import asyncio
import logging
from contextlib import contextmanager

# import aiosqlite

import CONFIG
from utils import llm_client
import utils
from states.knowledge_state import KnowledgeState


import os
import json
import sqlite3
from typing import List, Optional, Union

import numpy as np
import faiss
from datetime import datetime

from langchain.tools import tool
import dashscope
logger = logging.getLogger(__name__)

_conn=None

@contextmanager
def lifespan(path=getattr(CONFIG, "SQLITE_PATH", "resources/knowledge.db")):
    global _conn
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with sqlite3.connect(path) as conn:
        _conn = conn
        yield


class KnowledgeDatabase:
    """
    使用 SQLite + Faiss 实现的持久化知识库。
    - SQLite 保存所有元数据和向量（BLOB）
    - Faiss 负责向量检索（内存索引，启动时从 SQLite 重建）
    - 通过 collection_name 区分不同知识库
    - 支持 topic 关键字一对多检索
    """

    def __init__(
        self,
        dim: int = None,
        # model_name: str = "text-embedding-v2",
        collection_name: str = "knowledge_collection",
    ):
        self.collection_name = collection_name

        # ---- 1. 初始化 SQLite ----
        # db_path = getattr(CONFIG, "SQLITE_PATH", "resources/knowledge.db")
        # os.makedirs(os.path.dirname(db_path), exist_ok=True)
        # self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn = _conn
        self.conn.row_factory = sqlite3.Row
        self._init_tables()

        # ---- 2. 初始化向量维度 & Faiss ----
        self.dim = dim if dim else CONFIG.VECTOR_DIM
        self.index = self._load_faiss_index_from_disk()

        # ---- 3. 初始化 embedding 模型 & Redis ----
        dashscope.api_key = CONFIG.API_KEY
        # self.embedder = DashScopeEmbedding(model_name=model_name)
        self.embedder = llm_client.get_embed_client()

        self.redis = CONFIG.get_redis_client()

        # ---- 4. 同步 Faiss 索引与 SQL 数据 ----
        self._sync_faiss_with_sql()

    # =========================
    # 基础结构 & 持久化
    # =========================

    def _init_tables(self) -> None:
        """初始化 SQLite 表结构"""
        with self.conn:
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS knowledge
                (
                    id
                    INTEGER
                    PRIMARY
                    KEY
                    AUTOINCREMENT,
                    collection_name
                    TEXT
                    NOT
                    NULL,
                    url
                    TEXT
                    NOT
                    NULL,
                    summary
                    TEXT
                    NOT
                    NULL,
                    is_useful
                    INTEGER
                    NOT
                    NULL,
                    time
                    TEXT,
                    topic_json
                    TEXT
                    NOT
                    NULL,
                    embedding
                    BLOB
                    NOT
                    NULL,
                    is_picture
                    INTEGER
                    NOT
                    NULL,
                    site_name
                    TEXT
                    NOT
                    NULL
                    DEFAULT
                    '',
                    UNIQUE
                (
                    collection_name,
                    url
                )
                    );
                """
            )
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS knowledge_topic
                (
                    id
                    INTEGER
                    PRIMARY
                    KEY
                    AUTOINCREMENT,
                    collection_name
                    TEXT
                    NOT
                    NULL,
                    topic
                    TEXT
                    NOT
                    NULL,
                    knowledge_id
                    INTEGER
                    NOT
                    NULL
                );
                """
            )
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_knowledge_collection ON knowledge(collection_name);"
            )
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_topic_collection ON knowledge_topic(collection_name, topic);"
            )
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_topic_knowledge ON knowledge_topic(knowledge_id);"
            )

            # # 兼容已有 DB：如果老库没有 site_name 列，这里补上
            # cur = self.conn.execute("PRAGMA table_info(knowledge);")
            # cols = {row["name"] for row in cur.fetchall()}
            # if "site_name" not in cols:
            #     self.conn.execute(
            #         "ALTER TABLE knowledge ADD COLUMN site_name TEXT NOT NULL DEFAULT '';"
            #     )

    def _load_faiss_index_from_disk(self) -> faiss.Index:
        """
        从磁盘加载当前 collection 的 Faiss 索引
        如果索引文件不存在，创建一个新的 Faiss 索引
        """
        index_path = f"resources/{self.collection_name}_faiss.index"
        if os.path.exists(index_path):
            print(f"加载 Faiss 索引文件: {index_path}")
            index = faiss.read_index(index_path)
        else:
            # 如果没有索引文件，创建一个新的索引
            print(f"创建新的 Faiss 索引: {index_path}")
            index = faiss.IndexFlatIP(self.dim)  # 使用内积（cosine 可在外面做归一化）
            index = faiss.IndexIDMap(index)  # 创建带 ID 的索引

        return index

    def save_faiss_index_to_disk(self) -> None:
        """
        将 Faiss 索引保存到磁盘
        """
        index_path = f"resources/{self.collection_name}_faiss.index"
        faiss.write_index(self.index, index_path)
        print(f"Faiss 索引已保存至: {index_path}")

    def _sync_faiss_with_sql(self) -> None:
        """
        从 SQLite 加载所有数据并同步到 Faiss 索引。
        需要确保数据库和 Faiss 中的数据一致。
        """
        # print(f"同步 Faiss 索引与 SQL 数据...")

        cur = self.conn.execute(
            "SELECT id, embedding FROM knowledge WHERE collection_name = ?",
            (self.collection_name,),
        )
        rows = cur.fetchall()
        if not rows:
            print("没有数据需要同步到 Faiss 索引。")
            return

        ids = []
        embs = []
        for row in rows:
            emb = np.frombuffer(row["embedding"], dtype=np.float32)
            if emb.shape[0] != self.dim:
                # 维度不匹配的记录，跳过
                continue
            ids.append(int(row["id"]))
            embs.append(emb)

        if ids:
            ids_array = np.asarray(ids, dtype=np.int64)
            embs_array = np.vstack(embs).astype(np.float32)
            self.index.add_with_ids(embs_array, ids_array)

        print("Faiss 索引与 SQL 数据同步完成。")

        self.save_faiss_index_to_disk()

    def _load_faiss_index_from_db(self) -> faiss.Index:
        """
        从 SQLite 载入当前 collection 的向量，重建 Faiss Index。
        为简单起见，这里使用内存重建，不额外持久化 Faiss 文件。
        """
        # 使用内积（cosine 可在外面做归一化，有需要可以补）
        base_index = faiss.IndexFlatIP(self.dim)
        index = faiss.IndexIDMap(base_index)

        cur = self.conn.execute(
            "SELECT id, embedding FROM knowledge WHERE collection_name = ?",
            (self.collection_name,),
        )
        rows = cur.fetchall()
        if not rows:
            return index

        ids = []
        embs = []
        for row in rows:
            emb = np.frombuffer(row["embedding"], dtype=np.float32)
            if emb.shape[0] != self.dim:
                # 维度不匹配的记录，跳过
                continue
            ids.append(int(row["id"]))
            embs.append(emb)

        if ids:
            ids_array = np.asarray(ids, dtype=np.int64)
            embs_array = np.vstack(embs).astype(np.float32)
            index.add_with_ids(embs_array, ids_array)

        return index

    def __len__(self) -> int:
        cur = self.conn.execute(
            "SELECT COUNT(*) AS cnt FROM knowledge WHERE collection_name = ?",
            (self.collection_name,),
        )
        row = cur.fetchone()
        return row["cnt"] if row else 0

    # =========================
    # 核心 CRUD
    # =========================

    def get_by_url(self, url: str) -> Optional[KnowledgeState]:
        """根据 URL 从 SQLite 取出一条知识记录"""
        cur = self.conn.execute(
            """
            SELECT url, summary, is_useful, time, topic_json, is_picture, site_name
            FROM knowledge
            WHERE collection_name = ? AND url = ?
            """,
            (self.collection_name, url),
        )
        row = cur.fetchone()
        if not row:
            return None

        ks = KnowledgeState(
            url=row["url"],
            summary=row["summary"],
            is_useful=bool(row["is_useful"]),
            time=row["time"],
            is_picture=bool(row["is_picture"]),
            site_name=row["site_name"],
        )
        ks.load_topic(row["topic_json"])
        return ks

    async def add_knowledge(
        self, knowledge: List[KnowledgeState] | KnowledgeState
    ) -> None:
        """
        增加知识数据，支持单个或列表输入。
        - 使用 SQLite 持久化
        - 使用 Faiss 向量索引
        - 对已有 URL 做更新而不是重复插入
        """
        if isinstance(knowledge, KnowledgeState):
            knowledge = [knowledge]

        # 过滤无效
        seen_urls = set()
        knowledge = [
            k
            for k in knowledge
            if k is not None
               and k.summary is not None
               and getattr(k, "is_useful", True)
               and self.get_by_url(k.url) is None
               and (seen_urls.add(k.url.strip()) or True)  # 去重
        ]
        if not knowledge:
            return

        summaries = [k.summary for k in knowledge]

        try:
            embeddings = await self.embed_query(summaries)  # shape: [N, dim]
        except Exception as e:
            raise RuntimeError(f"Fail to add knowledge because error encoding texts: {e}")

        to_insert: List[tuple[KnowledgeState, np.ndarray]] = []

        for idx, k in enumerate(knowledge):
            if not k or not k.summary:
                continue

            existing = self.get_by_url(k.url)
            if existing:
                # 合并 topic 去重
                existing_topics = getattr(existing, "topic", [])
                new_topics = getattr(k, "topic", [])
                merged_topics = list({*existing_topics, *new_topics})
                existing.topic = merged_topics
                existing.summary = k.summary
                existing.time = k.time
                existing.is_useful = k.is_useful
                await self.update_knowledge(
                    k.url, existing, embedding=embeddings[idx]
                )
                continue

            to_insert.append((k, embeddings[idx]))

        if not to_insert:
            return

        with self.conn:
            for k, emb in to_insert:
                topic_list = getattr(k, "topic", [])
                topic_json = json.dumps(topic_list, ensure_ascii=False)
                emb_blob = emb.astype(np.float32).tobytes()
                try:
                    cur = self.conn.execute(
                        """
                        INSERT INTO knowledge
                        (collection_name, url, summary, is_useful, time, topic_json, embedding, is_picture, site_name)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            self.collection_name,
                            k.url,
                            k.summary,
                            int(k.is_useful),
                            k.time,
                            topic_json,
                            emb_blob,
                            int(k.is_picture),
                            k.site_name,
                        ),
                    )
                except Exception as e:
                    logger.info(f"[DB] Fail to add knowledge: {e}")
                    continue

                knowledge_id = cur.lastrowid

                # 写入 topic 映射
                for tp in topic_list:
                    self.conn.execute(
                        """
                        INSERT INTO knowledge_topic (collection_name, topic, knowledge_id)
                        VALUES (?, ?, ?)
                        """,
                        (self.collection_name, tp, knowledge_id),
                    )

                # 写入 Faiss 索引
                emb_vec = emb.reshape(1, -1).astype(np.float32)
                self.index.add_with_ids(
                    emb_vec, np.asarray([knowledge_id], dtype=np.int64)
                )

        self.conn.commit()
        # self.save_faiss_index_to_disk()

    def delete_knowledge(self, knowledge: "KnowledgeState") -> None:
        """删除指定 URL 的知识数据"""
        cur = self.conn.execute(
            """
            SELECT id FROM knowledge
            WHERE collection_name = ? AND url = ?
            """,
            (self.collection_name, knowledge.url),
        )
        row = cur.fetchone()
        if not row:
            return

        knowledge_id = int(row["id"])

        with self.conn:
            self.conn.execute(
                "DELETE FROM knowledge WHERE id = ?", (knowledge_id,)
            )
            self.conn.execute(
                "DELETE FROM knowledge_topic WHERE knowledge_id = ?",
                (knowledge_id,),
            )

        # 删除 Faiss 索引中的向量
        try:
            self.index.remove_ids(np.asarray([knowledge_id], dtype=np.int64))
        except Exception:
            # faiss 不存在此 id 时可能报错，忽略即可
            pass
        self.conn.commit()
        # self.save_faiss_index_to_disk()

    async def update_knowledge(
        self,
        url: str,
        new_knowledge: "KnowledgeState",
        embedding: Optional[np.ndarray] = None,
    ) -> None:
        """
        更新指定 URL 的知识数据。
        可通过 embedding 参数传入预计算好的向量，避免重复调接口。
        """
        cur = self.conn.execute(
            """
            SELECT id FROM knowledge
            WHERE collection_name = ? AND url = ?
            """,
            (self.collection_name, url),
        )
        row = cur.fetchone()
        if not row:
            # 若不存在，则视为新增
            await self.add_knowledge(new_knowledge)
            return

        knowledge_id = int(row["id"])

        if embedding is None:
            emb_arr = await self.embed_query(new_knowledge.summary)
            embedding = emb_arr[0]

        topic_list = getattr(new_knowledge, "topic", [])
        topic_json = json.dumps(topic_list, ensure_ascii=False)
        emb_blob = embedding.astype(np.float32).tobytes()

        with self.conn:
            self.conn.execute(
                """
                UPDATE knowledge
                SET summary    = ?,
                    is_useful  = ?,
                    time       = ?,
                    topic_json = ?,
                    embedding  = ?,
                    is_picture = ?,
                    site_name  = ?
                WHERE id = ?
                """,
                (
                    new_knowledge.summary,
                    int(new_knowledge.is_useful),
                    new_knowledge.time,
                    topic_json,
                    emb_blob,
                    int(new_knowledge.is_picture),
                    new_knowledge.site_name,
                    knowledge_id,
                ),
            )

            # 先清空再重建 topic 映射
            self.conn.execute(
                "DELETE FROM knowledge_topic WHERE knowledge_id = ?",
                (knowledge_id,),
            )
            for tp in topic_list:
                self.conn.execute(
                    """
                    INSERT INTO knowledge_topic (collection_name, topic, knowledge_id)
                    VALUES (?, ?, ?)
                    """,
                    (self.collection_name, tp, knowledge_id),
                )

        # 更新 Faiss：删除旧向量再添加新向量
        try:
            self.index.remove_ids(np.asarray([knowledge_id], dtype=np.int64))
        except Exception:
            pass

        self.index.add_with_ids(
            embedding.reshape(1, -1).astype(np.float32),
            np.asarray([knowledge_id], dtype=np.int64),
        )

        self.conn.commit()
        # self.save_faiss_index_to_disk()
    # =========================
    # 检索：向量 + 过滤 + topic
    # =========================

    async def search_knowledge(
            self,
            query: Union[str, List[str]],
            top_k: int = None,
            start_time: Optional[str] = None,
            end_time: Optional[str] = None,
            sim_threshold: float = CONFIG.SIM_THRESHOLD,
            # topic: Optional[str] = None,
    ) -> List[KnowledgeState]:
        """
        按照查询词进行向量检索，返回相关的知识，
        支持时间范围筛选和 topic 过滤，
        并逐步增大 top_k（最多到 CONFIG.RAG_MAX_TOP_K）。
        并且根据 CONFIG.SIM_THRESHOLD 过滤相似度低的结果。
        """
        if top_k is None:
            top_k = CONFIG.RAG_MAX_TOP_K

        if self.index.ntotal == 0:
            return []

        # 将 List[str] 合并为一个 query，和原来行为类似
        if isinstance(query, list):
            query_text = " ".join(query)
        else:
            query_text = query

        query_embs = await self.embed_query(query_text)
        query_vec = query_embs[0].reshape(1, -1).astype(np.float32)

        max_top_k = self.index.ntotal
        current_top_k = min(top_k, max_top_k)

        uniquer=set()
        results: List[KnowledgeState] = []

        while current_top_k <= max_top_k:
            # Faiss 检索
            distances, ids = self.index.search(query_vec, current_top_k)
            id_list = [int(i) for i in ids[0] if i != -1]
            if not id_list:
                break

            # SQLite 最大支持 999 个变量，所以要分批处理
            batch_size = 999
            all_results = []
            for i in range(0, len(id_list), batch_size):
                batch_ids = id_list[i:i + batch_size]
                placeholders = ",".join("?" for _ in batch_ids)

                cur = self.conn.execute(
                    f"""
                        SELECT id, url, summary, is_useful, time, topic_json, is_picture, site_name
                        FROM knowledge
                        WHERE id IN ({placeholders}) AND collection_name = ?
                        """,
                    (*batch_ids, self.collection_name),
                )
                rows = cur.fetchall()
                all_results.extend(rows)

            id_to_row = {int(r["id"]): r for r in rows}

            filtered: List[KnowledgeState] = []
            for idx, doc_id in enumerate(id_list):
                row = id_to_row.get(doc_id)
                if not row:
                    continue

                ks = KnowledgeState(
                    url=row["url"],
                    summary=row["summary"],
                    is_useful=bool(row["is_useful"]),
                    time=row["time"],
                    is_picture=bool(row["is_picture"]),
                    site_name=row["site_name"],
                )
                ks.load_topic(row["topic_json"])

                # 时间过滤
                time_ok = self._is_within_time_range(
                    ks.time, start_time, end_time
                )

                if distances[0][idx] < sim_threshold:continue
                if not time_ok:continue
                if ks.url not in uniquer:
                    uniquer.add(ks.url)

                filtered.append(ks)

            results = filtered
            if len(results) >= top_k:
                break

            # 扩大检索范围
            if current_top_k == max_top_k:
                break
            current_top_k = min(current_top_k * 2, max_top_k)

        return results[:top_k]

    def search_by_topic(
        self, topic: str, limit: int = 20
    ) -> List[KnowledgeState]:
        """
        通过 topic 关键字，一对多检索知识记录（不走向量检索）。
        """
        cur = self.conn.execute(
            """
            SELECT k.id,
                   k.url,
                   k.summary,
                   k.is_useful,
                   k.time,
                   k.topic_json,
                   k.is_picture,
                   k.site_name
            FROM knowledge_topic t
                     JOIN knowledge k ON t.knowledge_id = k.id
            WHERE t.collection_name = ?
              AND t.topic = ?
              AND k.collection_name = ?
            ORDER BY k.id DESC LIMIT ?
            """,
            (self.collection_name, topic, self.collection_name, limit),
        )
        rows = cur.fetchall()

        results: List[KnowledgeState] = []
        for row in rows:
            ks = KnowledgeState(
                url=row["url"],
                summary=row["summary"],
                is_useful=bool(row["is_useful"]),
                time=row["time"],
                is_picture=bool(row["is_picture"]),
                site_name=row["site_name"],
            )
            ks.load_topic(row["topic_json"])
            results.append(ks)
        return results

    # =========================
    # 时间过滤
    # =========================

    def _is_within_time_range(
        self,
        time_str: str,
        start_time: Optional[str],
        end_time: Optional[str],
        none_time_returns: bool = True,
    ) -> bool:
        """判断时间是否在给定范围内，格式：YYYY-MM-DD"""
        if not start_time and not end_time:
            return True

        if not time_str:
            return none_time_returns

        try:
            record_time = datetime.strptime(time_str, "%Y-%m-%d")
        except ValueError:
            # 非法时间格式，视为不过滤
            return none_time_returns

        if start_time:
            try:
                start_dt = datetime.strptime(start_time, "%Y-%m-%d")
                if record_time < start_dt:
                    return False
            except ValueError as e:
                logger.info(f"[RAG] Invalid time str: {start_time}")

        if end_time:
            try:
                end_dt = datetime.strptime(end_time, "%Y-%m-%d")
                if record_time > end_dt:
                    return False
            except ValueError as e:
                logger.info(f"[RAG] Invalid time str: {end_time}")

        return True

    # =========================
    # Embedding + Redis 缓存
    # =========================

    async def embed_query(
            self, sentences: List[str] | str
    ) -> np.ndarray:
        """
        为查询词生成 embedding 向量。
        - 每个句子单独异步获取 embedding，避免批量。
        - 使用 backoff 重试机制。
        - 使用 Redis 缓存每个句子的 embedding。
        """
        if isinstance(sentences, str):
            sentences = [sentences]

        embeddings: List[Optional[np.ndarray]] = [None] * len(sentences)
        missing_indices: List[int] = []
        missing_sentences: List[str] = []

        # 先从 Redis 里查缓存
        for idx, sent in enumerate(sentences):
            cache_key = utils.stable_cache_key(sent)
            cached = await self.redis.get(cache_key)
            if cached:
                emb = np.frombuffer(cached, dtype=np.float32)
                embeddings[idx] = emb
            else:
                missing_indices.append(idx)
                missing_sentences.append(sent)

        # 对缺失的句子异步获取 embedding（批量处理）
        if missing_sentences:
            # 使用 aget_batch_text_embedding 来批量获取嵌入
            batch_embeddings = await self.embedder.aget_batch_text_embedding(missing_sentences)

            # 将每个批量嵌入结果拆解并存入 embeddings 和 Redis
            for idx, (emb, sent) in zip(missing_indices, zip(batch_embeddings, missing_sentences)):
                embeddings[idx] = emb
                # 将嵌入向量存入 Redis
                cache_key = utils.stable_cache_key(sent)
                await self.redis.set(
                    cache_key,
                    emb.tobytes(),
                    ex=CONFIG.REDIS_EXPIRE,
                )

        if any(e is None for e in embeddings):
            raise RuntimeError("Failed to compute some embeddings")

        return np.vstack(embeddings).astype(np.float32)

    def __del__(self):
        self.save_faiss_index_to_disk()
        self.redis.close()
        self.conn.commit()
        self.conn.close()


_cache_context_knowledge_database={}

#避免同时创建的冲突
@utils.semaphore(1)
async def get_context_knowledge_database(session_id: str) -> KnowledgeDatabase:
    """
    根据 session_id 返回对应 collection 的 KnowledgeDatabase。
    与原始接口保持兼容。
    """
    global _cache_context_knowledge_database
    kdb=_cache_context_knowledge_database.get(session_id)
    if kdb is None:
        kdb=KnowledgeDatabase(collection_name=session_id)
        _cache_context_knowledge_database[session_id]=kdb
    return kdb

#签名
# @tool(return_direct=True)
@tool
async def get_knowledge_conditional_tool_call(query:str=None,start_time:str=None,end_time:str=None,top_k:int=None,sim_threshold:float=None)->List[KnowledgeState]:
    """
    调用RAG拿到知识库知识
    :param query: 用于RAG搜索的句子
    :param start_time: 限定知识起始时间
    :param end_time: 限定知识截止时间
    :param top_k: 返回的最大知识数量
    :param sim_threshold: 置信度阈值
    :return: List[KnowledgeState]
    """
    raise NotImplementedError("[Error]'get_knowledge_conditional_tool_call' is not callable, please call 'get_knowledge_conditional' instead.")

async def get_knowledge_conditional(session_id:str,query:str=None,start_time:str=None,end_time:str=None,top_k:int=None,sim_threshold:float=CONFIG.SIM_THRESHOLD)->List[KnowledgeState]:
    ck = await get_context_knowledge_database(session_id)
    if top_k is None:
        top_k = CONFIG.RAG_MAX_TOP_K
    return await ck.search_knowledge(query=query, start_time=start_time, end_time=end_time, top_k=top_k,sim_threshold=sim_threshold)


# 添加方法定时删除

async def main():
    print("==== 初始化知识库（collection = 'test_collection'）====")
    kb = KnowledgeDatabase(collection_name="test_collection")

    # 清理旧数据（可选）
    # 你如果希望每次测试干净一点可以删除 SQLite 文件
    # import os
    # os.remove("resources/knowledge.db")

    print("\n==== 构造知识对象 1 ====")
    k1 = KnowledgeState(
        url="https://example.com/a",
        summary="深度学习用于图像分类的方法包括 CNN、ResNet 等。",
        topic=["深度学习", "计算机视觉"],
        is_useful=True,
        time="2024-02-01",
    )

    print("==== 构造知识对象 2 ====")
    k2 = KnowledgeState(
        url="https://example.com/b",
        summary="Transformers 在 NLP 任务（如翻译、摘要）中表现优秀。",
        topic=["自然语言处理", "深度学习"],
        is_useful=True,
        time="2024-01-15",
    )

    print("\n==== 添加知识到数据库 ====")
    await kb.add_knowledge([k1, k2])
    print(f"当前知识库数据量: {len(kb)}")

    print("\n==== 使用向量检索 ====")
    results = await kb.search_knowledge("什么是深度学习？", top_k=3)
    print("向量检索结果：")
    for r in results:
        print(f"- {r.url} | topic={r.topic} | summary={r.summary}")

    print("\n==== 按 topic 检索（深度学习） ====")
    topic_results = kb.search_by_topic("深度学习")
    for r in topic_results:
        print(f"- {r.url} | summary={r.summary}")

    print("\n==== 更新某条知识 ====")
    k1_updated = KnowledgeState(
        url="https://example.com/a",
        summary="深度学习模型（如 CNN、ResNet、Transformer）可用于图像分类。",
        topic=["深度学习", "CV"],
        is_useful=True,
        time="2024-03-01",
    )
    await kb.update_knowledge(k1_updated.url, k1_updated)
    print("更新成功！")

    print("\n==== 再次向量检索 ====")
    results = await kb.search_knowledge("图像分类模型有哪些？", top_k=3)
    for r in results:
        print(f"- {r.url} | topic={r.topic} | summary={r.summary}")

    print("\n==== 删除一个知识 ====")
    kb.delete_knowledge(k2)
    print(f"删除后总数: {len(kb)}")

    print("\n==== 按 topic 检索（自然语言处理） ====")
    nlp_results = kb.search_by_topic("自然语言处理")
    print(f"NLP topic 数量: {len(nlp_results)}")

    print("\n==== 测试结束 ====")


if __name__ == "__main__":
    asyncio.run(main())