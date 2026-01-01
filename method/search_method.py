import json
import logging
from typing import List

import aiohttp
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

import CONFIG
from utils import llm_client,semaphore
from method import search_utils
from rag import knowledge_rag
from states.knowledge_state import KnowledgeState
from states.search_state import SearchState
from tqdm.asyncio import tqdm
import asyncio
from utils.llm_client import MultimodalityMessage

async def get_search_keys(state: SearchState):
    print(f"[get_search_keys] Running for {state.route}......")

    class SearchKeys(BaseModel):
        search_keys: List[str] = Field(description="用于迭代调用搜索引擎")

    llm = llm_client.get_llm()
    structured_llm = llm.with_structured_output(SearchKeys)
    if state.ref_outlines:
        formatted_prompt = (
            f"为了完成《{state.main_topic}》的深度研究报告的{state.route}，现在我将要进行网络搜索，我该搜索些什么内容。\n"
            f"{state.chapter}\n"
            f"每个需要搜索的内容必须将《{state.main_topic}》作为主旨，必须包含《{state.main_topic}》，再在其基础上考虑相关的视角。\n"
            f"我搜索了以后，期望这些内容有助于我撰写《{state.main_topic}》的深度研究报告的{state.route}。\n"
            f"各个搜索句需保持精辟。可以有英文的搜索词。\n"
            f"目前主要的疑惑有{state.confuses}" if state.confuses is not None else ""
        )
    else:
        formatted_prompt = (
            f"我想组织《{state.main_topic}》的深度研究报告大纲，但是我不知道怎么写这个大纲，我想看看网上的人怎么写这个大纲的。\n"
            f"现在我将要进行网络搜索，请帮我搜索来查找类似研究文章的大纲写法。\n"
        )

    human_message = HumanMessage(formatted_prompt)

    result = await structured_llm.ainvoke([state.system_message, human_message])

    return {"search_keys": result.search_keys}

#一次只做一个topic的search，避免多个同时搜，都超时都搜不完，搜了才能做后续
@semaphore(1)
async def topic_search(state: SearchState):
    print(f"[target_url] Running for {state.route}......")

    tasks = [search_utils.web_search(search_key) for search_key in state.search_keys]
    search_results = await asyncio.gather(*tasks)  # 并发执行所有的搜索请求
    search_results = [item for sublist in search_results for item in sublist]


    return {"search_results": search_results}

#和上个步骤分开，避免rag也加锁
async def search_results_filter(state:SearchState):
    seen_urls = set()
    search_results = []
    for i in state.search_results:
        if i is None:continue
        url = i.get("url") or i.get('contentUrl')
        if (
                url
                and url not in seen_urls
                # and await state.get_by_url(url) is None
        ):
            seen_urls.add(url)
            i["url"] = url
            if i.get("text") and i.get("raw_text"):
                i["raw_text"]=False
            if i.get("text") and i.get("snippet"):
                i["snippet"]=False
            no_none_i={k:v for k,v in i.items() if v}
            if no_none_i.get('contentUrl'):
                no_none_i["is_picture"]=True#加个标志位
            else:
                no_none_i["is_picture"]=False
            search_results.append(no_none_i)

    return {"search_results": search_results}


# 要用redis，有些无用的数据避免重复调用
# @utils.semaphore(1)
async def fetch_webpage(state: SearchState) -> List[KnowledgeState]:
    # 生成 search_rs 字典，将 search_keys 中的每个搜索结果对应的 URL 和内容提取出来
    print(f"[fetch_knowledge] Running for {state.route}......")

    async with aiohttp.ClientSession() as session:
        if state.para_knowledge:
            tasks = [search_utils.fetch_webpage_async(page,session=session) for page in state.search_results]
        else:
            tasks = [search_utils.fetch_webpage_async(page,session=session) for page in state.search_results if not page.get('is_picture')]#写大纲不需要图片
        # results = await tqdm.gather(*tasks, desc="Fetching Pages", ascii=True, total=len(tasks))
        results = await asyncio.gather(*tasks)

    total = len(results)
    knowledge = [rs for rs in results if rs]

    print(f"\nsuccess {len(knowledge)}, failure {total - len(knowledge)} for {state.route}")

    return {"temp_knowledge": knowledge}


async def outline_filter(state: SearchState):
    print(f"[outline_filter] Running for {state.main_topic}......")
    llm = llm_client.get_llm()
    structured_llm = llm.with_structured_output(KnowledgeState)

    prompt = (
        f"我正在撰写大纲，请你判断以下内容是否能用于《{state.main_topic}》的大纲撰写的参考。"
        f"如果可以，请你整理整篇文章的大纲（逻辑）到summary字段。如果内容无用，请你直接返回is_useful=False，并不用进行整理。"
        # f"另外，返回一个参数is_related_to_chapter,判断是否和本章节内容相关，是否有助于撰写：{state.chapter}"
        f"网页如下："
    )
    temp_knowledge=[page for page in state.temp_knowledge if not page.get('is_picture')]#过滤掉图像

    tasks = [structured_llm.ainvoke([state.system_message, HumanMessage(prompt + str(page))]) for page in temp_knowledge]

    results = await tqdm.gather(*tasks, desc="Processing Outline Ref", total=len(state.temp_knowledge))

    results = [rs for rs in results if rs and rs.is_useful]

    return {"ref_outlines": results}



# @utils.semaphore(1)
async def brief_web_pages(state: SearchState):
    """简化的网页总结函数，添加 backoff 重试逻辑"""
    print(f"[brief_web_pages] Running for {state.route}......")
    llm = llm_client.get_llm()
    vllm = llm_client.get_llm(visual=True)
    structured_llm = llm.with_structured_output(KnowledgeState)
    structured_vllm = vllm.with_structured_output(KnowledgeState)

    prompt = (
        f"我找到了一篇网页，请你判断其是否和《{state.main_topic}》的深度研究主题相关。"
        f"如果相关，请你撰写知识的完整总结，必须包括文章所有的信息(以摘要形式呈现，注意其中比较新颖的观点、信息或数字),方便后续撰写。如果不相关，请你直接返回False，并不用进行总结。"
        f"如果网页中有图片，我会附带其中的图片给你，![把图片描述添加在这里](图片的url)。"
        f"网页如下："
    )

    pic_prompt = (
        f"我找到了一张图片，请你判断其是否和《{state.main_topic}》的深度研究主题相关。"
        f"如果图片相关，请你撰写图片描述到summary。如果不相关，请你直接返回False，并不用进行总结。"
        f"这是这张图片的来源页面（可参考其中这张图片的相关描述）："
    )

    # 使用 backoff 包装的重试函数
    tasks = [_try_summarize_with_retry(k,  state, structured_llm, structured_vllm, prompt, pic_prompt) for k in
             state.temp_knowledge]

    # 使用 tqdm_asyncio 生成带有进度条的异步任务
    # results = await tqdm.gather(*tasks, desc="Processing Knowledge", total=len(state.temp_knowledge))
    results = await asyncio.gather(*tasks)

    # 过滤无用知识
    total = len(results)
    knowledge = [rs for rs in results if rs is not None and rs.summary is not None and rs.is_useful]

    print(f"\nsuccess {len(knowledge)}, drop {total - len(knowledge)} for {state.route}")

    for k in knowledge:
        k.topic = [state.main_topic]

    # 将返回的知识填充到 知识库 中
    await state.add_knowledge(knowledge)

    return {"para_knowledge": knowledge + state.para_knowledge,"recursion_depth":state.recursion_depth-1}




# @backoff.on_exception(
#     backoff.expo,  # 使用指数退避
#     Exception,     # 捕获所有异常
#     max_tries=CONFIG.MAX_TRY,  # 最大重试次数
# )
async def _try_summarize_with_retry(page: dict,  state, structured_llm, structured_vllm, prompt, pic_prompt):
    if page is None:return None
    url = page.get("url")
    cache_key = f"brief_summary:{state.main_topic}:{url}"
    redis = CONFIG.get_redis_client()

    # 2. 先查 Redis 缓存
    try:
        cached = await redis.get(cache_key)
    except Exception as e:
        cached = None
        logging.info(f"[brief_web_pages] redis get error: {e}, url={url}")

    if cached:
        try:
            data = json.loads(cached)
            cached_ks = KnowledgeState(**data)  # 反序列化为 KnowledgeState
            if not cached_ks.is_useful:
                # print(f"[brief_web_pages] cache hit (useless) for {url}")
                return None  # 跳过无效的缓存
            # print(f"[brief_web_pages] cache hit for {url}")
            return cached_ks
        except Exception as e:
            print(f"[brief_web_pages] parse cache error: {e}, url={url}")

    # 3. 查数据库
    db = await knowledge_rag.get_context_knowledge_database(state.session_id)
    db_knowledge = db.get_by_url(url)
    if db_knowledge: return db_knowledge

    # 4. 无缓存 / 缓存解析失败 -> 调 LLM 做总结
    if page.get('is_picture'):
        result: KnowledgeState = await structured_vllm.ainvoke([
            state.system_message,
            MultimodalityMessage(pic_prompt + str(page), url)
        ])
        if result is not None:
            result.is_picture=True
    elif img_urls:=await search_utils.extract_images_from_markdown(page.get("text","")):
        result: KnowledgeState = await structured_vllm.ainvoke([
            state.system_message,
            MultimodalityMessage(prompt+str(page), * img_urls)
        ])
        if result is not None:
            result.is_picture=False
    else:
        result: KnowledgeState = await structured_llm.ainvoke([
            state.system_message,
            HumanMessage(prompt + str(page))
        ])
        if result is not None:
            result.is_picture = False

    if not result or not result.is_useful or not result.summary:
        # 如果结果无效，缓存它
        dummy = KnowledgeState(is_useful=False)
        try:
            data = dummy.model_dump()
            await redis.set(cache_key, json.dumps(data, ensure_ascii=False), ex=CONFIG.REDIS_EXPIRE)
        except Exception as e:
            print(f"[brief_web_pages] redis set dummy error: {e}, url={url}")
        return None

    result.url = url
    # 缓存有效的结果
    try:
        data = result.model_dump()
        await redis.set(cache_key, json.dumps(data, ensure_ascii=False), ex=CONFIG.REDIS_EXPIRE)
    except Exception as e:
        logging.info(f"[brief_web_pages] redis set error: {e}, url={url}")

    return result




# 要用redis，有些无用的数据避免重复调用
# brief_web_pages_semaphore = asyncio.Semaphore(1)


# def brief_web_pages_synchronized(func):
#     @wraps(func)
#     async def wrapper(*args, **kwargs):
#         async with brief_web_pages_semaphore:  # 限制并发请求数
#             return await func(*args, **kwargs)
#
#     return wrapper
#
#
# @brief_web_pages_synchronized
# async def brief_web_pages(state: SearchState):
#     """同步函数，调用异步的 ainvoke 来处理网页总结"""
#     print(f"[brief_web_pages] Running for {state.route}......")
#     llm = llm_client.get_llm()
#     vllm= llm_client.get_llm(visual=True)
#     structured_llm = llm.with_structured_output(KnowledgeState)
#     structured_vllm = vllm.with_structured_output(KnowledgeState)
#
#     prompt = (
#         f"我找到了一篇网页，请你判断其是否和《{state.main_topic}》的深度研究主题相关。"
#         f"如果相关，请你撰写知识的完整总结，必须包括文章所有的信息(以摘要形式呈现，注意其中比较新颖的观点、信息或数字),方便后续撰写。如果不相关，请你直接返回False，并不用进行总结。"
#         # f"另外，返回一个参数is_related_to_chapter,判断是否和本章节内容相关，是否有助于撰写：{state.chapter}"
#         f"网页如下："
#     )
#
#     pic_prompt = (
#         f"我找到了一张图片，请你判断其是否和《{state.main_topic}》的深度研究主题相关。"
#         f"如果相关，请你撰写图片描述到summary。如果不相关，请你直接返回False，并不用进行总结。"
#     )
#
#     # ✅ 在外层初始化 Redis 客户端（假设是异步客户端）
#     redis = CONFIG.get_redis_client()
#
#     async def try_summarize(page: dict):
#         """
#         - 文本短：直接返回原始 k（不调用 LLM，也可以不用缓存）
#         - 文本长：先查缓存；无缓存才调用 LLM 总结，并把结果写入缓存
#         - 缓存 & LLM 的单位都是：url 对应的 KnowledgeState
#         """
#         # 1. 短文本直接跳过总结逻辑
#         # if utils.count_tokens_auto(k.summary) <= CONFIG.KNOWLEDGE_MAX_LENGTH_TOKEN:
#         #     return k
#         # 现在全部用LLM总结
#         url = page.get("url")
#
#         cache_key = f"brief_summary:{state.main_topic}:{url}"
#
#         # 2. 先查 Redis 缓存
#         try:
#             cached = await redis.get(cache_key)
#         except Exception as e:
#             cached = None
#             print(f"[brief_web_pages] redis get error: {e}, url={url}")
#
#         if cached:
#             try:
#                 data = json.loads(cached)
#                 cached_ks = KnowledgeState(**data)  # 反序列化为 KnowledgeState
#                 if not cached_ks.is_useful:
#                     # 之前已经判断为无用，直接跳过
#                     print(f"[brief_web_pages] cache hit (useless) for {url}")
#                     return None
#                 print(f"[brief_web_pages] cache hit for {url}")
#                 return cached_ks
#             except Exception as e:
#                 print(f"[brief_web_pages] parse cache error: {e}, url={url}")
#                 # 缓存坏了，当作没命中，继续往下走 LLM
#
#         # 3. 无缓存 / 缓存解析失败 -> 调 LLM 做总结（带重试）
#         t = CONFIG.MAX_TRY
#         while t := t - 1:
#             try:
#                 if not page.get('is_picture'):
#                     result: KnowledgeState = await structured_llm.ainvoke([
#                         state.system_message,
#                         HumanMessage(prompt + str(page)),
#                     ])
#                 else:
#                     result: KnowledgeState = await structured_vllm.ainvoke([
#                         state.system_message,
#                         MultimodalityMessage(pic_prompt + str(page),page.get('url'))
#                     ])
#
#                 if not result:
#                     return None
#
#                 # 3.1 LLM 按约定返回了一个 KnowledgeState
#                 # 不相关：is_useful=False -> 不入库，但写缓存以供下次快速跳过
#                 if not result.is_useful or not result.summary:
#                     try:
#                         data = result.model_dump()
#                         await redis.set(
#                             cache_key,
#                             json.dumps(data, ensure_ascii=False),
#                             ex=CONFIG.REDIS_EXPIRE,
#                         )
#                     except Exception as e:
#                         print(f"[brief_web_pages] redis set useless error: {e}, url={url}")
#                     return None
#
#                 # 3.2 相关且有摘要 -> 写缓存
#                 try:
#                     data = result.model_dump()
#                     await redis.set(
#                         cache_key,
#                         json.dumps(data, ensure_ascii=False),
#                         ex=CONFIG.REDIS_EXPIRE,
#                     )
#                 except Exception as e:
#                     print(f"[brief_web_pages] redis set error: {e}, url={url}")
#
#                 return result
#
#             except ValueError as e:
#                 # DataInspectionFailed 一般是结构化解析失败
#                 if "DataInspectionFailed" in str(e):
#                     print(f"[brief_web_pages] DataInspectionFailed for {url}. Skipping this content.")
#                     # 为了以后不再踩坑，这里也可以写一个 is_useful=False 的空结果进缓存
#                     dummy = KnowledgeState(
#                         topic=[],
#                         url=url,
#                         is_useful=False,
#                         time="",
#                         summary="",
#                     )
#                     try:
#                         data = dummy.model_dump()
#                         await redis.set(
#                             cache_key,
#                             json.dumps(data, ensure_ascii=False),
#                             ex=CONFIG.REDIS_EXPIRE,
#                         )
#                     except Exception as e2:
#                         print(f"[brief_web_pages] redis set dummy useless error: {e2}, url={url}")
#                     return None
#                 else:
#                     wait_time = (CONFIG.MAX_TRY - t + 1) ** 2
#                     print(
#                         f"[brief_web_pages] Unknown error: {e}, retrying {CONFIG.MAX_TRY - t}/{CONFIG.MAX_TRY} in {wait_time} seconds. url={url}"
#                     )
#                     await asyncio.sleep(wait_time)
#
#             # except Exception as e:
#             #     wait_time = (CONFIG.MAX_TRY - t + 1) ** 2
#             #     print(
#             #         f"[brief_web_pages] Error: {e}, retrying {CONFIG.MAX_TRY - t}/{CONFIG.MAX_TRY} in {wait_time} seconds. url={url}"
#             #     )
#             #     await asyncio.sleep(wait_time)
#
#         # 4. 重试耗尽 -> 记录一个 is_useful=False 的 dummy，避免以后死循环
#         dummy = KnowledgeState(
#             topic=[],
#             url=url,
#             is_useful=False,
#             time="",
#             summary="",
#         )
#         try:
#             data = dummy.model_dump()
#             await redis.set(
#                 cache_key,
#                 json.dumps(data, ensure_ascii=False),
#                 ex=CONFIG.REDIS_EXPIRE,
#             )
#         except Exception as e:
#             print(f"[brief_web_pages] redis set dummy after retries error: {e}, url={url}")
#
#         return None
#
#     tasks = [try_summarize(k) for k in state.temp_knowledge]
#
#     # 使用 tqdm_asyncio 生成带有进度条的异步任务
#     results = await tqdm.gather(*tasks, desc="Processing Knowledge", total=len(state.temp_knowledge))
#     # 过滤无用知识
#     total = len(results)
#     knowledge = [rs for rs in results if rs is not None and rs.summary is not None and rs.is_useful]
#
#     print(f"\nsuccess {len(knowledge)}, drop {total - len(knowledge)}")
#
#     for k in knowledge:
#         k.topic = [state.main_topic]
#
#     # 将返回的知识填充到 知识库 中
#     await state.add_knowledge(knowledge)
#
#     # 将返回的知识填充到 chapter知识 中
#     # knowledge = [rs for rs in knowledge if rs.is_related_to_chapter]
#     return {"para_knowledge": knowledge + state.para_knowledge}


