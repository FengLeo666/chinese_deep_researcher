import os

import redis

os.makedirs("resources",exist_ok=True)
REPORT_DIR = "resources/reports"

API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
API_KEY = os.environ["API_KEY"]
LLM_TYPE= "qwen3-max"
# LLM_TYPE= "qwen3-vl-plus"
VISUAL_LLM_TYPE= "qwen3-vl-plus-2025-12-19"
LLM_MAX_THREADS= 32

EMBEDDING_MODEL="text-embedding-v4"
VECTOR_DIM=2048
EMBED_BS=10
KNOWLEDGE_MAX_LENGTH=8192

RAG_MAX_TOP_K=5

RECURSION_DEPTH=3
MAX_PARAGRAPH_KNOWLEDGE=16

MAX_TRY=16

SIM_THRESHOLD = 0.45   # 可调，一般 0.40–0.55 最稳

REDIS_HOST="redis://localhost:6379"
REDIS_EXPIRE=3600000

_REDIS_CLIENT= redis.asyncio.from_url(REDIS_HOST)
def get_redis_client():
    global _REDIS_CLIENT
    return _REDIS_CLIENT



# GIVE_UP_STATUS_CODE=[468,555,400, 401, 403, 404, 405, 409, 410, 422, 502]
LOCAL_SEARCH_URL="http://127.0.0.1:8088"
BOCHA_SEARCH_URL= "https://api.bocha.cn/v1/web-search"
BOCHA_API_KEY=os.environ["BOCHA_API_KEY"]
MAX_SEARCH_RESULT=5
SEARCH_MAX_THREAD=2

TIME_WARN=60


HEADERS= {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
        'Accept-Encoding': 'gzip, deflate, br',
        'Accept-Language': 'en-US,en;q=0.9',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        # 'Referer': 'https://www.example.com',
    }









# import asyncio
#
# async def main():
#     r = get_redis_client()
#     # 初始化游标
#     cursor = 0
#
#     # 扫描并删除所有以 'page' 开头的键
#     while True:
#         # 使用 SCAN 命令扫描 Redis 键
#         cursor, keys = await r.scan(cursor=cursor, match='page*', count=1000)
#
#         # 删除匹配的键
#         if keys:
#             await r.unlink(*keys)  # 使用 unlink 删除键，避免阻塞
#
#         # 如果游标为 0，表示扫描结束
#         if cursor == 0:
#             break
#
#     print("删除完毕！")
# if __name__=="__main__":
#     asyncio.run(main())