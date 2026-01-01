
import aiosqlite
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from contextlib import asynccontextmanager



_checkpointer=None

@asynccontextmanager
async def lifespan(path="resources/checkpoints.db"):
    global _checkpointer
    async with AsyncSqliteSaver.from_conn_string(
        path
    ) as saver:
        _checkpointer = saver
        yield


# @utils.semaphore(1)
async def get_checkpointer():
    global _checkpointer
    # if checkpointer is None:
    #     conn = await aiosqlite.connect("resources/checkpoints.db", check_same_thread=False)
    #     checkpointer = AsyncSqliteSaver(conn)
    #     await checkpointer.setup()
    #
    #     # 注册程序结束时关闭数据库连接的回调函数
    #     # atexit.register(lambda: asyncio.create_task(close_checkpointer()))

    return _checkpointer

#
# async def close_checkpointer():
#     global checkpointer
#     if checkpointer:
#         # 关闭数据库连接
#         await checkpointer.conn.close()
#         checkpointer = None
#         print("Database connection closed.")

#
#
# async def