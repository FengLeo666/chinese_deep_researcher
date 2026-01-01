from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.constants import START, END
from langgraph.graph import StateGraph

from utils import checkpointer_pool
from method import search_method
from states.search_state import SearchState



async def get_graph(checkpointer: AsyncSqliteSaver=None):
    # 初始化 RedisSaver
    # checkpointer = AsyncRedisSaver(redis_url=CONFIG.REDIS_HOST, ttl={"default_ttl": CONFIG.REDIS_EXPIRE, "refresh_on_read": True})
    if checkpointer is None:
        checkpointer = await checkpointer_pool.get_checkpointer()

    _builder = StateGraph(SearchState)

    _builder.add_edge(START, "get_search_keys")
    _builder.add_node("get_search_keys", search_method.get_search_keys)

    _builder.add_edge("get_search_keys", "topic_search")
    _builder.add_node("topic_search", search_method.topic_search)

    _builder.add_edge("topic_search", "search_results_filter")
    _builder.add_node("search_results_filter",search_method.search_results_filter)

    _builder.add_edge("search_results_filter", "fetch_webpage")
    _builder.add_node("fetch_webpage", search_method.fetch_webpage)

    _builder.add_edge("fetch_webpage", "brief_web_pages")
    _builder.add_node("brief_web_pages", search_method.brief_web_pages)

    _builder.add_edge("brief_web_pages", END)

    return _builder.compile(name="search-agent", checkpointer=checkpointer)


async def get_graph4outline(checkpointer: AsyncSqliteSaver=None):
    # 初始化 RedisSaver
    # checkpointer = AsyncRedisSaver(redis_url=CONFIG.REDIS_HOST, ttl={"default_ttl": CONFIG.REDIS_EXPIRE, "refresh_on_read": True})
    if checkpointer is None:
        checkpointer = await checkpointer_pool.get_checkpointer()

    _builder = StateGraph(SearchState)

    _builder.add_edge(START, "get_search_keys")
    _builder.add_node("get_search_keys", search_method.get_search_keys)

    _builder.add_edge("get_search_keys", "topic_search")
    _builder.add_node("topic_search", search_method.topic_search)

    _builder.add_edge("topic_search", "search_results_filter")
    _builder.add_node("search_results_filter", search_method.search_results_filter)

    _builder.add_edge("search_results_filter", "fetch_webpage")
    _builder.add_node("fetch_webpage", search_method.fetch_webpage)

    _builder.add_edge("fetch_webpage", "outline_filter")
    _builder.add_node("outline_filter",search_method.outline_filter)

    _builder.add_edge("outline_filter", END)

    return _builder.compile(name="outline_search-agent", checkpointer=checkpointer)


# async def start_or_resume_research(input_state: SearchState, thread_id: str, recursion_depth: int,
#                                    checkpointer: AsyncSqliteSaver = None, substate: bool = False):
#     config: RunnableConfig = {
#         'configurable': {
#             'thread_id': thread_id,  # 这里设置线程 ID，确保每次调用的线程有唯一标识
#             # 'checkpoint_ns': 'my_namespace',  # 设置命名空间
#             # 'checkpoint_id': 'checkpoint-001'  # 设置检查点 ID
#         }
#     }
#
#     # 确保传递的是 DirectorState 类，而不是模块
#     graph = await get_graph(checkpointer)
#
#     checkpoint_list = graph.checkpointer.alist(config=config)
#
#     # 获取最晚的检查点继续
#     latest_checkpoint = None
#
#     async for i in checkpoint_list:
#         i_topic = i.checkpoint.get("channel_values").get('main_topic')
#         # 获取每个对象的时间戳
#         ts_str = i.checkpoint.get('ts')
#         # 将时间戳字符串转换为 datetime 对象
#         ts = datetime.fromisoformat(ts_str)
#         # 如果是第一次遍历或当前时间戳更大，更新最大时间戳
#         if ('latest_ts' not in locals() or ts > latest_ts) and i_topic == input_state.main_topic:
#             latest_ts = ts
#             latest_checkpoint = i
#
#     if latest_checkpoint:
#         print(
#             f"[LangGraph] 检测到未完成的研究流程 for {input_state.main_topic}，将从检查点继续执行: {list(latest_checkpoint.checkpoint.get('channel_values').keys())[-1]}")
#
#         config['configurable']["checkpoint"] = latest_checkpoint
#         # 自动恢复 + 继续执行
#         rs = await graph.ainvoke(
#             None,  # resume 时必须传 None，表示从 checkpoint 恢复
#             config=config
#         )
#     else:
#         print(f"[LangGraph] 未检测到 checkpoint，开始新的研究流程: {input_state.main_topic}")
#         rs = await graph.ainvoke(input_state, config=config)
#     return rs["report"].to_markdown()