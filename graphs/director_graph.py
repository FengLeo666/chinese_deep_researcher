from datetime import datetime

from langchain_core.runnables import RunnableConfig
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from utils import checkpointer_pool
from graphs import search_graph
from states.director_state import DirectorState
from method import director_method

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver





async def get_graph(checkpointer: AsyncSqliteSaver=None):
    # 初始化 RedisSaver
    # checkpointer = AsyncRedisSaver(redis_url=CONFIG.REDIS_HOST, ttl={"default_ttl": CONFIG.REDIS_EXPIRE, "refresh_on_read": True})
    if checkpointer is None:
        checkpointer = await checkpointer_pool.get_checkpointer()

    _builder = StateGraph(DirectorState)

    # 条件判断：如果 should_stop 为 True
    _builder.add_edge(START, "specify_user_needs")
    _builder.add_node("specify_user_needs", director_method.specify_user_needs)

    _builder.add_edge("specify_user_needs", "search4outline")
    _builder.add_node("search4outline", await search_graph.get_graph4outline())

    _builder.add_edge("search4outline", "get_outline")
    _builder.add_node("get_outline", director_method.get_outline)

    _builder.add_edge("get_outline", "final_writer")
    _builder.add_node("final_writer", director_method.final_writer)

    _builder.add_edge("final_writer", "last_paragraph_rewriter")
    _builder.add_node("last_paragraph_rewriter", director_method.last_paragraph_rewriter)

    _builder.add_edge("last_paragraph_rewriter", END)

    return _builder.compile(name="deep-research-agent", checkpointer=checkpointer)

    # _checkpointer_ctx = AsyncSqliteSaver.from_conn_string("resources/checkpoints.db")
    # checkpointer = await _checkpointer_ctx.__aenter__()
    # # return _builder.compile(name="deep-research-agent", checkpointer=checkpointer)
    # async with AsyncSqliteSaver.from_conn_string("resources/checkpoints.db") as checkpointer:
    #     # 构建图时指定 checkpointer
    #     await checkpointer.setup()
    #     return _builder.compile(name="deep-research-agent",checkpointer=checkpointer)


# graph = _builder.compile(name="deep-research-agent")


async def start_or_resume_research(input_state: DirectorState, thread_id: str=None,checkpointer: AsyncSqliteSaver = None)->str:

    thread_id = input_state.main_topic
    config: RunnableConfig = {
        'configurable': {
            'thread_id': thread_id,  # 这里设置线程 ID，确保每次调用的线程有唯一标识
            # 'checkpoint_ns': 'my_namespace',  # 设置命名空间
            # 'checkpoint_id': 'checkpoint-001'  # 设置检查点 ID
        }
    }

    # 确保传递的是 DirectorState 类，而不是模块
    graph = await get_graph(checkpointer)

    checkpoint_list = graph.checkpointer.alist(config=config)

    # 获取最晚的检查点继续
    latest_checkpoint = None

    async for i in checkpoint_list:
        i_topic = i.checkpoint.get("channel_values").get('main_topic')
        # 获取每个对象的时间戳
        ts_str = i.checkpoint.get('ts')
        # 将时间戳字符串转换为 datetime 对象
        ts = datetime.fromisoformat(ts_str)
        # 如果是第一次遍历或当前时间戳更大，更新最大时间戳
        if ('latest_ts' not in locals() or ts > latest_ts) and i_topic == input_state.main_topic:
            latest_ts = ts
            latest_checkpoint = i

    if latest_checkpoint:
        print(
            f"[LangGraph] 检测到未完成的研究流程 for {input_state.main_topic}，"
            f"将从检查点继续执行director_graph: {list(latest_checkpoint.checkpoint.get('channel_values').keys())[-1]}")

        config['configurable']["checkpoint"] = latest_checkpoint
        # 自动恢复 + 继续执行
        rs = await graph.ainvoke(
            None,  # resume 时必须传 None，表示从 checkpoint 恢复
            config=config
        )
    else:
        print(f"[LangGraph] 未检测到 checkpoint，开始新的研究流程: {input_state.main_topic}")
        rs = await graph.ainvoke(input_state, config=config)
    return rs["report"].to_markdown()

