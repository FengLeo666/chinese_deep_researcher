from datetime import datetime

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.constants import START, END
from langgraph.graph import StateGraph

import CONFIG
from utils import checkpointer_pool
from graphs import search_graph, plot_graph
from method import paragraph_method
from states.paragraph_state import ParagraphState


async def start2where(state):
    counter=sum([1 for k in state.para_knowledge if not k.is_picture ])
    if counter>CONFIG.MAX_PARAGRAPH_KNOWLEDGE:
        return "plot_graph"
    elif state.recursion_depth > 0:
    # 异步执行 `len_knowledge` 方法
        return "get_local_knowledge" if await state.len_knowledge() else "get_confuses"
    else:
        return "plot_graph"

async def get_graph(checkpointer: AsyncSqliteSaver=None):
    # 初始化 RedisSaver
    # checkpointer = AsyncRedisSaver(redis_url=CONFIG.REDIS_HOST, ttl={"default_ttl": CONFIG.REDIS_EXPIRE, "refresh_on_read": True})
    if checkpointer is None:
        checkpointer = await checkpointer_pool.get_checkpointer()

    _builder = StateGraph(ParagraphState)

    _builder.add_conditional_edges(START,  start2where)
    _builder.add_node("get_local_knowledge",paragraph_method.get_local_knowledge)

    _builder.add_edge("get_local_knowledge", "get_confuses")
    _builder.add_node("get_confuses", paragraph_method.get_confuses)

    _builder.add_conditional_edges("get_confuses", lambda state: "search_graph" if state.confuses else "chapter_writer" )

    _builder.add_node("search_graph", await search_graph.get_graph(checkpointer=checkpointer))

    _builder.add_edge("search_graph", "llm_chapter_knowledge_filter")
    _builder.add_node("llm_chapter_knowledge_filter", paragraph_method.llm_chapter_knowledge_filter)

    _builder.add_conditional_edges("llm_chapter_knowledge_filter", start2where)

    _builder.add_node("plot_graph",await plot_graph.get_graph(checkpointer=checkpointer))

    _builder.add_node("chapter_writer",paragraph_method.chapter_writer)

    _builder.add_edge("chapter_writer", END)

    return _builder.compile(name="paragraph-agent", checkpointer=checkpointer)


async def start_or_resume_research(input_state: ParagraphState, thread_id: str=None,checkpointer: AsyncSqliteSaver = None)->str:

    topic=input_state.route if input_state.route else input_state.main_topic

    config: RunnableConfig = {
        'configurable': {
            'thread_id': topic,  # 这里设置线程 ID，确保每次调用的线程有唯一标识
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
        i_topic = i.checkpoint.get("channel_values").get('route')#paragraph用route搜索
        # 获取每个对象的时间戳
        ts_str = i.checkpoint.get('ts')
        # 将时间戳字符串转换为 datetime 对象
        ts = datetime.fromisoformat(ts_str)
        # 如果是第一次遍历或当前时间戳更大，更新最大时间戳
        if ('latest_ts' not in locals() or ts > latest_ts) and i_topic == topic:
            latest_ts = ts
            latest_checkpoint = i

    if latest_checkpoint:
        print(
            f"[LangGraph] 检测到未完成的研究流程 for {topic}，"
            f"将从检查点继续执行paragraph_graph: {list(latest_checkpoint.checkpoint.get('channel_values').keys())[-1]}")

        config['configurable']["checkpoint"] = latest_checkpoint
        # 自动恢复 + 继续执行
        rs = await graph.ainvoke(
            None,  # resume 时必须传 None，表示从 checkpoint 恢复
            config=config
        )
    else:
        print(f"[LangGraph] 未检测到 checkpoint，开始新的研究流程: {input_state.route if input_state.route else input_state.main_topic}")
        rs = await graph.ainvoke(input_state, config=config)
    return rs["report"]