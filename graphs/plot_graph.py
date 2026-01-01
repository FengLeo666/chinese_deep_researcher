from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.constants import START, END
from langgraph.graph import StateGraph

from method import plot_method
from states.plot_state import PlotsState
from utils import checkpointer_pool


async def get_graph(checkpointer: AsyncSqliteSaver=None):
    # 初始化 RedisSaver
    # checkpointer = AsyncRedisSaver(redis_url=CONFIG.REDIS_HOST, ttl={"default_ttl": CONFIG.REDIS_EXPIRE, "refresh_on_read": True})
    if checkpointer is None:
        checkpointer = await checkpointer_pool.get_checkpointer()

    _builder = StateGraph(PlotsState)

    _builder.add_node("generate_codes",plot_method.generate_codes)
    _builder.add_node("do_plots",plot_method.do_plots)
    _builder.add_node("check_plots",plot_method.check_plots)
    _builder.add_node("to_knowledge",plot_method.to_knowledge)

    _builder.add_edge(START, "generate_codes")
    _builder.add_edge("generate_codes", "do_plots")
    _builder.add_edge("do_plots", "check_plots")
    _builder.add_edge("check_plots", "to_knowledge")
    _builder.add_edge("to_knowledge", END)

    return _builder.compile(name="plot-agent", checkpointer=checkpointer)


