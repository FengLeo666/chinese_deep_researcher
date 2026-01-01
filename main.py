import asyncio

import CONFIG
import utils
from graphs import director_graph
from rag import knowledge_rag

from utils import checkpointer_pool, sandbox_ploter
from states.director_state import DirectorState

async def _main(state,thread_id):
    async with checkpointer_pool.lifespan():
        with knowledge_rag.lifespan():
            async with sandbox_ploter.lifespan():
                report =await director_graph.start_or_resume_research(input_state=state, thread_id=thread_id)
    return report


if __name__ == "__main__":
    user_input = input("请输入深度研究主题：")
    thread_id = utils.stable_cache_key(user_input)

    d_s = DirectorState(
        main_topic=user_input,
        substate=False,
        recursion_depth=CONFIG.RECURSION_DEPTH,
        session_id=thread_id
    )
    report=asyncio.run(_main(d_s, thread_id=thread_id))
    utils.save_report(user_input, report,CONFIG.REPORT_DIR)
