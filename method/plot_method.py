import asyncio
import os
from dataclasses import asdict
from typing import List

from llm_sandbox import ExecutionResult
from pydantic import BaseModel, Field

import CONFIG
import utils
from states.knowledge_state import KnowledgeState
from states.plot_state import PlotsState,  PlotPending
from utils import sandbox_ploter, llm_client
from utils.llm_client import MultimodalityMessage, construct_base64img_url


# async def target_plot(state: PlotsState):
#     if state.para_knowledge:
#         print(f"[target_plot] Running for {state.route}......")
#     else:
#         warnings.warn(f"[target_plot] No knowledge for plot with path: {state.route}")
#
#     llm = llm_client.get_llm()
#     structured_llm = llm.with_structured_output(PlotMissions)
#
#     prompt = (
#         f"你是一个严格的绘图任务审核官。\n"
#         f"请帮我从以下知识中总结出关于'{state.route}'的绘图任务及数据。\n"
#         f"只要非常优质的，且画图出来好看的数据，如果没有优质数据，可以返回[]。\n"
#         f"知识如下：{state.para_knowledge}"
#     )
#
#     result:PlotMissions = await structured_llm.ainvoke(prompt)
#
#     if not result:
#         return
#
#
#     plots:List[PlotResults] = [PlotResults(**i.model_dump()) for i in result.missions]
#
#     return {"plots",plots}


async def generate_codes(state: PlotsState):
    print(f"[generate_codes] Running for {state.route}......")

    llm = llm_client.get_llm()

    structured_llm = llm.with_structured_output(PlotPending)

    base_prompt = (
        f"你是一个严格的绘图任务审核官和绘图代码程序员。\n"
        f"你需要先审核知识中的数据是否能满足以下要求：\n"
        f"1.以下知识是否存在优质的数据。\n"
        f"2.这些数据能否画一张精美的plot。\n"
        f"3.数据和画出来的plot内容是否和{state.route}相关。\n"
        f"4.画图后的效果是否比直接文字表达更好。\n"
        f"如果存在满足以上要求的数据请挑选出最优质的效果最好的，帮我生成其画图代码（用show方法输出）\n"
        f"给出code和libraries即可，其他的参数是可选的引用知识的参数，继承知识的即可。\n"
        f"知识如下：\n"
    )

    llm_result:PlotPending = await structured_llm.ainvoke(base_prompt+str(state.para_knowledge))

    return llm_result.model_dump()

async def do_plots(state: PlotsState):
    if not state.code:return
    print(f"[do_plots] Running for {state.route}......")

    e_r:ExecutionResult  = await sandbox_ploter.execute_code(state.code,state.libraries)

    return asdict(e_r)

async def check_plots(state: PlotsState):
    if not state.plots:return
    print(f"[check_plots] Running for {state.route}......")
    llm = llm_client.get_llm(visual=True)
    class PlotOK(BaseModel):
        reason:str = Field(default_factory=str,description="如果不ok，不ok的原因是什么。")
        ok: bool = Field(default_factory=bool, description="该plot是否OK。")
    structured_llm = llm.with_structured_output(PlotOK)
    prompt=(
        "你是一个严格的plot检查官。你需要检查这个程序跑出来的plot是否满足以下标准：\n"
        "1.plot排版是否混乱。\n"
        "2.有没有符合用户要求。\n"
        "3.是否是一张上得了台面的plot。\n"
        "如果上述条件都满足，请返回ok=True。\n"
        f"用户要求如下：{state.description}\n"
        f"代码如下{state.code}\n"
    )

    img_urls=[construct_base64img_url(plot.content_base64,plot.format.value) for plot in state.plots]

    results = await structured_llm.ainvoke(MultimodalityMessage(prompt,*img_urls))

    if results.ok:
        return
    else:
        return {"plots":[]}



async def to_knowledge(state: PlotsState):
    if not state.plots:return
    print(f"[to_knowledge] Running for {state.route}......")

    k_s:List[KnowledgeState] = list()

    for plot in state.plots:
        k_s.append(KnowledgeState())
        file_name=state.description+"."+plot.format.value
        file_path=os.path.join(CONFIG.REPORT_DIR,file_name)
        utils.base64_to_image_file(plot.content_base64,file_path)

        k:KnowledgeState = KnowledgeState(**state.model_dump())
        k.url=os.path.join(".",file_name)
        k.summary=state.description
        k.is_picture=True

        k_s.append(k)

    # await state.add_knowledge(k_s)

    return {"para_knowledge":state.para_knowledge+k_s}





