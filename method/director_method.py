import asyncio
import os.path
from typing import List

from langchain_core.messages import SystemMessage, HumanMessage

from utils import llm_client
from graphs import paragraph_graph
from states.director_state import DirectorState
from pydantic import BaseModel, Field
from datetime import datetime, timezone

from states.paragraph_state import ParagraphState
from states.report_state import ReportState
from tqdm.asyncio import tqdm

async def specify_user_needs(state: DirectorState):
    print(f"[specify_user_needs] Running for {state.main_topic}......")
    if state.substate:
        return

    class UserNeed(BaseModel):
        user_needs_claim: List[str] = Field(
            description=f"对于《{state.main_topic}》的深度研究主题，需要询问用户澄清的研究细节。"
                        "如果不需要澄清，返回[]即可。"
                        "以下是一些你可能需要提供的澄清问题：\n"
                        "1. 你希望研究的重点方向是什么？\n"
                        "2. 有哪些特定的假设、前提或背景信息需要明确？\n"
                        "3. 在研究过程中，是否有任何特定的数据源或方法学要求？\n"
                        "4. 你希望得到哪些具体的研究结果或答案？\n"
                        "5. 是否有任何特定的时间框架或优先级要求？\n"
                        "6. 研究是否有针对的时间或周期？\n"
                        "7. 还有其他需要考虑的因素或限制吗？\n\n"
                        "请提供你认为需要澄清的其他研究细节，帮助我们更好地理解你的需求。")

    llm = llm_client.get_llm()
    structured_llm = llm.with_structured_output(UserNeed)

    # str_knowledge = await state.stringify_knowledge()

    # 格式化提示词，用于询问用户哪些研究细节需要澄清
    human_message = HumanMessage(
        content=f"接下来，我们将对《{state.main_topic}》进行一个深度研究。在开始之前，我需要了解你觉得需要我澄清的研究细节。少量精辟的问题即可。"
                f"我已经澄清过了：{state.user_claim}" if state.user_claim else ""
        # f"这是我们已经收集到的资料：{str_knowledge}" if str_knowledge else ""
    )

    old_claim = state.user_claim

    # 使用结构化 LLM 调用，生成问题列表
    result = await structured_llm.ainvoke([state.system_message, human_message])

    if not result.user_needs_claim:
        print("LLM extends no user requirements needs to claim.")
        return
    state.user_claim["系统提问"] = state.user_claim.get("系统提问", "") + "\n".join(result.user_needs_claim)
    state.user_claim["用户回答"] = state.user_claim.get("用户回答", "") + input("\n".join(result.user_needs_claim))

    class UserRequest(BaseModel):
        user_request: str = Field(default_factory=str, description="用户对深度研究的要求，以‘用户要求.......’开头")

    structured_llm = llm.with_structured_output(UserRequest)

    human_message = HumanMessage(
        content=f"根据以下对话内容提取用户对深度研究《{state.main_topic}》的要求：{state.user_claim}"
                f"\n\n这是和用户以前对话的内容：{old_claim}" if old_claim else "")

    result = await structured_llm.ainvoke([human_message])

    now = datetime.now(timezone.utc)

    system_message = SystemMessage(str({
        "System time (UTC)": now,
        "用户要求": result.user_request,
        "研究主题": state.main_topic,
        "对话最终目标": f"生成一篇关于{state.main_topic}的详细报告",
        "注意":"不要进行任何政治敏感的研究"
    }))

    # 返回用户需要澄清的研究细节
    return {
        "user_claim": state.user_claim,
        "system_message": system_message,
    }




async def get_outline(state: DirectorState):
    print(f"[get_outline] Running for {state.main_topic}......")
    llm = llm_client.get_llm()  # 获取 LLM 客户端实例
    structured_llm = llm.with_structured_output(ReportState)  # 使用结构化输出

    # 格式化提示词，询问模型如何将研究主题拆分成多个部分
    formatted_prompt = (
        "请生成一个详细的高级研究大纲，涵盖该研究的各个章节及子章节，大纲层级应该具有最大的深度和广度，并具有结尾章节。\n\n"

        "大纲应包含以下要求：\n\n"

        "1) **大纲结构**：\n"
        "   - 整个大纲应包含至少多级递归结构，每一层应包含以下内容：\n"
        "     - 章节标题（`title`）\n"
        "     - 本节内容概述或导语（`content`），简洁明了地概述该部分的核心内容。\n"
        "     - 可能的子章节（`sections`），每个子章节应包含标题和内容概述。\n"
        "     - 子章节可以有进一步的子章节，递归展开。\n\n"

        "2) **叶子节点要求**：\n"
        "   - 叶子节点是最深层次的章节，这些章节必须包含有效的内容描述（`content`）。\n"
        "   - 叶子节点的 `content` 必须简明扼要地描述该节的核心内容。\n"
        "   - 叶子节点内容必须具体，避免过于笼统或空洞的描述。\n"
        "   - 叶子节点的 `sections` 为空列表。\n\n"

        "3) **分支节点要求**：\n"
        "   - 分支节点的 `ReportState.sections` 应递归展开。\n"
        "   - 分支子节点的 `content` 是可选的，可为空字符串，是用于简明扼要地描述该节的核心内容，或者进行一个导语。\n"
        "   - 分支子节点的 `sections` 是若干个子章节组成的列表（不能为空列表），其中元素为递归的本对象 `ReportState`。\n\n"

        "4) **格式要求**：\n"
        "   - 请确保标题中不包含数字标号（如 ‘1.1’ 或 ‘一、’）。\n"
        "   - `ReportState` 对象的字段应使用标准的 JSON 格式，不允许用字符串引号包裹对象、字典和列表。\n"
        "   - 请确保 `sections` 字段是一个有效的 JSON 格式 **列表结构**，并且其中包含内容，不应为空。\n"
        "   - 避免使用换行符，所有内容应以单一文本流形式返回。\n\n"

        "5) **内容要求**：\n"
        "   - 请确保每个章节和子章节都具有清晰、简洁的核心内容描述，避免过多无关信息或冗长内容。\n"
        "   - 确保所有章节的内容是独立且不重复的。\n\n"
        
        f"可以参考以下网络搜索到的大纲：\n{state.ref_outlines}"
    )
    outline = await structured_llm.ainvoke([state.system_message, HumanMessage(formatted_prompt)])

    # 返回生成的大纲
    return {"outline": outline}


async def _async_return_str(chapter):
    # 直接返回 str(chapter) 的字符串
    return str(chapter)


async def final_writer(state: DirectorState):
    print(f"[final_writer] Running for {state.main_topic}......")

    llm = llm_client.get_llm()
    class Guide(BaseModel):
        content: str

    structured_llm = llm.with_structured_output(Guide)  # 使用结构化输出

    async def generate_report_chapter_recursively(outline: ReportState, route: str) -> ReportState:
        result_state = ReportState(title=outline.title)

        tasks = []  # 用于存储所有任务的列表，先生成子章节

        for i, section in enumerate(outline.sections):
            task = generate_report_chapter_recursively(section, os.path.join(route, section.title))
            tasks.append(task)

        # 使用 asyncio.gather 并行处理所有任务
        if route == state.main_topic:
            results_list = await tqdm.gather(*tasks, desc="Writing Chapters: ", total=len(tasks))
        else:
            results_list = await asyncio.gather(*tasks)

        if len(results_list) == 0:
            paragraph_state=ParagraphState(
                **state.model_dump(),#继承主state所有参数
                chapter=outline.content,
                route=route,
            )

            content = await paragraph_graph.start_or_resume_research(
                input_state=paragraph_state,
                thread_id=paragraph_state.session_id,
            )
        else:
            # 再生成导语
            result_state.sections = results_list
            formatted_prompt = f"我已经完成了'{route}'部分的撰写，现在请帮我撰写'{route}'部分的导语，不用带任何格式，纯文本即可。内容精炼。我撰写的这部分内容如下：{result_state.to_markdown()}"
            guide = await structured_llm.ainvoke([state.system_message, HumanMessage(formatted_prompt)])
            content = guide.content if guide else ""
        result_state.content = content
        return result_state

    # 生成完整报告
    full_report = await generate_report_chapter_recursively(state.outline, state.main_topic)
    full_report.title = state.main_topic + "研究报告"

    # 返回最终生成的报告
    return {"report": full_report}


async def last_paragraph_rewriter(state: DirectorState):
    print(f"[last_paragraph_rewriter] Running for {state.main_topic}......")
    llm = llm_client.get_llm()  # 获取 LLM 客户端实例

    structured_llm = llm.with_structured_output(ReportState)  # 使用结构化输出，指定返回类型为 Outline 类

    state.report.sections[-1] = ReportState()

    # 格式化提示词，询问模型如何将研究主题拆分成多个部分
    formatted_prompt = f"现在我想请你根据全文帮我重写最后一个章节为一个总结性的章节，章节名为{state.report.sections[-1].title}，报告全文内容如下："

    human_message = HumanMessage(content=formatted_prompt + str(state.report))

    # 使用结构化 LLM 调用，生成问题列表
    rs = await structured_llm.ainvoke([state.system_message, human_message])

    state.report.sections[-1] = rs
