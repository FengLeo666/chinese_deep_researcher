import logging
from typing import List

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from utils import llm_client
from rag import knowledge_rag
from states.paragraph_state import ParagraphState


# 多次迭代需要再搜，万一其他线程拿到了相关信息
async def get_local_knowledge(state: ParagraphState):
    print(f"[get_local_knowledge] Running for {state.route} ......")
    llm = llm_client.get_llm()  # 获取大语言模型
    tool_llm = llm.bind_tools([knowledge_rag.get_knowledge_conditional_tool_call])

    formatted_prompt = (
        f"为了进行《{state.route}》报告生成，我们需要进行RAG检索\n"
        f"建议多次调用，小top_k。\n"
        f"请帮我调用检索方法,可以用多个query搜索。\n"
        f"{state.chapter}。\n"
        f"{"这是现有的疑惑：" + str(state.confuses) if state.confuses else ""}\n"
        f"{"如果已有知识已经比较完整，你可以不调用工具，这是已经查到的知识：" + str(state.para_knowledge) if state.para_knowledge else ""}"
    )

    human_message = HumanMessage(content=formatted_prompt)

    response = await tool_llm.ainvoke([state.system_message, human_message])

    para_knowledge = list()
    for tool_call in response.tool_calls:
        if tool_call['name'] == 'get_knowledge_conditional_tool_call':
            args = tool_call["args"]
            args["session_id"] = state.session_id
            # knowledge_rag.get_knowledge_conditional_tool_call(tool_call)#test 直接返回
            tool_rs = await knowledge_rag.get_knowledge_conditional(**args)
            para_knowledge += tool_rs
        else:
            logging.info(f"Unexpected tool call from LLM: {tool_call}.")

    uniquer = set()
    filtered_para_knowledge = []

    for k in [*para_knowledge, *state.para_knowledge]:
        url = k.url.strip()
        if url not in uniquer:
            uniquer.add(url)
            filtered_para_knowledge.append(k)

    return {"para_knowledge": filtered_para_knowledge}


async def get_confuses(state: ParagraphState):
    print(f"[get_confuses] Running for {state.route}......")

    # return
    class Doubt(BaseModel):
        confuses: List[str] = Field(description="疑惑的知识点（需要进一步研究或网络检索的）。")

    llm = llm_client.get_llm()
    structured_llm = llm.with_structured_output(Doubt)

    # 格式化提示词，用于询问用户哪些研究细节需要澄清
    formatted_prompt = (
        f"现在要进行{state.route}章节的撰写，{state.chapter}"
        f"{"现在我们已经查询到了一些资料，" if state.para_knowledge else ""}"
        f"如果我们现在要进行报告生成，是否还有知识是未知的，需要进一步研究的。"
        f"如果有，请告诉我疑惑的点，如果没有，请返回[]。"
        # f"返回的疑惑点请不要包含过于政治敏感的词。"
        f"资料如下：\n{state.para_knowledge if state.para_knowledge else ""}")

    human_message = HumanMessage(content=formatted_prompt)

    # 使用结构化 LLM 调用，生成问题列表
    result = await structured_llm.ainvoke([state.system_message, human_message])

    return {"confuses": result.confuses}


async def llm_chapter_knowledge_filter(state: ParagraphState):
    print(f"[llm_chapter_knowledge_filter] Running for {state.route}......")
    if not state.para_knowledge:
        return {}

    class Useless(BaseModel):
        useless: List[str] = Field(description="无用知识的url，我会在后续生成时删除这些知识。")

    llm = llm_client.get_llm()
    structured_llm = llm.with_structured_output(Useless)

    # 格式化提示词，用于询问用户哪些研究细节需要澄清
    formatted_prompt = (
        f"我准备进行{state.route}章节的撰写，{state.chapter}\n"
        # f"{"目前的主要疑惑有"+str(state.confuses) if state.confuses else ""}\n"
        f"我需要你找到下列的知识中哪些知识对这个章节的撰写是无用的。"
        f"如果有，请告诉我这些知识的url，如果没有，请返回[]。"
        # f"返回的疑惑点请不要包含过于政治敏感的词。"
        f"知识如下：\n{state.para_knowledge}")

    human_message = HumanMessage(content=formatted_prompt)

    # 使用结构化 LLM 调用，生成问题列表
    result = await structured_llm.ainvoke([state.system_message, human_message])

    # 根据result过滤
    return {"para_knowledge": [k for k in state.para_knowledge if k.url.strip() not in [u.strip() for u in
                                                                                        result.useless]] if result else state.para_knowledge}


async def chapter_writer(state: ParagraphState):
    llm = llm_client.get_llm()

    class Section(BaseModel):
        content: str = Field(default_factory=str, description="章节内容")

    structured_llm = llm.with_structured_output(Section)

    prompt = (
        f"现在请帮我生成'{state.route}'部分的最终内容。"
        f"{state.chapter}\n"
        f"整理已知信息，合并相关内容，保证是一个有逻辑的专业写作风格。"
        "整个章节请使用无格式纯文本流返回（不要用Markdown）。\n"
        "请不要包含任何标题（包括本小结标题）。\n"
        "如果生成的章节内容部分来自参考资料，请注意标注来源。\n"
        "标注链接时使用一个方括号标识网站主要部分（网站名），紧接着使用圆括号标出链接，然后用个中文括号把它们包裹起来，如果一个地方有多处引用，在中文括号内用逗号隔开，"
        "例如：（[百度](https://www.baidu.com/s?tn=68_7_oem_d)，[baidu.com](https://www.baidu.com/s?tn=68018em_d)）。\n"
        "若要在文本中携带图片(必须和内容高度相关)，请用![图片描述及详细题注](图片URL)。"
        "并可以在其他文本处添加引用，请用如图[](图片URL)所示。\n"
        "如果图片需要引用来源，请在题注内引用。\n"
        "如果有一系列数据，可以使用表格总结，其格式为：\n"
        "| 列1 | 列2 | 列3 |\n"
        "|-----|-----|-----|\n"
        "| 数据1 | 数据2 | 数据3 |\n"
        "| 数据4 | 数据5 | 数据6 |\n"
        "如果需要在文档中使用加粗，请使用`**加粗文字**`。\n"
        "如果需要使用斜体，请使用`*斜体文字*`。\n"
        # "引用某部分内容时，请使用`> 引用内容`的格式。\n"
        "如果需要列出项目，请使用无序列表格式（如：`- 项目`）或有序列表格式（如：`1. 项目`）。\n"
        "例如：\n"
        "- 项目1\n"
        "- 项目2\n"
        "1. 项目A\n"
        "2. 项目B\n"
        "代码块应使用三个反引号（``）包裹，例如：\n"
        "```\n"
        "print('Hello, World!')\n"
        "```\n"
        "生成内容请不要包含本章节以外的信息,特别是同级兄弟节点。\n"
        "但可选承上启下的内容。\n"
        f"以下是可以参考的资料(如果其中有知识不相关，可以不使用)：{state.para_knowledge}\n"
        f"大纲：{state.outline.to_markdown()}\n"
    )

    section = await structured_llm.ainvoke(prompt)

    return {"report": section.content}
