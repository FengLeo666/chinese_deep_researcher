from typing import List

from langchain_core.messages.tool import ToolCall
from pydantic import BaseModel, Field

from states.director_state import DirectorState
from states.knowledge_state import KnowledgeState


class ParagraphState(DirectorState):
    chapter: str=Field(default=str,description="章节描述。")
    route: str=Field(default=str,description="章节路径。")
    para_knowledge:List[KnowledgeState]=Field(default_factory=list,description="本地知识搜索结果。")
    # confuse_consumer:int=Field(default_factory=int,description="用于控制解决问题的迭代次数。")#用recursion_depth
    confuses: List[str] = Field(default_factory=list, description="目前疑惑的问题。")
    tool_calls: list[ToolCall] = Field(default_factory=list,description="工具调用。")
