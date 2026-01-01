import operator
from typing import List, Dict, Annotated

from pydantic import BaseModel, Field

from states.knowledge_state import KnowledgeState
from states.paragraph_state import ParagraphState


#继承DirectorState的参数
class SearchState(ParagraphState):
    # main_topic:str
    search_keys:List[str]=Field(default_factory=list,description="用于调用搜索引擎的搜索句。")
    search_results:List[dict]=Field(default_factory=list,description="#key为url，value为网站标题。")
    # web_pages:Dict[str,str]=Field(default_factory=dict,description="{url网站: page网页源内容}")
    temp_knowledge: Annotated[List[dict], operator.add] = Field(default_factory=list,description="暂存的知识，用于给brief。")


