import operator
from datetime import datetime,timezone
from typing import Annotated, List, Dict, Optional, ClassVar

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

import CONFIG
from states.knowledge_state import KnowledgeState
from states.report_state import ReportState
from langchain_core.messages import SystemMessage

class DirectorState(BaseModel):
    main_topic:str=Field(description="深度研究主题。")
    system_message:SystemMessage=Field(default=SystemMessage(content=str({"System time (UTC)": datetime.now(timezone.utc)})),description="研究相关meta 系统提示内容，比如现在的时间。")
    substate:bool=Field(default_factory=bool,description="是否属于某个深度研究主题的子主题。")
    user_claim:Dict[str,str]=Field(default_factory=dict,description="用户对研究过程中疑问的澄清。")
    outline:Optional[ReportState]=Field(default=None,description="深度研究报告的大纲，及其递归定义的章节。")
    ref_outlines:List[KnowledgeState]=Field(default_factory=list,description="网络上用于模型生成大纲前的参考。")
    report:Optional[ReportState]=Field(default=None,description="最后输出的深度研究报告，及其递归定义的章节及其内容。")
    recursion_depth:int=Field(default_factory=int,description="递归(迭代)调用控制。")#递归调用控制
    session_id:str=Field(description="用于获取session的knowledge。")

    async def get_knowledge(self)->List[KnowledgeState]:
        return await self.get_knowledge_conditional()

    async def add_knowledge(self,knowledge:KnowledgeState|List[KnowledgeState]):
        if not isinstance(knowledge,list):
            knowledge = [knowledge]
        from rag.knowledge_rag import get_context_knowledge_database
        ck = await get_context_knowledge_database(self.session_id)
        await ck.add_knowledge(knowledge)

    async def get_knowledge_conditional(self,query:str=None,start_time:str=None,end_time:str=None,top_k:int=None)->List[KnowledgeState]:
        if not query:
            query = self.main_topic
        from rag.knowledge_rag import get_context_knowledge_database
        ck = await get_context_knowledge_database(self.session_id)
        if top_k is None:
            top_k=CONFIG.RAG_MAX_TOP_K
        return await ck.search_knowledge(query,top_k=top_k,start_time=start_time,end_time=end_time)

    async def stringify_knowledge(self,*args,knowledge:Optional[List[KnowledgeState]],**kwargs):
        if knowledge is None:
            knowledge = await self.get_knowledge_conditional(*args,**kwargs)
        rs=""
        for i,knowledge in enumerate(knowledge):
            rs+=f"Info{i}:({knowledge})\n"
        return rs

    async def get_by_topic(self, *args, topic: Optional[str], **kwargs) -> List[KnowledgeState]:
        if topic is None:
            topic = self.main_topic
        from rag.knowledge_rag import get_context_knowledge_database
        ck = await get_context_knowledge_database(self.session_id)
        return ck.search_by_topic(*args,topic=topic,**kwargs)

    async def get_by_url(self,url:str) -> Optional[KnowledgeState]:
        from rag.knowledge_rag import get_context_knowledge_database
        ck = await get_context_knowledge_database(self.session_id)
        return ck.get_by_url(url)

    async def len_knowledge(self) -> int:
        from rag.knowledge_rag import get_context_knowledge_database
        ck = await get_context_knowledge_database(self.session_id)
        return len(ck)



if __name__=="__main__":
    ds = DirectorState(
        **{
            "main_topic": "user_input",
            "recursion_depth": 3,
            "substate": False
        }
    )

