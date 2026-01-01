import json
import operator
from datetime import datetime
from typing import Dict, Set, Annotated, List

from pydantic import BaseModel, Field

import utils


class KnowledgeState(BaseModel):
    topic:Annotated[List[str],operator.add] = Field(default_factory=list, description="本次研究主题的列表")#知识属于哪个（子）
    url: str = Field(default_factory=str,description="知识来源。")
    is_useful: bool = Field(default_factory=bool,description="是否和主题相关。")
    # is_related_to_chapter:bool = Field(default_factory=bool,description="是否和本章节内容相关，该知识是否有助于撰写本章节内容。")
    time: str = Field(default_factory=str,description="推测知识来源时间。格式为%Y-%m-%d。如果实在找不到时间，默认当前时间。")
    summary: str = Field(default_factory=str,description="知识关键信息总结，包括整篇文章的摘要，包括其中比较新颖的观点或信息。is_useful为False时，此处为空字符串。")
    is_picture:bool = Field(default_factory=bool,description="这个知识对象是不是图片形式的，如果是，那么summary中以![图片描述](图片链接)存在。")
    site_name:str=Field(default_factory=str,description="域名")




    def get_time(self):
        """
        将 time 字段转换为 datetime 对象
        """
        return utils.str2time(self.time)  # 假设时间格式为 YYYY-MM-DD

    def serialize_topic(self):
        """将 topic 转换为字符串，如果 topic 是列表，转换为 JSON 字符串"""
        # if type(self.topic) is str:
        #     self.topic={self.topic}
        return json.dumps(list(self.topic))  # 将列表转为字符串

    def load_topic(self, serialized_topic:str):
        """将 topic 字符串反序列化为原始格式（列表）"""
        try:
            # 尝试将字符串反序列化为列表
            self.topic = list(json.loads(serialized_topic))
        except (TypeError, json.JSONDecodeError):
            pass  # 如果不是字符串格式，保持原样

if "__main__" == __name__:

    knowledge = KnowledgeState()
    f=knowledge.model_fields["topic"]
    print(str(knowledge))
