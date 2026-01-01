import re
from typing import List,  Any

from pydantic import BaseModel, Field



class ReportState(BaseModel):
    title:str=Field(default_factory=str,description="标题。如果是根节点，则是文章标题。如果是文章分支节点，则是分支标题。")
    content:str=Field(default_factory=str,description="本节内容概述或导语。")
    sections:List["ReportState"]=Field(default_factory=list, description="子章节组成的列表，子章节使用递归的本对象（ReportState）表示，content和sections不允许全为空。")


    # 自定义递归转换为 Markdown 格式
    def to_markdown(self, level: int = 1,pre_title_code:str="",pic_idx:int=1) -> str | tuple[str, int]:
        pre_title_code=f"{pre_title_code}"

        md = f"{'#' * level} {pre_title_code} {self.title}\n"  # 标题部分，按层级叠加 # 号

        content,pic_idx = add_image_captions_and_update_references(self.content,pic_idx)

        md+=content+"\n" if content!="" else ""

        pre_title_code+="." if level > 1 else ""

        for i,paragraph in enumerate(self.sections):  # 递归调用
            sub_md, pic_idx = paragraph.to_markdown(level + 1,f"{pre_title_code}{i+1}",pic_idx)  # 递归调用子章节的 Markdown 转换，增加层级
            md+=sub_md
        if level == 1:
            return md+"\n"
        else:
            return md + "\n",pic_idx




# 在 Pydantic v2 中，必须调用 model_rebuild() 来重新构建模型
ReportState.model_rebuild()



def add_image_captions_and_update_references(text, start_number=1):
    if not text:
        return text, start_number
    # 用来保存图片URL及其题注
    image_urls = {}
    updated_text = text
    current_number = start_number

    # 匹配图片及其描述的正则表达式
    image_pattern = r'!\[([^\]]+)\]\((http[^\)]+)\)'

    # 搜索并为每个图片添加题注
    def add_caption(match):
        nonlocal current_number

        # 获取图片描述和URL
        description = match.group(1)
        url = match.group(2)

        # 创建题注
        caption = f"图{current_number}. {description}"

        # 保存图片的URL和对应的题注
        image_urls[url] = caption

        # 更新文本：添加题注在图片后面，并返回修改后的内容
        updated_image = f'\n\n![{description}]({url})\n\n{caption}\n\n'

        # 增加编号
        current_number += 1

        return updated_image

    # 为所有图片添加题注
    updated_text = re.sub(image_pattern, add_caption, updated_text)

    # 搜索并更新所有图片引用：替换为图n
    def update_reference(match):
        url = match.group(1)
        if url in image_urls:
            # 如果图片URL已被记录，替换为图n
            return f"图{list(image_urls.keys()).index(url) + 1}"
        return match.group(0)

    # 更新文本中的图片引用
    updated_text = re.sub(r'\[\]\((http[^\)]+)\)', update_reference, updated_text)

    # 返回更新后的文本以及下一个空闲的编号
    return updated_text, current_number