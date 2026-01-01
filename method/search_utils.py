import json
import logging
import random
import re
from functools import wraps
from typing import List

import magic
import pythoncom
import requests
import win32com.client
from PIL import Image
from aiohttp import InvalidUrlClientError

import CONFIG
import aiohttp
import backoff
import asyncio
import trafilatura
import fitz  # PyMuPDF
import pandas as pd
import os
from io import BytesIO
from docx import Document

import utils


@backoff.on_exception(backoff.expo, (aiohttp.ClientError, aiohttp.client_exceptions.ClientResponseError),
                      max_tries=CONFIG.MAX_TRY, raise_on_giveup=True)
@utils.rate_limited(qpm=60)
async def do_search_payload(url: str, headers: dict, payload: str) -> List[dict]:
    """
    重试请求逻辑，最多重试 max_retries 次，每次失败后等待 retry_delay 秒再重试。
    使用 backoff 库进行重试。

    :param url: 请求的 URL
    :param headers: 请求的头信息
    :param payload: 请求的有效载荷
    :return: 请求的结果字典
    """
    # async with _retry_request_semaphore:  # 限制并发请求数
    async with aiohttp.ClientSession() as session:
        # 发送请求
        async with session.post(url, headers=headers, data=payload) as response:
            response.raise_for_status()  # 如果请求失败，将引发 HTTPError
            results = await response.json()

            if results.get('code') == 200 and results.get('data'):
                data = results.get('data')
                # 返回文本
                web_pages = data.get('webPages')
                web_pages = web_pages.get('value') if web_pages else []
                # return web_pages
                # 返回图像
                images = data.get('images')
                images = images.get('value') if images else []

                rs = [*web_pages, *images]

                return rs
            else:
                raise Exception(f"Failed to search, code:{results.get("code")}, payload: {payload}")


@backoff.on_exception(backoff.expo, (
    requests.exceptions.HTTPError,
    requests.exceptions.Timeout,
    requests.exceptions.ConnectionError,
    asyncio.TimeoutError,
    aiohttp.ClientConnectorCertificateError
), max_tries=3)
async def _fetch_webpage_text_async(
        page: dict,
        timeout: int = 30,
        output_format="txt",
        session=None
) -> dict:
    """
    抓取并解析网页或文件内容（异步封装），支持网页、PDF、Word、Excel 等文件类型。
    使用 backoff 库来自动处理重试。

    :param page: 目标网页或文件 page信息字典
    :param timeout: 每次抓取的超时时间（秒）
    :param output_format: 输出格式，默认为 txt
    :return: 抽取后的正文内容；失败时返回空字符串 ""
    """
    if session is None:raise RuntimeError("Use 'async with aiohttp.ClientSession() as session' from outer.")
    if page.get("is_picture") and not await check_image(session, page.get("contentUrl")):
        raise requests.exceptions.RequestException(
            f"[Fetch] Picture is not available: {page.get("contentUrl")}")  # 不重试
    url = page.get('hostPageUrl') if page.get("is_picture") else page.get("url")  # 是图片就拿来源页面的内容
    # 异步抓取网页内容
    async with session.get(url,headers=CONFIG.HEADERS, timeout=timeout) as response:
        try:
            response.raise_for_status()
        except (
                aiohttp.ClientResponseError,
                aiohttp.ClientConnectorError,
                aiohttp.ClientConnectorSSLError,
        ) as e:
            logging.info(f"[Fetch Web] Fail for {url}: {e}")
            return {}#使用搜索引擎的总结
        if response.status != 200:
            return {}

        content = await response.read()

        # 使用 python-magic 来判断文件类型
        file_type = magic.from_buffer(content, mime=True)
        # print(f"Detected file type: {file_type}")  # 可以打印检测到的文件类型

        try:
            rs = {}
            if 'application/pdf' in file_type:
                # PDF 文件使用 PyMuPDF 解析
                rs = {"text": await _extract_text_from_pdf(content)}
            elif 'application/msword' in file_type or 'application/vnd.openxmlformats-officedocument.wordprocessingml.document' in file_type:
                # Word 文件使用 python-docx 解析
                rs = {"text": await _extract_text_from_word(content)}
            elif 'application/vnd.ms-excel' in file_type or 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' in file_type:
                # Excel 文件使用 pandas 解析
                rs = {"text": await _extract_text_from_excel(content)}
            else:  # 都不是，尝试用html解析
                if 'text/html' not in file_type:
                    logging.info(f"[Fetch Web] Force decode {file_type} from {url} with html decoder.")
                obj = await _extract_text_from_html(content, url, output_format)
                if obj:
                    rs = json.loads(obj)
            if page.get("is_picture"):
                return {"hostPageContent": rs.get("text")}
            else:
                return rs

        except Exception as e:
            logging.info(f"[Fetch] Page cannot resolve for {url}: {e}")
            return {}


async def _extract_text_from_html(content: bytes, url: str, output_format: str) -> str:
    """从 HTML 内容中提取文本"""
    downloaded = content.decode('utf-8', errors='ignore')
    extracted = trafilatura.extract(
        downloaded,
        url=url,
        output_format=output_format,
        include_comments=False,
        include_tables=True,
        include_images=True,
        with_metadata=True,
        deduplicate=True,
        url_blacklist={"zhihu.com", 'doc88.com'}
    )
    return extracted if extracted else ""


async def _extract_text_from_pdf(content: bytes) -> str:
    """从 PDF 文件中提取文本，使用 PyMuPDF"""
    try:
        # 打开 PDF 文件
        doc = fitz.open(stream=content, filetype="pdf")

        # 提取所有页面的文本
        text = ""
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text += page.get_text("text")  # 提取页面文本
        return text.strip()

    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""


async def _extract_text_from_word(content: bytes) -> str:
    """从 Word 文件中提取文本，无需依赖文件后缀名"""
    try:
        # 尝试直接用 python-docx 解析 .docx 文件
        doc = Document(BytesIO(content))
        text = ""
        for para in doc.paragraphs:
            text += para.text + "\n"
        return text.strip()

    except Exception as e:
        # print(f"Failed to extract text with python-docx, attempting to handle as .doc: {e}")
        # 如果转换失败，尝试用 antiword 提取 .doc 文件的文本
        temp_file_path = f"resources/temp_file{random.randint(0, 9999999999999999999)}.doc"
        try:
            with open(temp_file_path, 'wb') as f:
                f.write(content)
            doc_text = await doc_extractor.extract_text(temp_file_path)
            return doc_text
        except Exception as e:
            print(f"[extractor] Error while extracting text from .doc file:{e}")
            return ""
        finally:
            os.remove(temp_file_path)


_doc_semaphore = asyncio.Semaphore(1)  # 最多允许请求并发


def _doc_synchronized(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        async with _doc_semaphore:  # 限制并发请求数
            return await func(*args, **kwargs)

    return wrapper


class DocExtractor:
    def __init__(self):
        self.word = None  # 不立即创建 Word 对象

    @_doc_synchronized
    async def extract_text(self, url) -> str:
        return await asyncio.to_thread(self._extract_text, url)

    def _extract_text(self, url) -> str:
        if not os.path.exists(url):
            return ""

        pythoncom.CoInitialize()  # 初始化 COM
        try:
            if not self.word:  # 懒加载
                self.word = win32com.client.Dispatch("Word.Application")
                self.word.Visible = False  # 不显示 Word 窗口

            # 打开文档
            self.doc = self.word.Documents.Open(os.path.abspath(url))
            text = self.doc.Content.Text  # 获取文档的文本内容
            self.doc.Close()
            return text.replace("\r", "\n").strip()
        finally:
            pythoncom.CoUninitialize()  # 结束 COM 使用

    def __del__(self):
        # 关闭文档和 Word 应用程序
        if hasattr(self, "doc"):
            self.doc.Close()
        self.word.Quit()


doc_extractor = DocExtractor()
if __name__ == "__main__":
    print(asyncio.run(doc_extractor.extract_text("../resources/temp_file3157325990017675156.doc")))


async def _extract_text_from_excel(content: bytes) -> str:
    """从 Excel 文件中提取文本"""
    try:
        # 使用 pandas 读取 Excel 文件
        df = pd.read_excel(BytesIO(content), engine='openpyxl')
        # 返回所有表格的文本内容
        text = ""
        for col in df.columns:
            text += " | ".join(map(str, df[col].values)) + "\n"
        return text.strip()
    except Exception as e:
        logging.info(f"[Excel] Error extracting text from Excel: {e}")
        return ""


async def fetch_webpage_async(page: dict, timeout: int = 30,session=None) -> dict:
    redis_key = f"page:{page['url']}"
    redis_client = CONFIG.get_redis_client()
    cache_page = await redis_client.get(redis_key)
    if cache_page:
        # print("[Fetch] Returning cached result from Redis for page:", page['url'])
        return json.loads(cache_page)

    try:
        extracted = await _fetch_webpage_text_async(page, timeout=timeout, output_format="json",session=session)
    except Exception as e:
        logging.log(logging.ERROR, e)
        await redis_client.set(redis_key, json.dumps({}))
        return {}

    page.update(extracted)

    await redis_client.set(redis_key, json.dumps(page))
    return page


# _web_search_semaphore = asyncio.Semaphore(1)
# def _web_search_synchronized(func):
#     @wraps(func)
#     async def wrapper(*args, **kwargs):
#         async with _web_search_semaphore: # 限制并发请求数
#             return await func(*args, **kwargs)
#     return wrapper
#
# @_web_search_synchronized
async def web_search(query: str, categories: str = 'general') -> List[dict]:
    """
    执行网页搜索，支持重试机制（异步版本）。首先检查Redis是否有缓存结果，如果没有，则执行搜索。

    :param query: 搜索的查询词
    :param categories: 搜索类别，默认为 'general'
    :return: 搜索结果列表
    """
    # 生成 Redis 缓存的 key，这里可以根据需要调整
    redis_key = f"search_result:{query}:{categories}"

    redis_client = CONFIG.get_redis_client()

    # 尝试从 Redis 获取缓存的结果
    cached_result = await redis_client.get(redis_key)
    if cached_result:
        print("Returning cached result from Redis for query:", query)
        return json.loads(cached_result)

    url = CONFIG.BOCHA_SEARCH_URL
    payload = json.dumps({
        "query": query,
        "summary": True,  # 是否需要摘要
        "count": CONFIG.MAX_SEARCH_RESULT  # 返回最多10个结果
    })

    headers = {
        'Authorization': CONFIG.BOCHA_API_KEY,  # 使用实际的 Bearer Token
        'Content-Type': 'application/json'
    }

    print("Searching for:", query, "......")

    # 调用 _retry_request 来执行搜索
    result_list = await do_search_payload(url, headers, payload)

    # 将搜索结果缓存到 Redis
    await redis_client.set(redis_key, json.dumps(result_list), ex=CONFIG.REDIS_EXPIRE)

    return result_list


async def extract_images_from_markdown(markdown_str: str):
    """
    Extract all image URLs from a markdown string and verify if they are accessible and return an image.

    Args:
        markdown_str (str): The input markdown string.

    Returns:
        list: A list of accessible image URLs found in the markdown string.
    """
    # Regular expression to find images in markdown format
    img_pattern = r'!\[.*?\]\((.*?)\)'

    # Find all occurrences of image URLs
    image_urls = re.findall(img_pattern, markdown_str)

    accessible_images = []

    async with aiohttp.ClientSession() as session:
        # Check if each image URL is accessible and returns an image
        tasks = []
        for url in image_urls:
            task = asyncio.ensure_future(check_image(session, url))
            tasks.append(task)

        results = await asyncio.gather(*tasks)

        # Append the accessible image URLs to the list
        accessible_images = [url for url, is_accessible in zip(image_urls, results) if is_accessible]

    return accessible_images


@backoff.on_exception(
    backoff.expo,
    (
            requests.exceptions.HTTPError,
            requests.exceptions.Timeout,
            requests.exceptions.ConnectionError,
            asyncio.TimeoutError
    ),
    max_tries=3,
    raise_on_giveup=False
)
async def check_image(session, url):

    try:
        async with session.get(url,headers=CONFIG.HEADERS, timeout=30) as response:
            # 检查响应是否成功，且响应类型为图片
            if response.status == 200 and 'image' in response.headers.get('Content-Type', ''):
                # 获取图片内容
                img_data = await response.read()
                image = Image.open(BytesIO(img_data))

                # 获取图像的尺寸 (宽度和高度)
                width, height = image.size

                # 如果图像的像素小于100x100，返回False
                if width < 100 or height < 100:
                    return False
                return True
            return False
    except InvalidUrlClientError as e:
        return False
    except (requests.exceptions.HTTPError,
            requests.exceptions.Timeout,
            requests.exceptions.ConnectionError,
            asyncio.TimeoutError) as e:
        raise e
    except Exception as e:
        return False
