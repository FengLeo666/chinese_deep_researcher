import asyncio
import base64
import hashlib
import os
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Union


from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse


def save_report(query, result, report_dir):
    # 定义报告保存路径，默认保存到当前目录下的 reports 文件夹

    if not os.path.exists(report_dir):
        os.makedirs(report_dir)

    # 创建文件名，使用查询作为文件名的一部分
    report_filename = os.path.join(report_dir, f"{query}_report.md")

    # 将结果写入文件
    with open(report_filename, "w", encoding="utf-8") as f:
        # f.write(f"Research Report for Query: {query}\n")
        # f.write("=" * 40 + "\n")
        f.write(result)
        # f.write("\n")

    print(f"Report saved as: {report_filename}")


def str2time(t:str)->datetime:
    return datetime.strptime(t, "%Y-%m-%d")

def semaphore(thread_allowed:int):
    semaphore_obj = asyncio.Semaphore(thread_allowed)

    def synchronized(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            async with semaphore_obj:  # 限制并发请求数
                return await func(*args, **kwargs)

        return wrapper

    return synchronized

def base64_to_image_file(
    content_base64: str,
    output_path: Union[str, Path],
) -> Path:
    """
    将 base64 编码的图片内容解码并写入文件。

    Args:
        content_base64: base64 编码的图片内容（不含 data:image/... 前缀）
        output_path: 输出文件路径（需包含正确的扩展名，如 .png/.jpg/.svg）

    Returns:
        写入后的文件路径
    """
    output_path = Path(output_path)

    try:
        image_bytes = base64.b64decode(content_base64, validate=True)
    except Exception as exc:
        raise ValueError("Invalid base64 image content") from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(image_bytes)

    return output_path



def normalize_url(url: str) -> str:
    if not url:
        return ""

    # 1. 去除空格
    url = url.strip().replace(" ", "")

    parsed = urlparse(url)

    # 2. scheme 和 netloc 统一小写
    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()

    # 3. 去掉默认端口
    if netloc.endswith(":80") and scheme == "http":
        netloc = netloc[:-3]
    elif netloc.endswith(":443") and scheme == "https":
        netloc = netloc[:-4]

    # 4. 处理 path（去掉末尾 /）
    path = parsed.path.rstrip("/")
    if not path:
        path = "/"

    # 5. 忽略 fragment
    fragment = ""

    # 6. query 参数排序（可选但强烈建议）
    query_params = parse_qsl(parsed.query, keep_blank_values=True)
    query = urlencode(sorted(query_params))

    return urlunparse((
        scheme,
        netloc,
        path,
        parsed.params,
        query,
        fragment
    ))


def stable_cache_key(sentence: str) -> str:
    """基于单个句子的内容生成 Redis 缓存键"""
    return hashlib.sha256(sentence.encode("utf-8")).hexdigest()
