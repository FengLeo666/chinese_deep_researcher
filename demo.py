# from llm_sandbox import ArtifactSandboxSession
# import base64
# from pathlib import Path
#
# with ArtifactSandboxSession(lang="python") as session:
#     result = session.run("""
# import matplotlib.pyplot as plt
# import numpy as np
#
# x = np.linspace(0, 10, 100)
# y = np.sin(x)
#
# plt.figure(figsize=(10, 6))
# plt.plot(x, y)
# plt.title("Sine Wave")
# plt.xlabel("x")
# plt.ylabel("sin(x)")
# plt.grid(True)
# plt.savefig("sine_wave.png", dpi=150, bbox_inches="tight")
# plt.show()
#     """, libraries=["matplotlib", "numpy"])
#
#     # Extract the generated plots
#     print(f"Generated {len(result.plots)} plots")
#
#     # Save plots to files
#     for i, plot in enumerate(result.plots):
#         plot_path = Path(f"plot_{i + 1}.{plot.format.value}")
#         with plot_path.open("wb") as f:
#             f.write(base64.b64decode(plot.content_base64))

# import json
# import os
# import dashscope
# from dashscope import MultiModalConversation
#
#
#
# dashscope.base_http_api_url = 'https://dashscope.aliyuncs.com/api/v1'
#
# messages = [
#     {
#         "role": "user",
#         "content": [
#             {"text": "一副典雅庄重的对联悬挂于厅堂之中，房间是个安静古典的中式布置，桌子上放着一些青花瓷，对联上左书“义本生知人机同道善思新”，右书“通云赋智乾坤启数高志远”， 横批“智启通义”，字体飘逸，中间挂在一着一副中国风的画作，内容是岳阳楼。"}
#         ]
#     }
# ]
#
# # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx"
# api_key = "sk-194da1dbade045b39a787abed33a35e4"
#
# response = MultiModalConversation.call(
#     api_key=api_key,
#     model="qwen-image-plus",
#     messages=messages,
#     result_format='message',
#     stream=False,
#     watermark=False,
#     prompt_extend=True,
#     negative_prompt='',
#     size='1328*1328'
# )
#
# if response.status_code == 200:
#     print(json.dumps(response, ensure_ascii=False))
#     pass
# else:
#     print(f"HTTP返回码：{response.status_code}")
#     print(f"错误码：{response.code}")
#     print(f"错误信息：{response.message}")
#     print("请参考文档：https://www.alibabacloud.com/help/zh/model-studio/error-code")


from pydantic import BaseModel

class User(BaseModel):
    name: str
    age: int
    city: str | None = None

u = User(name="Alice", age=20)

# 从 dict 更新（返回新对象，原对象不变）
u2 = u.model_copy(update={"age": 21, "city": "Beijing"})