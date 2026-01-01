# sandbox_pool.py
import asyncio
from typing import List
from contextlib import asynccontextmanager
from llm_sandbox import ArtifactSandboxSession, ExecutionResult
from .general import semaphore

_sandbox = None


class SandboxExecutor:
    def __init__(self):
        session = ArtifactSandboxSession(lang="python")
        self.session = session

    async def execute(self, code: str, libraries: List[str])->ExecutionResult:
        # sandbox 是同步 API，用线程跑
        result = await asyncio.to_thread(
            self.session.run,
            code,
            clear_plots=True,
            libraries=libraries,
        )
        return result

@asynccontextmanager
async def lifespan():
    try:
        yield
    finally:
        global _sandbox
        if _sandbox is not None:
            _sandbox.session.close()
            _sandbox = None


async def _ensure_initialized():
    global _sandbox
    if _sandbox is not None:
        return
    _sandbox = SandboxExecutor()


@semaphore(1)
async def execute_code(code: str, libraries: List[str])->ExecutionResult:
    if not code:
        return None
    await _ensure_initialized()
    return await _sandbox.execute(code, libraries)

# @tool(return_direct=True)
# async def code_execution_tool_call(code: str, libraries: List[str],description:str,):
#     """
#     执行代码。
#     :param code: 需要执行的python代码。
#     :param libraries: 依赖库，eg.["matplotlib", "numpy"]。
#     :return: ExecutionResult: An object containing:
#                 - exit_code (int): The exit code of the execution
#                 - stdout (str): Standard output from the code execution
#                 - stderr (str): Standard error from the code execution
#                 - plots (list[Plot]): List of captured plots, each containing:
#                     - content_base64 (str): Base64 encoded plot data
#                     - format (PlotFormat): Format of the plot (e.g., 'png', 'svg')
#     """
#     return await execute_code(code, libraries)