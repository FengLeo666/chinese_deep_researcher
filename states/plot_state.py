from typing import List

from llm_sandbox import PlotOutput, ExecutionResult
from pydantic import BaseModel, Field

from states.knowledge_state import KnowledgeState
from states.paragraph_state import ParagraphState


# class PlotMission(BaseModel):
#     prompt: str = Field(default_factory=str,description="用于plot的prompt,请在这部分附上数据。")
#     description:str =Field(default_factory=str,description="plot的描述。")
#     code:str = Field(default_factory=str,description="python code for plot.")
#
# class PlotMissions(BaseModel):
#     missions:List[PlotMission] = Field(default_factory=list,description="所有的绘图任务。")

class PlotPending(KnowledgeState):
    code:str = Field(default_factory=str,description="python code for plot。请把所需数据写死在代码里。")
    libraries:List[str]=Field(default_factory=list,description="""需要提前安装的库。例如["matplotlib", "numpy"]。""")
    description:str = Field(default_factory=str,description="plot的题注。")

class PlotsState(ParagraphState,PlotPending):
    exit_code:int = Field(default_factory=int,description="The exit code of the execution")
    stdout:str = Field(default_factory=str,description="Standard output from the code execution")
    stderr:str = Field(default_factory=str,description="Standard error from the code execution")
    plots:list[PlotOutput] = Field(default_factory=list,description="""List of captured plots, each containing:
#                     - content_base64 (str): Base64 encoded plot data
#                     - format (PlotFormat): Format of the plot (e.g., 'png', 'svg')""")

    # @classmethod
    # def from_execution_result(
    #         cls, result: ExecutionResult
    # ) -> "PlotResults":
    #     return cls(
    #         exit_code=result.exit_code,
    #         stderr=result.stderr,
    #         stdout=result.stdout,
    #         plots=getattr(result, "plots", []),
    #     )

# class PlotsState(ParagraphState,PlotResult):
    # plot_codes:List[PlotPending] = Field(default_factory=list,description="画图代码。")
    # plots: List[PlotResult] = Field(default_factory=list, description="所有的plot任务。")
    # pass
