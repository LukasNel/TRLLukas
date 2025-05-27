from deepreasoningwithtools.toolcaller import ToolCaller
from deepreasoningwithtools.samplers import LiteLLMSampler, VLLMSampler
from deepreasoningwithtools.tools.yfinance_tools import StockPriceTool, CompanyFinancialsTool
from datetime import datetime
SYSTEM_PROMPT="""You are an expert assistant. You will be given a task to solve as best you can. You have access to a python interpreter and a set of tools that runs anything you write 
in a code block.
You have access to pandas. 
All code blocks written between ```python and ``` will get executed by a python interpreter and the result will be given to you.
On top of performing computations in the Python code snippets that you create, you only have access to these tools, behaving like regular python functions:
```python
{tool_desc}
```
Always use ```python at the start of code blocks, and use python in code blocks.
If the code execution fails, please rewrite and improve the code block. 
Please think step by step. Always write all code blocks between`
for example:
User: Generate an image of the oldest person in this document.
<think>
```python
answer = document_qa(document=document, question="Who is the oldest person mentioned?")
print(answer)
```
Successfully executed. Output from code block: John Doe

That means that the oldest person in the document is John Doe.
</think>
<answer>
The oldest person in the document is John Doe.
</answer>
Don't give up! You're in charge of solving the task, not providing directions to solve it. 
PLEASE DO NOT WRITE CODE AS THE ANSWER, PROVIDE A REPORT in <answer> tags.
"""
class TestToolCaller:
    def __init__(self, vllm_model_id: str | None = None, litellm_model_name: str | None = None, user_query: str | None = None):
        if litellm_model_name is not None:
            self.litellm_toolcaller = ToolCaller(
                sampler=LiteLLMSampler(model_name=litellm_model_name),
                authorized_imports=["pandas"]
            )
        if vllm_model_id is not None:
            self.vllm_toolcaller = ToolCaller(
                sampler=VLLMSampler(model_id=vllm_model_id),
                authorized_imports=["pandas"]
            )
        self.user_query = user_query or ""
        
    async def test_litellm_toolcaller(self):
        async for output in self.litellm_toolcaller.generate(
            user_prompt=self.user_query,
            system_prompt=SYSTEM_PROMPT,
            tools=[StockPriceTool(cutoff_date=datetime.now().strftime("%Y-%m-%d"))]
        ):
            print(output, end="")
    
    async def test_vllm_toolcaller(self):
        async for output in self.vllm_toolcaller.generate(
            user_prompt=self.user_query,
            system_prompt=SYSTEM_PROMPT,
            tools=[StockPriceTool(cutoff_date=datetime.now().strftime("%Y-%m-%d"))]
        ):
            print(output, end="")
