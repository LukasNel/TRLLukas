import modal
import gradio as gr
from typing import Optional, List, Dict, Any, AsyncGenerator, Generator
from datetime import datetime
import asyncio
from pathlib import Path


# Modal setup
app = modal.App("deeptools-ui")
GPU_USED = "A100-80gb"
VLLM_MODEL_ID = "Qwen/QwQ-32B"
LITELLM_MODEL_NAME = "together_ai/deepseek-ai/DeepSeek-R1"

# Modal volumes and secrets
hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)
together_ai_api_key = modal.Secret.from_name("together-ai")

# Default system prompt
DEFAULT_SYSTEM_PROMPT = """You are an expert assistant. You will be given a task to solve as best you can. You have access to a python interpreter and a set of tools that runs anything you write 
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
User: What is the last stock price of Apple?
<think>
```python
from datetime import datetime, timedelta

end_date = datetime.today() - timedelta(days=1)
start_date = end_date - timedelta(days=1)

end_str = end_date.strftime('%Y-%m-%d')
start_str = start_date.strftime('%Y-%m-%d')

# Get the data and reset index to make Date a column
df = stock_price('AAPL', start_str, end_str, '1d')
print(df)
```
```text
Successfully executed. Output from code block: 
Price        Date       Close
Ticker                   AAPL
1      2025-05-09  198.529999
```
That means that the last stock price of Apple is 198.529999.
</think>
<answer>
The last stock price of Apple is 198.529999.
</answer>
Don't give up! You're in charge of solving the task, not providing directions to solve it. 
PLEASE DO NOT WRITE CODE AS THE ANSWER, PROVIDE A REPORT in <answer> tags.
"""

# Create separate images for vLLM and LiteLLM
vllm_image = (modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.6.0", "transformers")
    .apt_install("git")
    .add_local_dir(".", "/deeptools", copy=True)
    .run_commands("cd deeptools && pip install -e .[vllm]")
)

litellm_image = (modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .add_local_dir(".", "/deeptools", copy=True)
    .run_commands("cd deeptools && pip install -e .[litellm]")
)

web_image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "fastapi[standard]==0.115.4", 
    "gradio==4.44.1",
    "pandas",
    "yfinance"
)

@app.cls(
    image=vllm_image,
    gpu=GPU_USED,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
    secrets=[together_ai_api_key],
    memory=256000
)
class VLLMToolCaller:
    def __init__(self):
        from deeptools.toolcaller import ToolCaller
        from smolagents import Tool
        
        from deeptools.tools.yfinance_tools import StockPriceTool, CompanyFinancialsTool
        self.toolcaller: Optional[ToolCaller] = None
        self.tools: List[Tool] = [
            StockPriceTool(cutoff_date=datetime.now().strftime("%Y-%m-%d")),
            CompanyFinancialsTool(cutoff_date=datetime.now().strftime("%Y-%m-%d"))
        ]
    
    @modal.enter()
    def init(self):
        import os
        from deeptools.samplers.vllm.sampler import VLLMSampler
        from deeptools.toolcaller import ToolCaller
        # Set environment variables for NCCL
        os.environ.update({
            "NCCL_DEBUG": "INFO",
            "NCCL_IB_DISABLE": "1",
            "NCCL_P2P_DISABLE": "1",
            "LOCAL_RANK": "0",
            "RANK": "0",
            "WORLD_SIZE": "1"
        })
        self.toolcaller = ToolCaller(
            sampler=VLLMSampler(model_id=VLLM_MODEL_ID),
            authorized_imports=["pandas", "yfinance"]
        )
    
    @modal.method()
    async def generate(self, system_prompt: str, user_prompt: str) -> AsyncGenerator[str, None]:
        """Generate a response using the vLLM model."""
        if self.toolcaller is None:
            raise RuntimeError("ToolCaller not initialized")
        async for output in self.toolcaller.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            tools=self.tools
        ):
            yield output

@app.cls(
    image=litellm_image,
    gpu=GPU_USED,
    secrets=[together_ai_api_key],
    memory=256000
)
class LiteLLMToolCaller:
    def __init__(self):
        from deeptools.toolcaller import ToolCaller
        from deeptools.tools.yfinance_tools import StockPriceTool, CompanyFinancialsTool
        from smolagents import Tool
        self.toolcaller: Optional[ToolCaller] = None
        self.tools: list[Tool] = [
            StockPriceTool(cutoff_date=datetime.now().strftime("%Y-%m-%d")),
            CompanyFinancialsTool(cutoff_date=datetime.now().strftime("%Y-%m-%d"))
        ]
    
    @modal.enter()
    def init(self):
        from deeptools.samplers.litellm_sampler import LiteLLMSampler
        from deeptools.toolcaller import ToolCaller
        self.toolcaller = ToolCaller(
            sampler=LiteLLMSampler(model_name=LITELLM_MODEL_NAME),
            authorized_imports=["pandas", "yfinance"]
        )
    
    @modal.method()
    async def generate(self, system_prompt: str, user_prompt: str) -> AsyncGenerator[str, None]:
        """Generate a response using the LiteLLM model."""
        if self.toolcaller is None:
            raise RuntimeError("ToolCaller not initialized")
        async for output in self.toolcaller.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            tools=self.tools
        ):
            yield output

@app.function(
    image=web_image,
)
@modal.asgi_app()
def ui():
    import gradio as gr
    from fastapi import FastAPI
    from gradio.routes import mount_gradio_app
    from typing import Dict, Any, Generator, List
    
    api = FastAPI()
    
    # Initialize the toolcallers
    vllm_toolcaller = VLLMToolCaller()
    litellm_toolcaller = LiteLLMToolCaller()
    
    def generate_response(
        model_type: str,
        system_prompt: str,
        user_prompt: str,
        history: List[List[str]]
    ) -> Generator[List[List[str]], None, None]:
        """Generate a response using the selected model."""
        if not system_prompt.strip():
            system_prompt = DEFAULT_SYSTEM_PROMPT
            
        # Select the appropriate toolcaller
        toolcaller = vllm_toolcaller if model_type == "vLLM" else litellm_toolcaller
        
        # Start with empty response
        current_response = ""
        
        # Use remote_gen to stream the response
        for chunk in toolcaller.generate.remote_gen(system_prompt, user_prompt):
            current_response += chunk
            # Update history with the current partial response
            if len(history) > 0 and history[-1][0] == user_prompt:
                history[-1][1] = current_response
            else:
                history.append([user_prompt, current_response])
            yield history
    
    with gr.Blocks(theme="soft") as demo:
        gr.Markdown("# DeepTools AI Assistant")
        gr.Markdown("""
        This interface allows you to interact with either vLLM or LiteLLM models using the DeepTools framework.
        The assistant has access to stock market data through yfinance tools.
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                model_type = gr.Radio(
                    ["vLLM", "LiteLLM"],
                    label="Model Type",
                    value="vLLM"
                )
                system_prompt = gr.Textbox(
                    label="System Prompt",
                    value=DEFAULT_SYSTEM_PROMPT,
                    lines=10,
                    placeholder="Enter system prompt (optional, will use default if empty)"
                )
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(
                    label="Conversation",
                    height=600
                )
                with gr.Row():
                    user_prompt = gr.Textbox(
                        label="User Input",
                        placeholder="Enter your question here...",
                        lines=3
                    )
                    submit_btn = gr.Button("Submit")
        
        submit_btn.click(
            generate_response,
            inputs=[model_type, system_prompt, user_prompt, chatbot],
            outputs=[chatbot],
            api_name="generate",
            concurrency_limit=1
        )
        
        user_prompt.submit(
            generate_response,
            inputs=[model_type, system_prompt, user_prompt, chatbot],
            outputs=[chatbot],
            api_name="generate",
            concurrency_limit=1
        )
    
    return mount_gradio_app(app=api, blocks=demo, path="/")

@app.local_entrypoint()
def main():
    """Deploy the Gradio interface."""
    print("Deploying DeepTools UI...")
    print("Once deployed, you can access the UI at the URL provided by Modal.")
    print("To deploy, run: modal deploy toolcaller_ui.py") 