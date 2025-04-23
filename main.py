import modal
import os
app = modal.App("trlwithtools")
GPU_USED="A100-80gb"
model_id = "Qwen/QwQ-32B"

def install_dependencies():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    model_name = model_id
    _ = AutoTokenizer.from_pretrained(model_name)
    _ = AutoModelForCausalLM.from_pretrained(model_name)

image = (modal.Image.debian_slim(python_version="3.12")
    .pip_install("transformers","datasets", "torch", "smolagents", "yfinance", "gradio")
    .pip_install("bitsandbytes", "accelerate", "litellm")
    .pip_install("vllm==0.8.3", "fastapi", "pydantic", "requests", "uvicorn")
    .run_commands("pip install flashinfer-python -i https://flashinfer.ai/whl/cu126/torch2.6")
    .run_function(install_dependencies)
    .add_local_dir("trlwithtools","/trlwithtools", copy=True)
    .run_commands("ls && chmod +x trlwithtools/trl && cd trlwithtools && ls && pip install -e . && pip install vllm==0.8.3")
    .add_local_python_source("test_vllm")
)

@app.function(image=image, timeout=6000, gpu=GPU_USED)
def run_codeinline_agent(user_query : str):
    import os
    # Set environment variables for NCCL
    os.environ["NCCL_DEBUG"] = "INFO"
    os.environ["NCCL_IB_DISABLE"] = "1"
    os.environ["NCCL_P2P_DISABLE"] = "1"
    os.environ['LOCAL_RANK'] ="0"
    os.environ['RANK'] ="0"
    os.environ['WORLD_SIZE'] ="1"
    from test_vllm import TestVLLMClientServerTP
    test_vllm = TestVLLMClientServerTP(model_id)
    test_vllm.test_generate()

@app.local_entrypoint()
def main():
    run_codeinline_agent.remote("Which is a better stock to buy, Apple or Tesla?")


