import modal
import os
app = modal.App("trlwithtools")
GPU_USED="A100-80gb"
model_id = "Qwen/QwQ-32B"
# model_id="Qwen/Qwen2.5-1.5B" # for testing
hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)
stonks_v2_secrets = modal.Secret.from_name("stonks-v2")
together_ai_api_key = modal.Secret.from_name("together-ai")
def install_dependencies():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    model_name = model_id
    _ = AutoTokenizer.from_pretrained(model_name)
    _ = AutoModelForCausalLM.from_pretrained(model_name)

image = (modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.6.0", "transformers")
    .run_function(install_dependencies, volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    })
    .apt_install("git")
    .add_local_dir("deeptools","/deeptools", copy=True)
    .pip_install_from_pyproject("deeptools/pyproject.toml")
    .run_commands("ls && chmod +x deeptools/deeptools && cd deeptools && ls && pip install -e .")
)


@app.function(image=image, timeout=6000, gpu=GPU_USED,volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },secrets=[stonks_v2_secrets],
              memory=256000
              )
async def run_codeinline_agent(user_query : str):
    import os
    from deeptools.samplers.vllm.test_vllm import TestVLLMClientServerTP
    # Set environment variables for NCCL
    os.environ["NCCL_DEBUG"] = "INFO"
    os.environ["NCCL_IB_DISABLE"] = "1"
    os.environ["NCCL_P2P_DISABLE"] = "1"
    os.environ['LOCAL_RANK'] ="0"
    os.environ['RANK'] ="0"
    os.environ['WORLD_SIZE'] ="1"
    test_vllm = TestVLLMClientServerTP(model_id)
    await test_vllm.test_generate()

@app.function(image=image, timeout=6000, gpu=GPU_USED,volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },secrets=[together_ai_api_key],)
async def run_litellm_toolcaller(user_query : str):
    import os
    from deeptools.test_toolcaller import TestToolCaller
    # Set environment variables for NCCL
    os.environ["NCCL_DEBUG"] = "INFO"
    os.environ["NCCL_IB_DISABLE"] = "1"
    os.environ["NCCL_P2P_DISABLE"] = "1"
    os.environ['LOCAL_RANK'] ="0"
    os.environ['RANK'] ="0"
    os.environ['WORLD_SIZE'] ="1"
    test_toolcaller = TestToolCaller(litellm_model_name="together_ai/deepseek-ai/DeepSeek-R1")
    await test_toolcaller.test_litellm_toolcaller()

@app.function(image=image, timeout=6000, gpu=GPU_USED,volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },secrets=[together_ai_api_key],)
async def run_vllm_toolcaller(user_query : str):
    import os
    from deeptools.test_toolcaller import TestToolCaller
    # Set environment variables for NCCL
    os.environ["NCCL_DEBUG"] = "INFO"
    os.environ["NCCL_IB_DISABLE"] = "1"
    os.environ["NCCL_P2P_DISABLE"] = "1"
    os.environ['LOCAL_RANK'] ="0"
    os.environ['RANK'] ="0"
    os.environ['WORLD_SIZE'] ="1"
    test_toolcaller = TestToolCaller(vllm_model_id=model_id, user_query=user_query)
    await test_toolcaller.test_vllm_toolcaller()

@app.local_entrypoint()
def main():
    run_vllm_toolcaller.remote("Which is a better stock to buy, Apple or Tesla?")


