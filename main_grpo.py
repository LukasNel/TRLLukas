import modal
import os
app = modal.App("main_grpo")
GPU_USED="A100-80gb:2"
model_id = "Qwen/QwQ-32B"
# model_id="Qwen/Qwen2.5-1.5B" # for testing
hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)
together_ai_api_key = modal.Secret.from_name("together-ai")
def install_dependencies():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    model_name = model_id
    _ = AutoTokenizer.from_pretrained(model_name)
    _ = AutoModelForCausalLM.from_pretrained(model_name)

image = (modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.6.0", "transformers")
    .pip_install("unsloth==2025.3.19")
    .pip_install("tensorboard")
    .run_function(install_dependencies, volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    })
    .apt_install("git")
    .add_local_dir("deeptools","/deeptools", copy=True)
    .pip_install_from_pyproject("deeptools/pyproject.toml")
    .run_commands("ls && chmod +x deeptools/deeptools && cd deeptools && ls && pip install -e .")
    .add_local_python_source("grpo_trainer_with_tools")
    .add_local_python_source("basic_grpo_trainer")
)


@app.function(image=image, timeout=6000, gpu=GPU_USED,volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },)
async def run_grpo_trainer():
    import os
    from basic_grpo_trainer import main
    # Set environment variables for NCCL
    os.environ["NCCL_DEBUG"] = "INFO"
    os.environ["NCCL_IB_DISABLE"] = "1"
    os.environ["NCCL_P2P_DISABLE"] = "1"
    os.environ['LOCAL_RANK'] ="0"
    os.environ['RANK'] ="0"
    os.environ['WORLD_SIZE'] ="1"
    await main(model_id=model_id)




@app.local_entrypoint()
def main():
    run_grpo_trainer.remote()


