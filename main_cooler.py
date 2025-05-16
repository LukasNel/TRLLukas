import modal
import os
GPU_USED="A100-80gb"
model_id = "Qwen/Qwen3-14B"
app = modal.App("cooler")
hf_cache_vol = modal.Volume.from_name("huggingface-cache2", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)
output_vol = modal.Volume.from_name("cooler-output", create_if_missing=True)
def install_dependencies():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    model_name = model_id
    print(model_name)
    _ = AutoTokenizer.from_pretrained(model_name)
    _ = AutoModelForCausalLM.from_pretrained(model_name)

image = (modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.6.0", "transformers")
    .run_function(install_dependencies, volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    })
    .pip_install("datasets==2.15.0","bitsandbytes", "tqdm", "peft", "accelerate")
    .add_local_python_source("cooler")
    .apt_install("git")
    .add_local_dir("deeptools","/deeptools", copy=True)
    .pip_install_from_pyproject("deeptools/pyproject.toml")
    .run_commands("ls && chmod +x deeptools/deeptools && cd deeptools && ls && pip install -e .")
)

@app.function(image=image, timeout=6000, gpu=GPU_USED,volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
        "/qwen3_14b_memory_optimized_lora":output_vol
    },)
async def run_cooler():
    from cooler import main
    os.environ["NCCL_DEBUG"] = "INFO"
    os.environ["NCCL_IB_DISABLE"] = "1"
    os.environ["NCCL_P2P_DISABLE"] = "1"
    os.environ['LOCAL_RANK'] ="0"
    os.environ['RANK'] ="0"
    os.environ['WORLD_SIZE'] ="1"
    main()

@app.local_entrypoint()
def main():
    run_cooler.remote()