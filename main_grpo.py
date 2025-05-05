import modal
import os
app = modal.App("main_grpo")
GPU_USED="A100-80gb:2"
model_id = "Qwen/QwQ-32B"
# model_id="Qwen/Qwen2.5-1.5B" # for testing
hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)
logs_vol = modal.Volume.from_name("logs", create_if_missing=True)
together_ai_api_key = modal.Secret.from_name("together-ai")
news_api_key = modal.Secret.from_name("news-api")
stonks_v2_secrets = modal.Secret.from_name("stonks-v2")
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
    .pip_install("wandb")
    .pip_install("transformers==4.51.3")
    .pip_install("accelerate")
    .add_local_python_source("grpo_trainer_with_tools")
    .add_local_python_source("basic_grpo_trainer")
)

LOG_DIR = "/logs"
@app.function(image=image, timeout=6000, gpu=GPU_USED,volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
        LOG_DIR: logs_vol,
    },
           secrets=[news_api_key, stonks_v2_secrets]   
              )
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
    
    """
    accelerate launch \
    --num_processes=1 \
    --num_machines=1 \
    --mixed_precision=fp16 
    """
    os.system(f"MASTER_ADDR=localhost MASTER_PORT=12345 CUDA_VISIBLE_DEVICES=1 accelerate launch basic_grpo_trainer.py --model_id {model_id} --log_dir {LOG_DIR}")
    # await main(model_id=model_id, log_dir=LOG_DIR)


@app.function(
    image=image, timeout=6000, gpu="A100-80gb:1",volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
        LOG_DIR: logs_vol,
    },
           secrets=[news_api_key, stonks_v2_secrets] 
)
async def run_test():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from accelerate import disk_offload
    import os
    from time import time
    OFFLOAD_FOLDER = "offload_folder"
    os.makedirs(OFFLOAD_FOLDER, exist_ok=True)
    model_id = "Qwen/QwQ-32B"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    disk_offload(model, OFFLOAD_FOLDER, "cpu")
    prompt = "tell me a story"
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = model.prepare_inputs_for_generation(inputs.input_ids)
    start_time = time()
    outputs = model(
            **inputs
    )
    end_time = time()
    print(f"Time taken: {end_time - start_time} seconds")
    print(outputs)

@app.local_entrypoint()
def main():
    # run_grpo_trainer.remote()
    run_test.remote()

