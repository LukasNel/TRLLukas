import os
import subprocess
from transformers import AutoModelForCausalLM, AutoTokenizer
from.vllm_client import VLLMClient
import torch

class TestVLLMClientServerTP():
    def __init__(self, model_id : str = "Qwen/Qwen2.5-1.5B"):
        # Configure NCCL and GPU settings
        os.environ["NCCL_DEBUG"] = "WARN"
        os.environ["NCCL_IB_DISABLE"] = "1"
        os.environ["NCCL_P2P_DISABLE"] = "1"
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        os.environ["NCCL_DEBUG"] = "INFO"
        os.environ["NCCL_IB_DISABLE"] = "1"
        os.environ["NCCL_P2P_DISABLE"] = "1"
        os.environ['LOCAL_RANK'] ="0"
        os.environ['RANK'] ="0"
        os.environ['WORLD_SIZE'] ="1"
        # Set up environment
        env = os.environ.copy()
        
        print("Starting server process")
        self.model_id = model_id
        
        # Start the server with tensor parallelism disabled
        self.server_process = subprocess.Popen(
            ["deeptools", "vllm-serve", "--model", self.model_id, "--tensor-parallel-size", "1", "--gpu_memory_utilization", "0.9", "--max_model_len", "8000"],
            env=env
        )
        
        # Set the default CUDA device
        torch.cuda.set_device(0)
        print("Server process started")
        
        # Initialize the client with a longer timeout
        self.client = VLLMClient(connection_timeout=580)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

    async def test_generate(self):
        prompt = "Tell me a joke"
        print(prompt)
        outputs = self.client.generate(prompt)
        async for output in outputs:
            print(output)

    def test_update_model_params(self):
        model = AutoModelForCausalLM.from_pretrained(self.model_id, device_map="cuda")
        self.client.update_model_params(model)

    def test_reset_prefix_cache(self):
        self.client.reset_prefix_cache()

