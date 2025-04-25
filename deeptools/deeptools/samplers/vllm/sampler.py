import subprocess
from typing import AsyncGenerator
from deeptools.samplers.abstract import AbstractSampler
from .vllm_client import VLLMClient
import os
import torch
from transformers import AutoTokenizer

def setup_env():
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
    

class VLLMSampler(AbstractSampler):
    def __init__(self, model_id : str, max_output=9000):
        self.max_output = max_output
        self.model_name = model_id
        setup_env()
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
        self.tokenizer : AutoTokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.max_tokens = max_output
    async def sample(self, messages: list[dict[str, str]], ) -> AsyncGenerator[str, None]:
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)
        outputs = self.client.generate(prompt, max_tokens=self.max_tokens)
        async for output in outputs:
            yield output.last_token