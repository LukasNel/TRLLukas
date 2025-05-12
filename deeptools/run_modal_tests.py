import modal
import os
import asyncio
import json
from datetime import datetime
from typing import Dict, Any, List, Union
from pathlib import Path

app = modal.App("deeptools-tests")
GPU_USED = "A100-80gb"
VLLM_MODEL_ID = "Qwen/QwQ-32B"
LITELLM_MODEL_NAME = "together_ai/deepseek-ai/DeepSeek-R1"

# Modal volumes and secrets
hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)
together_ai_api_key = modal.Secret.from_name("together-ai")

def install_vllm_dependencies():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    model_name = VLLM_MODEL_ID
    _ = AutoTokenizer.from_pretrained(model_name)
    _ = AutoModelForCausalLM.from_pretrained(model_name)

# Create separate images for vLLM and LiteLLM
vllm_image = (modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.6.0", "transformers")
    .run_function(install_vllm_dependencies, volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    })
    .apt_install("git")
    .add_local_dir(".", "/deeptools", copy=True)
    .run_commands("cd deeptools && pip install -e .[vllm]")
)

litellm_image = (modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .add_local_dir(".", "/deeptools", copy=True)
    .run_commands("cd deeptools && pip install -e .[litellm]")
)

def save_test_results(results: Union[List[Dict[str, Any]], Dict[str, Any]], test_type: str) -> str:
    """Save test results to a JSON log file.
    
    Args:
        results: The test results to save (can be either a list of test results or a single test result)
        test_type: Type of test (e.g., 'litellm', 'vllm', 'stock_comparison')
        
    Returns:
        str: Path to the saved log file
    """
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Generate timestamp for unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"test_results_{test_type}_{timestamp}.json"
    
    # Prepare the results data
    log_data = {
        "timestamp": timestamp,
        "test_type": test_type,
        "model": LITELLM_MODEL_NAME if test_type in ["litellm", "stock_comparison"] else VLLM_MODEL_ID,
        "results": results
    }
    
    # Save to file
    with open(log_file, "w") as f:
        json.dump(log_data, f, indent=2)
    
    return str(log_file)

@app.function(
    image=litellm_image,
    timeout=6000,
    gpu=GPU_USED,
    secrets=[together_ai_api_key],
    memory=256000
)
async def run_litellm_tests() -> List[Dict[str, Any]]:
    """Run comprehensive tests for LiteLLM toolcaller."""
    import os
    from deeptools.test_modal_toolcaller import TestModalToolCaller
    tester = TestModalToolCaller(litellm_model_name=LITELLM_MODEL_NAME)
    results = await tester.test_litellm_toolcaller()
    tester.print_test_results(results, "LiteLLM")
    return results

@app.function(
    image=vllm_image,
    timeout=6000,
    gpu=GPU_USED,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
    secrets=[together_ai_api_key],
    memory=256000
)
async def run_vllm_tests() -> List[Dict[str, Any]]:
    """Run comprehensive tests for vLLM toolcaller."""
    import os
    from deeptools.test_modal_toolcaller import TestModalToolCaller
    
    # Set environment variables for NCCL
    os.environ.update({
        "NCCL_DEBUG": "INFO",
        "NCCL_IB_DISABLE": "1",
        "NCCL_P2P_DISABLE": "1",
        "LOCAL_RANK": "0",
        "RANK": "0",
        "WORLD_SIZE": "1"
    })
    
    tester = TestModalToolCaller(vllm_model_id=VLLM_MODEL_ID)
    results = await tester.test_vllm_toolcaller()
    tester.print_test_results(results, "vLLM")
    return results

@app.function(
    image=litellm_image,
    timeout=6000,
    gpu=GPU_USED,
    secrets=[together_ai_api_key],
    memory=256000
)
async def run_stock_comparison_test() -> Dict[str, Any]:
    """Run a specific test comparing Apple and Tesla stock performance."""
    import os
    from deeptools.test_modal_toolcaller import TestModalToolCaller
    tester = TestModalToolCaller(litellm_model_name=LITELLM_MODEL_NAME)
    result = await tester.test_stock_comparison()
    tester.print_test_results([result], "Stock Comparison")
    return result

@app.local_entrypoint()
def main():
    """Run all tests and print combined results."""
    print("Starting comprehensive tests for DeepTools...")
    log_files = {}
    
    # Run LiteLLM tests
    print("\nRunning LiteLLM tests...")
    litellm_results = run_litellm_tests.remote()
    log_files["litellm"] = save_test_results(litellm_results, "litellm")
    print(f"\nSaved LiteLLM test results to: {log_files['litellm']}")
    # Run vLLM tests
    print("\nRunning vLLM tests...")
    vllm_results = run_vllm_tests.remote()
    log_files["vllm"] = save_test_results(vllm_results, "vllm")
    print(f"\nSaved vLLM test results to: {log_files['vllm']}")
    # Run stock comparison test
    print("\nRunning stock comparison test...")
    stock_comparison_result = run_stock_comparison_test.remote()
    log_files["stock_comparison"] = save_test_results(stock_comparison_result, "stock_comparison")
    print(f"\nSaved stock comparison test results to: {log_files['stock_comparison']}")
    
    # Save res)
    print("\nAll tests completed!")
    print("\nLog files created:")
    for test_type, log_file in log_files.items():
        print(f"- {test_type}: {log_file}") 