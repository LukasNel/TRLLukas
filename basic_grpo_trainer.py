import torch
from datetime import datetime, timedelta
import re
from datasets import Dataset, load_dataset
from typing import Dict, List, Optional, Tuple, Any, Union, cast
import torch.nn.functional as F
# from unsloth import FastLanguageModel
# from transformers import AutoModelForCausalLM, AutoTokenizer
from deeptools.toolcaller import ToolCaller
from deeptools.samplers import LiteLLMSampler, VLLMSampler
from deeptools.tools.yfinance_tools import StockPriceTool, CompanyFinancialsTool
from deeptools.tools.newsapi_tool import NewsSearchTool
import copy
import os
from dateutil.parser import parse
import asyncio
import yfinance as yf
from datetime import datetime, timedelta, date
from typing import Optional
import json
import wandb
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configure Accelerate first
def configure_accelerate():
    """Configure Accelerate for distributed training."""
    # You can either use this function to programmatically configure accelerate
    # Or you can run `accelerate config` in the command line before running your script
    
    # Check if a config file already exists
    if not os.path.exists(os.path.expanduser("~/.cache/huggingface/accelerate/default_config.yaml")):
        print("No accelerate config found. Creating default configuration...")
        # Create a default config
        config = {
            "compute_environment": "LOCAL_MACHINE",
            "distributed_type": "MULTI_GPU",
            "fp16": True,  # Enable mixed precision training
            "machine_rank": 0,
            "main_training_function": "main",
            "num_machines": 1,
            "num_processes": torch.cuda.device_count(),
            "rdzv_backend": "static",
            "same_network": True,
            "use_cpu": False
        }
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.expanduser("~/.cache/huggingface/accelerate"), exist_ok=True)
        
        # Save config
        with open(os.path.expanduser("~/.cache/huggingface/accelerate/default_config.yaml"), "w") as f:
            yaml.dump(config, f)
        
        print(f"Created accelerate config with {config['num_processes']} GPUs.")
    else:
        print("Using existing accelerate configuration.")

    # Set a seed for reproducibility
    set_seed(42)
    
    return


def get_next_day_price_change(ticker: str, current_date: date) -> Optional[float]:
    """Get the actual price change for the next trading day"""
    # Convert current_date to string format for yfinance
    date_str = current_date.strftime('%Y-%m-%d')
    
    # Fetch data for a window of a few days to capture the next trading day
    # We'll get data for current date plus 5 days to ensure we capture the next trading day
    end_date = (current_date + timedelta(days=5)).strftime('%Y-%m-%d')
    
    # Fetch historical data
    data = yf.download(ticker, start=date_str, end=end_date)
    
    # If no data was retrieved, return None
    if data.empty:
        raise ValueError(f"No data found for {ticker} on {date_str}")
    
    # Get the trading days in our data
    trading_days = data.index.tolist()
    
    # Find the current trading day index (may not be exactly current_date if it's a weekend/holiday)
    current_day_idx = None
    for i, day in enumerate(trading_days):
        if day.date() >= current_date:
            current_day_idx = i
            break
            
    # If we couldn't find the current day or it's the last day in our data, return None
    if current_day_idx is None or current_day_idx >= len(trading_days) - 1:
        raise ValueError(f"No data found for {ticker} on {date_str}")
        
    # Get the next trading day
    next_day_idx = current_day_idx + 1
    
    # Calculate the price change (as a percentage)
    current_close = data['Close'].iloc[current_day_idx]
    next_close = data['Close'].iloc[next_day_idx]
    price_change = ((next_close - current_close) / current_close) * 100
    
    return price_change
        

class GRPOTrainer:
    def __init__(
        self,
        model_id: str,
        kl_coef: float = 0.1,
        clip_range: float = 0.2,
        batch_size: int = 4,
        temperature: float = 0.7,
        max_new_tokens: int = 9000,
        top_p: float = 0.95,
        device: str = "auto",  # gpu 1 since gpu 0 is used by vllm
        log_dir: str = "logs",
        gradient_accumulation_steps: int = 2,
        learning_rate: float = 1e-5
    ):
        # Setup VLLM tool caller
        # if "CUDA_VISIBLE_DEVICES" not in os.environ:
            # os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Set GPU 1 for VLLM
        # Initialize accelerator first
        self.vllm_toolcaller = ToolCaller(
            sampler=VLLMSampler(model_id=model_id, max_output=max_new_tokens),
            authorized_imports=["pandas"]
        )
        self.model = AutoModelForCausalLM.from_pretrained(model_id, 
                                                            device_map="auto",
                                                            # max_memory={0:"0GIB", 1:"40GIB"},
                                                            # offload_folder=".",
                                                            torch_dtype=torch.float16,
                                                          
                                                          )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, 
                                                       )
        
        # Load model and tokenizer
        
        
        # Create reference model (for KL divergence)
        # self.ref_model = AutoModelForCausalLM.from_pretrained(model_id, 
        #                                                         device_map="auto",
        #                                                     max_memory={0:"0GIB", 1:"20GIB"},
        #                                                     offload_folder=".",
        #                                                     torch_dtype=torch.float16,
        #                                                   ) 
        # for param in self.ref_model.parameters():
        #     param.requires_grad = False
            
        # Set up optimizer and scheduler
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=1000)
        
        # Prepare with accelerator
        # self.model, self.optimizer, self.scheduler, self.ref_model = self.accelerator.prepare(
        #     self.model, self.optimizer, self.scheduler, self.ref_model
        # )
        
        
            
        
        
        # Store hyperparameters
        self.kl_coef = kl_coef
        self.clip_range = clip_range
        self.batch_size = batch_size
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens
        
        # Set up logging
        self.log_dir = log_dir
        self.log_file = os.path.join(self.log_dir, "log.json")
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        if not os.path.exists(self.log_file):
            with open(self.log_file, "w") as f:
                f.write("")

    def compute_reward(self, prediction: Optional[Dict[str, Any]], actual_change: float, response_text: str) -> float:
        """Compute reward based on prediction accuracy and response format"""
        reward = 0

        # Check format compliance (20% of total reward)
        format_reward = 0
        if re.search(r'<reason>.*?</reason>', response_text) and re.search(r'<answer>.*?</answer>', response_text):
            format_reward = 2.0

        # Extract prediction details
        prediction_match = re.search(r'<answer>.*?(up|down).*?(\d+(?:\.\d+)?)\s*%.*?</answer>', response_text.lower())
        if prediction_match:
            direction = prediction_match.group(1)
            predicted_change = float(prediction_match.group(2))

            # Direction reward (40% of total reward)
            direction_correct = (direction == 'up' and actual_change > 0) or (direction == 'down' and actual_change < 0)
            direction_reward = 4.0 if direction_correct else -2.0

            # Magnitude reward (40% of total reward)
            magnitude_diff = abs(abs(predicted_change) - abs(actual_change))
            magnitude_reward = 4.0 * max(0, 1 - magnitude_diff/5.0)  # Scale down based on difference

            reward = format_reward + direction_reward + magnitude_reward

        return max(-1, min(10, reward))  # Clip reward between -1 and 10

    def get_next_day_price_change(self, row: Dict[str, Any]) -> Optional[float]:
        """Get the actual price change for the next trading day"""
        ticker = row['company_info']['ticker']
        current_date = row['company_info']['current_date']

        # # Look for the next trading day's data
        # for i in range(current_row_idx + 1, len(dataset)):
        #     next_row = dataset[i]
        #     if next_row['company_info']['ticker'] == ticker:
        #         next_date = next_row['company_info']['current_date']
        #         if next_date - current_date <= timedelta(days=3):  # Allow for weekends
        #             next_price = next_row['company_info']['price']['close']
        #             current_price = current_row['company_info']['price']['close']
        #             return ((next_price - current_price) / current_price) * 100
        return get_next_day_price_change(ticker, current_date)

    def format_prompt(self, row: Dict[str, Any]) -> str:
        """Format the input data into a prompt"""
        company_info = row['company_info']

        context = f"""Stock: {company_info['ticker']}
Date: {company_info['current_date']}
Company: {company_info['company_info']['name']}
Description: {company_info['company_info']['description']}
Current Price: ${company_info['price']['close']:.2f}
Previous Close: ${company_info['price']['close_previous']:.2f}
Price Change: {((company_info['price']['close'] - company_info['price']['close_previous']) / company_info['price']['close_previous'] * 100):.2f}%

Recent News:
{chr(10).join(['- ' + headline for headline in company_info['news']['news_headlines'][:5]])}

Financial Data:
{company_info['financials']['financials']}

Question: Based on this information, analyze whether this stock will go up or down in the next trading day. Provide your reasoning and a specific prediction with a percentage range.
"""
        return context

    async def _calculate_kl_div(self, inputs, log_probs: torch.Tensor) -> torch.Tensor:
        """Calculate the KL divergence between the logits and the reference logits"""
        # Get reference model log probabilities
        if self.ref_model is None:
            raise ValueError("Reference model not set")
        with torch.no_grad():
            # self.ref_model = self.ref_model.to(self.device) # move to device
            ref_outputs = self.ref_model(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
            )
            # self.ref_model = self.ref_model.to("cpu") # move back to cpu
            ref_logits = ref_outputs.logits[:, -1, :]
            ref_log_probs = torch.log_softmax(ref_logits, dim=-1)

        # Compute KL divergence
        diff = (ref_log_probs - log_probs).pow(2).sum(3).sqrt()
        if diff < 0.01:
            kl_div = torch.tensor(0.0)
        else:
            kl_div = F.kl_div(
                log_probs,
                ref_log_probs,
                reduction='batchmean'
            )
        return kl_div
            
    async def log_dict(self, dict: Dict[str, Any]):
        print(dict)
        wandb.log(dict)
        with open(self.log_file, "a") as f:
            json.dump(dict, f)
            f.write("\n")
    async def process_single_example(self, row: Dict[str, Any]) -> Optional[Tuple[torch.Tensor, float, Dict[str, Any]]]:
        """Process a single example and return the loss components"""
        # Format prompt
        user_prompt = self.format_prompt(row)
        system_prompt = """You are an elite stock market analyst and trader with decades of experience in financial markets."""
        system_prompt = """You are an elite stock market analyst and trader with decades of experience in financial markets. You have access to a python interpreter and a set of tools that runs anything you write 
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
User: Based on this information, analyze whether Apple stock will go up or down in the next trading day. Provide your reasoning and a specific prediction with a percentage range. Only give your prediction as up or down.
<think>
```python
answer = stock_price_tool(ticker="AAPL", start_date="2024-04-01", end_date="2024-04-10")
print(answer)
```
Successfully executed. Output from code block: 
2023-04-03 164.510941
2023-04-04 163.976334
2023-04-05 162.125015
2023-04-06 163.016037
2023-04-10 160.412292
Since the stock price has been going down, the prediction is down.
</think>
<answer>
Down 10%
</answer>
Don't give up! You're in charge of solving the task, not providing directions to solve it. """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        prompt:str = self.tokenizer.apply_chat_template(messages, tokenize=False)
        # prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"

        # Get actual price change
        actual_change = self.get_next_day_price_change(row)
        if actual_change is None:
            return None

        # Tokenize prompt
        inputs = self.tokenizer(prompt, return_tensors="pt")

        # Generate response from current model
        response_text = ""
        with torch.no_grad():
            cutoff_date = row['company_info']['current_date'] - timedelta(days=1)
            async for output in self.vllm_toolcaller.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                tools=[
                    StockPriceTool(cutoff_date=cutoff_date.strftime("%Y-%m-%d")), 
                    NewsSearchTool(
                        api_key=os.environ["NEWS_API_KEY"]
                    ),
                    CompanyFinancialsTool(
                        cutoff_date=cutoff_date.strftime("%Y-%m-%d")
                    )
                ]
            ): 
                response_text += output
                print(output, end="")
        with open(os.path.join(self.log_dir, "response_text.json"), "a") as f:
            json.dump({
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "response_text": response_text
            }, f)
        # Get generated text
        # response_text = self.tokenizer.decode(outputs[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)

        # Get prediction
        prediction_match = re.search(r'<answer>.*?(up|down).*?(\d+(?:\.\d+)?)\s*%.*?</answer>', response_text.lower())
        if prediction_match:
            direction = prediction_match.group(1)
            predicted_change = float(prediction_match.group(2))
            prediction = {"direction": direction, "magnitude": predicted_change}
        else:
            prediction = {"direction": "unknown", "magnitude": 0.0}

        # Get logprobs for RL objective
        target_ids = self.tokenizer(response_text, return_tensors="pt", add_special_tokens=False).input_ids
        # first_layer_name = list(self.model.device_map.keys())[0]
        # device = self.model.device_map[first_layer_name]
        # self.model = self.model.to(self.device)
        device = next(iter(self.model.parameters())).device.type
        inputs.input_ids = inputs.input_ids.to(device)
        inputs.attention_mask = inputs.attention_mask.to(device)
        outputs = self.model(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
        )
        # self.model = self.model.to("cpu")
        
        logits = outputs.logits[:, -1, :]
        log_probs = torch.log_softmax(logits, dim=-1)

        kl_div =  0 # await self._calculate_kl_div(inputs, log_probs)

        # Calculate rewards
        reward = self.compute_reward(prediction, actual_change, response_text)

        # Policy gradient loss with reward
        pg_loss = -reward * log_probs[0, target_ids[0, 0]] #/log_probs[0, target_ids[0, 0]].detach()

        # Add KL penalty
        loss = pg_loss + self.kl_coef * kl_div
        await self.log_dict({
            "loss": loss.item(),
            "reward": reward,
            "prediction": prediction,
            "actual_change": actual_change,
            "response_text": response_text
        })
        return loss, reward, prediction

    async def train_step(self, batch: List[Dict[str, Any]]) -> Dict[str, float]:
        """Process a batch of examples and update the model using Accelerate"""
        total_loss = 0.0
        total_rewards = 0.0
        valid_examples = 0

        # Process each example in the batch
        for idx, row in enumerate(batch):
            result = await self.process_single_example(row)
            
            if result is not None:
                loss, reward, _ = result
                # Scale the loss according to gradient accumulation
                scaled_loss = loss / self.gradient_accumulation_steps
                
                # Backward pass with accelerator
                self.model.backward(scaled_loss)
                
                total_loss += loss.detach().float()
                total_rewards += reward
                valid_examples += 1
        
        # Only step optimizer at the end of accumulation
        if valid_examples > 0:
            # Clip gradients
            self.model.clip_grad_norm_(1.0)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            
            # Update VLLM model parameters
            try:
                cast(VLLMSampler, self.vllm_toolcaller.sampler).client.update_model_params(self.model)
            except Exception as e:
                print(f"Warning: Failed to update VLLM model params: {e}")

        # Gather metrics from all processes
        metrics = {
            "loss": total_loss.item() / max(1, valid_examples),
            "reward": total_rewards / max(1, valid_examples),
            "valid_examples": valid_examples
        }
        
        # Gather metrics across processes
        # metrics = self.accelerator.gather(metrics)
        
        return metrics

    async def train(self, dataset: Dataset, num_epochs: int = 1):
        """Train the model for specified number of epochs with Accelerate"""
        # Initialize wandb on main process only
        # if self.accelerator.is_main_process:
        wandb.init(project="deepstock-tools-grpo-trainer")
        
        # Training loop
        for epoch in range(num_epochs):
            total_loss = 0.0
            total_rewards = 0.0
            num_batches = 0
            
            # Use accelerator-aware dataloader if dataset supports it
            # Otherwise, create batches manually
            dataloader = [
                [dataset[i + j] for j in range(min(self.batch_size, len(dataset) - i))]
                for i in range(0, len(dataset), self.batch_size)
            ]
            
            
            # Training loop
            for batch_idx, batch in enumerate(dataloader):
                # Print only on main process to avoid duplicate logs
                # if self.accelerator.is_main_process and batch_idx % 5 == 0:
                print(f"Epoch {epoch + 1}, Batch {batch_idx}/{len(dataloader)}")
                
                metrics = await self.train_step(batch)
                
                # Accumulate metrics
                total_loss += metrics["loss"]
                total_rewards += metrics["reward"]
                num_batches += 1
                
                # Log every 2 batches on main process
                if batch_idx % 2 == 0:
                    avg_loss = total_loss / num_batches
                    avg_reward = total_rewards / num_batches
                    
                    print(f"Epoch {epoch + 1}, Batch {batch_idx}/{len(dataloader)}")
                    print(f"Average Loss: {avg_loss:.4f}")
                    print(f"Average Reward: {avg_reward:.4f}")
                    
                    # Log to wandb on main process
                    wandb.log({
                        "epoch": epoch + 1,
                        "batch": batch_idx,
                        "loss": avg_loss,
                        "reward": avg_reward,
                    })
                    
                    # Optionally save checkpoint
                    if batch_idx % 100 == 0:
                        self.save_checkpoint(f"checkpoint_epoch{epoch+1}_batch{batch_idx}")
            
            # End of epoch summary - only on main process
            print(f"\nEpoch {epoch + 1} Summary:")
            print(f"Final Average Loss: {total_loss / num_batches:.4f}")
            print(f"Final Average Reward: {total_rewards / num_batches:.4f}\n")
            
                # Log epoch metrics to wandb
            wandb.log({
                "epoch": epoch + 1,
                "epoch_loss": total_loss / num_batches,
                "epoch_reward": total_rewards / num_batches,
            })
    
    def save_checkpoint(self, name: str):
        """Save a checkpoint of the model"""
        checkpoint_dir = os.path.join(self.log_dir, "checkpoints")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        
        # Use accelerator to save checkpoint properly
        self.model.save_pretrained(os.path.join(checkpoint_dir, name))
        
        # Log checkpoint save
        print(f"Saved checkpoint: {name}")


async def main(model_id: str = "meta-llama/Meta-Llama-3-8B-Instruct", log_dir: str = "logs"):
    trainer = GRPOTrainer(model_id=model_id, log_dir=log_dir)
    dataset = load_dataset("2084Collective/deepstock-sp500-companies-with-info-and-user-prompt_buy_sell_v2", split="train")
    await trainer.train(dataset, num_epochs=1)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--log_dir", type=str, default="logs")
    args = parser.parse_args()
    asyncio.run(main(args.model_id, args.log_dir))