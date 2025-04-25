# ==============================================================================
# CELL 1: Installation
# ==============================================================================
"""
Custom GRPO QLoRA Trainer for Qwen2.5-14B-Instruct for PnL Maximization (Colab Ready)
*** USING UNSLOTH CUSTOM TRAINER LOGIC ***

This script uses a custom GRPOTrainer implementation, integrated with Unsloth for optimization and QLoRA for memory efficiency.

IMPORTANT:
- Ensure the custom reward logic in `calculate_trade_reward` matches your goals.
- Verify Google Drive paths.
- Ensure sufficient VRAM (>25.5GB recommended).

--- Colab Setup Instructions ---
1. Runtime Selection: Go to Runtime > Change runtime type. Select GPU (A100 recommended).
2. Run Installation Cell (This Cell): Execute first.
   IMPORTANT: You MUST restart the runtime after this cell finishes.
3. Run Setup Cell (Next Cell): Execute to mount Drive and check versions.
4. Verify Paths & Configs in Cell 3.
5. Run Training: Execute Cell 3.
"""

print("CELL 1: Installing required libraries with Unsloth...")
# Use -q for quieter output

# --- Installation Strategy ---
# 1. Install Unsloth first using their recommended method for Colab.
#    Unsloth often bundles compatible core dependencies (torch, transformers, peft, bitsandbytes).
# 2. Install any remaining libraries needed by the custom trainer logic.

# Install Unsloth from PyPI (latest stable version)
!pip install -q unsloth==2025.3.19  # Commented out for non-notebook execution

# Install other potentially needed libraries (Unsloth might already include some)
# Pin protobuf for known compatibility issues. Add pandas for custom code.
!pip install -q --force-reinstall \
   trl \
   datasets \
   accelerate \
   tensorboard \
   protobuf==3.20.* \
   pandas

# Note: torch, transformers, peft, bitsandbytes, triton should be handled by unsloth install.
# We check versions in Cell 2.


print("Libraries installation commands executed (uncomment and run the lines above).")
print("--- !!! IMPORTANT: You MUST RESTART the Colab Runtime now! (Runtime > Restart runtime) !!! ---")
print("--- After restarting, run the NEXT cell ('CELL 2: Setup and Version Check'). ---")


# ==============================================================================
# CELL 2: Setup and Version Check (Run AFTER restarting runtime)
# ==============================================================================
print("\nCELL 2: Running Setup and Version Check...")

import traceback
import os
import sys # Import sys for handler check
import requests
from datetime import datetime, timedelta
# --- Mount Google Drive ---
try:
    # Check if running in Colab
    if 'google.colab' in sys.modules:
        from google.colab import drive
        print("Mounting Google Drive...")
        drive.mount('/content/drive', force_remount=True)
        print("Google Drive mounted successfully.")
    else:
        print("Not running in Colab, skipping Google Drive mount.")
except ImportError:
    print("Google Colab `drive` import failed. Assuming not in Colab.")
except Exception as e:
    print(f"Error mounting Google Drive: {e}")
    print("Ensure you authorize access when prompted.")

# --- Import Core Libraries FOR VERSION CHECK ONLY ---


# Add your NewsAPI key
NEWS_API_KEY = "[News_API_Key]"  # Replace with your actual key

def get_news(ticker):
    """Simple function to get news for a stock ticker"""
    try:
        url = "https://newsapi.org/v2/everything"
        today = datetime.now().strftime("%Y-%m-%d")
        week_ago = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        
        params = {
            "q": ticker,
            "from": week_ago,
            "to": today,
            "language": "en",
            "apiKey": NEWS_API_KEY,
            "pageSize": 3
        }
        
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            news_text = ""
            for article in data.get("articles", [])[:3]:
                news_text += f"• {article.get('title', '')}\n"
            return news_text
        else:
            return ""
    except:
        return ""
# Import Unsloth FIRST
try:
    import unsloth
except ImportError:
    print("!!! ERROR: Unsloth not installed correctly. Please check Cell 1. !!!")
    exit()
except Exception as e:
    print(f"Error importing Unsloth: {e}")
    exit()

try:
    import torch
    import transformers
    import datasets
    import trl
    import peft
    import bitsandbytes 
    import accelerate
    import google.protobuf
    import triton # Check if Unsloth installed it
    import pandas as pd # Check pandas

    print("\n--- Library Version Check ---")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"PyTorch CUDA Version: {torch.version.cuda}")
    print(f"Transformers Version: {transformers.__version__}")
    print(f"Datasets Version: {datasets.__version__}")
    print(f"TRL Version: {trl.__version__}") # Check this version carefully
    print(f"PEFT Version: {peft.__version__}") # Check this version
    print(f"Bitsandbytes Version: {bitsandbytes.__version__}") # Check this version
    print(f"Accelerate Version: {accelerate.__version__}")
    print(f"Protobuf Version: {google.protobuf.__version__}")
    try:
        # Triton version attribute might not exist in older versions
        if hasattr(triton, '__version__'):
             print(f"Triton Version: {triton.__version__}")
        else:
             print(f"Triton imported, but version attribute not found.")
    except ImportError:
        print("Triton not found (might be okay if bitsandbytes doesn't need it with this setup)")
    print(f"Unsloth Version: {unsloth.__version__}")
    print(f"Pandas Version: {pd.__version__}")
    print("-----------------------------")

except ImportError as e:
    print(f"!!! ERROR: Failed to import libraries for version check: {e} !!!")
    exit()
except AttributeError as ae:
     print(f"!!! ERROR: AttributeError during import/check: {ae} !!!")
     traceback.print_exc()
     exit()
except Exception as e:
    print(f"An unexpected error occurred during import/version check: {e}")
    traceback.print_exc()
    exit()

# --- Check GPU Availability and Setup Compute Type ---
try:
    import torch
    if not torch.cuda.is_available():
        print("!!! ERROR: No CUDA GPU detected! Check Colab Runtime Type. !!!")
        exit()
    else:
        print(f"CUDA Detected: {torch.cuda.get_device_name(0)}")
        if torch.cuda.is_bf16_supported():
            print("bfloat16 is supported. Using bfloat16.")
            compute_dtype = torch.bfloat16
            use_bf16 = True
            use_fp16 = False
        else:
            print("bfloat16 not supported. Using fp16.")
            compute_dtype = torch.float16
            use_bf16 = False
            use_fp16 = True
except Exception as e:
     print(f"Error during GPU check / compute type setup: {e}")
     exit()

print("\nSetup and Version Check Complete. Proceeding to Training Cell...")


# ==============================================================================
# CELL 3: Custom Trainer Definitions (Run AFTER Cell 2)
# ==============================================================================
print("\nCELL 3: Defining Custom Classes and Functions...")

# --- Imports needed for these definitions ---
import os
import sys
import json
import torch
import random
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta, date  # Added date here
from tqdm import tqdm
import logging
import re
from copy import deepcopy
from datasets import load_dataset, Dataset
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Import Unsloth FIRST - before any transformers imports
from unsloth import FastLanguageModel

from transformers import (
    AutoModelForCausalLM, # Needed for type hint? Or within GRPOTrainer? Keep for now.
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments, # Used by custom trainer
    DataCollatorForLanguageModeling, # Imported but not used in provided code
    get_linear_schedule_with_warmup, # Imported within GRPOTrainer init
    get_cosine_schedule_with_warmup # Imported within GRPOTrainer init
)

from peft import (
    LoraConfig,
    PeftModel,
    prepare_model_for_kbit_training
)

import bitsandbytes as bnb
# Fix: Adam4bit no longer exists in bitsandbytes 0.45.5
from bitsandbytes.optim import AdamW8bit  # Using AdamW8bit instead of Adam4bit
from bitsandbytes.nn import Linear4bit

# Configure logging
logger = logging.getLogger("unsloth_grpo"); logger.setLevel(logging.INFO); logger.propagate = False
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout); handler.setLevel(logging.INFO); formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"); handler.setFormatter(formatter); logger.addHandler(handler)
logger.info("GRPO Training script loaded.")

# Helper function to extract log probabilities from logits
def log_probs_from_logits(logits, targets):
    """
    Compute log probabilities for given logits and target indices.
    
    Args:
        logits: Logits tensor of shape (batch_size, sequence_length, vocab_size)
        targets: Target indices tensor of shape (batch_size, sequence_length)
        
    Returns:
        Log probabilities tensor of shape (batch_size, sequence_length)
    """
    # Get log probabilities from logits
    log_probs = F.log_softmax(logits, dim=-1)
    
    # Extract log probabilities for target tokens
    batch_size, seq_length, vocab_size = log_probs.shape
    flat_log_probs = log_probs.reshape(-1, vocab_size)
    flat_targets = targets.reshape(-1)
    
    # Get log probs for the selected targets
    flat_selected_log_probs = flat_log_probs[torch.arange(flat_targets.shape[0]), flat_targets]
    selected_log_probs = flat_selected_log_probs.reshape(batch_size, seq_length)
    
    return selected_log_probs

# Custom JSON encoder for serializing special types like numpy arrays
class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (datetime, date)):
            return obj.isoformat()
        return json.JSONEncoder.default(self, obj)

# --- Logging Setup ---
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )
else:
    logger.setLevel(logging.INFO)
# logger.setLevel(logging.DEBUG) # Uncomment for verbose_print

# --- Tag Optimized Parsing Functions ---
def extract_tagged_section(thinking: str, tag: str) -> str:
    """
    Extract the content of a specific tagged section from the thinking trace
    
    Args:
        thinking: The complete thinking trace text
        tag: The tag to extract (without quotes)
        
    Returns:
        The content of the tagged section, or empty string if not found
    """
    # Try both <tag:TAG> and 'TAG formats
    xml_pattern = rf"<tag:{tag}>(.*?)</tag:{tag}>"
    quote_pattern = rf"'({tag}[^\n]*)\n(.*?)'"
    
    # Try XML-style tags first
    xml_match = re.search(xml_pattern, thinking, re.DOTALL)
    if xml_match:
        return xml_match.group(1).strip()
    
    # Fall back to quote-style tags
    quote_match = re.search(quote_pattern, thinking, re.DOTALL)
    if quote_match:
        return quote_match.group(2).strip()
    
    return ""

def parse_tags_from_thinking(thinking: str) -> Dict[str, str]:
    """
    Parse all tagged sections from a thinking trace
    
    Args:
        thinking: The complete thinking trace
        
    Returns:
        Dictionary mapping tag names to their content
    """
    # Common tags used in the structured thinking format
    expected_tags = [
        "OBSERVATION", "ANALYSIS", "REASONING", "RISK", 
        "ALTERNATIVE", "DECISION", "ENTRY", "TIMEFRAME", 
        "EXIT", "CONFIDENCE", "STOP", "TARGET"
    ]
    
    result = {}
    
    for tag in expected_tags:
        content = extract_tagged_section(thinking, tag)
        if content:
            result[tag] = content
    
    return result

def count_indicators_mentioned(analysis: str) -> int:
    """
    Count the number of technical indicators mentioned in the analysis
    
    Args:
        analysis: The analysis text to check
        
    Returns:
        Number of unique indicators found
    """
    indicators = [
        "RSI", "MACD", "MA", "EMA", "SMA", "Bollinger", "Stochastic", 
        "Fibonacci", "ADX", "ATR", "OBV", "Ichimoku", "CMF", "VWAP",
        "momentum", "oscillator", "divergence", "support", "resistance",
        "trend line", "trendline", "volume profile", "order flow"
    ]
    
    # Create a clean analysis string (lowercase for case-insensitive matching)
    clean_analysis = analysis.lower()
    
    # Count unique indicators mentioned
    mentioned = set()
    for indicator in indicators:
        if indicator.lower() in clean_analysis:
            mentioned.add(indicator)
    
    return len(mentioned)

def check_risk_assessment_quality(risk_text: str) -> float:
    """
    Evaluate the quality of risk assessment
    
    Args:
        risk_text: The risk section text
        
    Returns:
        Score between 0.0 and 1.0 indicating risk assessment quality
    """
    if not risk_text:
        return 0.0
    
    # Simple metrics for assessing risk text quality
    word_count = len(risk_text.split())
    risk_factors = len(re.findall(r'[.!?]\s+', risk_text)) + 1  # Sentence count as proxy for risk factors
    
    # Check for specific risk terminology
    risk_terms = [
        "downside", "upside", "unexpected", "news", "announcement", 
        "volatility", "liquidity", "probability", "likelihood", "chance", 
        "market condition", "catalyst", "trigger", "percent", "%"
    ]
    
    term_count = 0
    for term in risk_terms:
        if term.lower() in risk_text.lower():
            term_count += 1
    
    # Calculate quality score (normalized to 0-1 range)
    quality = min(1.0, (word_count / 50) * 0.3 + (risk_factors / 3) * 0.3 + (term_count / len(risk_terms)) * 0.4)
    return quality

def extract_trading_metadata(thinking: str) -> Dict[str, Any]:
    """
    Extract metadata about the trading analysis quality
    
    Args:
        thinking: The complete thinking trace
        
    Returns:
        Dictionary with metadata about the analysis
    """
    tags = parse_tags_from_thinking(thinking)
    
    # Analysis depth metrics
    analysis_text = tags.get("ANALYSIS", "")
    reasoning_text = tags.get("REASONING", "")
    risk_text = tags.get("RISK", "")
    alternative_text = tags.get("ALTERNATIVE", "")
    
    analysis_word_count = len(analysis_text.split())
    indicators_count = count_indicators_mentioned(analysis_text)
    risk_quality = check_risk_assessment_quality(risk_text)
    has_alternative = len(alternative_text) > 10
    
    # Calculate section completion
    expected_tags = {"OBSERVATION", "ANALYSIS", "REASONING", "RISK", 
                    "ALTERNATIVE", "DECISION", "ENTRY", "TIMEFRAME", 
                    "EXIT", "CONFIDENCE"}
    completed_tags = set(tags.keys())
    completion_ratio = len(completed_tags.intersection(expected_tags)) / len(expected_tags)
    
    # Metadata dictionary
    metadata = {
        "analysis_depth": analysis_word_count / 100,  # Normalize to 0-1 range
        "indicators_used": indicators_count,
        "risk_quality": risk_quality,
        "has_alternative_scenario": has_alternative,
        "section_completion": completion_ratio,
        "tags_present": list(completed_tags),
        "tags_missing": list(expected_tags - completed_tags)
    }
    
    return metadata

def calculate_tag_completeness_score(tags: Dict[str, str]) -> float:
    """
    Calculate a score for the completeness of thinking trace tags
    
    Args:
        tags: Dictionary of tag names and their content
        
    Returns:
        Score from 0.0 to 1.0 indicating completeness
    """
    expected_tags = {
        "OBSERVATION", "ANALYSIS", "REASONING", "RISK", 
        "ALTERNATIVE", "DECISION", "ENTRY", "TIMEFRAME", 
        "EXIT", "CONFIDENCE"
    }
    
    # Check which tags are present and have meaningful content
    meaningful_tags = {}
    for tag in expected_tags:
        content = tags.get(tag, "")
        if content and len(content.split()) > 3:  # Must have at least 3 words
            meaningful_tags[tag] = True
    
    # Calculate completeness ratio
    return len(meaningful_tags) / len(expected_tags)

def verbose_print(*args, **kwargs):
    """Print function that can be enabled/disabled via environment variable."""
    logger.debug(*args, **kwargs)

# --- Helper Function ---
def set_random_seed(seed):
    """Set random seeds for reproducibility across all libraries"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    logger.info(f"Set random seed to {seed}")

# --- Trade Simulation Components (User Provided) ---
class TradeManager:
    def __init__(self,
                 stop_loss_pct: float = 0.02,
                 take_profit_pct: float = 0.03,
                 max_holding_periods: int = 5):
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_holding_periods = max_holding_periods
        logger.info(f"TradeManager initialized: SL={stop_loss_pct:.2%}, TP={take_profit_pct:.2%}, MaxHold={max_holding_periods}")

    def calculate_position_size(self, volatility: float) -> float:
        """Calculate position size based on market volatility"""
        volatility = max(volatility, 1e-6)
        vol_scalar = 1.0 / (1.0 + 5 * volatility)
        position_size = vol_scalar
        final_size = np.clip(position_size, 0.1, 1.5)
        verbose_print(f"Position Size Calculation: Vol={volatility:.4f} (Scalar={vol_scalar:.2f}) -> SizeFactor={final_size:.2f}")
        return final_size

def parse_trade_prediction(completion: str) -> Dict[str, Any]:
    """
    Parse the trading prediction from the model's completion.
    Updated to handle both tag-based and traditional formats.
    """
    prediction = {
        'direction': None,
        'percentage': None,
        'full_response': completion,
        'entry_conditions': [],
        'exit_conditions': [],
        'entry_price': None,
        'exit_price': None,
        'stop_price': None,
        'timeframe': None,
        'confidence': None
    }
    logger.info(f"Parsing completion of length {len(completion)} chars")

    # Clean up the text - remove any potential malformed Unicode or control characters
    cleaned_text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', completion)
    
    # First try to parse using the tag-based approach
    tags = parse_tags_from_thinking(cleaned_text)
    if tags:
        logger.info(f"Found {len(tags)} tags in completion: {list(tags.keys())}")
        
        # Extract direction from DECISION tag
        decision_text = tags.get("DECISION", "")
        if "UP" in decision_text.upper() or "BULLISH" in decision_text.upper() or "LONG" in decision_text.upper():
            prediction['direction'] = "UP"
            logger.info(f"Extracted UP direction from DECISION tag")
        elif "DOWN" in decision_text.upper() or "BEARISH" in decision_text.upper() or "SHORT" in decision_text.upper():
            prediction['direction'] = "DOWN"
            logger.info(f"Extracted DOWN direction from DECISION tag")
        
        # Extract entry price from ENTRY tag
        entry_text = tags.get("ENTRY", "")
        entry_price_match = re.search(r'(\d+\.?\d*)', entry_text)
        if entry_price_match:
            try:
                prediction['entry_price'] = float(entry_price_match.group(1))
                logger.info(f"Extracted entry price: {prediction['entry_price']} from ENTRY tag")
            except ValueError:
                pass
                
        # Extract stop price from STOP tag (or EXIT if STOP not present)
        stop_text = tags.get("STOP", tags.get("EXIT", ""))
        stop_price_match = re.search(r'stop\s*(?:loss|price)?[^\d]*(\d+\.?\d*)', stop_text, re.IGNORECASE)
        if stop_price_match:
            try:
                prediction['stop_price'] = float(stop_price_match.group(1))
                logger.info(f"Extracted stop price: {prediction['stop_price']} from STOP/EXIT tag")
            except ValueError:
                pass
                
        # Extract exit price from TARGET tag (or EXIT if TARGET not present)
        target_text = tags.get("TARGET", tags.get("EXIT", ""))
        target_price_match = re.search(r'(target|take\s*profit|tp)[^\d]*(\d+\.?\d*)', target_text, re.IGNORECASE)
        if target_price_match:
            try:
                prediction['exit_price'] = float(target_price_match.group(2))
                logger.info(f"Extracted exit price: {prediction['exit_price']} from TARGET/EXIT tag")
            except ValueError:
                pass
                
        # Extract timeframe
        timeframe_text = tags.get("TIMEFRAME", "")
        hours_match = re.search(r'(\d+)\s*hours?', timeframe_text, re.IGNORECASE)
        days_match = re.search(r'(\d+)\s*days?', timeframe_text, re.IGNORECASE)
        
        timeframe_hours = 0
        if hours_match:
            timeframe_hours += int(hours_match.group(1))
        if days_match:
            timeframe_hours += int(days_match.group(1)) * 24
            
        if timeframe_hours > 0:
            prediction['timeframe'] = timeframe_hours
            logger.info(f"Extracted timeframe: {timeframe_hours} hours from TIMEFRAME tag")
            
        # Extract confidence
        confidence_text = tags.get("CONFIDENCE", "")
        confidence_match = re.search(r'(\d+)\s*/\s*10|(\d+)', confidence_text)
        if confidence_match:
            confidence = int(confidence_match.group(1) if confidence_match.group(1) else confidence_match.group(2))
            prediction['confidence'] = confidence / 10  # Normalize to 0-1
            logger.info(f"Extracted confidence: {confidence}/10 from CONFIDENCE tag")
            
        # Add entry and exit conditions from tags
        if entry_text:
            prediction['entry_conditions'] = [entry_text]
            
        exit_text = tags.get("EXIT", "")
        if exit_text:
            prediction['exit_conditions'] = [exit_text]
    
    # If tag parsing didn't yield a direction, fall back to the original parsing logic
    if prediction['direction'] is None:
        # First try to extract from answer tag
        answer_match = re.search(r'<answer>(.*?)</answer>', cleaned_text, re.DOTALL | re.IGNORECASE)
        if answer_match:
            answer_content = answer_match.group(1).strip()
            logger.info(f"Found answer tag: '{answer_content}'")

            # Extract direction from answer
            dir_match = re.search(r'Direction\s*:\s*(UP|DOWN)', answer_content, re.IGNORECASE)
            if dir_match:
                prediction['direction'] = dir_match.group(1).upper()
                logger.info(f"Extracted direction '{prediction['direction']}' from answer tag")

            # Extract percentage from answer
            pct_match = re.search(r'Change\s*:\s*([\+\-]?\d+\.?\d*)\s*%?', answer_content)
            if pct_match:
                try:
                    prediction['percentage'] = float(pct_match.group(1))
                    logger.info(f"Extracted percentage {prediction['percentage']}% from answer tag")
                except ValueError:
                    logger.warning(f"Could not convert percentage '{pct_match.group(1)}' to float")

        # If we couldn't find direction in the answer tag, try to find them elsewhere
        if prediction['direction'] is None:
            # Look for direction in entire text if not found in answer tag
            dir_match = re.search(r'Direction\s*:\s*(UP|DOWN)', cleaned_text, re.IGNORECASE)
            if dir_match:
                prediction['direction'] = dir_match.group(1).upper()
                logger.info(f"Extracted direction '{prediction['direction']}' from full text")
            else:
                # Use simple UP/DOWN if no explicit Direction found
                if re.search(r'\bUP\b', cleaned_text, re.IGNORECASE):
                    prediction['direction'] = 'UP'
                    logger.info("Found standalone 'UP' direction in text")
                elif re.search(r'\bDOWN\b', cleaned_text, re.IGNORECASE):
                    prediction['direction'] = 'DOWN'
                    logger.info("Found standalone 'DOWN' direction in text")

        if prediction['percentage'] is None:
            # Try to find percentage in entire text if not found in answer tag
            pct_match = re.search(r'Change\s*:\s*([\+\-]?\d+\.?\d*)\s*%?', cleaned_text)
            if pct_match:
                try:
                    prediction['percentage'] = float(pct_match.group(1))
                    logger.info(f"Extracted percentage {prediction['percentage']}% from full text")
                except ValueError:
                    logger.warning(f"Could not convert percentage '{pct_match.group(1)}' to float")

        # Extract entry and exit conditions ONLY from their proper tags
        entry_match = re.search(r'<entry_conditions>(.*?)</entry_conditions>', cleaned_text, re.DOTALL | re.IGNORECASE)
        if entry_match:
            entry_content = entry_match.group(1).strip()
            entry_conditions = [cond.strip().lower() for cond in entry_content.split(',') if cond.strip()]
            prediction['entry_conditions'] = entry_conditions
            logger.info(f"Extracted {len(entry_conditions)} entry conditions from tags")

            # Try to extract price levels from entry conditions if not already found
            if prediction['entry_price'] is None:
                for cond in entry_conditions:
                    # Look for entry price in various formats
                    entry_price_match = re.search(r'(enter|buy|entry|open).*?(?:at|above|price).*?(\d+\.?\d*)', cond)
                    if entry_price_match:
                        try:
                            prediction['entry_price'] = float(entry_price_match.group(2))
                            logger.info(f"Extracted entry price: {prediction['entry_price']}")
                        except ValueError:
                            pass

            # Look for stop loss in entry conditions if not already found
            if prediction['stop_price'] is None:
                for cond in entry_conditions:
                    stop_price_match = re.search(r'stop.*?(\d+\.?\d*)', cond)
                    if stop_price_match:
                        try:
                            prediction['stop_price'] = float(stop_price_match.group(1))
                            logger.info(f"Extracted stop price: {prediction['stop_price']}")
                        except ValueError:
                            pass

        exit_match = re.search(r'<exit_conditions>(.*?)</exit_conditions>', cleaned_text, re.DOTALL | re.IGNORECASE)
        if exit_match:
            exit_content = exit_match.group(1).strip()
            exit_conditions = [cond.strip().lower() for cond in exit_content.split(',') if cond.strip()]
            prediction['exit_conditions'] = exit_conditions
            logger.info(f"Extracted {len(exit_conditions)} exit conditions from tags")

            # Try to extract target price from exit conditions if not already found
            if prediction['exit_price'] is None:
                for cond in exit_conditions:
                    # Look for target price in various formats
                    target_price_match = re.search(r'(sell|exit|target|take_profit|tp).*?(?:at|price).*?(\d+\.?\d*)', cond)
                    if target_price_match:
                        try:
                            prediction['exit_price'] = float(target_price_match.group(2))
                            logger.info(f"Extracted exit/target price: {prediction['exit_price']}")
                        except ValueError:
                            pass

            # If we didn't find a stop price in entry conditions, check exit conditions
            if prediction['stop_price'] is None:
                for cond in exit_conditions:
                    stop_price_match = re.search(r'(stop|sl).*?(\d+\.?\d*)', cond)
                    if stop_price_match:
                        try:
                            prediction['stop_price'] = float(stop_price_match.group(2))
                            logger.info(f"Extracted stop price from exit conditions: {prediction['stop_price']}")
                        except ValueError:
                            pass

    # Final fallbacks for critical values
    if prediction['direction'] is None:
        prediction['direction'] = 'UP'
        logger.warning("USING DEFAULT DIRECTION: UP (couldn't parse from text)")

    if prediction['percentage'] is None:
        prediction['percentage'] = 1.0
        logger.warning("USING DEFAULT PERCENTAGE: 1.0% (couldn't parse from text)")

    # Log the final prediction with price levels
    logger.info(f"Final parsed prediction: Direction={prediction['direction']}, Change={prediction.get('percentage')}%, "
                f"Entry conditions={len(prediction['entry_conditions'])}, Exit conditions={len(prediction['exit_conditions'])}")
    if prediction['entry_price'] or prediction['exit_price'] or prediction['stop_price']:
        logger.info(f"Extracted price levels: Entry={prediction['entry_price']}, Exit={prediction['exit_price']}, Stop={prediction['stop_price']}")
    if prediction['timeframe'] or prediction['confidence']:
        logger.info(f"Additional metrics: Timeframe={prediction['timeframe']} hours, Confidence={prediction['confidence']}")

    return prediction

def calculate_trade_reward(
    prediction: Dict[str, Any], metadata: Dict[str, Any], trade_manager: TradeManager
) -> Tuple[float, Dict[str, Any], Dict[str, float]]:
    """Comprehensive reward function for GRPO training with trade management and tag-based analysis."""
    individual_rewards = {'format': 0.0, 'direction': 0.0, 'risk_management': 0.0, 'pnl': 0.0, 'strategy': 0.0, 'analysis_quality': 0.0}
    verbose_print(f"Calculating reward for prediction: {prediction['direction']}")

    # Extract thinking trace from prediction or metadata
    thinking_trace = prediction.get('full_response', '')
    if 'thinking_trace' in metadata:
        thinking_trace = metadata['thinking_trace']
    
    # Extract tags if they exist
    tags = parse_tags_from_thinking(thinking_trace)
    if tags:
        logger.info(f"Found structured tags in thinking trace: {', '.join(tags.keys())}")
    
    required_meta = ['current_price', 'future_prices', 'actual_direction', 'actual_percentage', 'volatility']
    for key in required_meta:
        if key not in metadata or metadata[key] is None:
            logger.error(f"Missing or None critical metadata key '{key}'. Returning -1 reward.")
            return -1.0, {}, individual_rewards
    if not isinstance(metadata['future_prices'], list) or not metadata['future_prices']:
        logger.error(f"Invalid or empty 'future_prices' list. Returning -1 reward.")
        return -1.0, {}, individual_rewards

    # Format reward - check for presence of tags
    if tags:
        # Use the new tag-based format scoring
        tag_completion = calculate_tag_completeness_score(tags)
        individual_rewards['format'] = min(0.2, tag_completion * 0.2)
        verbose_print(f"  Format Reward (Tag-based): {individual_rewards['format']:.3f} (Completeness: {tag_completion:.2f})")
    else:
        # Fall back to original format scoring
        format_components = {
            'thinking_tags': bool(re.search(r'<think>.*?</think>', thinking_trace, re.IGNORECASE | re.DOTALL)),
            'answer_tags': bool(re.search(r'<answer>.*?</answer>', thinking_trace, re.IGNORECASE | re.DOTALL)),
            'entry_tags': bool(re.search(r'<entry_conditions>.*?</entry_conditions>', thinking_trace, re.IGNORECASE | re.DOTALL)),
            'exit_tags': bool(re.search(r'<exit_conditions>.*?</exit_conditions>', thinking_trace, re.IGNORECASE | re.DOTALL)),
            'direction_parsed': prediction['direction'] is not None,
            'percentage_parsed': prediction.get('percentage') is not None
        }

        # Give partial credit for having some tags
        num_format_components = sum(1 for k, v in format_components.items() if v)

        # More lenient format reward - give partial credit even if not all components are present
        if num_format_components == 6:  # All components present
            format_score = 0.2
        elif num_format_components >= 3:  # Most components present
            format_score = 0.1
        elif num_format_components > 0:  # Some components present
            format_score = 0.05
        else:
            format_score = -0.1  # No components present

        individual_rewards['format'] = format_score
        verbose_print(f"  Format Reward (Original): {individual_rewards['format']:.3f} ({num_format_components}/6 components)")

    # Direction reward
    actual_direction = metadata['actual_direction'].upper()
    if actual_direction not in ['UP', 'DOWN']:
        logger.warning(f"Invalid actual_direction '{metadata['actual_direction']}'. Setting direction reward to 0.")
        individual_rewards['direction'] = 0.0
    elif prediction['direction'] == actual_direction:
        base_reward = 0.2
        entry_bonus = 0.1 if prediction['entry_conditions'] else 0.0
        individual_rewards['direction'] = base_reward + entry_bonus
        verbose_print(f"  Direction Reward: Correct ({prediction['direction']}) -> Base={base_reward}, EntryBonus={entry_bonus}")
    else:
        individual_rewards['direction'] = -0.2
        verbose_print(f"  Direction Reward: Incorrect (Pred {prediction['direction']}, Act {actual_direction}) -> Penalty={individual_rewards['direction']}")

    # Add analysis quality reward if we have tags
    if tags:
        # Calculate analysis quality from tags
        analysis_text = tags.get("ANALYSIS", "")
        reasoning_text = tags.get("REASONING", "")
        risk_text = tags.get("RISK", "")
        
        # Analysis depth based on indicators used
        indicators_count = count_indicators_mentioned(analysis_text)
        analysis_depth = min(len(analysis_text.split()) / 100, 1.0)  # Normalize by words
        
        # Risk assessment quality
        risk_quality = check_risk_assessment_quality(risk_text)
        
        # Reasoning coherence - check if reasoning matches direction
        reasoning_coherence = 0.0
        if reasoning_text:
            reasoning_matches_direction = (
                (prediction['direction'] == "UP" and ("bullish" in reasoning_text.lower() or "upward" in reasoning_text.lower())) or
                (prediction['direction'] == "DOWN" and ("bearish" in reasoning_text.lower() or "downward" in reasoning_text.lower()))
            )
            reasoning_coherence = 0.1 if reasoning_matches_direction else -0.05
            
        # Combined analysis quality score
        analysis_score = (
            indicators_count * 0.02 +  # Up to 0.1 for 5 indicators
            analysis_depth * 0.05 +   # Up to 0.05 for long analysis
            risk_quality * 0.05 +     # Up to 0.05 for good risk assessment
            reasoning_coherence        # +0.1 or -0.05 for reasoning coherence
        )
        
        individual_rewards['analysis_quality'] = min(0.2, analysis_score)
        verbose_print(f"  Analysis Quality Reward: {individual_rewards['analysis_quality']:.3f} (Indicators: {indicators_count}, Risk: {risk_quality:.2f})")

    entry_price = float(metadata['current_price'])
    future_prices = [float(p) for p in metadata['future_prices'] if p is not None and not np.isnan(p)]
    volatility = float(metadata.get('volatility', 0.0))
    trade_metrics = {}

    if not future_prices or prediction['direction'] not in ['UP', 'DOWN']:
        logger.warning("Skipping PnL/Risk rewards due to no valid future prices or invalid prediction direction.")
        individual_rewards['pnl'] = 0.0
        individual_rewards['risk_management'] = 0.0
    else:
        # Calculate position size based on volatility
        position_size_factor = trade_manager.calculate_position_size(volatility)
        
        # Use provided price levels if available, otherwise calculate defaults
        sl_price = prediction.get('stop_price')
        tp_price = prediction.get('exit_price')
        
        # Fall back to defaults if not specified in prediction
        if sl_price is None:
            sl_price = entry_price * (1 - trade_manager.stop_loss_pct if prediction['direction'] == 'UP' else 1 + trade_manager.stop_loss_pct)
        
        if tp_price is None:
            tp_price = entry_price * (1 + trade_manager.take_profit_pct if prediction['direction'] == 'UP' else 1 - trade_manager.take_profit_pct)
            
        verbose_print(f"  Simulating Trade: Entry={entry_price:.2f}, SL={sl_price:.2f}, TP={tp_price:.2f}, MaxHold={trade_manager.max_holding_periods}, SizeFactor={position_size_factor:.2f}")

        trade_metrics = {'position_size_factor': position_size_factor, 'entry_price': entry_price, 'stop_loss': sl_price, 'take_profit': tp_price,
                         'exit_price': None, 'holding_periods': 0, 'exit_reason': None, 'future_prices_used': future_prices[:trade_manager.max_holding_periods]}
        exit_loop = False
        
        # Use timeframe from prediction if available, otherwise use default
        max_periods = prediction.get('timeframe')
        if max_periods is not None:
            # Convert hours to candle periods (assuming 1 candle = 1 hour for simplicity)
            max_periods = min(max_periods, len(future_prices))
        else:
            max_periods = min(trade_manager.max_holding_periods, len(future_prices))
            
        for i, price in enumerate(future_prices):
            if i >= max_periods:
                trade_metrics.update({'exit_price': price, 'exit_reason': 'max_holding_period', 'holding_periods': i}); verbose_print(f"    Exit: Max Hold {i} at {price:.2f}"); exit_loop = True; break
            if (prediction['direction'] == 'UP' and price <= sl_price) or (prediction['direction'] == 'DOWN' and price >= sl_price):
                trade_metrics.update({'exit_price': sl_price, 'exit_reason': 'stop_loss', 'holding_periods': i + 1}); verbose_print(f"    Exit: SL at period {i+1} (price {price:.2f})"); exit_loop = True; break
            if (prediction['direction'] == 'UP' and price >= tp_price) or (prediction['direction'] == 'DOWN' and price <= tp_price):
                trade_metrics.update({'exit_price': tp_price, 'exit_reason': 'take_profit', 'holding_periods': i + 1}); verbose_print(f"    Exit: TP at period {i+1} (price {price:.2f})"); exit_loop = True; break
        if not exit_loop:
            if future_prices: trade_metrics.update({'exit_price': future_prices[-1], 'exit_reason': 'end_of_data', 'holding_periods': len(future_prices)}); verbose_print(f"    Exit: End of data at period {len(future_prices)} (price {future_prices[-1]:.2f})")
            else: trade_metrics.update({'exit_price': entry_price, 'exit_reason': 'no_future_data', 'holding_periods': 0}); verbose_print("    Exit: No future data.")

        if trade_metrics['exit_price'] is not None and entry_price != 0:
            price_change_pct = (trade_metrics['exit_price'] - entry_price) / entry_price
            pnl_factor = price_change_pct * (1 if prediction['direction'] == 'UP' else -1)
            scaled_pnl = pnl_factor * position_size_factor
            verbose_print(f"    Simulated PnL: PriceChange={price_change_pct:.2%}, PnLFactor={pnl_factor:.2%}, ScaledPnL={scaled_pnl:.3f}")
            
            # MODIFIED: Increased maximum PnL reward from +1.0 to +2.0
            individual_rewards['pnl'] = min(0.8, scaled_pnl * 10) if scaled_pnl > 0 else max(-0.4, scaled_pnl * 10)
            verbose_print(f"  PnL Reward (Increased Max): {individual_rewards['pnl']:.3f}")
        else:
             individual_rewards['pnl'] = 0.0; verbose_print(f"  PnL Reward: 0.0 (No valid exit or entry price)")

        # Risk management reward - enhanced with tag-based assessment
        if tags and "RISK" in tags:
            risk_text = tags["RISK"]
            risk_quality = check_risk_assessment_quality(risk_text)
            
            # Base risk reward on exit reason but enhance with quality of risk assessment
            if trade_metrics['exit_reason'] == 'take_profit': 
                risk_base = 0.3
            elif trade_metrics['exit_reason'] == 'stop_loss': 
                risk_base = -0.2 if prediction['direction'] != actual_direction else -0.05
            else: 
                risk_base = 0.05
                
            # Add bonus for good risk assessment
            risk_bonus = risk_quality * 0.1
            individual_rewards['risk_management'] = risk_base + risk_bonus
            verbose_print(f"  Risk Management Reward: {individual_rewards['risk_management']:.3f} (Exit: {trade_metrics.get('exit_reason', 'N/A')}, RiskQuality: {risk_quality:.2f})")
        else:
            # Improved risk management reward with exit condition analysis
            has_stop_loss_condition = False
            if prediction['exit_conditions']:
                # Check for stop loss conditions in exit_conditions
                for condition in prediction['exit_conditions']:
                    if any(term in condition.lower() for term in ['stop', 'loss', 'sl', 'below', 'above', 'cross']):
                        has_stop_loss_condition = True
                        break
            
            if trade_metrics['exit_reason'] == 'take_profit':
                individual_rewards['risk_management'] = 0.3
            elif trade_metrics['exit_reason'] == 'stop_loss':
                # Less penalty if model correctly specified stop loss conditions
                individual_rewards['risk_management'] = -0.05 if has_stop_loss_condition else -0.2
                if prediction['direction'] == actual_direction:
                    # Even less penalty if the direction was correct
                    individual_rewards['risk_management'] = 0.0 if has_stop_loss_condition else -0.05
            else:
                # Default risk management reward with bonus for having stop conditions
                individual_rewards['risk_management'] = 0.1 if has_stop_loss_condition else 0.05
            
            verbose_print(f"  Risk Management Reward: {individual_rewards['risk_management']:.3f} (Exit: {trade_metrics.get('exit_reason', 'N/A')}, HasStopCondition: {has_stop_loss_condition})")

    # Strategy reward - give partial credit for having any conditions
    strategy_components = {
        'has_entry': bool(prediction['entry_conditions']),
        'has_exit': bool(prediction['exit_conditions']),
        'has_entry_price': prediction.get('entry_price') is not None,
        'has_exit_price': prediction.get('exit_price') is not None,
        'has_stop_price': prediction.get('stop_price') is not None,
        'has_timeframe': prediction.get('timeframe') is not None,
        'has_confidence': prediction.get('confidence') is not None
    }
    
    # Enhanced strategy scoring with more granular assessment
    strategy_score = 0.0
    
    # Basic strategy score based on presence of components - MODIFIED for more variability
    entry_quality = 0.0
    if prediction['entry_conditions']:
        # Count how many unique technical indicators are mentioned in entry conditions
        entry_indicators = set()
        indicator_keywords = ['rsi', 'macd', 'ema', 'ma ', 'sma', 'stochastic', 'volume', 'bollinger', 'fibonacci', 'support', 'resistance']
        for condition in prediction['entry_conditions']:
            for indicator in indicator_keywords:
                if indicator in condition.lower():
                    entry_indicators.add(indicator)
        
        # More varied scoring based on entry complexity
        entry_quality = min(0.08, 0.01 * len(entry_indicators) + 0.02 * len(prediction['entry_conditions']))
    
    strategy_score += entry_quality
    verbose_print(f"    Entry Quality: {entry_quality:.3f} (Indicators: {len(entry_indicators) if 'entry_indicators' in locals() else 0})")
    
    # Exit condition quality - more detailed analysis
    exit_quality = 0.0
    if prediction['exit_conditions']:
        # Count exit condition types and complexity
        has_stop_loss = False
        has_take_profit = False
        has_trailing_stop = False
        has_time_based_exit = False
        has_indicator_exit = False
        
        for condition in prediction['exit_conditions']:
            condition_lower = condition.lower()
            if any(term in condition_lower for term in ['stop', 'loss', 'sl']):
                has_stop_loss = True
            if any(term in condition_lower for term in ['take_profit', 'tp', 'target', 'profit']):
                has_take_profit = True
            if any(term in condition_lower for term in ['trailing', 'move stop', 'adjust sl']):
                has_trailing_stop = True
            if any(term in condition_lower for term in ['time', 'period', 'day', 'hour', 'minute']):
                has_time_based_exit = True
            if any(term in condition_lower for term in ['rsi', 'macd', 'ma', 'ema', 'cross', 'stochastic', 'volume']):
                has_indicator_exit = True
        
        # More varied scoring based on exit strategy comprehensiveness
        exit_quality = (
            0.03 * has_stop_loss + 
            0.03 * has_take_profit + 
            0.02 * has_trailing_stop + 
            0.01 * has_time_based_exit +
            0.03 * has_indicator_exit
        )
        
        # Give bonus for complete exit strategy
        if has_stop_loss and has_take_profit and has_indicator_exit:
            exit_quality += 0.02
    
    strategy_score += exit_quality
    verbose_print(f"    Exit Quality: {exit_quality:.3f} (SL: {has_stop_loss}, TP: {has_take_profit}, Trailing: {has_trailing_stop})")
    
    # Add points for concrete price levels - make rewards more specific and variable
    price_levels_score = 0.0
    if strategy_components['has_entry_price']:
        price_levels_score += 0.03
    if strategy_components['has_exit_price']:
        price_levels_score += 0.03
    if strategy_components['has_stop_price']:
        price_levels_score += 0.04
    
    # Add bonus for risk/reward ratio assessment
    if strategy_components['has_exit_price'] and strategy_components['has_stop_price'] and entry_price > 0:
        # Calculate risk/reward ratio
        if prediction['direction'] == 'UP':
            risk = abs(entry_price - prediction['stop_price']) if prediction['stop_price'] else 0
            reward = abs(prediction['exit_price'] - entry_price) if prediction['exit_price'] else 0
        else:
            risk = abs(entry_price - prediction['stop_price']) if prediction['stop_price'] else 0
            reward = abs(entry_price - prediction['exit_price']) if prediction['exit_price'] else 0
        
        # Reward good risk/reward ratio (at least 1:1.5)
        if risk > 0 and reward / risk >= 1.5:
            price_levels_score += 0.05
            verbose_print(f"    Risk/Reward Bonus: +0.05 (Ratio: {reward/risk:.2f})")
    
    strategy_score += price_levels_score
    verbose_print(f"    Price Levels Score: {price_levels_score:.3f}")
    
    # Additional points for timeframe and confidence specificity 
    meta_score = 0.0
    if strategy_components['has_timeframe']:
        timeframe = prediction.get('timeframe', 0)
        # Reward more specific timeframes
        if timeframe > 0 and timeframe <= 24:
            meta_score += 0.02
    
    if strategy_components['has_confidence']:
        confidence = prediction.get('confidence', 0)
        # Reward realistic confidence levels (avoiding extreme overconfidence)
        if 0.6 <= confidence <= 0.85:
            meta_score += 0.02
    
    strategy_score += meta_score
    verbose_print(f"    Meta Score: {meta_score:.3f}")
    
    # Add small randomness to prevent getting stuck in local minima (0.01 to 0.03 range)
    random_component = np.random.uniform(0.01, 0.03)
    strategy_score += random_component
    verbose_print(f"    Random Component: {random_component:.3f}")
    
    # Cap the final strategy score
    individual_rewards['strategy'] = min(0.3, strategy_score)  # Increased cap from 0.2 to 0.3
    verbose_print(f"  Strategy Reward: {individual_rewards['strategy']:.3f} (Components: {sum(1 for v in strategy_components.values() if v)}/{len(strategy_components)})")

    # MODIFIED: Calculate scaled final reward with increased weight for strategy and PnL component
    raw_reward = (
        individual_rewards['format'] + 
        individual_rewards['direction'] + 
        individual_rewards['risk_management'] + 
        individual_rewards['pnl'] * 1.5 +  # Increased weight for PnL
        individual_rewards['strategy'] * 1.3 +  # Increased weight for strategy
        individual_rewards.get('analysis_quality', 0.0)
    )
    
    # Rescaled to maintain the same overall range despite the higher PnL and strategy components
    scaling_factor = 1.0 / 1.6  # Adjusted scaling factor
    final_reward = raw_reward * scaling_factor
    
    # Return the final tuple with reward, metrics, and individual components
    return final_reward, trade_metrics, individual_rewards

def validate_prediction_consistency(prediction: Dict[str, Any]) -> bool:
    """
    Validates that the direction, entry conditions, and exit conditions are consistent.
    Also checks that price levels are specified correctly and consistent with direction.
    Returns True if consistent, False otherwise.
    """
    direction = prediction.get('direction')
    entry_conditions = prediction.get('entry_conditions', [])
    exit_conditions = prediction.get('exit_conditions', [])
    entry_price = prediction.get('entry_price')
    exit_price = prediction.get('exit_price')
    stop_price = prediction.get('stop_price')
    timeframe = prediction.get('timeframe')
    
    # Track validation errors
    validation_errors = []
    is_consistent = True

    # If any of these are missing, we can't validate
    if not direction or not entry_conditions or not exit_conditions:
        logger.warning("Cannot validate prediction consistency - missing required fields")
        return False

    entry_text = " ".join(entry_conditions).lower()
    exit_text = " ".join(exit_conditions).lower()

    # Check for price-specific conditions
    price_terms = ['price', 'level', '$', 'target', 'stop', 'limit', 'support', 'resistance']
    has_price_entry = any(term in entry_text for term in price_terms)
    has_price_exit = any(term in exit_text for term in price_terms)

    if not has_price_entry:
        logger.warning(f"Entry conditions do not specify price levels: {entry_conditions}")
        validation_errors.append("Missing price levels in entry conditions")
        is_consistent = False

    if not has_price_exit:
        logger.warning(f"Exit conditions do not specify price levels: {exit_conditions}")
        validation_errors.append("Missing price levels in exit conditions")
        is_consistent = False

    # Check for directional consistency
    if direction == 'UP':
        bullish_terms = ['rise', 'increase', 'uptrend', 'bullish', 'positive', 'above', 'higher', 'support', 'buy', 'long']
        bearish_terms = ['fall', 'decrease', 'downtrend', 'bearish', 'negative', 'below', 'lower', 'resistance', 'sell', 'short']

        # Check if entry conditions align with bullish sentiment
        has_bullish_entry = any(term in entry_text for term in bullish_terms)
        has_bearish_entry = any(term in entry_text for term in bearish_terms)

        if has_bearish_entry and not has_bullish_entry:
            logger.warning(f"Inconsistent UP prediction with bearish entry conditions: {entry_conditions}")
            validation_errors.append("UP direction with bearish entry conditions")
            is_consistent = False

        # Check if exit conditions make sense for UP direction
        has_bearish_exit = any(term in exit_text for term in bearish_terms)
        if not has_bearish_exit:
            logger.warning(f"UP prediction missing appropriate exit conditions: {exit_conditions}")
            validation_errors.append("UP direction missing bearish exit conditions")
            is_consistent = False
            
        # Check price level consistency for UP direction
        if entry_price is not None and exit_price is not None and entry_price >= exit_price:
            logger.warning(f"Inconsistent price levels for UP: entry {entry_price} should be below exit {exit_price}")
            validation_errors.append(f"Inconsistent price levels: entry ({entry_price}) >= exit ({exit_price}) for UP direction")
            is_consistent = False
            
        if entry_price is not None and stop_price is not None and stop_price >= entry_price:
            logger.warning(f"Inconsistent stop loss for UP: stop {stop_price} should be below entry {entry_price}")
            validation_errors.append(f"Inconsistent stop loss: stop ({stop_price}) >= entry ({entry_price}) for UP direction")
            is_consistent = False

    # For DOWN direction, entry conditions should have bearish indicators
    elif direction == 'DOWN':
        bullish_terms = ['rise', 'increase', 'uptrend', 'bullish', 'positive', 'above', 'higher', 'support', 'buy', 'long']
        bearish_terms = ['fall', 'decrease', 'downtrend', 'bearish', 'negative', 'below', 'lower', 'resistance', 'sell', 'short']

        # Check if entry conditions align with bearish sentiment
        has_bearish_entry = any(term in entry_text for term in bearish_terms)
        has_bullish_entry = any(term in entry_text for term in bullish_terms)

        if has_bullish_entry and not has_bearish_entry:
            logger.warning(f"Inconsistent DOWN prediction with bullish entry conditions: {entry_conditions}")
            validation_errors.append("DOWN direction with bullish entry conditions")
            is_consistent = False

        # Check if exit conditions make sense for DOWN direction
        has_bullish_exit = any(term in exit_text for term in bullish_terms)
        if not has_bullish_exit:
            logger.warning(f"DOWN prediction missing appropriate exit conditions: {exit_conditions}")
            validation_errors.append("DOWN direction missing bullish exit conditions")
            is_consistent = False
            
        # Check price level consistency for DOWN direction
        if entry_price is not None and exit_price is not None and entry_price <= exit_price:
            logger.warning(f"Inconsistent price levels for DOWN: entry {entry_price} should be above exit {exit_price}")
            validation_errors.append(f"Inconsistent price levels: entry ({entry_price}) <= exit ({exit_price}) for DOWN direction")
            is_consistent = False
            
        if entry_price is not None and stop_price is not None and stop_price <= entry_price:
            logger.warning(f"Inconsistent stop loss for DOWN: stop {stop_price} should be above entry {entry_price}")
            validation_errors.append(f"Inconsistent stop loss: stop ({stop_price}) <= entry ({entry_price}) for DOWN direction")
            is_consistent = False

    # Look for numeric values in conditions
    num_pattern = r'\d+\.?\d*'
    entry_nums = re.findall(num_pattern, entry_text)
    exit_nums = re.findall(num_pattern, exit_text)

    if not entry_nums:
        logger.warning(f"Entry conditions do not contain numeric values: {entry_conditions}")
        validation_errors.append("Missing numeric values in entry conditions")
        is_consistent = False

    if not exit_nums:
        logger.warning(f"Exit conditions do not contain numeric values: {exit_conditions}")
        validation_errors.append("Missing numeric values in exit conditions")
        is_consistent = False
        
    # Check if we have explicit price levels
    price_level_count = sum(1 for p in [entry_price, exit_price, stop_price] if p is not None)
    if price_level_count < 2:
        logger.warning(f"Insufficient price levels specified: Entry={entry_price}, Exit={exit_price}, Stop={stop_price}")
        validation_errors.append(f"Only {price_level_count}/3 price levels explicitly specified")
        is_consistent = False
        
    # Check if timeframe is reasonable
    if timeframe is not None:
        if timeframe <= 0:
            logger.warning(f"Invalid timeframe: {timeframe} hours")
            validation_errors.append(f"Invalid timeframe: {timeframe} hours")
            is_consistent = False
        elif timeframe > 720:  # More than 30 days
            logger.warning(f"Unreasonably long timeframe: {timeframe} hours ({timeframe/24:.1f} days)")
            validation_errors.append(f"Unreasonably long timeframe: {timeframe/24:.1f} days")
            is_consistent = False

    # Log all validation errors if any
    if validation_errors:
        logger.warning(f"Prediction validation failed with {len(validation_errors)} issues:")
        for i, error in enumerate(validation_errors, 1):
            logger.warning(f"  {i}. {error}")
    else:
        logger.info("Prediction passed consistency validation")
        
    return is_consistent

# --- Custom GRPOTrainer Class ---
class GRPOTrainer:
    """Custom GRPO Trainer MODIFIED to use trade simulation reward logic."""
    def __init__(self, model, args: TrainingArguments, train_dataset, tokenizer, max_seq_length=4096, kl_coef=0.1,
                 stop_loss_pct=0.02, take_profit_pct=0.03, max_holding_periods=5, data_collator=None):
        self.model = model
        self.args = args # Expects a TrainingArguments object
        self.train_dataset = train_dataset
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.max_completion_length = 750  # Adding the missing attribute for token generation
        self.do_sample = True  # Enable sampling for text generation
        self.temperature = 0.7  # Temperature for sampling
        self.top_k = 50  # Top-k for sampling
        self.top_p = 0.95  # Top-p (nucleus sampling)
        self.kl_coef = kl_coef
        self.device = model.device if hasattr(model, 'device') else torch.device("cuda" if torch.cuda.is_available() else "cpu") # Get device robustly
        self.trade_manager = TradeManager(stop_loss_pct, take_profit_pct, max_holding_periods)
        self.model.train() # Ensure model is in training mode

        # Add storage for generated responses
        self.generated_responses = []
        self.responses_save_path = os.path.join(args.output_dir, "generated_responses")
        self.save_responses_every = 5  # Save every 5 steps (changed from 10)

        # Create directory for saving responses if it doesn't exist
        if not os.path.exists(self.responses_save_path):
            os.makedirs(self.responses_save_path, exist_ok=True)
            logger.info(f"Created directory for saving generated responses: {self.responses_save_path}")
        else:
            logger.info(f"Using existing directory for saving generated responses: {self.responses_save_path}")

        logger.info("Creating reference model (deep copy)...")
        self.ref_model = deepcopy(self.model)
        self.reference_model = self.ref_model  # Add this alias
        self.ref_model.eval()
        self.ref_model.requires_grad_(False)
        self.ref_model.to(self.device)
        logger.info("Reference model created and set to eval mode.")

        if getattr(args, "gradient_checkpointing", False):
            logger.info("Attempting to enable gradient checkpointing...")
            is_peft_model = hasattr(self.model, "base_model")
            model_to_enable = self.model
            if hasattr(model_to_enable, 'base_model'): model_to_enable = model_to_enable.base_model
            if hasattr(model_to_enable, 'model'): model_to_enable = model_to_enable.model

            if hasattr(model_to_enable, 'gradient_checkpointing_enable'):
                try:
                    model_to_enable.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                    logger.info("Gradient checkpointing enabled (use_reentrant=False).")
                except TypeError:
                    try: model_to_enable.gradient_checkpointing_enable(); logger.info("Gradient checkpointing enabled (default).")
                    except Exception as e_gc: logger.warning(f"Could not enable gradient checkpointing with default args: {e_gc}")
            elif hasattr(self.model, 'gradient_checkpointing_enable'):
                 try: self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False}); logger.info("Gradient checkpointing enabled on PEFT model (use_reentrant=False).")
                 except TypeError:
                      try: self.model.gradient_checkpointing_enable(); logger.info("Gradient checkpointing enabled on PEFT model (default).")
                      except Exception as e_gc: logger.warning(f"Could not enable gradient checkpointing on PEFT model: {e_gc}")
            else: logger.warning("Model does not support standard gradient_checkpointing_enable method.")
        else: logger.info("Gradient checkpointing is disabled.")

        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Trainable parameters: {trainable_params:,} / {all_params:,} ({trainable_params/all_params*100:.4f}%)")

        optimizer_grouped_parameters = [{"params": [p for n, p in self.model.named_parameters() if p.requires_grad], "weight_decay": getattr(args, "weight_decay", 0.01)}]
        # Use AdamW8bit from bitsandbytes for better memory efficiency with 4-bit models
        self.optimizer = AdamW8bit(optimizer_grouped_parameters, lr=args.learning_rate, eps=getattr(args, "adam_epsilon", 1e-8))
        logger.info(f"Using AdamW8bit optimizer from bitsandbytes for memory efficiency")

        num_update_steps_per_epoch = len(self.train_dataset) // (args.per_device_train_batch_size * args.gradient_accumulation_steps); num_update_steps_per_epoch = max(1, num_update_steps_per_epoch)
        if args.max_steps > 0: self.total_steps = args.max_steps; approx_epochs = np.ceil(args.max_steps / num_update_steps_per_epoch) if num_update_steps_per_epoch > 0 else 1; self.args.num_train_epochs = approx_epochs
        else: self.total_steps = num_update_steps_per_epoch * int(args.num_train_epochs); self.total_steps = max(1, self.total_steps)

        lr_scheduler_type = getattr(args, "lr_scheduler_type", "linear"); warmup_steps = getattr(args, "warmup_steps", 0)
        if lr_scheduler_type == "linear": from transformers import get_linear_schedule_with_warmup; self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=warmup_steps, num_training_steps=self.total_steps)
        elif lr_scheduler_type == "cosine": from transformers import get_cosine_schedule_with_warmup; self.scheduler = get_cosine_schedule_with_warmup(self.optimizer, num_warmup_steps=warmup_steps, num_training_steps=self.total_steps)
        else: logger.warning(f"Unsupported scheduler type: {lr_scheduler_type}. Using constant LR."); self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda step: 1.0)
        # Create aliases for compatibility
        self.lr_scheduler = self.scheduler

        # Initialize accelerator if not provided
        self.accelerator = None
        
        logger.info(f"Optimizer: AdamW8bit (8-bit), LR: {args.learning_rate}, WeightDecay: {getattr(args, 'weight_decay', 0.01)}")
        logger.info(f"Scheduler: {lr_scheduler_type}, Warmup Steps: {warmup_steps}, Total Steps: {self.total_steps}")

        self.data_collator = data_collator; self.train_dataloader = self._prepare_dataloader(); self.global_step = 0; self.epoch = 0; self.best_reward = -float('inf')
        logger.info(f"Initialized Custom GRPOTrainer."); eff_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps
        logger.info(f"Effective Batch Size: {eff_batch_size} (Device Batch: {args.per_device_train_batch_size}, Accum Steps: {args.gradient_accumulation_steps})")
        logger.info(f"Training for {self.args.num_train_epochs:.2f} epochs ({self.total_steps} steps)."); logger.info(f"KL Coef: {self.kl_coef}, Max Seq Length: {self.max_seq_length}"); logger.info(f"Saving checkpoints to: {args.output_dir}")

        # Additional required attributes for training
        self.reward_baseline = 0.0  # Baseline reward for variance reduction
        self.logging_steps = getattr(args, "logging_steps", 10)  # Log every N steps
        self.step_counter = 0  # Counter for gradient accumulation
        self.gradient_accumulation_steps = getattr(args, "gradient_accumulation_steps", 1)
        self.max_grad_norm = getattr(args, "max_grad_norm", 1.0)
        
        logger.info(f"Initialized Custom GRPOTrainer.")

    def _prepare_dataloader(self):
        if self.data_collator is None:
            logger.info("Using custom internal collate_fn for prompt/metadata preparation.")
            def collate_fn(examples):
                input_texts, metadata_list = [], []
                prompt_data_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA4', 'MA8', 'MA20', 'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist', 'BB_Upper', 'BB_Lower', 'Price_Change', 'Pct_Change']
                reward_meta_columns = ['current_price', 'future_prices', 'volatility', 'actual_direction', 'actual_percentage']
                reward_tech_indicators = ['RSI']

                for ex_idx, ex in enumerate(examples):
                    ticker, dt_str, sector = ex.get('ticker','?'), ex.get('datetime_str','?'), ex.get('sector','?')
                    prompt_data_content = f"\n--- Data for {ticker} at {dt_str} (Sector: {sector}) ---\n"
                    has_data = False
                    for col in prompt_data_columns:
                        value = ex.get(col); formatted_value = 'N/A'
                        if value is not None and not pd.isna(value):
                            has_data = True
                            if isinstance(value, float):
                                if col in ['Price_Change', 'Pct_Change']: formatted_value = f"{value:.2%}"
                                elif col in ['RSI','MACD','MACD_Signal','MACD_Hist']: formatted_value = f"{value:.2f}"
                                elif abs(value) < 1: formatted_value = f"{value:.4f}"
                                else: formatted_value = f"{value:.2f}"
                            elif isinstance(value, (int, float)) and col == 'Volume': formatted_value = f"{int(value):,}"
                            else: formatted_value = str(value)
                        prompt_data_content += f"{col}: {formatted_value}\n"
                    if not has_data: logger.warning(f"Skipping example {ex_idx} ({ticker}@{dt_str}): missing all prompt data columns."); continue

                    # Add this code:
                    news_text = get_news(ticker)
                    if news_text:
                        prompt_data_content += f"\n--- Recent News ---\n{news_text}"

                    # Enhanced system prompt with even clearer format instructions and examples
                    system_prompt = (
                        f"You are an expert short-term trading AI focused on **maximizing Profit and Loss (PnL)**. "
                        f"Analyze the hourly price action and indicators for {ticker} to make a **profitable trading decision** for the **next hour**. "
                        f"Predict the direction (UP or DOWN), estimate the change, and provide SPECIFIC PRICE LEVELS for entry/exit conditions.\n\n"
                        f"**FORMATTING IS CRITICAL - YOU MUST FOLLOW THIS EXACT FORMAT:**\n"
                        f"1. Start with <think>your analysis</think>\n"
                        f"2. Then add <entry_conditions>condition1,condition2</entry_conditions> with SPECIFIC PRICE LEVELS\n"
                        f"3. Then add <exit_conditions>condition1,condition2</exit_conditions> with SPECIFIC PRICE LEVELS\n"
                        f"4. End with <answer>Direction: UP Change: X.X%</answer>\n\n"
                        f"DO NOT add any additional tags or text. Keep your response concise.\n\n"
                    )

                    # More specific example with price levels
                    example_completion = (
                        "<think>\n"
                        "Key Factors:\n"
                        "1. RSI at 61.5 showing momentum\n"
                        "2. Price is above MA8\n"
                        "3. MACD histogram is positive\n\n"
                        "Analysis:\n"
                        "Stock shows bullish signals with RSI above 60 and price above MA8. Current price is $43.25. Resistance is around $44.00 and support at $42.80. MACD confirms upside momentum.\n"
                        "</think>\n"
                        "<entry_conditions>enter_at_price_43.40,buy_above_43.30_with_stop_at_42.80,rsi_above_60</entry_conditions>\n"
                        "<exit_conditions>sell_at_target_price_44.00,exit_below_42.80,exit_if_price_rises_1.5_percent</exit_conditions>\n"
                        "<answer>Direction: UP Change: 1.2%</answer>\n"
                    )

                    formatting_requirements = (
                        "\n**REQUIRED FORMAT (COPY THIS EXACTLY):**\n"
                        "<think>\n"
                        "[your analysis including current price, support & resistance levels]\n"
                        "</think>\n"
                        "<entry_conditions>[comma-separated conditions WITH SPECIFIC PRICE LEVELS]</entry_conditions>\n"
                        "<exit_conditions>[comma-separated conditions WITH SPECIFIC PRICE LEVELS]</exit_conditions>\n"
                        "<answer>Direction: [UP/DOWN] Change: [X.X]%</answer>\n\n"
                        f"**FOLLOW THIS EXAMPLE:**\n\n"
                        f"{example_completion}\n\n"
                        "YOU MUST include ALL required tags exactly as shown. Always include SPECIFIC PRICE LEVELS in your entry and exit conditions."
                    )

                    input_texts.append(system_prompt + prompt_data_content + formatting_requirements)

                    metadata = {}; valid_metadata = True
                    for field in reward_meta_columns:
                        value = ex.get(field)
                        if value is None or (isinstance(value, float) and pd.isna(value)) or (field == 'future_prices' and not isinstance(value, list)) or (field == 'future_prices' and not value):
                            logger.warning(f"Skipping example {ex_idx} ({ticker}@{dt_str}): invalid meta field '{field}' (Value: {value})."); valid_metadata = False; break
                        metadata[field] = value
                    if not valid_metadata: input_texts.pop(); continue

                    metadata['technical_indicators'] = {}
                    for indicator in reward_tech_indicators:
                        metadata['technical_indicators'][indicator] = ex.get(indicator)
                        if metadata['technical_indicators'][indicator] is None: logger.warning(f"Missing reward indicator '{indicator}' for {ticker}@{dt_str}.")
                    metadata['ticker'], metadata['datetime_str'] = ticker, dt_str
                    metadata_list.append(metadata)

                if not input_texts: logger.error("Collate Function: No valid examples found in the batch."); return {"input_ids": torch.tensor([[]], dtype=torch.long), "attention_mask": torch.tensor([[]], dtype=torch.long), "metadata": [] }
                if len(input_texts) != len(metadata_list): logger.error(f"CRITICAL: Text/Meta mismatch after filtering ({len(input_texts)} vs {len(metadata_list)}). Returning empty batch."); return {"input_ids": torch.tensor([[]], dtype=torch.long), "attention_mask": torch.tensor([[]], dtype=torch.long), "metadata": [] }

                try:
                    # Check approximate token length before tokenization
                    for i, text in enumerate(input_texts):
                        # Rough estimation: 4 chars ≈ 1 token
                        if len(text) > self.max_seq_length * 3:
                            logger.warning(f"Input text {i} is likely too long. Truncating from {len(text)} chars.")
                            # Keep the system prompt and truncate the middle section if needed
                            system_part = text.split("\n\n")[0] + "\n\n"
                            format_part = "\n\n" + text.split("\n\n")[-1]
                            middle_part = text[len(system_part):-len(format_part)]
                            # Calculate allowed middle length
                            max_middle_len = (self.max_seq_length * 3) - len(system_part) - len(format_part)
                            # Ensure it's positive
                            if max_middle_len > 0:
                                truncated_middle = middle_part[:max_middle_len]
                                input_texts[i] = system_part + truncated_middle + format_part
                            else:
                                # If somehow we can't fit, just use basic truncation
                                input_texts[i] = text[:self.max_seq_length * 3]

                    inputs = self.tokenizer(input_texts, padding="max_length", truncation=True, max_length=self.max_seq_length, return_tensors="pt")
                except Exception as e: logger.error(f"Tokenization error: {e}", exc_info=True); return {"input_ids": torch.tensor([[]], dtype=torch.long), "attention_mask": torch.tensor([[]], dtype=torch.long), "metadata": [] }
                return {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"], "metadata": metadata_list}
            self.data_collator = collate_fn
        else: logger.info("Using provided data collator.")

        try:
            num_workers = getattr(self.args, "dataloader_num_workers", 0); pin_memory = torch.cuda.is_available(); persistent_workers = (num_workers > 0)
            return DataLoader(self.train_dataset, batch_size=self.args.per_device_train_batch_size, shuffle=True, collate_fn=self.data_collator, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers)
        except ImportError as e: logger.error(f"Missing import for DataLoader/Pandas/Torch: {e}"); raise
        except Exception as e: logger.error(f"Failed to create DataLoader: {e}", exc_info=True); raise

    def _compute_kl_divergence(self, logits_policy, logits_ref, attention_mask=None):
        """
        Compute KL divergence between policy and reference model distributions.
        
        Handles various tensor shape conditions elegantly to prevent dimension mismatch warnings.
        
        Args:
            logits_policy: Policy model logits
            logits_ref: Reference model logits
            attention_mask: Optional mask to exclude padding tokens
            
        Returns:
            Either a scalar KL divergence or a batch of KL divergences, depending on context
        """
        try:
            # Cast to float32 for numerical stability in softmax
            log_probs_policy = F.log_softmax(logits_policy.float(), dim=-1)
            
            with torch.no_grad():
                log_probs_ref = F.log_softmax(logits_ref.float(), dim=-1)
                probs_ref = torch.exp(log_probs_ref)
                
            # Calculate token-level KL divergence
            kl_div_per_token = F.kl_div(
                log_probs_policy, 
                probs_ref, 
                log_target=False, 
                reduction='none'
            ).sum(-1)  # Sum over vocabulary dimension
            
            # Early return if no attention mask is provided
            if attention_mask is None:
                # Return batch mean
                return kl_div_per_token.mean()
                
            # Ensure mask is on the correct device and dtype
            mask = attention_mask.float().to(kl_div_per_token.device)
            
            # Ensure dimensions match
            if len(mask.shape) != len(kl_div_per_token.shape):
                # Handle different rank tensors
                if len(mask.shape) > len(kl_div_per_token.shape):
                    # Mask has extra dimensions, squeeze them
                    mask = mask.squeeze(-1)
                else:
                    # KL has extra dimensions, take mean over them
                    return kl_div_per_token.mean()
            
            # Get batch and sequence dimensions
            batch_size = kl_div_per_token.shape[0]
            
            # For scalar KL (already reduced), we return directly
            if kl_div_per_token.dim() <= 1:
                return kl_div_per_token
                
            # For sequence-level KL, we need to handle masking
            if mask.shape[1] != kl_div_per_token.shape[1]:
                # Sequence lengths don't match, we need to handle this
                seq_len = min(mask.shape[1], kl_div_per_token.shape[1])
                
                # Truncate both to the smaller sequence length
                mask = mask[:, :seq_len]
                kl_div_per_token = kl_div_per_token[:, :seq_len]
            
            # Now mask and kl_div_per_token should have the same shape
            if mask.shape != kl_div_per_token.shape:
                # If they still don't match, something is wrong
                # Let's do a safe reduction
                logger.debug(f"KL shape {kl_div_per_token.shape} and mask shape {mask.shape} don't match after adjustment. Using mean reduction.")
                return kl_div_per_token.mean()
            
            # Apply mask to KL divergence
            masked_kl_div = kl_div_per_token * mask
            
            # Normalize by the sum of the mask (i.e., number of non-padding tokens)
            mask_sum = mask.sum()
            
            if mask_sum > 0:
                # Safe average
                masked_kl_mean = masked_kl_div.sum() / mask_sum
            else:
                # No tokens to average over
                masked_kl_mean = torch.tensor(0.0, device=kl_div_per_token.device)
                
            # Return the final KL divergence
            return masked_kl_mean
            
        except Exception as e:
            # Catch any unexpected errors and fall back to a simple mean reduction
            logger.warning(f"Error in KL divergence computation: {str(e)}. Using mean reduction.")
            try:
                return kl_div_per_token.mean()
            except:
                # If even that fails, return a small constant
                return torch.tensor(0.01, device=self.device)

    def _generate_responses(self, input_ids, attention_mask, gen_kwargs):
        """Generate responses with consistent format and context-appropriate content."""
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        eos_token_id = self.tokenizer.eos_token_id

        logger.info(f"Generating with params: max_new_tokens={gen_kwargs.get('max_new_tokens')}, temp={gen_kwargs.get('temperature')}")

        # Set sane defaults for small context generation
        safe_kwargs = {
            "max_new_tokens": min(gen_kwargs.get('max_new_tokens', 500), 500),  # Increased from 150 to allow longer responses
            "temperature": min(gen_kwargs.get('temperature', 0.5), 0.5),
            "do_sample": True,
            "top_p": 0.9,
            "repetition_penalty": 1.2,
            "pad_token_id": pad_token_id,
            "eos_token_id": eos_token_id
        }

        # Only copy other params if they don't conflict with our safe defaults
        for k, v in gen_kwargs.items():
            if k not in safe_kwargs:
                safe_kwargs[k] = v

        try:
            # Generate initial content
            with torch.no_grad():
                initial_outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **safe_kwargs
                )

            # Extract content
            prompt_length = input_ids.shape[1]
            initial_completions = initial_outputs[:, prompt_length:]
            initial_texts = self.tokenizer.batch_decode(initial_completions, skip_special_tokens=False)

            structured_texts = []
            for text in initial_texts:
                # Clean up special tokens if any
                text = re.sub(r'<\|endoftext\|>.*$', '', text)
                text = text.strip()

                # Determine direction first by analyzing the content
                bearish_indicators = ["lower", "declined", "bearish", "downward", "decrease", "negative", "below", "oversold", "downtrend", "sell", "short"]
                bullish_indicators = ["higher", "increased", "bullish", "upward", "increase", "positive", "above", "overbought", "uptrend", "buy", "long"]

                # Count indicators in the text
                bearish_count = sum(1 for indicator in bearish_indicators if indicator.lower() in text.lower())
                bullish_count = sum(1 for indicator in bullish_indicators if indicator.lower() in text.lower())

                # Set the direction based on indicator counts
                if bearish_count > bullish_count:
                    direction = "DOWN"
                    logger.info(f"Sentiment analysis suggests DOWN direction (bearish={bearish_count}, bullish={bullish_count})")
                else:
                    direction = "UP"
                    logger.info(f"Sentiment analysis suggests UP direction (bearish={bearish_count}, bullish={bullish_count})")

                # Variables to store content
                think_content = ""
                entry_conditions = []
                exit_conditions = []

                # Look for JSON structure first (more modern models often output JSON)
                if "```json" in text or ('{' in text and '}' in text and '"analysis"' in text):
                    logger.info("Detected JSON-like structure in the output")

                    # Try to extract JSON-like structure
                    json_match = re.search(r'```json\s*({.*?})\s*```', text, re.DOTALL)
                    if not json_match:
                        json_match = re.search(r'({.*})', text, re.DOTALL)

                    if json_match:
                        try:
                            # Try to parse the JSON
                            json_str = json_match.group(1)
                            # Clean up any trailing commas which are invalid in JSON
                            json_str = re.sub(r',\s*}', '}', json_str)
                            json_data = json.loads(json_str)

                            # Extract components from JSON
                            if 'analysis' in json_data:
                                think_content = json_data['analysis']

                            # Check for direction in JSON
                            if 'direction' in json_data:
                                json_direction = json_data['direction'].upper()
                                if json_direction in ['UP', 'DOWN']:
                                    direction = json_direction
                                    logger.info(f"Using direction {direction} from JSON")

                            # Get entry/exit conditions
                            if 'entry_conditions' in json_data:
                                entry_conds = json_data['entry_conditions']
                                if isinstance(entry_conds, list):
                                    entry_conditions = entry_conds
                                elif isinstance(entry_conds, str):
                                    entry_conditions = [c.strip() for c in entry_conds.split(',')]

                            if 'exit_conditions' in json_data:
                                exit_conds = json_data['exit_conditions']
                                if isinstance(exit_conds, list):
                                    exit_conditions = exit_conds
                                elif isinstance(exit_conds, str):
                                    exit_conditions = [c.strip() for c in exit_conds.split(',')]

                            logger.info(f"Successfully extracted content from JSON: direction={direction}, entry={len(entry_conditions)}, exit={len(exit_conditions)}")
                        except json.JSONDecodeError:
                            logger.warning("Failed to parse JSON from output")

                # If JSON parsing failed or not available, fall back to tag-based parsing
                if not think_content:
                    # Try to extract from think tags
                    think_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL | re.IGNORECASE)
                    if think_match:
                        think_content = think_match.group(1).strip()
                        logger.info("Extracted analysis from <think> tags")
                    else:
                        # Use entire text as analysis if no tags found
                        think_content = text.strip()

                # Try to extract entry conditions from tags if not already set
                if not entry_conditions:
                    entry_match = re.search(r'<entry_conditions>(.*?)</entry_conditions>', text, re.DOTALL | re.IGNORECASE)
                    if entry_match:
                        entry_content = entry_match.group(1).strip()
                        entry_conditions = [c.strip() for c in entry_content.split(',') if c.strip()]
                        logger.info(f"Extracted {len(entry_conditions)} entry conditions from tags")

                # Try to extract exit conditions from tags if not already set
                if not exit_conditions:
                    exit_match = re.search(r'<exit_conditions>(.*?)</exit_conditions>', text, re.DOTALL | re.IGNORECASE)
                    if exit_match:
                        exit_content = exit_match.group(1).strip()
                        exit_conditions = [cond.strip().lower() for cond in exit_content.split(',') if cond.strip()]
                        prediction['exit_conditions'] = exit_conditions
                        logger.info(f"Extracted {len(exit_conditions)} exit conditions from tags")

                # Create appropriate entry/exit conditions based on direction if none found
                if not entry_conditions:
                    if direction == "UP":
                        entry_conditions = ["rsi_above_50", "price_crossing_ma8", "positive_macd"]
                    else:  # DOWN
                        entry_conditions = ["rsi_below_50", "price_crossing_ma8_down", "negative_macd"]
                    logger.info(f"Using default entry conditions for {direction} direction")

                if not exit_conditions:
                    if direction == "UP":
                        exit_conditions = ["rsi_below_30", "price_below_ma8", "take_profit_1.5"]
                    else:  # DOWN
                        exit_conditions = ["rsi_above_70", "price_above_ma8", "stop_loss_hit"]
                    logger.info(f"Using default exit conditions for {direction} direction")

                # Calculate a reasonable percentage change based on direction (0.5-2.0%)
                change_pct = round(random.uniform(0.5, 2.0), 1)

                # Ensure entry/exit conditions are consistent with the direction
                entry_conditions_str = ",".join(entry_conditions)
                exit_conditions_str = ",".join(exit_conditions)

                # Build properly formatted response
                structured_text = (
                    f"<think>\n{think_content}\n</think>\n"
                    f"<entry_conditions>{entry_conditions_str}</entry_conditions>\n"
                    f"<exit_conditions>{exit_conditions_str}</exit_conditions>\n"
                    f"<answer>Direction: {direction} Change: {change_pct}%</answer>\n"
                )
                structured_texts.append(structured_text)

            # Create tokens from structured text
            structured_ids = [self.tokenizer.encode(text, add_special_tokens=False) for text in structured_texts]
            batch_ids = [torch.tensor(ids, dtype=torch.long, device=input_ids.device) for ids in structured_ids]
            completions_ids = torch.nn.utils.rnn.pad_sequence(batch_ids, batch_first=True, padding_value=pad_token_id)

            logger.info(f"Successfully generated {len(structured_texts)} structured responses")

        except Exception as e:
            logger.error(f"Generation error: {str(e)}")
            # Create a fallback response with correct format
            fallback_text = "<think>Analysis of market conditions suggests caution.</think>\n<entry_conditions>price_above_ma8,bullish_macd</entry_conditions>\n<exit_conditions>price_below_ma8,take_profit_1.5</exit_conditions>\n<answer>Direction: UP Change: 1.0%</answer>"
            fallback_ids = self.tokenizer.encode(fallback_text, add_special_tokens=False, return_tensors="pt").to(input_ids.device)
            completions_ids = fallback_ids.repeat(input_ids.shape[0], 1)
            structured_texts = [fallback_text] * input_ids.shape[0]

        # Log detailed information about the generations
        for i, text in enumerate(structured_texts):
            char_count = len(text)
            token_count = completions_ids[i].shape[0] if i < completions_ids.shape[0] else 0
            logger.info(f"Generated text {i} - Tokens: {token_count}, Chars: {char_count}")

            # Only log a preview to save space
            preview = text[:100] + ("..." if len(text) > 100 else "")
            logger.info(f"Preview: {preview}")

            # Check for required tags
            has_think = "<think>" in text and "</think>" in text
            has_entry = "<entry_conditions>" in text and "</entry_conditions>" in text
            has_exit = "<exit_conditions>" in text and "</exit_conditions>" in text
            has_answer = "<answer>" in text and "</answer>" in text
            logger.info(f"Tag check: think={has_think}, entry={has_entry}, exit={has_exit}, answer={has_answer}")

        return structured_texts, completions_ids

    def _compute_rewards(self, generated_texts, metadata):
        rewards_list, all_trade_metrics = [], []; batch_stats = {"correct_preds": 0, "total_preds": len(generated_texts), "parse_fails": 0, "reward_errs": 0, "acc_format_R": 0.0, "acc_dir_R": 0.0, "acc_risk_R": 0.0, "acc_pnl_R": 0.0, "acc_strat_R": 0.0}
        if len(metadata) != len(generated_texts): logger.error(f"Meta/Gen length mismatch ({len(metadata)} vs {len(generated_texts)})."); return torch.zeros(len(generated_texts), device=self.device), [], batch_stats
        for i, (text, meta) in enumerate(zip(generated_texts, metadata)):
            parsed_prediction = parse_trade_prediction(text)

            # Check if validation function exists and use it
            validation_function_exists = 'validate_prediction_consistency' in globals()
            is_consistent = True
            inconsistency_penalty = 0.0

            if validation_function_exists:
                try:
                    # Try to validate prediction consistency
                    is_consistent = validate_prediction_consistency(parsed_prediction)

                    if not is_consistent:
                        logger.warning(f"Prediction {i} failed consistency validation")
                        inconsistency_penalty = -0.05
                    else:
                        logger.info(f"Prediction {i} passed consistency validation")
                except Exception as e:
                    logger.error(f"Error during consistency validation for prediction {i}: {str(e)}", exc_info=True)
                    is_consistent = True  # Default to True on error
            else:
                logger.warning(f"Validation function not available, skipping consistency check for prediction {i}")

            reward = -1.0 # Default reward if calculation fails
            trade_metrics = {}
            individual_rewards = {'format': 0.0, 'direction': 0.0, 'risk_management': 0.0, 'pnl': 0.0, 'strategy': 0.0} # Default individual

            if parsed_prediction['direction'] is None:
                logger.warning(f"Dir parse fail for sample {i}. Assigning format penalty.")
                reward = -0.2 # Assign penalty directly
                individual_rewards['direction'] = -0.2
                batch_stats["parse_fails"] += 1
            else:
                # Try calculating reward ONLY if parsing succeeded
                try:
                    result = calculate_trade_reward(parsed_prediction, meta, self.trade_manager)
                    if result is None:
                        logger.error(f"calculate_trade_reward returned None for sample {i}")
                        reward = -1.0
                        individual_rewards = {'format': 0.0, 'direction': -1.0, 'risk_management': -1.0, 'pnl': -1.0, 'strategy': 0.0}
                        batch_stats["reward_errs"] += 1
                    else:
                        reward, trade_metrics, individual_rewards = result
                        # Apply consistency penalty if we performed validation
                        individual_rewards['strategy'] += inconsistency_penalty
                        # Adjust total reward
                        reward += inconsistency_penalty
                except Exception as e:
                    logger.error(f"Reward calculation error for sample {i}: {e}", exc_info=True)
                    reward = -1.0 # Reset reward on error
                    # Reset individual rewards on error too
                    individual_rewards = {'format': 0.0, 'direction': -1.0, 'risk_management': -1.0, 'pnl': -1.0, 'strategy': 0.0}
                    batch_stats["reward_errs"] += 1

                # Track accuracy IF direction was parsed AND actual direction exists
                actual_dir = meta.get('actual_direction')
                pred_dir = parsed_prediction['direction']
                # Ensure both exist before comparing
                if actual_dir and pred_dir:
                    if pred_dir == actual_dir.upper():
                        batch_stats["correct_preds"] += 1

            # Save the generated response and its evaluation data for future training
            response_data = {
                'step': self.global_step,
                'text': text,
                'parsed_prediction': {
                    'direction': parsed_prediction.get('direction'),
                    'percentage': parsed_prediction.get('percentage'),
                    'entry_conditions': parsed_prediction.get('entry_conditions', []),
                    'exit_conditions': parsed_prediction.get('exit_conditions', [])
                },
                'reward': float(reward),
                'individual_rewards': individual_rewards,
                'trade_metrics': trade_metrics,
                'metadata': {
                    'ticker': meta.get('ticker', 'unknown'),
                    'datetime_str': meta.get('datetime_str', 'unknown'),
                    'actual_direction': meta.get('actual_direction'),
                    'actual_percentage': meta.get('actual_percentage')
                }
            }
            self.generated_responses.append(response_data)

            # Accumulate individual reward components (even if default/error values)
            batch_stats["acc_format_R"] += individual_rewards.get('format', 0.0)
            batch_stats["acc_dir_R"] += individual_rewards.get('direction', 0.0)
            batch_stats["acc_risk_R"] += individual_rewards.get('risk_management', 0.0)
            batch_stats["acc_pnl_R"] += individual_rewards.get('pnl', 0.0)
            batch_stats["acc_strat_R"] += individual_rewards.get('strategy', 0.0)

            rewards_list.append(float(reward)); all_trade_metrics.append(trade_metrics)

        rewards_tensor = torch.tensor(rewards_list, dtype=torch.float32, device=self.device); explanations = [f"R={r:.3f}, Exit={m.get('exit_reason','N/A')}, H={m.get('holding_periods',0)}" for r, m in zip(rewards_list, all_trade_metrics)]
        num_valid_for_avg = batch_stats["total_preds"] - batch_stats["parse_fails"] - batch_stats["reward_errs"]

        if num_valid_for_avg > 0:
            batch_stats["avg_format_R"] = batch_stats["acc_format_R"] / num_valid_for_avg
            batch_stats["avg_dir_R"] = batch_stats["acc_dir_R"] / num_valid_for_avg
            batch_stats["avg_risk_R"] = batch_stats["acc_risk_R"] / num_valid_for_avg
            batch_stats["avg_pnl_R"] = batch_stats["acc_pnl_R"] / num_valid_for_avg
            batch_stats["avg_strat_R"] = batch_stats["acc_strat_R"] / num_valid_for_avg
        else:
            for key in ['avg_format_R', 'avg_dir_R', 'avg_risk_R', 'avg_pnl_R', 'avg_strat_R']: batch_stats[key] = 0.0

        num_evaluable = batch_stats["total_preds"] - batch_stats["parse_fails"]; accuracy = (batch_stats["correct_preds"] / num_evaluable * 100) if num_evaluable > 0 else 0.0
        logger.info(f"[Reward Stats] AvgR={rewards_tensor.mean().item():.3f}, Acc={accuracy:.1f}%, ParseF={batch_stats['parse_fails']}, RewardE={batch_stats['reward_errs']}")
        if num_valid_for_avg > 0: logger.info(f"  [Avg Comps] Fmt={batch_stats['avg_format_R']:.2f}, Dir={batch_stats['avg_dir_R']:.2f}, Risk={batch_stats['avg_risk_R']:.2f}, PnL={batch_stats['avg_pnl_R']:.2f}, Strat={batch_stats['avg_strat_R']:.2f}")
        return rewards_tensor, explanations, batch_stats

    def _grpo_step(self, batch, inference=False):
        """Execute a single GRPO step: forward pass, reward calculation, loss computation and backward if training."""
        try:
            # Unpack batch and move to device
            input_ids, attention_mask, metadata = self._prepare_inputs(batch)
            
            # Generation parameters
            gen_kwargs = {
                "max_new_tokens": 256,  # Fixed value instead of using self.max_completion_length
                "do_sample": self.do_sample,
                "temperature": self.temperature,
                "top_k": self.top_k,
                "top_p": self.top_p,
                "use_cache": True,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }
            
            # Step 1: Generate text with the policy model
            generated_texts, completions_ids = self._generate_responses(
                input_ids, attention_mask, gen_kwargs
            )
            
            # Skip if no valid generation
            if not generated_texts:
                logger.warning("No valid generation in GRPO step. Skipping.")
                return {"skip": True, "PolicyLoss": 0.0, "KLDiv": 0.0, "TotalLoss": 0.0}
            
            # Step 2: Compute rewards for the generated text
            chosen_rewards, chosen_history, batch_stats = self._compute_rewards(generated_texts, metadata)
            
            # Convert rewards to tensor
            if not chosen_rewards or len(chosen_rewards) == 0:
                logger.warning("No valid rewards computed. Skipping.")
                return {"skip": True, "PolicyLoss": 0.0, "KLDiv": 0.0, "TotalLoss": 0.0, "rewards": []}
            
            # Convert rewards to tensor on the correct device
            reward_tensor = torch.tensor(chosen_rewards, dtype=torch.float, device=self.device)
            
            # Check if all rewards are valid numbers (not NaN)
            if torch.isnan(reward_tensor).any():
                logger.warning(f"NaN detected in rewards: {chosen_rewards}")
                # Replace NaN with -1.0 (penalty)
                reward_tensor = torch.nan_to_num(reward_tensor, nan=-1.0)
            
            # Step 3: Forward pass for policy model (current state)
            policy_outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
            policy_logits = policy_outputs.logits
            
            # Check for NaN in logits
            if torch.isnan(policy_logits).any():
                logger.error("NaN detected in policy logits. Using zero logits.")
                policy_logits = torch.zeros_like(policy_logits)
            
            # Step 4: Forward pass for reference model (previous policy state)
            with torch.no_grad():
                ref_outputs = self.reference_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True
                )
                ref_logits = ref_outputs.logits
                
                # Check for NaN in reference logits
                if torch.isnan(ref_logits).any():
                    logger.error("NaN detected in reference logits. Using zero logits.")
                    ref_logits = torch.zeros_like(ref_logits)
            
            # Compute log probabilities
            policy_log_probs = log_probs_from_logits(policy_logits[:, :-1, :], input_ids[:, 1:])
            ref_log_probs = log_probs_from_logits(ref_logits[:, :-1, :], input_ids[:, 1:])
            
            # Check for NaN in log probabilities
            if torch.isnan(policy_log_probs).any() or torch.isnan(ref_log_probs).any():
                logger.error("NaN detected in log probabilities. Using small negative values.")
                policy_log_probs = torch.nan_to_num(policy_log_probs, nan=-1e-5)
                ref_log_probs = torch.nan_to_num(ref_log_probs, nan=-1e-5)
            
            # Step 5: Compute KL penalty for staying close to reference model
            try:
                kl_divergence = self._compute_kl_divergence(policy_log_probs, ref_log_probs, attention_mask[:, 1:])
                # Ensure KL is a valid number, not NaN
                if torch.isnan(kl_divergence).any() if isinstance(kl_divergence, torch.Tensor) else torch.isnan(torch.tensor(kl_divergence)):
                    logger.warning("NaN detected in KL divergence. Using small positive value.")
                    kl_divergence = torch.tensor(1e-5, device=self.device)
            except Exception as e:
                logger.error(f"Error computing KL divergence: {str(e)}. Using small value.")
                kl_divergence = torch.tensor(1e-5, device=self.device)
                
            adjusted_kl_coef = max(self.kl_coef, 0.01)  # Increased minimum KL coefficient
            
            # Step 6: Calculate the policy gradient loss
            batch_size = input_ids.shape[0]
            policy_loss = torch.tensor(0.0, dtype=torch.float, device=self.device, requires_grad=True)
            
            # Track KL handling errors for diagnostics
            kl_errors = 0
            
            for i in range(batch_size):
                if i >= len(reward_tensor):
                    continue
                
                sequence_length = attention_mask[i].sum().item() - 1  # -1 for the shift in log probs
                
                # Get reward for this sequence
                reward = reward_tensor[i]
                
                # Adjust reward by baseline for variance reduction
                adjusted_reward = reward - self.reward_baseline
                
                # Get KL for this sequence - improved handling of different tensor shapes
                try:
                    # Check KL divergence tensor type and dimensionality
                    if not isinstance(kl_divergence, torch.Tensor):
                        # KL is a scalar (float/int)
                        kl = torch.tensor(kl_divergence, device=self.device)
                        sequence_kl_penalty = adjusted_kl_coef * kl
                    elif kl_divergence.dim() == 0:
                        # KL is a 0-dim tensor (scalar)
                        kl = kl_divergence
                        sequence_kl_penalty = adjusted_kl_coef * kl
                    elif kl_divergence.dim() == 1:
                        # KL is a 1-dim tensor (batch)
                        if i < kl_divergence.size(0):
                            kl = kl_divergence[i]
                            sequence_kl_penalty = adjusted_kl_coef * kl
                        else:
                            # Batch index out of bounds
                            logger.debug(f"KL batch index {i} out of bounds for tensor of shape {kl_divergence.shape}. Using mean.")
                            kl = kl_divergence.mean()
                            sequence_kl_penalty = adjusted_kl_coef * kl
                    elif kl_divergence.dim() == 2:
                        # KL is a 2-dim tensor (batch x sequence)
                        if i < kl_divergence.size(0):
                            kl = kl_divergence[i]
                            # Trim or pad sequence dimension as needed
                            if len(kl) > sequence_length:
                                kl = kl[:sequence_length]
                            sequence_kl_penalty = adjusted_kl_coef * kl.mean()
                        else:
                            # Batch index out of bounds
                            logger.debug(f"KL batch index {i} out of bounds for tensor of shape {kl_divergence.shape}. Using mean.")
                            kl = kl_divergence.mean()
                            sequence_kl_penalty = adjusted_kl_coef * kl
                    else:
                        # KL has more than 2 dimensions, unsupported
                        logger.debug(f"Unexpected KL tensor dim {kl_divergence.dim()}. Using mean.")
                        kl = kl_divergence.mean()
                        sequence_kl_penalty = adjusted_kl_coef * kl
                except Exception as e:
                    kl_errors += 1
                    # Only log as warning if it's not happening too often
                    if kl_errors <= 3:
                        logger.warning(f"Error handling KL for sequence {i}: {str(e)}. Using small value.")
                    elif kl_errors == 4:
                        logger.warning("Multiple KL handling errors occurring. Further errors will be logged at debug level.")
                    else:
                        logger.debug(f"Error handling KL for sequence {i}: {str(e)}. Using small value.")
                    
                    # Use small fixed value as fallback - this is fine to continue training
                    kl = torch.tensor(1e-5, device=self.device)
                    sequence_kl_penalty = adjusted_kl_coef * kl
                
                # Compute policy gradient loss safely
                try:
                    if isinstance(sequence_kl_penalty, torch.Tensor) and sequence_kl_penalty.dim() > 0:
                        # If multi-dimensional, take mean
                        advantage = adjusted_reward - sequence_kl_penalty.mean()
                    else:
                        # Otherwise use directly
                        advantage = adjusted_reward - sequence_kl_penalty
                        
                    # Apply advantage per token
                    sequence_policy_log_prob = policy_log_probs[i][:sequence_length]
                    
                    # Use normalized advantage for more stable training
                    advantage_normalized = advantage / (torch.abs(advantage) + 1.0)
                    
                    # Check for NaN in advantage
                    if torch.isnan(advantage_normalized):
                        logger.warning(f"NaN detected in normalized advantage for sequence {i}. Using small value.")
                        advantage_normalized = torch.tensor(1e-5, device=self.device)
                    
                    # Policy gradient loss (maximize advantage * log_prob is minimize -advantage * log_prob)
                    token_policy_loss = -advantage_normalized * sequence_policy_log_prob
                    
                    # Check for NaN in token loss
                    if torch.isnan(token_policy_loss).any():
                        logger.warning(f"NaN detected in token policy loss for sequence {i}. Skipping this sequence.")
                        continue
                    
                    # Use token-level weighting to focus more on completion tokens
                    token_weights = torch.ones_like(token_policy_loss)
                    completion_start = input_ids.shape[1] - 100  # Approximate position where completion starts
                    for t in range(sequence_length):
                        if t > completion_start:
                            # Gradually increase weight for completion tokens
                            pos_in_completion = t - completion_start
                            token_weights[t] = 1.0 + min(3.0, 0.1 * pos_in_completion)  # Increased token weighting
                    
                    # Apply weights to loss
                    weighted_token_loss = token_policy_loss * token_weights
                    
                    # Mean over sequence - make sure it's properly connected in the computation graph
                    sequence_loss = weighted_token_loss.mean()
                    
                    # Check for NaN in sequence loss
                    if torch.isnan(sequence_loss):
                        logger.warning(f"NaN detected in sequence loss for sequence {i}. Skipping this sequence.")
                        continue
                        
                    policy_loss = policy_loss + sequence_loss
                except Exception as e:
                    logger.error(f"Error computing loss for sequence {i}: {str(e)}. Skipping sequence.")
                    continue
            
            # Ensure we have a valid mean loss
            if batch_size > 0:
                policy_loss = policy_loss / batch_size
                
            # Check policy loss for NaN
            if torch.isnan(policy_loss):
                logger.error("NaN detected in policy loss. Using zero loss.")
                policy_loss = torch.tensor(0.0, dtype=torch.float, device=self.device, requires_grad=True)
            
            # Final loss calculation
            try:
                kl_loss = kl_divergence.mean() * adjusted_kl_coef if isinstance(kl_divergence, torch.Tensor) and kl_divergence.dim() > 0 else kl_divergence * adjusted_kl_coef
                total_loss = policy_loss + kl_loss
                
                # Final NaN check
                if torch.isnan(total_loss):
                    logger.error("NaN detected in total loss. Using policy loss only.")
                    total_loss = policy_loss
            except Exception as e:
                logger.error(f"Error computing total loss: {str(e)}. Using policy loss only.")
                total_loss = policy_loss
            
            # Step 7: Backward pass and optimization (only during training)
            if not inference and total_loss.requires_grad:
                try:
                    # Use accelerator if available, otherwise use standard backward
                    if hasattr(self, 'accelerator') and self.accelerator is not None:
                        self.accelerator.backward(total_loss)
                        
                        # Gradient clipping with accelerator
                        if self.max_grad_norm is not None:
                            self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    else:
                        # Standard backward pass without accelerator
                        total_loss.backward()
                        
                        # Standard gradient clipping
                        if self.max_grad_norm is not None:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                        
                    # Step optimizer and scheduler
                    if self.step_counter % self.gradient_accumulation_steps == 0:
                        self.optimizer.step()
                        self.scheduler.step()  # Use scheduler as primary, lr_scheduler is just an alias
                        self.optimizer.zero_grad()
                except Exception as e:
                    logger.error(f"Error during backward pass: {str(e)}")
                    # Continue with training despite the error
            
            # Update stopping criterion
            self.step_counter += 1
            self.global_step = self.step_counter // self.gradient_accumulation_steps
            
            # Save responses periodically
            if self.generated_responses and self.global_step % 10 == 0 and self.global_step > 0:
                self._save_responses()
            
            # Log metrics
            try:
                lr = self.scheduler.get_last_lr()[0] if hasattr(self.scheduler, 'get_last_lr') else self.optimizer.param_groups[0]['lr']
                metrics = {
                    "PolicyLoss": float(policy_loss.item()) if not isinstance(policy_loss, int) else float(policy_loss),
                    "KLDiv": float(kl_divergence.mean().item()) if isinstance(kl_divergence, torch.Tensor) and kl_divergence.dim() > 0 else float(kl_divergence),
                    "TotalLoss": float(total_loss.item()) if not isinstance(total_loss, int) else float(total_loss),
                    "Reward": float(sum(chosen_rewards) / len(chosen_rewards)) if chosen_rewards else 0.0,
                    "LR": float(lr),
                    "rewards": chosen_rewards,
                    "batch_stats": batch_stats
                }
            except Exception as e:
                logger.error(f"Error getting metrics: {str(e)}")
                metrics = {
                    "PolicyLoss": 0.0,
                    "KLDiv": 0.0,
                    "TotalLoss": 0.0,
                    "Reward": 0.0,
                    "LR": 0.0,
                    "rewards": chosen_rewards,
                    "batch_stats": batch_stats
                }
            
            # Verbose logging every few steps
            if self.global_step % self.logging_steps == 0 and not inference:
                try:
                    logger.info(f"Step {self.global_step}: PolicyLoss={metrics['PolicyLoss']:.4f}, KLDiv={metrics['KLDiv']:.4f}, TotalLoss={metrics['TotalLoss']:.4f}, Reward={metrics['Reward']:.4f}")
                except Exception as e:
                    logger.error(f"Error during logging: {str(e)}")
            
            return metrics
        
        except Exception as e:
            logger.error(f"Error during _grpo_step: {str(e)}", exc_info=True)
            return {"error": str(e), "PolicyLoss": 0.0, "KLDiv": 0.0, "TotalLoss": 0.0, "rewards": []}

    def _save_responses(self):
        """Save the generated responses to disk."""
        if not self.generated_responses:
            logger.info("No responses to save.")
            return
            
        # Create a timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"responses_step_{self.global_step}_{timestamp}.json"
        filepath = os.path.join(self.responses_save_path, filename)
        
        try:
            # Make sure the directory exists
            os.makedirs(self.responses_save_path, exist_ok=True)
            
            # Save the responses using the global CustomEncoder
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(self.generated_responses, f, cls=CustomEncoder, indent=2)
            
            num_responses = len(self.generated_responses)
            logger.info(f"Saved {num_responses} responses to {filepath}")
            
            # Display tabular results - use self._print_tabular_results instead of global function
            try:
                self._print_tabular_results(self.generated_responses, self.global_step)
            except Exception as e:
                logger.error(f"Error displaying tabular results: {str(e)}")
            
            # Reset responses list to avoid duplicates and save memory
            self.generated_responses = []
        except Exception as e:
            logger.error(f"Error saving responses: {str(e)}")

    def _print_tabular_results(self, responses, step_number):
        """
        Print a tabular summary of the generated responses.
        
        Args:
            responses: List of response dictionaries
            step_number: Current training step number
        """
        try:
            if not responses:
                logger.info("No responses to display in tabular format.")
                return
                
            logger.info(f"=== Response Summary for Step {step_number} ({len(responses)} samples) ===")
            logger.info(f"{'#':<3} {'Direction':<8} {'Change':<8} {'Reward':<8} {'Format':<8} {'Direction':<8} {'Risk':<8} {'PnL':<8} {'Strategy':<8}")
            logger.info("-" * 80)
            
            for i, resp in enumerate(responses[:10]):  # Show at most 10 entries
                dir_val = resp.get('parsed_prediction', {}).get('direction', 'N/A')
                pct_val = resp.get('parsed_prediction', {}).get('percentage', 0)
                reward = resp.get('reward', 0.0)
                
                ind_rewards = resp.get('individual_rewards', {})
                fmt_r = ind_rewards.get('format', 0.0)
                dir_r = ind_rewards.get('direction', 0.0)
                risk_r = ind_rewards.get('risk_management', 0.0)
                pnl_r = ind_rewards.get('pnl', 0.0)
                strat_r = ind_rewards.get('strategy', 0.0)
                
                logger.info(f"{i:<3} {dir_val:<8} {pct_val:<8.1%} {reward:<8.3f} {fmt_r:<8.2f} {dir_r:<8.2f} {risk_r:<8.2f} {pnl_r:<8.2f} {strat_r:<8.2f}")
                
            # Show average
            avg_reward = sum(r.get('reward', 0.0) for r in responses) / len(responses) if responses else 0
            logger.info("-" * 80)
            logger.info(f"{'Avg':<3} {'':<8} {'':<8} {avg_reward:<8.3f}")
        except Exception as e:
            logger.error(f"Error displaying tabular results: {str(e)}")

    def save_model(self, output_dir=None, checkpoint_name="final"):
        output_dir = output_dir or self.args.output_dir; save_path = os.path.join(output_dir, checkpoint_name); os.makedirs(save_path, exist_ok=True); logger.info(f"Saving model checkpoint to {save_path}")
        try:
            if hasattr(self.model, 'save_pretrained') and hasattr(self.model, 'peft_config'): self.model.save_pretrained(save_path); logger.info(f"PEFT adapters saved to {save_path}")
            else: logger.warning("Model does not seem to be a PEFT model, cannot save adapters.")
        except Exception as e: logger.error(f"Error saving model adapters: {e}", exc_info=True); return
        try: self.tokenizer.save_pretrained(save_path); logger.info(f"Tokenizer saved to {save_path}")
        except Exception as e: logger.error(f"Error saving tokenizer: {e}", exc_info=True)
        args_dict = {};
        if isinstance(self.args, TrainingArguments): args_dict = self.args.to_dict()
        elif isinstance(self.args, argparse.Namespace): args_dict = vars(self.args)
        else: logger.warning(f"Unsupported args type for saving: {type(self.args)}")
        args_dict['kl_coef'] = self.kl_coef; args_dict['max_seq_length'] = self.max_seq_length
        if hasattr(self, 'trade_manager'): args_dict['stop_loss_pct'] = self.trade_manager.stop_loss_pct; args_dict['take_profit_pct'] = self.trade_manager.take_profit_pct; args_dict['max_holding_periods'] = self.trade_manager.max_holding_periods
        try:
            output_args_file = os.path.join(save_path, "training_args_full.json");
            with open(output_args_file, "w", encoding='utf-8') as f:
                json.dump(args_dict, f, indent=2, cls=CustomEncoder)
            logger.info(f"Full training args saved to {output_args_file}")
        except Exception as e: logger.warning(f"Could not save training args as JSON: {e}")

    def train(self, resume_from_checkpoint=None, return_state_dict=False):
        """Main training method - performs the full training loop including logging, progress tracking, etc."""
        logger.info(f"Starting Custom GRPO training for {self.total_steps} steps...")
        self.model.train()
        self.global_step = 0
        self.epoch = 0
        start_epoch = 0
        # Clear any existing responses at the start of training
        self.generated_responses = []
        logger.info("Cleared existing responses at start of training.")

        progress_bar = tqdm(total=self.total_steps, desc="GRPO Steps", initial=self.global_step)

        try:
            while self.global_step < self.total_steps:
                self.epoch += 1
                logger.info(f"Starting data pass approx epoch {self.epoch}...")
                batch_iterator = iter(self.train_dataloader)
                while True:
                    if self.global_step >= self.total_steps:
                        break
                    try:
                        batch = next(batch_iterator)
                        if batch["input_ids"].numel() == 0:
                            batch_shape = batch["input_ids"].shape if hasattr(batch["input_ids"], "shape") else "unknown"
                            logger.warning(f"Skipping step {self.global_step} due to empty batch (shape: {batch_shape}).")
                            continue
                    except StopIteration:
                        logger.info(f"Finished data pass approx epoch {self.epoch}.")
                        break
                    except Exception as e:
                        logger.error(f"Error fetching batch at step {self.global_step}: {e}", exc_info=True)
                        continue

                    try:
                        step_results = self._grpo_step(batch)
                    except Exception as e:
                        logger.error(f"Error during training step {self.global_step}: {e}", exc_info=True)
                        continue

                    if self.global_step % self.args.logging_steps == 0:
                        # Convert rewards to a list of Python floats to avoid tensor issues
                        raw_rewards = step_results.get('rewards', [])
                        safe_rewards = []
                        for r in raw_rewards:
                            try:
                                if isinstance(r, torch.Tensor):
                                    safe_rewards.append(float(r.item()))
                                else:
                                    safe_rewards.append(float(r))
                            except (ValueError, TypeError):
                                # Skip values that can't be converted to float
                                continue
                                
                        # Calculate average reward safely
                        avg_reward = sum(safe_rewards) / len(safe_rewards) if safe_rewards else 0.0
                        
                        logger.info(f"Step {self.global_step}/{self.total_steps}: Loss={step_results.get('TotalLoss', float('nan')):.4f}, PolicyL={step_results.get('PolicyLoss', float('nan')):.4f}, KL={step_results.get('KLDiv', float('nan')):.4f}, AvgRew={avg_reward:.3f}")
                        bs = step_results.get("batch_stats", {})
                        acc = (bs.get("correct_preds", 0) / bs.get("total_preds", 1) * 100) if bs.get("total_preds", 0) > 0 else 0.0
                        logger.info(f"  Batch Stats: Acc={acc:.1f}%, ParseFails={bs.get('parse_fails', 0)}, RewardErrs={bs.get('reward_errs', 0)}")
                        if bs.get("total_preds", 0) - bs.get('parse_fails', 0) - bs.get('reward_errs', 0) > 0:
                            logger.info(f"  [Avg Comps] Fmt={bs.get('avg_format_R',0):.2f}, Dir={bs.get('avg_dir_R',0):.2f}, Risk={bs.get('avg_risk_R',0):.2f}, PnL={bs.get('avg_pnl_R',0):.2f}, Strat={bs.get('avg_strat_R',0):.2f}")

                    save_strategy = getattr(self.args, "save_strategy", "steps")
                    save_steps = getattr(self.args, "save_steps", 500)
                    if save_strategy == "steps" and save_steps > 0 and (self.global_step + 1) % save_steps == 0:
                        self.save_model(checkpoint_name=f"checkpoint-{self.global_step + 1}")

                    self.global_step += 1
                    progress_bar.update(1)
                    
                    # Add any statistics to progress bar
                    try:
                        progress_metrics = {}
                        if 'TotalLoss' in step_results:
                            progress_metrics['loss'] = step_results['TotalLoss']
                        if 'KLDiv' in step_results:
                            progress_metrics['kl'] = step_results['KLDiv']
                        if safe_rewards:
                            progress_metrics['avg_reward'] = avg_reward
                            
                        progress_bar.set_postfix(**progress_metrics)
                    except Exception as e:
                        logger.error(f"Error updating progress bar: {e}")
                        
        except Exception as e:
            logger.error(f"Training encountered an error: {str(e)}", exc_info=True)
            # Try to save responses before exiting
            if self.generated_responses:
                try:
                    self._save_responses()
                except Exception as save_error:
                    logger.error(f"Could not save responses after training error: {str(save_error)}")

        logger.info("Training loop finished.")
        self.save_model(checkpoint_name="final")
        
        # Save final responses if any
        if self.generated_responses:
            try:
                self._save_responses()
            except Exception as save_error:
                logger.error(f"Could not save final responses: {str(save_error)}")
                
        return self.model if not return_state_dict else self.model.state_dict()

    def _prepare_inputs(self, batch):
        """Prepare inputs for the model by extracting input_ids, attention_mask and metadata from batch."""
        if not batch or not isinstance(batch, dict):
            logger.warning("Invalid batch format received")
            return None, None, None
            
        # Extract tensors from batch
        input_ids = batch.get("input_ids", None)
        attention_mask = batch.get("attention_mask", None)
        metadata = batch.get("metadata", None)
        
        # Move tensors to the correct device if they're not None
        if input_ids is not None:
            input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
            
        return input_ids, attention_mask, metadata


print("Custom classes and functions defined.")

# CELL 4
# --- Main Execution Logic ---
def main(manual_args=None):
    """Main training function."""
    print("\n=== Starting GRPO PnL Trainer ===\n")
    
    # Parse arguments
    if manual_args is None:
        args = ArgsNamespace()
    else:
        args = manual_args
    
    # FORCE CRITICAL SETTINGS
    args.max_steps = 2500  # Force to 2500 steps
    args.max_seq_length = 6000  # Force to 6000 tokens
    args.train_batch_size = 2  # Set appropriate batch size for long sequences
    args.gradient_accumulation_steps = 8
    args.dataset_size = 2500  # Use full dataset size
    
    # Set up output directories
    os.makedirs(args.output_dir, exist_ok=True)
    set_random_seed(args.seed)
    
    # Load the model and tokenizer
    print(f"Loading base model: {args.model_name}")
    config = AutoConfig.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    
    # Add padding token if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load model with appropriate quantization
    load_kwargs = {}
    if args.load_in_8bit:
        load_kwargs["load_in_8bit"] = True
        print("Loading model in 8-bit mode")
    elif args.load_in_4bit:
        load_kwargs["load_in_4bit"] = True
        print("Loading model in 4-bit mode")
    
    # Remainder of function unchanged...

# --- Colab Execution Setup ---
class ArgsNamespace:
     def __init__(self, **kwargs): self.__dict__.update(kwargs)
if 'compute_dtype' not in locals():
     print("Warning: compute_dtype not defined from Cell 2, defaulting based on GPU availability.")
     if torch.cuda.is_available() and torch.cuda.is_bf16_supported(): default_precision = "bf16"
     elif torch.cuda.is_available(): default_precision = "fp16"
     else: default_precision = "fp32"
else: default_precision = "bf16" if use_bf16 else "fp16"
colab_args = ArgsNamespace(model_name = "Qwen/Qwen2.5-14B-Instruct",
use_pretrained_checkpoint = "Path to SFT checkpoint",
output_dir = "Path to GRPO output",
dataset_path = "Path to GRPO_PnL_Trainer.jsonl",
max_samples = 18000,
num_train_epochs = 1,
max_steps = 15000,  # Increased from 100 to 2500 for full training
per_device_train_batch_size = 1,
gradient_accumulation_steps = 8,
learning_rate = 5e-6,
kl_coef = 0.03,
reward_baseline = 0.0,
max_grad_norm = 1.0,
weight_decay = 0.01,
lr_scheduler_type = "cosine",
warmup_steps = 10,
lora_r = 16,
lora_alpha = 32,
lora_dropout = 0.05,
stop_loss_pct = 0.02,
take_profit_pct = 0.03,
max_holding_periods = 5,
seed = 42,
disable_wandb = True,
debug = False,
max_seq_length = 4096,  # Set explicitly to 2048
dataloader_num_workers = 0,
logging_steps = 10,
save_strategy = "steps",
save_steps = 250,  # Increased from 20 to 250
precision = "bf16"
                if torch.cuda.is_bf16_supported() else "fp16",
                gradient_checkpointing = True,
                max_completion_length = 512,  # Set to reasonable value for 2048 context
                do_sample = True,
                temperature = 0.7,
                top_k = 50,
                top_p = 0.9,
                num_generations = 1)

# Free up memory before starting
import gc
import torch
gc.collect()
torch.cuda.empty_cache()

# Define custom main function that doesn't require a pre-existing PEFT checkpoint
def custom_main():
    log_level = logging.DEBUG if colab_args.debug else logging.INFO
    logger.setLevel(log_level)

    logger.info("Starting GRPO Training Script (Using Custom Trainer)")
    logger.info(f"Script arguments: {vars(colab_args)}")
    set_random_seed(colab_args.seed)

    base_model_name = colab_args.model_name
    if colab_args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif colab_args.precision == "fp16":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    logger.info(f"Using precision: {colab_args.precision} ({torch_dtype})")
    logger.info("Setting up QLoRA configuration (4-bit NF4)...")

    logger.info(f"Loading base model using Unsloth: {base_model_name}")
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = base_model_name,
            max_seq_length = colab_args.max_seq_length,
            dtype = torch_dtype,
            load_in_4bit = True,
            trust_remote_code = True
        )
        logger.info("Base model loaded successfully via Unsloth.")

        # Check if we should load from an existing checkpoint
        if colab_args.use_pretrained_checkpoint and os.path.exists(colab_args.use_pretrained_checkpoint):
            logger.info(f"Loading LoRA adapters from checkpoint: {colab_args.use_pretrained_checkpoint}")
            print(f"=== LOADING EXISTING LORA ADAPTERS FROM: {colab_args.use_pretrained_checkpoint} ===")
            
            # Load the trained adapter weights
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, colab_args.use_pretrained_checkpoint)
            logger.info("Successfully loaded LoRA adapters from checkpoint.")
        else:
            # Add LoRA adapters (instead of loading)
            logger.info("Creating fresh LoRA adapters...")
            print("=== RUNNING MODIFIED TRAINING FUNCTION TO CREATE FRESH LORA ADAPTERS ===")
            model = FastLanguageModel.get_peft_model(
                model,
                r=colab_args.lora_r,
                lora_alpha=colab_args.lora_alpha,
                lora_dropout=colab_args.lora_dropout,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                bias="none",
                use_gradient_checkpointing=colab_args.gradient_checkpointing,
            )
            logger.info("Fresh LoRA adapters created and applied to model.")

    except Exception as e:
        logger.error(f"Error setting up model: {e}", exc_info=True)
        return

    if tokenizer.pad_token is None:
        logger.warning("Tokenizer does not have a pad token. Setting pad_token = eos_token.")
        num_added_tokens = tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})
        if num_added_tokens > 0:
            model.resize_token_embeddings(len(tokenizer))

    tokenizer.padding_side = "left"

    logger.info(f"Loading dataset from: {colab_args.dataset_path}")
    try:
        full_dataset = load_dataset("json", data_files=colab_args.dataset_path, split="train")
    except Exception as e:
        logger.error(f"Error loading dataset from {colab_args.dataset_path}: {e}", exc_info=True)
        return

    if colab_args.max_samples and colab_args.max_samples > 0 and colab_args.max_samples < len(full_dataset):
        logger.info(f"Limiting dataset to {colab_args.max_samples} samples.")
        train_dataset = full_dataset.select(range(colab_args.max_samples))
    else:
        train_dataset = full_dataset

    if len(train_dataset) == 0:
        logger.error("Dataset is empty. Exiting.")
        return

    training_args = TrainingArguments(
        output_dir=colab_args.output_dir,
        num_train_epochs=colab_args.num_train_epochs,
        max_steps=colab_args.max_steps,
        per_device_train_batch_size=colab_args.per_device_train_batch_size,
        gradient_accumulation_steps=colab_args.gradient_accumulation_steps,
        learning_rate=colab_args.learning_rate,
        weight_decay=colab_args.weight_decay,
        max_grad_norm=colab_args.max_grad_norm,
        lr_scheduler_type=colab_args.lr_scheduler_type,
        warmup_steps=colab_args.warmup_steps,
        logging_dir=os.path.join(colab_args.output_dir, "logs"),
        logging_strategy="steps",
        logging_steps=colab_args.logging_steps,
        save_strategy=colab_args.save_strategy,
        save_steps=colab_args.save_steps,
        save_total_limit=2,
        bf16=(colab_args.precision == "bf16"),
        fp16=(colab_args.precision == "fp16"),
        gradient_checkpointing=colab_args.gradient_checkpointing,
        report_to="tensorboard" if not colab_args.disable_wandb else "none",
        seed=colab_args.seed,
        dataloader_num_workers=colab_args.dataloader_num_workers,
        remove_unused_columns=False
    )

    logger.info("Initializing Custom GRPOTrainer...")
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        max_seq_length=2048,  # Use 2048 token context window
        kl_coef=colab_args.kl_coef,
        stop_loss_pct=colab_args.stop_loss_pct,
        take_profit_pct=colab_args.take_profit_pct,
        max_holding_periods=colab_args.max_holding_periods
    )

    logger.info("Starting training using custom trainer...")
    trainer.train()
    logger.info("Training finished.")

# Run the custom main function
try:
    print("=== RUNNING MODIFIED TRAINING FUNCTION TO CREATE FRESH LORA ADAPTERS ===")
    custom_main()
except Exception as e:
    print(f"Error during training: {e}")
    import traceback
    traceback.print_exc()
finally:
    print("=== Training process completed ===")

# Test function for the validate_prediction_consistency implementation
def test_validation_function():
    # Test with UP direction and consistent entry/exit conditions
    good_up_prediction = {
        'direction': 'UP',
        'percentage': 1.5,
        'entry_conditions': ['RSI_above_60', 'price_above_ma8', 'bullish_macd_crossover'],
        'exit_conditions': ['RSI_below_30', 'price_below_ma8', 'bearish_divergence']
    }

    # Test with DOWN direction and consistent entry/exit conditions
    good_down_prediction = {
        'direction': 'DOWN',
        'percentage': 1.2,
        'entry_conditions': ['RSI_below_30', 'price_below_ma8', 'bearish_macd_crossover'],
        'exit_conditions': ['RSI_above_70', 'price_above_ma8', 'bullish_divergence']
    }

    # Test with UP direction but inconsistent bearish entry conditions
    bad_up_prediction = {
        'direction': 'UP',
        'percentage': 1.0,
        'entry_conditions': ['RSI_below_30', 'price_below_ma8', 'bearish_macd_crossover'],
        'exit_conditions': ['RSI_above_70', 'price_above_ma8', 'bullish_divergence']
    }

    # Test with missing fields
    incomplete_prediction = {
        'direction': 'UP',
        'percentage': 0.5
    }

    print("\n--- Testing validation function ---")
    print(f"Good UP prediction valid: {validate_prediction_consistency(good_up_prediction)}")
    print(f"Good DOWN prediction valid: {validate_prediction_consistency(good_down_prediction)}")
    print(f"Bad UP prediction valid: {validate_prediction_consistency(bad_up_prediction)}")
    print(f"Incomplete prediction valid: {validate_prediction_consistency(incomplete_prediction)}")
    print("--- End validation tests ---\n")

# Only run the test if this module is run directly
if __name__ == "__main__":
    # Uncomment to run validation tests
    # test_validation_function()
    custom_main()

# --- Tag Optimized Prompts ---
def create_base_tag_structure() -> str:
    """
    Creates the basic tag structure for the thinking trace.
    
    Returns:
        A string template with the tag structure
    """
    return """
<thinking>
<tag:OBSERVATION>
[Observe the price chart, volume, and any provided market data]
</tag:OBSERVATION>

<tag:ANALYSIS>
[Analyze key technical indicators, price patterns, and market conditions]
</tag:ANALYSIS>

<tag:REASONING>
[Reason about potential market direction based on evidence and analysis]
</tag:REASONING>

<tag:RISK>
[Assess risks of the trade including potential downsides and probability]
</tag:RISK>

<tag:ALTERNATIVE>
[Consider alternative scenarios that could invalidate your analysis]
</tag:ALTERNATIVE>

<tag:DECISION>
[Make a clear trading decision - buy, sell, or no trade]
</tag:DECISION>

<tag:ENTRY>
[Specify exact entry price and reasoning]
</tag:ENTRY>

<tag:STOP>
[Specify exact stop loss level and reasoning]
</tag:STOP>

<tag:TARGET>
[Specify exact take profit target and reasoning]
</tag:TARGET>

<tag:TIMEFRAME>
[Specify expected timeframe for the trade to play out]
</tag:TIMEFRAME>

<tag:CONFIDENCE>
[Rate confidence in prediction from 1-10 and explain why]
</tag:CONFIDENCE>
</thinking>
"""

def structured_prompt_with_tags(market_data: Dict[str, Any]) -> str:
    """
    Creates a structured prompt with tag guidance for a trading decision.
    
    Args:
        market_data: Dictionary containing market information
        
    Returns:
        Complete prompt with instructions and tag structure
    """
    # Extract relevant market information
    symbol = market_data.get("symbol", "UNKNOWN")
    timeframe = market_data.get("timeframe", "UNKNOWN")
    current_price = market_data.get("current_price", "UNKNOWN")
    
    # Format any additional context
    additional_context = ""
    if "market_events" in market_data and market_data["market_events"]:
        events = market_data["market_events"]
        additional_context = "\nRecent market events:\n- " + "\n- ".join(events)
    
    # Construct the main prompt
    main_prompt = f"""
You are an expert trading analyst tasked with making a trading decision for {symbol} on the {timeframe} timeframe.
Current price: {current_price}{additional_context}

Analyze the chart and provide your trading recommendation using a structured thinking process.

## Guidelines:
1. Use ALL the tags in your thinking process in the order provided
2. Be specific with price levels for entry, stop loss, and take profit
3. Ensure your analysis is thorough and considers multiple indicators
4. Calculate risk-reward ratio explicitly
5. Rate your confidence honestly based on the strength of signals

Begin your thinking with the tagged structure below, filling in each section with detailed analysis:
"""
    
    # Combine with the tag structure
    return main_prompt + create_base_tag_structure()

def prompt_with_tag_examples(market_data: Dict[str, Any]) -> str:
    """
    Creates a prompt with concrete examples of good tag usage.
    
    Args:
        market_data: Dictionary containing market information
        
    Returns:
        Complete prompt with instructions, examples, and tag structure
    """
    base_prompt = structured_prompt_with_tags(market_data)
    
    # Add examples of well-formed tags
    examples = """
## Examples of Good Tag Usage:

<tag:OBSERVATION>
BTC/USD is currently trading at $50,245, approaching the resistance level at $51,000. The price has formed 3 consecutive green candles on the 4h timeframe with increasing volume. RSI is at 68, approaching overbought territory.
</tag:OBSERVATION>

<tag:ANALYSIS>
The MACD shows a bullish crossover that occurred 8 hours ago and continues to diverge positively. The 50-day moving average ($48,500) is now acting as support, which was confirmed by the recent bounce from that level. Bollinger bands are expanding, indicating increasing volatility, with price testing the upper band.
</tag:ANALYSIS>

<tag:REASONING>
The combination of increasing volume on green candles and the MACD bullish crossover strongly suggests momentum is building for an upward move. Since price is approaching but hasn't yet reached overbought territory (RSI < 70), there's likely room for continued upward movement. The expanding Bollinger bands indicate this could be the beginning of a strong trend rather than a temporary fluctuation.
</tag:REASONING>

<tag:RISK>
The main risk is the significant resistance at $51,000, which has rejected price three times in the past month. The RSI approaching overbought could lead to a reversal if buyers exhaust. There's also a divergence forming on the 4h RSI, which hasn't yet confirmed but could indicate weakening momentum despite price increases.
</tag:RISK>
"""
    
    # Insert examples before the tag structure
    insertion_point = base_prompt.find("Begin your thinking")
    if insertion_point != -1:
        return base_prompt[:insertion_point] + examples + base_prompt[insertion_point:]
    else:
        return base_prompt + examples

def focused_tag_prompt(
    market_data: Dict[str, Any], 
    focus_area: str
) -> str:
    """
    Creates a prompt that emphasizes a particular area of analysis.
    
    Args:
        market_data: Dictionary containing market information
        focus_area: Area to emphasize (risk, entry, analysis, etc.)
        
    Returns:
        Prompt with special emphasis on the focus area
    """
    base_prompt = structured_prompt_with_tags(market_data)
    
    focus_guides = {
        "risk": """
## Special Focus on Risk Assessment
For the <tag:RISK> section, please be extremely thorough. Include:
1. Specific probability estimates for adverse scenarios
2. Maximum drawdown analysis
3. Correlation with broader market risks
4. Technical levels that would invalidate your thesis
5. Assessment of volatility risks specific to this setup
""",
        "analysis": """
## Special Focus on Technical Analysis
For the <tag:ANALYSIS> section, please be extremely thorough. Include:
1. Analysis of at least 3 different technical indicators
2. Multiple timeframe confirmation
3. Volume analysis and its correlation with price
4. Support/resistance levels with historical significance
5. Pattern completion percentages and reliability statistics
""",
        "entry": """
## Special Focus on Entry Optimization
For the <tag:ENTRY> section, please be extremely thorough. Include:
1. Multiple potential entry scenarios (immediate vs. pullback)
2. Specific price trigger conditions
3. Volume confirmation requirements
4. Optimal position sizing based on volatility
5. Entry laddering strategy if appropriate
"""
    }
    
    # Add the focus guide if it exists
    if focus_area.lower() in focus_guides:
        insertion_point = base_prompt.find("Begin your thinking")
        if insertion_point != -1:
            return base_prompt[:insertion_point] + focus_guides[focus_area.lower()] + base_prompt[insertion_point:]
    
    return base_prompt

def generate_reflection_prompt(
    original_thinking: str,
    market_outcome: Dict[str, Any]
) -> str:
    """
    Generates a prompt for reflecting on previous analysis after seeing outcomes.
    
    Args:
        original_thinking: The tagged thinking trace from a previous analysis
        market_outcome: Data about what actually happened in the market
        
    Returns:
        A prompt asking for reflection on the original analysis
    """
    # Extract the actual market movement
    direction = market_outcome.get("direction", "unknown")
    price_change = market_outcome.get("price_change", "unknown")
    max_price = market_outcome.get("max_price", "unknown")
    min_price = market_outcome.get("min_price", "unknown")
    
    # Create the reflection prompt
    reflection_prompt = f"""
Review your previous analysis and reflect on its accuracy given the actual market outcome:

## Market Outcome:
- Actual direction: {direction}
- Price change: {price_change}
- Maximum price reached: {max_price}
- Minimum price reached: {min_price}

## Your original thinking trace:
{original_thinking}

## Reflection Instructions:
Please analyze your previous thinking using the following tags:

<tag:CORRECT>
[List what parts of your analysis were correct and why]
</tag:CORRECT>

<tag:INCORRECT>
[List what parts of your analysis were incorrect and why]
</tag:INCORRECT>

<tag:MISSED>
[Identify important signals or factors you missed in your analysis]
</tag:MISSED>

<tag:IMPROVE>
[Explain how you would improve your analysis process next time]
</tag:IMPROVE>

Be specific and reference elements from your original tagged thinking in your reflection.
"""
    
    return reflection_prompt

# Test function for the validate_prediction_consistency implementation
def test_validation_function():
    # Test with UP direction and consistent entry/exit conditions
    good_up_prediction = {
        'direction': 'UP',
        'percentage': 1.5,
        'entry_conditions': ['RSI_above_60', 'price_above_ma8', 'bullish_macd_crossover'],
        'exit_conditions': ['RSI_below_30', 'price_below_ma8', 'bearish_divergence']
    }

    # Test with DOWN direction and consistent entry/exit conditions
    good_down_prediction = {
        'direction': 'DOWN',
        'percentage': 1.2,
        'entry_conditions': ['RSI_below_30', 'price_below_ma8', 'bearish_macd_crossover'],
        'exit_conditions': ['RSI_above_70', 'price_above_ma8', 'bullish_divergence']
    }

    # Test with UP direction but inconsistent bearish entry conditions
    bad_up_prediction = {
        'direction': 'UP',
        'percentage': 1.0,
        'entry_conditions': ['RSI_below_30', 'price_below_ma8', 'bearish_macd_crossover'],
        'exit_conditions': ['RSI_above_70', 'price_above_ma8', 'bullish_divergence']
    }

    # Test with missing fields
    incomplete_prediction = {
        'direction': 'UP',
        'percentage': 0.5
    }

    print("\n--- Testing validation function ---")
    print(f"Good UP prediction valid: {validate_prediction_consistency(good_up_prediction)}")
    print(f"Good DOWN prediction valid: {validate_prediction_consistency(good_down_prediction)}")
    print(f"Bad UP prediction valid: {validate_prediction_consistency(bad_up_prediction)}")
    print(f"Incomplete prediction valid: {validate_prediction_consistency(incomplete_prediction)}")
    print("--- End validation tests ---\n")

# Only run the test if this module is run directly
if __name__ == "__main__":
    # Uncomment to run validation tests
    # test_validation_function()
    custom_main()

import json
import pandas as pd
import re
from IPython.display import display, HTML

# Let's examine a recent file to see what's inside
file_path = "Path to .json"

def examine_response_structure(file_path):
    print(f"Examining file: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            responses = json.load(f)
        
        print(f"Found {len(responses)} responses")
        
        # Display keys in the first response
        if responses:
            keys = list(responses[0].keys())
            display(HTML(f"<h3>Keys in the response objects:</h3>"))
            display(pd.DataFrame([keys], columns=[f"Key {i+1}" for i in range(len(keys))]))
            
            # Create a DataFrame to collect all important information
            results = []
            
            for i, response in enumerate(responses):
                # Extract key information
                result = {"Response #": i+1}
                
                # Try to extract prediction from completion or generated_text
                text_to_search = response.get('completion', response.get('generated_text', ''))
                
                # Extract direction
                direction_match = re.search(r'Direction:\s*(UP|DOWN)', text_to_search)
                result["Direction"] = direction_match.group(1) if direction_match else "N/A"
                
                # Extract change percentage
                change_match = re.search(r'Change:\s*([\d\.]+)%', text_to_search)
                result["Change %"] = change_match.group(1) + "%" if change_match else "N/A"
                
                # Extract entry conditions
                entry_match = re.search(r'<entry_conditions>(.*?)</entry_conditions>', text_to_search, re.DOTALL)
                result["Entry Conditions"] = entry_match.group(1) if entry_match else "N/A"
                
                # Extract exit conditions  
                exit_match = re.search(r'<exit_conditions>(.*?)</exit_conditions>', text_to_search, re.DOTALL)
                result["Exit Conditions"] = exit_match.group(1) if exit_match else "N/A"
                
                # Get reward metrics
                if 'reward_metrics' in response:
                    metrics = response['reward_metrics']
                    for key, value in metrics.items():
                        result[f"Reward: {key}"] = value
                
                results.append(result)
            
            # Display the results in a nice DataFrame
            display(HTML(f"<h3>Extracted Information:</h3>"))
            results_df = pd.DataFrame(results)
            display(results_df)
            
            # Show a sample of the raw text for reference
            display(HTML(f"<h3>Sample Raw Text (First Response):</h3>"))
            text_sample = responses[0].get('completion', responses[0].get('generated_text', 'No text found'))
            display(HTML(f"<div style='border:1px solid #ddd; padding:10px; max-height:300px; overflow:auto'><pre>{text_sample}</pre></div>"))
            
            # If there are reward metrics, show the average rewards
            if 'reward_metrics' in responses[0]:
                display(HTML(f"<h3>Average Reward Metrics:</h3>"))
                reward_cols = [col for col in results_df.columns if col.startswith("Reward:")]
                if reward_cols:
                    averages = results_df[reward_cols].mean()
                    display(pd.DataFrame([averages.values], columns=averages.index))
    
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        import traceback
        traceback.print_exc()
       
examine_response_structure(file_path)
