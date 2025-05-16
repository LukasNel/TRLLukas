from datetime import datetime


async def main():
    import os
    import json
    import pandas as pd
    from datasets import Dataset, load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig
    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, TaskType
    import torch.nn.functional as F
    import re
    import json
    import pandas as pd
    from datasets import Dataset, concatenate_datasets
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    from deeptools.samplers.vllm.sampler import VLLMSampler

    def external_custom_reward(sample, completion, **kwargs):
        """
        sample: dict, the input sample (with 'has_thinking' flag)
        completion: str, the model's output
        kwargs: any other fields passed from the dataset
        Returns: float, the reward
        """
        reward = 0.0
        # Example: reward for correct answer (you can expand this)
        if "expected_answer" in sample:
            if sample["expected_answer"].strip().lower() in completion.strip().lower():
                reward += 1.0

        # Reward logic for thinking traces
        if sample.get("has_thinking"):
            # Reward for step-by-step reasoning (very simple example)
            if "let's think" in completion.lower() or "step by step" in completion.lower():
                reward += 0.5
            # Reward for length/structure (customize as needed)
            if len(completion.split()) > 30:
                reward += 0.2
        else:
            # Reward for brevity
            if len(completion.split()) < 20:
                reward += 0.2
            # Penalize if it rambles
            if "let's think" in completion.lower():
                reward -= 0.2

        return reward


    # 1. Load and merge datasets
    with_thinking_ds = load_dataset("DatOneDue/with_thinking")
    without_thinking_ds = load_dataset("DatOneDue/without_thinking")

    def add_thinkbool(example, thinkbool):
        return {
            "is_thinking":thinkbool,
        }

    with_thinking_ds = with_thinking_ds.map(add_thinkbool, fn_kwargs={"thinkbool": True})['train']
    without_thinking_ds = without_thinking_ds.map(add_thinkbool, fn_kwargs={"thinkbool": False})['train']

    # 2. Convert to HuggingFace Dataset
    combined_dataset = concatenate_datasets([with_thinking_ds, without_thinking_ds])

    # 3. Load model and tokenizer (Qwen3-14B as example)
    model_name = "Qwen/Qwen3-14B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="cuda:1",
        quantization_config=bnb_config,
        trust_remote_code=True
    )
    vllm_sampler = VLLMSampler(model_id=model_name, max_output=512)
    vllm_sampler.client.update_model_params(model)
    # Create a fixed generation config to avoid any issues with default settings
    fixed_gen_config = GenerationConfig(
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_length=2048,  # Set an appropriate max length
        do_sample=False,  # Force greedy decoding by default
        num_beams=1,      # Simple greedy search
        temperature=1.0,  # No temperature adjustment
        use_cache=True,   # Enable KV-caching
    )
    model.generation_config = fixed_gen_config

    # Add QLoRA for 4-bit quantized model training
    print("Preparing model for QLoRA training...")
    model = prepare_model_for_kbit_training(model)

    # Define LoRA config
    lora_config = LoraConfig(
        r=16,  # Rank of the update matrices
        lora_alpha=32,  # Alpha parameter for LoRA scaling
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    # Apply LoRA adapters to the model
    model = get_peft_model(model, lora_config)
    # model = torch.compile(model)
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Percentage trainable: {100 * sum(p.numel() for p in model.parameters() if p.requires_grad) / sum(p.numel() for p in model.parameters()):.4f}%")

    # Enable gradient checkpointing explicitly with use_reentrant=False
    print("Enabling gradient checkpointing...")
    model.gradient_checkpointing_enable()

    # Manually set use_reentrant flag where possible
    try:
        # For future compatibility, try to set the flag via attribute for newer PyTorch versions
        for module in model.modules():
            if hasattr(module, "gradient_checkpointing") and hasattr(module, "gradient_checkpointing_kwargs"):
                module.gradient_checkpointing_kwargs = {"use_reentrant": False}
    except Exception as e:
        print(f"Note: Could not set use_reentrant flag: {e}")
        print("Continuing with default gradient checkpointing settings")

    # Verify that LoRA parameters require gradients (debugging)
    lora_params_require_grad = all(p.requires_grad for n, p in model.named_parameters() if 'lora' in n)
    print(f"All LoRA parameters require gradients: {lora_params_require_grad}")

    # Print list of LoRA parameter names
    print("LoRA parameters:")
    for name, param in model.named_parameters():
        if 'lora' in name:
            print(f"  {name} - requires_grad: {param.requires_grad}, shape: {param.shape}")

    model.train()

    # Check and set pad_token_id if not already set
    if tokenizer.pad_token is None:
        print("Tokenizer does not have a pad token. Setting pad_token = eos_token.")
        if tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
            # Ensure the pad_token_id is also set if pad_token was None
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            print("CRITICAL WARNING: EOS token not found, and no pad token. Padding will likely fail.")
            # As a last resort, you might need to add a new pad token,
            # but this requires careful handling of model embeddings.
            # For now, we'll assume eos_token exists for Qwen.

    # Ensure model's config is aware of the pad_token_id
    if tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
        print(f"Model config pad_token_id set to: {model.config.pad_token_id}")
    else:
        print("Warning: pad_token_id is still None. Generation might be problematic.")

    # Resize token embeddings if tokenizer vocabulary is larger than model's embedding layer
    current_embedding_size = model.get_input_embeddings().weight.size(0)
    if current_embedding_size < len(tokenizer):
        print(f"Resizing token embeddings from {current_embedding_size} to {len(tokenizer)}")
        model.resize_token_embeddings(len(tokenizer))
        # After resizing, it's good to re-tie weights if applicable, though for LoRA this is less of a concern.
        # model.tie_weights() # Uncomment if you were doing full fine-tuning and had tied embeddings

    # Set padding_side for tokenizer (important for decoder-only models)
    # For generation, "left" padding is typical so the actual sequence starts on the right.
    tokenizer.padding_side = "left"
    print(f"Tokenizer padding_side set to '{tokenizer.padding_side}'")

    # 4. DataLoader
    batch_size = 1
    dataloader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: x)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    def custom_reward(sample, completion):
        """
        Calculate a reward for the model's completion based on the input sample.

        Args:
            sample: dict, the input sample with metadata
            completion: str, the generated text from the model

        Returns:
            float: the calculated reward
        """
        print(f"DEBUG - custom_reward: Received completion of length {len(completion)}")

        # Initialize the reward
        reward = 0.0
        reward_components = {}

        # 1. Base reward just for generating something non-empty (minimum reward)
        if len(completion.strip()) > 10:  # At least 10 characters
            reward += 0.1
            reward_components['non_empty'] = 0.1

        # 2. Check for stock/technical analysis keywords
        analysis_keywords = [
            "bullish", "bearish", "uptrend", "downtrend", "support", "resistance",
            "overbought", "oversold", "momentum", "volume", "breakout", "reversal",
            "moving average", "rsi", "macd", "divergence", "consolidation",
            "volatility", "trend", "indicator",
            # Options-specific terminology
            "call option", "put option", "strike price", "expiration", "premium",
            "implied volatility", "delta", "gamma", "theta", "vega", "option chain",
            "in-the-money", "out-of-the-money", "at-the-money", "covered call",
            "protective put", "straddle", "strangle", "spread", "iron condor", "butterfly"
        ]

        analysis_count = 0
        for keyword in analysis_keywords:
            if keyword in completion.lower():
                analysis_count += 1

        if analysis_count >= 1:
            reward += 0.2  # Reward for using at least 1 technical term
            reward_components['technical_terms'] = 0.2

        if analysis_count >= 3:
            reward += 0.3  # Additional reward for using 3+ technical terms
            reward_components['multiple_technical_terms'] = 0.3

        # 3. Reward for making a clear prediction (UP or DOWN)
        prediction_found = False
        prediction_terms = {
            "UP": ["UP", "BULLISH", "UPWARD", "RISING", "BUY", "LONG", "STRONG_UP"],
            "DOWN": ["DOWN", "BEARISH", "DOWNWARD", "FALLING", "SELL", "SHORT", "STRONG_DOWN"]
        }
        # Add separate checks for problematic terms that need more context
        ambiguous_terms = {
            "UP": ["POSITIVE"],
            "DOWN": ["NEGATIVE"]
        }

        # Add phrase patterns that indicate a prediction
        prediction_phrases = {
            "UP": [
                r"PRICE\s+(?:WILL|SHOULD|COULD|MAY|MIGHT)?\s+(?:GO|MOVE|TREND|HEAD|CONTINUE)?\s+UP",
                r"STOCK\s+(?:WILL|SHOULD|COULD|MAY|MIGHT)?\s+(?:GO|MOVE|TREND|HEAD|CONTINUE)?\s+UP",
                r"LOOKS?\s+BULLISH",
                r"EXPECT(?:ING)?\s+(?:A|AN)?\s+(?:UP|UPWARD|BULLISH)",
                r"LIKELY\s+TO\s+(?:GO|MOVE|TREND|HEAD|CONTINUE)?\s+UP",
                r"SHOULD\s+(?:GO|MOVE|TREND|HEAD|CONTINUE)?\s+(?:UP|HIGHER)",
                r"INCREASE\s+IN\s+PRICE",
                r"HIGHER\s+PRICES?",
                r"BULLISH\s+MOMENTUM",
                r"BUYING\s+OPPORTUNITY",
                r"UPWARD\s+TREND",
                r"BOUNCE\s+(?:BACK|UP)"
            ],
            "DOWN": [
                r"PRICE\s+(?:WILL|SHOULD|COULD|MAY|MIGHT)?\s+(?:GO|MOVE|TREND|HEAD|CONTINUE)?\s+DOWN",
                r"STOCK\s+(?:WILL|SHOULD|COULD|MAY|MIGHT)?\s+(?:GO|MOVE|TREND|HEAD|CONTINUE)?\s+DOWN",
                r"LOOKS?\s+BEARISH",
                r"EXPECT(?:ING)?\s+(?:A|AN)?\s+(?:DOWN|DOWNWARD|BEARISH)",
                r"LIKELY\s+TO\s+(?:GO|MOVE|TREND|HEAD|CONTINUE)?\s+DOWN",
                r"SHOULD\s+(?:GO|MOVE|TREND|HEAD|CONTINUE)?\s+(?:DOWN|LOWER)",
                r"DECREASE\s+IN\s+PRICE",
                r"LOWER\s+PRICES?",
                r"BEARISH\s+MOMENTUM",
                r"SELLING\s+OPPORTUNITY",
                r"DOWNWARD\s+TREND",
                r"DROP\s+(?:FURTHER)?"
            ]
        }

        strong_prediction_pattern = r'\*\*Prediction:\s*(UP|DOWN|BULLISH|BEARISH|STRONG_UP|STRONG_DOWN)\*\*'
        prediction_pattern = r'Prediction:\s*(UP|DOWN|BULLISH|BEARISH|STRONG_UP|STRONG_DOWN)'
        clear_prediction_pattern = r'(UP|DOWN|BULLISH|BEARISH|STRONG_UP|STRONG_DOWN)\s*([^\w]|$)'

        # Check for prediction-indicating phrases first
        if not prediction_found:
            for direction, phrases in prediction_phrases.items():
                for phrase in phrases:
                    if re.search(phrase, completion.upper()):
                        match = re.search(phrase, completion.upper())
                        start = max(0, match.start() - 25)
                        end = min(len(completion), match.end() + 25)
                        context = completion[start:end]
                        prediction_found = True
                        detected_direction = direction
                        print(f"DEBUG - Found prediction phrase indicating {direction}: '{phrase}'")
                        print(f"DEBUG - Phrase context: '{context}'")
                        reward += 0.5
                        reward_components['phrase_prediction'] = 0.5
                        break
                if prediction_found:
                    break

        # Check for explicit prediction formats
        if not prediction_found:
            strong_match = re.search(strong_prediction_pattern, completion, re.IGNORECASE)
            if strong_match:
                prediction_found = True
                detected_direction = strong_match.group(1).upper()
                print(f"DEBUG - Found strong prediction format: {strong_match.group(0)}")
                reward += 0.5
                reward_components['clear_prediction'] = 0.5

        # Check for regular prediction format if strong format not found
        if not prediction_found:
            pred_match = re.search(prediction_pattern, completion, re.IGNORECASE)
            if pred_match:
                prediction_found = True
                detected_direction = pred_match.group(1).upper()
                print(f"DEBUG - Found prediction format: {pred_match.group(0)}")
                reward += 0.5
                reward_components['clear_prediction'] = 0.5

        # Check for occurrences of prediction terms if no prediction format was found
        if not prediction_found:
            for direction, terms in prediction_terms.items():
                for term in terms:
                    # Use a more flexible pattern that looks for the term in various formats
                    term_pattern = r'\b' + term + r'\b'
                    if re.search(term_pattern, completion.upper()):
                        match = re.search(term_pattern, completion.upper())
                        start = max(0, match.start() - 40)
                        end = min(len(completion), match.end() + 40)
                        context = completion[start:end]

                        # Check if this is in a context of describing an indicator rather than a prediction
                        indicator_explanation_patterns = [
                            r"INDICATING\s+(?:A|AN)?\s*" + term,
                            r"SUGGESTS?\s+(?:A|AN)?\s*" + term,
                            r"MACD\s+(?:IS|SHOWS|INDICATES|SUGGESTS)\s+(?:A|AN)?\s*" + term,
                            r"RSI\s+(?:IS|SHOWS|INDICATES|SUGGESTS)\s+(?:A|AN)?\s*" + term,
                            r"INDICATOR\s+(?:IS|SHOWS|SUGGESTS)\s+(?:A|AN)?\s*" + term,
                            r"TECHNICALS?\s+(?:ARE|IS|SHOW|SUGGEST)\s+(?:A|AN)?\s*" + term
                        ]

                        is_indicator_explanation = any(re.search(pattern, context.upper()) for pattern in indicator_explanation_patterns)

                        if not is_indicator_explanation:
                            prediction_found = True
                            detected_direction = direction
                            print(f"DEBUG - Found prediction term: {term}")
                            print(f"DEBUG - Context: '{context}'")
                            reward += 0.5
                            reward_components['clear_prediction'] = 0.5
                            break
                if prediction_found:
                    break

        # Special check for ambiguous terms (POSITIVE/NEGATIVE) that need context
        if not prediction_found:
            for direction, terms in ambiguous_terms.items():
                for term in terms:
                    term_pattern = r'\b' + term + r'\b'
                    matches = list(re.finditer(term_pattern, completion.upper()))
                    for match in matches:
                        # Get context around the term
                        start = max(0, match.start() - 40)
                        end = min(len(completion), match.end() + 40)
                        context = completion[start:end].upper()

                        # Check if this is likely referring to market direction vs just a descriptive term
                        print(f"DEBUG - Checking context for ambiguous term {term}: '{context}'")

                        # Look for indicator context that makes it clear it's about market direction
                        direction_indicators = ["TREND", "MARKET", "PRICE", "MOVEMENT", "MOMENTUM", "DIRECTION"]
                        has_direction_context = any(indicator in context for indicator in direction_indicators)

                        # Check for words that suggest it's NOT about market direction
                        non_direction_indicators = ["SENTIMENT", "NEWS", "FEELING", "ATTITUDE", "OUTLOOK", "MACD IS POSITIVE", "MACD IS NEGATIVE", "SIGNAL IS POSITIVE", "SIGNAL IS NEGATIVE"]
                        has_non_direction_context = any(indicator in context for indicator in non_direction_indicators)

                        if has_direction_context and not has_non_direction_context:
                            prediction_found = True
                            detected_direction = direction
                            print(f"DEBUG - Ambiguous term {term} used in directional context")
                            reward += 0.5
                            reward_components['clear_prediction'] = 0.5
                            break
                    if prediction_found:
                        break

        # Check if prediction matches the expected one from thinking_structured
        thinking = sample.get("thinking_structured", {})
        if prediction_found and "prediction" in thinking:
            expected_prediction = thinking["prediction"].upper()

            print(f"DEBUG - Expected prediction: {expected_prediction}")
            print(f"DEBUG - Detected prediction: {detected_direction}")

            # Extract strong variants
            expected_base = expected_prediction.replace("STRONG_", "")

            # Check if detected direction matches expected direction or base
            if detected_direction == expected_prediction or detected_direction == expected_base:
                reward += 1.0  # Major reward for matching the expected prediction
                reward_components['correct_prediction'] = 1.0
                print(f"DEBUG - Found matching prediction '{expected_prediction}' in completion")

                # NEW: Extra reward for early prediction (in first half of text)
                first_half = completion[:len(completion)//2]
                if (expected_base == "UP" and any(term in first_half.upper() for term in prediction_terms["UP"])) or \
                    (expected_base == "DOWN" and any(term in first_half.upper() for term in prediction_terms["DOWN"])) or \
                    expected_prediction in first_half.upper():
                    reward += 0.5  # Bonus for early prediction
                    reward_components['early_prediction'] = 0.5
                    print(f"DEBUG - Found early prediction in first half of completion")

                # NEW: Extra reward for prediction in first few sentences
                first_sentences = '. '.join(completion.split('.')[:3])
                if (expected_base == "UP" and any(term in first_sentences.upper() for term in prediction_terms["UP"])) or \
                    (expected_base == "DOWN" and any(term in first_sentences.upper() for term in prediction_terms["DOWN"])) or \
                    expected_prediction in first_sentences.upper():
                    reward += 0.7  # Even bigger bonus for very early prediction
                    reward_components['very_early_prediction'] = 0.7
                    print(f"DEBUG - Found prediction in first few sentences")

        # Track if prediction missing and penalize slightly
        if not prediction_found and "prediction" in sample.get("thinking_structured", {}):
            reward -= 0.3
            reward_components['missing_prediction'] = -0.3

        # 4. Reward for thinking process if this should include thinking
        if sample.get("has_thinking", False):
            # Reward for using specific reasoning phrases
            reasoning_phrases = ["because", "due to", "as a result", "therefore", "since", "given that"]
            for phrase in reasoning_phrases:
                if phrase in completion.lower():
                    reward += 0.2
                    reward_components['reasoning_phrase'] = 0.2
                    break

            # Reward for detailed thinking
            if len(completion.split()) > 50:  # Reasonably detailed response
                reward += 0.3
                reward_components['detailed_thinking'] = 0.3
        else:
            # For non-thinking mode, reward conciseness and clarity
            if 10 < len(completion.split()) < 30:  # Short but not too short
                reward += 0.2
                reward_components['conciseness'] = 0.2

        # 5. Reward for mentioning specific price numbers
        price_pattern = r'\$\d+\.?\d*|\d+\.?\d*\s*dollars'
        if re.search(price_pattern, completion):
            reward += 0.2
            reward_components['price_mention'] = 0.2

        # 6. Reward for options trading recommendations
        options_pattern = r'(call|put)\s*option|options?(?:\s+trade|\s+strategy)?|(bull|bear)\s*(?:call|put)\s*spread|iron\s*condor|butterfly|straddle|strangle'
        if re.search(options_pattern, completion.lower()):
            reward += 0.3
            reward_components['options_recommendation'] = 0.3

            # Check for correct options strategy based on prediction
            thinking = sample.get("thinking_structured", {})
            if "prediction" in thinking:
                expected_prediction = thinking["prediction"].upper().replace("STRONG_", "")

                # Define bullish and bearish options strategies
                bullish_strategies = ["call option", "bull call spread", "bull put spread",
                                    "long call", "covered call", "leap call", "call debit spread"]
                bearish_strategies = ["put option", "bear put spread", "bear call spread",
                                    "long put", "protective put", "leap put", "put debit spread"]

                # Get timeframe-appropriate strategies
                timeframe = sample.get("detected_timeframe", "hourly")

                # Check if the recommended options strategy matches the direction
                if expected_prediction == "UP":
                    for strategy in bullish_strategies:
                        if strategy.lower() in completion.lower():
                            reward += 0.5
                            reward_components['correct_options_strategy'] = 0.5
                            print(f"DEBUG - Found appropriate bullish options strategy: {strategy}")
                            break
                elif expected_prediction == "DOWN":
                    for strategy in bearish_strategies:
                        if strategy.lower() in completion.lower():
                            reward += 0.5
                            reward_components['correct_options_strategy'] = 0.5
                            print(f"DEBUG - Found appropriate bearish options strategy: {strategy}")
                            break

            # Extra reward for including specific strike prices or expiration
            strike_pattern = r'strike\s*(?:price|)?\s*(?:of|at|:)?\s*\$?\d+(?:\.\d+)?'
            if re.search(strike_pattern, completion.lower()):
                reward += 0.2
                reward_components['strike_price_specified'] = 0.2

            # Detect expiration recommendations
            expiry_pattern = r'(?:expir(?:y|ation)|expiring)\s*(?:date|on|at|:)?\s*\d{1,2}[-/]\d{1,2}(?:[-/]\d{2,4})?|\d+\s*(?:day|week|month)'
            expiry_match = re.search(expiry_pattern, completion.lower())
            if expiry_match:
                reward += 0.2
                reward_components['expiration_specified'] = 0.2

                # Get the detected timeframe
                timeframe = sample.get("detected_timeframe", "hourly")

                # Check if expiration timeframe matches analysis timeframe
                expiry_text = expiry_match.group(0).lower()
                appropriate_expiry = False

                if timeframe == "hourly" and ("day" in expiry_text or "24 hour" in expiry_text or "same day" in expiry_text):
                    appropriate_expiry = True
                elif timeframe == "daily" and ("week" in expiry_text or "7 day" in expiry_text or "few day" in expiry_text):
                    appropriate_expiry = True
                elif timeframe == "weekly" and ("month" in expiry_text or "30 day" in expiry_text or "4 week" in expiry_text):
                    appropriate_expiry = True
                elif timeframe == "monthly" and ("leap" in expiry_text or "quarter" in expiry_text or "3 month" in expiry_text):
                    appropriate_expiry = True

                if appropriate_expiry:
                    reward += 0.3
                    reward_components['appropriate_expiry_timeframe'] = 0.3
                    print(f"DEBUG - Found appropriate expiry for {timeframe} timeframe: {expiry_text}")

        # Check for dual prediction format (hourly + options)
        hourly_prediction_pattern = r'\*\*\s*HOURLY\s*PREDICTION\s*:\s*(UP|DOWN|BULLISH|BEARISH|STRONG_UP|STRONG_DOWN)\s*\*\*'
        options_prediction_pattern = r'\*\*\s*2-WEEK\s*OPTIONS\s*STRATEGY\s*:\s*\*'

        hourly_match = re.search(hourly_prediction_pattern, completion, re.IGNORECASE)
        options_match = re.search(options_prediction_pattern, completion, re.IGNORECASE)

        if hourly_match and options_match:
            reward += 0.4  # Bonus for providing both predictions
            reward_components['dual_prediction_format'] = 0.4
            print(f"DEBUG - Found both hourly and options predictions")

            # Extract the hourly prediction to use for directional matching
            hourly_pred = hourly_match.group(1).upper()
            print(f"DEBUG - Extracted hourly prediction: {hourly_pred}")

            # Check if the extracted prediction matches the expected one
            if "thinking_structured" in sample and "prediction" in sample["thinking_structured"]:
                expected_pred = sample["thinking_structured"]["prediction"].upper()
                if hourly_pred == expected_pred or (hourly_pred == "UP" and expected_pred == "STRONG_UP") or (hourly_pred == "DOWN" and expected_pred == "STRONG_DOWN"):
                    reward += 0.7  # Higher reward for matching the expected prediction in the correct format
                    reward_components['matched_hourly_prediction'] = 0.7
                    print(f"DEBUG - Hourly prediction {hourly_pred} matches expected {expected_pred}")

        # Check if the raw term STRONG_UP or STRONG_DOWN appears anywhere in the text
        if "thinking_structured" in sample and "prediction" in sample["thinking_structured"]:
            expected_pred = sample["thinking_structured"]["prediction"].upper()
            if expected_pred in ["STRONG_UP", "STRONG_DOWN"] and expected_pred in completion.upper():
                print(f"DEBUG - Found raw strong prediction term: {expected_pred}")
                if not prediction_found:
                    prediction_found = True
                    reward += 0.6
                    reward_components['strong_prediction_term'] = 0.6

        # Check for exit conditions in options recommendation
        exit_pattern = r'(?:exit|sell|close)\s*(?:at|when|if)?\s*(?:price|target|profit|reaches|hits)?\s*[^.]*?\$?\d+(?:\.\d+)?'
        exit_match = re.search(exit_pattern, completion.lower())
        if exit_match:
            reward += 0.3
            reward_components['exit_condition'] = 0.3
            print(f"DEBUG - Found exit condition: {exit_match.group(0)}")

        # Check for accuracy-based reward if actual percentage change is available
        if "actual_percentage" in sample:
            try:
                actual_pct = float(sample["actual_percentage"])

                # Get the prediction
                if "prediction" in thinking:
                    predicted_direction = thinking["prediction"].upper().replace("STRONG_", "")

                    # Calculate accuracy reward
                    accuracy_reward, tier = calculate_accuracy_reward(predicted_direction, actual_pct)

                    if accuracy_reward > 0:
                        reward += accuracy_reward
                        reward_components[f'accuracy_reward_{tier}'] = accuracy_reward
                        print(f"DEBUG - Added accuracy reward of {accuracy_reward} for {tier} change ({actual_pct}%)")
            except (ValueError, TypeError) as e:
                print(f"DEBUG - Error calculating accuracy reward: {e}")

        # Check for both entry and exit price specifications
        entry_price_pattern = r'(?:entry|buy|open)\s*(?:at|price|target|when)?\s*[^.]*?\$?\d+(?:\.\d+)?'
        entry_match = re.search(entry_price_pattern, completion.lower())
        exit_match = re.search(exit_pattern, completion.lower())

        if entry_match and exit_match:
            reward += 0.3  # Bonus for specifying both entry and exit
            reward_components['complete_trade_plan'] = 0.3
            print(f"DEBUG - Found complete trade plan with entry and exit")

        # Specific check for early exit recommendation before expiration
        early_exit_pattern = r'(?:exit|sell|close)\s*(?:before|prior to|ahead of)\s*(?:expiration|expiry)'
        if re.search(early_exit_pattern, completion.lower()):
            reward += 0.2
            reward_components['early_exit_strategy'] = 0.2
            print(f"DEBUG - Found recommendation to exit before expiration")

        # Print the reward components for debugging
        print(f"Reward components: {reward_components}")
        print(f"Total reward: {reward}")

        # Simulate trade impact on bankroll if future prices are available
        if "future_prices" in sample and sample["future_prices"]:
            try:
                # Create prediction dict with extracted direction
                pred_direction = None

                # Try to get direction from thinking_structured first
                if "thinking_structured" in sample and "prediction" in sample["thinking_structured"]:
                    pred_direction = sample["thinking_structured"]["prediction"].upper().replace("STRONG_", "")
                # Otherwise try to extract it from the completion
                elif prediction_found:
                    # Extract the direction from matched patterns
                    if hourly_match:
                        pred_direction = hourly_match.group(1).upper()
                    elif strong_match:
                        pred_direction = strong_match.group(1).upper()
                    elif pred_match:
                        pred_direction = pred_match.group(1).upper()

                # Only proceed if we have a direction
                if pred_direction:
                    # Parse additional parameters (stop loss, take profit) if available
                    stop_loss_pct = 0.05  # Default 5%
                    take_profit_pct = 0.1  # Default 10%

                    # Try to extract stop loss percentage
                    sl_pattern = r'stop\s*(?:loss|price)(?:\s*at|\s*:|\s*=)?\s*(?:\$?(\d+(?:\.\d+)?)|\s*(\d+(?:\.\d+)?)%)'
                    sl_match = re.search(sl_pattern, completion.lower())
                    if sl_match:
                        # Check if dollar amount or percentage
                        if sl_match.group(1):  # Dollar amount
                            # Handle dollar-based stop loss (would need current price)
                            if "current_price" in sample:
                                current_price = float(sample["current_price"])
                                stop_price = float(sl_match.group(1))
                                # Calculate percentage
                                if pred_direction == "UP":
                                    stop_loss_pct = (current_price - stop_price) / current_price
                                else:
                                    stop_loss_pct = (stop_price - current_price) / current_price
                        else:  # Percentage
                            stop_loss_pct = float(sl_match.group(2)) / 100.0

                    # Try to extract take profit percentage
                    tp_pattern = r'take\s*profit(?:\s*at|\s*:|\s*=)?\s*(?:\$?(\d+(?:\.\d+)?)|\s*(\d+(?:\.\d+)?)%)'
                    tp_match = re.search(tp_pattern, completion.lower())
                    if tp_match:
                        # Check if dollar amount or percentage
                        if tp_match.group(1):  # Dollar amount
                            # Handle dollar-based take profit (would need current price)
                            if "current_price" in sample:
                                current_price = float(sample["current_price"])
                                take_price = float(tp_match.group(1))
                                # Calculate percentage
                                if pred_direction == "UP":
                                    take_profit_pct = (take_price - current_price) / current_price
                                else:
                                    take_profit_pct = (current_price - take_price) / current_price
                        else:  # Percentage
                            take_profit_pct = float(tp_match.group(2)) / 100.0

                    # Create bankroll manager if not already in sample
                    if "bankroll_manager" not in sample:
                        sample["bankroll_manager"] = BankrollManager(initial_capital=200.0, position_size_pct=0.20, max_position_pct=0.50)

                    # Create prediction dict
                    prediction_dict = {
                        "direction": pred_direction,
                        "stop_loss_pct": stop_loss_pct,
                        "take_profit_pct": take_profit_pct
                    }

                    # Get future prices from sample
                    future_prices = sample["future_prices"]

                    # Evaluate the prediction
                    bankroll_reward, bankroll_components = sample["bankroll_manager"].evaluate_prediction(
                        prediction_dict, future_prices, sample.get("detected_timeframe", "hourly"))

                    # Update the global bankroll manager with the updated instance
                    global global_bankroll_manager
                    global_bankroll_manager = sample["bankroll_manager"]

                    # Add bankroll reward to total
                    reward += bankroll_reward

                    # Add bankroll components to reward components
                    for key, value in bankroll_components.items():
                        reward_components[f"bankroll_{key}"] = value

                    print(f"DEBUG - Bankroll evaluation results: {bankroll_components}")
                    print(f"DEBUG - Current capital: ${sample['bankroll_manager'].current_capital:.2f}")

                    # Try to extract confidence from the completion
                    confidence_pattern = r'confidence[:\s]+(\d+(?:\.\d+)?)[%\s]*'
                    confidence_match = re.search(confidence_pattern, completion.lower())
                    if confidence_match:
                        try:
                            confidence_value = float(confidence_match.group(1))
                            # Convert percentage to decimal if needed
                            if confidence_value > 1:
                                confidence_value = confidence_value / 100.0
                            # Cap between 0 and 1
                            confidence_value = max(0.0, min(1.0, confidence_value))
                            prediction_dict['confidence'] = confidence_value
                            print(f"DEBUG - Extracted confidence: {confidence_value:.2f}")
                        except (ValueError, TypeError) as e:
                            print(f"DEBUG - Error parsing confidence: {e}")
                            prediction_dict['confidence'] = 0.5  # Default 50% confidence

                    # Also try to extract position size percentage
                    position_size_pattern = r'(?:position\s*size|allocate|risk)[:\s]+(\d+(?:\.\d+)?)[%\s]*'
                    position_match = re.search(position_size_pattern, completion.lower())
                    if position_match:
                        try:
                            position_size = float(position_match.group(1))
                            prediction_dict['position_size_pct'] = position_size
                            print(f"DEBUG - Extracted position size: {position_size:.2f}%")
                        except (ValueError, TypeError) as e:
                            print(f"DEBUG - Error parsing position size: {e}")

            except Exception as e:
                print(f"DEBUG - Error in bankroll evaluation: {e}")

        # Add an explicit debug section for prediction detection
        print(f"DEBUG - Full prediction detection details:")
        print(f"  - Prediction found: {prediction_found}")
        print(f"  - Raw completion (first 200 chars): {completion[:200]}")
        print(f"  - Primary prediction terms: {prediction_terms}")
        print(f"  - Ambiguous terms requiring context: {ambiguous_terms}")
        print(f"  - Using prediction phrases for UP/DOWN detection: {len(prediction_phrases['UP'])} UP phrases, {len(prediction_phrases['DOWN'])} DOWN phrases")

        if prediction_found:
            print(f"  - Detected direction: {detected_direction}")

        if "thinking_structured" in sample and "prediction" in sample["thinking_structured"]:
            print(f"  - Expected prediction: {sample['thinking_structured']['prediction']}")

            # Add more detailed detection that analyzes why prediction was or wasn't found
            expected_pred = sample["thinking_structured"]["prediction"].upper()
            expected_base = expected_pred.replace("STRONG_", "")

            print(f"  - Direct STRONG_UP/DOWN in text: {expected_pred in completion.upper()}")

            # Check for base forms (UP/DOWN)
            base_match = any(term in completion.upper() for term in prediction_terms[expected_base])
            print(f"  - Related {expected_base} terms in text: {base_match}")

            # Check for phrase patterns
            phrase_match = False
            for phrase in prediction_phrases[expected_base]:
                if re.search(phrase, completion.upper()):
                    phrase_match = True
                    phrase_context = re.search(phrase, completion.upper())
                    start = max(0, phrase_context.start() - 20)
                    end = min(len(completion), phrase_context.end() + 20)
                    print(f"  - Found {expected_base} phrase pattern: '{completion[start:end]}'")
                    break

            print(f"  - {expected_base} phrase patterns in text: {phrase_match}")

            # Check for ambiguous forms
            if expected_base in ambiguous_terms:
                ambig_match = any(term in completion.upper() for term in ambiguous_terms[expected_base])
                print(f"  - Ambiguous {expected_base} terms in text: {ambig_match}")

            # Look for the context around any matching term
            matching_contexts = []
            for term in prediction_terms[expected_base]:
                term_pattern = r'\b' + term + r'\b'
                matches = re.finditer(term_pattern, completion.upper())
                for match in matches:
                    start = max(0, match.start() - 20)
                    end = min(len(completion), match.end() + 20)
                    matching_contexts.append(f"'{completion[start:end]}'")

            # Also check ambiguous terms
            if expected_base in ambiguous_terms:
                for term in ambiguous_terms[expected_base]:
                    term_pattern = r'\b' + term + r'\b'
                    matches = re.finditer(term_pattern, completion.upper())
                    for match in matches:
                        start = max(0, match.start() - 20)
                        end = min(len(completion), match.end() + 20)
                        matching_contexts.append(f"'{completion[start:end]}' (ambiguous)")

            if matching_contexts:
                print(f"  - Matching contexts: {', '.join(matching_contexts[:3])}")

        # Extract any special formatting markers in the text that might be causing issues
        special_format = re.findall(r'[\*\#\<\>\_\`\~\|]+', completion[:200])
        if special_format:
            print(f"  - Special formatting detected: {special_format}")

        # Update the prompt format to include a single, efficient thinking section
        if last_user_idx >= 0:
            # Add hint that requests both predictions with specific formatting
            hint = (
                f"\nProvide your analysis in this exact format:\n"
                f"<thinking>\n"
                f"[Include short analysis for both hourly price movement AND options strategy. Be concise but thorough.]\n"
                f"</thinking>\n\n"
                f"**HOURLY PREDICTION: [UP/DOWN/STRONG_UP/STRONG_DOWN]**\n"
                f"[Brief hourly conclusion and reasoning]\n\n"
                f"**2-WEEK OPTIONS STRATEGY:**\n"
                f"[Options recommendation with strike price, expiration, entry/exit conditions, and position size as % of capital based on confidence]"
            )
            messages[last_user_idx]["content"] = messages[last_user_idx]["content"] + hint

        # Add pattern for combined thinking section
        combined_thinking_pattern = r'<thinking>(.*?)</thinking>'

        # Add reward for concise but comprehensive thinking
        # Add this right before the full_format_match check
        thinking_match = re.search(combined_thinking_pattern, completion, re.DOTALL)
        if thinking_match:
            thinking_content = thinking_match.group(1).strip()
            # Reward for reasonable thinking length - not too short, not excessively long
            if 100 <= len(thinking_content) <= 800:  # Efficient length
                reward += 0.4
                reward_components['efficient_thinking'] = 0.4
                print(f"DEBUG - Found efficient thinking section ({len(thinking_content)} chars)")
            elif len(thinking_content) > 800:  # Too verbose
                reward += 0.1  # Smaller reward for thinking that's too long
                reward_components['verbose_thinking'] = 0.1
                print(f"DEBUG - Found thinking section but it's too verbose ({len(thinking_content)} chars)")

        # Update the full format match to include thinking
        full_format_match = (
            re.search(combined_thinking_pattern, completion, re.DOTALL) and
            re.search(r'\*\*\s*HOURLY\s*PREDICTION\s*:\s*(UP|DOWN|BULLISH|BEARISH|STRONG_UP|STRONG_DOWN)\s*\*\*', completion, re.IGNORECASE) and
            re.search(r'\*\*\s*2-WEEK\s*OPTIONS\s*STRATEGY\s*:\s*\*\*', completion, re.IGNORECASE)
        )

        if full_format_match:
            reward += 0.5  # Bonus for following complete format with both predictions
            reward_components['complete_format'] = 0.5
            print(f"DEBUG - Found complete format with hourly prediction and options recommendation")

        # Final report about prediction detection for debugging
        if "thinking_structured" in sample and "prediction" in sample["thinking_structured"]:
            expected_pred = sample["thinking_structured"]["prediction"].upper()
            expected_base = expected_pred.replace("STRONG_", "")

            # Check for phrase patterns
            phrase_match = False
            for phrase in prediction_phrases[expected_base]:
                if re.search(phrase, completion.upper()):
                    phrase_match = True
                    break

            # Check for ambiguous terms
            ambiguous_match = False
            if expected_base in ambiguous_terms:
                for term in ambiguous_terms[expected_base]:
                    term_pattern = r'\b' + term + r'\b'
                    matches = list(re.finditer(term_pattern, completion.upper()))
                    for match in matches:
                        # Get context around the term
                        start = max(0, match.start() - 40)
                        end = min(len(completion), match.end() + 40)
                        context = completion[start:end].upper()

                        # Look for indicator context that makes it clear it's about market direction
                        direction_indicators = ["TREND", "MARKET", "PRICE", "MOVEMENT", "MOMENTUM", "DIRECTION"]
                        has_direction_context = any(indicator in context for indicator in direction_indicators)

                        # Check for words that suggest it's NOT about market direction
                        non_direction_indicators = ["SENTIMENT", "NEWS", "FEELING", "ATTITUDE", "OUTLOOK", "MACD IS POSITIVE", "MACD IS NEGATIVE", "SIGNAL IS POSITIVE", "SIGNAL IS NEGATIVE"]
                        has_non_direction_context = any(indicator in context for indicator in non_direction_indicators)

                        if has_direction_context and not has_non_direction_context:
                            ambiguous_match = True
                            break

            # Determine if prediction is found based on our enhanced detection logic
            prediction_in_completion = (
                expected_pred in completion.upper() or  # Direct match for STRONG_UP/DOWN
                any(term in completion.upper() for term in prediction_terms[expected_base]) or  # Base direction terms
                phrase_match or  # Phrase patterns indicating direction
                ambiguous_match  # Correctly contextualized ambiguous terms
            )

            print(f"Reward components for debugging:")
            print(f"- has_thinking: {sample.get('has_thinking', False)}")
            if "thinking_structured" in sample:
                print(f"- thinking keys: {list(sample['thinking_structured'].keys())}")
                if "prediction" in sample.get("thinking_structured", {}):
                    print(f"- prediction in thinking: {sample['thinking_structured']['prediction']}")
                    print(f"- prediction found in completion: {prediction_in_completion}")
                    print(f"- prediction_found flag value: {prediction_found}")
                if "reasoning" in sample.get("thinking_structured", {}):
                    reasoning_len = len(sample['thinking_structured']['reasoning'].split())
                    print(f"- reasoning length: {reasoning_len} words")

        return reward

    # Add timeframe detection and appropriate expiration matching
    # First, add a function to detect the timeframe from the datetime string
    def detect_timeframe(sample):
        """Detect the timeframe from the sample's datetime string"""
        timeframe = "hourly"  # Default timeframe

        if "datetime_str" in sample:
            datetime_str: datetime = sample["datetime_str"]

            # Check if the datetime contains hour:minute format (implies hourly/intraday)
            if datetime_str.hour > 0 and datetime_str.minute > 0 and datetime_str.second > 0:
                # If it has minutes, it's likely hourly data
                timeframe = "hourly"
            elif datetime_str.day > 0:
                # If it only has date but no time, likely daily data
                timeframe = "daily"

        # Check for explicit timeframe mentions in messages
        if "messages" in sample:
            for msg in sample["messages"]:
                if "content" in msg:
                    content = msg["content"].lower()
                    if "weekly" in content or "week" in content:
                        timeframe = "weekly"
                    elif "monthly" in content or "month" in content:
                        timeframe = "monthly"
                    elif "daily" in content or "day" in content:
                        timeframe = "daily"
                    elif "hourly" in content or "hour" in content:
                        timeframe = "hourly"

        return timeframe

    # Add a function to calculate prediction accuracy reward on a sliding scale
    def calculate_accuracy_reward(prediction, actual_change_pct):
        """Calculate reward based on sliding scale of prediction accuracy"""
        # Default reward for making any prediction
        reward = 0.0
        reward_tier = "none"

        # Handle null cases
        if prediction is None or actual_change_pct is None:
            return reward, reward_tier

        prediction = prediction.upper()

        # Calculate the reward based on correctness and percentage
        if (prediction == "UP" and actual_change_pct > 0) or (prediction == "DOWN" and actual_change_pct < 0):
            # Correct directional prediction
            abs_change = abs(actual_change_pct)

            if abs_change <= 10:
                reward = 0.3
                reward_tier = "0-10%"
            elif abs_change <= 25:
                reward = 0.6
                reward_tier = "11-25%"
            elif abs_change <= 50:
                reward = 0.8
                reward_tier = "26-50%"
            else:
                reward = 1.25
                reward_tier = "51%+"

        # No penalty for incorrect prediction as requested
        return reward, reward_tier

    # Add a BankrollManager class to track trades, P&L and account value
    class BankrollManager:
        """Manages the AI's trading bankroll and P&L calculations."""
        def __init__(self, initial_capital=200.0, position_size_pct=0.20, max_position_pct=0.50):
            self.initial_capital = initial_capital
            self.current_capital = initial_capital
            self.position_size_pct = position_size_pct  # Default position size as % of capital
            self.max_position_pct = max_position_pct    # Maximum position size as % of capital
            self.trade_history = []
            self.positions = {}  # Currently open positions
            self.min_capital = initial_capital  # Track lowest capital point
            self.max_capital = initial_capital  # Track highest capital point
            self.total_trades = 0
            self.winning_trades = 0
            self.losing_trades = 0
            self.bankrupt_count = 0  # Track number of bankruptcies
            self.position_size_variety = set()  # Track variety of position sizes used

        def calculate_position_size(self, price, confidence=None, risk_pct=None):
            """Calculate position size based on current capital, confidence and risk."""
            if self.current_capital <= 0:
                # Reset if bankrupt
                self.reset_after_bankruptcy()

            # If specific risk percentage is provided, use it
            if risk_pct is not None:
                position_pct = risk_pct
            # Otherwise if confidence is provided, scale position size
            elif confidence is not None:
                # Scale from min 5% to max position size based on confidence (0-1)
                min_position = 0.05  # Minimum 5% position size
                position_pct = min_position + (self.max_position_pct - min_position) * confidence
            else:
                position_pct = self.position_size_pct

            # Cap the position size at the maximum percentage
            position_pct = min(position_pct, self.max_position_pct)

            # Calculate position size in dollars
            position_size_dollars = self.current_capital * position_pct

            # Calculate number of shares/contracts
            num_shares = position_size_dollars / price

            # Track position size variety
            self.position_size_variety.add(round(position_pct * 100))

            return num_shares, position_size_dollars, position_pct

        def reset_after_bankruptcy(self):
            """Reset the account after bankruptcy."""
            self.bankrupt_count += 1
            self.current_capital = self.initial_capital
            print(f"DEBUG - Bankroll reset after bankruptcy #{self.bankrupt_count}. New capital: ${self.current_capital:.2f}")

        def open_position(self, symbol, direction, entry_price, confidence=None, size_pct=None,
                        stop_loss_pct=0.05, take_profit_pct=0.1, expiration=None):
            """Open a new trading position with dynamic sizing based on confidence."""
            # Determine position size based on confidence or explicit size
            num_shares, position_dollars, actual_position_pct = self.calculate_position_size(
                entry_price, confidence, size_pct)

            # Create position object
            position = {
                'symbol': symbol,
                'direction': direction.upper(),  # UP or DOWN
                'entry_price': entry_price,
                'num_shares': num_shares,
                'position_dollars': position_dollars,
                'position_pct': actual_position_pct,
                'confidence': confidence,
                'stop_loss': entry_price * (1 - stop_loss_pct) if direction.upper() == 'UP' else entry_price * (1 + stop_loss_pct),
                'take_profit': entry_price * (1 + take_profit_pct) if direction.upper() == 'UP' else entry_price * (1 - take_profit_pct),
                'entry_time': 'now',  # In a real implementation, use actual timestamp
                'expiration': expiration,
                'status': 'open'
            }

            # Generate unique position ID
            position_id = f"{symbol}_{self.total_trades + 1}"

            # Store position
            self.positions[position_id] = position

            # Increment trade counter
            self.total_trades += 1

            return position_id

        def evaluate_prediction(self, prediction, future_prices, timeframe="hourly"):
            """
            Simulate a trade based on prediction and future price data.
            Returns the reward based on P&L and account growth/bankruptcy.
            """
            reward = 0
            reward_components = {}

            try:  # Add try/except to catch and log any errors
                # Skip if no prediction or future prices
                if not prediction or not future_prices or len(future_prices) < 2:
                    return reward, reward_components

                # Extract key data
                current_price = future_prices[0]
                direction = prediction.get('direction', '').upper()
                confidence = prediction.get('confidence', 0.5)  # Default 50% confidence if not specified

                if not direction or not current_price:
                    return reward, reward_components

                # Get basic simulation parameters
                stop_loss_pct = 0.05  # Default 5% stop loss
                take_profit_pct = 0.1  # Default 10% take profit

                # Extract custom parameters if available
                if 'stop_loss_pct' in prediction:
                    try:
                        stop_loss_pct = float(prediction['stop_loss_pct'])
                    except (ValueError, TypeError):
                        pass

                if 'take_profit_pct' in prediction:
                    try:
                        take_profit_pct = float(prediction['take_profit_pct'])
                    except (ValueError, TypeError):
                        pass

                # Add position sizing based on confidence
                size_pct = None
                if 'position_size_pct' in prediction:
                    try:
                        size_pct = float(prediction['position_size_pct']) / 100.0  # Convert from percentage
                    except (ValueError, TypeError):
                        pass

                # Determine position size based on confidence or explicit size
                num_shares, position_dollars, actual_position_pct = self.calculate_position_size(
                    current_price, confidence, size_pct)

                # IMPORTANT: Remove the invested capital from the bankroll
                # This implements fractional trading - we're setting aside this portion of capital for the trade
                starting_capital = self.current_capital
                self.current_capital -= position_dollars

                print(f"DEBUG - Starting trade with ${starting_capital:.2f} capital")
                print(f"DEBUG - Investing ${position_dollars:.2f} ({actual_position_pct*100:.1f}%) in {direction} trade")
                print(f"DEBUG - Remaining capital: ${self.current_capital:.2f}")

                # Create position object
                position = {
                    'symbol': 'SYMBOL',
                    'direction': direction,
                    'entry_price': current_price,
                    'num_shares': num_shares,
                    'position_dollars': position_dollars,
                    'position_pct': actual_position_pct,
                    'confidence': confidence,
                    'stop_loss_pct': stop_loss_pct,
                    'take_profit_pct': take_profit_pct,
                    'entry_time': 'now',
                }

                # Generate unique position ID
                position_id = f"SYMBOL_{self.total_trades + 1}"

                # Store position
                self.positions[position_id] = position

                # Simulate the position based on future prices
                max_price = current_price
                min_price = current_price
                exit_price = future_prices[-1]  # Default to last price if no SL/TP hit
                position_outcome = "held"  # Default outcome

                # Calculate stop loss and take profit prices
                if direction == "UP":
                    stop_loss_price = current_price * (1 - stop_loss_pct)
                    take_profit_price = current_price * (1 + take_profit_pct)
                else:  # DOWN
                    stop_loss_price = current_price * (1 + stop_loss_pct)
                    take_profit_price = current_price * (1 - take_profit_pct)

                # Simulate price action through future prices
                for i, price in enumerate(future_prices[1:], 1):  # Skip first price (current)
                    # Track price extremes
                    max_price = max(max_price, price)
                    min_price = min(min_price, price)

                    # Check if stop loss hit
                    if (direction == "UP" and price <= stop_loss_price) or \
                    (direction == "DOWN" and price >= stop_loss_price):
                        exit_price = stop_loss_price
                        position_outcome = "stop_loss"
                        break

                    # Check if take profit hit
                    if (direction == "UP" and price >= take_profit_price) or \
                    (direction == "DOWN" and price <= take_profit_price):
                        exit_price = take_profit_price
                        position_outcome = "take_profit"
                        break

                # Calculate P&L
                if direction == "UP":
                    pnl_pct = (exit_price - current_price) / current_price
                else:  # DOWN
                    pnl_pct = (current_price - exit_price) / current_price

                # Calculate absolute P&L
                pnl_dollars = position_dollars * pnl_pct

                # Calculate the final value of the investment
                final_position_value = position_dollars + pnl_dollars

                # Return the investment plus profit/loss to the bankroll
                self.current_capital += final_position_value

                # Prevent negative balance
                if self.current_capital <= 0:
                    self.reset_after_bankruptcy()

                # Track high/low watermarks
                self.min_capital = min(self.min_capital, self.current_capital)
                self.max_capital = max(self.max_capital, self.current_capital)

                # Update trade history
                self.trade_history.append({
                    'position_id': position_id,
                    'direction': direction,
                    'entry_price': current_price,
                    'exit_price': exit_price,
                    'pnl_pct': pnl_pct,
                    'pnl_dollars': pnl_dollars,
                    'investment': position_dollars,
                    'final_value': final_position_value,
                    'outcome': position_outcome
                })

                # Update trade statistics
                self.total_trades += 1
                if pnl_dollars > 0:
                    self.winning_trades += 1
                elif pnl_dollars < 0:
                    self.losing_trades += 1

                # Log the trade results
                print(f"DEBUG - Trade {position_id} {direction} from ${current_price:.2f} to ${exit_price:.2f} ({position_outcome})")
                print(f"DEBUG - Invested: ${position_dollars:.2f}, Final value: ${final_position_value:.2f}, P&L: ${pnl_dollars:.2f} ({pnl_pct*100:.2f}%)")
                print(f"DEBUG - New account balance: ${self.current_capital:.2f}")

                # Calculate reward components based on trade outcome
                if pnl_dollars > 0:
                    # Reward for winning trade
                    reward += 0.5
                    reward_components['winning_trade'] = 0.5

                    if position_outcome == "take_profit":
                        # Extra reward for hitting take profit
                        reward += 0.3
                        reward_components['hit_take_profit'] = 0.3

                    # Reward based on P&L percentage (larger wins get more reward)
                    if pnl_pct >= 0.05:  # 5%+
                        reward += 0.4
                        reward_components['large_profit'] = 0.4

                elif pnl_dollars < 0:
                    # Small penalty for losing trade
                    reward -= 0.1
                    reward_components['losing_trade'] = -0.1

                    if position_outcome == "stop_loss":
                        # Small penalty for hitting stop loss, but not too harsh since stops are good practice
                        reward -= 0.1
                        reward_components['hit_stop_loss'] = -0.1

                # Add reward for appropriate position sizing
                if confidence and abs(confidence - position['position_pct']/self.max_position_pct) < 0.2:
                    # Reward if position size is appropriate to confidence
                    reward += 0.2
                    reward_components['appropriate_position_size'] = 0.2

                # Reward for position size diversity (if we have enough trades)
                if self.total_trades >= 3 and len(self.position_size_variety) >= 3:
                    reward += 0.4
                    reward_components['position_size_diversity'] = 0.4
                    print(f"DEBUG - Using diverse position sizes: {sorted(self.position_size_variety)}")

            except Exception as e:
                print(f"DEBUG - Error in bankroll evaluation: {e}")

            return reward, reward_components

    # Initialize a global bankroll manager for the entire run
    global_bankroll_manager = BankrollManager(initial_capital=200.0, position_size_pct=0.20, max_position_pct=0.50)

    # 5. RL/GRPO Training Loop
    num_epochs = 1
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx, batch in enumerate(tqdm(dataloader)):
            try:
                sample = batch[0]  # batch_size=1

                # Add detection code here
                # Detect and store timeframe in sample
                detected_timeframe = detect_timeframe(sample)
                print(f"DEBUG - Detected timeframe: {detected_timeframe}")
                sample["detected_timeframe"] = detected_timeframe  # Store in the sample dictionary

                # Initialize bankroll manager using the global instance
                if "bankroll_manager" not in sample:
                    sample["bankroll_manager"] = global_bankroll_manager
                    print(f"DEBUG - Using global bankroll manager with current capital ${global_bankroll_manager.current_capital:.2f}")

                # Continue with DEBUG and sample processing
                print("\nDEBUG - Sample structure:")
                print(f"Sample keys: {list(sample.keys())}")
                if "messages" in sample:
                    print(f"Number of messages: {len(sample['messages'])}")
                    print(f"Message keys: {list(sample['messages'][0].keys()) if sample['messages'] else 'No messages'}")
                if "thinking_structured" in sample:
                    print(f"Thinking structured keys: {list(sample['thinking_structured'].keys())}")

                messages = sample["messages"]

                # Correctly restore the user message detection code but use only the formatted hint
                # Add this after the message variable is set:
                # Add special instruction at the end of user message to encourage early prediction
                if len(messages) > 0:
                    # Get the last user message
                    last_user_idx = -1
                    for i, msg in enumerate(messages):
                        if msg.get("role") == "user":
                            last_user_idx = i

                # Add historical data context for better options predictions
                # Before processing the messages, add historical context if available
                if "thinking_structured" in sample and "historical_context" in sample["thinking_structured"]:
                    # Try to extract historical data from thinking_structured
                    historical_data = sample["thinking_structured"]["historical_context"]

                    # Add this to the user message if not already present
                    if last_user_idx >= 0 and historical_data and "historical data" not in messages[last_user_idx]["content"].lower():
                        historical_context = f"\n\nHistorical data for the past 2 weeks:\n{historical_data}"
                        messages[last_user_idx]["content"] = messages[last_user_idx]["content"] + historical_context
                        print(f"DEBUG - Added historical context to prompt")
                elif "ticker" in sample:
                    # If we have a ticker but no historical data, add a generic historical reference
                    ticker = sample["ticker"]
                    if last_user_idx >= 0 and "historical data" not in messages[last_user_idx]["content"].lower():
                        historical_note = f"\n\nConsider recent 2-week price action for {ticker} when making options recommendations."
                        messages[last_user_idx]["content"] = messages[last_user_idx]["content"] + historical_note
                        print(f"DEBUG - Added historical note for {ticker}")

                prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                print(inputs['input_ids'].device, model.device)
                print(inputs['input_ids'].dtype)

                # Get the ACTUAL vocab size from the model (not the tokenizer)
                model_vocab_size = model.get_input_embeddings().weight.shape[0]
                print(f"Max token ID: {inputs['input_ids'].max().item()}, Model vocab size: {model_vocab_size}, Tokenizer vocab size: {len(tokenizer)}")

                if inputs['input_ids'].max().item() >= model_vocab_size:
                    print("ERROR: Out-of-vocab token detected!")
                    print("Max token ID:", inputs['input_ids'].max().item())
                    print("Model vocab size:", model_vocab_size)
                    continue  # Skip this batch

                # STEP 1: Generate completions with the model (inference only)
                model.eval()  # Set to eval mode for generation
                with torch.no_grad():
                    generated = ""
                    async for output in vllm_sampler.sample(
                        messages[:-1]
                    ):
                        generated += output
                    # output_ids = model.generate(
                    #     **inputs,
                    #     max_new_tokens=256,
                    #     do_sample=False,
                    #     num_beams=1,
                    #     use_cache=True,
                    #     pad_token_id=tokenizer.pad_token_id,
                    #     eos_token_id=tokenizer.eos_token_id,
                    # )
                # generated = tokenizer.decode(output_ids[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

                # Add a limit to the maximum output tokens to prevent OOM errors
                MAX_OUTPUT_TOKENS = 196  # Increased from 128 for more complete outputs
                # output_length = output_ids[0][inputs['input_ids'].shape[1]:].shape[0]
                generated_tokens = tokenizer.encode(generated)
                output_length = len(generated_tokens)
                if output_length > MAX_OUTPUT_TOKENS:
                    print(f"WARNING: Output length {output_length} exceeds limit. Truncating to {MAX_OUTPUT_TOKENS} tokens")
                    # Truncate the output_ids sequence for training
                    output_ids = torch.cat([
                        output_ids[0][:inputs['input_ids'].shape[1]],
                        output_ids[0][inputs['input_ids'].shape[1]:inputs['input_ids'].shape[1]+MAX_OUTPUT_TOKENS]
                    ]).unsqueeze(0)
                    output_length = MAX_OUTPUT_TOKENS

                # Print the actual generated text
                print("\n" + "="*50)
                print("INPUT PROMPT:")
                # Show more of the prompt to provide better context
                if len(prompt) > 300:
                    print("..." + prompt[-300:])
                else:
                    print(prompt)
                print("\nGENERATED OUTPUT:")
                print(generated)
                print("="*50 + "\n")

                # Compute reward with detailed logging for debugging
                print("Computing reward...")
                reward = custom_reward(sample, generated)
                print(f"Reward components for debugging:")
                print(f"- has_thinking: {sample.get('has_thinking', False)}")
                if "thinking_structured" in sample:
                    print(f"- thinking keys: {list(sample['thinking_structured'].keys())}")
                    if "prediction" in sample.get("thinking_structured", {}):
                        print(f"- prediction in thinking: {sample['thinking_structured']['prediction']}")
                        print(f"- prediction found in completion: {'prediction' in sample['thinking_structured'] and sample['thinking_structured']['prediction'].lower() in generated.lower()}")
                    if "reasoning" in sample.get("thinking_structured", {}):
                        reasoning_len = len(sample['thinking_structured']['reasoning'].split())
                        print(f"- reasoning length: {reasoning_len} words")

                # STEP 2: Prepare model for training
                model.train()  # Set back to train mode for the backward pass

                # STEP 3: Create the training labels: concatenate prompt with generated completion
                label_ids = torch.cat([inputs['input_ids'], output_ids[0][inputs['input_ids'].shape[1]:].unsqueeze(0)], dim=1)

                # Create a reward tensor based on the reward value (repeated for each output token)
                reward_tensor = torch.ones(output_length, device=model.device) * reward

                # STEP 4: Forward pass - do the actual training with gradient accumulation
                # Create a tensor to hold all token-level losses
                total_loss = 0.0

                # Get prompt length to use as offset
                prompt_length = inputs['input_ids'].shape[1]

                # Define how many tokens to process before each gradient update
                tokens_per_update = 64  # Increased from 32 for better speed

                # We'll process the forward pass in chunks with gradient accumulation
                for i in range(0, output_length, tokens_per_update):
                    # Clear gradients before each chunk to avoid accumulating too much memory
                    optimizer.zero_grad()

                    # Process a chunk of tokens
                    chunk_end = min(i + tokens_per_update, output_length)
                    chunk_size = chunk_end - i
                    chunk_loss = 0.0

                    for j in range(i, chunk_end):
                        # The sequence so far (prompt + generated tokens so far)
                        prefix_length = prompt_length + j
                        current_input_ids = label_ids[:, :prefix_length]

                        # Get the label for the next token
                        target_id = label_ids[:, prefix_length]

                        # Forward pass through the model to get logits
                        outputs = model(input_ids=current_input_ids, attention_mask=torch.ones_like(current_input_ids))
                        logits = outputs.logits[:, -1, :]  # Get logits for the last token

                        # Apply log softmax to get log probabilities
                        log_probs = F.log_softmax(logits, dim=-1)

                        # Get log prob for the correct next token
                        token_log_prob = log_probs[0, target_id[0]]

                        # Apply weighted loss (REINFORCE with reward rescaling)
                        token_loss = -token_log_prob * reward_tensor[j]

                        # Add to chunk loss
                        chunk_loss += token_loss

                        # Accumulate total loss for reporting
                        total_loss += token_loss.detach()

                        # Print details for first few tokens (for debugging)
                        if j < 3:
                            print(f"Token {j} - ID: {target_id[0].item()}, " +
                                f"Token: '{tokenizer.decode([target_id[0].item()])}', " +
                                f"Log prob: {token_log_prob.item():.4f}, " +
                                f"Loss: {token_loss.item():.4f}")

                    # Average the chunk loss and backpropagate
                    chunk_loss = chunk_loss / chunk_size
                    chunk_loss.backward()

                    # Apply gradients
                    optimizer.step()

                    # Optimize the memory clearing to be less frequent
                    # Clear memory
                    if i % (tokens_per_update * 4) == 0:  # Only clear every 4 chunks instead of every chunk
                        torch.cuda.empty_cache()

                # Average the token-level losses for reporting
                avg_loss = total_loss / output_length

                # Print final loss value
                print(f"Reward: {reward:.3f}, Total loss: {avg_loss.item():.3f}")

                # STEP 5: Already did optimization in chunks, so only need to clear gradients
                optimizer.zero_grad()
                vllm_sampler.client.update_model_params(model)
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    print(f"WARNING: CUDA OOM error on sample {batch_idx}. Skipping and clearing cache.")
                    # Clear CUDA cache to recover
                    torch.cuda.empty_cache()
                    # Wait a moment for memory to clear
                    import time
                    time.sleep(5)
                    continue
                else:
                    # Re-raise other errors
                    raise
            except Exception as e:
                import traceback
                print(traceback.format_exc())
                print(f"ERROR processing sample {batch_idx}: {str(e)}")
                continue

        print(f"Epoch {epoch+1} complete.")

    # 6. Save model and tokenizer
    output_dir = "/qwen3_14b_memory_optimized_lora"
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Training complete. LoRA adapters saved to {output_dir}.")

    # To generate after training, first load the original model and tokenizer with merged LoRA weights
    prompt = "Hello, how are you?"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=10)
    print(tokenizer.decode(output_ids[0], skip_special_tokens=True))