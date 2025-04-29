import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import json
import re
from typing import List, Dict
from datasets import load_dataset
from tqdm import tqdm
from dotenv import load_dotenv
import torch
from unsloth import FastLanguageModel
from pathlib import Path

# MC GRPO
# Evaluation Results:
# Average Precision: 0.4293
# Average Recall: 0.4116
# Average F1: 0.3870
# Total True Positives: 290
# Total False Positives: 366
# Total False Negatives: 636

# MC SFT Only 

# Base Model

load_dotenv()

MAX_INPUT_LENGTH = 8192

# Implement similar parsing function as in the original non_mc_eval.py
def parse_response(response: str) -> List[str]:
    """
    Parse the model's response to extract GO terms from non-multiple choice format.
    Expects GO terms to be in \boxed{} notation with GO IDs.
    """
    # Extract content between \boxed{} tags
    boxed_matches = re.findall(r'\\boxed{(.*?)}', response)
    if not boxed_matches:
        # Try to find GO terms in the answer tag if no boxed notation
        answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
        if answer_match:
            # Extract GO IDs from the answer section
            go_ids = re.findall(r'GO:\d{7}', answer_match.group(1))
            if go_ids:
                return go_ids
        return []
    
    # Extract GO IDs from boxed content
    go_terms = []
    for match in boxed_matches:
        # Find all GO IDs in the boxed content
        terms = re.findall(r'GO:\d{7}', match)
        if terms:
            go_terms.extend(terms)
        else:
            # If no GO IDs found, split by commas and clean up
            terms = [term.strip() for term in match.split(',')]
            go_terms.extend(terms)
    
    return go_terms

def evaluate_predictions(predicted: List[str], actual: List[str]) -> Dict[str, float]:
    """Calculate precision, recall, and F1 score for predictions."""
    predicted_set = set(predicted)
    actual_set = set(actual)
    
    tp = len(predicted_set.intersection(actual_set))
    fp = len(predicted_set - actual_set)
    fn = len(actual_set - predicted_set)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn
    }

def run_evaluation_local(model_name: str, checkpoint_path: str, dataset_path: str, output_path: str):
    """Run evaluation using local model with loaded adapter."""
    # Load dataset
    dataset = load_dataset('json', data_files=dataset_path)['train']
    
    # Initialize model with adapter
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=MAX_INPUT_LENGTH,
        load_in_4bit=True,
        fast_inference=True,
        max_lora_rank=32,
        gpu_memory_utilization=0.6,
    )
    
    # Load the adapter weights
    model.load_adapter(checkpoint_path)
    FastLanguageModel.for_inference(model)
    
    results = []
    total_metrics = {
        "precision": 0,
        "recall": 0,
        "f1": 0,
        "true_positives": 0,
        "false_positives": 0,
        "false_negatives": 0
    }
    
    for example in tqdm(dataset, desc="Evaluating"):
        # Extract system and user parts from the prompt
        try:
            prompt_parts = example['prompt'].split('|eot_id|>')
            system_message = prompt_parts[0].replace('<|start_header_id|>system<|end_header_id|>', '').strip()
            user_message = prompt_parts[1].split('<|eot_id|>')[0].replace('<|start_header_id|>user<|end_header_id|>', '').strip()
            
            # Get model response using local model
            inputs = tokenizer(user_message, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    temperature=0.7,
                    top_p=0.95,
                    do_sample=True,
                )
            
            response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Extract just the model's response, handling different response formats
            if '<|start_header_id|>assistant<|end_header_id|>' in response_text:
                response_text = response_text.split('<|start_header_id|>assistant<|end_header_id|>')[1].strip()
            elif '[ASSISTANT]:' in response_text:
                response_text = response_text.split('[ASSISTANT]:')[1].strip()
            elif '<|assistant|>' in response_text:
                response_text = response_text.split('<|assistant|>')[1].strip()
            
            print(f"Response for {example['protein_id']}:")
            print(response_text[:2000] + "..." if len(response_text) > 2000 else response_text)
            
            predicted_terms = parse_response(response_text)
            actual_terms = example.get('correct_choices', [])
            
            # Extract aspect from prompt if available or default to 'MFO'
            aspect = 'MFO'  # Default to MFO for multiple choice examples
            for line in user_message.split('\n'):
                if line.strip().startswith('Target Aspect:'):
                    aspect = line.split(':')[1].strip()
                    break
            
            metrics = evaluate_predictions(predicted_terms, actual_terms)
            results.append({
                "protein_id": example['protein_id'],
                "aspect": aspect,
                "predicted_terms": predicted_terms,
                "actual_terms": actual_terms,
                "metrics": metrics
            })
            
            print(f"Metrics: {metrics}")
            
            # Update total metrics
            for metric in total_metrics:
                total_metrics[metric] += metrics[metric]
                
        except Exception as e:
            print(f"Error processing example {example.get('protein_id', 'unknown')}: {str(e)}")
            continue
    
    # Calculate averages for ratio metrics
    num_examples = len(results)
    if num_examples > 0:
        for metric in ["precision", "recall", "f1"]:
            total_metrics[metric] /= num_examples
    
    # Save results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump({
            "results": results,
            "average_metrics": total_metrics
        }, f, indent=2)
    
    print("\nEvaluation Results:")
    print(f"Average Precision: {total_metrics['precision']:.4f}")
    print(f"Average Recall: {total_metrics['recall']:.4f}")
    print(f"Average F1: {total_metrics['f1']:.4f}")
    print(f"Total True Positives: {total_metrics['true_positives']}")
    print(f"Total False Positives: {total_metrics['false_positives']}")
    print(f"Total False Negatives: {total_metrics['false_negatives']}")

def run_evaluation_base_model(model_name: str, dataset_path: str, output_path: str):
    """Run evaluation using base model without adapter."""
    # Load dataset
    dataset = load_dataset('json', data_files=dataset_path)['train']
    
    # Initialize model without adapter
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=MAX_INPUT_LENGTH,
        load_in_4bit=True,
        fast_inference=True,
        gpu_memory_utilization=0.6,
    )
    
    # Set model to inference mode
    FastLanguageModel.for_inference(model)
    
    results = []
    total_metrics = {
        "precision": 0,
        "recall": 0,
        "f1": 0,
        "true_positives": 0,
        "false_positives": 0,
        "false_negatives": 0
    }
    
    for example in tqdm(dataset, desc="Evaluating Base Model"):
        # Extract system and user parts from the prompt
        try:
            prompt_parts = example['prompt'].split('|eot_id|>')
            system_message = prompt_parts[0].replace('<|start_header_id|>system<|end_header_id|>', '').strip()
            user_message = prompt_parts[1].split('<|eot_id|>')[0].replace('<|start_header_id|>user<|end_header_id|>', '').strip()
            
            # Get model response using base model
            inputs = tokenizer(user_message, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    temperature=0.7,
                    top_p=0.95,
                    do_sample=True,
                )
            
            response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Extract just the model's response, handling different response formats
            if '<|start_header_id|>assistant<|end_header_id|>' in response_text:
                response_text = response_text.split('<|start_header_id|>assistant<|end_header_id|>')[1].strip()
            elif '[ASSISTANT]:' in response_text:
                response_text = response_text.split('[ASSISTANT]:')[1].strip()
            elif '<|assistant|>' in response_text:
                response_text = response_text.split('<|assistant|>')[1].strip()
            
            print(f"Base Model Response for {example['protein_id']}:")
            print(response_text[:2000] + "..." if len(response_text) > 2000 else response_text)
            
            predicted_terms = parse_response(response_text)
            actual_terms = example.get('correct_choices', [])
            
            # Extract aspect from prompt if available or default to 'MFO'
            aspect = 'MFO'  # Default to MFO for multiple choice examples
            for line in user_message.split('\n'):
                if line.strip().startswith('Target Aspect:'):
                    aspect = line.split(':')[1].strip()
                    break
            
            metrics = evaluate_predictions(predicted_terms, actual_terms)
            results.append({
                "protein_id": example['protein_id'],
                "aspect": aspect,
                "predicted_terms": predicted_terms,
                "actual_terms": actual_terms,
                "metrics": metrics
            })
            
            print(f"Metrics: {metrics}")
            
            # Update total metrics
            for metric in total_metrics:
                total_metrics[metric] += metrics[metric]
                
        except Exception as e:
            print(f"Error processing example {example.get('protein_id', 'unknown')}: {str(e)}")
            continue
    
    # Calculate averages for ratio metrics
    num_examples = len(results)
    if num_examples > 0:
        for metric in ["precision", "recall", "f1"]:
            total_metrics[metric] /= num_examples
    
    # Save results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump({
            "results": results,
            "average_metrics": total_metrics
        }, f, indent=2)
    
    print("\nBase Model Evaluation Results:")
    print(f"Average Precision: {total_metrics['precision']:.4f}")
    print(f"Average Recall: {total_metrics['recall']:.4f}")
    print(f"Average F1: {total_metrics['f1']:.4f}")
    print(f"Total True Positives: {total_metrics['true_positives']}")
    print(f"Total False Positives: {total_metrics['false_positives']}")
    print(f"Total False Negatives: {total_metrics['false_negatives']}")

if __name__ == "__main__":
    # Configuration
    BASE_MODEL = "unsloth/meta-Llama-3.1-8B-Instruct"
    CHECKPOINT_PATH = "function-sft-checkpoint-1125/function-sft-checkpoint-1125"
    DATASET_PATH = "eval_dataset/dataset.jsonl"  # Update with your actual dataset path
    OUTPUT_PATH = "results/mc_base_eval_results.json"
    
    # Run evaluation
    # run_evaluation_local(BASE_MODEL, CHECKPOINT_PATH, DATASET_PATH, OUTPUT_PATH)

    # # Run evaluation on base model (without adapter)
    run_evaluation_base_model(BASE_MODEL, DATASET_PATH, OUTPUT_PATH)