import json
import os
import re
from typing import List, Dict
from datasets import load_dataset
from tqdm import tqdm
from dotenv import load_dotenv

# llama-3.3-70b-instruct-turbo Evaluation Results:
# Average Precision: 0.2655
# Average Recall: 0.1229
# Average F1: 0.1525
# Total True Positives: 82
# Total False Positives: 242
# Total False Negatives: 796

##gpt-4o-mini Evaluation Results:
# Average Precision: 0.2207
# Average Recall: 0.1796
# Average F1: 0.1787
# Total True Positives: 104
# Total False Positives: 346
# Total False Negatives: 774

# You can choose which API to use by uncommenting the appropriate import
import together  # For Together API
import openai   # For OpenAI API

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

def run_evaluation_together(model_name: str, dataset_path: str, output_path: str):
    """Run evaluation on the non-multiple choice dataset using Together API."""
    # Load dataset
    dataset = load_dataset('json', data_files=dataset_path)['train']
    
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
        system_message = example['prompt'].split('|eot_id|>')[0] + '|eot_id|>'
        user_message = example['prompt'].split('|eot_id|>')[1].split('<|eot_id|>')[0] + '<|eot_id|>'
        
        try:
            # Get model response using Together API
            response = together.Complete.create(
                model=model_name,
                prompt=f"{system_message}{user_message}",
                max_tokens=1024,
                temperature=0.7,
                top_p=0.7,
                top_k=50,
                repetition_penalty=1.1
            )
            
            # Parse response based on actual API structure
            if 'choices' not in response or not response['choices']:
                print(f"Response keys: {response.keys()}")
                raise ValueError(f"Response missing 'choices' field or empty choices")
            
            response_text = response['choices'][0]['text']
            print(f"Response for {example['protein_id']}:")
            print(response_text[:200] + "..." if len(response_text) > 200 else response_text)
            
            predicted_terms = parse_response(response_text)
            actual_terms = example.get('ground_truth_terms', [])
            
            metrics = evaluate_predictions(predicted_terms, actual_terms)
            results.append({
                "protein_id": example['protein_id'],
                "aspect": example['aspect'],
                "predicted_terms": predicted_terms,
                "actual_terms": actual_terms,
                "metrics": metrics
            })
            
            print(f"Metrics: {metrics}")
            
            # Update total metrics
            for metric in total_metrics:
                total_metrics[metric] += metrics[metric]
                
        except Exception as e:
            print(f"Error processing example {example['protein_id']}: {str(e)}")
            if 'response' in locals():
                print(f"Response structure: {response.keys() if hasattr(response, 'keys') else 'Not a dict'}")
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

def run_evaluation_openai(model_name: str, dataset_path: str, output_path: str, client):
    """Run evaluation on the non-multiple choice dataset using OpenAI API."""
    # Load dataset
    dataset = load_dataset('json', data_files=dataset_path)['train']
    
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
        system_message = example['prompt'].split('|eot_id|>')[0].replace('<|start_header_id|>system<|end_header_id|>', '').strip()
        user_message = example['prompt'].split('|eot_id|>')[1].replace('<|start_header_id|>user<|end_header_id|>', '').strip()
        
        try:
            # Get model response using OpenAI API
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.7,
                max_tokens=1024
            )
            
            response_text = response.choices[0].message.content
            print(f"Response for {example['protein_id']}:")
            print(response_text[:200] + "..." if len(response_text) > 200 else response_text)
            
            predicted_terms = parse_response(response_text)
            actual_terms = example.get('ground_truth_terms', [])
            
            metrics = evaluate_predictions(predicted_terms, actual_terms)
            results.append({
                "protein_id": example['protein_id'],
                "aspect": example['aspect'],
                "predicted_terms": predicted_terms,
                "actual_terms": actual_terms,
                "metrics": metrics
            })
            
            print(f"Metrics: {metrics}")
            
            # Update total metrics
            for metric in total_metrics:
                total_metrics[metric] += metrics[metric]
                
        except Exception as e:
            print(f"Error processing example {example['protein_id']}: {str(e)}")
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

if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    
    # Choose which API to use (uncomment one)
    
    # For Together API
    # together.api_key = os.getenv('TOGETHER_API_KEY')
    # MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct-Turbo"  # or your preferred Together model
    # DATASET_PATH = "non_mc_eval_dataset/dataset.jsonl"
    # OUTPUT_PATH = "eval_results/non_mc_llama-3.3-70b-instruct-turbo.json"
    # run_evaluation_together(MODEL_NAME, DATASET_PATH, OUTPUT_PATH)
    
    # For OpenAI API
    client = openai.OpenAI(api_key='')
    MODEL_NAME = "o3-mini-2025-01-31"  # or your preferred OpenAI model
    DATASET_PATH = "non_mc_eval_dataset/dataset.jsonl"
    OUTPUT_PATH = "eval_results/non_mc_o3-mini-2025-01-31.json"
    run_evaluation_openai(MODEL_NAME, DATASET_PATH, OUTPUT_PATH, client) 