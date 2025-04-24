import json
import os
from typing import List, Dict
import openai
from datasets import load_dataset
from tqdm import tqdm
import re
from dotenv import load_dotenv


def parse_response(response: str) -> List[str]:
    """Parse the model's response to extract GO terms."""
    # Extract content between \boxed{} tags
    boxed_matches = re.findall(r'\\boxed{(.*?)}', response)
    if not boxed_matches:
        return []
    
    # Split by commas and clean up whitespace
    go_terms = []
    for match in boxed_matches:
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
        "f1": f1
    }

def run_evaluation(model_name: str, dataset_path: str, output_path: str):
    """Run evaluation on the protein function prediction task."""
    # Load dataset
    dataset = load_dataset('json', data_files=dataset_path)['train']
    
    results = []
    total_metrics = {
        "precision": 0,
        "recall": 0,
        "f1": 0
    }
    for example in tqdm(dataset, desc="Evaluating"):
        system_message = example['prompt'].split('|eot_id|>')[0]
        user_message = example['prompt'].split('|eot_id|>')[1]
        try:
            # Get model response
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
            )
            
            # Parse response and evaluate
            predicted_terms = parse_response(response.choices[0].message.content)
            actual_terms = example.get('correct_choices', [])
            
            metrics = evaluate_predictions(predicted_terms, actual_terms)
            results.append({
                "protein_id": example['protein_id'],
                "predicted_terms": predicted_terms,
                "actual_terms": actual_terms,
                "metrics": metrics
            })

            print(response.choices[0].message.content)
            print(metrics)
            
            # Update total metrics
            for metric in total_metrics:
                total_metrics[metric] += metrics[metric]
                
        except Exception as e:
            print(f"Error processing example {example['protein_id']}: {str(e)}")
            continue
    
    # Calculate averages
    num_examples = len(results)
    for metric in total_metrics:
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

if __name__ == "__main__":
    # Initialize OpenAI client with API key
    load_dotenv()
    client = openai.OpenAI(api_key='')
    
    # Configuration
    MODEL_NAME = "gpt-4o-mini-2024-07-18"  # or your preferred model
    DATASET_PATH = "eval_dataset/dataset.jsonl"
    OUTPUT_PATH = "eval_results/gpt-4o-mini-2024-07-18.json"
    
    run_evaluation(MODEL_NAME, DATASET_PATH, OUTPUT_PATH)
