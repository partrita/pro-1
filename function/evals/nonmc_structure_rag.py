import json
import os
from typing import List, Dict
import openai
import together
from datasets import load_dataset
from tqdm import tqdm
import re
from dotenv import load_dotenv
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from structure_similarity_search import StructureSimilaritySearch

# Evaluation Results:
# Average Precision: 0.9461
# Average Recall: 0.6174
# Average F1: 0.7233

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

def format_similar_structures_prompt(similar_structures) -> str:
    """Format similar structures and their GO terms into a prompt."""
    prompt_parts = ["\nHere are some proteins with similar structures and their known GO terms:"]
    
    for protein_id, go_terms, similarity, seq_sim in similar_structures:
        prompt_parts.append(f"\nProtein {protein_id} (Structural Similarity: {similarity:.3f}, Sequence Similarity: {seq_sim:.3f})")
        # for aspect, terms in go_terms.items():
        #     if terms:  # Only include aspects that have terms
        #         prompt_parts.append(f"{aspect}: {', '.join(terms)}")
    
    return "\n".join(prompt_parts)

def run_evaluation_openai_with_rag(model_name: str, dataset_path: str, output_path: str, client):
    """Run evaluation on the non-multiple choice dataset using OpenAI API with structure similarity RAG."""
    # Initialize structure similarity search
    searcher = StructureSimilaritySearch(
        train_sequences_path="function/cafa/Train/train_sequences.fasta",
        train_terms_path="function/cafa/Train/train_terms.tsv",
        cache_dir="cache"  # Specify the cache directory where the index exists
    )
    print("Loading structural similarity index from cache...")
    searcher.build_index()
    
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
            print(f"\nProcessing protein: {example['protein_id']}")
            
            # Get similar structures and their GO terms if sequence is available
            if 'sequence' in example:
                print(f"Sequence length: {len(example['sequence'])}")
                try:
                    similar_structures = searcher.find_similar_structures(
                        query_sequence=example['sequence'],
                        k=5  # Get top 5 similar structures
                    )
                    print(f"Found {len(similar_structures)} similar structures")
                    
                    # Add similar structures information to the prompt
                    similar_structures_prompt = format_similar_structures_prompt(similar_structures)
                    enhanced_user_message = user_message + similar_structures_prompt
                except Exception as e:
                    print(f"Error in structure similarity search: {str(e)}")
                    print(f"Error line: {e.__traceback__.tb_lineno}")
                    enhanced_user_message = user_message
            else:
                print("No sequence available for this example, skipping similarity search")
                enhanced_user_message = user_message
            
            # Get model response using OpenAI API
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": enhanced_user_message}
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
            result = {
                "protein_id": example['protein_id'],
                "aspect": example.get('aspect', ''),
                "predicted_terms": predicted_terms,
                "actual_terms": actual_terms,
                "metrics": metrics
            }
            
            # Add similar structures information to result if available
            if 'sequence' in example and 'similar_structures' in locals():
                result["similar_structures"] = [
                    {
                        "protein_id": pid,
                        "similarity": sim,
                        "sequence_similarity": seq_sim,
                        "go_terms": {k: list(v) for k, v in terms.items()}
                    }
                    for pid, terms, sim, seq_sim in similar_structures
                ]
            
            results.append(result)
            
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

def run_evaluation_together_with_rag(model_name: str, dataset_path: str, output_path: str):
    """Run evaluation on the non-multiple choice dataset using Together API with structure similarity RAG."""
    # Initialize structure similarity search
    searcher = StructureSimilaritySearch(
        train_sequences_path="function/cafa/Train/train_sequences.fasta",
        train_terms_path="function/cafa/Train/train_terms.tsv",
        cache_dir="cache"  # Specify the cache directory where the index exists
    )
    print("Loading structural similarity index from cache...")
    searcher.build_index()
    
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
        user_message = example['prompt'].split('|eot_id|>')[1].split('<|eot_id|>')[0]
        
        try:
            print(f"\nProcessing protein: {example['protein_id']}")
            
            # Get similar structures and their GO terms if sequence is available
            if 'sequence' in example:
                print(f"Sequence length: {len(example['sequence'])}")
                try:
                    similar_structures = searcher.find_similar_structures(
                        query_sequence=example['sequence'],
                        k=5  # Get top 5 similar structures
                    )
                    print(f"Found {len(similar_structures)} similar structures")
                    
                    # Add similar structures information to the prompt
                    similar_structures_prompt = format_similar_structures_prompt(similar_structures)
                    enhanced_user_message = user_message + similar_structures_prompt + '<|eot_id|>'
                except Exception as e:
                    print(f"Error in structure similarity search: {str(e)}")
                    print(f"Error line: {e.__traceback__.tb_lineno}")
                    enhanced_user_message = user_message + '<|eot_id|>'
            else:
                print("No sequence available for this example, skipping similarity search")
                enhanced_user_message = user_message + '<|eot_id|>'
            
            # Get model response using Together API
            response = together.Complete.create(
                model=model_name,
                prompt=f"{system_message}{enhanced_user_message}",
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
            result = {
                "protein_id": example['protein_id'],
                "aspect": example.get('aspect', ''),
                "predicted_terms": predicted_terms,
                "actual_terms": actual_terms,
                "metrics": metrics
            }
            
            # Add similar structures information to result if available
            if 'sequence' in example and 'similar_structures' in locals():
                result["similar_structures"] = [
                    {
                        "protein_id": pid,
                        "similarity": sim,
                        "sequence_similarity": seq_sim,
                        "go_terms": {k: list(v) for k, v in terms.items()}
                    }
                    for pid, terms, sim, seq_sim in similar_structures
                ]
            
            results.append(result)
            
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

if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    
    # Choose which API to use (uncomment one)
    
    # For Together API
    # together.api_key = os.getenv('TOGETHER_API_KEY')
    # MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct-Turbo"  # or your preferred Together model
    # DATASET_PATH = "non_mc_eval_dataset/dataset.jsonl"
    # OUTPUT_PATH = "eval_results/non_mc_llama-3.3-70b-instruct-turbo_with_rag.json"
    # run_evaluation_together_with_rag(MODEL_NAME, DATASET_PATH, OUTPUT_PATH)
    
    # For OpenAI API
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    MODEL_NAME = "gpt-4o-mini-2024-07-18"  # or your preferred OpenAI model
    DATASET_PATH = "non_mc_eval_dataset/dataset.jsonl"
    OUTPUT_PATH = "eval_results/non_mc_gpt4o-mini_with_rag.json"
    run_evaluation_openai_with_rag(MODEL_NAME, DATASET_PATH, OUTPUT_PATH, client) 