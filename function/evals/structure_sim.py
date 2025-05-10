import json
import os
from typing import List, Dict
import openai
from datasets import load_dataset
from tqdm import tqdm
import re
from dotenv import load_dotenv
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from structure_similarity_search import StructureSimilaritySearch


# Evaluation Results:
# Average Precision: 0.9451
# Average Recall: 0.6259
# Average F1: 0.7270

# Evaluation Results:
# Average Precision: 0.9267
# Average Recall: 0.6265
# Average F1: 0.7210

# Evaluation Results:
# Average Precision: 0.9444
# Average Recall: 0.6006
# Average F1: 0.7060

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

def format_similar_structures_prompt(similar_structures) -> str:
    """Format similar structures and their GO terms into a prompt."""
    prompt_parts = ["\nHere are some proteins with similar structures and their known GO terms:"]
    
    for protein_id, go_terms, similarity, seq_sim in similar_structures:
        prompt_parts.append(f"\nProtein {protein_id} (Structural Similarity: {similarity:.3f}, Sequence Similarity: {seq_sim:.3f})")
        for aspect, terms in go_terms.items():
            if terms:  # Only include aspects that have terms
                prompt_parts.append(f"{aspect}: {', '.join(terms)}")
    
    return "\n".join(prompt_parts)

def run_evaluation(model_name: str, dataset_path: str, output_path: str):
    """Run evaluation on the protein function prediction task with structural similarity RAG."""
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
        "f1": 0
    }
    
    for example in tqdm(dataset, desc="Evaluating"):
        system_message = example['prompt'].split('|eot_id|>')[0]
        user_message = example['prompt'].split('|eot_id|>')[1]
        
        try:
            print(f"\nProcessing protein: {example['protein_id']}")
            print(f"Sequence length: {len(example['sequence'])}")
            
            # Get similar structures and their GO terms
            try:
                similar_structures = searcher.find_similar_structures(
                    query_sequence=example['sequence'],
                    k=5  # Get top 5 similar structures
                )
                print(f"Found {len(similar_structures)} similar structures")
            except Exception as e:
                print(f"Error in structure similarity search: {str(e)}")
                print(f"Error line: {e.__traceback__.tb_lineno}")
                raise
            
            # Add similar structures information to the prompt
            similar_structures_prompt = format_similar_structures_prompt(similar_structures)
            enhanced_user_message = user_message + similar_structures_prompt
            
            print("Sending request to OpenAI API...")
            print(enhanced_user_message)
            try:
                # Get model response
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": enhanced_user_message}
                    ],
                )
                print("Successfully received API response")
            except Exception as e:
                print(f"Error in OpenAI API call: {str(e)}")
                print(f"Error line: {e.__traceback__.tb_lineno}")
                print("System message length:", len(system_message))
                print("User message length:", len(enhanced_user_message))
                raise
            
            # Parse response and evaluate
            predicted_terms = parse_response(response.choices[0].message.content)
            actual_terms = example.get('correct_choices', [])
            
            metrics = evaluate_predictions(predicted_terms, actual_terms)
            results.append({
                "protein_id": example['protein_id'],
                "predicted_terms": predicted_terms,
                "actual_terms": actual_terms,
                "metrics": metrics,
                "similar_structures": [
                    {
                        "protein_id": pid,
                        "similarity": sim,
                        "sequence_similarity": seq_sim,
                        "go_terms": {k: list(v) for k, v in terms.items()}
                    }
                    for pid, terms, sim, seq_sim in similar_structures
                ]
            })

            print(f"\nProtein: {example['protein_id']}")
            print("Model Response:", response.choices[0].message.content)
            print("Metrics:", metrics)
            
            # Update total metrics
            for metric in total_metrics:
                total_metrics[metric] += metrics[metric]
                
        except Exception as e:
            print(f"Error processing example {example['protein_id']}: {str(e)}")
            print(f"Error type: {type(e).__name__}")
            print(f"Error line: {e.__traceback__.tb_lineno}")
            if hasattr(e, 'response'):
                print(f"Response status: {e.response.status_code if hasattr(e.response, 'status_code') else 'N/A'}")
                print(f"Response content: {e.response.text if hasattr(e.response, 'text') else 'N/A'}")
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
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Configuration
    MODEL_NAME = "gpt-4o-mini-2024-07-18"  # or your preferred model
    DATASET_PATH = "non_mc_eval_dataset/dataset.jsonl"
    OUTPUT_PATH = "eval_results/nonMC-gpt-4o-mini-2024-07-18_with_structure_rag.json"
    
    run_evaluation(MODEL_NAME, DATASET_PATH, OUTPUT_PATH)
