import os
import json
from typing import List, Dict
import openai
from Bio import SeqIO
from tqdm import tqdm
import re
from dotenv import load_dotenv
def parse_response(response: str) -> List[str]:
    """Parse the model's response to extract GO terms."""
    # Extract GO terms from {GO:XXXXXXX} format
    go_terms = re.findall(r'{GO:\d{7}}', response)
    # Remove the {} wrapper to get just the GO terms
    go_terms = [term.replace('{', '').replace('}', '') for term in go_terms]
    return go_terms

def save_predictions(all_predictions: Dict[str, List[str]], output_file: str):
    """Save predictions to both JSON and TXT files."""
    # Save all predictions to JSON file
    with open(output_file, 'w') as f:
        json.dump(all_predictions, f, indent=2)
    
    # Create the predictions.txt file in the required format
    txt_output = output_file.replace('.json', '.txt')
    with open(txt_output, 'w') as f:
        for protein_id, terms in all_predictions.items():
            for term in terms:
                f.write(f"{protein_id}\t{term}\t1\n")

def generate_predictions(test_fasta: str, output_file: str, model_name: str):
    """
    Generate predictions for protein sequences using O3.
    
    Args:
        test_fasta (str): Path to test superset FASTA file
        output_file (str): Path to output predictions file
        model_name (str): Name of the O3 model to use
    """
    # Initialize OpenAI client
    client = openai.OpenAI(api_key='')
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Store all predictions
    all_predictions: Dict[str, List[str]] = {}
    
    try:
        # Process all sequences
        for record in tqdm(SeqIO.parse(test_fasta, "fasta"), desc="Processing sequences"):
            sequence = str(record.seq)
            protein_id = record.id
            
            # Construct prompt
            system_message = """You are an expert in protein function prediction. Your task is to analyze protein sequences and predict their functions using Gene Ontology (GO) terms."""
            
            user_message = f"Given a protein sequence, reason through the protein's sequence and predict its Gene Ontology (GO) terms. Format your response with the \boxed{{GO:XXXXXXX}} tags and predict the function of this protein sequence: {sequence}"
            
            try:
                # Get model response
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": user_message}
                    ],
                )
                
                print(f"\nProcessing protein: {protein_id}")
                print("Model response:", response.choices[0].message.content)
                
                # Parse response
                predicted_terms = parse_response(response.choices[0].message.content)
                print('Predicted GO terms:', predicted_terms)
                
                # Store predictions
                all_predictions[protein_id] = predicted_terms
                
                # Save progress after each successful prediction
                save_predictions(all_predictions, output_file)
                
            except Exception as e:
                print(f"Error processing sequence {protein_id}: {str(e)}")
                all_predictions[protein_id] = []
                # Save progress even if there's an error with a sequence
                save_predictions(all_predictions, output_file)
    
    except KeyboardInterrupt:
        print("\nInterrupted by user. Saving current predictions...")
        save_predictions(all_predictions, output_file)
        print(f"Predictions saved to {output_file} and {output_file.replace('.json', '.txt')}")
        raise  # Re-raise the KeyboardInterrupt to exit the program

if __name__ == "__main__":
    load_dotenv()
    
    # Configuration
    TEST_FASTA = "function/cafa/Test/testsuperset.fasta"
    OUTPUT_FILE = "function/cafa/predictions.json"  # Changed to .json
    MODEL_NAME = "o3-2025-04-16"
    
    generate_predictions(TEST_FASTA, OUTPUT_FILE, MODEL_NAME) 