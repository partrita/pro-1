import numpy as np
from typing import List, Dict, Tuple, Set
import json
from pathlib import Path
import random
import os
from dotenv import load_dotenv
from tqdm import tqdm
from together import Together
from Bio import SeqIO
import csv
import tiktoken

load_dotenv()  # Load environment variables from .env file
client = Together(api_key=os.getenv("TOGETHER_API_KEY"))
# Initialize tokenizer for counting tokens
tokenizer = tiktoken.get_encoding("cl100k_base")  # Using OpenAI's tokenizer as approximation

def count_tokens(text: str) -> int:
    """Count the number of tokens in a text string"""
    return len(tokenizer.encode(text))

def load_go_structure(go_structure_path: str) -> Dict[str, Dict]:
    """Load GO structure and extract term information"""
    try:
        go_terms = {}
        current_term = None
        
        with open(go_structure_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line == '[Term]':
                    current_term = {}
                elif line.startswith('id:') and current_term is not None:
                    current_term['id'] = line.split('id:')[1].strip()
                elif line.startswith('name:') and current_term is not None:
                    current_term['name'] = line.split('name:')[1].strip()
                elif line.startswith('namespace:') and current_term is not None:
                    current_term['namespace'] = line.split('namespace:')[1].strip()
                elif line.startswith('def:') and current_term is not None:
                    current_term['def'] = line.split('def:')[1].strip()
                elif line == '' and current_term is not None and 'id' in current_term:
                    go_terms[current_term['id']] = current_term
                    current_term = None
        
        return go_terms
    except Exception as e:
        print(f"Error loading GO structure: {e}")
        return {}

def load_protein_annotations(train_terms_path: str) -> Dict[str, Dict[str, Dict[str, Set[str]]]]:
    """Load protein GO term annotations from tsv file"""
    try:
        protein_annotations = {
            'MFO': {'experimental': {}},
            'BPO': {'experimental': {}},
            'CCO': {'experimental': {}}
        }
        
        with open(train_terms_path, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            header = next(reader)  # Skip header row
            
            for row in reader:
                if len(row) < 3:
                    continue
                    
                protein_id, term_id, aspect = row[:3]
                protein_id = protein_id.strip()
                term_id = term_id.strip()
                aspect = aspect.strip()
                
                if aspect not in protein_annotations:
                    continue
                
                if protein_id not in protein_annotations[aspect]['experimental']:
                    protein_annotations[aspect]['experimental'][protein_id] = set()
                
                protein_annotations[aspect]['experimental'][protein_id].add(term_id)
        
        return protein_annotations
        
    except Exception as e:
        print(f"Error loading protein annotations: {e}")
        return {
            'MFO': {'experimental': {}},
            'BPO': {'experimental': {}},
            'CCO': {'experimental': {}}
        }

def load_protein_sequences(train_sequences_path: str) -> Dict[str, str]:
    """Load protein sequences from FASTA file"""
    try:
        protein_sequences = {}
        for record in SeqIO.parse(train_sequences_path, "fasta"):
            header = record.description
            if 'sp|' in header:
                uniprot_id = header.split('sp|')[1].split('|')[0]
            elif 'tr|' in header:
                uniprot_id = header.split('tr|')[1].split('|')[0]
            else:
                uniprot_id = header.split()[0]
                
            protein_sequences[uniprot_id] = str(record.seq)
        
        return protein_sequences
    except Exception as e:
        print(f"Error loading protein sequences: {e}")
        return {}

def generate_reasoning(
    protein_id: str,
    sequence: str,
    true_terms: Set[str],
    protein_annotations: Dict,
    go_terms: Dict,
    target_aspect: str
) -> str:
    """Generate reasoning for GO term prediction using Together API with DeepSeek model"""
    
    # Format known annotations for other aspects
    def format_terms(terms: Set[str], aspect_name: str) -> str:
        if not terms:
            return f"  No experimentally validated {aspect_name} terms known"
        formatted = []
        for term_id in sorted(terms):
            term_info = go_terms.get(term_id, {})
            name = term_info.get('name', 'Unknown term')
            definition = term_info.get('def', '').split('"')[1] if 'def' in term_info else ''
            formatted.append(f"  - {term_id}: {name}")
            if definition:
                formatted.append(f"    Definition: {definition}")
        return "\n".join(formatted)
    
    # Get annotations for non-target aspects
    other_aspects = [asp for asp in ['MFO', 'BPO', 'CCO'] if asp != target_aspect]
    aspect_annotations = {}
    for aspect in other_aspects:
        terms = protein_annotations.get(aspect, {}).get('experimental', {}).get(protein_id, set())
        aspect_annotations[aspect] = format_terms(terms, aspect)
    
    # Format true terms for target aspect
    formatted_true_terms = []
    for term_id in sorted(true_terms):
        term_info = go_terms.get(term_id, {})
        name = term_info.get('name', 'Unknown term')
        definition = term_info.get('def', '').split('"')[1] if 'def' in term_info else ''
        formatted_true_terms.append(f"{term_id}: {name}\nDefinition: {definition}")

    # Map aspect codes to full names
    aspect_names = {
        'MFO': 'Molecular Function',
        'BPO': 'Biological Process',
        'CCO': 'Cellular Component'
    }
    
    meta_instruction = """You are helping to create a dataset of expert reasoning for protein function prediction. For each example, you will:

1. See a prompt that asks to predict GO terms for a protein
2. See the correct GO terms that were experimentally verified
3. Generate detailed scientific reasoning as if you were solving the problem, but incorporate knowledge of the correct terms
4. Format your response with:
   - Detailed reasoning in <think> </think> tags (at least 3000 words)
   - Concise summary in <answer> </answer> tags
   - The known correct terms in \\boxed{} notation, example: \\boxed{{GO:0003674,GO:0005515, GO:0006810, etc.}}

Your goal is to create high-quality reasoning that explains WHY these GO terms are correct for this protein, citing scientific evidence and sequence analysis.

DO NOT INCLUDE ANY OTHER TEXT OTHER THAN WHAT CAN BE USED TO DIRECTLY FINETUNE A MODEL WITH OUR DATASET. DO NOT MENTION THE META INSTRUCTIONS OR TASK. RESPOND AS IF YOU WERE SOLVING THE TASK.

Here is the prompt you are responding to:

PROMPT:
"""

    prompt = f"""You are an expert in protein function prediction using the Gene Ontology (GO) framework. Your task is to explain why this protein has the following {aspect_names[target_aspect]} terms.

Protein ID: {protein_id}

PROTEIN SEQUENCE:
{sequence}

KNOWN FUNCTIONAL ANNOTATIONS:
"""

    # Add annotations for non-target aspects
    for aspect in other_aspects:
        prompt += f"\n{aspect_names[aspect]} ({aspect}) - describes {aspect_names[aspect].lower()}:\n"
        prompt += f"{aspect_annotations[aspect]}\n"

    prompt += f"""
TRUE {aspect_names[target_aspect]} TERMS:
{chr(10).join(formatted_true_terms)}

Generate detailed scientific reasoning explaining why this protein has these {aspect_names[target_aspect]} terms. Consider:
1. The protein's amino acid sequence features (length, composition, motifs)
2. Known annotations in other aspects and their implications
3. The cellular context and biological roles
4. How the sequence and structure support these functions

Your reasoning should be specific to this protein and these GO terms. Cite scientific literature where relevant. Consider similar enzymes and reactions.

Remember to format your response with:
1. Your detailed reasoning process in <think> </think> tags (aim for 3000+ words)
2. A concise summary in <answer> </answer> tags
3. The GO terms indicated by the \\boxed{{}} notation, example: \\boxed{{GO:0003674,GO:0005515, GO:0006810, etc.}}
"""

    full_prompt = meta_instruction + "\n\n" + prompt

    try:
        # Create response using Together API
        response = client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
            messages=[
                {"role": "user", "content": full_prompt}
            ],
            max_tokens=4192,
            stream=False
        )
    except Exception as e:
        print(f"Error generating response: {e}")
        return ""

    print(response.choices[0].message.content)

    return response.choices[0].message.content

def create_reasoning_dataset(
    protein_sequences: Dict[str, str],
    protein_annotations: Dict[str, Dict[str, Dict[str, Set[str]]]],
    go_terms: Dict[str, Dict],
    max_examples: int = 1000
) -> List[Dict]:
    """Create dataset of GO term predictions with reasoning"""
    dataset_records = []
    
    # Get proteins with sequences
    proteins_with_sequences = set(protein_sequences.keys())
    print(f"Found {len(proteins_with_sequences)} proteins with sequences")
    
    # Create examples for each aspect
    for aspect in ['MFO', 'BPO', 'CCO']:
        if len(dataset_records) >= max_examples:
            continue
            
        # Get proteins with experimental annotations
        proteins_with_exp = set(protein_annotations[aspect]['experimental'].keys())
        proteins_with_exp = proteins_with_exp & proteins_with_sequences
        
        print(f"\n{aspect}:")
        print(f"  Proteins with experimental annotations: {len(proteins_with_exp)}")
        
        # Create records for proteins with experimental annotations
        for i in tqdm(range(max_examples), desc=f"Generating {aspect} reasoning"):
            protein_id = list(proteins_with_exp)[i]
            if len(dataset_records) >= max_examples:
                break
                
            # Get GO terms for this protein in this aspect
            true_terms = protein_annotations[aspect]['experimental'].get(protein_id, set())
            
            # Generate reasoning
            reasoning = generate_reasoning(
                protein_id,
                protein_sequences[protein_id],
                true_terms,
                protein_annotations,
                go_terms,
                target_aspect=aspect
            )
            
            record = {
                "protein_id": protein_id,
                "aspect": aspect,
                "sequence": protein_sequences[protein_id],
                "true_terms": list(true_terms),
                "reasoning": reasoning
            }
            dataset_records.append(record)
    
    print(f"\nCreated {len(dataset_records)} total dataset records")
    return dataset_records

if __name__ == "__main__":
    # Configuration
    TRAIN_SEQUENCES_PATH = "function/cafa/Train/train_sequences.fasta"
    TRAIN_TERMS_PATH = "function/cafa/Train/train_terms.tsv"
    GO_STRUCTURE_PATH = "function/cafa/Train/go-basic.obo"
    MAX_EXAMPLES = 40000  # Adjust as needed
    
    # Load data
    print("Loading GO structure...")
    go_terms = load_go_structure(GO_STRUCTURE_PATH)
    
    print("Loading protein annotations...")
    protein_annotations = load_protein_annotations(TRAIN_TERMS_PATH)
    
    print("Loading protein sequences...")
    protein_sequences = load_protein_sequences(TRAIN_SEQUENCES_PATH)
    
    # Create dataset
    print("\nGenerating reasoning dataset...")
    dataset = create_reasoning_dataset(
        protein_sequences,
        protein_annotations,
        go_terms,
        max_examples=MAX_EXAMPLES
    )
    
    # Save dataset
    print("\nSaving dataset...")
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / "go_reasoning.json", "w") as f:
        json.dump({"records": dataset}, f, indent=2)
    
    print("Done!") 