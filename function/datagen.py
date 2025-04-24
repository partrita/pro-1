import random
import json
import os
from pathlib import Path
from typing import Dict, Set, List
from Bio import SeqIO
import csv

def get_plausible_distractors(true_terms: Set[str], go_terms: Dict, aspect: str) -> List[str]:
    """
    Select plausible but incorrect GO terms as distractors for multiple choice.
    Number of distractors will be 1-3x the number of true terms.
    
    Args:
        true_terms: Set of correct GO terms
        go_terms: Dictionary of all GO terms and their information
        aspect: The GO aspect (MFO, BPO, CCO)
    
    Returns:
        List of distractor GO terms
    """
    # Map aspect codes to namespaces
    aspect_to_namespace = {
        'MFO': 'molecular_function',
        'BPO': 'biological_process', 
        'CCO': 'cellular_component'
    }
    namespace = aspect_to_namespace[aspect]
    
    # Filter GO terms by aspect
    aspect_terms = {
        term_id for term_id, term_info in go_terms.items()
        if term_info.get('namespace', '').lower() == namespace
    }
    
    # Remove true terms from potential distractors
    potential_distractors = aspect_terms - true_terms
    
    # Randomly choose multiplier between 1 and 3
    multiplier = random.randint(1, 3)
    num_distractors = len(true_terms) * multiplier
    
    # Select distractors randomly
    distractors = random.sample(list(potential_distractors), min(num_distractors, len(potential_distractors)))
    print(distractors)
    return distractors

def load_go_terms(go_structure_path: str) -> Dict[str, Dict]:
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

def format_terms(terms: Set[str], aspect_name: str, go_terms: Dict) -> str:
    """Format GO terms for display in prompt"""
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

def create_eval_dataset(
    protein_sequences: Dict[str, str],
    protein_annotations: Dict[str, Dict[str, Dict[str, Set[str]]]],
    go_terms: Dict[str, Dict],
    max_examples: int = 100
) -> List[Dict]:
    """
    Create evaluation dataset with held-out GO terms.
    
    Args:
        protein_sequences: Dictionary mapping protein IDs to sequences
        protein_annotations: Dictionary containing protein annotations by aspect
        go_terms: Dictionary containing GO term information
        max_examples: Maximum number of examples to include
    """
    eval_records = []
    
    # Get proteins with sequences
    proteins_with_sequences = set(protein_sequences.keys())
    print(f"Found {len(proteins_with_sequences)} proteins with sequences")
    
    # Create examples for each aspect
    for aspect in ['MFO', 'BPO', 'CCO']:
        if len(eval_records) >= max_examples:
            break
            
        # Get proteins with experimental annotations for this aspect
        proteins_with_exp = set(protein_annotations[aspect]['experimental'].keys())
        proteins_without_exp = proteins_with_sequences - proteins_with_exp
        
        # Select proteins for evaluation
        eval_proteins = random.sample(list(proteins_with_exp), min(max_examples - len(eval_records), len(proteins_with_exp)))
        
        for protein_id in eval_proteins:
            # Get sequence
            sequence = protein_sequences[protein_id]
            
            # Get true terms for this protein and aspect
            true_terms = protein_annotations[aspect]['experimental'].get(protein_id, set())
            
            # Skip if no true terms
            if not true_terms:
                continue
                
            # Get distractor terms
            distractors = get_plausible_distractors(true_terms, go_terms, aspect)
            
            # Combine true terms and distractors
            all_choices = list(true_terms) + distractors
            
            # Shuffle all choices
            random.shuffle(all_choices)
            
            # Format choices with descriptions
            formatted_choices = []
            for i, term_id in enumerate(all_choices):
                term_info = go_terms.get(term_id, {})
                name = term_info.get('name', 'Unknown term')
                formatted_choices.append(f"Option {chr(65+i)}. {term_id}: {name}")
            
            # Map aspect codes to full names
            aspect_names = {
                'MFO': 'Molecular Function',
                'BPO': 'Biological Process',
                'CCO': 'Cellular Component'
            }
            
            # Get protein description
            protein_description = f"Protein ID: {protein_id}"
            
            # Format known annotations for non-target aspects
            other_aspects = [asp for asp in ['MFO', 'BPO', 'CCO'] if asp != aspect]
            aspect_annotations = {}
            for asp in other_aspects:
                terms = protein_annotations.get(asp, {}).get('experimental', {}).get(protein_id, set())
                aspect_annotations[asp] = format_terms(terms, asp, go_terms)
            
            # Construct prompt
            go_prompt = f"""You are an expert in protein function prediction using the Gene Ontology (GO) framework. Your task is to select the correct {aspect_names[aspect]} terms for this protein from the given options. There are multiple correct answers, you must select all of them.

Target Aspect: {aspect}

{protein_description}

PROTEIN SEQUENCE:
{sequence}

KNOWN FUNCTIONAL ANNOTATIONS:
"""

            # Add annotations for non-target aspects
            for asp in other_aspects:
                go_prompt += f"\n{aspect_names[asp]} ({asp}) - describes {aspect_names[asp].lower()}:\n"
                go_prompt += f"{aspect_annotations[asp]}\n"

            go_prompt += f"""
CANDIDATE GO TERMS:
{chr(10).join(formatted_choices)}

Based on:
1. The protein's amino acid sequence
2. Known annotations in other aspects
3. Your knowledge of protein domains, motifs, and cellular organization

Select the correct GO term(s) for this protein. Consider:
- A protein can have multiple functions
- Only select terms that you are confident are supported by the sequence or known annotations
- Explain your reasoning for each selected and rejected term

Your final answer should be the letter(s) of the correct option(s) enclosed in \\boxed{{}} notation.
Example: \\boxed{{A,C}} for selecting options A and C.

First, think through your reasoning process considering:
1. Sequence features (length, composition, motifs)
2. Known annotations and their implications
3. Cellular context and biological roles
4. How each option relates to the protein's likely function"""

            whole_prompt = f"""<|start_header_id|>system<|end_header_id|>
You are a helpful assistant that helps users with protein function prediction. You first think about the reasoning process and then provide the answer. The reasoning process and answer should be enclosed within <think> </think> and <answer> </answer> tags respectively. Your thinking should be at least 3000 words. |eot_id|><|start_header_id|>user<|end_header_id|>
{go_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

            # Create record
            record = {
                "prompt": whole_prompt,
                "protein_id": protein_id,
                "sequence": sequence,
                "ground_truth_terms": list(true_terms),
                "choices": {chr(65+i): term_id for i, term_id in enumerate(all_choices)}
            }
            
            eval_records.append(record)
    
    return eval_records

def save_eval_dataset(eval_records: List[Dict], output_dir: str = "eval_dataset"):
    """
    Save evaluation dataset to a single JSONL file.
    
    Args:
        eval_records: List of evaluation records
        output_dir: Directory to save the dataset
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save all data to a single file
    dataset_file = os.path.join(output_dir, "dataset.jsonl")
    with open(dataset_file, 'w') as f:
        for record in eval_records:
            # Combine all information in a single record
            output_record = {
                "prompt": record["prompt"],
                "protein_id": record["protein_id"],
                "sequence": record["sequence"],
                "choices": record["choices"],
                "ground_truth_terms": record["ground_truth_terms"],
                "correct_choices": [k for k, v in record["choices"].items() 
                                  if v in record["ground_truth_terms"]]
            }
            json.dump(output_record, f)
            f.write('\n')
    
    print(f"Saved {len(eval_records)} evaluation examples to {dataset_file}")

if __name__ == "__main__":
    # Load required data
    print("Loading protein sequences...")
    protein_sequences = load_protein_sequences("function/cafa/Train/train_sequences.fasta")
    
    print("Loading GO terms...")
    go_terms = load_go_terms("function/cafa/Train/go-basic.obo")
    
    print("Loading protein annotations...")
    protein_annotations = load_protein_annotations("function/cafa/Train/train_terms.tsv")

    
    # Generate evaluation dataset
    print("\nGenerating evaluation dataset...")
    eval_records = create_eval_dataset(
        protein_sequences=protein_sequences,
        protein_annotations=protein_annotations,
        go_terms=go_terms,
        max_examples=100  # Adjust this number as needed
    )
    
    # Save the dataset
    save_eval_dataset(eval_records)
    
    print(f"\nGenerated {len(eval_records)} evaluation examples")