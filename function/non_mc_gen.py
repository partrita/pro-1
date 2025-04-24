import random
import json
import os
from pathlib import Path
from typing import Dict, Set, List
from Bio import SeqIO
import csv

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

def create_non_mc_eval_dataset(
    protein_sequences: Dict[str, str],
    protein_annotations: Dict[str, Dict[str, Dict[str, Set[str]]]],
    go_terms: Dict[str, Dict],
    max_examples: int = 100
) -> List[Dict]:
    """
    Create evaluation dataset without multiple choice framing.
    
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
        
        # Filter to proteins that have at least one annotation for the target aspect
        valid_proteins = [p for p in proteins_with_exp if len(protein_annotations[aspect]['experimental'].get(p, set())) > 0]
        
        # Select proteins for evaluation
        eval_proteins = random.sample(valid_proteins, min(max_examples - len(eval_records), len(valid_proteins)))
        
        for protein_id in eval_proteins:
            # Get sequence
            sequence = protein_sequences[protein_id]
            
            # Get true terms for this protein and aspect
            true_terms = protein_annotations[aspect]['experimental'].get(protein_id, set())
            
            # Skip if no true terms
            if not true_terms:
                continue
            
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
            
            # Create non-multiple choice prompt
            go_prompt = f"""You are an expert in protein function prediction using the Gene Ontology (GO) framework. Your task is to predict the correct {aspect_names[aspect]} GO terms for this protein.

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
Based on:
1. The protein's amino acid sequence
2. Known annotations in other aspects
3. Your knowledge of protein domains, motifs, and cellular organization

Predict GO terms for this protein within the {aspect_names[aspect]} ({aspect}) aspect. Consider:
- A protein typically has multiple functions
- Only predict terms that you are confident are supported by the sequence or known annotations
- Explain your reasoning for each predicted term
- Provide your answer as GO term IDs (e.g., GO:0046872)

Your final answer should be the full list of predicted GO term IDs enclosed in \\boxed{{}} notation.
Example: \\boxed{{GO:0046872, GO:0005515}}

First, think through your reasoning process considering:
1. Sequence features (length, composition, motifs)
2. Known annotations and their implications
3. Cellular context and biological roles
4. Potential molecular functions based on the sequence and cellular context"""

            whole_prompt = f"""<|start_header_id|>system<|end_header_id|>
You are a helpful assistant that helps users with protein function prediction. You first think about the reasoning process and then provide the answer. The reasoning process and answer should be enclosed within <think> </think> and <answer> </answer> tags respectively. Your thinking should be at least 3000 words. |eot_id|><|start_header_id|>user<|end_header_id|>
{go_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

            # Create record
            record = {
                "prompt": whole_prompt,
                "protein_id": protein_id,
                "sequence": sequence,
                "ground_truth_terms": list(true_terms),
                "aspect": aspect
            }
            
            eval_records.append(record)
    
    return eval_records

def save_non_mc_eval_dataset(eval_records: List[Dict], output_dir: str = "non_mc_eval_dataset"):
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
                "ground_truth_terms": record["ground_truth_terms"],
                "aspect": record["aspect"]
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

    
    # Generate non-multiple choice evaluation dataset
    print("\nGenerating non-multiple choice evaluation dataset...")
    eval_records = create_non_mc_eval_dataset(
        protein_sequences=protein_sequences,
        protein_annotations=protein_annotations,
        go_terms=go_terms,
        max_examples=100  # Adjust this number as needed
    )
    
    # Save the dataset
    save_non_mc_eval_dataset(eval_records)
    
    print(f"\nGenerated {len(eval_records)} non-multiple choice evaluation examples")
