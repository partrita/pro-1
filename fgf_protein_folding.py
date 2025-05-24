#!/usr/bin/env python3
"""
Script to extract protein sequences with specific iteration values from adaptyv.json,
fold them using ESMFold, and save PDB files to the structures directory.

Based on the existing stability_reward.py implementation.
"""

import json
import os
import sys
import torch
from transformers import AutoTokenizer, EsmForProteinFolding
from transformers.models.esm.openfold_utils.protein import to_pdb, Protein as OFProtein
from transformers.models.esm.openfold_utils.feats import atom14_to_atom37
import time

class ProteinFolder:
    def __init__(self, model_path="facebook/esmfold_v1", device="cuda"):
        """Initialize the protein folder with ESMFold model."""
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load ESMFold model
        self.model = self._load_esmfold_model(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
        
    def _load_esmfold_model(self, model_path):
        """Load ESMFold model for structure prediction."""
        start_time = time.time()
        local_path = 'model_cache/'
        
        if os.path.exists(local_path):
            print(f"Loading cached model from {local_path}")
            model = EsmForProteinFolding.from_pretrained(local_path)
        else:
            print(f"Downloading model {model_path}")
            model = EsmForProteinFolding.from_pretrained(model_path)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            model.save_pretrained(local_path)
            print(f"Model cached to {local_path}")

        # Move model to specified device
        model = model.to(self.device)
        print(f"ESMFold model loaded in {time.time() - start_time:.2f} seconds")
        return model

    def predict_structure(self, sequence, iteration_id):
        """Predict protein structure using ESMFold."""
        print(f"Predicting structure for iteration {iteration_id}...")
        start_time = time.time()
        
        # Tokenize sequence
        tokenized_input = self.tokenizer(
            [sequence], 
            return_tensors="pt", 
            add_special_tokens=False
        )['input_ids'].to(self.device)

        # Predict structure
        with torch.no_grad():
            output = self.model(tokenized_input)

        # Convert to PDB format
        pdb_content = self.convert_outputs_to_pdb(output, iteration_id)
        
        print(f"Structure prediction completed in {time.time() - start_time:.2f} seconds")
        return pdb_content

    def convert_outputs_to_pdb(self, outputs, iteration_id):
        """Convert model outputs to PDB format."""
        final_atom_positions = atom14_to_atom37(outputs["positions"][-1], outputs)
        outputs_cpu = {k: v.to("cpu").numpy() for k, v in outputs.items()}
        final_atom_positions = final_atom_positions.cpu().numpy()
        final_atom_mask = outputs_cpu["atom37_atom_exists"]

        # Process the first (and only) sequence in the batch
        aa = outputs_cpu["aatype"][0]
        pred_pos = final_atom_positions[0]
        mask = final_atom_mask[0]
        resid = outputs_cpu["residue_index"][0] + 1
        
        # Create protein object
        pred = OFProtein(
            aatype=aa,
            atom_positions=pred_pos,
            atom_mask=mask,
            residue_index=resid,
            b_factors=outputs_cpu["plddt"][0],
            chain_index=outputs_cpu["chain_index"][0] if "chain_index" in outputs_cpu else None,
        )
        
        return to_pdb(pred)

    def save_pdb(self, pdb_content, iteration_id, output_dir):
        """Save PDB content to file."""
        os.makedirs(output_dir, exist_ok=True)
        
        pdb_filename = f"iteration_{iteration_id}.pdb"
        pdb_path = os.path.join(output_dir, pdb_filename)
        
        with open(pdb_path, "w") as f:
            f.write(pdb_content)
        
        print(f"Saved PDB file: {pdb_path}")
        return pdb_path


def load_adaptyv_data(json_file):
    """Load and parse the adaptyv.json file."""
    print(f"Loading data from {json_file}")
    with open(json_file, 'r') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} entries from adaptyv.json")
    return data


def extract_target_sequences(data, target_iterations):
    """Extract sequences for specific iteration values."""
    target_sequences = {}
    
    for entry in data:
        iteration = entry.get("iteration")
        if iteration in target_iterations:
            sequence = entry.get("sequence")
            stability_score = entry.get("stability_score")
            
            target_sequences[iteration] = {
                "sequence": sequence,
                "stability_score": stability_score
            }
            print(f"Found iteration {iteration}: {len(sequence)} amino acids, stability score: {stability_score}")
    
    # Check if we found all target iterations
    missing = set(target_iterations) - set(target_sequences.keys())
    if missing:
        print(f"Warning: Could not find sequences for iterations: {missing}")
    
    return target_sequences


def main():
    """Main function to extract sequences and fold proteins."""
    # Configuration
    target_iterations = [14, 27, 37]
    adaptyv_file = "fgf-1/adaptyv.json"
    output_dir = "fgf-1/structures"
    
    print("=== FGF-1 Protein Folding Script ===")
    print(f"Target iterations: {target_iterations}")
    print(f"Input file: {adaptyv_file}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Check if input file exists
    if not os.path.exists(adaptyv_file):
        print(f"Error: Input file {adaptyv_file} not found!")
        sys.exit(1)
    
    # Load data and extract target sequences
    data = load_adaptyv_data(adaptyv_file)
    target_sequences = extract_target_sequences(data, target_iterations)
    
    if not target_sequences:
        print("Error: No target sequences found!")
        sys.exit(1)
    
    print(f"\nSuccessfully extracted {len(target_sequences)} target sequences")
    print()
    
    # Initialize protein folder
    print("Initializing ESMFold model...")
    folder = ProteinFolder()
    print()
    
    # Process each target sequence
    for iteration in sorted(target_sequences.keys()):
        sequence_data = target_sequences[iteration]
        sequence = sequence_data["sequence"]
        stability_score = sequence_data["stability_score"]
        
        print(f"=== Processing Iteration {iteration} ===")
        print(f"Sequence length: {len(sequence)} amino acids")
        print(f"Stability score: {stability_score}")
        print(f"Sequence: {sequence[:50]}..." if len(sequence) > 50 else f"Sequence: {sequence}")
        print()
        
        try:
            # Predict structure
            pdb_content = folder.predict_structure(sequence, iteration)
            
            # Save PDB file
            pdb_path = folder.save_pdb(pdb_content, iteration, output_dir)
            
            print(f"Successfully processed iteration {iteration}")
            print()
            
        except Exception as e:
            print(f"Error processing iteration {iteration}: {str(e)}")
            print()
            continue
    
    print("=== Protein folding completed ===")
    print(f"PDB files saved to: {output_dir}")


if __name__ == "__main__":
    main() 