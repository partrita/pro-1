import os
from Bio import SeqIO

def create_submission_file(predictions_file, output_file):
    """
    Create a submission file in CAFA format from predictions.
    
    Args:
        predictions_file (str): Path to file containing predictions
        output_file (str): Path to output submission file
    """
    # Read test sequences
    test_seqs = []
    with open("function/cafa/testsuperset.fasta", "r") as f:
        for record in SeqIO.parse(f, "fasta"):
            test_seqs.append(record.id)
    
    # Read predictions
    with open(predictions_file, "r") as f:
        predictions = f.readlines()
    
    # Write submission file
    with open(output_file, "w") as f:
        for seq_id in test_seqs:
            for pred in predictions:
                pred = pred.strip()
                if pred.startswith("GO:"):
                    f.write(f"{seq_id}\t{pred}\t1.0\n")

if __name__ == "__main__":
    predictions_file = "function/cafa/predictions.txt"  # Path to your predictions file
    output_file = "function/cafa/submission.tsv"
    create_submission_file(predictions_file, output_file)
