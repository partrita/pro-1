import torch
import esm
import biotite.structure as struc
import biotite.structure.io.pdb as pdb
from Bio import SeqIO
from typing import Dict, List, Tuple, Optional
import numpy as np
from pathlib import Path
import faiss
import pickle
import os
from tqdm import tqdm

class StructureSimilaritySearch:
    def __init__(self, train_sequences_path: str, train_terms_path: str, cache_dir: str = "cache"):
        """
        Initialize the structure similarity search system.
        
        Args:
            train_sequences_path: Path to the training sequences FASTA file
            train_terms_path: Path to the training GO terms TSV file
            cache_dir: Directory to store cached embeddings and index
        """
        self.train_sequences_path = train_sequences_path
        self.train_terms_path = train_terms_path
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load ESM-3 model for structure prediction
        self.model, self.alphabet = esm.pretrained.esm2_t36_3B_UR50D()
        self.model = self.model.eval().cuda()
        self.batch_converter = self.alphabet.get_batch_converter()
        
        # Initialize FAISS index for similarity search
        self.index = None
        self.protein_ids = []
        self.go_terms = {}
        
        # Load training data
        self._load_training_data()
        
    def _load_training_data(self):
        """Load training sequences and their GO terms"""
        # Load sequences
        self.train_sequences = {}
        for record in SeqIO.parse(self.train_sequences_path, "fasta"):
            if 'sp|' in record.description:
                uniprot_id = record.description.split('sp|')[1].split('|')[0]
            elif 'tr|' in record.description:
                uniprot_id = record.description.split('tr|')[1].split('|')[0]
            else:
                uniprot_id = record.description.split()[0]
            self.train_sequences[uniprot_id] = str(record.seq)
            
        # Load GO terms
        self.train_terms = {}
        with open(self.train_terms_path, 'r') as f:
            next(f)  # Skip header
            for line in f:
                protein_id, term_id, aspect = line.strip().split('\t')
                if protein_id not in self.train_terms:
                    self.train_terms[protein_id] = {'MFO': set(), 'BPO': set(), 'CCO': set()}
                self.train_terms[protein_id][aspect].add(term_id)
    
    def generate_structure_embedding(self, sequence: str) -> np.ndarray:
        """
        Generate structure embedding for a protein sequence using ESM-3.
        
        Args:
            sequence: Protein sequence as string
            
        Returns:
            numpy.ndarray: Structure embedding
        """
        # Prepare input
        data = [("protein", sequence)]
        batch_labels, batch_strs, batch_tokens = self.batch_converter(data)
        
        # Generate embedding
        with torch.no_grad():
            results = self.model(batch_tokens.cuda(), repr_layers=[36])
            token_representations = results["representations"][36]
        
        # Average pool over sequence length
        sequence_embedding = token_representations.mean(dim=1)
        return sequence_embedding.cpu().numpy()
    
    def build_index(self, force_rebuild: bool = False):
        """
        Build FAISS index for fast similarity search.
        
        Args:
            force_rebuild: If True, rebuild index even if cache exists
        """
        cache_file = self.cache_dir / "structure_index.pkl"
        
        if not force_rebuild and cache_file.exists():
            print("Loading cached index...")
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
                self.index = cached_data['index']
                self.protein_ids = cached_data['protein_ids']
            return
        
        print("Building structure similarity index...")
        embeddings = []
        self.protein_ids = []
        
        # Generate embeddings for all training sequences
        for protein_id, sequence in tqdm(self.train_sequences.items()):
            embedding = self.generate_structure_embedding(sequence)
            embeddings.append(embedding)
            self.protein_ids.append(protein_id)
            
        # Convert to numpy array
        embeddings = np.vstack(embeddings)
        
        # Build FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
        
        # Cache the index
        with open(cache_file, 'wb') as f:
            pickle.dump({
                'index': self.index,
                'protein_ids': self.protein_ids
            }, f)
    
    def find_similar_structures(self, 
                              query_sequence: str, 
                              k: int = 5) -> List[Tuple[str, Dict[str, set], float]]:
        """
        Find proteins with similar structures and their GO terms.
        
        Args:
            query_sequence: Query protein sequence
            k: Number of similar structures to return
            
        Returns:
            List of tuples (protein_id, GO_terms, similarity_score)
        """
        # Generate embedding for query sequence
        query_embedding = self.generate_structure_embedding(query_sequence)
        
        # Search in FAISS index
        distances, indices = self.index.search(query_embedding, k)
        
        # Get results
        results = []
        for i, idx in enumerate(indices[0]):
            protein_id = self.protein_ids[idx]
            go_terms = self.train_terms[protein_id]
            similarity_score = 1.0 / (1.0 + distances[0][i])  # Convert distance to similarity
            results.append((protein_id, go_terms, similarity_score))
            
        return results

def main():
    # Example usage
    searcher = StructureSimilaritySearch(
        train_sequences_path="function/cafa/Train/train_sequences.fasta",
        train_terms_path="function/cafa/Train/train_terms.tsv"
    )
    
    # Build index (only needed once)
    searcher.build_index()
    
    # Example query sequence
    query_sequence = "MSLLTEVETPIRNEWECRCSGLPEWLQAEPMQTRGLFGAIAGFIENGWEGMVDGWYGFRHQNSEGRGQAADLKSTQAAIDQINGKLNRLIGKTNEKFHQIEKEFSEVEGRIQDLEKYVEDTKIDLWSYNAELLVALENQHTIDLTDSEMNKLFEKTKKQLRENAEDMGNGCFKIYHKCDNACIGSIRNGTYDHDVYRDEALNNRFQIKGVELKSGYKDWILWISFAISCFLLCVALLGFIMWACQKGNIRCNICI"
    
    # Find similar structures
    results = searcher.find_similar_structures(query_sequence, k=5)
    
    # Print results
    print("\nMost similar structures and their GO terms:")
    for protein_id, go_terms, similarity in results:
        print(f"\nProtein: {protein_id} (Similarity: {similarity:.3f})")
        print("GO Terms:")
        for aspect, terms in go_terms.items():
            print(f"  {aspect}: {', '.join(terms)}")

if __name__ == "__main__":
    main() 