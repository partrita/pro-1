import torch
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import Dict, List, Tuple
import re

class GOTermEmbedder:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """Initialize the embedder with a sentence transformer model."""
        self.model = SentenceTransformer(model_name)
        self.embeddings: Dict[str, np.ndarray] = {}
        
    def parse_obo_file(self, file_path: str) -> Dict[str, str]:
        """Parse the GO term obo file and extract term IDs and their descriptions."""
        terms = {}
        current_id = None
        current_def = None
        
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('id: '):
                    current_id = line[4:].strip()
                elif line.startswith('def: '):
                    # Extract the definition text between quotes
                    match = re.search(r'"([^"]*)"', line)
                    if match:
                        current_def = match.group(1)
                        if current_id:
                            terms[current_id] = current_def
                            current_id = None
                            current_def = None
        return terms
    
    def compute_embeddings(self, terms: Dict[str, str]) -> None:
        """Compute embeddings for all term definitions."""
        import os
        import pickle
        from tqdm import tqdm
        
        # Define cache file path
        cache_file = "go_term_embeddings.pkl"
        
        # Try to load from cache if exists
        if os.path.exists(cache_file):
            print("Loading embeddings from cache...")
            with open(cache_file, 'rb') as f:
                self.embeddings = pickle.load(f)
            return
            
        # Compute embeddings if not cached
        print("Computing embeddings (this may take a while)...")
        for term_id, definition in tqdm(terms.items(), desc="Computing embeddings"):
            embedding = self.model.encode(definition)
            self.embeddings[term_id] = embedding
            
        # Save to cache
        print("Saving embeddings to cache...")
        with open(cache_file, 'wb') as f:
            pickle.dump(self.embeddings, f)
            
    def load_from_obo(self, file_path: str) -> None:
        """Load and embed all terms from an OBO file."""
        terms = self.parse_obo_file(file_path)
        self.compute_embeddings(terms)
    
    def compute_reward(self, pred_go_term: str, true_go_term: str, scale: float = 10.0) -> float:
        """
        Compute a continuous reward based on embedding similarity.
        Returns a value between 0 and 1, where 1 means perfect match.
        """
        if pred_go_term not in self.embeddings or true_go_term not in self.embeddings:
            return 0.0
            
        pred_embedding = self.embeddings[pred_go_term]
        true_embedding = self.embeddings[true_go_term]
        
        # Compute cosine similarity
        similarity = np.dot(pred_embedding, true_embedding) / \
                    (np.linalg.norm(pred_embedding) * np.linalg.norm(true_embedding))
        
        # Transform similarity to [0, 1] range
        reward = (similarity + 1) / 2
        return reward

def create_embedder(obo_file_path: str) -> GOTermEmbedder:
    """Helper function to create and initialize an embedder."""
    embedder = GOTermEmbedder()
    embedder.load_from_obo(obo_file_path)
    print(f"Loaded {len(embedder.embeddings)} term embeddings")
    return embedder

if __name__ == "__main__":
    # Example usage
    obo_file = "function/cafa/Train/go-basic.obo"
    embedder = create_embedder(obo_file)
    
    # Example of computing reward between two GO terms
    go_term1 = "GO:0000001"
    go_term2 = "GO:0000002"
    reward = embedder.compute_reward(go_term1, go_term2)
    print(f"Reward between {go_term1} and {go_term2}: {reward:.4f}")
