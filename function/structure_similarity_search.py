import torch
import esm
import biotite.structure as struc
import biotite.structure.io.pdb as pdb
from Bio import SeqIO
from typing import Dict, List, Tuple, Optional
import numpy as np
from pathlib import Path
import pickle
import os
from tqdm import tqdm
from huggingface_hub import login
from esm.models.esm3 import ESM3
from esm.sdk.api import ESM3InferenceClient, ESMProtein, LogitsConfig, GenerationConfig
from dotenv import load_dotenv

load_dotenv()

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
        
        # Load ESM3 model for protein embedding
        try:
            # Try to load ESM3-open model
            huggingface_token = os.getenv('HUGGINGFACE_API_KEY')
            if not huggingface_token:
                raise ValueError("HUGGINGFACE_API_KEY not found in environment variables")
            login(token=huggingface_token)
            print("Successfully logged into Hugging Face")
            self.model = ESM3.from_pretrained("esm3-open").to("cuda")
        except Exception as e:
            print(f"Error loading ESM3 from HuggingFace: {e}")
            # Fallback to ESMC open model if available
            try:
                from esm.pretrained import ESMC_300M_202412
                self.model = ESMC_300M_202412().to("cuda")
            except:
                raise Exception("Could not load any ESM3 or ESMC model. Make sure to install and configure the esm package correctly.")
        
        # Initialize embedding storage
        self.embeddings = None
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
        Generate structure-specific embedding for a protein sequence using ESM3's structure track.
        
        Args:
            sequence: Protein sequence as string
            
        Returns:
            numpy.ndarray: Structure-specific embedding
        """ 
        # Create ESMProtein object from sequence
        protein = ESMProtein(sequence=sequence)
        
        # Generate the structure if it doesn't exist
        if protein.coordinates is None:
            # This will predict the structure for the sequence
            protein = self.model.generate(protein, 
                                         GenerationConfig(track="structure", 
                                                         num_steps=8))
        
        # Now encode the protein WITH the generated structure
        protein_tensor = self.model.encode(protein)        
        
        # Get structure-specific embeddings
        structure_output = self.model.logits(
            protein_tensor,
            LogitsConfig(structure=True, return_embeddings=True)
        )
        # Return the structure-specific embeddings
        return structure_output.embeddings.cpu().numpy()
    
    def generate_structure_embeddings_batch(self, sequences: List[str], batch_size: int = 4) -> List[np.ndarray]:
        """
        Generate structure-specific embeddings for multiple protein sequences in batches.
        
        Args:
            sequences: List of protein sequences as strings
            batch_size: Number of sequences to process in each batch
            
        Returns:
            List of numpy.ndarray: Structure-specific embeddings for each sequence
        """
        all_embeddings = []
        
        # Process sequences in batches
        for i in range(0, len(sequences), batch_size):
            batch_sequences = sequences[i:i+batch_size]
            batch_proteins = []
            
            # Create ESMProtein objects for each sequence in the batch
            for sequence in batch_sequences:
                protein = ESMProtein(sequence=sequence)
                batch_proteins.append(protein)
            
            # Generate structures for all proteins in the batch that don't have coordinates
            batch_proteins_with_structure = []
            for protein in batch_proteins:
                if protein.coordinates is None:
                    protein = self.model.generate(protein,
                                                 GenerationConfig(track="structure",
                                                                 num_steps=8))
                batch_proteins_with_structure.append(protein)
            
            # Process the batch with a simple loop, avoiding potential memory issues
            batch_embeddings = []
            for protein in batch_proteins_with_structure:
                # Encode the protein with the structure
                protein_tensor = self.model.encode(protein)
                
                # Get structure-specific embeddings
                structure_output = self.model.logits(
                    protein_tensor,
                    LogitsConfig(structure=True, return_embeddings=True)
                )
                
                # Add to batch embeddings
                batch_embeddings.append(structure_output.embeddings.cpu().numpy())
            
            # Add all batch embeddings to the results
            all_embeddings.extend(batch_embeddings)
            
            print(f"Processed batch {i//batch_size + 1}/{(len(sequences) + batch_size - 1)//batch_size}")
            
        return all_embeddings
    
    def build_index(self, force_rebuild: bool = False, batch_size: int = 4):
        """
        Build embeddings matrix for similarity search.
        
        Args:
            force_rebuild: If True, rebuild embeddings even if cache exists
            batch_size: Number of sequences to process in each batch
        """
        cache_file = self.cache_dir / "structure_embeddings.pkl"
        
        if not force_rebuild and cache_file.exists():
            print("Loading cached embeddings...")
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
                self.embeddings = cached_data['embeddings']
                self.protein_ids = cached_data['protein_ids']
            return
        
        print("Building structure similarity embeddings...")
        self.protein_ids = list(self.train_sequences.keys())
        sequences = [self.train_sequences[pid] for pid in self.protein_ids]
        
        # Process embeddings in batches
        print(f"Processing {len(sequences)} sequences in batches of {batch_size}")
        embeddings_list = self.generate_structure_embeddings_batch(sequences, batch_size)
        
        # Handle different embedding shapes (ensuring we have a 1D or 2D array)
        processed_embeddings = []
        for embedding in embeddings_list:
            if embedding.ndim > 2:
                embedding = embedding.reshape(1, -1)
            processed_embeddings.append(embedding)
            
        # Convert to numpy array
        self.embeddings = np.vstack(processed_embeddings)
        
        # Cache the embeddings
        with open(cache_file, 'wb') as f:
            pickle.dump({
                'embeddings': self.embeddings,
                'protein_ids': self.protein_ids
            }, f)
    
    def compute_similarity(self, query_embedding, database_embeddings):
        """
        Compute similarity scores between query and database embeddings.
        Uses Euclidean distance to match the behavior of the previous FAISS implementation.
        
        Args:
            query_embedding: Query embedding
            database_embeddings: Database embeddings
            
        Returns:
            Distances and indices of nearest neighbors
        """
        # Compute squared Euclidean distances
        distances = np.sum((database_embeddings - query_embedding) ** 2, axis=1)
        
        # Get indices sorted by distance (ascending)
        indices = np.argsort(distances)
        
        return distances, indices
    
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
        
        # Ensure the embedding is in the right shape
        if query_embedding.ndim > 2:
            query_embedding = query_embedding.reshape(1, -1)
        elif query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Search in embeddings matrix
        distances, indices = self.compute_similarity(query_embedding, self.embeddings)
        
        # Get top k results
        top_indices = indices[:k]
        
        # Get results
        results = []
        for idx in top_indices:
            protein_id = self.protein_ids[idx]
            go_terms = self.train_terms[protein_id]
            distance = distances[idx]
            similarity_score = 1.0 / (1.0 + distance)  # Convert distance to similarity
            results.append((protein_id, go_terms, similarity_score))
            
        return results

def main():
    # Example usage
    searcher = StructureSimilaritySearch(
        train_sequences_path="function/cafa/Train/train_sequences.fasta",
        train_terms_path="function/cafa/Train/train_terms.tsv"
    )
    
    # Build embeddings matrix (only needed once)
    searcher.build_index(batch_size=4)  # Adjust batch size based on available GPU memory
    
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