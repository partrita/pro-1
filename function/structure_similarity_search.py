import torch
import esm
import biotite.structure as struc
import biotite.structure.io.pdb as pdb
from Bio import SeqIO
from typing import Dict, List, Tuple, Optional, Set
import numpy as np
from pathlib import Path
import pickle
import os
import json
from tqdm import tqdm
from huggingface_hub import login
from esm.models.esm3 import ESM3
from esm.sdk.api import ESM3InferenceClient, ESMProtein, LogitsConfig, GenerationConfig
from dotenv import load_dotenv
import time
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
        self.sequences_db: List[str] = []  # Sequences that correspond to each embedding (same order)
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
    
    def _load_eval_sequences(self) -> Set[str]:
        """
        Load sequences from evaluation datasets to exclude them from structure embeddings.
        
        Returns:
            Set of protein sequences that should be excluded
        """
        eval_sequences = set()
        
        # Load sequences from eval_dataset/dataset.jsonl
        eval_paths = ["eval_dataset/dataset.jsonl", "non_mc_eval_dataset/dataset.jsonl"]
        
        for path in eval_paths:
            try:
                with open(path, 'r') as f:
                    for line in f:
                        try:
                            data = json.loads(line)
                            if 'sequence' in data:
                                eval_sequences.add(data['sequence'])
                        except json.JSONDecodeError:
                            print(f"Error parsing JSON in {path}")
                            continue
                print(f"Loaded {path} for filtering evaluation sequences")
            except Exception as e:
                print(f"Warning: Could not load {path} for filtering: {e}")
        
        print(f"Loaded {len(eval_sequences)} sequences to exclude from training")
        return eval_sequences
    
    def generate_structure_embedding(self, sequence: str) -> np.ndarray:
        """
        Generate structure-specific embedding for a protein sequence using ESM3's structure track.
        
        Args:
            sequence: Protein sequence as string
            
        Returns:
            numpy.ndarray: Normalized global structure-specific embedding
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
        
        # Extract global protein representation from the last position and make it 1D
        embedding_tensor = structure_output.embeddings
        global_embedding = embedding_tensor[0, -1, :].cpu().numpy().flatten()
        
        # Normalize the embedding
        norm = np.linalg.norm(global_embedding)
        if norm > 0:
            global_embedding = global_embedding / norm
        
        # Print information about the embedding
        print("Sequence start:", sequence[:20])
        
        return global_embedding
    
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
    
    def generate_structure_embeddings_one_by_one(self, sequences: List[str]) -> Tuple[List[np.ndarray], List[int]]:
        """
        Generate structure-specific embeddings for protein sequences one at a time to avoid
        dimensionality inconsistencies. Takes the last vector from the sequence length dimension
        to get a fixed-size embedding and normalizes it for efficient similarity computation.
        
        In case of GPU/CPU memory exhaustion, the offending sequence is skipped and the process
        continues. The index of successfully processed sequences is returned so the caller can
        keep the protein ID list in sync with the produced embeddings.
        
        Args:
            sequences: List of protein sequences as strings
            
        Returns:
            Tuple consisting of:
              1. List of numpy.ndarray: Normalized structure-specific embeddings for each **successfully** processed sequence
              2. List[int]: Indices in the original *sequences* list that were processed successfully
        """
        all_embeddings: List[np.ndarray] = []
        success_indices: List[int] = []
        
        print(f"Processing {len(sequences)} sequences one by one")
        start_time = time.time()
        for i, sequence in enumerate(sequences):
            try:
                print(f"Processing sequence {i+1}/{len(sequences)}")
                # Calculate and print estimated time remaining
                if i > 0:
                    elapsed_time = time.time() - start_time
                    avg_time_per_sequence = elapsed_time / i
                    remaining_sequences = len(sequences) - i
                    estimated_time_remaining = avg_time_per_sequence * remaining_sequences
                    print(f"Estimated time remaining: {estimated_time_remaining/60:.1f} minutes")
                else:
                    start_time = time.time()
                
                # Generate embedding for the sequence
                protein = ESMProtein(sequence=sequence)
                
                # Generate the structure if it doesn't exist
                if protein.coordinates is None:
                    protein = self.model.generate(protein,
                                                 GenerationConfig(track="structure",
                                                                 num_steps=8))
                
                # Encode the protein with the structure
                protein_tensor = self.model.encode(protein)
                
                # Get structure-specific embeddings
                structure_output = self.model.logits(
                    protein_tensor,
                    LogitsConfig(structure=True, return_embeddings=True)
                )
                
                # Get the embedding and take the last vector from sequence length dimension
                embedding = structure_output.embeddings.cpu().numpy()
                # Extract the vector to make it 1D (not 3D or 2D)
                embedding = embedding[0, -1, :].flatten()  # Now shape is just (hidden_size,)
                
                # Normalize the embedding for efficient similarity computation
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm
                
                # Print embedding information
                print("Sequence start:", sequence[:20])
                print("First few elements:", embedding[:5])  # Print first 5 elements of the embedding
                
                all_embeddings.append(embedding)
                success_indices.append(i)
            except KeyboardInterrupt:
                # Graceful early exit on user interrupt – stop processing further sequences
                print("KeyboardInterrupt detected during sequence processing. Stopping further processing and returning partial embeddings.")
                break
            except (torch.cuda.OutOfMemoryError, RuntimeError, MemoryError) as e:
                # Handle memory errors gracefully and continue.
                print(f"RAM limit failed with sequence of length {len(sequence)}. Skipping. Error: {e}")
                # Free GPU memory if possible
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
        
        return all_embeddings, success_indices
    
    def build_index(self, force_rebuild: bool = False, max_sequences: Optional[int] = None):
        """
        Build embeddings matrix for similarity search.
        
        Args:
            force_rebuild: If True, rebuild embeddings even if cache exists
            max_sequences: Maximum number of sequences to embed. If None, process all sequences.
        """
        cache_file = self.cache_dir / "structure_embeddings.pkl"
        
        if not force_rebuild and cache_file.exists():
            print("Loading cached embeddings...")
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
                self.embeddings = cached_data['embeddings']
                self.protein_ids = cached_data['protein_ids']
                # sequences may not exist in older caches; default to empty list
                self.sequences_db = cached_data.get('sequences', [])
            return
        
        print("Building structure similarity embeddings...")
        
        # Load evaluation sequences to exclude
        eval_sequences = self._load_eval_sequences()
        
        # Filter out protein IDs with sequences in evaluation datasets
        all_protein_ids = list(self.train_sequences.keys())
        filtered_protein_ids = []
        
        for pid in all_protein_ids:
            if self.train_sequences[pid] not in eval_sequences:
                filtered_protein_ids.append(pid)
        
        self.protein_ids = filtered_protein_ids
        print(f"Filtered out {len(all_protein_ids) - len(filtered_protein_ids)} sequences that appear in evaluation datasets")
        
        # Limit the number of sequences if max_sequences is specified
        if max_sequences is not None:
            self.protein_ids = self.protein_ids[:max_sequences]
            print(f"Processing {max_sequences} sequences out of {len(filtered_protein_ids)} filtered sequences")
        else:
            print(f"Processing all {len(filtered_protein_ids)} filtered sequences")
        
        sequences_all = [self.train_sequences[pid] for pid in self.protein_ids]
        
        # Process embeddings one by one to extract global embeddings
        embeddings_list, success_indices = self.generate_structure_embeddings_one_by_one(sequences_all)
        
        # Keep only the protein IDs that were successfully embedded
        self.protein_ids = [self.protein_ids[i] for i in success_indices]
        self.sequences_db = [sequences_all[i] for i in success_indices]
        
        if not embeddings_list:
            raise ValueError("Failed to generate any valid embeddings")
        
        # Convert list of embeddings to a 2D array
        self.embeddings = np.stack(embeddings_list)
        
        # Cache the embeddings along with their associated metadata
        with open(cache_file, 'wb') as f:
            pickle.dump({
                'embeddings': self.embeddings,
                'protein_ids': self.protein_ids,
                'sequences': self.sequences_db
            }, f)
    
    def compute_similarity(self, query_embedding, database_embeddings):
        """
        Compute similarity scores between query and database embeddings.
        Uses cosine similarity to measure similarity between embeddings.
        Assumes embeddings are already normalized to unit length.
        
        Args:
            query_embedding: Query embedding (1D array, normalized)
            database_embeddings: Database embeddings (2D array, normalized)
            
        Returns:
            Similarity scores and indices of nearest neighbors
        """

        print(f"Query embedding shape: {query_embedding.shape}")
        print(f"Query embedding: {query_embedding}")
        # Compute cosine similarity (dot product of normalized vectors)
        similarities = np.dot(database_embeddings, query_embedding)
        
        # Get indices sorted by similarity (descending)
        indices = np.argsort(similarities)[::-1]

        # print("Database shape:", database_embeddings.shape)
        # print("First database element norm:", np.linalg.norm(database_embeddings[0]))
        # print("Query shape:", query_embedding.shape)
        # print("Similarity range:", similarities.min(), "to", similarities.max())
        
        return similarities, indices
    
    def find_similar_structures(self, 
                              query_sequence: str, 
                              k: int = 5) -> List[Tuple[str, Dict[str, set], float]]:
        """
        Find proteins with similar structures and their GO terms.
        Filters out sequences with >90% raw sequence similarity.
        
        Args:
            query_sequence: Query protein sequence
            k: Number of similar structures to return
            
        Returns:
            List of tuples (protein_id, GO_terms, similarity_score)
        """
        # Generate normalized embedding for query sequence
        query_embedding = self.generate_structure_embedding(query_sequence)
        
        # Search in embeddings matrix - gets cosine similarities
        similarities, indices = self.compute_similarity(query_embedding, self.embeddings)
        
        # Filter results based on raw sequence similarity
        results = []
        for idx in indices:
            protein_id = self.protein_ids[idx]
            candidate_seq = self.sequences_db[idx]
            
            # Calculate max raw sequence similarity by trying different alignments
            max_similarity = 0
            query_len = len(query_sequence)
            cand_len = len(candidate_seq)
            
            # Try different starting positions in both sequences
            for q_start in range(query_len):
                for c_start in range(cand_len):
                    # Get overlapping region length
                    overlap_len = min(query_len - q_start, cand_len - c_start)
                    if overlap_len < 20:  # Skip if overlap too small
                        continue
                        
                    # Count matches in overlapping region
                    matches = sum(q == c for q, c in zip(
                        query_sequence[q_start:q_start+overlap_len],
                        candidate_seq[c_start:c_start+overlap_len]
                    ))
                    similarity = (matches / overlap_len) * 100
                    max_similarity = max(max_similarity, similarity)
                    
                    if max_similarity > 90:  # Early exit if we find high similarity
                        break
                if max_similarity > 90:
                    break
            
            # Skip if max similarity is too high
            if max_similarity > 90:
                continue
                
            go_terms = self.train_terms.get(protein_id, {'MFO': set(), 'BPO': set(), 'CCO': set()})
            similarity = float(similarities[idx])
            results.append((protein_id, go_terms, similarity, max_similarity))
            
            # Break if we have enough results
            if len(results) >= k:
                break
        
        return results[:k]  # Return at most k results

def main():
    # Example usage
    searcher = StructureSimilaritySearch(
        train_sequences_path="function/cafa/Train/train_sequences.fasta",
        train_terms_path="function/cafa/Train/train_terms.tsv"
    )
    
    searcher.build_index(max_sequences=72000) 
    
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