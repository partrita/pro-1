import json
import tiktoken
from pathlib import Path
import re
from collections import defaultdict
import statistics

def count_tokens(text: str) -> int:
    """Count tokens in text using tiktoken"""
    encoding = tiktoken.encoding_for_model("gpt-4")  # Using GPT-4's tokenizer as a standard
    return len(encoding.encode(text))

def analyze_reasoning_file(file_path: str):
    """Analyze token counts in reasoning file"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    records = data.get('records', [])
    
    # Initialize counters
    total_tokens = 0
    token_counts = []
    aspect_tokens = defaultdict(list)
    section_tokens = defaultdict(list)
    
    print(f"\nAnalyzing {len(records)} records...")
    
    for record in records:
        reasoning = record.get('reasoning', '')
        aspect = record.get('aspect', 'Unknown')
        
        # Count total tokens
        tokens = count_tokens(reasoning)
        total_tokens += tokens
        token_counts.append(tokens)
        aspect_tokens[aspect].append(tokens)
        
        # Extract and count tokens in think and answer sections
        think_match = re.search(r'<think>(.*?)</think>', reasoning, re.DOTALL)
        answer_match = re.search(r'<answer>(.*?)</answer>', reasoning, re.DOTALL)
        
        if think_match:
            think_tokens = count_tokens(think_match.group(1))
            section_tokens['think'].append(think_tokens)
        
        if answer_match:
            answer_tokens = count_tokens(answer_match.group(1))
            section_tokens['answer'].append(answer_tokens)
    
    # Calculate statistics
    print("\nOverall Statistics:")
    print(f"Total records: {len(records)}")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Average tokens per record: {statistics.mean(token_counts):,.1f}")
    print(f"Median tokens per record: {statistics.median(token_counts):,.1f}")
    print(f"Min tokens: {min(token_counts):,}")
    print(f"Max tokens: {max(token_counts):,}")
    
    print("\nTokens by Aspect:")
    for aspect, tokens in aspect_tokens.items():
        print(f"\n{aspect}:")
        print(f"  Records: {len(tokens)}")
        print(f"  Average tokens: {statistics.mean(tokens):,.1f}")
        print(f"  Median tokens: {statistics.median(tokens):,.1f}")
    
    print("\nTokens by Section:")
    for section, tokens in section_tokens.items():
        if tokens:
            print(f"\n{section.title()}:")
            print(f"  Average tokens: {statistics.mean(tokens):,.1f}")
            print(f"  Median tokens: {statistics.median(tokens):,.1f}")

if __name__ == "__main__":
    data_path = Path("data/go_reasoning.json")
    analyze_reasoning_file(data_path) 