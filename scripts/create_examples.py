import json
import re
import os
import glob
import subprocess
from collections import defaultdict
from tqdm import tqdm
import sys

# Add script directory to path to allow importing local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from formality_rules import FORMALITY_RULES
except ImportError:
    print("Error: Could not import formality_rules.py")
    sys.exit(1)

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
INPUT_DIR = "data/unprocessed"
OUTPUT_FILE = "data/processed/stratified_data_set.jsonl"
LOG_FILE = "data/processed/stratification_log.json"
MAX_EXAMPLES_PER_RULE = 5000
CHUNK_SIZE = 5000
CACHE_INTERVAL = 50000  # Save every N lines processed

# ------------------------------------------------------------
# Compile Rules
# ------------------------------------------------------------
COMPILED_RULES = []
ALL_RULE_KEYS = set()
for rule_group in FORMALITY_RULES:
    sorted_variants = sorted(rule_group, key=len, reverse=True)
    escaped_variants = [re.escape(v) for v in sorted_variants if v]
    
    if not escaped_variants:
        continue
        
    pattern = "|".join(escaped_variants)
    regex = re.compile(f"({pattern})")
    
    rule_key = tuple(sorted_variants) # Use sorted tuple as key
    ALL_RULE_KEYS.add(rule_key)
    
    COMPILED_RULES.append({
        "variants": rule_group,
        "regex": regex,
        "key": rule_key
    })

# ------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------
def count_lines(filepath):
    try:
        result = subprocess.run(['wc', '-l', filepath], stdout=subprocess.PIPE, text=True)
        return int(result.stdout.split()[0])
    except Exception as e:
        print(f"Error counting lines: {e}")
        return None

def load_existing_counts():
    counts = defaultdict(int)
    
    # Load Log
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, 'r', encoding='utf-8') as f:
                loaded_counts = json.load(f)
                # Convert string keys back to tuples? 
                # The log keys are strings like "('a', 'b')".
                # We need to map them back to our rule keys to enforce limits correctly.
                # Since we use the tuple as key in COMPILED_RULES, we need to match it.
                # Let's try to reconstruct the key mapping.
                
                # Create a map of str(key) -> key
                str_to_key = {str(rule["key"]): rule["key"] for rule in COMPILED_RULES}
                
                for k, v in loaded_counts.items():
                    if k in str_to_key:
                        counts[str_to_key[k]] = v
        except Exception as e:
            print(f"Warning: Could not load log file: {e}")
            pass

    return counts

def append_data(new_examples, counts):
    # Append Data to JSONL
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
        for example_set in new_examples:
            # Write each example set as a single line JSON object
            f.write(json.dumps(example_set, ensure_ascii=False) + "\n")

    # Save Log (Overwrite)
    with open(LOG_FILE, 'w', encoding="utf-8") as f:
        log_counts = {str(k): v for k, v in counts.items()}
        json.dump(log_counts, f, ensure_ascii=False, indent=2)

# ------------------------------------------------------------
# Processing Logic
# ------------------------------------------------------------
def process_dataset():
    if not os.path.exists(INPUT_DIR):
        print(f"Input directory {INPUT_DIR} not found.")
        return

    input_files = glob.glob(os.path.join(INPUT_DIR, "*"))
    if not input_files:
        print(f"No files found in {INPUT_DIR}.")
        return

    print(f"Found {len(input_files)} files to process.")

    counts = load_existing_counts()
    
    # Buffer for new examples before saving
    new_examples_buffer = []

    for input_file in input_files:
        print(f"Processing {input_file}...")
        total_lines = count_lines(input_file)
        
        lines_processed_since_save = 0
        
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                # Use tqdm iterator
                iterator = tqdm(f, total=total_lines, unit="lines")
                
                for line in iterator:
                    sentence = line.strip()
                    if not sentence:
                        continue

                    # Check saturation
                    saturated_rules = 0
                    for rule in COMPILED_RULES:
                        if counts[rule["key"]] >= MAX_EXAMPLES_PER_RULE:
                            saturated_rules += 1
                    
                    if saturated_rules == len(COMPILED_RULES):
                        print("\nAll rules reached max examples. Stopping early.")
                        iterator.close()
                        break

                    # Find all matching rules first
                    matching_rules = []
                    for rule in COMPILED_RULES:
                        rule_key = rule["key"]
                        
                        if counts[rule_key] >= MAX_EXAMPLES_PER_RULE:
                            continue

                        match = rule["regex"].search(sentence)
                        if match:
                            matching_rules.append((rule, match))
                    
                    if not matching_rules:
                        continue
                        
                    # Sort by current count (ascending) to prefer rules with fewer examples
                    # If counts are equal, maybe sort by something else? 
                    # For now, just count is enough. Stable sort preserves order (priority in list).
                    matching_rules.sort(key=lambda x: counts[x[0]["key"]])
                    
                    # Select the best rule (first one after sorting)
                    best_rule, match = matching_rules[0]
                    rule_key = best_rule["key"]
                    
                    # Apply the best rule
                    start, end = match.span()
                    variants = best_rule["variants"]
                    
                    example_set = []
                    for variant in variants:
                        if variant == "": 
                            new_sentence = sentence[:start] + sentence[end:]
                        else:
                            new_sentence = sentence[:start] + variant + sentence[end:]
                        example_set.append(new_sentence)
                    
                    new_examples_buffer.append(example_set)
                    counts[rule_key] += 1
                    
                    lines_processed_since_save += 1
                    
                    # Periodic Cache
                    if lines_processed_since_save >= CACHE_INTERVAL:
                        if new_examples_buffer:
                            append_data(new_examples_buffer, counts)
                            new_examples_buffer = [] # Clear buffer
                        
                        lines_processed_since_save = 0

            # End of file loop
            
            # Final flush for this file
            if new_examples_buffer:
                append_data(new_examples_buffer, counts)
                new_examples_buffer = []
            
            # Check if we stopped early due to saturation
            saturated_rules = 0
            for rule in COMPILED_RULES:
                if counts[rule["key"]] >= MAX_EXAMPLES_PER_RULE:
                    saturated_rules += 1
            if saturated_rules == len(COMPILED_RULES):
                print("Stopping processing as all rules are satisfied.")
                break
                
        except Exception as e:
            print(f"Error processing {input_file}: {e}")

    # Final Save (redundant if flushed, but good for safety)
    if new_examples_buffer:
         append_data(new_examples_buffer, counts)

    print("\n--- Stratification Counts (This Run) ---")
    for rule, count in counts.items():
        if count > 0:
            print(f"{rule}: {count}")

if __name__ == "__main__":
    process_dataset()
