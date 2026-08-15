import os
import csv
import requests
import time
import re
import concurrent.futures
import threading
from datetime import datetime
from tqdm import tqdm

# ================= CONFIGURATION =================
LM_STUDIO_URL = "http://192.168.2.192:1234/v1/chat/completions"
MODEL_ID = "qwen3.5-9b-abliterated"

# I/O Settings
INPUT_CSV = "shot_list.csv"       # Change this to your input file name
OUTPUT_CSV = "shot_list_reimagined.csv"

# Reliability Settings
LM_TIMEOUT = 120  
MAX_RETRIES = 2

# Clarification Settings
REQUIRED_KEYWORD = "silly hat" 
MAX_CLARIFICATIONS = 2 

# --- Keyword Replacement Settings ---
ENABLE_SWAPS = True
# The number of swap pairs defined below
NUM_SWAPS = 2 
# List of (Target Word, Replacement Word)
KEYWORD_SWAPS = [
    ("wheel", "Toaster"),
    ("hat", "silly hat")
]
# =================================================

stop_requested = False

def rewrite_prompt(original_prompt):
    """Sends the text prompt to the LLM to be reimagined."""
    if not original_prompt or not str(original_prompt).strip():
        return original_prompt
        
    messages = [
        {"role": "user", "content": f"Rewrite the following image or video generation prompt. Change all of the characters to be wearing a silly hat. Be creative in your description of the hats. Provide the details and organized scene description ONLY as your response, no additional information.\n\nOriginal Prompt:\n{original_prompt}"}
    ]

    total_loops = 1 + (MAX_CLARIFICATIONS if REQUIRED_KEYWORD else 0)
    current_description = None

    for loop_index in range(total_loops):
        payload = {
            "model": MODEL_ID,
            "messages": messages,
            "temperature": 0.7
        }

        success = False
        for attempt in range(MAX_RETRIES + 1):
            try:
                response = requests.post(LM_STUDIO_URL, json=payload, timeout=LM_TIMEOUT)
                response.raise_for_status()
                current_description = response.json()['choices'][0]['message']['content'].strip()
                success = True
                break
            except requests.exceptions.Timeout:
                print(f"\n[!] Timeout on LLM request (Net Attempt {attempt+1})")
                if attempt < MAX_RETRIES:
                    time.sleep(1)
            except Exception as e:
                print(f"\n[!] LM Studio Error: {e}")
                break
        
        if not success:
            return original_prompt # Fallback to original if LLM completely fails

        if not REQUIRED_KEYWORD:
            return current_description

        if REQUIRED_KEYWORD.lower() in current_description.lower():
            return current_description 
        
        if loop_index < total_loops - 1:
            print(f"\n[?] Missing '{REQUIRED_KEYWORD}'. Asking for clarification (Attempt {loop_index+1}/{MAX_CLARIFICATIONS})...")
            messages.append({"role": "assistant", "content": current_description})
            messages.append({
                "role": "user", 
                "content": f"You missed a key detail. The prompt definitely needs to contain a {REQUIRED_KEYWORD}. Please rewrite the description and ensure you explicitly include the {REQUIRED_KEYWORD}."
            })
        else:
            print(f"\n[!] Warning: '{REQUIRED_KEYWORD}' still missing after max retries. Using last result.")

    return current_description

def process_csv():
    global stop_requested

    if not os.path.exists(INPUT_CSV):
        print(f"[!] Error: Input file '{INPUT_CSV}' not found.")
        return

    print(f"--- PRESS CTRL+C TO CANCEL ---")
    print(f"Targeting Instance: {LM_STUDIO_URL}")
    print(f"Clarification Keyword: '{REQUIRED_KEYWORD}'")
    print(f"Threads: 4 (Make sure LM Studio Parallel requests is set to 4)")
    
    if ENABLE_SWAPS:
        print(f"Keyword Swaps Active: {NUM_SWAPS} rules applied.")

    # We need a lock so multiple threads don't try to write to the CSV at the exact same millisecond
    write_lock = threading.Lock()

    def process_and_write(row, writer):
        """Worker function that processes a single row and writes it safely."""
        for col in ["Video_Prompt", "First_Frame_Prompt"]:
            if col in row and row[col].strip():
                original_text = row[col]
                new_text = rewrite_prompt(original_text)

                if ENABLE_SWAPS and new_text:
                    for i in range(min(len(KEYWORD_SWAPS), NUM_SWAPS)):
                        old_word, new_word = KEYWORD_SWAPS[i]
                        pattern = re.compile(re.escape(old_word), re.IGNORECASE)
                        new_text = pattern.sub(new_word, new_text)
                
                row[col] = new_text

        # Safely write the row to the file
        with write_lock:
            writer.writerow(row)

    try:
        with open(INPUT_CSV, 'r', encoding='utf-8') as infile:
            reader = csv.DictReader(infile)
            fieldnames = reader.fieldnames
            rows = list(reader)

        with open(OUTPUT_CSV, 'w', encoding='utf-8', newline='') as outfile:
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()

            # Execute up to 4 requests at the same time
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(process_and_write, row, writer) for row in rows]
                
                # Update the progress bar as each thread finishes
                for future in tqdm(concurrent.futures.as_completed(futures), total=len(rows), unit="row"):
                    if stop_requested:
                        # Attempt to cancel pending tasks if the user hits CTRL+C
                        executor.shutdown(wait=False, cancel_futures=True)
                        break

    except KeyboardInterrupt:
        print("\n[!] Stop signal received. Finishing active threads...")
        stop_requested = True
    except Exception as e:
        print(f"\n[!] Error processing CSV: {e}")

    print(f"\nProcessing complete. Reimagined data saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    process_csv()