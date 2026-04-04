import os
import json
import base64
import requests
import random
import csv
import time
import io
import shutil
import re  # Added for case-insensitive keyword swapping
from datetime import datetime
from tqdm import tqdm
from PIL import Image

# ================= CONFIGURATION =================
LM_STUDIO_URL = "http://192.168.2.192:1234/v1/chat/completions"
COMFY_URL = "http://127.0.0.1:8188/prompt" 
MODEL_ID = "qwen3-vl-8b-instruct-abliterated-v2.0"
WORKFLOW_FILE = "ZImage_Poster_API.json" 
LOG_FILE = "reimagine_log.csv"

# Reliability Settings
LM_TIMEOUT = 120  
MAX_RETRIES = 2

# Clarification Settings
REQUIRED_KEYWORD = "silly hat" 
MAX_CLARIFICATIONS = 2 

# --- NEW: Keyword Replacement Settings ---
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

def log_task(filename, ratio, prompt):
    file_exists = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Timestamp", "Filename", "Ratio", "Prompt"])
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), filename, ratio, prompt])

def load_existing_prompts(log_file):
    prompts = {}
    if not os.path.exists(log_file):
        return prompts
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if "Filename" in row and "Prompt" in row:
                    prompts[row["Filename"]] = row["Prompt"]
    except Exception as e:
        print(f"[!] Error reading log file: {e}")
    return prompts

def process_and_encode_image(image_path, max_size=768):
    with Image.open(image_path) as img:
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        if max(img.size) > max_size:
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG", quality=85)
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

def get_image_description(image_path):
    try:
        base64_image = process_and_encode_image(image_path)
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

    messages = [
        {"role": "user", "content": [
            {"type": "text", "text": "If you think the image is a poster or magazine cover, mention this first! Describe this image in extreme detail for an image generation prompt. Change all of the characters to be wearing a silly hat.  Be creative in your description of the hats. Provide the details and organized image description ONLY as your response, no additional information."},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
        ]}
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
                print(f"\n[!] Timeout on {image_path} (Net Attempt {attempt+1})")
                if attempt < MAX_RETRIES:
                    time.sleep(1)
            except Exception as e:
                print(f"\n[!] LM Studio Error: {e}")
                break
        
        if not success:
            return None

        if not REQUIRED_KEYWORD:
            return current_description

        if REQUIRED_KEYWORD.lower() in current_description.lower():
            return current_description 
        
        if loop_index < total_loops - 1:
            print(f"\n[?] Missing '{REQUIRED_KEYWORD}'. Asking for clarification (Attempt {loop_index+1}/{MAX_CLARIFICATIONS})...")
            messages.append({"role": "assistant", "content": current_description})
            messages.append({
                "role": "user", 
                "content": f"You missed a key detail. The image definitely contains a {REQUIRED_KEYWORD}. Please rewrite the description and ensure you explicitly include the {REQUIRED_KEYWORD}."
            })
        else:
            print(f"\n[!] Warning: '{REQUIRED_KEYWORD}' still missing after max retries. Using last result.")

    return current_description

def get_smart_dimensions(image_path):
    with Image.open(image_path) as img:
        w, h = img.size
        ratio = w / h
        
        if ratio > 1.1: 
            return 1152, 896, "landscape (4:3)"
        elif ratio > 0.9:
            return 1024, 1024, "square (1:1)"
        elif ratio > 0.72:
            return 896, 1152, "portrait (3:4)"
        else:
            return 832, 1216, "portrait (2:3)"

def send_to_comfy(prompt_text, width, height, output_prefix):
    try:
        with open(WORKFLOW_FILE, 'r') as f:
            workflow = json.load(f)

        if isinstance(workflow, list) or "nodes" in workflow:
            print(f"[!] FATAL: {WORKFLOW_FILE} is in 'Saved' format.")
            return False

        if "6" in workflow: 
            workflow["6"]["inputs"]["text"] = str(prompt_text)
        if "57" in workflow:
            workflow["57"]["inputs"]["seed"] = random.randint(1, 10**15)
        if "61" in workflow:
            workflow["61"]["inputs"]["width"] = width
            workflow["61"]["inputs"]["height"] = height
        if "73" in workflow:
            workflow["73"]["inputs"]["filename_prefix"] = output_prefix

        response = requests.post(COMFY_URL, json={"prompt": workflow}, timeout=15)
        return response.status_code == 200
    except Exception as e:
        print(f"[!] ComfyUI Error: {e}")
        return False

def main():
    global stop_requested
    valid_exts = ('.jpg', '.jpeg', '.png', '.webp')
    files = [f for f in os.listdir('.') if f.lower().endswith(valid_exts)]
    
    random.shuffle(files)
    
    output_dir = "reimagine"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"--- PRESS CTRL+C TO CANCEL ---")
    print(f"Targeting Primary Instance: {COMFY_URL}")
    print(f"Clarification Keyword: '{REQUIRED_KEYWORD}'")
    
    if ENABLE_SWAPS:
        print(f"Keyword Swaps Active: {NUM_SWAPS} rules applied.")

    cached_prompts = {}
    use_cache = False
    
    if os.path.exists(LOG_FILE):
        user_input = input("Would you like to reuse prompts from the log? (y/n): ").strip().lower()
        if user_input == 'y':
            use_cache = True
            cached_prompts = load_existing_prompts(LOG_FILE)
    
    for filename in tqdm(files, unit="img"):
        if stop_requested:
            break
            
        try:
            try:
                shutil.copy2(filename, os.path.join(output_dir, filename))
            except Exception as e:
                print(f"\n[!] Error copying original file {filename}: {e}")

            description = None
            if use_cache and filename in cached_prompts:
                description = cached_prompts[filename]
            
            if not description:
                description = get_image_description(filename)
            
            if not description:
                print(f"\n[!] Could not get description for {filename}")
                continue

            # --- KEYWORD SWAP LOGIC (The Last Step) ---
            if ENABLE_SWAPS:
                # We only process up to the count specified in NUM_SWAPS
                for i in range(min(len(KEYWORD_SWAPS), NUM_SWAPS)):
                    old_word, new_word = KEYWORD_SWAPS[i]
                    # Uses regex for case-insensitive replacement
                    pattern = re.compile(re.escape(old_word), re.IGNORECASE)
                    description = pattern.sub(new_word, description)
            # ------------------------------------------
                
            w, h, ratio_desc = get_smart_dimensions(filename)
            base_name = os.path.splitext(filename)[0]
            output_prefix = f"{output_dir}/{base_name}_reimagined"
            
            if send_to_comfy(description, w, h, output_prefix):
                log_task(filename, f"{w}x{h} ({ratio_desc})", description)
            else:
                print(f"\n[!] Failed to queue {filename} to ComfyUI")

        except KeyboardInterrupt:
            print("\n[!] Stop signal received. Finishing current task...")
            stop_requested = True

    print(f"\nProcessing complete. Logs updated in {LOG_FILE}")

if __name__ == "__main__":
    main()