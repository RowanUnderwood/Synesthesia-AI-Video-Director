import os
import csv
import json
import re
import time
import glob
import shutil
import threading

import requests
import pandas as pd

import config
from utils import get_file_path, get_ltx_duration

# ==========================================
# LLM BRIDGE
# ==========================================

class LLMBridge:
    def __init__(self, base_url=None):
        self.base_url = base_url if base_url is not None else config.LM_STUDIO_URL

    def _is_minimax_model(self, model):
        return str(model).lower().startswith("minimax")

    def _get_headers(self, model):
        if self._is_minimax_model(model) and config.MINIMAX_API_KEY:
            return {"Authorization": f"Bearer {config.MINIMAX_API_KEY}"}
        return {}

    def _get_base_url(self, model):
        if self._is_minimax_model(model):
            return config.MINIMAX_BASE_URL
        return self.base_url

    def get_models(self):
        models = []
        if config.LLM_PROVIDER == "MiniMax":
            return list(config.MINIMAX_MODELS)
        try:
            resp = requests.get(f"{self.base_url}/models", timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                models = [m['id'] for m in data['data']]
        except Exception:
            pass
        if not models:
            models = ["qwen3-vl-8b-instruct-abliterated-v2.0"]
        return models

    def query(self, system_prompt, user_prompt, model, temperature=0.7):
        base_url = self._get_base_url(model)
        url = f"{base_url}/chat/completions"
        # MiniMax requires temperature in (0.0, 1.0]; clamp if needed
        if self._is_minimax_model(model):
            temperature = max(0.01, min(1.0, temperature))
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": temperature
        }
        headers = self._get_headers(model)
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=600)
            if resp.status_code != 200:
                return f"Error {resp.status_code} from LLM: {resp.text}"
            return resp.json()['choices'][0]['message']['content'].strip()
        except Exception as e:
            return f"Error: {str(e)}"

# ==========================================
# PROJECT MANAGER
# ==========================================

class ProjectManager:
    def __init__(self):
        self.current_project = None
        self.base_dir = "projects"
        os.makedirs(self.base_dir, exist_ok=True)
        self.df = pd.DataFrame(columns=config.REQUIRED_COLUMNS)
        self.stop_generation = False
        self.stop_video_generation = False
        self.is_generating = False
        self.llm_busy = False
        self.character_bibles = {}  # {character_name: description}

        # Render Queue
        self.render_queue = []
        self.queue_lock = threading.Lock()
        self.queue_paused = False
        self.queue_processor_running = False

        # Time Tracking Variables
        self.total_time_spent = 0
        self.session_start_time = None

        # GPU / RAM leakage warning (set by Tab 3 after each render)
        self.ltx_ram_warning = ""

    def __deepcopy__(self, memo):
        import copy
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if isinstance(v, type(threading.Lock())):
                setattr(result, k, threading.Lock())
            else:
                setattr(result, k, copy.deepcopy(v, memo))
        return result

    def sanitize_name(self, name):
        return re.sub(r'[\\/*?:"<>|]', "", name).strip().strip(".").replace(" ", "_")

    def get_current_total_time(self):
        if self.session_start_time and self.current_project:
            elapsed = time.time() - self.session_start_time
            self.session_start_time = time.time()
            self.total_time_spent += elapsed

            settings = self.load_project_settings()
            settings["total_time_spent"] = self.total_time_spent
            path = os.path.join(self.base_dir, self.current_project, "settings.json")
            try:
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump(settings, f, indent=4)
            except Exception:
                pass

        return self.total_time_spent

    def create_project(self, name):
        if not name: return "Invalid name"
        clean_name = self.sanitize_name(name)
        path = os.path.join(self.base_dir, clean_name)

        folders = ["assets", "audio_chunks", "videos", "renders", "cutting_room", "first_frames"]

        if os.path.exists(path):
            return f"Project '{clean_name}' already exists."

        for f in folders:
            os.makedirs(os.path.join(path, f), exist_ok=True)

        self.df = pd.DataFrame(columns=config.REQUIRED_COLUMNS)
        self.df.to_csv(os.path.join(path, "shot_list.csv"), index=False)
        self.current_project = clean_name

        self.total_time_spent = 0
        self.session_start_time = time.time()

        with open(os.path.join(path, "lyrics.txt"), "w") as f:
            f.write("")

        initial_settings = config.get_global_defaults()
        settings_path = os.path.join(path, "settings.json")
        try:
            with open(settings_path, "w", encoding="utf-8") as f:
                json.dump(initial_settings, f, indent=4)
        except Exception:
            pass

        return f"Project '{clean_name}' created."

    def load_project(self, name):
        path = os.path.join(self.base_dir, name)
        csv_path = os.path.join(path, "shot_list.csv")
        if os.path.exists(csv_path):
            self.df = pd.read_csv(csv_path)
            for col in config.REQUIRED_COLUMNS:
                if col not in self.df.columns:
                    self.df[col] = ""
            self.current_project = name

            settings = self.load_project_settings()
            self.total_time_spent = settings.get("total_time_spent", 0)
            self.session_start_time = time.time()

            sync_video_directory(self)
            self.load_character_bibles()
            return f"Loaded '{name}'", self.df
        return "Project not found.", pd.DataFrame()

    def import_csv(self, file_obj):
        if not self.current_project:
            return "No project loaded.", self.df

        try:
            new_df = pd.read_csv(get_file_path(file_obj))
        except Exception as e:
            return f"❌ Error reading CSV: {e}", self.df

        mode = self.load_project_settings().get("video_mode", "Intercut")

        if mode == "Intercut":
            return self._import_csv_intercut(new_df)
        else:
            return self._import_csv_flexible(new_df, mode)

    def _import_csv_intercut(self, new_df):
        """Import validation for Intercut mode: shot IDs and row count must match exactly."""
        if "Shot_ID" not in new_df.columns:
            return "❌ Error: Uploaded CSV is missing the 'Shot_ID' column.", self.df

        if len(new_df) != len(self.df):
            return f"❌ Error: Uploaded CSV has {len(new_df)} rows, but current project has {len(self.df)} rows.", self.df

        new_df = new_df.set_index("Shot_ID")
        curr_df = self.df.set_index("Shot_ID")

        missing_shots = set(curr_df.index) - set(new_df.index)
        if missing_shots:
            return f"❌ Error: CSV is missing required Shot IDs: {', '.join(str(s) for s in missing_shots)}", self.df

        if 'Type' not in new_df.columns:
            return "❌ Error: CSV is missing 'Type' column.", self.df

        valid_types = {"Vocal", "Action"}
        invalid_types = set(new_df['Type'].unique()) - valid_types
        if invalid_types:
            return f"❌ Error: Invalid Type values: {', '.join(map(str, invalid_types))}. Must be 'Vocal' or 'Action'.", self.df

        type_changed = new_df['Type'] != curr_df['Type']
        changed_shots = curr_df[type_changed].index.tolist()

        if type_changed.any():
            curr_df['Type'] = new_df['Type']

        if 'Video_Prompt' in new_df.columns:
            curr_df['Video_Prompt'] = new_df['Video_Prompt']
            self.df = curr_df.reset_index()
            self.save_data()
            if changed_shots:
                return f"✅ CSV imported. Type changed for: {', '.join(map(str, changed_shots))}. Prompts updated.", self.df
            return "✅ CSV Uploaded & Verified. Prompts successfully updated.", self.df
        else:
            if changed_shots:
                self.df = curr_df.reset_index()
                self.save_data()
                return f"✅ Type changed for: {', '.join(map(str, changed_shots))}.", self.df
            return "✅ No changes detected. Shot list unchanged.", self.df

    def _import_csv_flexible(self, new_df, mode):
        """Import for All Vocals, All Action, and Scripted modes.

        Users may freely change shot count and durations. The Duration column
        drives all timing — Start_Time, End_Time, and frame columns are recalculated
        from scratch. Total duration is validated against the original audio length
        for All Vocals and All Action, but is skipped for Scripted (which provides
        its own audio or no audio).
        """
        fps = 24.0
        original_count = len(self.df)

        # --- Required columns ---
        for col in ("Shot_ID", "Duration", "Type"):
            if col not in new_df.columns:
                return f"❌ Error: Uploaded CSV is missing the '{col}' column.", self.df

        n = len(new_df)

        # --- Shot_ID integrity: no duplicates ---
        if new_df["Shot_ID"].duplicated().any():
            dupes = sorted(new_df.loc[new_df["Shot_ID"].duplicated(keep=False), "Shot_ID"].unique().tolist())
            return f"❌ Error: Duplicate Shot IDs found: {', '.join(str(s) for s in dupes)}.", self.df

        # --- Shot_ID integrity: gap-free sequential S001, S002, ... ---
        sorted_user_ids = sorted(new_df["Shot_ID"].tolist())
        expected_ids = [f"S{i+1:03d}" for i in range(n)]
        if sorted_user_ids != expected_ids:
            missing = sorted(set(expected_ids) - set(sorted_user_ids))
            extra = sorted(set(sorted_user_ids) - set(expected_ids))
            parts = []
            if missing:
                parts.append(f"missing: {', '.join(missing)}")
            if extra:
                parts.append(f"unexpected IDs: {', '.join(extra)}")
            return f"❌ Error: Shot IDs must be sequential with no gaps (S001, S002, …). {'; '.join(parts)}.", self.df

        # --- Type validation ---
        valid_types = {"Vocal", "Action"}
        invalid_types = set(new_df['Type'].dropna().unique()) - valid_types
        if invalid_types:
            return f"❌ Error: Invalid Type values: {', '.join(map(str, invalid_types))}. Must be 'Vocal' or 'Action'.", self.df

        # --- Duration validation and snapping ---
        snapped_durations = []
        snap_changes = []
        for _, row in new_df.iterrows():
            shot_id = row["Shot_ID"]
            raw_val = row["Duration"]
            try:
                raw_dur = float(raw_val)
            except (ValueError, TypeError):
                return (
                    f"❌ Error: Duration for {shot_id} is not a valid number ('{raw_val}'). "
                    f"Use '.' as the decimal separator."
                ), self.df

            if raw_dur < 1.0 or raw_dur > 10.05:
                return f"❌ Error: Duration for {shot_id} is {raw_dur}s — must be between 1 and 10 seconds.", self.df

            snapped = get_ltx_duration(raw_dur)
            if abs(snapped - raw_dur) > 0.001:
                snap_changes.append(shot_id)
            snapped_durations.append(snapped)

        # --- Total duration check (skipped for Scripted — it manages its own audio) ---
        if mode != "Scripted":
            original_total = float(self.df['Duration'].sum())
            snapped_total = sum(snapped_durations)
            tolerance = n * 0.5
            if abs(snapped_total - original_total) > tolerance:
                return (
                    f"❌ Total duration mismatch: your shots sum to {snapped_total:.2f}s "
                    f"but the project audio is {original_total:.2f}s. "
                    f"(Tolerance: ±{tolerance:.1f}s for {n} shots)"
                ), self.df

        # --- Default Render_Resolution: most common in existing shot list ---
        default_res = "540p"
        if not self.df.empty and 'Render_Resolution' in self.df.columns:
            res_counts = self.df['Render_Resolution'].value_counts()
            if not res_counts.empty:
                default_res = res_counts.index[0]

        has_prompt = 'Video_Prompt' in new_df.columns
        has_lyrics = 'Lyrics' in new_df.columns
        has_characters = 'Characters' in new_df.columns
        has_resolution = 'Render_Resolution' in new_df.columns

        # --- Recalculate all timing columns from snapped durations ---
        new_rows = []
        cursor = 0.0
        for i, (snapped_dur, (_, row)) in enumerate(zip(snapped_durations, new_df.iterrows())):
            start = cursor
            end = cursor + snapped_dur
            new_rows.append({
                "Shot_ID": f"S{i+1:03d}",
                "Type": row["Type"],
                "Start_Time": round(start, 4),
                "End_Time": round(end, 4),
                "Duration": round(snapped_dur, 4),
                "Start_Frame": int(round(start * fps)),
                "End_Frame": int(round(end * fps)),
                "Total_Frames": int(round(end * fps)) - int(round(start * fps)),
                "Lyrics": row["Lyrics"] if has_lyrics else "",
                "Video_Prompt": row["Video_Prompt"] if has_prompt else "",
                "Characters": row["Characters"] if has_characters else "",
                "Video_Path": "",
                "All_Video_Paths": "",
                "Status": "Pending",
                "Render_Resolution": (
                    str(row["Render_Resolution"]).strip()
                    if has_resolution and pd.notna(row.get("Render_Resolution"))
                    and str(row["Render_Resolution"]).strip() not in ("", "nan")
                    else default_res
                ),
            })
            cursor = end

        result_df = pd.DataFrame(new_rows)
        for col in config.REQUIRED_COLUMNS:
            if col not in result_df.columns:
                result_df[col] = ""
        self.df = result_df[config.REQUIRED_COLUMNS]
        self.save_data()

        # --- Build success message ---
        msg_parts = [f"✅ CSV imported. Shots: {original_count} → {n}."]
        if has_prompt:
            msg_parts.append("Prompts updated.")
        if snap_changes:
            msg_parts.append(f"{len(snap_changes)} duration(s) snapped to LTX-compatible values ({', '.join(snap_changes)}).")

        return " ".join(msg_parts), self.df

    def export_csv(self):
        if not self.current_project or self.df.empty:
            return None
        return os.path.join(self.base_dir, self.current_project, "shot_list.csv")

    def export_character_bibles(self):
        if not self.current_project or not self.character_bibles:
            return None
        return os.path.join(self.base_dir, self.current_project, "character_bibles.csv")

    def import_character_bibles(self, file_obj):
        if file_obj is None:
            return "❌ No file provided.", None
        try:
            path = file_obj.name if hasattr(file_obj, "name") else str(file_obj)
            df = pd.read_csv(path)
            if "character_name" not in df.columns or "description" not in df.columns:
                return "❌ CSV must have 'character_name' and 'description' columns.", None
            bibles = {}
            for _, row in df.iterrows():
                name = str(row.get("character_name", "")).strip()
                desc = str(row.get("description", "")).strip()
                if name and name.lower() != "nan":
                    bibles[name] = desc
            if not bibles:
                return "❌ No valid entries found in CSV.", None
            self.character_bibles = bibles
            self.save_character_bibles()
            self.update_characters_column()
            self.save_data()
            bible_df = pd.DataFrame(list(bibles.items()), columns=["character_name", "description"])
            return f"✅ Imported {len(bibles)} character(s).", bible_df
        except Exception as e:
            return f"❌ Error importing bibles: {e}", None

    def save_data(self):
        if self.current_project:
            path = os.path.join(self.base_dir, self.current_project, "shot_list.csv")
            self.df.to_csv(path, index=False)

    def save_character_bibles(self):
        if not self.current_project:
            return
        path = os.path.join(self.base_dir, self.current_project, "character_bibles.csv")
        try:
            with open(path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f, quoting=csv.QUOTE_ALL)
                writer.writerow(["character_name", "description"])
                for name, desc in self.character_bibles.items():
                    writer.writerow([name, desc])
        except Exception as e:
            print(f"Error saving character bibles: {e}")

    def load_character_bibles(self):
        if not self.current_project:
            self.character_bibles = {}
            return {}
        path = os.path.join(self.base_dir, self.current_project, "character_bibles.csv")
        bibles = {}
        if os.path.exists(path):
            try:
                with open(path, 'r', newline='', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    next(reader, None)  # skip header
                    for row in reader:
                        if len(row) >= 2 and row[0].strip():
                            bibles[row[0].strip()] = row[1].strip()
            except Exception as e:
                print(f"Error loading character bibles: {e}")
        self.character_bibles = bibles
        return bibles

    def update_characters_column(self):
        """Scan each shot's Video_Prompt and record which bible characters appear in it."""
        if "Characters" not in self.df.columns:
            self.df["Characters"] = ""
        for idx, row in self.df.iterrows():
            prompt_raw = row.get("Video_Prompt", "")
            prompt = "" if pd.isna(prompt_raw) else str(prompt_raw)
            found = []
            for name in self.character_bibles:
                pattern = re.compile(r'\b' + re.escape(name) + r'\b', re.IGNORECASE | re.UNICODE)
                if pattern.search(prompt):
                    found.append(name)
            self.df.at[idx, "Characters"] = ", ".join(found)

    def get_path(self, subfolder):
        if not self.current_project: return None
        return os.path.join(self.base_dir, self.current_project, subfolder)

    def save_lyrics(self, text):
        if not self.current_project: return
        path = os.path.join(self.base_dir, self.current_project, "lyrics.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

    def get_lyrics(self):
        if not self.current_project: return ""
        path = os.path.join(self.base_dir, self.current_project, "lyrics.txt")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        return ""

    def save_project_settings(self, settings_dict):
        if not self.current_project: return "No project loaded."

        existing_settings = self.load_project_settings()
        existing_settings.update(settings_dict)
        existing_settings["total_time_spent"] = self.get_current_total_time()

        path = os.path.join(self.base_dir, self.current_project, "settings.json")
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(existing_settings, f, indent=4)
            return f"Settings saved."
        except Exception as e:
            return f"Error saving settings: {e}"

    def load_project_settings(self):
        if not self.current_project: return {}
        path = os.path.join(self.base_dir, self.current_project, "settings.json")
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def save_asset(self, source_path, filename):
        if not self.current_project or not source_path: return None
        dest = os.path.join(self.get_path("assets"), filename)
        if os.path.abspath(source_path) == os.path.abspath(dest): return dest
        shutil.copy(source_path, dest)
        return dest

    def get_asset_path_if_exists(self, filename):
        if not self.current_project: return None
        path = os.path.join(self.get_path("assets"), filename)
        return path if os.path.exists(path) else None

# ==========================================
# LOGIC: DIRECTORY SYNC
# ==========================================

def sync_video_directory(pm):
    if not pm.current_project: return "No project loaded."
    vid_dir = pm.get_path("videos")
    if not os.path.exists(vid_dir): return "No videos directory."

    mp4s = glob.glob(os.path.join(vid_dir, "*.mp4"))
    shot_vids = {}

    for v in mp4s:
        fname = os.path.basename(v)
        shot_id = fname.split("_")[0].upper()
        if shot_id not in shot_vids: shot_vids[shot_id] = []
        shot_vids[shot_id].append(v)

    if "All_Video_Paths" not in pm.df.columns:
        pm.df["All_Video_Paths"] = ""

    for idx, row in pm.df.iterrows():
        sid = str(row.get("Shot_ID", "")).upper()
        if sid in shot_vids:
            vids = sorted(shot_vids[sid], key=os.path.getmtime, reverse=True)
            pm.df.at[idx, "All_Video_Paths"] = ",".join(vids)

            curr_path_raw = row.get("Video_Path", "")
            curr_path = "" if pd.isna(curr_path_raw) else str(curr_path_raw)

            if not curr_path or not os.path.exists(curr_path) or curr_path not in vids:
                pm.df.at[idx, "Video_Path"] = vids[0]
                pm.df.at[idx, "Status"] = "Done"
        else:
            pm.df.at[idx, "All_Video_Paths"] = ""

            curr_path_raw = row.get("Video_Path", "")
            curr_path = "" if pd.isna(curr_path_raw) else str(curr_path_raw)

            if not curr_path or not os.path.exists(curr_path):
                pm.df.at[idx, "Video_Path"] = ""
                pm.df.at[idx, "Status"] = "Pending"

    pm.save_data()
    return "Directory sync complete."
