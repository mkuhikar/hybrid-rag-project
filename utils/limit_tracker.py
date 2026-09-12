import os
import json
from datetime import datetime

LIMITS_FILE = "usage_limits.json"
MAX_DAILY_MB = 10.0
MAX_DAILY_PROMPTS = 10

def _get_today_str() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d")

def load_usage_data() -> dict:
    today = _get_today_str()
    default_data = {"date": today, "uploaded_bytes": 0, "prompt_count": 0}
    
    if os.path.exists(LIMITS_FILE):
        try:
            with open(LIMITS_FILE, "r") as f:
                data = json.load(f)
            # Auto-reset if the saved data belongs to a previous day
            if data.get("date") != today:
                return default_data
            return data
        except Exception:
            return default_data
    return default_data

def save_usage_data(data: dict) -> None:
    with open(LIMITS_FILE, "w") as f:
        json.dump(data, f, indent=2)

def check_upload_allowed(additional_bytes: int) -> tuple[bool, str]:
    data = load_usage_data()
    current_mb = data["uploaded_bytes"] / (1024 * 1024)
    additional_mb = additional_bytes / (1024 * 1024)
    
    if (current_mb + additional_mb) > MAX_DAILY_MB:
        remaining_mb = max(0.0, MAX_DAILY_MB - current_mb)
        return False, f"Global daily limit reached ({MAX_DAILY_MB} MB/day). Remaining quota today: {remaining_mb:.2f} MB."
    return True, ""

def record_upload(additional_bytes: int) -> None:
    data = load_usage_data()
    data["uploaded_bytes"] += additional_bytes
    save_usage_data(data)

def check_prompt_allowed() -> tuple[bool, str]:
    data = load_usage_data()
    if data["prompt_count"] >= MAX_DAILY_PROMPTS:
        return False, f"Global daily query limit reached ({MAX_DAILY_PROMPTS} prompts/day). Please try again tomorrow!"
    return True, ""

def record_prompt() -> None:
    data = load_usage_data()
    data["prompt_count"] += 1
    save_usage_data(data)