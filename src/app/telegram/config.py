import os
from dotenv import load_dotenv
import logging

load_dotenv()  # Load environment variables from .env file

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("config")

# --- Telegram Bot API Configuration ---
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not TELEGRAM_BOT_TOKEN:
    logger.error(
        "TELEGRAM_BOT_TOKEN environment variable not set. Bot will not function.")
    # In a real app, you might raise an exception or exit here.
    # For now, we'll let it proceed but log the error.

TELEGRAM_WEBHOOK_SECRET_TOKEN = os.getenv(
    "TELEGRAM_WEBHOOK_SECRET_TOKEN")  # Optional but recommended
if not TELEGRAM_WEBHOOK_SECRET_TOKEN:
    logger.warning(
        "TELEGRAM_WEBHOOK_SECRET_TOKEN not set. Webhook requests will not be authenticated.")

TELEGRAM_API_BASE_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}" if TELEGRAM_BOT_TOKEN else ""

# --- AssemblyAI API Configuration ---
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")
if not ASSEMBLYAI_API_KEY:
    logger.error(
        "ASSEMBLYAI_API_KEY environment variable not set. Transcription will not function.")

# --- Other Configurations ---
# Example: 10 minutes max audio to process to avoid excessive costs/time
MAX_AUDIO_DURATION_SECONDS = 60
