import httpx
import logging
from typing import Optional, List, Dict, Any
from .config import TELEGRAM_API_BASE_URL, TELEGRAM_BOT_TOKEN  # Import configuration

logger = logging.getLogger("telegram_api")

# --- Async HTTP Client ---
http_client = httpx.AsyncClient()


async def send_telegram_request(method: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Helper to send POST requests to the Telegram Bot API."""
    if not TELEGRAM_BOT_TOKEN:
        logger.error("Telegram bot token is missing. Cannot send API request.")
        raise ValueError("Telegram bot token not configured.")

    url = f"{TELEGRAM_API_BASE_URL}/{method}"
    try:
        # Add a timeout
        response = await http_client.post(url, json=payload, timeout=10)
        response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)
        return response.json()
    except httpx.HTTPStatusError as e:
        logger.error(
            f"Telegram API {method} call failed: {e.response.status_code} - {e.response.text}")
        raise
    except httpx.RequestError as e:
        logger.error(f"Error making request to Telegram API {method}: {e}")
        raise


async def send_telegram_text(chat_id: int, text: str, parse_mode: Optional[str] = None) -> Dict[str, Any]:
    """Sends a text message."""
    payload = {"chat_id": str(chat_id), "text": text}
    if parse_mode:
        payload["parse_mode"] = parse_mode
    return await send_telegram_request("sendMessage", payload)


async def send_telegram_photo(chat_id: int, photo: str, caption: Optional[str] = None) -> Dict[str, Any]:
    """Sends a photo (file_id or URL)."""
    payload = {"chat_id": str(chat_id), "photo": photo}
    if caption:
        payload["caption"] = caption
    return await send_telegram_request("sendPhoto", payload)


async def send_telegram_audio(chat_id: int, audio: str, caption: Optional[str] = None, duration: Optional[int] = None) -> Dict[str, Any]:
    """Sends an audio file (file_id or URL). Can be a voice note if is_voice=True is added by Telegram."""
    payload = {"chat_id": str(chat_id), "audio": audio}
    if caption:
        payload["caption"] = caption
    if duration:
        payload["duration"] = duration  # Recommended for voice/audio
    return await send_telegram_request("sendAudio", payload)


async def send_telegram_voice(chat_id: int, voice: str, caption: Optional[str] = None, duration: Optional[int] = None) -> Dict[str, Any]:
    """Sends a voice note (file_id or URL)."""
    payload = {"chat_id": str(chat_id), "voice": voice}
    if caption:
        payload["caption"] = caption
    if duration:
        payload["duration"] = duration
    return await send_telegram_request("sendVoice", payload)


async def send_telegram_video(
    chat_id: int,
    video: str,  # This can be a file_id or a direct HTTP URL
    caption: Optional[str] = None,
    duration: Optional[int] = None,  # Video duration in seconds
    width: Optional[int] = None,    # Video width
    height: Optional[int] = None,   # Video height
    # Pass True if the uploaded video is suitable for streaming
    supports_streaming: Optional[bool] = None,
    # Mode for parsing entities in the video caption
    parse_mode: Optional[str] = None
) -> Dict[str, Any]:
    """
    Sends a video file to a specified chat ID.

    Args:
        chat_id (int): Unique identifier for the target chat.
        video (str): Video to send. Pass a file_id as a string to send a video that
                     exists on Telegram servers (recommended), or pass an HTTP URL
                     as a string for Telegram to get a video from the Internet.
        caption (Optional[str]): Video caption, 0-1024 characters after entities parsing.
        duration (Optional[int]): Duration of sent video in seconds.
        width (Optional[int]): Video width.
        height (Optional[int]): Video height.
        supports_streaming (Optional[bool]): Pass True if the uploaded video is suitable for streaming.
        parse_mode (Optional[str]): Mode for parsing entities in the caption. E.g., "MarkdownV2", "HTML".

    Returns:
        Dict[str, Any]: The JSON response from the Telegram API.
    """
    logger.info(f"Sending video to chat {chat_id} from: {video[:50]}...")
    payload = {
        "chat_id": str(chat_id),
        "video": video,
    }
    if caption:
        payload["caption"] = caption
    if duration:
        payload["duration"] = duration
    if width:
        payload["width"] = width
    if height:
        payload["height"] = height
    if supports_streaming is not None:
        payload["supports_streaming"] = supports_streaming
    if parse_mode:
        payload["parse_mode"] = parse_mode

    return await send_telegram_request("sendVideo", payload)


async def get_telegram_file_url(file_id: str) -> Optional[str]:
    """
    Retrieves the direct download URL for a file given its file_id.
    Returns None if the file_path is not found.
    """
    payload = {"file_id": file_id}
    try:
        response_data = await send_telegram_request("getFile", payload)
        if response_data.get("ok") and response_data.get("result"):
            file_path = response_data["result"].get("file_path")
            if file_path:
                download_url = f"https://api.telegram.org/file/bot{TELEGRAM_BOT_TOKEN}/{file_path}"
                return download_url
            else:
                logger.warning(
                    f"File path not found in getFile response for file_id {file_id}: {response_data}")
                return None
        else:
            logger.error(
                f"Error in getFile API response for file_id {file_id}: {response_data.get('description', 'Unknown error')}")
            return None
    except Exception as e:
        logger.error(f"Failed to get file URL for {file_id}: {e}")
        return None

# Optional: Client shutdown for clean exit


async def close_http_client():
    await http_client.aclose()
    logger.info("HTTP client for Telegram API closed.")
