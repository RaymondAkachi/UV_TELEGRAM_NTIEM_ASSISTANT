import logging
# Make sure to import all relevant Pydantic models
from app.telegram.models import Update, Message
import app.telegram.telegram_api as telegram_api
import app.telegram.assemblyai_transcriber as assemblyai_transcriber
from .config import MAX_AUDIO_DURATION_SECONDS  # Import max duration
from typing import Optional

logger = logging.getLogger("message_handler")


async def process_telegram_update_background(update: Update):
    """
    This function processes the Telegram update in a background task.
    It filters messages (text, audio, voice), transcribes audio/voice (if applicable),
    and orchestrates responses using preferred methods (text, image/audio via URL).
    """
    if not update.message:
        logger.info(
            f"Update {update.update_id} received, but no message. Skipping processing.")
        return

    message = update.message
    chat_id = message.chat.id

    logger.info(
        f"Starting background processing for chat_id {chat_id}, update_id {update.update_id}")

    # This variable will hold the text that our bot's core logic will process
    processed_text: Optional[str] = None
    message_type_received: str = "unrecognized"  # Default to 'unrecognized'

    # --- Step 1: Filter Message Types and Prepare Text for Processing ---

    # Prioritize direct text messages
    if message.text:
        processed_text = message.text.strip()
        message_type_received = "text message"
        logger.info(
            f"Received text message from chat ID {chat_id}: \"{processed_text[:100]}...\"")

    # If no text, check for voice notes
    elif message.voice:
        file_id_to_transcribe = message.voice.file_id
        audio_duration = message.voice.duration
        message_type_received = "voice note"
        logger.info(
            f"Received voice note from chat ID {chat_id}. File ID: {file_id_to_transcribe}")

        if audio_duration is not None and audio_duration > MAX_AUDIO_DURATION_SECONDS:
            logger.warning(
                f"{message_type_received} duration ({audio_duration}s) exceeds max limit ({MAX_AUDIO_DURATION_SECONDS}s) for chat ID {chat_id}.")
            try:
                await telegram_api.send_telegram_text(chat_id, f"That {message_type_received} is too long ({audio_duration} seconds). Please send an audio file or voice note shorter than {MAX_AUDIO_DURATION_SECONDS} seconds.")
            except Exception as e:
                logger.error(
                    f"Failed to send audio too long warning to chat {chat_id}: {e}")
            return  # Exit if audio is too long

        # Get direct URL and transcribe
        try:
            audio_url = await telegram_api.get_telegram_file_url(file_id_to_transcribe)
            if not audio_url:
                logger.error(
                    f"Could not get direct URL for {message_type_received} {file_id_to_transcribe} from Telegram.")
                await telegram_api.send_telegram_text(chat_id, f"Sorry, I had trouble accessing your {message_type_received}. Please try again.")
                return
            logger.info(
                f"Got direct URL for {message_type_received}: {audio_url[:50]}...")

            await telegram_api.send_telegram_text(chat_id, f"Got your {message_type_received}! Please wait a moment while I transcribe it...")
            processed_text = await assemblyai_transcriber.transcribe_audio_from_url(audio_url)
            if not processed_text:
                await telegram_api.send_telegram_text(chat_id, f"I couldn't get text from your {message_type_received}. Please speak clearly or try a different file.")
                return  # Exit if transcription failed
            logger.info(
                f"Transcribed text from chat ID {chat_id}: \"{processed_text[:100]}...\"")

        except Exception as e:
            logger.error(
                f"Error during {message_type_received} processing for {file_id_to_transcribe}: {e}")
            await telegram_api.send_telegram_text(chat_id, f"Sorry, there was an error processing your {message_type_received}. Please try again.")
            return

    # If no text and no voice, check for audio files
    elif message.audio:
        file_id_to_transcribe = message.audio.file_id
        audio_duration = message.audio.duration
        message_type_received = "audio file"
        logger.info(
            f"Received audio file from chat ID {chat_id}. File ID: {file_id_to_transcribe}")

        if audio_duration is not None and audio_duration > MAX_AUDIO_DURATION_SECONDS:
            logger.warning(
                f"{message_type_received} duration ({audio_duration}s) exceeds max limit ({MAX_AUDIO_DURATION_SECONDS}s) for chat ID {chat_id}.")
            try:
                await telegram_api.send_telegram_text(chat_id, f"That {message_type_received} is too long ({audio_duration} seconds). Please send an audio file or voice note shorter than {MAX_AUDIO_DURATION_SECONDS} seconds.")
            except Exception as e:
                logger.error(
                    f"Failed to send audio too long warning to chat {chat_id}: {e}")
            return  # Exit if audio is too long

        # Get direct URL and transcribe
        try:
            audio_url = await telegram_api.get_telegram_file_url(file_id_to_transcribe)
            if not audio_url:
                logger.error(
                    f"Could not get direct URL for {message_type_received} {file_id_to_transcribe} from Telegram.")
                await telegram_api.send_telegram_text(chat_id, f"Sorry, I had trouble accessing your {message_type_received}. Please try again.")
                return
            logger.info(
                f"Got direct URL for {message_type_received}: {audio_url[:50]}...")

            await telegram_api.send_telegram_text(chat_id, f"Got your {message_type_received}! Please wait a moment while I transcribe it...")
            processed_text = await assemblyai_transcriber.transcribe_audio_from_url(audio_url)
            if not processed_text:
                await telegram_api.send_telegram_text(chat_id, f"I couldn't get text from your {message_type_received}. Please speak clearly or try a different file.")
                return  # Exit if transcription failed
            logger.info(
                f"Transcribed text from chat ID {chat_id}: \"{processed_text[:100]}...\"")

        except Exception as e:
            logger.error(
                f"Error during {message_type_received} processing for {file_id_to_transcribe}: {e}")
            await telegram_api.send_telegram_text(chat_id, f"Sorry, there was an error processing your {message_type_received}. Please try again.")
            return

    # If none of the above specific types, it's an unsupported message type
    else:
        logger.info(
            f"Received unsupported message type from chat ID {chat_id}. Message: {message.dict()}")
        try:
            await telegram_api.send_telegram_text(chat_id, "I only process text, audio files, and voice notes. Please send me one of those!")
        except Exception as e:
            logger.error(
                f"Failed to send unsupported message type warning to chat {chat_id}: {e}")
        return  # Exit if unsupported message type

    # --- Step 2: Process the 'processed_text' and Formulate Response ---
    # This is where your core bot logic goes, operating solely on the 'processed_text'.
    if not processed_text:  # Should ideally be caught by specific errors above, but a final check
        logger.error(
            f"No text available for processing for chat ID {chat_id} after all attempts.")
        try:
            await telegram_api.send_telegram_text(chat_id, "I received your message but couldn't process it into text. Can you try again?")
        except Exception as e:
            logger.error(
                f"Failed to send generic processing error to chat {chat_id}: {e}")
        return

    response_text = f"You sent a {message_type_received}. "
    if message_type_received != "text message":
        response_text += f"I transcribed it as: \"{processed_text}\""
    else:
        response_text += f"You said: \"{processed_text}\""

    response_image_url: Optional[str] = None
    response_audio_url: Optional[str] = None

    # Example simple logic based on processed_text
    if "hello" in processed_text.lower():
        response_text += "\nHello there! How can I help you today?"
    if "image" in processed_text.lower() or "picture" in processed_text.lower():
        # Example image URL (replace with your actual hosted image links)
        response_image_url = "https://picsum.photos/400/300"
        response_text += "\nHere's an image for you from a link!"
    if "sound" in processed_text.lower() or "music" in processed_text.lower():
        # Example audio URL (replace with your actual hosted audio links)
        response_audio_url = "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-4.mp3"
        response_text += "\nHere's a short audio clip from a link!"

    # --- Step 3: Send Responses Back to Telegram ---
    try:
        # Always send the primary text response
        await telegram_api.send_telegram_text(chat_id, response_text)

        # Send media using links if determined by logic
        if response_image_url:
            await telegram_api.send_telegram_photo(chat_id, response_image_url, caption="From your request!")
        if response_audio_url:
            await telegram_api.send_telegram_audio(chat_id, response_audio_url, caption="Responding with audio!")

    except Exception as e:
        logger.error(f"Failed to send response(s) to chat {chat_id}: {e}")

    logger.info(
        f"Finished background processing for update_id {update.update_id}")
