from .telegram.models import Update
from app.telegram.config import MAX_AUDIO_DURATION_SECONDS  # Import max duration
import app.telegram.telegram_api as telegram_api
import app.telegram.assemblyai_transcriber as assemblyai_transcriber
import logging
from .main_graph import create_workflow_graph
from app.components.query_rewriters import update_chat_history
from app.components.polly import VoiceCreation

logger = logging.getLogger("message_handler")


async def process_user_message(update: Update, validators, scheduler):
    """
    This function processes the Telegram update in a background task.
    It filters messages (text, audio, voice), transcribes audio/voice (if applicable),
    and orchestrates responses using preferred methods (text, image/audio via URL).
    """

    if not update.message:
        logger.info(
            f"Update {update.update_id} received, but no message. Skipping processing.")
        return

    from_audio_file = False

    message = update.message
    chat_id = message.chat.id

    logger.info(
        f"Starting background processing for chat_id {chat_id}, update_id {update.update_id}")

    logger.info(f"Message received: {update.model_dump()}")

    is_bot = message.from_user.is_bot
    if not is_bot:
        first_name = message.from_user.first_name
        last_name = message.from_user.last_name
        username = message.from_user.username

        name = first_name
        if first_name and last_name:
            name = first_name + ' ' + last_name

        name = name.strip()

        print(f"ChatID: {chat_id}, username: {username}")

        if message.text:
            processed_text = message.text.strip()
            message_type_received = "text message"
            logger.info(
                f"Received text message from chat ID {chat_id}: \"{processed_text[:100]}...\"")

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

        elif message.audio:
            # from_audio_file = True
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

        else:
            logger.info(
                f"Received unsupported message type from chat ID {chat_id}. Message: {message.model_dump()}")
            try:
                await telegram_api.send_telegram_text(chat_id, "I only process text, audio files, and voice notes. Please send me one of those!")
            except Exception as e:
                logger.error(
                    f"Failed to send unsupported message type warning to chat {chat_id}: {e}")
            return  # Exit if unsupported message type

        try:
            rag = validators['rag']
            prayer_embeddings = validators['prayer']
            counselling_embeddings = validators['counselling']

            graph = create_workflow_graph().compile()

            rag_validator = [rag]
            p_and_c_validators = {
                'validator': prayer_embeddings,
                "counselling_validator": counselling_embeddings
            }

            graph_input = {
                'user_request': processed_text,
                'name': name,
                'username': username,
                'chat_id': chat_id,
                'rag_validator': rag_validator,
                'p_and_c_validators': p_and_c_validators,
                'scheduler': [scheduler]
            }

            results = await graph.ainvoke(graph_input)
            output_format = results['output_format']
            response = results['response']
            user_request = results['user_request']
            app_state = results['app_state']

            await process_message_response(chat_id, response, from_audio_file, output_format, user_request, app_state)
        except Exception as e:
            logger.error(f"An error occured: {e}")


async def process_message_response(chat_id, response, from_audio_file, ouput_format, user_input, state):
    if ouput_format == 'text':
        if from_audio_file:
            url = VoiceCreation(response).text_to_speech()
            api_response = await telegram_api.send_telegram_voice(chat_id, url)
        else:
            api_response = await telegram_api.send_telegram_text(chat_id, response)
        await update_chat_history(chat_id, user_message=user_input, bot_response=response, state=state)
        return api_response

    if ouput_format == 'video':
        collective_result = ''
        for result in response:
            if result['match_type'] == 'title':
                link = result['s3VideoLink']
                text = f"We found this result based on the title you provided, watch the rest of the sermon here: {result['socialVideoLink']}\n"
            else:
                link = result['s3VideoLink']
                text = f"We found this result based on the date you provided, watch the rest of the sermon here: {result['socialVideoLink']}\n"
            response = await telegram_api.send_telegram_video(chat_id, link, text)
            collective_result += text
        await update_chat_history(chat_id, user_message=user_input, bot_response=collective_result, state=state)
        return response

    if ouput_format == 'image':
        response = await telegram_api.send_telegram_photo(chat_id, photo=response)
        await update_chat_history(chat_id, user_message=user_input, bot_response=f"Here is the link to the image we just created fpr you based on your query: {response}", state=state)
        return response
