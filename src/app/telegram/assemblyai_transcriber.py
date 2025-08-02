import httpx
import logging
from typing import Optional
from app.telegram.config import ASSEMBLYAI_API_KEY  # Import configuration
import asyncio

logger = logging.getLogger("assemblyai_transcriber")


async def transcribe_audio_from_url(audio_url: str) -> Optional[str]:
    """
    Transcribes audio from a given public URL using AssemblyAI.
    Returns the transcribed text or None if transcription fails.
    """

    # Log first 50 chars
    logger.info(
        f"Starting AssemblyAI transcription for URL: {audio_url[:50]}...")
    try:
        # Step 2: Download the audio file from WhatsApp
        async with httpx.AsyncClient() as client:
            audio_response = await client.get(audio_url)
            audio_response.raise_for_status()
            audio_data = audio_response.content  # raw bytes of the audio file

        # Step 3: Upload audio file to AssemblyAI
        assembly_headers = {
            "authorization": ASSEMBLYAI_API_KEY,
        }

        async with httpx.AsyncClient() as client:
            upload_response = await client.post(
                "https://api.assemblyai.com/v2/upload",
                headers=assembly_headers,
                content=audio_data
            )
            upload_response.raise_for_status()
            upload_data = upload_response.json()
            # This is what AssemblyAI will transcribe
            upload_url = upload_data["upload_url"]

        # Step 4: Request transcription using the uploaded file
        transcript_request = {
            "audio_url": upload_url
        }

        async with httpx.AsyncClient() as client:
            transcript_response = await client.post(
                "https://api.assemblyai.com/v2/transcript",
                headers={**assembly_headers,
                         "content-type": "application/json"},
                json=transcript_request
            )
            transcript_response.raise_for_status()
            transcript_data = transcript_response.json()
            transcript_id = transcript_data["id"]

        # Step 5: Poll AssemblyAI until transcription is complete
        polling_url = f"https://api.assemblyai.com/v2/transcript/{transcript_id}"

        while True:
            async with httpx.AsyncClient() as client:
                poll_response = await client.get(polling_url, headers=assembly_headers)
                poll_response.raise_for_status()
                poll_data = poll_response.json()

                if poll_data["status"] == "completed":
                    return poll_data["text"]
                elif poll_data["status"] == "failed":
                    logger.warning(f"Transcription failed: {poll_data}")

            await asyncio.sleep(3)  # Wait a few seconds before polling again
    except Exception as e:
        logger.error(
            f"An unexpected error occurred during AssemblyAI transcription of {audio_url[:50]}: {e}")
        return None
