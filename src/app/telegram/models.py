from pydantic import BaseModel, Field
from typing import List, Optional


class User(BaseModel):
    id: int
    is_bot: bool
    first_name: str
    last_name: Optional[str] = None
    username: Optional[str] = None


class AddAdmin(BaseModel):
    username: str
    name: str
    chat_id: str


class Chat(BaseModel):
    id: int
    type: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    username: Optional[str] = None


class PhotoSize(BaseModel):
    file_id: str
    file_unique_id: str
    width: int
    height: int
    file_size: Optional[int] = None


class Audio(BaseModel):
    file_id: str
    file_unique_id: str
    duration: int  # in seconds
    mime_type: Optional[str] = None
    file_size: Optional[int] = None
    title: Optional[str] = None
    performer: Optional[str] = None
    file_name: Optional[str] = None  # Added for completeness


class Voice(BaseModel):  # For voice notes
    file_id: str
    file_unique_id: str
    duration: int  # in seconds
    mime_type: Optional[str] = None
    file_size: Optional[int] = None


class Message(BaseModel):
    message_id: int
    from_user: Optional[User] = Field(None, alias="from")
    chat: Chat
    date: int
    # Will be None for voice/audio notes unless a caption is present
    text: Optional[str] = None

    # Media fields specific to your use case
    photo: Optional[List[PhotoSize]] = None
    audio: Optional[Audio] = None  # For regular audio files
    voice: Optional[Voice] = None  # For voice notes
    caption: Optional[str] = None  # Caption for media messages


class Update(BaseModel):
    update_id: int
    message: Optional[Message] = None
    edited_message: Optional[Message] = None
    # Add other update types if you might need them later (e.g., callback_query, channel_post)
