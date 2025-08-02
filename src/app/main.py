from fastapi import FastAPI, Depends, Request, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
# from .main_graph import create_workflow_graph
from qdrant_client import AsyncQdrantClient
# from app_reminder import setup_scheduler, schedule_appointment_reminder
# from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
# from .components.polly import VoiceCreation
from pytz import timezone
from .RAG.validator import TopicValidator
from .db_logic.database import engine
from .p_and_c.prayer_embeddings import PrayerRelation
from .p_and_c.counselling_embeddings import CounsellingRelation
from .db_logic.models import create_tables, Appointment, check_insert_admin_users
# from .components.query_rewriters import update_chat_history
from typing import AsyncGenerator
# import httpx
# import json
import asyncio
import logging
from contextlib import asynccontextmanager
from sqlalchemy import delete
from zoneinfo import ZoneInfo
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
# from apscheduler.triggers.date import DateTrigger
from apscheduler.triggers.cron import CronTrigger
from datetime import datetime, timedelta
from .settings import settings
# import traceback
# from .components import response_types
import app.telegram.models as telegram_models
from app.telegram.config import TELEGRAM_WEBHOOK_SECRET_TOKEN
from app.settings import settings
from . import background_process_message

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define timezone (West Africa Time)
WAT = ZoneInfo("Africa/Lagos")
WAT_timezone = timezone('Africa/Lagos')

# Global scheduler instance (for access without passing as argument)
scheduler = None

# Initialize FastAPI app

# Constants
ASSEMBLYAI_API_KEY = settings.ASSEMBLYAI_API_KEY
WHATSAPP_API_AUTHORIZATION = settings.WHATSAPP_API_AUTHORIZATION


async def get_async_db():
    """Dependency for async database session."""
    async with AsyncSession(engine) as session:
        yield session


def setup_scheduler():
    """Initialize the scheduler with a SQLAlchemy job store."""
    jobstores = {
        'default': SQLAlchemyJobStore(
            url=settings.NON_ASYNC_DATABASE_URL
        )
    }

    global scheduler
    scheduler = AsyncIOScheduler(jobstores=jobstores, timezone=WAT)
    scheduler.add_job(
        check_and_delete_old_appointments,
        trigger=CronTrigger(day="28-31", hour=23, minute=59, timezone=WAT),
        id="delete_old_appointments",
        replace_existing=True,
        args=[]  # Pass database session factory
    )

    return scheduler


async def delete_old_appointments(db: AsyncSession):
    """Delete appointments older than 30 days and remove associated scheduler jobs."""
    try:
        cutoff_date = datetime.now(WAT_timezone).date() - timedelta(days=30)
        result = await db.execute(
            select(Appointment).filter(
                Appointment.appointment_date < cutoff_date.strftime("%Y-%m-%d")
            )
        )
        old_appointments = result.scalars().all()
        if not old_appointments:
            logger.info("No appointments older than one month found.")
            return

        for appointment in old_appointments:
            await db.execute(
                delete(Appointment).filter(Appointment.id == appointment.id)
            )
            job_id = f"reminder_{appointment.id}"
            if scheduler and scheduler.get_job(job_id):
                scheduler.remove_job(job_id)
                logger.info(
                    f"Removed scheduler job for appointment {appointment.id}")
        await db.commit()
        logger.info(
            f"Deleted {len(old_appointments)} appointments older than {cutoff_date}")
    except Exception as e:
        logger.error(f"Failed to delete old appointments: {str(e)}")
        await db.rollback()


async def check_and_delete_old_appointments():
    async with AsyncSession(engine) as db:
        """Check if today is the last day of the month and run delete_old_appointments."""
        today = datetime.now(WAT).date()
        # Check if tomorrow is the 1st of the next month
        tomorrow = today + timedelta(days=1)
        if tomorrow.day == 1:
            logger.info("Running end-of-month appointment cleanup")
            await delete_old_appointments(db)
        else:
            logger.debug(
                f"Skipping cleanup; today ({today}) is not the last day of the month")

# Initialize scheduler
scheduler = setup_scheduler()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    FastAPI lifespan context manager for application startup and shutdown events.
    Code before `yield` runs on startup.
    Code after `yield` runs on shutdown.
    """
    # --- Startup Events ---
    scheduler.start()
    await create_tables()
    app.state.validators = {
        "rag": TopicValidator(),
        "prayer": PrayerRelation(),
        "counselling": CounsellingRelation()
    }
    logger.info("Scheduler started and validators initialized")
    logger.info("Validators initialized")  # Typo corrected from original

    # Yield control to the FastAPI application
    # The application will now start receiving requests
    yield

    # --- Shutdown Events ---
    scheduler.shutdown()
    for validator in app.state.validators.values():
        if hasattr(validator, 'client') and isinstance(validator.client, AsyncQdrantClient):
            await validator.client.close()
            logger.info(f"Closed Qdrant client for {type(validator).__name__}")
    logger.info("Scheduler stopped")
    logger.info("All Qdrant clients closed")

app = FastAPI(lifespan=lifespan)


def get_validators(request: Request) -> dict:
    """
    Dependency to get the initialized validators from the app state.
    """
    return request.app.state.validators


@app.get("/")
async def read_root():
    return {"message": "Telegram Voice/Audio/Text Bot Webhook is running! Send POST requests to /webhook"}


@app.get("/health_check")
async def health_check():
    return {
        "status": "healthy" if hasattr(app.state, "validators") and app.state.validators else "starting",
        "validators_initialized": bool(app.state.validators)
    }


@app.post(f"/webhook/{TELEGRAM_WEBHOOK_SECRET_TOKEN}" if TELEGRAM_WEBHOOK_SECRET_TOKEN else "/webhook")
async def telegram_webhook(request: Request, update: telegram_models.Update, validators: dict = Depends(get_validators)):
    """
    Receives Telegram webhook updates.
    Immediately returns 200 OK, then processes the update in the background.
    """
    # Optional: Verify X-Telegram-Bot-Api-Secret-Token header if you set it
    if TELEGRAM_WEBHOOK_SECRET_TOKEN:
        received_secret = request.headers.get(
            "X-Telegram-Bot-Api-Secret-Token")
        if not received_secret or received_secret != TELEGRAM_WEBHOOK_SECRET_TOKEN:
            logger.warning(
                "Unauthorized webhook access attempt with incorrect secret token.")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail="Unauthorized")

    logger.info(
        f"Received webhook update_id: {update.update_id}. Immediately returning 200 OK.")

    # IMPORTANT: Offload processing to a background task
    asyncio.create_task(background_process_message.process_user_message(
        update, validators, scheduler))

    return {"status": "ok"}  # Telegram expects a 200 OK response


@app.post("/add-admin")
async def add_admin_user(admin: telegram_models.AddAdmin):
    try:
        await check_insert_admin_users({"name": admin.name, "username": admin.username, "chat_id": admin.chat_id})
        return {"status": "success", "message": f"Admin user {admin.username} added successfully."}
    except Exception as e:
        logger.error(f"Error adding admin user: {str(e)}")
        raise HTTPException(
            status_code=500, detail="Failed to add admin user.")
