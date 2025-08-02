from sqlalchemy.ext.asyncio import AsyncSession
from app.db_logic.models import AdminUser
from app.db_logic.database import engine
from sqlalchemy import select


async def is_admin(username):
    async with AsyncSession(engine) as session:
        result = await session.execute(select(AdminUser.username).where(AdminUser.username == username))
        return result.scalar_one_or_none()
