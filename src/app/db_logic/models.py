from sqlalchemy import Column, Integer, String, Boolean, ForeignKey
from sqlalchemy.sql.sqltypes import TIMESTAMP
from sqlalchemy.sql.expression import text
from sqlalchemy import select
from .database import Base, engine
from sqlalchemy.ext.asyncio import AsyncSession
import asyncio


class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True, nullable=False)
    name = Column(String, nullable=False, unique=False)
    created_at = Column(TIMESTAMP(timezone=True),
                        server_default=text('now()'), nullable=False)
    username = Column(String, nullable=False, unique=False)
    chat_id = Column(String, nullable=False, unique=False)


class Appointment(Base):
    __tablename__ = 'appointments'
    id = Column(Integer, primary_key=True, nullable=False)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    name = Column(String, nullable=False)
    username = Column(String, nullable=False)
    chat_id = Column(String, nullable=False)
    appointment_date = Column(String, nullable=False)
    appointment_time = Column(String, nullable=False)
    is_confirmed = Column(Boolean, nullable=False, default=False)
    created_at = Column(TIMESTAMP(timezone=True),
                        server_default=text('now()'), nullable=False)


class AdminUser(Base):
    __tablename__ = 'admin_users'
    id = Column(Integer, primary_key=True, nullable=False)
    # user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    name = Column(String, nullable=False)
    username = Column(String, nullable=False)
    chat_id = Column(String, nullable=False)


async def create_tables():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("All tables needed for applications confirmed to exist")


# //TODO: Get your chat id and user id to insert in table print it out first, for testing make yourself the defualt admin
async def check_insert_admin_users(admin_user):
    """
    Insert defualt admin parameters
    """

    # Admin users to check/insert
    # admin_user = [
    #     {"chat_id": "7950346489",  "username": "RaymondAkachi", "name": "Akachi"},
    #     # {"phone_number": "2348032235209", "user_name": "Dad 1"}
    # ]

    async with AsyncSession(engine) as session:
        try:
            for admin in admin_user:
                # Check if phone_number exists
                result = await session.execute(
                    select(AdminUser).where(
                        AdminUser.username == admin["username"])
                )
                existing_user = result.scalars().first()

                if existing_user:
                    print(
                        f"Username: {admin['username']} already exists with user_name")
                else:
                    # Insert new admin user
                    new_user = AdminUser(
                        name=admin["name"],
                        username=admin["username"],
                        chat_id=admin["chat_id"]
                    )
                    session.add(new_user)
                    await session.commit()
                    print(
                        f"Inserted {admin['name']} with username: {admin['username']}")

        except Exception as e:
            print(f"Error: {e}")
            await session.rollback()


if __name__ == "__main__":
    # asyncio.run(create_tables())
    asyncio.run(check_insert_admin_users(
        {"chat_id": "7950346489",  "username": "RaymondAkachi", "name": "Akachi"}))


# async def add_admin_users():
#     try:
#         async with AsyncSession(engine) as session:
#             for user in [{'user_id': 2, 'user_name': "*******", 'phone_number': "*******"}]:
#                 new_admin_user = AdminUser(**user)
#                 session.add(new_admin_user)
#             await session.commit()
#             await session.refresh(new_admin_user)
#     except BaseException as e:
#         print(e)
# if __name__ == "__main__":
#     asyncio.run(add_admin_users())
