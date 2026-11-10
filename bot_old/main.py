"""Hello-world Telegram bot (polling mode). Run: python -m app.bot.main"""
import asyncio
import logging

from aiogram import Bot, Dispatcher
from aiogram.filters import Command, CommandStart
from aiogram.types import Message

from app.config import settings

logging.basicConfig(level=logging.INFO)  # never log message text or phone numbers
dp = Dispatcher()


@dp.message(CommandStart())
async def start(message: Message) -> None:
    await message.answer(
        "Hello! Describe your maintenance issue by text or voice note.\n"
        "مرحبا! اوصف مشكلة الصيانة برسالة نصية أو صوتية."
    )


@dp.message(Command("reset"))
async def reset(message: Message) -> None:
    # TODO (backend): clear Redis session for this chat
    await message.answer("Session cleared.")


@dp.message(Command("forget"))
async def forget(message: Message) -> None:
    # TODO (backend): delete saved profile from PostgreSQL
    await message.answer("Your saved profile has been deleted.")


@dp.message()
async def echo(message: Message) -> None:
    if message.voice:
        await message.answer("Voice note received (transcription not wired yet).")
    elif message.text:
        await message.answer(f"Echo: {message.text}")


async def main() -> None:
    bot = Bot(token=settings.telegram_bot_token)
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
