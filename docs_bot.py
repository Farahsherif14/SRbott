import asyncio
import os

from aiogram import Bot, Dispatcher, F, types
from aiogram.filters import Command, CommandStart
from dotenv import load_dotenv

import rag

load_dotenv()
bot = Bot(token=os.environ["TELEGRAM_BOT_TOKEN"])
dp = Dispatcher()
index = rag.Index(os.getenv("DOCS_DIR", "docs"))
ALLOWED = {int(x) for x in os.getenv("ALLOWED_USER_IDS", "").replace(" ", "").split(",") if x.isdigit()}


def allowed(message: types.Message) -> bool:
    """Public bot: anyone can ask questions. (If ALLOWED_USER_IDS is set, it becomes private again.)"""
    return not ALLOWED or message.from_user.id in ALLOWED


def is_admin(message: types.Message) -> bool:
    """/files and /reindex are only for the IDs listed in ALLOWED_USER_IDS."""
    return message.from_user.id in ALLOWED


@dp.message(Command("id"))
async def my_id(message: types.Message):
    await message.answer(f"Your Telegram user ID: {message.from_user.id}\n"
                         "Put it in ALLOWED_USER_IDS in your .env file, then restart the bot.")


@dp.message(CommandStart())
async def start(message: types.Message):
    if not allowed(message):
        await message.answer("This bot is private. Send /id and add your ID to ALLOWED_USER_IDS in .env.")
        return
    await message.answer("Ask me anything about the project. I answer only from the project files.")


@dp.message(Command("files"))
async def files(message: types.Message):
    if is_admin(message):
        await message.answer("\n".join(index.files) or "No files indexed. Put files in the docs folder, then /reindex.")


@dp.message(Command("reindex"))
async def reindex(message: types.Message):
    if is_admin(message):
        await asyncio.to_thread(index.build)
        await message.answer(f"Indexed {len(index.files)} files, {len(index.chunks)} chunks.")


@dp.message(F.text)
async def ask(message: types.Message):
    if not allowed(message):
        await message.answer("This bot is private. Send /id to get your user ID.")
        return
    await bot.send_chat_action(message.chat.id, "typing")
    try:
        reply = await asyncio.to_thread(rag.answer, index, message.text)
    except Exception as e:
        print("error:", type(e).__name__)
        reply = "Sorry, something went wrong. Check the terminal."
    await message.answer(reply[:4000])


async def main():
    mode = "private" if ALLOWED else "public (anyone can ask)"
    print(f"Indexed {len(index.files)} files, {len(index.chunks)} chunks. Mode: {mode}")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())