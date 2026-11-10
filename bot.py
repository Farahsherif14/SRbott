import asyncio
import os

import httpx
from aiogram import Bot, Dispatcher, F, types
from aiogram.filters import Command, CommandObject, CommandStart
from dotenv import load_dotenv

load_dotenv()
bot = Bot(token=os.environ["TELEGRAM_BOT_TOKEN"])
dp = Dispatcher()
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
UNAVAILABLE = "Sorry, the system is unavailable right now. / النظام غير متاح حالياً."

WELCOME = ("Hi! Describe your maintenance issue by text or voice note (Arabic or English).\n"
           "أهلاً! اكتب أو ابعت رسالة صوتية بالمشكلة اللي عندك.\n\n"
           "/status SR-000001 · /reset · /forget")


async def call(method: str, path: str, **kw) -> dict | None:
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.request(method, f"{API_URL}{path}", **kw)
            r.raise_for_status()
            return r.json()
    except Exception:
        return None


async def answer(message: types.Message, data: dict | None, prefix: str = ""):
    await message.answer(prefix + (data["reply"] if data else UNAVAILABLE))


@dp.message(CommandStart())
async def start(message: types.Message):
    await message.answer(WELCOME)


@dp.message(Command("reset"))
async def reset(message: types.Message):
    await answer(message, await call("POST", "/reset", json={
        "user_id": message.from_user.id, "chat_id": message.chat.id}))


@dp.message(Command("forget"))
async def forget(message: types.Message):
    await answer(message, await call("POST", "/forget", json={
        "user_id": message.from_user.id, "chat_id": message.chat.id}))


@dp.message(Command("status"))
async def status(message: types.Message, command: CommandObject):
    if not command.args:
        await message.answer("Usage: /status SR-000001")
        return
    await answer(message, await call("GET", f"/status/{command.args.strip()}",
                                     params={"user_id": message.from_user.id}))


@dp.message(F.voice)
async def voice(message: types.Message):
    await bot.send_chat_action(message.chat.id, "typing")
    buf = await bot.download(message.voice)
    data = await call("POST", "/voice", content=buf.getvalue(),
                      headers={"Content-Type": "application/octet-stream"},
                      params={"user_id": message.from_user.id, "chat_id": message.chat.id})
    prefix = f"🎙 {data['transcript']}\n\n" if data and data.get("transcript") else ""
    await answer(message, data, prefix)


@dp.message(F.text)
async def text(message: types.Message):
    await bot.send_chat_action(message.chat.id, "typing")
    await answer(message, await call("POST", "/message", json={
        "user_id": message.from_user.id, "chat_id": message.chat.id, "text": message.text}))


@dp.message()
async def other(message: types.Message):
    await message.answer("Please send text or a voice note. / ابعت نص أو رسالة صوتية.")


async def main():
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())