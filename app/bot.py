import asyncio, tempfile
from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command
from aiogram.types import Message
from faster_whisper import WhisperModel
from app.config import settings
from app.flow import run_request_flow
from app.flow import has_arabic
from app.session import clear_session, load_session

bot = Bot(settings.telegram_bot_token)
dp = Dispatcher()
whisper = None

def language_for(message: Message, text: str = "") -> str:
    session = load_session(str(message.chat.id), str(message.from_user.id))
    return session.get("response_language") or (
        "ar" if has_arabic(text) or (message.from_user.language_code or "").startswith("ar") else "en")

@dp.message(Command("start"))
async def start(message: Message):
    if language_for(message) == "ar":
        await message.answer("مرحبًا! اشرح مشكلة الصيانة بالعربية أو الإنجليزية، وسأجمع التفاصيل ثم أطلب منك تأكيدها.")
    else:
        await message.answer("Hello! Describe your maintenance issue in Arabic or English. I’ll collect details and ask you to confirm before saving it.")

@dp.message(Command("reset"))
async def reset(message: Message):
    language = language_for(message)
    clear_session(str(message.chat.id))
    await message.answer("تم مسح المحادثة. اشرح طلبك للبدء من جديد." if language == "ar" else "Session cleared. Describe your request to start again.")

@dp.message(Command("forget"))
async def forget(message: Message):
    language = language_for(message)
    from app.main import forget as forget_profile
    forget_profile(str(message.from_user.id)); clear_session(str(message.chat.id))
    await message.answer("تم حذف بيانات التواصل المحفوظة." if language == "ar" else "Your saved contact details were removed.")

@dp.message(Command("status"))
async def status(message: Message):
    language = language_for(message)
    ticket = (message.text or "").split(maxsplit=1)
    if len(ticket) < 2:
        await message.answer("استخدم: /status SR-XXXXXXXX" if language == "ar" else "Use /status SR-XXXXXXXX")
        return
    from app.main import status as lookup
    from app.schemas import StatusRequest
    try: result = lookup(StatusRequest(ticket_number=ticket[1]))
    except Exception: result = None
    if result:
        if language == "ar":
            states = {"open": "مفتوحة", "closed": "مغلقة", "in_progress": "قيد التنفيذ"}
            await message.answer(f"التذكرة {result['ticket_number']}: {states.get(result['status'], 'قيد المتابعة')}")
        else:
            await message.answer(f"{result['ticket_number']}: {result['status']}")
    else:
        await message.answer("لم يتم العثور على التذكرة." if language == "ar" else "Ticket not found.")

async def intake_text(message: Message, text: str):
    try:
        result = await asyncio.to_thread(run_request_flow, chat_id=str(message.chat.id),
            user_id=str(message.from_user.id), message=text, transcript=text)
    except Exception:
        lang = language_for(message, text)
        await message.answer("تعذر معالجة الطلب الآن. حاول مرة أخرى بعد قليل." if lang == "ar" else "I couldn’t process that request just now. Please try again shortly.")
        return
    await message.answer(result.get("response") or "I’ve updated your request details.")

@dp.message(F.voice)
async def voice(message: Message):
    global whisper
    if whisper is None: whisper = WhisperModel(settings.whisper_model, device="cpu", compute_type="int8")
    info = await bot.get_file(message.voice.file_id)
    with tempfile.NamedTemporaryFile(suffix=".ogg") as f:
        await bot.download_file(info.file_path, destination=f.name)
        segments, _ = whisper.transcribe(f.name, language=None)
        text = " ".join(s.text for s in segments).strip()
    if not text:
        language = language_for(message)
        await message.answer("لم أتمكن من سماع التسجيل بوضوح. أرسله مرة أخرى أو اكتب طلبك." if language == "ar" else "I couldn’t hear that clearly. Please try again or type your request.")
        return
    await intake_text(message, text)

@dp.message(F.text & ~F.text.startswith("/"))
async def text(message: Message):
    await intake_text(message, message.text)

async def main():
    if not settings.telegram_bot_token: raise RuntimeError("Set TELEGRAM_BOT_TOKEN before starting the bot")
    await dp.start_polling(bot)

if __name__ == "__main__": asyncio.run(main())
