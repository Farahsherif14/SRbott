import asyncio
import fcntl
import json
import logging
from io import BytesIO
from pathlib import Path
import wave
from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command
from aiogram.types import Message
from vosk import KaldiRecognizer, Model
from app.config import settings
from app.flow import run_request_flow
from app.flow import has_arabic
from app.db import init_db
from app.session import clear_session, load_session

bot = Bot(settings.telegram_bot_token)
dp = Dispatcher()
vosk_model = None
logger = logging.getLogger(__name__)
BOT_BUILD = "2026-11-01-followup-fix-2"

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

@dp.message(Command("version"))
async def version(message: Message):
    await message.answer(
        f"إصدار البوت: {BOT_BUILD}\nملف التشغيل: {Path(__file__).resolve()}"
        if language_for(message) == "ar" else
        f"Bot build: {BOT_BUILD}\nRunning file: {Path(__file__).resolve()}"
    )

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

async def intake_text(message: Message, text: str, *, is_voice: bool = False):
    try:
        result = await asyncio.to_thread(run_request_flow, chat_id=str(message.chat.id),
            user_id=str(message.from_user.id), message=text, transcript=text,
            is_voice=is_voice)
    except Exception as exc:
        detail = str(exc) if str(exc).startswith(("Extraction relay error:",
                                                  "Extraction relay is unreachable")) else type(exc).__name__
        logger.error("Request processing failed: %s", detail)
        lang = language_for(message, text)
        await message.answer("تعذر معالجة الطلب الآن. حاول مرة أخرى بعد قليل." if lang == "ar" else "I couldn’t process that request just now. Please try again shortly.")
        return
    await message.answer(result.get("response") or "I’ve updated your request details.")

@dp.message(F.voice)
async def voice(message: Message):
    global vosk_model
    language = language_for(message)
    if message.voice.file_size and message.voice.file_size > 25 * 1024 * 1024:
        await message.answer("التسجيل كبير جدًا. أرسل رسالة صوتية أقصر." if language == "ar"
                             else "That recording is too large. Please send a shorter voice message.")
        return
    try:
        info = await bot.get_file(message.voice.file_id)
        audio_file = BytesIO()
        await bot.download_file(info.file_path, destination=audio_file)
        if vosk_model is None:
            model_path = Path(settings.vosk_model_path).expanduser()
            if not model_path.is_dir():
                raise RuntimeError(f"Vosk Arabic model not found: {model_path}")
            vosk_model = Model(str(model_path))
        text = await asyncio.to_thread(transcribe_vosk, audio_file.getvalue(), vosk_model)
    except Exception:
        await message.answer("تعذر تحويل التسجيل إلى نص الآن. حاول مرة أخرى أو أرسل التفاصيل كتابةً."
                             if language == "ar"
                             else "I couldn’t transcribe that recording just now. Please try again or type the details.")
        return
    if not text:
        await message.answer("لم أتمكن من سماع التسجيل بوضوح. أرسله مرة أخرى أو اكتب طلبك." if language == "ar" else "I couldn’t hear that clearly. Please try again or type your request.")
        return
    await intake_text(message, text, is_voice=True)


def transcribe_vosk(audio_content: bytes, model: Model) -> str:
    """Decode Telegram OGG/Opus with the bundled ffmpeg binary, then run Vosk."""
    import subprocess
    from imageio_ffmpeg import get_ffmpeg_exe

    decoded = subprocess.run(
        [get_ffmpeg_exe(), "-v", "error", "-i", "pipe:0", "-f", "wav", "-ac", "1",
         "-ar", "16000", "-acodec", "pcm_s16le", "pipe:1"],
        input=audio_content, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        check=True, timeout=120,
    ).stdout
    with wave.open(BytesIO(decoded), "rb") as wav:
        pcm = wav.readframes(wav.getnframes())
    recognizer = KaldiRecognizer(model, 16000)
    recognizer.SetWords(False)
    for offset in range(0, len(pcm), 8000):
        recognizer.AcceptWaveform(pcm[offset:offset + 8000])
    result = json.loads(recognizer.FinalResult())
    return result.get("text", "").strip()

@dp.message(F.text & ~F.text.startswith("/"))
async def text(message: Message):
    await intake_text(message, message.text)

async def main():
    if not settings.telegram_bot_token:
        raise RuntimeError("Set TELEGRAM_BOT_TOKEN before starting the bot")
    lock_path = Path(__file__).resolve().parent.parent / ".service-request-bot.lock"
    lock_file = lock_path.open("a+")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        lock_file.close()
        raise SystemExit("The Service Request bot is already running. Stop that instance before starting another.") from exc
    init_db()
    print(f"Service Request bot build {BOT_BUILD} loaded from {Path(__file__).resolve()}", flush=True)
    try:
        await dp.start_polling(bot)
    finally:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        lock_file.close()

if __name__ == "__main__": asyncio.run(main())
