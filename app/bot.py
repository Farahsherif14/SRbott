import asyncio
import socket
import json
import logging
import os
from io import BytesIO
from pathlib import Path
import tempfile
import wave
from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command
from aiogram.types import Message
from app.config import settings
from app.flow import run_request_flow
from app.flow import has_arabic
from app.db import init_db
from app.session import clear_session, load_session

bot = Bot(settings.telegram_bot_token)
dp = Dispatcher()
asr_model = None
logger = logging.getLogger(__name__)
BOT_BUILD = "2026-11-10-windows-lock"

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
    global asr_model
    language = language_for(message)
    if message.voice.file_size and message.voice.file_size > 25 * 1024 * 1024:
        await message.answer("التسجيل كبير جدًا. أرسل رسالة صوتية أقصر." if language == "ar"
                             else "That recording is too large. Please send a shorter voice message.")
        return
    stage = "Telegram download"
    try:
        info = await bot.get_file(message.voice.file_id)
        audio_file = BytesIO()
        await bot.download_file(info.file_path, destination=audio_file)
        if asr_model is None:
            # Numba cannot write its cache into protected site-packages paths.
            os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "service-request-numba"))
            import nemo.collections.asr as nemo_asr

            model_path = Path(settings.tawasul_model_path).expanduser()
            if not model_path.is_file():
                raise RuntimeError(f"Tawasul NeMo checkpoint not found: {model_path}")
            asr_model = nemo_asr.models.ASRModel.restore_from(str(model_path), map_location="cpu")
            asr_model.eval()
        stage = "audio decoding / Tawasul inference"
        text = await asyncio.to_thread(transcribe_tawasul, audio_file.getvalue(), asr_model)
    except Exception as exc:
        logger.exception("Voice transcription failed during %s (%s)", stage, type(exc).__name__)
        await message.answer("تعذر تحويل التسجيل إلى نص الآن. حاول مرة أخرى أو أرسل التفاصيل كتابةً."
                             if language == "ar"
                             else "I couldn’t transcribe that recording just now. Please try again or type the details.")
        return
    if not text:
        await message.answer("لم أتمكن من سماع التسجيل بوضوح. أرسله مرة أخرى أو اكتب طلبك." if language == "ar" else "I couldn’t hear that clearly. Please try again or type your request.")
        return
    await intake_text(message, text, is_voice=True)


def transcribe_tawasul(audio_content: bytes, model) -> str:
    """Decode Telegram OGG/Opus to mono 16 kHz WAV and run Tawasul ASR."""
    import subprocess
    from imageio_ffmpeg import get_ffmpeg_exe

    decoded = subprocess.run(
        [get_ffmpeg_exe(), "-v", "error", "-i", "pipe:0", "-f", "wav", "-ac", "1",
         "-ar", "16000", "-acodec", "pcm_s16le", "pipe:1"],
        input=audio_content, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        check=True, timeout=120,
    ).stdout
    # NeMo's transcribe API accepts filenames; use a temporary WAV and remove it
    # automatically when inference finishes.
    with tempfile.NamedTemporaryFile(suffix=".wav") as audio_path:
        audio_path.write(decoded)
        audio_path.flush()
        prediction = model.transcribe([audio_path.name])
    item = prediction[0] if prediction else ""
    return (item.text if hasattr(item, "text") else str(item)).strip()

@dp.message(F.text & ~F.text.startswith("/"))
async def text(message: Message):
    await intake_text(message, message.text)

async def main():
    if not settings.telegram_bot_token:
        raise RuntimeError("Set TELEGRAM_BOT_TOKEN before starting the bot")
    # Single-instance guard that works on Windows and Linux (the old fcntl lock is Unix-only).
    lock_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        lock_socket.bind(("127.0.0.1", 47653))
    except OSError as exc:
        lock_socket.close()
        raise SystemExit("The Service Request bot is already running. Stop that instance before starting another.") from exc
    init_db()
    print(f"Service Request bot build {BOT_BUILD} loaded from {Path(__file__).resolve()}", flush=True)
    try:
        await dp.start_polling(bot)
    finally:
        lock_socket.close()

if __name__ == "__main__": asyncio.run(main())
