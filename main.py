"""STF SUPPORT TEAM - Telegram customer support bot.

Support-only bot. Conversation context lives in RAM only and expires after
5 minutes of inactivity. No database, no persistent message storage.
"""

import asyncio
import base64
import io
import logging
import os
import re
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import httpx
from telegram import Update
from telegram.constants import ChatAction
from telegram.error import TelegramError
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

# ---------------------------------------------------------------- logging
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
log = logging.getLogger("stf-support")

# ----------------------------------------------------------------- config
BOT_NAME = "STF SUPPORT TEAM"

BOT_TOKEN = os.environ.get("BOT_TOKEN", "").strip()
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "").strip()
ADMIN_ID_RAW = os.environ.get("ADMIN_ID", "").strip()
MODEL = os.environ.get("OPENROUTER_MODEL", "openrouter/free").strip()

# Do not exit at import time; validate at runtime in main().
if not BOT_TOKEN or not OPENROUTER_API_KEY:
    log.warning("BOT_TOKEN and OPENROUTER_API_KEY are not configured yet.")

try:
    ADMIN_ID = int(ADMIN_ID_RAW) if ADMIN_ID_RAW else None
except ValueError:
    ADMIN_ID = None
if ADMIN_ID is None:
    log.warning("ADMIN_ID missing or invalid; admin-only features are disabled.")

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
IDLE_TIMEOUT = 5 * 60
CLEANUP_INTERVAL = 30
MAX_HISTORY = 8
MAX_INPUT_CHARS = 1000
MAX_REPLY_CHARS = 3500
RATE_LIMIT_COUNT = 5
RATE_LIMIT_WINDOW = 60
STARTED_AT = time.time()

SYSTEM_PROMPT = (
    f"You are the automated assistant of {BOT_NAME}, a customer support bot on Telegram.\n"
    "Rules:\n"
    "- ALWAYS reply in natural Hinglish using Hindi written in Roman script mixed with simple English. "
    "Never reply in Devanagari Hindi, Italian, or fully English. Explain technical terms in simple Hinglish. "
    "Keep commands, error codes, product names, and official technical identifiers unchanged where necessary.\n"
    "- Only help with customer-support topics: troubleshooting, explaining how to report a problem, and guiding the user to the official support team.\n"
    "- You have NO access to accounts, orders, payments or internal systems. Never claim to look anything up or change anything.\n"
    "- Never invent company policies, prices, refund rules, timelines, account details, server addresses, account status, or contact details. If you do not have verified information, say so and tell the user to contact the official support administrator.\n"
    "- Never ask for passwords, OTPs, PINs, card numbers, API keys or tokens. Tell users never to share them.\n"
    "- If the request is outside support (general chat, coding, writing, roleplay, opinions, etc.), politely decline and redirect to the official support team. Ask the user to describe a support issue.\n"
    "- Keep replies short, polite, plain text, and in Hinglish."
)

SECRET_PATTERNS = [
    re.compile(r"\b\d{6,}:[A-Za-z0-9_-]{30,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}"),
    re.compile(r"\b(otp|cvv|pin)\b\s*(is|=|:)?\s*\d{4,8}\b", re.I),
    re.compile(r"\b(password|passcode)\s*(=|:)\s*\S+", re.I),
]


@dataclass
class Session:
    history: list = field(default_factory=list)
    last_active: float = field(default_factory=time.time)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


sessions: dict[int, Session] = {}
rate_log: dict[int, deque] = {}


def get_session(user_id: int) -> Session:
    if user_id not in sessions:
        sessions[user_id] = Session()
    return sessions[user_id]


def rate_limited(user_id: int) -> bool:
    now = time.time()
    q = rate_log.setdefault(user_id, deque())
    while q and now - q[0] > RATE_LIMIT_WINDOW:
        q.popleft()
    if len(q) >= RATE_LIMIT_COUNT:
        return True
    q.append(now)
    return False


def is_admin(user_id: int) -> bool:
    return ADMIN_ID is not None and user_id == ADMIN_ID


def contains_secret(text: str) -> bool:
    return any(p.search(text) for p in SECRET_PATTERNS)


def clean_reply(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    for secret in (BOT_TOKEN, OPENROUTER_API_KEY):
        if secret and secret in text:
            text = text.replace(secret, "[removed]")
    if len(text) > MAX_REPLY_CHARS:
        text = text[:MAX_REPLY_CHARS].rstrip() + "…"
    return text


async def safe_reply(update: Update, text: str) -> None:
    try:
        await update.effective_message.reply_text(text)
    except TelegramError as exc:
        log.warning("Failed to send reply: %s", type(exc).__name__)


# ---------------------------------------------------------------- heuristics & canned support replies
CASUAL_PATTERNS = [
    r"^hi\b",
    r"^hello\b",
    r"^hey\b",
    r"how are you",
    r"what's up",
    r"sup\b",
]
CASUAL_RE = re.compile("|".join(CASUAL_PATTERNS), re.I)


def is_casual_message(text: str) -> bool:
    if not text:
        return False
    if len(text) > 120:
        return False
    return bool(CASUAL_RE.search(text))


def message_has_image(message) -> bool:
    if not message:
        return False
    if getattr(message, "photo", None):
        return True
    doc = getattr(message, "document", None)
    if doc and getattr(doc, "mime_type", "").startswith("image"):
        return True
    return False


def is_mt5_question(text: str) -> bool:
    if not text:
        return False
    t = text.lower()
    return ("metatrader" in t or "mt5" in t) and (
        "connect" in t or "connect kaise" in t or "connect kar" in t or "login" in t or "se connect" in t
    )


def mt5_canned_reply_hinglish() -> str:
    return (
        "MetaTrader 5 se connect karne ke liye yeh steps follow karo:\n"
        "1) MetaTrader 5 app open karo.\n"
        "2) 'Login to an existing account' ya 'Login' option select karo.\n"
        "3) Broker ne diya hua login/account number, password, aur correct server enter karo.\n"
        "4) Correct server broker hi provide karta hai. Agar server unknown hai, toh pehle batao kaunsa broker use kar rahe ho.\n"
        "5) Kabhi bhi apna password, OTP, ya sensitive details Telegram mein share mat karo."
    )


# ---------------------------------------------------------------- AI call and model capability
class AIError(Exception):
    pass


def model_supports_vision_heuristic(model_name: str) -> bool:
    n = model_name.lower()
    keys = ["vl", "vision", "gemini", "flash", "multimodal", "image", "img"]
    return any(k in n for k in keys)


async def get_model_capability(client: httpx.AsyncClient, model_name: str, timeout: float = 6.0) -> Tuple[Optional[Dict[str, Any]], Optional[bool]]:
    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}"} if OPENROUTER_API_KEY else {}
    try:
        resp = await client.get(OPENROUTER_MODELS_URL, headers=headers, timeout=timeout)
    except Exception as exc:
        log.info("Could not fetch OpenRouter models: %s", type(exc).__name__)
        return None, None

    if resp.status_code != 200:
        log.info("OpenRouter models endpoint returned status %s", resp.status_code)
        return None, None

    try:
        j = resp.json()
    except Exception:
        log.info("Invalid JSON from OpenRouter models endpoint")
        return None, None

    data = j.get("data") or j.get("models") or []
    if not isinstance(data, list):
        return None, None

    target = model_name.lower()
    for m in data:
        mid = (m.get("id") or "").lower()
        name = (m.get("name") or "").lower()
        desc = (m.get("description") or "").lower()
        if target == mid or target in mid or target in name or target in desc:
            modalities = m.get("modalities") or m.get("input") or m.get("capabilities")
            if isinstance(modalities, list):
                has_vision = any("image" in str(x).lower() or "vision" in str(x).lower() for x in modalities)
                return m, bool(has_vision)
            if any(k in name or k in desc for k in ("vision", "image", "multimodal", "flash", "gemini")):
                return m, True
            return m, False

    return None, None


async def ask_ai(client: httpx.AsyncClient, messages: list[Any]) -> str:
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {"model": MODEL, "messages": messages, "max_tokens": 600}

    async def _call_once() -> str:
        max_retries = 4
        backoff = 1.0
        last_error = "unknown"

        for attempt in range(1, max_retries + 1):
            try:
                resp = await client.post(OPENROUTER_URL, headers=headers, json=payload, timeout=30)
            except httpx.TimeoutException:
                last_error = "timeout"
                log.info("OpenRouter timeout on attempt %d", attempt)
                await asyncio.sleep(backoff)
                backoff *= 2
                continue
            except httpx.HTTPError as exc:
                last_error = type(exc).__name__
                log.warning("HTTP error contacting OpenRouter: %s", last_error)
                await asyncio.sleep(backoff)
                backoff *= 2
                continue

            status = resp.status_code
            if status == 200:
                try:
                    data = resp.json()
                except ValueError:
                    last_error = "invalid-json"
                    log.warning("OpenRouter returned invalid JSON")
                    await asyncio.sleep(backoff)
                    backoff *= 2
                    continue

                try:
                    choices = data.get("choices")
                    if not choices:
                        last_error = "empty choices"
                        log.info("OpenRouter returned empty choices")
                        await asyncio.sleep(backoff)
                        backoff *= 2
                        continue
                    msg = choices[0].get("message")
                    if not msg:
                        last_error = "missing message"
                        await asyncio.sleep(backoff)
                        backoff *= 2
                        continue
                    content = msg.get("content")
                except Exception:
                    last_error = "bad response format"
                    await asyncio.sleep(backoff)
                    backoff *= 2
                    continue

                if isinstance(content, str):
                    if content.strip():
                        return content.strip()
                    last_error = "empty response"
                    await asyncio.sleep(backoff)
                    backoff *= 2
                    continue

                if isinstance(content, list):
                    parts = []
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            t = item.get("text") or item.get("content")
                            if t:
                                parts.append(t.strip())
                        elif isinstance(item, dict) and item.get("type") == "image":
                            parts.append("[image output omitted]")
                    if parts:
                        return "\n\n".join(parts).strip()
                    last_error = "no text parts in content"
                    await asyncio.sleep(backoff)
                    backoff *= 2
                    continue

                last_error = "unknown content type"
                await asyncio.sleep(backoff)
                backoff *= 2
                continue

            last_error = f"HTTP {status}"
            if status in (401, 402, 403, 404):
                log.error("OpenRouter permanent error: %s", last_error)
                break

            if status == 429:
                retry_after = 0
                try:
                    j = resp.json()
                    retry_after = j.get("retry_after", 0)
                except Exception:
                    pass
                wait = max(backoff, retry_after)
                log.info("OpenRouter rate limited (429). Waiting %.1fs before retry", wait)
                await asyncio.sleep(wait)
                backoff *= 2
                continue

            if 500 <= status < 600:
                log.info("OpenRouter server error %s; attempt %d", status, attempt)
                await asyncio.sleep(backoff)
                backoff *= 2
                continue

            log.warning("OpenRouter unexpected status %s", status)
            break

        raise AIError(last_error)

    text = await _call_once()

    def _is_valid_hinglish(s: str) -> bool:
        if re.search(r"[\u0900-\u097F]", s):
            return False
        tokens = re.findall(r"\w+", s.lower())
        hinglish_tokens = {
            "kaise", "ka", "kya", "hai", "nahi", "karo", "kar", "login", "server",
            "broker", "account", "password", "otp", "error", "support", "issue"
        }
        if any(t in hinglish_tokens for t in tokens):
            return True
        if len(tokens) <= 6:
            return True
        return False

    if not _is_valid_hinglish(text):
        log.info("AI reply failed Hinglish validation; retrying once with stronger hint")
        retry_messages = []
        for m in messages:
            if m.get("role") == "system":
                retry_messages.append({"role": "system", "content": SYSTEM_PROMPT + "\nALWAYS REPLY IN HINGLISH IN ROMAN SCRIPT."})
            else:
                retry_messages.append(m)
        try:
            text2 = await _call_once()
            if _is_valid_hinglish(text2):
                return text2
        except AIError:
            pass
    return text


# --------------------------------------------------------------- commands
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await safe_reply(
        update,
        f"Welcome to {BOT_NAME}! 🛟\n\n"
        "Main support-only bot hoon — batao aapko kis issue mein help chahiye.\n"
        "Screenshot bhejo caption ke saath.\n"
        "Safety: kabhi password, OTP, PIN ya API keys yahan mat bhejo.\n\n"
        "Type /help for commands.",
    )


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await safe_reply(
        update,
        f"{BOT_NAME} - Help\n\n"
        "/start - welcome message\n"
        "/help - this message\n"
        "/clear - erase our current conversation\n"
        "/status - check if the bot is online\n\n"
        "Support issue ko short mein batao ya screenshot bhejo.\n"
        "Conversation memory 5 minutes tak hoti hai.\n"
        "Main account aapka access nahi kar sakta — account-specific help ke liye official support se contact karo.",
    )


async def cmd_clear(update: Update, context: ContextTypes.DEFAULT_TYPE):
    sessions.pop(update.effective_user.id, None)
    await safe_reply(update, "Aapki conversation clear kar di gayi. 🧹")


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = f"✅ {BOT_NAME} online hai."
    if is_admin(update.effective_user.id):
        uptime = int(time.time() - STARTED_AT)
        h, rem = divmod(uptime, 3600)
        m, s = divmod(rem, 60)
        text += (
            f"\n\n[Admin]\nUptime: {h}h {m}m {s}s"
            f"\nModel: {MODEL}"
            f"\nActive conversations: {len(sessions)}"
        )
        ms = context.application.bot_data.get("model_supports_vision")
        if ms is True:
            text += "\nModel vision: yes"
        elif ms is False:
            text += "\nModel vision: no"
        else:
            text += "\nModel vision: unknown"
    await safe_reply(update, text)


async def on_unknown_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await safe_reply(update, "Unknown command. Type /help to see what I can do.")


# ------------------------------------------------------------ chat handler
async def on_non_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await safe_reply(update, "Main text padh sakta hoon. Issue ko text mein batao ya screenshot bhejo caption ke saath.")


async def build_user_message_for_openrouter(text: Optional[str], image_data_url: Optional[str]):
    if image_data_url:
        content = []
        if text:
            content.append({"type": "text", "text": text})
        content.append({"type": "image_url", "image_url": {"url": image_data_url}})
        return content
    return text or ""


async def on_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.effective_user:
        return
    user_id = update.effective_user.id

    if rate_limited(user_id):
        await safe_reply(update, "Aap jaldi messages bhej rahe ho. Thoda ruk ke try karo.")
        return

    text = (update.message.text or update.message.caption or "").strip()
    has_image = message_has_image(update.message)

    if contains_secret(text):
        await safe_reply(
            update,
            "⚠️ Yeh message kis secret jaisa lagta hai (password/OTP). Maine isse process nahi kiya. Sensitive info yahan mat bhejo.",
        )
        return

    if is_mt5_question(text):
        await safe_reply(update, mt5_canned_reply_hinglish())
        return

    if is_casual_message(text):
        await safe_reply(update, "Hi! Main support bot hoon — batao aapko kis support issue mein help chahiye.")
        return

    if not text and not has_image:
        await safe_reply(update, "Please apna support issue short mein batao ya screenshot bhejo caption ke saath.")
        return

    if len(text) > MAX_INPUT_CHARS:
        await safe_reply(update, f"Message bahut lamba hai. Kripya {MAX_INPUT_CHARS} characters ke andar rakho.")
        return

    session = get_session(user_id)
    if session.lock.locked():
        await safe_reply(update, "Pehle wale message pe kaam chal raha hai. Thoda intezaar karo.")
        return

    image_data_url = None
    if has_image:
        ms = context.application.bot_data.get("model_supports_vision")
        if ms is None:
            try:
                _, cap = await get_model_capability(context.application.bot_data["http"], MODEL)
                context.application.bot_data["model_supports_vision"] = cap if cap is not None else model_supports_vision_heuristic(MODEL)
                ms = context.application.bot_data["model_supports_vision"]
            except Exception:
                ms = model_supports_vision_heuristic(MODEL)
        if not ms:
            await safe_reply(update, "Yeh model image analyze nahi kar sakta. Image ke liye vision-capable model use karo ya text mein apna issue likho.")
            return

        photo = update.message.photo[-1] if getattr(update.message, "photo", None) else None
        document = update.message.document if getattr(update.message, "document", None) else None
        file = await (photo.get_file() if photo else document.get_file())
        max_bytes = 5 * 1024 * 1024
        if getattr(file, "file_size", 0) and file.file_size > max_bytes:
            await safe_reply(update, "Image size zyada hai. Kripya chhota screenshot bhejo (5 MB se kam).")
            return
        try:
            bio = io.BytesIO()
            await file.download(out=bio)
            bio.seek(0)
            data = bio.read()
            if len(data) > max_bytes:
                await safe_reply(update, "Image size zyada hai. Chhota image bhejo.")
                return
            mime = "image/jpeg"
            if data.startswith(b"\x89PNG"):
                mime = "image/png"
            b64 = base64.b64encode(data).decode("ascii")
            image_data_url = f"data:{mime};base64,{b64}"
        except Exception:
            log.warning("Failed to download image from Telegram; skipping image")
            await safe_reply(update, "Image download mein problem aayi. Dobara try karo.")
            return

    async with session.lock:
        session.history.append({"role": "user", "content": text or "[screenshot]"})
        session.history = session.history[-MAX_HISTORY:]
        session.last_active = time.time()

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        user_content = await build_user_message_for_openrouter(text, image_data_url)
        messages.append({"role": "user", "content": user_content})

        try:
            await context.bot.send_chat_action(update.effective_chat.id, ChatAction.TYPING)
        except TelegramError:
            pass

        try:
            reply = clean_reply(await ask_ai(context.application.bot_data["http"], messages))
        except AIError as exc:
            log.error("AI request failed: %s", exc)
            session.history.pop()
            err = str(exc)
            if "429" in err or "rate" in err.lower():
                await safe_reply(update, "AI service temporarily rate-limited hai. Thode der baad try karo.")
            elif "404" in err or "not found" in err.lower():
                await safe_reply(update, "Configured model nahi mila. Admin se OPENROUTER_MODEL verify karo.")
            elif "401" in err or "403" in err:
                await safe_reply(update, "AI service authentication fail hui. Admin ko inform karo.")
            else:
                await safe_reply(update, "Sorry, main abhi help nahi kar pa raha. Thodi der baad try karo ya admin se contact karo.")
            return

        session.history.append({"role": "assistant", "content": reply})
        session.history = session.history[-MAX_HISTORY:]
        session.last_active = time.time()

    await safe_reply(update, reply)


async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE):
    name = type(context.error).__name__ if context.error else "Unknown"
    log.error("Unhandled error: %s", name)
    if name == "Conflict":
        log.warning("A 'Conflict' error often means multiple bot instances or replicas are running. Ensure only one Railway worker/replica runs this bot.")


# ------------------------------------------------------- background tasks
async def cleanup_loop():
    while True:
        await asyncio.sleep(CLEANUP_INTERVAL)
        now = time.time()
        expired = [uid for uid, s in sessions.items() if now - s.last_active > IDLE_TIMEOUT and not s.lock.locked()]
        for uid in expired:
            sessions.pop(uid, None)
        for uid in [u for u, q in rate_log.items() if not q or now - q[-1] > RATE_LIMIT_WINDOW]:
            rate_log.pop(uid, None)
        if expired:
            log.info("Expired %d idle conversation(s)", len(expired))


async def post_init(app: Application):
    app.bot_data["http"] = httpx.AsyncClient()
    try:
        _, cap = await get_model_capability(app.bot_data["http"], MODEL)
        app.bot_data["model_supports_vision"] = cap if cap is not None else model_supports_vision_heuristic(MODEL)
    except Exception:
        app.bot_data["model_supports_vision"] = model_supports_vision_heuristic(MODEL)
    app.bot_data["cleanup"] = asyncio.create_task(cleanup_loop())
    log.info("%s started", BOT_NAME)


async def post_shutdown(app: Application):
    task = app.bot_data.get("cleanup")
    if task:
        task.cancel()
    client = app.bot_data.get("http")
    if client:
        await client.aclose()


# ------------------------------------------------------------------- main
def main():
    if not BOT_TOKEN or not OPENROUTER_API_KEY:
        log.error("BOT_TOKEN and OPENROUTER_API_KEY environment variables are required.")
        sys.exit(1)

    app = (
        Application.builder()
        .token(BOT_TOKEN)
        .post_init(post_init)
        .post_shutdown(post_shutdown)
        .build()
    )

    private = filters.ChatType.PRIVATE
    app.add_handler(CommandHandler("start", cmd_start, filters=private))
    app.add_handler(CommandHandler("help", cmd_help, filters=private))
    app.add_handler(CommandHandler("clear", cmd_clear, filters=private))
    app.add_handler(CommandHandler("status", cmd_status, filters=private))
    app.add_handler(MessageHandler(private & filters.TEXT & ~filters.COMMAND, on_message))
    app.add_handler(MessageHandler(private & filters.PHOTO, on_message))
    app.add_handler(MessageHandler(private & filters.Document.IMAGE, on_message))
    app.add_handler(MessageHandler(private & filters.COMMAND, on_unknown_command))
    app.add_handler(
        MessageHandler(
            private & ~filters.TEXT & ~filters.COMMAND & ~filters.StatusUpdate.ALL,
            on_non_text,
        )
    )
    app.add_error_handler(on_error)

    # One polling instance only per process; Railway replicas must be 1.
    app.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)


if __name__ == "__main__":
    main()
