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
# Never log message content, tokens or keys. Only event types and counts.
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
)
# httpx logs full request URLs, and Telegram URLs contain the bot token.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
log = logging.getLogger("stf-support")

# ----------------------------------------------------------------- config
BOT_NAME = "STF SUPPORT TEAM"  # fixed; users cannot change it

BOT_TOKEN = os.environ.get("BOT_TOKEN", "").strip()
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "").strip()
ADMIN_ID_RAW = os.environ.get("ADMIN_ID", "").strip()
MODEL = os.environ.get("OPENROUTER_MODEL", "nvidia/nemotron-3.5-super-vl:free").strip()

# Do not exit at import time (helps tests). Validate at runtime in main().
if not BOT_TOKEN or not OPENROUTER_API_KEY:
    log.warning("BOT_TOKEN and OPENROUTER_API_KEY environment variables are not set. Runtime will fail unless provided.")

try:
    ADMIN_ID = int(ADMIN_ID_RAW) if ADMIN_ID_RAW else None
except ValueError:
    ADMIN_ID = None
if ADMIN_ID is None:
    log.warning("ADMIN_ID missing or invalid; admin-only features are disabled.")

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
IDLE_TIMEOUT = 5 * 60        # seconds of inactivity before context is erased
CLEANUP_INTERVAL = 30        # seconds between cleanup passes
MAX_HISTORY = 8              # messages kept per user (RAM only)
MAX_INPUT_CHARS = 1000
MAX_REPLY_CHARS = 3500
RATE_LIMIT_COUNT = 5         # messages allowed ...
RATE_LIMIT_WINDOW = 60       # ... per this many seconds, per user
STARTED_AT = time.time()

# SYSTEM PROMPT now enforces Hinglish and strict support-only behaviour.
SYSTEM_PROMPT = (
    f"You are the automated assistant of {BOT_NAME}, a customer support bot on Telegram.\n"
    "Rules:\n"
    "- ALWAYS reply in natural Hinglish using Hindi written in Roman script mixed with simple English. "
    "Never reply in Devanagari Hindi, Italian, or fully English. Explain technical terms in simple Hinglish. "
    "Keep commands, error codes, product names, and official technical identifiers unchanged where necessary.\n"
    "- Only help with customer-support topics: troubleshooting, explaining how to report a problem, and guiding the user to the official support team.\n"
    "- You have NO access to accounts, orders, payments or internal systems. Never claim to look anything up or change anything.\n"
    "- Never invent company policies, prices, refund rules, timelines, account details or contact details. If you do not have verified information, say so and tell the user to contact the official support administrator.\n"
    "- Never ask for passwords, OTPs, PINs, card numbers, API keys or tokens. Tell users never to share them.\n"
    "- If the request is outside support (general chat, coding, writing, roleplay, opinions, etc.), politely decline and redirect to the official support team. Ask the user to describe a support issue.\n"
    "- Keep replies short, polite, plain text, and in Hinglish."
)

# Messages that look like they contain secrets are NOT sent to the AI or stored.
SECRET_PATTERNS = [
    re.compile(r"\b\d{6,}:[A-Za-z0-9_-]{30,}\b"),                 # Telegram bot token
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}"),                        # API-key style
    re.compile(r"\b(otp|cvv|pin)\b\s*(is|=|:)?\s*\d{4,8}\b", re.I),
    re.compile(r"\b(password|passcode)\s*(=|:)\s*\S+", re.I),
]


# ------------------------------------------------------------ RAM storage
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


# ------------------------------------------------------------------ utils
def is_admin(user_id: int) -> bool:
    return ADMIN_ID is not None and user_id == ADMIN_ID


def contains_secret(text: str) -> bool:
    return any(p.search(text) for p in SECRET_PATTERNS)


def clean_reply(text: str) -> str:
    """Redact our own credentials if the model ever echoes them, and cap length."""
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


# ---------------------------------------------------------- heuristics & canned replies
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


def is_mt5_question(text: str) -> bool:
    if not text:
        return False
    t = text.lower()
    return ("metatrader" in t or "mt5" in t) and ("connect" in t or "connect kaise" in t or "connect kaise karun" in t or "se connect" in t or "connect kar" in t or "login" in t)


def mt5_canned_reply_hinglish() -> str:
    return (
        "Yeh steps follow karo to connect MetaTrader 5 account:\n"
        "1) MetaTrader 5 app open karo.\n"
        "2) 'Login to an existing account' ya 'Login' option dhoondo.\n"
        "3) Broker ne jo login/account number diya hai, woh enter karo, uska password aur correct server select karo.\n"
        "4) Server details aapke broker se milti hain — broker hi correct server batayega. Agar aapko server ka naam nahi pata, toh pehle batao kaunsa broker use kar rahe ho.\n"
        "5) Kabhi apna password, OTP ya sensitive details yahan mat bhejo. Agar login fail ho, broker support se server or account details confirm karo."
    )


# ---------------------------------------------------------------- AI call and model capability
class AIError(Exception):
    pass


def model_supports_vision_heuristic(model_name: str) -> bool:
    """Best-effort heuristic to determine whether the configured model supports
    image/vision inputs.
    """
    n = model_name.lower()
    keys = ["vl", "vision", "gemini", "flash", "multimodal", "image", "img"]
    return any(k in n for k in keys)


async def get_model_capability(client: httpx.AsyncClient, model_name: str, timeout: float = 6.0) -> Tuple[Optional[Dict[str, Any]], Optional[bool]]:
    """Query the OpenRouter models endpoint and try to find the named model.
    Returns (model_info, supports_vision) where model_info is the raw model object
    or None if not found / on error, and supports_vision is True/False if
    determinable, otherwise None.
    """
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

    # Try to match model by id or name
    target = model_name.lower()
    for m in data:
        mid = (m.get("id") or "").lower()
        name = (m.get("name") or "").lower()
        desc = (m.get("description") or "").lower()
        if target == mid or target in mid or target in name or target in desc:
            # Determine vision support from fields if possible
            modalities = m.get("modalities") or m.get("input") or m.get("capabilities")
            if isinstance(modalities, list):
                has_vision = any("image" in str(x).lower() or "vision" in str(x).lower() for x in modalities)
                return m, bool(has_vision)
            if any(k in name or k in desc for k in ("vision", "image", "multimodal", "flash", "gemini")):
                return m, True
            return m, False

    return None, None


async def ask_ai(client: httpx.AsyncClient, messages: list[Any]) -> str:
    """Call OpenRouter with messages. Validate response language (Hinglish) and retry once if necessary."""
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {"model": MODEL, "messages": messages, "max_tokens": 600}

    async def _call():
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

                # Expecting choices -> message -> content
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

                # content may be a string, or a list of content objects for multimodal
                if isinstance(content, str):
                    if content.strip():
                        return content.strip()
                    last_error = "empty response"
                    await asyncio.sleep(backoff)
                    backoff *= 2
                    continue

                if isinstance(content, list):
                    # collect text pieces
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

            # Non-200 handling
            last_error = f"HTTP {status}"
            if status in (401, 402, 403, 404):
                # Permanent authorization / model not found errors: do not retry
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

    # First call
    text = await _call()

    # Validate Hinglish: reject Devanagari and retry once if invalid
    def _is_valid_hinglish(s: str) -> bool:
        # reject Devanagari/Hindi script characters
        if re.search(r"[\u0900-\u097F]", s):
            return False
        # simple heuristic: must contain at least one Hindi-roman token or be short
        tokens = re.findall(r"\w+", s.lower())
        hinglish_tokens = {"kaise", "ka", "kya", "hai", "nahi", "karo", "kar", "login", "server", "broker", "account", "password", "otp"}
        if any(t in hinglish_tokens for t in tokens):
            return True
        # allow very short replies
        if len(tokens) <= 6:
            return True
        return False

    if not _is_valid_hinglish(text):
        log.info("AI reply failed Hinglish validation; retrying once with stronger hint")
        # inject a stronger instruction at start of system prompt
        messages_copy = []
        for m in messages:
            if m.get("role") == "system":
                messages_copy.append({"role": "system", "content": SYSTEM_PROMPT + "\nALWAYS REPLY IN HINGLISH IN ROMAN SCRIPT."})
            else:
                messages_copy.append(m)
        try:
            text2 = await _call()
            if _is_valid_hinglish(text2):
                return text2
        except AIError:
            pass
        # If still invalid, return the original but mark it cleaned
    return text


# --------------------------------------------------------------- commands
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await safe_reply(
        update,
        f"Welcome to {BOT_NAME}! 🛟\n\n"
        "Main support-only bot hoon — batao kya issue hai, ya screenshot bhejo caption ke saath.\n"
        "Safety: kabhi apna password, OTP, PIN yahan mat bhejo.\n\n"
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
        "Support karo: apna issue short mein batao ya screenshot bhejo.\n"
        "Conversation memory 5 minutes tak hoti hai.\n"
        "Main aapka account access nahi kar sakta — account-specific help ke liye broker/official support se contact karo.",
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
    """Return a message content structure compatible with OpenRouter when
    including optional image data. If only text is provided, returns the text
    as a simple string to preserve backwards compatibility.
    """
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

    # handle rate limiting early
    if rate_limited(user_id):
        await safe_reply(update, "Aap jaldi messages bhej rahe ho. Thoda ruk ke phir try karo.")
        return

    # extract text and possible caption
    text = (update.message.text or update.message.caption or "").strip()
    photo = None
    doc = None
    image_data_url = None

    if getattr(update.message, "photo", None):
        photo = update.message.photo[-1]
    elif getattr(update.message, "document", None) and getattr(update.message.document, "mime_type", "").startswith("image"):
        doc = update.message.document

    # secrets check
    if contains_secret(text):
        await safe_reply(
            update,
            "⚠️ Yeh message kisi secret jaisa lagta hai (password/OTP). Maine isse process nahi kiya. Sensitive info yahan mat bhejo.",
        )
        return

    # Catch MetaTrader 5 question with a canned, safe Hinglish reply
    if is_mt5_question(text):
        await safe_reply(update, mt5_canned_reply_hinglish())
        return

    # Casual chat redirection
    if is_casual_message(text):
        await safe_reply(update, "Hi! Main support bot hoon — batao kya support issue hai jispar main help kar sakun?")
        return

    if not text and not photo and not doc:
        await safe_reply(update, "Please apna support issue short mein batao ya screenshot bhejo caption ke saath.")
        return

    if len(text) > MAX_INPUT_CHARS:
        await safe_reply(update, f"Message bahut lamba hai. Kripya {MAX_INPUT_CHARS} characters ke andar rakho.")
        return

    session = get_session(user_id)
    if session.lock.locked():
        await safe_reply(update, "Pehle wale message pe kaam chal raha hai. Thoda intezaar karo.")
        return

    # Image handling only when an image is actually present
    if photo or doc:
        # Check model capability: prefer authoritative info if available
        ms = context.application.bot_data.get("model_supports_vision")
        if ms is None:
            try:
                info, cap = await get_model_capability(context.application.bot_data["http"], MODEL)
                context.application.bot_data["model_info"] = info
                context.application.bot_data["model_supports_vision"] = cap if cap is not None else model_supports_vision_heuristic(MODEL)
                ms = context.application.bot_data["model_supports_vision"]
            except Exception:
                ms = model_supports_vision_heuristic(MODEL)
        if not ms:
            await safe_reply(update, "Yeh model image analyze nahi kar sakta. Agar aap image bhejte ho toh model ko vision-capable set karo ya text mein problem batao.")
            return

        # download image
        file = await (photo.get_file() if photo else doc.get_file())
        max_bytes = 5 * 1024 * 1024  # 5 MB
        if getattr(file, "file_size", 0) and file.file_size > max_bytes:
            await safe_reply(update, "Image size zyada hai. Kripya chhota screenshot bhejo (5 MB se kam).")
            return
        try:
            bio = io.BytesIO()
            await file.download(out=bio)
            bio.seek(0)
            data = bio.read()
            if len(data) > max_bytes:
                await safe_reply(update, "Image size zyada hai baad mein. Chhota image bhejo.")
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

    # Build conversation and call AI
    async with session.lock:
        placeholder = text or "[screenshot]"
        session.history.append({"role": "user", "content": placeholder})
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
                await safe_reply(update, "Configured model nahi mila. Admin se OPENROUTER_MODEL check karne kehdo.")
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
        expired = [
            uid
            for uid, s in sessions.items()
            if now - s.last_active > IDLE_TIMEOUT and not s.lock.locked()
        ]
        for uid in expired:
            sessions.pop(uid, None)
        for uid in [u for u, q in rate_log.items() if not q or now - q[-1] > RATE_LIMIT_WINDOW]:
            rate_log.pop(uid, None)
        if expired:
            log.info("Expired %d idle conversation(s)", len(expired))


async def post_init(app: Application):
    app.bot_data["http"] = httpx.AsyncClient()
    try:
        info, cap = await get_model_capability(app.bot_data["http"], MODEL)
        app.bot_data["model_info"] = info
        app.bot_data["model_supports_vision"] = cap if cap is not None else model_supports_vision_heuristic(MODEL)
        log.info("Model capability loaded: vision=%s", app.bot_data.get("model_supports_vision"))
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

    private = filters.ChatType.PRIVATE  # private chats only
    app.add_handler(CommandHandler("start", cmd_start, filters=private))
    app.add_handler(CommandHandler("help", cmd_help, filters=private))
    app.add_handler(CommandHandler("clear", cmd_clear, filters=private))
    app.add_handler(CommandHandler("status", cmd_status, filters=private))
    # Accept text, captions, photos and image documents as support input
    app.add_handler(MessageHandler(private & filters.TEXT & ~filters.COMMAND, on_message))
    app.add_handler(MessageHandler(private & filters.PHOTO, on_message))
    app.add_handler(MessageHandler(private & filters.Document.IMAGE, on_message))
    app.add_handler(MessageHandler(private & filters.COMMAND, on_unknown_command))
    # Any other non-text updates get a helpful note
    app.add_handler(
        MessageHandler(
            private & ~filters.TEXT & ~filters.COMMAND & ~filters.StatusUpdate.ALL,
            on_non_text,
        )
    )
    app.add_error_handler(on_error)

    # Only call polling once per process. If you deploy multiple Railway replicas,
    # each will start its own polling instance; set replicas to 1 in Railway to
    # avoid Telegram 'Conflict' errors.
    app.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)


if __name__ == "__main__":
    main()
