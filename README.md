# STF SUPPORT TEAM - Telegram Support Bot

A support-only Telegram bot (not a general chatbot). Built with `python-telegram-bot`, uses OpenRouter for short support replies.

## Architecture
- `main.py` - the whole bot (polling, handlers, OpenRouter call, RAM-only sessions)
- Telegram -> bot (long polling, private chats only) -> OpenRouter (HTTPS) -> reply
- Conversation context is stored in RAM only and erased after 5 minutes of inactivity or on `/clear`. No database, no message logging.

## Commands
`/start`, `/help`, `/clear`, `/status` (admins see extra stats)

## Environment variables
| Name | Required | Notes |
|---|---|---|
| `BOT_TOKEN` | yes | from @BotFather |
| `OPENROUTER_API_KEY` | yes | from openrouter.ai/keys |
| `ADMIN_ID` | recommended | your numeric Telegram ID |
| `OPENROUTER_MODEL` | no | defaults to `nvidia/nemotron-3.5-super-vl:free` |

Never commit real values. Set them in Railway > Variables.

## Deploy on Railway
1. New Project > Deploy from GitHub repo > select this repo.
2. Add the variables above in the Variables tab.
3. The `Procfile` runs `python main.py` as a worker (no web port needed). If Railway doesn't pick it up, set Start Command to `python main.py`.
4. Check Logs for "STF SUPPORT TEAM started".

## Security notes and limitations
- Secrets only via environment variables; logs contain no message text, tokens or keys.
- Per-user rate limit (5 msgs/min), 1000-character input limit, private chats only.
- Messages that look like passwords/OTPs/keys are rejected before reaching the AI (pattern-based, not perfect).
- The AI is instructed to stay in support scope and never invent policies or claim fixes, but no prompt can fully prevent AI mistakes or prompt injection. Treat replies as guidance, not verified facts.
- All data is lost on restart (by design). Railway and OpenRouter process messages under their own policies.
- Free models can be slow, rate-limited, or removed without notice.
