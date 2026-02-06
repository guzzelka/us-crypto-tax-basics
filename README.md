# 🤖 US Crypto Tax Basics Bot v2.0

A lightweight Telegram bot that explains US cryptocurrency tax rules in plain English.

## What's New in v2.0

**Lightweight version** - Uses OpenAI Embeddings API instead of local sentence-transformers model:
- ✅ Works on 512MB RAM (Render Starter plan)
- ✅ No HuggingFace token needed
- ✅ Faster startup time
- ✅ Only 3 Python dependencies

## Features

- 📚 **Knowledge Base**: 50+ topics from official IRS guidance
- 🔍 **RAG Search**: Semantic search using OpenAI embeddings
- 🤖 **AI Responses**: GPT-4o-mini for natural language explanations
- 📊 **Analytics**: SQLite logging for queries and feedback
- 👍👎 **Feedback**: User rating buttons

## Quick Start (Render)

### 1. Create GitHub Repository

Upload these files to a new GitHub repo:
- `crypto_tax_bot_v2.py`
- `crypto_tax_rag_chunks_v1.1.json`
- `requirements.txt`
- `Dockerfile`

### 2. Deploy on Render

1. Go to https://render.com
2. New → Background Worker
3. Connect your GitHub repo
4. Add Environment Variables:
   - `TELEGRAM_BOT_TOKEN` = your token from @BotFather
   - `OPENAI_API_KEY` = your OpenAI API key
5. Deploy!

## Cost

- **Render Starter**: $7/month
- **OpenAI API**: ~$0.01-0.05 per question (very cheap with text-embedding-3-small + gpt-4o-mini)

## Files

| File | Description |
|------|-------------|
| `crypto_tax_bot_v2.py` | Main bot code |
| `crypto_tax_rag_chunks_v1.1.json` | Knowledge base (50+ IRS topics) |
| `requirements.txt` | Python dependencies (only 3!) |
| `Dockerfile` | Docker configuration |

## Bot Commands

- `/start` - Welcome message
- `/help` - Show help
- `/topics` - List available topics
- `/disclaimer` - Legal disclaimer

## Technical Details

### Why v2.0?

The original v1.1 used `sentence-transformers` library which requires 500MB+ RAM just for the ML model. This made it impossible to run on Render's Starter plan (512MB limit).

v2.0 replaces local embeddings with OpenAI's `text-embedding-3-small` API:
- Embeddings computed via API call (no local model)
- Minimal RAM usage (~100MB)
- Same quality semantic search
- Slightly higher cost per query (~$0.0001)

### Architecture

```
User Query → OpenAI Embedding → Cosine Similarity Search → Top 3 Chunks → GPT-4o-mini → Response
```

## License

MIT
