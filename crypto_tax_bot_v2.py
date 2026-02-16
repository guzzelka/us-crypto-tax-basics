"""
Crypto Tax Bot v2.1 - Lightweight Version with Professional Tone
Telegram bot with RAG using OpenAI Embeddings (no local ML models)

This version uses OpenAI's text-embedding-3-small instead of sentence-transformers,
making it lightweight enough to run on 512MB RAM (Render Starter plan).

v2.1 changes: Updated system prompt to professional/cautious tone (v1.2)

Requirements:
pip install python-telegram-bot openai aiosqlite

Environment variables:
- TELEGRAM_BOT_TOKEN (required)
- OPENAI_API_KEY (required)
"""

import os
import json
import logging
import asyncio
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

# Telegram
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    filters,
    ContextTypes
)
from telegram.error import TelegramError

# OpenAI
from openai import OpenAI

# Database
import aiosqlite


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class Config:
    """Bot configuration from environment variables."""
    telegram_token: str
    openai_api_key: str
    knowledge_base_path: str
    db_path: str

    @classmethod
    def from_env(cls) -> 'Config':
        telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
        openai_key = os.getenv("OPENAI_API_KEY")

        if not telegram_token:
            raise ValueError("TELEGRAM_BOT_TOKEN environment variable not set")
        if not openai_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")

        return cls(
            telegram_token=telegram_token,
            openai_api_key=openai_key,
            knowledge_base_path=os.getenv("KNOWLEDGE_BASE_PATH", "crypto_tax_rag_chunks_v1.1.json"),
            db_path=os.getenv("DB_PATH", "bot_analytics.db"),
        )


# =============================================================================
# SYSTEM PROMPT (v1.2 — Professional Tone)
# =============================================================================

SYSTEM_PROMPT = """You are a specialized crypto tax education assistant for US taxpayers.
You have access to a curated knowledge base built from official IRS guidance.
Your job is to explain how tax rules work — not to suggest strategies, optimize taxes, or give personalized advice.

IDENTITY & LIMITATIONS:
- You are NOT a CPA, tax attorney, financial advisor, or tax strategist
- Everything you provide is EDUCATIONAL and INFORMATIONAL only
- You do NOT make tax projections or tell users what they "should" do
- You do NOT suggest strategies to minimize, reduce, or avoid taxes
- You only cover U.S. federal income tax rules. State tax treatment may differ.

CRITICAL RULES:
1. ONLY answer based on the provided context from the knowledge base
2. If information is NOT in the context, say: "I don't have verified information about that topic. Please consult a CPA or check irs.gov/digital-assets."
3. NEVER invent tax rules, rates, or deadlines
4. NEVER state proposed or pending rules as settled law
5. Always cite the source at the end of your response

LANGUAGE RULES (CRITICAL):

NEVER use these phrases:
- "You should..." / "You can safely..."
- "You must..." / "You are required to..." / "You are obligated to..."
- "You can reduce your taxes by..." / "This will lower your liability..."
- "This allows you to avoid tax..." / "This is a strategy for..."
- "You could potentially..." / "Consider doing..."
- "One approach is to..." / "To mitigate this risk..."
- "You do NOT owe any tax." / "There is no taxable event." (too categorical)
- "This provides a double benefit..."
- "It is currently legal to..." (sounds like endorsement — use "current law does not apply X to Y")
- "There is no provision that allows..." / "cannot" / "never" / "always" / "guaranteed"
- "Some advisors recommend..." / "allowing a short delay..." (this is operational guidance)
- "You will be fine" / "You can safely"

NEVER provide transaction structuring advice:
- NEVER suggest timing of purchases or sales
- NEVER suggest swapping into similar/different assets
- NEVER suggest delays between transactions
- NEVER suggest approaches "to reduce audit risk"
- NEVER describe how to structure transactions for tax benefit
- If the knowledge base context contains such advice, DO NOT repeat it — rephrase as a neutral explanation of the legal framework only

NEVER make absolute statements about law:
- NEVER say "there is no provision" — instead say "historically... however Congress has broad authority"
- NEVER say "cannot" about legislation — instead say "under current law"
- NEVER say "always" or "never" about tax treatment — instead say "generally" or "typically"

ALWAYS use hedging language:
- "May" instead of "will" or "can"
- "Generally" / "typically" / "historically" instead of absolute statements
- "Under current federal tax law..."
- "In most cases..."
- "Depends on individual circumstances"
- "Based on available guidance"
- "This reflects current law as of 2025"
- Instead of "you must report" say "taxpayers are generally expected to report"
- Instead of "you are required" say "reporting may be required under"
- Instead of "you need to" say "it may be necessary to"

CORE PRINCIPLE — THIS IS A BASIC EDUCATIONAL BOT:
- You explain IRS rules and definitions — THAT'S IT
- You are NOT a tax expert simulator
- You do NOT model user scenarios or walk through action steps
- You do NOT answer "could I do X and then Y" type questions with step-by-step analysis
- When users ask practical/situational questions, explain the RELEVANT RULE from IRS, then direct them to a CPA for their specific situation
- Keep answers SHORT and focused on the rule itself
- You must describe legal frameworks, not simulate user action steps

Example of WRONG approach:
User: "Could I sell in December and buy back in January to double the loss?"
Bot: "Since the wash sale rule does not apply, a taxpayer could sell their crypto at a loss and then buy it back right away. This allows the taxpayer to claim the loss on their taxes. For example, if a taxpayer sells Bitcoin for a $30,000 loss and then immediately buys it back..."
(This is too detailed, models the scenario, sounds like guidance)

Example of RIGHT approach:
User: "Could I sell in December and buy back in January to double the loss?"
Bot: "**Rule**: Under current federal tax law, the wash sale rule (IRC §1091) applies to securities but not to cryptocurrency, which is classified as property per IRS Notice 2014-21. Capital losses are generally recognized in the tax year the sale occurs (IRC §1211, §1212). **Limitations**: This is a complex area where multiple rules may interact. The specific tax treatment of any transaction depends on individual circumstances. Consult a CPA for guidance on your situation."
(Short, rule-focused, no scenario modeling, directs to CPA)

Instead of strategic language, explain the MECHANISM:
- Bad: "Selling before Dec 31 can be a strategy to harvest losses"
- Good: "Some taxpayers realize losses before year-end. Realized losses may offset capital gains under current rules."
- Bad: "You can sell and immediately buy back crypto to harvest losses"
- Good: "Under current federal tax law, the wash sale rule applies to securities but not to property. Crypto is classified as property."
- Bad: "This provides a double benefit: a charitable deduction and avoidance of capital gains tax"
- Good: "Donating appreciated property held over one year may allow a deduction at fair market value. The specific tax treatment depends on multiple factors including holding period, charity type, and AGI limitations."
- Bad: "To mitigate this risk, some advisors recommend allowing a short delay before repurchasing"
- Good: "The IRS may consider whether a transaction has both a meaningful economic effect and a substantial non-tax purpose, as outlined in IRC Section 7701(o)."
- Bad: "There is no provision that allows Congress to retroactively deny a capital loss"
- Good: "Historically, changes to capital loss rules have generally applied prospectively. However, Congress has broad authority over tax legislation, and effective dates can vary."
- Bad: "While it is currently legal to sell crypto at a loss and immediately buy it back"
- Good: "Under current federal tax law, wash sale rules do not apply to cryptocurrency."

IMPORTANT — when the knowledge base context contains strategic or operational advice, you must REWRITE it as a neutral legal explanation. Do not parrot the context. Apply these language rules even if the context uses different wording.

RESPONSE STRUCTURE (every substantive answer):

**Rule**
State the relevant tax rule in 1-3 sentences. Cite the IRS source. Do NOT model the user's scenario.

**Limitations**
- "This reflects current federal law as of 2025."
- "State tax treatment may differ."
- "Individual circumstances vary — consult a CPA for your specific situation."
- Any relevant gray areas or pending changes.

Keep answers SHORT — 2 paragraphs max for simple questions. Do NOT write long explanations or walk through hypothetical scenarios. If the user's question requires detailed analysis of their specific situation, say so and refer them to a CPA.

Numerical examples: ONLY use brief, generic examples to illustrate a rule's mechanics. Never model the user's specific numbers or scenario.

End every response with:
📎 Source: [IRS source name]
⚠️ This is general educational information about U.S. federal income tax rules. It does not consider your full tax profile, filing status, state taxation, or other income factors. Consult a qualified tax professional for your specific situation.

HANDLING EVOLVING LAW:
- Never state proposed rules as settled law
- Use: "Recent IRS guidance indicates..." or "As of 2025, the IRS has stated..."
- Never give specific effective dates for rules that are not yet final
- Bad: "IRS will default to FIFO starting in 2026"
- Good: "Recent IRS guidance (Rev. Proc. 2024-28) addresses cost basis accounting methods for digital assets. The specific requirements are complex — consult a CPA for details."

HANDLING GRAY AREAS:
- Clearly state: "IRS has not issued specific guidance on this topic."
- Explain general principles that may apply
- Always recommend consulting a tax professional

RISK LEVEL HANDLING:
- If context shows risk_level="high": "⚠️ This is a gray area with limited IRS guidance. Professional consultation is strongly recommended."
- If context shows risk_level="medium": "Note: IRS guidance on this topic is limited."

HANDLING OFF-TOPIC OR HARMFUL REQUESTS:
If someone asks how to hide income, evade taxes, or avoid reporting:
"I can only provide information about how tax rules work. I cannot assist with tax evasion, which is illegal under federal law. If you have concerns about your tax situation, please consult a licensed tax professional."

TONE: Professional but approachable. Informative, not advisory. Cautious, not bold. Explain the mechanism, never suggest the tactic."""


# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging() -> logging.Logger:
    """Configure logging."""
    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    return logging.getLogger(__name__)

logger = setup_logging()


# =============================================================================
# KNOWLEDGE BASE & RAG (Lightweight - OpenAI Embeddings)
# =============================================================================

class KnowledgeBase:
    """Knowledge base with OpenAI embeddings for semantic search."""

    def __init__(self, json_path: str, openai_client: OpenAI):
        self.chunks = []
        self.embeddings = []
        self.openai = openai_client
        self._load_knowledge_base(json_path)
        self._compute_embeddings()

    def _load_knowledge_base(self, json_path: str):
        """Load chunks from JSON file."""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.chunks = data.get('chunks', [])
            logger.info(f"Loaded {len(self.chunks)} chunks from {json_path}")
        except FileNotFoundError:
            logger.error(f"Knowledge base file not found: {json_path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in knowledge base: {e}")
            raise

    def _compute_embeddings(self):
        """Compute embeddings for all chunks using OpenAI."""
        logger.info("Computing embeddings for knowledge base...")

        texts = []
        for chunk in self.chunks:
            triggers = ' '.join(chunk.get('question_triggers', []))
            text = f"{chunk['title']} {triggers} {chunk.get('simple_answer', '')}"
            texts.append(text)

        try:
            response = self.openai.embeddings.create(
                model="text-embedding-3-small",
                input=texts
            )
            self.embeddings = [item.embedding for item in response.data]
            logger.info(f"Computed {len(self.embeddings)} embeddings")
        except Exception as e:
            logger.error(f"Error computing embeddings: {e}")
            raise

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot_product / (norm_a * norm_b)

    def search(self, query: str, n_results: int = 3) -> List[Dict]:
        """Search for relevant chunks using semantic similarity."""
        try:
            response = self.openai.embeddings.create(
                model="text-embedding-3-small",
                input=[query]
            )
            query_embedding = response.data[0].embedding
        except Exception as e:
            logger.error(f"Error getting query embedding: {e}")
            return []

        similarities = []
        for i, chunk_embedding in enumerate(self.embeddings):
            sim = self._cosine_similarity(query_embedding, chunk_embedding)
            similarities.append((sim, i))

        similarities.sort(reverse=True)
        top_indices = [idx for _, idx in similarities[:n_results]]

        return [self.chunks[i] for i in top_indices]

    def format_context(self, chunks: List[Dict]) -> str:
        """Format chunks into context string for LLM."""
        if not chunks:
            return "No relevant information found in knowledge base."

        context_parts = []
        for chunk in chunks:
            risk = chunk.get('risk_level', 'low')
            part = f"""---
Topic: {chunk.get('topic', '')} / {chunk.get('subtopic', '')}
Title: {chunk.get('title', '')}
Content: {chunk.get('content', '')}
Simple Answer: {chunk.get('simple_answer', '')}
Example: {chunk.get('example', 'N/A')}
Source: {chunk.get('source', 'N/A')}
Risk Level: {risk}
---"""
            context_parts.append(part)

        return '\n'.join(context_parts)


# =============================================================================
# LLM HANDLER
# =============================================================================

class LLMHandler:
    """Handles LLM interactions using OpenAI."""

    def __init__(self, openai_client: OpenAI):
        self.client = openai_client
        self.model = os.getenv("MODEL_NAME", "gpt-4o-mini")

    def generate_response(self, query: str, context: str) -> str:
        """Generate a response using the LLM with RAG context."""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"""Based on the following context from the knowledge base, answer the user's question.

CONTEXT:
{context}

USER QUESTION: {query}

CRITICAL REMINDERS:
- Follow the response structure (Rule/Explanation/Limitations)
- Use hedging language (may, generally, under current law)
- NEVER give strategic or operational advice, even if the context contains it
- NEVER suggest transaction timing, delays, or asset swaps
- NEVER make absolute statements about what Congress can or cannot do
- Explain the legal framework ONLY — not how to use it for tax benefit
- Cite sources and include the disclaimer
- If context contains phrases like "advisors recommend" or "to mitigate risk" — DO NOT repeat them, rephrase as neutral legal explanation"""}
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=int(os.getenv("MAX_TOKENS", "1000")),
                temperature=float(os.getenv("TEMPERATURE", "0.3"))
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM error: {e}")
            return "Sorry, I'm having trouble processing your question right now. Please try again later."


# =============================================================================
# ANALYTICS DATABASE
# =============================================================================

class AnalyticsDB:
    """SQLite database for analytics and feedback."""

    def __init__(self, db_path: str):
        self.db_path = db_path

    async def initialize(self):
        """Create tables if they don't exist."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute('''
                CREATE TABLE IF NOT EXISTS queries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    user_id INTEGER,
                    query TEXT,
                    response TEXT,
                    tokens_used INTEGER DEFAULT 0,
                    error TEXT
                )
            ''')
            await db.execute('''
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    query_id INTEGER,
                    user_id INTEGER,
                    feedback TEXT
                )
            ''')
            await db.commit()

    async def log_query(self, user_id: int, query: str, response: str) -> int:
        """Log a query and return its ID."""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute('''
                INSERT INTO queries (timestamp, user_id, query, response)
                VALUES (?, ?, ?, ?)
            ''', (datetime.utcnow().isoformat(), user_id, query, response))
            await db.commit()
            return cursor.lastrowid

    async def log_feedback(self, query_id: int, user_id: int, feedback: str):
        """Log user feedback."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute('''
                INSERT INTO feedback (timestamp, query_id, user_id, feedback)
                VALUES (?, ?, ?, ?)
            ''', (datetime.utcnow().isoformat(), query_id, user_id, feedback))
            await db.commit()


# =============================================================================
# TELEGRAM BOT HANDLERS
# =============================================================================

# Global instances
config: Optional[Config] = None
knowledge_base: Optional[KnowledgeBase] = None
llm_handler: Optional[LLMHandler] = None
analytics_db: Optional[AnalyticsDB] = None


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start command."""
    welcome_message = """👋 Welcome to **US Crypto Tax Basics**!

I help you understand U.S. federal crypto tax rules in plain English.

**What I can explain:**
• How the IRS classifies crypto
• Taxable vs non-taxable events
• Capital gains (short-term vs long-term)
• Staking, mining, airdrops
• DeFi taxes (swaps, LPs, lending)
• NFT taxation
• Reporting requirements

**Just ask me a question!** For example:
› _"Is swapping ETH for USDC taxable?"_
› _"How are staking rewards taxed?"_
› _"What is the wash sale rule for crypto?"_

⚠️ I provide educational info only, not tax advice. Always consult a CPA for your specific situation.

Type /help for commands."""

    await update.message.reply_text(welcome_message, parse_mode='Markdown')


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /help command."""
    help_text = """**Commands:**
/start - Welcome message
/help - This help menu
/topics - Topics I can explain
/disclaimer - Full legal disclaimer

**Tips:**
• Ask specific questions for better answers
• I'll cite IRS sources when available
• Rate my answers with 👍 or 👎

**Can't find what you need?**
I'll let you know when to consult a CPA."""

    await update.message.reply_text(help_text, parse_mode='Markdown')


async def cmd_topics(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /topics command."""
    topics = """**Topics I can explain:**

📌 **Basics** - IRS classification, taxable events
💰 **Capital Gains** - Short/long-term, cost basis methods
⛏️ **Earning Crypto** - Mining, staking, airdrops, forks
🔄 **DeFi** - Swaps, LPs, lending, borrowing, wrapping
🎨 **NFTs** - Buying, selling, creating
📋 **Reporting** - Forms 8949, Schedule D, 1099-DA
📜 **Rules** - Wash sale, holding periods, donations

Ask me anything about these!"""

    await update.message.reply_text(topics, parse_mode='Markdown')


async def cmd_disclaimer(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /disclaimer command."""
    disclaimer = """⚠️ **IMPORTANT DISCLAIMER**

This bot provides **general educational information only** about U.S. federal cryptocurrency taxation.

**This is NOT:**
• Tax advice
• Legal advice
• Financial advice
• A substitute for a CPA

**This information:**
• Does not consider your full tax profile, filing status, state taxation, or other income factors
• Is based on IRS guidance as of 2025
• May become outdated as laws change

**Sources used:**
• IRS Notice 2014-21
• IRS Revenue Rulings 2019-24, 2023-14
• IRS FAQ on Virtual Currency
• IRS Final Regulations 2024
• Revenue Procedure 2024-28

**Always consult a qualified tax professional (CPA or tax attorney) for guidance specific to your situation.**

Knowledge base last updated: December 2025"""

    await update.message.reply_text(disclaimer, parse_mode='Markdown')


def create_feedback_keyboard(query_id: int) -> InlineKeyboardMarkup:
    """Create inline keyboard for feedback."""
    keyboard = [
        [
            InlineKeyboardButton("👍 Helpful", callback_data=f"feedback_good_{query_id}"),
            InlineKeyboardButton("👎 Not helpful", callback_data=f"feedback_bad_{query_id}")
        ]
    ]
    return InlineKeyboardMarkup(keyboard)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle user text messages."""
    if not update.message or not update.message.text:
        return

    user_message = update.message.text.strip()
    user_id = update.effective_user.id

    if not user_message:
        return

    logger.info(f"User {user_id}: {user_message[:100]}")

    # Show typing indicator
    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id,
        action="typing"
    )

    try:
        # Search knowledge base
        relevant_chunks = knowledge_base.search(user_message, n_results=3)
        context_str = knowledge_base.format_context(relevant_chunks)

        # Generate response
        response = llm_handler.generate_response(user_message, context_str)

        # Log to analytics
        query_id = await analytics_db.log_query(user_id, user_message, response)

        # Create feedback keyboard
        keyboard = create_feedback_keyboard(query_id)

        # Send response (split if too long for Telegram)
        if len(response) > 4096:
            parts = [response[i:i+4096] for i in range(0, len(response), 4096)]
            for i, part in enumerate(parts):
                if i == len(parts) - 1:
                    await update.message.reply_text(part, reply_markup=keyboard)
                else:
                    await update.message.reply_text(part)
        else:
            await update.message.reply_text(response, reply_markup=keyboard)

    except Exception as e:
        logger.error(f"Error handling message: {e}", exc_info=True)
        await update.message.reply_text(
            "Sorry, I encountered an error processing your question. Please try again."
        )


async def handle_feedback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle feedback button presses."""
    query = update.callback_query
    await query.answer()

    data = query.data
    user_id = update.effective_user.id

    if data.startswith("feedback_"):
        parts = data.split("_")
        if len(parts) >= 3:
            feedback_type = parts[1]  # "good" or "bad"
            try:
                query_id = int(parts[2])
            except ValueError:
                return

            await analytics_db.log_feedback(query_id, user_id, feedback_type)

            if feedback_type == "good":
                await query.edit_message_reply_markup(reply_markup=None)
                # Don't send extra message, just remove buttons
            else:
                await query.edit_message_reply_markup(reply_markup=None)
                # Don't send extra message, just remove buttons


async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle errors."""
    logger.error(f"Update {update} caused error {context.error}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Start the bot."""
    global config, knowledge_base, llm_handler, analytics_db

    # Load config
    config = Config.from_env()

    # Initialize OpenAI client
    openai_client = OpenAI(api_key=config.openai_api_key)

    # Initialize knowledge base with embeddings
    logger.info("Loading knowledge base and computing embeddings...")
    knowledge_base = KnowledgeBase(config.knowledge_base_path, openai_client)

    # Initialize LLM handler
    logger.info("Initializing LLM handler...")
    llm_handler = LLMHandler(openai_client)

    # Initialize analytics
    analytics_db = AnalyticsDB(config.db_path)

    # Create application
    logger.info("Starting Telegram bot...")
    application = Application.builder().token(config.telegram_token).build()

    # Initialize database on startup
    async def post_init(app):
        await analytics_db.initialize()

    application.post_init = post_init

    # Add handlers
    application.add_handler(CommandHandler("start", cmd_start))
    application.add_handler(CommandHandler("help", cmd_help))
    application.add_handler(CommandHandler("topics", cmd_topics))
    application.add_handler(CommandHandler("disclaimer", cmd_disclaimer))
    application.add_handler(CallbackQueryHandler(handle_feedback))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Error handler
    application.add_error_handler(error_handler)

    # Start polling
    logger.info("Bot is running!")
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
