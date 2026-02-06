"""
Crypto Tax Bot v2.0 - Lightweight Version
Telegram bot with RAG using OpenAI Embeddings (no local ML models)

This version uses OpenAI's text-embedding-3-small instead of sentence-transformers,
making it lightweight enough to run on 512MB RAM (Render Starter plan).

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
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = """You are a specialized crypto tax education assistant for US taxpayers. 
You have access to a curated knowledge base built from official IRS guidance.

IDENTITY & LIMITATIONS:
- You are NOT a CPA, tax attorney, or financial advisor
- Everything you provide is EDUCATIONAL and INFORMATIONAL only
- You do NOT make tax projections or tell users what they "should" do
- You only cover US federal tax rules

CRITICAL RULES:
1. ONLY answer based on the provided context from the knowledge base
2. If information is NOT in the context, say: "I don't have verified information about that topic. Please consult a CPA or check irs.gov."
3. NEVER invent tax rules, rates, or deadlines
4. Always cite the source at the end of your response

RESPONSE FORMAT:
- Lead with the direct answer
- Explain briefly (2-3 paragraphs max)
- Include an example with numbers if available
- End with: "📎 Source: [source name]"
- End with: "⚠️ Educational info only, not tax advice. Consult a CPA for your situation."

RISK LEVEL HANDLING:
- If context shows risk_level="high", warn: "⚠️ This is a gray area with limited IRS guidance."
- If context shows risk_level="medium", mention: "Note: IRS guidance on this is limited."
"""


# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


# =============================================================================
# KNOWLEDGE BASE WITH OPENAI EMBEDDINGS
# =============================================================================

class KnowledgeBase:
    """Knowledge base with OpenAI embeddings for semantic search."""
    
    def __init__(self, json_path: str, openai_client: OpenAI):
        self.chunks: List[Dict] = []
        self.embeddings: List[List[float]] = []
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
            # Combine searchable text
            triggers = ' '.join(chunk.get('question_triggers', []))
            text = f"{chunk['title']} {triggers} {chunk.get('simple_answer', '')}"
            texts.append(text)
        
        # Batch embed all texts (OpenAI allows up to 2048 inputs)
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
        # Get query embedding
        try:
            response = self.openai.embeddings.create(
                model="text-embedding-3-small",
                input=[query]
            )
            query_embedding = response.data[0].embedding
        except Exception as e:
            logger.error(f"Error getting query embedding: {e}")
            return []
        
        # Calculate similarities
        similarities = []
        for i, chunk_embedding in enumerate(self.embeddings):
            sim = self._cosine_similarity(query_embedding, chunk_embedding)
            similarities.append((sim, i))
        
        # Sort by similarity and get top results
        similarities.sort(reverse=True)
        top_indices = [idx for _, idx in similarities[:n_results]]
        
        return [self.chunks[i] for i in top_indices]
    
    def format_context(self, chunks: List[Dict]) -> str:
        """Format chunks into context string for LLM."""
        if not chunks:
            return "No relevant information found in the knowledge base."
        
        context_parts = []
        for chunk in chunks:
            part = f"""
---
TOPIC: {chunk['title']}
SOURCE: {chunk.get('source', 'Knowledge Base')}
RISK_LEVEL: {chunk.get('risk_level', 'low')}

{chunk['content']}

SIMPLE_ANSWER: {chunk.get('simple_answer', '')}
"""
            if chunk.get('example'):
                part += f"\nEXAMPLE: {chunk['example']}"
            context_parts.append(part)
        
        return "\n".join(context_parts)


# =============================================================================
# LLM HANDLER
# =============================================================================

class LLMHandler:
    """Handles LLM interactions."""
    
    def __init__(self, openai_client: OpenAI):
        self.client = openai_client
    
    async def generate_response(self, user_query: str, context: str) -> Tuple[str, int]:
        """Generate a response using the LLM with RAG context."""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"""
Based on the following knowledge base context, answer the user's question.
If the context doesn't contain relevant information, say so clearly.

CONTEXT:
{context}

USER QUESTION: {user_query}
"""}
        ]
        
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    max_tokens=1000,
                    temperature=0.3
                )
            )
            
            tokens_used = response.usage.total_tokens if response.usage else 0
            return response.choices[0].message.content, tokens_used
            
        except Exception as e:
            logger.error(f"LLM error: {e}")
            return "Sorry, I'm having trouble processing your request. Please try again.", 0


# =============================================================================
# ANALYTICS DATABASE
# =============================================================================

class AnalyticsDB:
    """SQLite database for logging queries and feedback."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        
    async def initialize(self):
        """Create tables if they don't exist."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute('''
                CREATE TABLE IF NOT EXISTS queries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    user_id INTEGER NOT NULL,
                    username TEXT,
                    query TEXT NOT NULL,
                    response TEXT,
                    tokens_used INTEGER,
                    response_time_ms INTEGER
                )
            ''')
            await db.execute('''
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    query_id INTEGER,
                    user_id INTEGER NOT NULL,
                    feedback TEXT NOT NULL
                )
            ''')
            await db.commit()
    
    async def log_query(self, user_id: int, username: str, query: str, 
                       response: str, tokens_used: int, response_time_ms: int) -> int:
        """Log a user query and return the query ID."""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute('''
                INSERT INTO queries (timestamp, user_id, username, query, response, tokens_used, response_time_ms)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (datetime.utcnow().isoformat(), user_id, username, query, response, tokens_used, response_time_ms))
            await db.commit()
            return cursor.lastrowid
    
    async def log_feedback(self, query_id: int, user_id: int, feedback: str):
        """Log user feedback for a response."""
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

I help you understand US crypto tax rules in plain English.

**What I can explain:**
• Taxable vs non-taxable events
• Capital gains (short-term vs long-term)
• Staking, mining, airdrops
• DeFi taxes (swaps, LPs, lending)
• NFT taxation
• Tax-saving strategies
• Reporting requirements

**Just ask me a question!** For example:
› _"Is swapping ETH for USDC taxable?"_
› _"How are staking rewards taxed?"_
› _"What is tax-loss harvesting?"_

⚠️ I provide educational info only, not tax advice.

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
I'll tell you when to consult a CPA."""
    
    await update.message.reply_text(help_text, parse_mode='Markdown')


async def cmd_topics(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /topics command."""
    topics = """**Topics I can explain:**

📌 **Basics** - IRS classification, taxable events
💰 **Capital Gains** - Short/long-term rates, cost basis
⛏️ **Earning Crypto** - Mining, staking, airdrops, forks
🔄 **DeFi** - Swaps, LPs, lending, borrowing, wrapping
🎨 **NFTs** - Buying, selling, creating
📋 **Reporting** - Forms 8949, Schedule D, 1099-DA
💡 **Strategies** - Tax-loss harvesting, donations

Ask me anything about these!"""
    
    await update.message.reply_text(topics, parse_mode='Markdown')


async def cmd_disclaimer(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /disclaimer command."""
    disclaimer = """⚠️ **IMPORTANT DISCLAIMER**

This bot provides **educational information only**.

**This is NOT:**
• Tax advice
• Legal advice  
• Financial advice
• A substitute for a CPA

**Information is based on:**
• IRS Notice 2014-21
• IRS Revenue Rulings 2019-24, 2023-14
• IRS FAQ on Virtual Currency
• IRS Final Regulations 2024

**Always:**
• Consult a qualified tax professional
• Verify with official IRS sources
• Consider your specific situation

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
    user_message = update.message.text
    user_id = update.effective_user.id
    username = update.effective_user.username
    
    logger.info(f"Query from {user_id} (@{username}): {user_message[:100]}...")
    
    # Show typing indicator
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    
    start_time = datetime.now()
    
    try:
        # Search knowledge base
        relevant_chunks = knowledge_base.search(user_message, n_results=3)
        context_str = knowledge_base.format_context(relevant_chunks)
        
        # Generate response
        response_text, tokens_used = await llm_handler.generate_response(user_message, context_str)
        
        # Calculate response time
        response_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)
        
        # Log to database
        query_id = await analytics_db.log_query(
            user_id=user_id,
            username=username or "",
            query=user_message,
            response=response_text,
            tokens_used=tokens_used,
            response_time_ms=response_time_ms
        )
        
        # Send response with feedback buttons
        keyboard = create_feedback_keyboard(query_id)
        
        # Split long messages if needed
        if len(response_text) <= 4096:
            await update.message.reply_text(response_text, reply_markup=keyboard, parse_mode='Markdown')
        else:
            chunks = [response_text[i:i+4000] for i in range(0, len(response_text), 4000)]
            for i, chunk in enumerate(chunks):
                if i == len(chunks) - 1:
                    await update.message.reply_text(chunk, reply_markup=keyboard)
                else:
                    await update.message.reply_text(chunk)
        
        logger.info(f"Response sent. Tokens: {tokens_used}, Time: {response_time_ms}ms")
        
    except Exception as e:
        logger.error(f"Error handling message: {e}")
        await update.message.reply_text(
            "❌ Sorry, something went wrong. Please try again later."
        )


async def handle_feedback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle feedback button clicks."""
    query = update.callback_query
    await query.answer()
    
    data = query.data
    user_id = update.effective_user.id
    
    if data.startswith("feedback_"):
        parts = data.split("_")
        if len(parts) >= 3:
            feedback_type = parts[1]
            query_id = int(parts[2])
            
            await analytics_db.log_feedback(query_id, user_id, feedback_type)
            
            await query.edit_message_reply_markup(reply_markup=None)
            
            if feedback_type == "bad":
                await query.message.reply_text(
                    "Thanks for the feedback. Try rephrasing your question for a better answer!"
                )
            
            logger.info(f"Feedback: {feedback_type} for query {query_id}")


async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle errors."""
    logger.error(f"Update {update} caused error {context.error}")


# =============================================================================
# MAIN
# =============================================================================

async def post_init(application: Application):
    """Initialize async components."""
    global analytics_db
    await analytics_db.initialize()
    logger.info("Database initialized")


def main():
    """Start the bot."""
    global config, knowledge_base, llm_handler, analytics_db
    
    logger.info("Starting US Crypto Tax Basics Bot v2.0...")
    
    # Load configuration
    config = Config.from_env()
    
    # Initialize OpenAI client
    openai_client = OpenAI(api_key=config.openai_api_key)
    
    # Initialize components
    logger.info(f"Loading knowledge base from {config.knowledge_base_path}...")
    knowledge_base = KnowledgeBase(config.knowledge_base_path, openai_client)
    
    logger.info("Initializing LLM handler...")
    llm_handler = LLMHandler(openai_client)
    
    logger.info("Initializing analytics database...")
    analytics_db = AnalyticsDB(config.db_path)
    
    # Create application
    application = Application.builder().token(config.telegram_token).build()
    application.post_init = post_init
    
    # Add handlers
    application.add_handler(CommandHandler("start", cmd_start))
    application.add_handler(CommandHandler("help", cmd_help))
    application.add_handler(CommandHandler("topics", cmd_topics))
    application.add_handler(CommandHandler("disclaimer", cmd_disclaimer))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_handler(CallbackQueryHandler(handle_feedback, pattern="^feedback_"))
    application.add_error_handler(error_handler)
    
    # Start bot
    logger.info("Bot is starting...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
