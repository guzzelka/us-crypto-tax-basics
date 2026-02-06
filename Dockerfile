FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY crypto_tax_bot_v2.py bot.py
COPY crypto_tax_rag_chunks_v1.1.json knowledge_base.json

# Set environment variables
ENV KNOWLEDGE_BASE_PATH=knowledge_base.json
ENV DB_PATH=/data/bot_analytics.db

# Create data directory
RUN mkdir -p /data

# Run the bot
CMD ["python", "bot.py"]
