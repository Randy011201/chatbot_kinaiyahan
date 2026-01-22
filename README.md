# Kinaiyahan Chatbot (Python/FastAPI + Google Gemini)

A conversational chatbot for Kinaiyahan Forest Park powered by Google's Gemini API. The chatbot provides natural, human-like responses about room bookings, dining, events, and general inquiries.

## Features

✨ **AI-Powered Responses**: Uses Google Gemini API for natural, conversational answers
🏨 **Resort Knowledge**: Equipped with Kinaiyahan Forest Park information
💬 **Fallback Support**: Works with or without API key (graceful fallback to template responses)
📱 **Image Support**: Ready for image uploads and analysis
⚡ **Fast & Lightweight**: Built with FastAPI for high performance

## Setup

### 1. Get Gemini API Key

1. Visit [Google AI Studio](https://ai.google.dev/)
2. Click "Get API Key"
3. Create a new API key for use in the project

### 2. Install Dependencies

```bash
cd chatbot_kinaiyahan
python -m venv .venv

# On Windows:
.venv\Scripts\activate

# On macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Configure Environment

```bash
cp .env.example .env
```

Then edit `.env` and add your Gemini API key:

```env
GEMINI_API_KEY=your-actual-api-key-here
HOST=0.0.0.0
PORT=8000
```

## Run the Chatbot

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### API Endpoints

- **Health Check**: `GET http://localhost:8000/health`
  - Response: `{"status": "ok"}`

- **Chat Endpoint**: `POST http://localhost:8000/chat`
  - Body:
    ```json
    {
      "message": "What are your room rates?",
      "history": [],
      "imageBase64": null
    }
    ```
  - Response:
    ```json
    {
      "answer": "Our rooms range from ₱1,500 to ₱5,000 per night depending on room type and date...",
      "model": "kinaiyahan-gemini"
    }
    ```

## Integration with Next.js Frontend

Set the chatbot proxy URL in your frontend environment:

```bash
CHATBOT_PROXY_URL=http://localhost:8000/chat
```

Then the frontend's `/api/ai-chat` endpoint will proxy requests to this service.

## How It Works

1. **User Message**: User sends a message through the chat interface
2. **Knowledge Search**: System searches local knowledge base for relevant context
3. **Gemini API Call**: Message + context is sent to Google Gemini API
4. **Response Generation**: Gemini generates a natural, human-like response
5. **Response Return**: Answer is returned to the user

### System Prompt

The chatbot is configured with a detailed system prompt that ensures:
- Warm, conversational tone (like a real receptionist)
- Focus on Kinaiyahan Forest Park services
- Relevant context from the knowledge base
- Requests for necessary booking details (dates, guests, etc.)
- Offers to escalate to direct contact when needed

## Fallback Mode

If the Gemini API key is not configured or unavailable:
- The chatbot gracefully falls back to template-based responses
- Uses the tool-based approach with knowledge snippets
- No loss of core functionality, just less natural responses

## Knowledge Base

The chatbot has built-in knowledge about:
- 🏠 **Rooms**: Pricing, check-in/out times
- 🍽️ **Dining**: Restaurant hours, popular dishes
- 🎉 **Events**: Wedding, birthday, and corporate event hosting
- 💳 **Payments**: Accepted payment methods
- 📧 **Contact**: Email and location information
- ❌ **Cancellations**: Cancellation policies

## Extending the Chatbot

### Add More Knowledge

Edit the `KNOWLEDGE` list in `main.py`:

```python
KNOWLEDGE = [
    ("topic", "Information about the topic"),
    # Add more entries...
]
```

### Add Tool Functions

Create new tool functions like:

```python
def tool_spa_services(message: str) -> Optional[str]:
    if "spa" in message or "massage" in message:
        return "Our spa offers..."
    return None
```

Then add it to the `build_answer_fallback()` function's tool loop.

## Requirements

- Python 3.9+
- FastAPI
- Uvicorn
- Google Generative AI SDK
- Python Dotenv

## Troubleshooting

### "GEMINI_API_KEY not found"

Make sure you have:
1. Created a `.env` file from `.env.example`
2. Added your actual API key to the `.env` file
3. The `.env` file is in the `chatbot_kinaiyahan` directory

### API Key Quota Exceeded

- Check your [Google AI Studio](https://ai.google.dev/) usage
- Gemini API has generous free tier limits
- Consider rate limiting for production use

### Slow Responses

- First API call may be slightly slower (initialization)
- Subsequent calls should be fast
- Consider implementing caching for common queries

## Production Deployment

For production:

1. Remove `--reload` flag
2. Use a production ASGI server like `gunicorn`:
   ```bash
   pip install gunicorn
   gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app
   ```
3. Set up proper environment variables and secrets management
4. Consider rate limiting and authentication
5. Add logging and monitoring

## License

Part of Kinaiyahan Forest Park project
