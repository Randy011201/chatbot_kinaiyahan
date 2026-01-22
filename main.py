from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Tuple, Dict, Any
import os
import json
import re
from pathlib import Path
import joblib
import httpx

app = FastAPI(title="Kinaiyahan Chatbot API", version="0.4.0")

# CORS for local frontend (Next.js dev on :3000 or Vercel previews)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://0.0.0.0:3000",
        "*",  # relax for now; tighten in prod
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load dataset
DATASET = None
dataset_path = Path(__file__).parent / "dataset.json"
try:
    with open(dataset_path, 'r') as f:
        DATASET = json.load(f)
    print(f"✓ Dataset loaded successfully")
except FileNotFoundError:
    print(f"WARNING: Dataset not found")
    DATASET = {}
except json.JSONDecodeError as e:
    print(f"WARNING: Could not parse dataset: {e}")
    DATASET = {}

# Load trained ML models
INTENT_CLASSIFIER = None
ANSWER_DATABASE = None
models_dir = Path(__file__).parent / "models"

try:
    classifier_path = models_dir / "intent_classifier.pkl"
    if classifier_path.exists():
        INTENT_CLASSIFIER = joblib.load(classifier_path)
        print(f"✓ Intent classifier loaded")
    else:
        print("WARNING: Intent classifier model not found. Run: python train_chatbot_model.py")
except Exception as e:
    print(f"WARNING: Could not load intent classifier: {e}")

try:
    answer_db_path = models_dir / "answer_database.json"
    if answer_db_path.exists():
        with open(answer_db_path, 'r') as f:
            ANSWER_DATABASE = json.load(f)
        print(f"✓ Answer database loaded")
    else:
        print("WARNING: Answer database not found")
except Exception as e:
    print(f"WARNING: Could not load answer database: {e}")

# Configure Gemini API (google.genai)
GEMINI_API_KEY = None
GEMINI_MODEL_NAME = "gemini-1.5-flash"
GEMINI_CLIENT = None

try:
    from dotenv import load_dotenv
    load_dotenv()

    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL", GEMINI_MODEL_NAME)

    if GEMINI_API_KEY:
        try:
            from google import genai

            # google.genai client (replacement for deprecated google.generativeai)
            GEMINI_CLIENT = genai.Client(api_key=GEMINI_API_KEY)
        except ImportError:
            print("INFO: google.genai not installed. Install with: pip install google-genai")
    else:
        print("INFO: GEMINI_API_KEY not found (optional)")
except ImportError:
    print("INFO: python-dotenv not installed; skipping .env load for Gemini")


class ChatMessage(BaseModel):
    type: str
    text: str
    imageBase64: Optional[str] = None


class ChatRequest(BaseModel):
    message: str
    history: Optional[List[ChatMessage]] = None
    imageBase64: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    model: str = "kinaiyahan-gemini"
    items: Optional[List[Dict[str, Any]]] = None  # For food/drinks items with images


# --- In-memory knowledge snippets (replace with real RAG/vector store) ---
KNOWLEDGE = [
    (
        "rooms",
        "Rooms range ~₱1,500-₱5,000/night; ask for your dates so we can quote exact availability and promos."
    ),
    (
        "checkin",
        "Check-in 2 PM, check-out 12 NN. Late check-in ok; tell us ETA for coordination."
    ),
    (
        "dining",
        "Restaurant 8:00 AM - 10:00 PM; popular picks: adobo sa gata, grilled fish, halo-halo."
    ),
    (
        "events",
        "We host weddings, birthdays, and corporate events; indoor/outdoor venues available."
    ),
    (
        "payments",
        "Payments: Cash, Visa/Mastercard, GCash, Bank Transfer, PayMaya."
    ),
    (
        "contact",
        "Contact: kinaiyahanforestpark@gmail.com; location: Bilar, Bohol 6317."
    ),
    (
        "cancellation",
        "Cancellation policies vary by rate; refundable rates allow free cancel before cutoff. Share your booking code to check." 
    ),
]

# Backend API configuration
BACKEND_API_URL = os.getenv("BACKEND_API_URL", "http://127.0.0.1:8000/api")


async def fetch_from_api(endpoint: str, timeout: float = 10.0) -> Optional[Any]:
    """
    Generic API fetching function.
    """
    try:
        url = f"{BACKEND_API_URL}/{endpoint}"
        print(f"[API] Fetching from: {url}")
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=timeout)
            print(f"[API] Response status for {endpoint}: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"[API] Received {len(data) if isinstance(data, list) else 'object'} from {endpoint}")
                return data
            else:
                print(f"[API] Error for {endpoint}: {response.status_code} - {response.text[:200]}")
                return None
    except Exception as e:
        print(f"[API] Exception fetching {endpoint}: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


async def fetch_food_drinks_from_api(category: Optional[str] = None) -> List[Dict]:
    """
    Fetch food and drinks from the backend API.
    Returns a list of food/drink items with images.
    """
    items = await fetch_from_api("food-drinks")
    if not items:
        return []
    
    # Filter by category if specified
    if category:
        category_lower = category.lower()
        if category_lower == 'drinks':
            items = [item for item in items if item.get('category', '').lower() == 'drinks']
        elif category_lower == 'food':
            items = [item for item in items if item.get('category', '').lower() != 'drinks']
    return items


async def fetch_rooms_from_api() -> List[Dict]:
    """Fetch rooms from the backend API."""
    rooms = await fetch_from_api("rooms")
    return rooms if rooms else []


async def fetch_offers_from_api() -> List[Dict]:
    """Fetch offers from the backend API."""
    offers = await fetch_from_api("offers")
    return offers if offers else []


async def fetch_gallery_from_api() -> List[Dict]:
    """Fetch gallery images from the backend API."""
    gallery = await fetch_from_api("gallery")
    return gallery if gallery else []


async def fetch_drivers_from_api() -> List[Dict]:
    """Fetch drivers from the backend API."""
    drivers = await fetch_from_api("drivers")
    return drivers if drivers else []


def search_knowledge(query: str, limit: int = 2) -> List[str]:
    q = query.lower()
    scored: List[Tuple[int, str]] = []
    for _, text in KNOWLEDGE:
        score = sum(1 for token in text.lower().split() if token in q)
        if score:
            scored.append((score, text))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [text for _, text in scored[:limit]]


# --- Tool stubs (replace with real integrations) ---
def tool_check_availability(message: str) -> Optional[str]:
    if any(k in message for k in ["room", "availability", "available", "stay", "dates"]):
        return (
            "I can check rooms. Please share: check-in date, nights, adults/children. "
            "Typical range: ₱1,500-₱5,000/night; I will quote exact once I have dates."
        )
    return None


def tool_quote_price(message: str) -> Optional[str]:
    if any(k in message for k in ["price", "rates", "rate", "magkano", "pila"]):
        return "Prices depend on date and room. Tell me your dates and guests; I will quote and hold the best refundable rate."
    return None


def tool_booking_actions(message: str) -> Optional[str]:
    if "cancel" in message or "modify" in message or "change" in message:
        return "I can modify/cancel if you share your booking code and last name. I will check penalties first."
    if "book" in message or "reserve" in message or "hold" in message:
        return "I can place a hold or confirm. Send dates, guests, and room preference; I will return a payment link."
    return None


def tool_promos(message: str) -> Optional[str]:
    if "promo" in message or "discount" in message:
        return "I will apply any eligible promos once you share dates/room type."
    return None


def tool_offers(message: str) -> Optional[str]:
    if any(k in message for k in ["dining", "food", "restaurant", "menu", "breakfast", "pickup", "transport"]):
        return (
            "Dining 8:00 AM - 10:00 PM; favorites: adobo sa gata, grilled fish, halo-halo. "
            "Need airport pickup or table booking? Tell me time and headcount."
        )
    return None


def tool_contact(message: str) -> Optional[str]:
    if "contact" in message or "email" in message or "location" in message:
        return "Email kinaiyahanforestpark@gmail.com; Bilar, Bohol 6317."
    return None


def build_answer(message: str, image_present: bool) -> str:
    """
    Build an answer using Gemini API with knowledge context.
    Falls back to basic response if Gemini is not available.
    """
    # Prepare context from knowledge base
    context_snippets = search_knowledge(message)
    context = " ".join(context_snippets) if context_snippets else ""
    
    # Build the prompt with context
    system_prompt = """You are a helpful, friendly chatbot assistant for Kinaiyahan Forest Park, 
a resort/hotel in Bilar, Bohol, Philippines. You are knowledgeable about room bookings, dining, events, and general inquiries.
Always respond in a warm, conversational manner like a real receptionist would.

IMPORTANT CONTEXT ABOUT KINAIYAHAN:
- Rooms: ₱1,500-₱5,000/night depending on room type and date
- Check-in: 2 PM, Check-out: 12 NN
- Restaurant hours: 8:00 AM - 10:00 PM
- Popular dishes: adobo sa gata, grilled fish, halo-halo
- Accepts: Cash, Visa/Mastercard, GCash, Bank Transfer, PayMaya
- Email: kinaiyahanforestpark@gmail.com
- Location: Bilar, Bohol 6317
- Services: Room bookings, dining, airport pickup, event hosting (weddings, birthdays, corporate events)

When responding:
1. Be warm and helpful
2. If you don't know something specific, offer to have them contact the resort
3. For bookings, always ask for dates and number of guests
4. For pricing, mention the range but suggest checking exact rates based on their dates
5. Keep responses concise but friendly (2-3 sentences typically)
"""
    
    user_prompt = f"{message}"
    if context:
        user_prompt += f"\n\n[Additional Context: {context}]"
    
    if image_present:
        user_prompt += "\n\n[Note: User has uploaded an image with this message]"
    
    try:
        if GEMINI_CLIENT:
            # Use Gemini API for natural responses
            response = GEMINI_CLIENT.models.generate_content(
                model=GEMINI_MODEL_NAME,
                contents=f"{system_prompt}\n\nUser: {user_prompt}",
            )
            text = getattr(response, "text", None)
            if text:
                return text.strip()
            return build_answer_fallback(message, image_present)
        else:
            # Fallback if API key or client is not configured
            return build_answer_fallback(message, image_present)
    except Exception as e:
        print(f"Gemini API error: {e}")
        return build_answer_fallback(message, image_present)


def search_dataset(query: str, category: Optional[str] = None) -> Dict[str, Any]:
    """Search dataset for relevant information."""
    if not DATASET:
        return {}
    
    query_lower = query.lower()
    results = {}
    
    # Search contact info
    if any(k in query_lower for k in ['contact', 'phone', 'email', 'location', 'address']):
        results['contact'] = DATASET.get('resort', {}).get('contact', {})
        results['location'] = DATASET.get('resort', {}).get('location', {})

    # Search hours / availability / open daily
    if any(k in query_lower for k in [
        'hour', 'hours', 'open', 'opening', 'closing', 'close', 'schedule',
        'time', 'times', 'daily', 'everyday', 'every day', 'available everyday', 'days open'
    ]):
        results['hours'] = DATASET.get('resort', {}).get('hours', {})
        # Also include dining/restaurant hours if present
        dining = DATASET.get('dining', {})
        if dining:
            results['restaurant'] = dining.get('restaurant', {})
    
    # Search pricing
    if any(k in query_lower for k in ['price', 'rates', 'cost', 'magkano', 'how much']):
        results['pricing'] = DATASET.get('pricing', {})
    
    # Search accommodations
    if any(k in query_lower for k in ['room', 'accommodation', 'stay', 'overnight', 'hotel', 'suite', 'deluxe', 'standard', 'family']):
        results['accommodations'] = DATASET.get('accommodations', {})
    
    # Search dining
    if any(k in query_lower for k in ['food', 'cafe', 'restaurant', 'menu', 'dining', 'eat']):
        results['dining'] = DATASET.get('dining', {})
    
    # Search activities
    if any(k in query_lower for k in ['activity', 'atv', 'trail', 'tour', 'adventure', 'things to do']):
        results['activities'] = DATASET.get('activities', [])
    
    # Search booking/policies
    if any(k in query_lower for k in ['book', 'reserve', 'cancel', 'policy', 'payment']):
        results['booking'] = DATASET.get('booking', {})
        results['policies'] = DATASET.get('policies', {})
    
    # Search events
    if any(k in query_lower for k in ['event', 'wedding', 'birthday', 'corporate', 'celebration']):
        results['events'] = DATASET.get('events', {})
    
    # Search FAQs
    if any(k in query_lower for k in ['how', 'why', 'when', 'what']):
        faqs = DATASET.get('faqs', [])
        matching_faqs = [faq for faq in faqs if any(kw in faq['question'].lower() for kw in query_lower.split())]
        if matching_faqs:
            results['faqs'] = matching_faqs[:3]
    
    return results


def classify_intent(message: str) -> Tuple[Optional[str], float]:
    """Classify the intent of the user message using trained model"""
    if not INTENT_CLASSIFIER:
        return None, 0.0
    
    try:
        # Predict intent
        predicted_intent = INTENT_CLASSIFIER.predict([message])[0]
        
        # Get confidence scores
        probabilities = INTENT_CLASSIFIER.predict_proba([message])[0]
        confidence = max(probabilities)
        
        return predicted_intent, confidence
    except Exception as e:
        print(f"Intent classification error: {e}")
        return None, 0.0


def build_answer_from_intent(intent: str, message: str) -> Optional[str]:
    """Build answer from classified intent using answer database - with human-like tone."""
    if not ANSWER_DATABASE or intent not in ANSWER_DATABASE:
        return None
    
    msg_lower = message.lower()
    intent_data = ANSWER_DATABASE.get(intent, {})
    
    # Handle GREETING intent
    if intent == "greeting":
        greetings = [
            "Hey there! 👋 I'm the Kinaiyahan AI Assistant. I'd be happy to help you with bookings, rooms, activities, dining, pricing—basically anything about our forest park! What's on your mind?",
            "Hello! 😊 Welcome to Kinaiyahan Forest Park! I'm here to answer questions about accommodation, food, activities, and how to make a reservation. What can I do for you?",
            "Hi! 🌿 Thanks for reaching out! I can help with room info, pricing, dining, tours, and event planning. What would you like to know?"
        ]
        return greetings[len(message) % len(greetings)]
    
    # Handle THANKS intent
    if intent == "thanks":
        thanks_responses = [
            "You're welcome! 😊 Anytime you need help planning your visit to Kinaiyahan, I'm here!",
            "Happy to help! Feel free to ask anything else. 🌳",
            "My pleasure! 😄 Don't hesitate to come back if you have more questions!"
        ]
        return thanks_responses[len(message) % len(thanks_responses)]
    
    # Handle CONTACT intent
    if intent == "contact":
        contact = intent_data.get('contact', {})
        location = intent_data.get('location', {})
        return (
            f"📍 **Location**: {location.get('city', 'Bilar')}, {location.get('province', 'Bohol')} {location.get('postal_code', '6317')}\n"
            f"📞 **Phone**: {contact.get('phone', '0960-385-1464')}\n"
            f"📧 **Email**: {contact.get('email', 'kinaiyahanforestpark@gmail.com')}\n"
            f"🌐 **Website**: {contact.get('website', 'kinaiyahan.com')}\n\n"
            f"💬 You can reach us anytime — we'd love to hear from you!"
        )
    
    # Handle PRICING intent
    if intent == "pricing":
        pricing = intent_data.get('data', {})
        day_pass = pricing.get('day_pass', {})
        tour = pricing.get('tour_package', {})
        return (
            f"💰 **Our Pricing**:\n\n"
            f"🎫 **{day_pass.get('name', 'Nature Escape Pass')}**: ₱{day_pass.get('price', 350)}/person\n"
            f"   Includes: Park entry + ₱75 food credit + bottled water\n\n"
            f"🚗 **{tour.get('name', 'Countryside Tour')}**: ₱{tour.get('price', 1400)}/person\n"
            f"   Guided tour (optional ATV/buffet add-ons available)\n\n"
            f"🏨 **Rooms**: ₱1,500-₱5,000/night depending on room type & date\n\n"
            f"*Want exact pricing for your dates? Just tell me when you're planning to visit!* 😊"
        )
    
    # Handle ROOMS intent
    if intent == "rooms":
        accommodations = intent_data.get('data', {})
        room_types = accommodations.get('room_types', [])
        msg = "🏨 **Our Room Options**:\n\n"
        for room in room_types[:3]:
            msg += f"• **{room.get('type')}** ({room.get('capacity')})\n"
            msg += f"  {room.get('price')}\n"
            amenities = room.get('amenities', [])
            if amenities:
                msg += f"  Amenities: {', '.join(amenities[:2])}"
                if len(amenities) > 2:
                    msg += f" +{len(amenities)-2} more"
                msg += "\n"
            msg += "\n"
        msg += "⏰ **Check-in**: 2 PM | **Check-out**: 12 noon\n\n"
        msg += "Want to know more or book? Call us at 0960-385-1464! 📞"
        return msg
    
    # Handle DINING intent
    if intent == "dining":
        dining = intent_data.get('data', {})
        restaurant = dining.get('restaurant', {})
        return (
            f"🍽️ **{restaurant.get('name', 'Forest Café')}**\n\n"
            f"⏰ **Hours**: {restaurant.get('opening_time', '8:00 AM')} - {restaurant.get('closing_time', '10:00 PM')} daily\n"
            f"🍜 **Specialty**: Filipino & International cuisine\n\n"
            f"Popular dishes include adobo sa gata, grilled fish, and delicious halo-halo! 😋\n\n"
            f"*Guests with Nature Escape Pass get ₱75 credit to enjoy our café!*"
        )
    
    # Handle ACTIVITIES intent
    if intent == "activities":
        activities = intent_data.get('data', [])
        msg = "🏃 **Popular Activities**:\n\n"
        for activity in activities[:5]:
            msg += f"• {activity.get('name')} ({activity.get('duration')})\n"
        msg += "\n*Want to try something specific? I can tell you more!* ✨"
        return msg
    
    # Handle EVENTS intent
    if intent == "events":
        events = intent_data.get('data', {})
        types = events.get('types', [])
        msg = "🎉 **Event Services**:\n\n"
        for event in types:
            msg += f"• {event.get('type')} (up to {event.get('capacity')} guests)\n"
        msg += "\n📧 **Contact us**: kinaiyahanforestpark@gmail.com or call 0960-385-1464\n"
        msg += "We'd love to make your event special! 🎊"
        return msg
    
    # Handle BOOKING intent
    if intent == "booking":
        booking = intent_data.get('data', {})
        methods = booking.get('methods', [])
        msg = "📅 **How to Book**:\n\n"
        for method in methods[:3]:
            if method.get('method') == 'Website':
                msg += f"🌐 **Website**: {method.get('url')}\n"
            elif method.get('method') == 'Phone':
                msg += f"📞 **Phone**: {method.get('number')}\n"
            elif method.get('method') == 'Email':
                msg += f"📧 **Email**: {method.get('email')}\n"
        msg += "\n*We're here to help make your booking smooth and easy!* 😊"
        return msg
    
    # Handle FAQS intent
    if intent == "faqs":
        faqs = intent_data.get('data', [])
        if faqs:
            return f"ℹ️ **{faqs[0].get('question', 'FAQ')}**\n\nA: {faqs[0].get('answer', 'Thank you for your question!')}\n\n*Any other questions? I'm here to help!*"
    
    return None


def build_answer_from_dataset(message: str) -> str:
    """Build answer using dataset information - more human-like responses."""
    msg_lower = message.lower()
    
    # Search dataset
    data = search_dataset(message)
    
    # Check for contact/location
    if 'contact' in data or 'location' in data:
        contact = data.get('contact', {})
        location = data.get('location', {})
        return (
            f"📍 **Where to find us**: {location.get('city', 'Bilar')}, {location.get('province', 'Bohol')} {location.get('postal_code', '6317')}\n"
            f"📞 **Call us**: {contact.get('phone', '0960-385-1464')}\n"
            f"📧 **Email**: {contact.get('email', 'kinaiyahanforestpark@gmail.com')}\n\n"
            f"We're always ready to help! 😊"
        )

    # Check for hours / availability
    if 'hours' in data:
        hours = data.get('hours', {})
        restaurant = data.get('restaurant', {})
        resort_days = hours.get('days_open', 'Daily')
        resort_open = hours.get('opening_time', '8:00 AM')
        resort_close = hours.get('closing_time', '6:00 PM')
        cafe_name = restaurant.get('name', 'Forest Café')
        cafe_open = restaurant.get('opening_time', '8:00 AM')
        cafe_close = restaurant.get('closing_time', '10:00 PM')

        if any(k in msg_lower for k in ['cafe', 'restaurant', 'food', 'drink', 'menu']):
            return (
                f"✅ **Yes, we're open {resort_days.lower()}!** 🎉\n\n"
                f"🍽️ **{cafe_name}**: {cafe_open} - {cafe_close} daily\n"
                f"🏞️ **Park hours**: {resort_open} - {resort_close}\n\n"
                f"Perfect time to visit us! 🌿"
            )
        else:
            return (
                f"✅ **Great news — we're open {resort_days.lower()}!**\n\n"
                f"🏞️ **Park**: {resort_open} - {resort_close}\n"
                f"🍽️ **{cafe_name}**: {cafe_open} - {cafe_close}\n\n"
                f"Come visit us anytime! 😊"
            )
    
    # Check for pricing
    if 'pricing' in data:
        pricing = data.get('pricing', {})
        day_pass = pricing.get('day_pass', {})
        tour = pricing.get('tour_package', {})
        msg = f"💰 **Our Pricing Guide**:\n\n"
        msg += f"🎫 **{day_pass.get('name', 'Nature Escape Pass')}**: ₱{day_pass.get('price', 350)}/person\n"
        msg += f"   ✓ Park entrance | ✓ ₱75 food credit | ✓ Bottled water\n\n"
        msg += f"🚗 **{tour.get('name', 'Countryside Tour')}**: ₱{tour.get('price', 1400)}/person\n"
        msg += f"   ✓ Guided experience | Optional ATV & buffet add-ons\n\n"
        msg += f"🏨 **Room rates**: ₱1,500-₱5,000/night (depends on room type & date)\n\n"
        msg += f"*Want exact quotes? Just tell me your dates and group size!* 😊"
        return msg
    
    # Check for rooms
    if 'accommodations' in data:
        accommodations = data.get('accommodations', {})
        room_types = accommodations.get('room_types', [])
        msg = "🏨 **Our Cozy Rooms**:\n\n"
        for room in room_types:
            msg += f"**{room.get('type')}** (sleeps {room.get('capacity')})\n"
            msg += f"💰 {room.get('price')}\n"
            amenities = room.get('amenities', [])
            if amenities:
                msg += f"✓ {', '.join(amenities[:3])}"
                if len(amenities) > 3:
                    msg += f", +{len(amenities)-3} more"
                msg += "\n"
            msg += "\n"
        msg += "⏰ **Check-in**: 2 PM | **Check-out**: 12 noon\n\n"
        msg += "📞 Ready to book? Call 0960-385-1464!"
        return msg
    
    # Check for dining
    if 'dining' in data:
        dining = data.get('dining', {})
        restaurant = dining.get('restaurant', {})
        return (
            f"🍽️ **{restaurant.get('name', 'Forest Café')}**\n\n"
            f"⏰ Open {restaurant.get('opening_time', '8 AM')} - {restaurant.get('closing_time', '10 PM')} daily\n"
            f"🍜 Filipino & International cuisine\n\n"
            f"Your Nature Escape Pass includes ₱75 to spend here! 🤤\n"
            f"Favorites: adobo sa gata, grilled fish, halo-halo ✨"
        )
    
    # Check for activities
    if 'activities' in data:
        activities = data.get('activities', [])
        msg = "🏃 **Things to Do at Kinaiyahan**:\n\n"
        for activity in activities[:5]:
            msg += f"• {activity.get('name')} ({activity.get('duration')})\n"
        msg += "\n*Want adventure? We've got you covered!* 🌳✨"
        return msg
    
    # Check for events
    if 'events' in data:
        events = data.get('events', {})
        types = events.get('types', [])
        msg = "🎉 **Host Your Special Event Here!**\n\n"
        for event in types:
            msg += f"• {event.get('type')} (up to {event.get('capacity')} guests)\n"
        msg += "\n📧 kinaiyahanforestpark@gmail.com or 📞 0960-385-1464\n"
        msg += "Let's make your event unforgettable! 🎊"
        return msg
    
    # Check for booking
    if 'booking' in data:
        booking = data.get('booking', {})
        methods = booking.get('methods', [])
        msg = "📅 **Easy Booking Options**:\n\n"
        for method in methods[:3]:
            if method.get('method') == 'Website':
                msg += f"🌐 Visit: {method.get('url')}\n"
            elif method.get('method') == 'Phone':
                msg += f"📞 Call: {method.get('number')}\n"
            elif method.get('method') == 'Email':
                msg += f"📧 Email: {method.get('email')}\n"
        msg += "\n*We'll make sure you get the best deal available!* 😊"
        return msg
    
    # Check for FAQs
    if 'faqs' in data:
        faqs = data.get('faqs', [])
        if faqs:
            return f"**Q: {faqs[0].get('question')}**\n\nA: {faqs[0].get('answer', 'Great question!')}\n\n*Need more info? Let me know!* 😊"
    
    return None


def build_answer_fallback(message: str, image_present: bool) -> str:
    """Enhanced response builder with ML intent classification and dataset support - more human-like."""
    msg_lower = message.lower()
    
    # Step 1: Try ML intent classification
    if INTENT_CLASSIFIER:
        intent, confidence = classify_intent(message)
        print(f"[ML] Classified intent: {intent} (confidence: {confidence:.2%})")
        
        if intent and confidence > 0.65:  # Higher threshold for confident answers
            answer = build_answer_from_intent(intent, message)
            if answer:
                return answer
        elif intent and confidence > 0.45:
            # Moderate confidence - answer with disclaimer
            answer = build_answer_from_intent(intent, message)
            if answer:
                return f"{answer}\n\n💡 *Tip: For the most accurate info, call us at 0960-385-1464!*"
    
    # Step 2: Try dataset-based search
    dataset_answer = build_answer_from_dataset(message)
    if dataset_answer:
        return dataset_answer
    
    # Step 3: Intelligent fallback based on question type
    # Check if it's a question we might not know
    question_keywords = ['how', 'what', 'when', 'where', 'why', 'can', 'do', 'is', 'are']
    is_question = any(msg_lower.startswith(kw) for kw in question_keywords) or '?' in message
    
    if is_question:
        # More honest and human response for questions we can't answer
        honest_responses = [
            f"🤔 That's a great question! I'm not entirely sure about this specific detail. "
            f"But our team at Kinaiyahan knows everything! Give us a call at 0960-385-1464 or email kinaiyahanforestpark@gmail.com, "
            f"and they'll give you the perfect answer. What else can I help with?",
            
            f"😊 I appreciate the question! I don't have specific information about that in my knowledge base. "
            f"However, our friendly team would love to help! Reach out at 0960-385-1464 or kinaiyahanforestpark@gmail.com. "
            f"In the meantime, is there anything else about rooms, dining, or activities I can tell you about?",
            
            f"That's something I'm not 100% sure about 😅. But don't worry - our team knows all the details! "
            f"Call 0960-385-1464 or email kinaiyahanforestpark@gmail.com for accurate information. "
            f"I'm here to help with bookings, pricing, dining, and more though!",
        ]
        return honest_responses[len(message) % len(honest_responses)]
    else:
        # For statements or commands
        return (
            "I understand! 😊 Here's what I can definitely help you with:\n"
            "✅ Room bookings & pricing\n"
            "✅ Dining & menu questions\n"
            "✅ Activities & tours\n"
            "✅ Event planning\n"
            "✅ Contact information\n\n"
            "For other questions, our amazing team is just a call away: 0960-385-1464 or kinaiyahanforestpark@gmail.com"
        )



@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    message = req.message.strip()
    if not message:
        return ChatResponse(answer="Please enter a message so I can help.")

    msg_lower = message.lower()
    
    # Check for food/drinks menu requests
    is_food_request = any(keyword in msg_lower for keyword in [
        'food', 'menu', 'eat', 'drink', 'beverage', 'meal', 'dish', 
        'cafe', 'restaurant', 'available food', 'available drinks',
        'what food', 'what drinks', 'food items', 'drink items'
    ])
    
    # Check for room requests
    is_room_request = any(keyword in msg_lower for keyword in [
        'room', 'rooms', 'accommodation', 'stay', 'overnight', 'suite', 'deluxe', 'standard',
        'available rooms', 'show me rooms', 'what rooms'
    ])
    
    # Check for offers/deals requests
    is_offer_request = any(keyword in msg_lower for keyword in [
        'offer', 'offers', 'deal', 'deals', 'promo', 'promotion', 'discount', 'package',
        'available offers', 'show me offers'
    ])
    
    # Check for gallery/photos requests
    is_gallery_request = any(keyword in msg_lower for keyword in [
        'photo', 'photos', 'picture', 'pictures', 'gallery', 'image', 'images',
        'show me photos', 'view photos'
    ])
    
    # Check for driver/transportation requests
    is_driver_request = any(keyword in msg_lower for keyword in [
        'driver', 'drivers', 'transport', 'transportation', 'pickup', 'ride',
        'available drivers', 'show me drivers'
    ])
    
    # Handle food/drinks requests
    if is_food_request and not is_room_request:
        category = None
        if 'drink' in msg_lower or 'beverage' in msg_lower:
            category = 'drinks'
        elif 'food' in msg_lower or 'menu' in msg_lower or 'eat' in msg_lower:
            if 'food and drink' in msg_lower or 'food & drink' in msg_lower:
                category = None
            else:
                category = 'food'
        
        items = await fetch_food_drinks_from_api(category)
        print(f"[Chat] Fetched {len(items) if items else 0} food/drink items")
        
        if items:
            # Generate natural language response
            if category == 'drinks':
                greetings = [
                    f"🍹 Great choice! We have {len(items)} refreshing drinks available.",
                    f"🍹 Here's our drink menu with {len(items)} delicious options!",
                    f"🍹 Perfect! Let me show you our {len(items)} beverage selections."
                ]
                answer = greetings[len(items) % 3] + "\n\n"
            elif category == 'food':
                greetings = [
                    f"🍽️ Wonderful! Our kitchen has {len(items)} delicious dishes for you.",
                    f"🍽️ Here's our food menu featuring {len(items)} tasty options!",
                    f"🍽️ Excellent! Check out these {len(items)} mouth-watering dishes."
                ]
                answer = greetings[len(items) % 3] + "\n\n"
            else:
                food_count = sum(1 for item in items if item.get('category', '').lower() != 'drinks')
                drink_count = len(items) - food_count
                answer = f"🍽️ Fantastic! Here's our complete Forest Café menu with {food_count} delicious food items and {drink_count} refreshing drinks!\n\n"
            
            # Show sample items
            for i, item in enumerate(items[:3], 1):
                name = item.get('name', 'Unknown')
                price = item.get('price', '0')
                category_name = item.get('category', 'Item')
                availability = item.get('availability', 'Available')
                desc = item.get('description', '')
                answer += f"{i}. **{name}** - ₱{price}\n"
                if desc:
                    answer += f"   {desc[:50]}...\n"
                answer += f"   Category: {category_name} | Status: {availability}\n\n"
            
            if len(items) > 3:
                answer += f"...and {len(items) - 3} more delicious options below!\n\n"
            
            # Add contextual closing based on user query
            if 'show' in msg_lower or 'see' in msg_lower:
                answer += "All items are displayed below with photos and details. 😊"
            else:
                answer += "Forest Café is open daily from 8:00 AM to 10:00 PM. Enjoy! 🌿"
            
            return ChatResponse(answer=answer, items=items)
        else:
            # More natural error message based on context
            if 'what' in msg_lower:
                answer = "🍽️ I'd love to tell you about our menu, but I'm having trouble accessing it right now. Our Forest Café serves Filipino and international cuisine from 8:00 AM to 10:00 PM daily. Call us at 0960-385-1464 and we'll tell you all about our delicious offerings!"
            else:
                answer = "🍽️ I couldn't load the menu at this moment. But don't worry! Our Forest Café is open 8:00 AM - 10:00 PM daily with amazing Filipino and international dishes. Call 0960-385-1464 to hear about our specialties!"
            return ChatResponse(answer=answer)
    
    # Handle room requests
    if is_room_request:
        rooms = await fetch_rooms_from_api()
        if rooms:
            answer = f"🏨 Here are our available rooms ({len(rooms)} types):\n\n"
            for i, room in enumerate(rooms[:5], 1):
                name = room.get('name', 'Room')
                room_type = room.get('type', 'Standard')
                capacity = room.get('capacity', 'N/A')
                price = room.get('price', '0')
                answer += f"{i}. **{name}** ({room_type})\n   Capacity: {capacity} | Price: ₱{price}/night\n\n"
            
            if len(rooms) > 5:
                answer += f"...and {len(rooms) - 5} more rooms!\n\n"
            
            answer += "📞 Call 0960-385-1464 to book or check availability!"
            return ChatResponse(answer=answer, items=rooms)
        else:
            answer = "🏨 I couldn't fetch room info right now. Please call 0960-385-1464 for room availability."
            return ChatResponse(answer=answer)
    
    # Handle offers requests
    if is_offer_request:
        offers = await fetch_offers_from_api()
        if offers:
            answer = f"🎁 Here are our current offers ({len(offers)} available):\n\n"
            for i, offer in enumerate(offers[:5], 1):
                title = offer.get('title', 'Special Offer')
                description = offer.get('description', '')
                answer += f"{i}. **{title}**\n   {description[:100]}...\n\n"
            
            if len(offers) > 5:
                answer += f"...and {len(offers) - 5} more offers!\n\n"
            
            answer += "Visit our website or call 0960-385-1464 for details!"
            return ChatResponse(answer=answer, items=offers)
        else:
            answer = "🎁 I couldn't fetch current offers. Please visit our website or call 0960-385-1464."
            return ChatResponse(answer=answer)
    
    # Handle gallery requests
    if is_gallery_request:
        gallery = await fetch_gallery_from_api()
        if gallery:
            answer = f"📸 Here are some photos from Kinaiyahan ({len(gallery)} images):\n\n"
            answer += "Check out our beautiful forest park, accommodations, and dining areas!\n\n"
            answer += "Visit our gallery page to see all photos. 🌳✨"
            return ChatResponse(answer=answer, items=gallery[:10])
        else:
            answer = "📸 I couldn't fetch gallery images right now. Please visit our website gallery."
            return ChatResponse(answer=answer)
    
    # Handle driver requests
    if is_driver_request:
        drivers = await fetch_drivers_from_api()
        if drivers:
            answer = f"🚗 Here are our available drivers ({len(drivers)}):\n\n"
            for i, driver in enumerate(drivers[:3], 1):
                name = driver.get('name', 'Driver')
                vehicle = driver.get('vehicle_type', 'Vehicle')
                answer += f"{i}. **{name}** - {vehicle}\n"
            
            if len(drivers) > 3:
                answer += f"\n...and {len(drivers) - 3} more drivers available!\n\n"
            
            answer += "\n📞 Call 0960-385-1464 to book transportation!"
            return ChatResponse(answer=answer, items=drivers)
        else:
            answer = "🚗 I couldn't fetch driver info right now. Please call 0960-385-1464 for transportation."
            return ChatResponse(answer=answer)
    
    # Regular chatbot response for general queries
    answer = build_answer(message, bool(req.imageBase64))
    return ChatResponse(answer=answer)


@app.get("/health")
async def health():
    return {"status": "ok"}
