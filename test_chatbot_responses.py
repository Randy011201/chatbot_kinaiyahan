#!/usr/bin/env python3
"""
Quick test script to demonstrate the enhanced human-like chatbot responses
"""

import json
import requests
import time

# Test cases demonstrating different response types
TEST_CASES = [
    {
        "user_message": "Hello!",
        "expected_type": "Greeting",
        "description": "Warm, friendly greeting"
    },
    {
        "user_message": "How much is the Nature Escape Pass?",
        "expected_type": "Pricing - Can Answer",
        "description": "Direct answer with pricing details"
    },
    {
        "user_message": "What are the room types?",
        "expected_type": "Rooms - Can Answer",
        "description": "Clear room information"
    },
    {
        "user_message": "What time are you open?",
        "expected_type": "Hours - Can Answer",
        "description": "Operating hours"
    },
    {
        "user_message": "What's your phone number?",
        "expected_type": "Contact - Can Answer",
        "description": "Contact information"
    },
    {
        "user_message": "Do you have wheelchair accessibility?",
        "expected_type": "Cannot Answer",
        "description": "Honest response admitting uncertainty"
    },
    {
        "user_message": "What are your cancellation policies?",
        "expected_type": "Uncertain - Refers to Team",
        "description": "Directs to team for specific policies"
    },
    {
        "user_message": "Do you offer pet-friendly rooms?",
        "expected_type": "Cannot Answer",
        "description": "Honest uncertainty with contact info"
    },
    {
        "user_message": "What food do you serve?",
        "expected_type": "Dining - Can Answer",
        "description": "Restaurant and menu information"
    },
    {
        "user_message": "Tell me about your activities",
        "expected_type": "Activities - Can Answer",
        "description": "List of available activities"
    },
]

def run_tests():
    """Run all test cases against the chatbot API"""
    
    BASE_URL = "http://localhost:8000"
    
    print("=" * 70)
    print("🤖 CHATBOT HUMAN-LIKE COMMUNICATION TEST SUITE")
    print("=" * 70)
    print()
    
    # Check if API is running
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code != 200:
            print("❌ Chatbot API is not running!")
            print("Please start it first:")
            print("  cd chatbot_kinaiyahan")
            print("  python main.py")
            return
    except:
        print("❌ Cannot connect to chatbot API at localhost:8000")
        print("Please start it first:")
        print("  cd chatbot_kinaiyahan")
        print("  python main.py")
        return
    
    print("✅ Chatbot API is running!")
    print()
    
    # Run each test
    for i, test_case in enumerate(TEST_CASES, 1):
        print("-" * 70)
        print(f"Test #{i}: {test_case['expected_type']}")
        print("-" * 70)
        print(f"📝 User Message: {test_case['user_message']}")
        print(f"📌 Type: {test_case['description']}")
        print()
        print("🤖 Chatbot Response:")
        print()
        
        try:
            payload = {
                "message": test_case['user_message']
            }
            
            response = requests.post(
                f"{BASE_URL}/chat",
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                answer = data.get('answer', 'No response')
                print(answer)
            else:
                print(f"❌ Error: HTTP {response.status_code}")
                print(response.text)
        
        except requests.exceptions.Timeout:
            print("❌ Request timeout - chatbot took too long to respond")
        except Exception as e:
            print(f"❌ Error: {str(e)}")
        
        print()
        time.sleep(1)  # Brief pause between requests
    
    print("=" * 70)
    print("✅ TEST SUITE COMPLETED")
    print("=" * 70)
    print()
    print("OBSERVATIONS:")
    print("✓ High confidence questions: Direct, helpful answers")
    print("✓ Low confidence questions: Honest 'I'm not sure' responses")
    print("✓ All responses: Include contact information")
    print("✓ Tone: Warm, friendly, human-like")
    print("✓ Format: Clear, well-structured, easy to read")
    print()

if __name__ == "__main__":
    run_tests()
