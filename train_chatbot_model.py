#!/usr/bin/env python3
"""
Kinaiyahan Chatbot ML Training Pipeline
Trains intent classification and question-answer models from the dataset
"""

import json
import os
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
import joblib
from datetime import datetime

class ChatbotTrainer:
    def __init__(self):
        self.dataset_path = Path(__file__).parent / "dataset.json"
        self.models_dir = Path(__file__).parent / "models"
        self.models_dir.mkdir(exist_ok=True)
        self.dataset = {}
        self.training_data = []
        
    def load_dataset(self):
        """Load the dataset"""
        print("[LOAD] Loading dataset...")
        try:
            with open(self.dataset_path, 'r') as f:
                self.dataset = json.load(f)
            print(f"✓ Dataset loaded successfully")
            return True
        except FileNotFoundError:
            print(f"✗ Dataset not found at {self.dataset_path}")
            return False
        except json.JSONDecodeError as e:
            print(f"✗ Invalid JSON in dataset: {e}")
            return False
    
    def extract_training_samples(self):
        """Extract training samples from dataset"""
        print("\n[EXTRACT] Extracting training samples...")
        
        samples = []
        
        # Intent: PRICING
        samples.extend([
            ("what is the price", "pricing"),
            ("how much does it cost", "pricing"),
            ("what are your rates", "pricing"),
            ("how much for a room", "pricing"),
            ("pricing information", "pricing"),
            ("cost per person", "pricing"),
            ("price of day pass", "pricing"),
            ("room rates", "pricing"),
            ("magkano ang bayad", "pricing"),
            ("pila ang presyo", "pricing"),
        ])
        
        # Intent: BOOKING
        samples.extend([
            ("how do i book", "booking"),
            ("make a reservation", "booking"),
            ("i want to reserve", "booking"),
            ("book a room", "booking"),
            ("check my booking code", "booking"),
            ("booking methods", "booking"),
            ("how to reserve", "booking"),
            ("reservation process", "booking"),
            ("gusto ko mag-book", "booking"),
            ("paano mag-reserve", "booking"),
        ])
        
        # Intent: CONTACT
        samples.extend([
            ("what is your phone number", "contact"),
            ("how do i contact you", "contact"),
            ("email address", "contact"),
            ("where are you located", "contact"),
            ("what is your address", "contact"),
            ("contact information", "contact"),
            ("phone number", "contact"),
            ("location", "contact"),
            ("saan kayo", "contact"),
            ("ano ang number ninyo", "contact"),
        ])
        
        # Intent: ROOMS
        samples.extend([
            ("do you have rooms", "rooms"),
            ("room types", "rooms"),
            ("accommodations available", "rooms"),
            ("types of rooms", "rooms"),
            ("room features", "rooms"),
            ("can i stay overnight", "rooms"),
            ("lodge information", "rooms"),
            ("available rooms", "rooms"),
            ("may rooms ba kayo", "rooms"),
            ("ano ang rooms ninyo", "rooms"),
        ])
        
        # Intent: DINING
        samples.extend([
            ("do you have a restaurant", "dining"),
            ("food menu", "dining"),
            ("what can i eat", "dining"),
            ("dining options", "dining"),
            ("cafe hours", "dining"),
            ("food and beverage", "dining"),
            ("what do you serve", "dining"),
            ("restaurant hours", "dining"),
            ("may kain ba dito", "dining"),
            ("anong pagkain ang meron", "dining"),
        ])
        
        # Intent: ACTIVITIES
        samples.extend([
            ("what activities do you have", "activities"),
            ("things to do", "activities"),
            ("adventures available", "activities"),
            ("tour options", "activities"),
            ("atv rental", "activities"),
            ("guided tours", "activities"),
            ("activities list", "activities"),
            ("what can we do", "activities"),
            ("ano ang activities", "activities"),
            ("may tour ba kayo", "activities"),
        ])
        
        # Intent: EVENTS
        samples.extend([
            ("can you host events", "events"),
            ("wedding venue", "events"),
            ("birthday party", "events"),
            ("corporate events", "events"),
            ("event hosting", "events"),
            ("celebration packages", "events"),
            ("event services", "events"),
            ("venue rental", "events"),
            ("pwede ba mag-event", "events"),
            ("may event space ba", "events"),
        ])
        
        # Intent: GREETING
        samples.extend([
            ("hello", "greeting"),
            ("hi there", "greeting"),
            ("hey", "greeting"),
            ("good morning", "greeting"),
            ("how are you", "greeting"),
            ("kumusta ka", "greeting"),
            ("hello there", "greeting"),
            ("greetings", "greeting"),
            ("hi", "greeting"),
            ("buenos dias", "greeting"),
        ])
        
        # Intent: THANK YOU
        samples.extend([
            ("thank you", "thanks"),
            ("thanks", "thanks"),
            ("appreciate it", "thanks"),
            ("salamat", "thanks"),
            ("maraming salamat", "thanks"),
            ("thank you so much", "thanks"),
            ("thanks a lot", "thanks"),
            ("danke", "thanks"),
            ("gracias", "thanks"),
            ("much appreciated", "thanks"),
        ])
        
        self.training_data = samples
        print(f"✓ Extracted {len(samples)} training samples")
        print(f"  - Intents: {len(set([s[1] for s in samples]))}")
        for intent in set([s[1] for s in samples]):
            count = len([s for s in samples if s[1] == intent])
            print(f"    • {intent}: {count} samples")
        
        return samples
    
    def train_intent_classifier(self):
        """Train intent classification model"""
        print("\n[TRAIN] Training intent classifier...")
        
        texts = [s[0] for s in self.training_data]
        labels = [s[1] for s in self.training_data]
        
        # Create pipeline
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                lowercase=True,
                stop_words=['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'],
                ngram_range=(1, 2),
                max_features=500
            )),
            ('classifier', MultinomialNB())
        ])
        
        # Train
        pipeline.fit(texts, labels)
        
        # Save
        model_path = self.models_dir / "intent_classifier.pkl"
        joblib.dump(pipeline, model_path)
        print(f"✓ Intent classifier trained and saved to {model_path}")
        
        # Evaluate on training data
        accuracy = pipeline.score(texts, labels)
        print(f"  Training accuracy: {accuracy:.2%}")
        
        return pipeline
    
    def train_answer_db(self):
        """Create answer database from dataset"""
        print("\n[BUILD] Building answer database...")
        
        answer_db = {
            "pricing": {
                "category": "Pricing",
                "data": self.dataset.get("pricing", {})
            },
            "booking": {
                "category": "Booking",
                "data": self.dataset.get("booking", {})
            },
            "contact": {
                "category": "Contact",
                "contact": self.dataset.get("resort", {}).get("contact", {}),
                "location": self.dataset.get("resort", {}).get("location", {})
            },
            "rooms": {
                "category": "Accommodations",
                "data": self.dataset.get("accommodations", {})
            },
            "dining": {
                "category": "Dining",
                "data": self.dataset.get("dining", {})
            },
            "activities": {
                "category": "Activities",
                "data": self.dataset.get("activities", [])
            },
            "events": {
                "category": "Events",
                "data": self.dataset.get("events", {})
            },
            "faqs": {
                "category": "FAQs",
                "data": self.dataset.get("faqs", [])
            }
        }
        
        # Save answer database
        db_path = self.models_dir / "answer_database.json"
        with open(db_path, 'w') as f:
            json.dump(answer_db, f, indent=2)
        
        print(f"✓ Answer database saved to {db_path}")
        print(f"  Covered intents: {list(answer_db.keys())}")
        
        return answer_db
    
    def create_training_report(self):
        """Create training report"""
        print("\n[REPORT] Creating training report...")
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "model_version": "1.0",
            "training_samples": len(self.training_data),
            "intents_trained": list(set([s[1] for s in self.training_data])),
            "intent_distribution": {
                intent: len([s for s in self.training_data if s[1] == intent])
                for intent in set([s[1] for s in self.training_data])
            },
            "features": [
                "Intent Classification (TF-IDF + Naive Bayes)",
                "Answer Database from Dataset",
                "Multi-language support (English + Tagalog)",
                "FAQ matching"
            ],
            "model_files": [
                "intent_classifier.pkl",
                "answer_database.json"
            ]
        }
        
        report_path = self.models_dir / "training_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"✓ Training report saved to {report_path}")
        
        return report
    
    def train(self):
        """Run full training pipeline"""
        print("\n" + "="*60)
        print("KINAIYAHAN CHATBOT ML TRAINING PIPELINE")
        print("="*60)
        
        # Step 1: Load dataset
        if not self.load_dataset():
            return False
        
        # Step 2: Extract training samples
        self.extract_training_samples()
        
        # Step 3: Train intent classifier
        self.train_intent_classifier()
        
        # Step 4: Build answer database
        self.train_answer_db()
        
        # Step 5: Create report
        self.create_training_report()
        
        print("\n" + "="*60)
        print("✓ TRAINING COMPLETE!")
        print("="*60)
        print(f"\nModels saved in: {self.models_dir}")
        print("\nTo use the trained model:")
        print("1. Restart the chatbot server")
        print("2. The chatbot will automatically load the trained models")
        print("3. Intent classification will be used for better accuracy")
        print("\n")
        
        return True

if __name__ == "__main__":
    trainer = ChatbotTrainer()
    success = trainer.train()
    exit(0 if success else 1)
