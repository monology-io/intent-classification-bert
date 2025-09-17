import json
import torch
import logging
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Dict, List, Union
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IntentClassifier:
    """Intent classification inference class for real-time predictions."""
    
    def __init__(self, model_path: str, device: str = None):
        """
        Initialize the intent classifier.
        
        Args:
            model_path: Path to the trained model directory
            device: Device to run inference on ('cpu', 'cuda', or None for auto)
        """
        self.model_path = model_path
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model, tokenizer, and mappings
        self._load_model()
        self._load_mappings()
        
        logger.info(f"IntentClassifier initialized on {self.device}")
        logger.info(f"Available intents: {list(self.id_to_intent.values())}")
    
    def _load_model(self):
        """Load the trained model and tokenizer."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"Model loaded successfully from {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def _load_mappings(self):
        """Load intent mappings from saved configuration."""
        try:
            mapping_path = f"{self.model_path}/intent_mapping.json"
            with open(mapping_path, 'r') as f:
                self.intent_to_id = json.load(f)
            
            # Create reverse mapping
            self.id_to_intent = {v: k for k, v in self.intent_to_id.items()}
            logger.info("Intent mappings loaded successfully")
        except FileNotFoundError:
            logger.warning("Intent mapping file not found, using default mappings")
            self._create_default_mappings()
    
    def _create_default_mappings(self):
        """Create default intent mappings if file is not found."""
        self.intent_to_id = {
            "Requirement Submission": 0,
            "General Query": 1,
            "Contact Details": 2,
            "Feedback Submission": 3,
            "Appreciation": 4,
            "Greeting": 5,
            "Job Application Submission": 6,
        }
        self.id_to_intent = {v: k for k, v in self.intent_to_id.items()}
    
    def predict(self, text: str, return_probabilities: bool = False) -> Dict:
        """
        Predict intent for a single text input.
        
        Args:
            text: Input text to classify
            return_probabilities: Whether to return all class probabilities
            
        Returns:
            Dictionary containing predicted intent, confidence, and optionally all probabilities
        """
        if not text or not text.strip():
            return {
                "intent": "Unknown",
                "confidence": 0.0,
                "error": "Empty input text"
            }
        
        try:
            # Tokenize input
            inputs = self.tokenizer(
                text.strip(),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128
            ).to(self.device)
            
            # Get prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predicted_id = torch.argmax(probabilities, dim=-1).item()
                confidence = probabilities[0][predicted_id].item()
            
            # Prepare result
            result = {
                "intent": self.id_to_intent.get(predicted_id, "Unknown"),
                "confidence": confidence,
                "predicted_id": predicted_id
            }
            
            if return_probabilities:
                all_probs = {
                    self.id_to_intent[i]: float(prob) 
                    for i, prob in enumerate(probabilities[0])
                }
                result["all_probabilities"] = all_probs
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            return {
                "intent": "Error",
                "confidence": 0.0,
                "error": str(e)
            }
    
    def predict_batch(self, texts: List[str], return_probabilities: bool = False) -> List[Dict]:
        """
        Predict intents for a batch of texts.
        
        Args:
            texts: List of input texts to classify
            return_probabilities: Whether to return all class probabilities
            
        Returns:
            List of prediction dictionaries
        """
        if not texts:
            return []
        
        try:
            # Filter out empty texts
            valid_texts = [text.strip() for text in texts if text and text.strip()]
            if not valid_texts:
                return [{"intent": "Unknown", "confidence": 0.0, "error": "Empty input"} for _ in texts]
            
            # Tokenize all inputs
            inputs = self.tokenizer(
                valid_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128
            ).to(self.device)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predicted_ids = torch.argmax(probabilities, dim=-1)
            
            # Prepare results
            results = []
            for i, (pred_id, text) in enumerate(zip(predicted_ids, valid_texts)):
                pred_id = pred_id.item()
                confidence = probabilities[i][pred_id].item()
                
                result = {
                    "text": text,
                    "intent": self.id_to_intent.get(pred_id, "Unknown"),
                    "confidence": confidence,
                    "predicted_id": pred_id
                }
                
                if return_probabilities:
                    all_probs = {
                        self.id_to_intent[j]: float(prob) 
                        for j, prob in enumerate(probabilities[i])
                    }
                    result["all_probabilities"] = all_probs
                
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Batch prediction failed: {str(e)}")
            return [{"intent": "Error", "confidence": 0.0, "error": str(e)} for _ in texts]
    
    def get_intent_distribution(self, texts: List[str]) -> Dict:
        """
        Get distribution of intents across a list of texts.
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            Dictionary with intent counts and percentages
        """
        predictions = self.predict_batch(texts)
        intent_counts = {}
        total = len(predictions)
        
        for pred in predictions:
            intent = pred.get("intent", "Unknown")
            intent_counts[intent] = intent_counts.get(intent, 0) + 1
        
        # Calculate percentages
        intent_distribution = {
            intent: {
                "count": count,
                "percentage": (count / total) * 100 if total > 0 else 0
            }
            for intent, count in intent_counts.items()
        }
        
        return {
            "total_texts": total,
            "distribution": intent_distribution
        }

def main():
    """Example usage of the IntentClassifier."""
    # Initialize classifier
    classifier = IntentClassifier("./intent_model")
    
    # Updated example texts that match the 350-sample dataset patterns
    test_texts = [
        # Greeting examples
        "Hello there, how can I get started?",
        "Hi, good morning!",
        "Hey, what's up?",
        
        # Requirement Submission examples
        "I need to submit a technical requirement for my project",
        "Can you help me submit my system requirements?",
        "Here are the specifications for my software project",
        
        # General Query examples
        "What services do you offer?",
        "Can you tell me more about your platform?",
        "How does your system work?",
        
        # Contact Details examples
        "What are your contact details?",
        "How can I reach your support team?",
        "I need your phone number and email address",
        
        # Feedback Submission examples
        "I want to share my feedback about your service",
        "Here's my review of your platform",
        "I'd like to provide some suggestions for improvement",
        
        # Appreciation examples
        "Thank you for the excellent support!",
        "Great job on the quick response time",
        "I really appreciate your help with this issue",
        
        # Job Application Submission examples
        "I would like to apply for the software engineer position",
        "Can I submit my resume for the open developer role?",
        "I'm interested in the data scientist job posting"
    ]
    
    print("=== Single Predictions ===")
    for text in test_texts:
        result = classifier.predict(text, return_probabilities=True)
        print(f"Text: {text}")
        print(f"Intent: {result['intent']} (Confidence: {result['confidence']:.3f})")
        print("---")
    
    print("\n=== Batch Predictions ===")
    batch_results = classifier.predict_batch(test_texts)
    for result in batch_results:
        print(f"'{result['text'][:50]}...' -> {result['intent']} ({result['confidence']:.3f})")
    
    print("\n=== Intent Distribution ===")
    distribution = classifier.get_intent_distribution(test_texts)
    print(f"Total texts: {distribution['total_texts']}")
    for intent, stats in distribution['distribution'].items():
        print(f"{intent}: {stats['count']} ({stats['percentage']:.1f}%)")

if __name__ == "__main__":
    main()