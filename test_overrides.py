import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from app import predict_intent, load_model

clf, vectorizer = load_model()

tests = [
    "Where is my order?",
    "I want a refund for my broken product.",
    "My package was stolen.",
    "I received the wrong item.",
    "Cancel my prime membership.",
    "I cannot access my account.",
    "Talk to a real person.",
    "Thank you."
]

for t in tests:
    intent, conf, needs_esc = predict_intent(t, clf, vectorizer)
    print(f"[{t}] -> Intent: {intent}, Conf: {conf*100:.1f}%, Escalated: {needs_esc}")
