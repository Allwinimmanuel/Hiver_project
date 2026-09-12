"""
Test the agent on sample messages for submission verification.
"""
import sys
import importlib.util

spec = importlib.util.spec_from_file_location("classifier", "scripts/13_agent_classifier.py")
classifier = importlib.util.module_from_spec(spec)
sys.modules["classifier"] = classifier
spec.loader.exec_module(classifier)
predict_intent = classifier.predict_intent

samples = [
    "Where is my order?",
    "My package was stolen.",
    "I want a refund.",
    "My payment failed.",
    "I cannot log into my account.",
    "Tell me about this product.",
    "I want a refund for my broken product.",
    "A completely unusual problem."
]

print("=" * 60)
print("  SAMPLE MESSAGE TEST RESULTS")
print("=" * 60)

for msg in samples:
    intent, conf = predict_intent(msg)
    flag = "[ESCALATE - LOW CONFIDENCE]" if conf < 0.60 else "[OK]"
    print(f"\nMessage:    {msg}")
    print(f"Predicted:  {intent}")
    print(f"Confidence: {conf:.2%}  {flag}")

print("\n" + "=" * 60)
print("  ALL TESTS COMPLETE")
print("=" * 60)
