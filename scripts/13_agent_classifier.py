import pandas as pd
from pathlib import Path
import sys
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Recommended output directory
MODELS_DIR = Path("models")
MODEL_PATH = MODELS_DIR / "intent_classifier.joblib"
VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.joblib"
TRAIN_PATH = Path("data/processed/train_data.csv")

def train_model():
    if not TRAIN_PATH.exists():
        print(f"[ERROR] Training data not found at {TRAIN_PATH}")
        sys.exit(1)
        
    print("Loading training data...")
    df = pd.read_csv(TRAIN_PATH)
    
    # Drop NaNs just in case
    df = df.dropna(subset=['customer_text', 'intent'])
    
    X_text = df['customer_text'].astype(str)
    y = df['intent']
    
    print(f"Training on {len(X_text)} examples...")
    
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_features=50000,
        sublinear_tf=True
    )
    
    X_vec = vectorizer.fit_transform(X_text)
    
    clf = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    )
    
    print("Fitting Logistic Regression model...")
    clf.fit(X_vec, y)
    
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(clf, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    
    print(f"[OK] Model saved to {MODEL_PATH}")
    print(f"[OK] Vectorizer saved to {VECTORIZER_PATH}")

def load_model():
    if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
        return None, None
    clf = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return clf, vectorizer

def predict_intent(text):
    """
    Reusable function to predict intent from text.
    Handles empty/invalid input safely.
    Returns: (predicted_intent, confidence)
    """
    if not text or not isinstance(text, str) or not text.strip():
        return "other_unknown", 1.0
        
    clf, vectorizer = load_model()
    if clf is None:
        return "other_unknown", 0.0
        
    vec = vectorizer.transform([text])
    pred = clf.predict(vec)[0]
    
    # Get probabilities
    proba = clf.predict_proba(vec)[0]
    confidence = max(proba)
    
    return pred, round(confidence, 4)

def interactive_mode():
    clf, vectorizer = load_model()
    if clf is None:
        print("Model not trained. Training now...")
        train_model()
        
    print("\n" + "="*50)
    print("  AMAZONHELP INTENT CLASSIFIER - INTERACTIVE MODE")
    print("="*50)
    print("Type your message below (or 'quit' to exit).")
    
    while True:
        try:
            user_input = input("\nCustomer Message: ")
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
                
            intent, conf = predict_intent(user_input)
            print(f"Predicted Intent: {intent}")
            print(f"Confidence:       {conf}")
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error processing input: {e}")
            
    print("\nExiting interactive mode.")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--train":
        train_model()
    elif not MODEL_PATH.exists():
        print("Model not found. Running training first...")
        train_model()
        # Don't launch interactive mode by default if we just trained, 
        # unless specifically requested. We'll exit to allow scripts to run.
        print("Training complete. Run without args for interactive mode.")
    else:
        interactive_mode()
