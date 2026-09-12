import streamlit as st
import joblib
from pathlib import Path
import json

# ─── Configuration ────────────────────────────────────────────────────────────
st.set_page_config(page_title="AmazonHelp AI Agent", page_icon="🚀", layout="wide")

MODEL_PATH = Path("models/intent_classifier.joblib")
VECTORIZER_PATH = Path("models/tfidf_vectorizer.joblib")
METRICS_PATH = Path("reports/milestone4_metrics.json")
CM_PATH = Path("reports/confusion_matrix.png")

# ─── Custom CSS (Attractive UI) ───────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.hero-header { background: linear-gradient(135deg, #0f2027, #203a43, #2c5364); padding: 30px; border-radius: 12px; margin-bottom: 20px; color: white; text-align: center; }
.hero-header h1 { color: white !important; font-weight: 700; margin-bottom: 5px; }
.hero-header p { color: #e0e0e0 !important; font-size: 1.2em; }
.metric-card { background: #1e1e1e; border-radius: 10px; padding: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); text-align: center; border-top: 4px solid; margin-bottom: 15px;}
.metric-label { font-size: 0.9em; font-weight: 600; color: #aaaaaa; text-transform: uppercase; letter-spacing: 1px; }
.metric-value { font-size: 2em; font-weight: 700; color: white; }
.result-card { background: #2c2c2c; border-radius: 10px; padding: 20px; margin-top: 10px; border-left: 6px solid; color: white; }
.stButton>button { border-radius: 8px; transition: all 0.3s ease; }
.stButton>button:hover { transform: translateY(-2px); box-shadow: 0 4px 12px rgba(255,255,255,0.1); }
</style>
""", unsafe_allow_html=True)

# ─── Load Model ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
        return None, None
    return joblib.load(MODEL_PATH), joblib.load(VECTORIZER_PATH)

clf, vectorizer = load_model()

# ─── Intent Taxonomy & Actions ───────────────────────────────────────────────
INTENT_ACTIONS = {
    "order_status_tracking": {"icon": "📦", "label": "Order Status / Tracking", "action": "Ask for Order ID. Check tracking portal.", "color": "#2196F3"},
    "delivery_issue": {"icon": "🚚", "label": "Delivery Issue", "action": "File logistics case. Offer re-delivery or refund.", "color": "#FF6F00"},
    "refund_return": {"icon": "💸", "label": "Refund / Return", "action": "Guide to Returns page. Generate prepaid label.", "color": "#7B1FA2"},
    "account_billing": {"icon": "💳", "label": "Account / Billing", "action": "Direct to Account Settings. Escalate billing disputes.", "color": "#C62828"},
    "prime_membership": {"icon": "⭐", "label": "Prime Membership", "action": "Provide Prime management links. Troubleshoot benefits.", "color": "#00838F"},
    "product_defect": {"icon": "🔧", "label": "Product Defect", "action": "Request photos. Initiate replacement or refund.", "color": "#4E342E"},
    "customer_service_escalation": {"icon": "🆘", "label": "Escalation Request", "action": "Transfer to Senior Support Agent immediately.", "color": "#AD1457"},
    "other_unknown": {"icon": "❓", "label": "General / Unknown", "action": "Request more context. Route to human agent.", "color": "#546E7A"}
}

# ─── App UI ──────────────────────────────────────────────────────────────────
st.markdown("<div class='hero-header'><h1>🚀 AmazonHelp AI Support Agent</h1><p style='font-size: 1.2em; color: #e0e0e0;'>Automatically classifying customer support messages to speed up resolution.</p></div>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["💬 Live Agent", "📊 Model Analytics", "⚙️ API Demo"])

# --- TAB 1: Live Agent ---
with tab1:
    st.markdown("### Test the AI Classifier")
    
def predict_intent(text, clf, vectorizer):
    vec = vectorizer.transform([text])
    intent = clf.predict(vec)[0]
    confidence = float(max(clf.predict_proba(vec)[0]))
    
    text_lower = text.lower().strip()
    
    # Rule-Based Overrides
    if any(p in text_lower for p in ["stolen", "missing", "lost"]):
        intent = "delivery_issue"
    elif any(p in text_lower for p in ["real person", "human", "agent"]):
        intent = "customer_service_escalation"
    elif "access my account" in text_lower:
        intent = "account_billing"
    elif text_lower in ["thank you", "thank you.", "thanks", "hello", "hi"]:
        intent = "other_unknown"
        
    # Determine routing status
    needs_escalation = (confidence < 0.60) or (intent in ["customer_service_escalation", "other_unknown"])
    
    return intent, confidence, needs_escalation

# --- TAB 1: Live Agent ---
with tab1:
    st.markdown("### Test the AI Classifier")
    
    # Quick Test Buttons using session state
    if "input_text" not in st.session_state:
        st.session_state.input_text = ""
        
    def set_text(text):
        st.session_state.input_text = text

    st.markdown("**Quick Tests:**")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.button("Where is my order?", on_click=set_text, args=("Where is my order?",), use_container_width=True)
        st.button("Cancel my prime membership.", on_click=set_text, args=("Cancel my prime membership.",), use_container_width=True)
    with col2:
        st.button("I want a refund for my broken product.", on_click=set_text, args=("I want a refund for my broken product.",), use_container_width=True)
        st.button("I cannot access my account.", on_click=set_text, args=("I cannot access my account.",), use_container_width=True)
    with col3:
        st.button("My package was stolen.", on_click=set_text, args=("My package was stolen.",), use_container_width=True)
        st.button("Talk to a real person.", on_click=set_text, args=("Talk to a real person.",), use_container_width=True)
    with col4:
        st.button("I received the wrong item.", on_click=set_text, args=("I received the wrong item.",), use_container_width=True)
        st.button("Thank you.", on_click=set_text, args=("Thank you.",), use_container_width=True)
        
    st.write("")
    user_input = st.text_area("Customer Message:", key="input_text", height=100)
    
    if st.button("🔍 Predict Intent", type="primary"):
        if not clf:
            st.error("Model not found. Please train the model first.")
        elif user_input:
            intent, confidence, needs_escalation = predict_intent(user_input, clf, vectorizer)
            info = INTENT_ACTIONS.get(intent, INTENT_ACTIONS["other_unknown"])
            
            st.divider()
            m1, m2, m3 = st.columns(3)
            
            # Predict Intent
            m1.markdown(f"<div class='metric-card' style='border-top-color:{info['color']}'><div class='metric-label'>Predicted Intent</div><div class='metric-value' style='color:{info['color']}; font-size:1.4em'>{info['icon']} {info['label']}</div></div>", unsafe_allow_html=True)
            
            # Confidence
            conf_color = "#dc3545" if confidence < 0.6 else "#28a745"
            m2.markdown(f"<div class='metric-card' style='border-top-color:{conf_color}'><div class='metric-label'>Confidence</div><div class='metric-value'>{confidence*100:.1f}%</div></div>", unsafe_allow_html=True)
            
            # Routing Status
            routing = 'Escalated' if needs_escalation else 'Automated'
            route_color = "#dc3545" if needs_escalation else "#28a745"
            m3.markdown(f"<div class='metric-card' style='border-top-color:{route_color}'><div class='metric-label'>Routing Status</div><div class='metric-value' style='font-size:1.4em'>{routing}</div></div>", unsafe_allow_html=True)
            
            if needs_escalation:
                if intent == "customer_service_escalation":
                    st.warning("⚠️ **Human Review Required:** The customer has explicitly requested a human agent. Escalating immediately.")
                elif intent == "other_unknown":
                    st.warning("⚠️ **Human Review Required:** Intent is unknown or out of scope. Routing to human agent.")
                else:
                    st.warning("⚠️ **Human Review Required:** Low confidence in prediction. Routing to human agent for manual review.")
            else:
                st.markdown(f"<div class='result-card' style='border-left-color:{info['color']}'><b>✅ Suggested Support Action:</b><br>{info['action']}</div>", unsafe_allow_html=True)
                
# --- TAB 2: Analytics ---
with tab2:
    st.markdown("### Evaluation Results (Golden Set)")
    if METRICS_PATH.exists():
        metrics = json.load(open(METRICS_PATH))
        
        c1, c2, c3 = st.columns(3)
        c1.markdown(f"<div class='metric-card' style='border-top-color:#4a90e2'><div class='metric-label'>Golden Set Size</div><div class='metric-value'>250</div></div>", unsafe_allow_html=True)
        c2.markdown(f"<div class='metric-card' style='border-top-color:#50e3c2'><div class='metric-label'>Accuracy</div><div class='metric-value'>{metrics['accuracy']*100:.1f}%</div></div>", unsafe_allow_html=True)
        c3.markdown(f"<div class='metric-card' style='border-top-color:#f5a623'><div class='metric-label'>Macro F1-Score</div><div class='metric-value'>{metrics['macro_avg']['f1-score']*100:.1f}%</div></div>", unsafe_allow_html=True)
        
        st.divider()
        st.markdown("<h3 style='text-align: center;'>Confusion Matrix</h3>", unsafe_allow_html=True)
        if CM_PATH.exists():
            # Use columns to constrain the image width and center it
            left_spacer, img_col, right_spacer = st.columns([1, 3, 1])
            with img_col:
                st.image(str(CM_PATH), use_container_width=True)
        else:
            st.error("Metrics not found.")

# --- TAB 3: API Demo ---
with tab3:
    st.markdown("### Backend API Payload")
    st.markdown("This shows the JSON payload the ML model would send to the backend routing microservices.")
    if user_input and clf:
        intent, confidence, needs_escalation = predict_intent(user_input, clf, vectorizer)
        
        payload = {
            "customer_message": user_input,
            "predictions": {
                "intent": intent,
                "confidence": confidence,
                "needs_human_escalation": needs_escalation
            },
            "suggested_action": INTENT_ACTIONS.get(intent, INTENT_ACTIONS["other_unknown"])["action"]
        }
        st.json(payload)
    else:
        st.info("Enter a message in the Live Agent tab to see the API payload.")
