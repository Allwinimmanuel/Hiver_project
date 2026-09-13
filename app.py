import streamlit as st
import joblib
from pathlib import Path
import json
import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


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
    "order_status_tracking": {"icon": "📦", "label": "Order Status / Tracking", "action": "Ask for Order ID. Check tracking portal.", "color": "#2196F3", "priority": "Medium", "department": "Logistics Team"},
    "delivery_issue": {"icon": "🚚", "label": "Delivery Issue", "action": "File logistics case. Offer re-delivery or refund.", "color": "#FF6F00", "priority": "High", "department": "Logistics Escalations"},
    "refund_return": {"icon": "💸", "label": "Refund / Return", "action": "Guide to Returns page. Generate prepaid label.", "color": "#7B1FA2", "priority": "Medium", "department": "Returns & Refunds"},
    "account_billing": {"icon": "💳", "label": "Account / Billing", "action": "Direct to Account Settings. Escalate billing disputes.", "color": "#C62828", "priority": "High", "department": "Billing & Security"},
    "prime_membership": {"icon": "⭐", "label": "Prime Membership", "action": "Provide Prime management links. Troubleshoot benefits.", "color": "#00838F", "priority": "Medium", "department": "Prime Subscriptions"},
    "product_defect": {"icon": "🔧", "label": "Product Defect", "action": "Request photos. Initiate replacement or refund.", "color": "#4E342E", "priority": "Medium", "department": "Product Quality"},
    "customer_service_escalation": {"icon": "🆘", "label": "Escalation Request", "action": "Transfer to Senior Support Agent immediately.", "color": "#AD1457", "priority": "Urgent", "department": "Senior Escalations"},
    "other_unknown": {"icon": "❓", "label": "General / Unknown", "action": "Request more context. Route to human agent.", "color": "#546E7A", "priority": "Low", "department": "General Support"}
}

# ─── Historical Evidence Base ────────────────────────────────────────────────
EVIDENCE_DB = {
    "order_status_tracking": "Historical Agent Resolution: 'Your order is currently processing and will ship soon. You can track its progress in Your Orders.'",
    "delivery_issue": "Historical Agent Resolution: 'I am sorry to hear your package was not delivered correctly. We will investigate with the carrier and issue a replacement if it is lost.'",
    "refund_return": "Historical Agent Resolution: 'You can easily return this item. I have generated a prepaid shipping label for you. Your refund will process within 3-5 business days after we receive the item.'",
    "account_billing": "Historical Agent Resolution: 'I have reviewed your account billing details and removed the unauthorized charge. You should see a credit on your statement shortly.'",
    "prime_membership": "Historical Agent Resolution: 'I can help you manage your Prime membership. I have cancelled the auto-renewal as requested.'",
    "product_defect": "Historical Agent Resolution: 'I apologize that the item arrived defective. We can issue a full refund or send a free replacement immediately.'",
    "customer_service_escalation": "Historical Agent Resolution: 'I understand your frustration. I am escalating this ticket immediately to our senior support team for priority resolution.'",
    "other_unknown": "Historical Agent Resolution: 'Thank you for reaching out. Please provide more details about your issue so we can assist you better.'"
}

# ─── Reply Generation ────────────────────────────────────────────────────────
def generate_reply(intent, evidence, customer_message):
    provider = os.getenv("LLM_PROVIDER", "").lower()
    api_key = os.getenv("OPENAI_API_KEY", "")
    
    fallback_templates = {
        "order_status_tracking": "We can help you track your order. Please check the order tracking page for the latest updates.",
        "delivery_issue": "I'm sorry to hear about the delivery problem. We recommend reviewing the order and delivery details. If the issue requires further investigation, our support team can assist.",
        "refund_return": "I'm sorry your product arrived damaged or you wish to return it. We can help review your refund or return request. Please provide your order details so the appropriate next step can be confirmed.",
        "account_billing": "I'm sorry you're having trouble accessing your account or have a billing issue. Please try the account recovery options. If the issue continues, a support representative can help.",
        "prime_membership": "We can help manage your Prime membership. Please visit your account settings to review your subscription.",
        "product_defect": "I'm sorry the item is defective. Please initiate a return or replacement from your orders page.",
        "customer_service_escalation": "I understand that you would like to speak with a support representative. I'm routing this request for human review.",
        "other_unknown": "I'm happy to help. Could you provide a few more details about your issue so we can route it appropriately?"
    }
    
    fallback_reply = fallback_templates.get(intent, fallback_templates["other_unknown"])
    
    if not OPENAI_AVAILABLE or not api_key or provider != "openai":
        return fallback_reply, "Template fallback (LLM provider not configured)"
        
    try:
        client = openai.OpenAI(api_key=api_key)
        prompt = f"""You are a professional Amazon customer support agent.
Generate a concise, helpful reply to the customer message below.
Your response MUST be grounded in the following historical resolution evidence. Do not invent policies, dates, or personal information.

Customer Message: {customer_message}
Historical Evidence: {evidence}
"""
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=150,
            timeout=10
        )
        return response.choices[0].message.content.strip(), "LLM Generated"
    except Exception as e:
        # Suppress scary 401/429 errors from the UI to avoid confusing the user
        return fallback_reply, "Template fallback (LLM generation unavailable)"

# ─── App UI ──────────────────────────────────────────────────────────────────
st.markdown("<div class='hero-header'><h1>🚀 AmazonHelp AI Support Agent</h1><p style='font-size: 1.2em; color: #e0e0e0;'>Automatically classifying customer support messages to speed up resolution.</p></div>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["💬 Live Agent", "📊 Model Analytics", "⚙️ API Demo"])

# --- TAB 1: Live Agent ---
# ─── Prediction Logic ──────────────────────────────────────────────────────────
def predict_intent(text, clf, vectorizer):
    vec = vectorizer.transform([text])
    intent = clf.predict(vec)[0]
    confidence = float(max(clf.predict_proba(vec)[0]))
    
    text_lower = text.lower().strip()
    
    # Rule-Based Overrides
    override_reason = ""
    if any(p in text_lower for p in ["stolen", "missing", "lost"]):
        intent = "delivery_issue"
    elif any(p in text_lower for p in ["real person", "human", "agent"]):
        intent = "customer_service_escalation"
        override_reason = "Customer explicitly requested a human."
    elif "access my account" in text_lower:
        intent = "account_billing"
        override_reason = "Sensitive account/billing issue."
    elif text_lower in ["thank you", "thank you.", "thanks", "hello", "hi"]:
        intent = "other_unknown"
        
    needs_escalation = False
    escalation_reason = "No escalation required."
    
    # Determine routing status
    if confidence < 0.60:
        needs_escalation = True
        escalation_reason = f"Low intent confidence: {confidence*100:.0f}%"
    elif intent == "customer_service_escalation":
        needs_escalation = True
        escalation_reason = override_reason if override_reason else "Escalation intent predicted."
    elif intent == "other_unknown":
        needs_escalation = True
        escalation_reason = "Unsupported or ambiguous customer request."
    elif override_reason:
        if "Sensitive" in override_reason:
            needs_escalation = True
            escalation_reason = override_reason
    
    return intent, confidence, needs_escalation, escalation_reason

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
            intent, confidence, needs_escalation, escalation_reason = predict_intent(user_input, clf, vectorizer)
            info = INTENT_ACTIONS.get(intent, INTENT_ACTIONS["other_unknown"])
            
            st.divider()
            
            # Row 1: Intent, Priority, Assigned Team
            m1, m2, m3 = st.columns(3)
            
            # Predict Intent
            m1.markdown(f"<div class='metric-card' style='border-top-color:{info['color']}'><div class='metric-label'>Predicted Intent</div><div class='metric-value' style='color:{info['color']}; font-size:1.4em'>{info['icon']} {info['label']}</div></div>", unsafe_allow_html=True)
            
            # Priority
            p_color = {"Low": "#28a745", "Medium": "#ffc107", "High": "#fd7e14", "Urgent": "#dc3545"}.get(info['priority'], "#6c757d")
            m2.markdown(f"<div class='metric-card' style='border-top-color:{p_color}'><div class='metric-label'>Priority</div><div class='metric-value' style='color:{p_color}; font-size:1.4em'>{info['priority']}</div></div>", unsafe_allow_html=True)
            
            # Department
            m3.markdown(f"<div class='metric-card' style='border-top-color:#17a2b8'><div class='metric-label'>Assigned Team</div><div class='metric-value' style='font-size:1.4em'>{info['department']}</div></div>", unsafe_allow_html=True)
            
            # Row 2: Confidence & Routing
            m4, m5 = st.columns(2)
            
            # Confidence
            conf_color = "#dc3545" if confidence < 0.6 else "#28a745"
            m4.markdown(f"<div class='metric-card' style='border-top-color:{conf_color}'><div class='metric-label'>Confidence</div><div class='metric-value'>{confidence*100:.1f}%</div></div>", unsafe_allow_html=True)
            
            # Routing Status
            routing = 'Escalated' if needs_escalation else 'Automated'
            route_color = "#dc3545" if needs_escalation else "#28a745"
            m5.markdown(f"<div class='metric-card' style='border-top-color:{route_color}'><div class='metric-label'>Routing Status</div><div class='metric-value' style='font-size:1.4em'>{routing}</div></div>", unsafe_allow_html=True)
            
            if needs_escalation:
                st.warning(f"⚠️ **Human Review Required:** {escalation_reason}")
            else:
                st.markdown(f"<div class='result-card' style='border-left-color:{info['color']}'><b>✅ Suggested Support Action:</b><br>{info['action']}</div>", unsafe_allow_html=True)
                
            # Reply Generation
            st.markdown("### 📝 Grounded Reply Generation")
            evidence = EVIDENCE_DB.get(intent, EVIDENCE_DB["other_unknown"])
            reply, mode = generate_reply(intent, evidence, user_input)
            
            st.info(f"**🔍 Historical Evidence Used:** {evidence}")
            st.success(f"**✉️ Draft Customer Reply:**\n\n{reply}")
            st.caption(f"Reply Generation Mode: {mode}")
                
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
            
        st.info("ℹ️ **Note:** Offline model metrics evaluate *intent classification*. Reply-quality scores evaluate *generated responses* separately (see LLM Judge reports).")

# --- TAB 3: API Demo ---
with tab3:
    st.markdown("### Backend API Payload")
    st.markdown("This shows the JSON payload the ML model would send to the backend routing microservices.")
    if user_input and clf:
        intent, confidence, needs_escalation, escalation_reason = predict_intent(user_input, clf, vectorizer)
        evidence = EVIDENCE_DB.get(intent, EVIDENCE_DB["other_unknown"])
        reply, mode = generate_reply(intent, evidence, user_input)
        
        payload = {
            "customer_message": user_input,
            "predictions": {
                "intent": intent,
                "confidence": confidence,
                "priority": INTENT_ACTIONS.get(intent, INTENT_ACTIONS["other_unknown"])["priority"],
                "needs_human_escalation": needs_escalation,
                "escalation_reason": escalation_reason
            },
            "routing": {
                "assigned_department": INTENT_ACTIONS.get(intent, INTENT_ACTIONS["other_unknown"])["department"]
            },
            "suggested_action": INTENT_ACTIONS.get(intent, INTENT_ACTIONS["other_unknown"])["action"],
            "reply_generation": {
                "draft_reply": reply,
                "generation_mode": mode,
                "evidence_used": evidence
            }
        }
        st.json(payload)
    else:
        st.info("Enter a message in the Live Agent tab to see the payload.")
