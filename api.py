from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
from pathlib import Path
import os
import openai
from dotenv import load_dotenv

load_dotenv()
OPENAI_AVAILABLE = True
try:
    import openai
except ImportError:
    OPENAI_AVAILABLE = False

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

def predict_intent(text, clf, vectorizer):
    vec = vectorizer.transform([text])
    intent = clf.predict(vec)[0]
    confidence = float(max(clf.predict_proba(vec)[0]))
    
    text_lower = text.lower().strip()
    
    # Heuristic overrides
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
        # Some overrides like account_billing might not explicitly force escalation unless we want them to,
        # but the prompt asks to escalate sensitive account issues.
        # Let's escalate them if the reason is set.
        if "Sensitive" in override_reason:
            needs_escalation = True
            escalation_reason = override_reason
            
    return intent, confidence, needs_escalation, escalation_reason

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
        prompt = f"You are a professional Amazon customer support agent.\nGenerate a concise, helpful reply to the customer message below.\nYour response MUST be grounded in the following historical resolution evidence. Do not invent policies, dates, or personal information.\n\nCustomer Message: {customer_message}\nHistorical Evidence: {evidence}\n"
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

app = FastAPI(title="AmazonHelp AI Support API", version="1.0")

MODEL_PATH = Path("models/intent_classifier.joblib")
VECTORIZER_PATH = Path("models/tfidf_vectorizer.joblib")

# Load models globally
try:
    clf = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
except Exception as e:
    clf = None
    vectorizer = None
    print(f"Warning: Could not load models. {e}")

class TicketRequest(BaseModel):
    customer_message: str

class PredictionResponse(BaseModel):
    intent: str
    confidence: float
    priority: str
    needs_human_escalation: bool
    escalation_reason: str

class RoutingResponse(BaseModel):
    assigned_department: str

class ReplyResponse(BaseModel):
    draft_reply: str
    generation_mode: str
    evidence_used: str

class APIResponse(BaseModel):
    customer_message: str
    predictions: PredictionResponse
    routing: RoutingResponse
    suggested_action: str
    reply_generation: ReplyResponse

@app.post("/predict", response_model=APIResponse)
async def predict(request: TicketRequest):
    if not clf or not vectorizer:
        raise HTTPException(status_code=500, detail="Model not loaded.")
        
    text = request.customer_message
    if not text.strip():
        raise HTTPException(status_code=400, detail="Customer message cannot be empty.")
        
    # Reuse prediction logic from app.py
    intent, confidence, needs_escalation, escalation_reason = predict_intent(text, clf, vectorizer)
    info = INTENT_ACTIONS.get(intent, INTENT_ACTIONS["other_unknown"])
    evidence = EVIDENCE_DB.get(intent, EVIDENCE_DB["other_unknown"])
    
    reply, mode = generate_reply(intent, evidence, text)
    
    return APIResponse(
        customer_message=text,
        predictions=PredictionResponse(
            intent=intent,
            confidence=confidence,
            priority=info["priority"],
            needs_human_escalation=needs_escalation,
            escalation_reason=escalation_reason
        ),
        routing=RoutingResponse(
            assigned_department=info["department"]
        ),
        suggested_action=info["action"],
        reply_generation=ReplyResponse(
            draft_reply=reply,
            generation_mode=mode,
            evidence_used=evidence
        )
    )
