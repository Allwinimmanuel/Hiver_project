# Labeling Guidelines for AmazonHelp Intents

This document provides instructions for human labelers assigning intents to the Golden Set.

## General Rule
Read the customer message and determine the **primary** reason the customer is reaching out. Select **exactly one** intent from the taxonomy.

## Handling Multiple Issues
If a customer's message contains multiple issues (e.g., "My package is late and I want a refund"), choose the intent that is the **root cause** or the most actionable. In this case, `delivery_issue` is the root cause. If equal, prioritize `refund_return` as it directly impacts financials.

## Handling Vague or Insufficient Context Messages
If a message is too short (e.g., "help me", "fix this") or lacks enough context to confidently assign a specific category, label it as `other_unknown`. Do not guess if there is no evidence.

## Handling PII and Order Numbers
Ignore the presence of order numbers, phone numbers, or personal information when classifying. Classify based on the surrounding text (e.g., "Where is order #1234?" -> `order_status_tracking`).

## Abusive or Emotional Language
Customers may be angry. Ignore the tone unless the *entire message* is just an angry rant demanding a manager with no actionable issue mentioned, in which case use `customer_service_escalation`. If they say "I'm so angry my package is late", use `delivery_issue`.

## Non-English Messages
If the message is not in English but you can clearly understand the intent (e.g., "donde esta mi orden" -> `order_status_tracking`), classify it. If you cannot understand it, use `other_unknown`.

## Fallback Category (`other_unknown`)
Only use `other_unknown` as a last resort. This includes:
- Single-word confirmations ("yes", "no", "thanks")
- Completely unrelated spam
- Extremely vague complaints

## Difficulty Rating
When labeling, also assign a difficulty:
- **Easy**: Clear keywords, perfectly fits one category.
- **Medium**: Slight ambiguity or minor overlap, requires reading carefully.
- **Hard**: Message contains multiple conflicting issues, is very vague, or heavily overlaps two intents.
