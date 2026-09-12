# Intent Taxonomy for AmazonHelp

Based on the intent discovery phase (analyzing bigrams, trigrams, and TF-IDF clusters), we have defined **8 core intents** for Amazon customer support messages. This taxonomy is designed to be practical, mutually exclusive where possible, and highly relevant to actual TWCS dataset patterns.

## 1. `order_status_tracking`
- **Description**: Inquiries about the status of an order, when it will ship, or tracking information.
- **Inclusion**: Messages asking "where is my order", "has my item shipped", tracking number questions.
- **Exclusion**: Items that have already been marked as delivered but are missing (use `delivery_issue`).
- **Examples**:
  - *"When is my package going to ship?"*
  - *"Can I get an update on order #12345?"*

## 2. `delivery_issue`
- **Description**: Problems with the actual delivery of a package.
- **Inclusion**: Missing packages, late deliveries, delivered to wrong address, damaged packages.
- **Exclusion**: Items that haven't shipped yet (use `order_status_tracking`).
- **Examples**:
  - *"It says delivered but nothing is at my door."*
  - *"My guaranteed next day delivery is late."*

## 3. `refund_return`
- **Description**: Questions or issues regarding returning an item or getting a refund.
- **Inclusion**: Requests for refund status, how to print a return label, return policy questions.
- **Exclusion**: General complaints about the product without mentioning return/refund.
- **Examples**:
  - *"How long does it take for a refund to process?"*
  - *"I want to return this item, it doesn't fit."*

## 4. `account_billing`
- **Description**: Issues related to the user's Amazon account, login, passwords, or unauthorized charges.
- **Inclusion**: Locked accounts, password resets, unrecognized charges, updating payment methods.
- **Exclusion**: Prime membership questions (use `prime_membership`).
- **Examples**:
  - *"I was charged twice for my last order."*
  - *"I can't log into my account, it says password incorrect."*

## 5. `prime_membership`
- **Description**: Issues or questions specifically about Amazon Prime services.
- **Inclusion**: Prime Video streaming issues, cancelling Prime, Prime subscription costs, Prime Music.
- **Exclusion**: Generic delivery questions that happen to mention the word prime.
- **Examples**:
  - *"Prime video isn't working on my smart TV."*
  - *"How do I cancel my prime membership?"*

## 6. `product_defect`
- **Description**: Complaints about the item itself being broken, defective, or not as described.
- **Inclusion**: Item arrived broken, wrong item sent, defective electronics, missing parts.
- **Exclusion**: Package itself was damaged in transit but item is fine (use `delivery_issue`).
- **Examples**:
  - *"I ordered a red shirt but received a blue one."*
  - *"The screen on the tablet I bought is shattered."*

## 7. `customer_service_escalation`
- **Description**: Messages where the user is primarily demanding to speak to a human or escalating a complaint.
- **Inclusion**: Demands for a phone number, "call me", "talk to a real person", generic angry complaints about service.
- **Exclusion**: Actionable complaints that fit a specific category like `delivery_issue`.
- **Examples**:
  - *"Give me a phone number to call right now."*
  - *"Your customer service is terrible, I want to speak to a manager."*

## 8. `other_unknown`
- **Description**: Messages that do not fit the above categories, or are too vague to classify.
- **Inclusion**: Thank you messages, single words ("yes", "no"), vague complaints ("fix it").
- **Exclusion**: Any message with enough context to fit a specific category.
- **Examples**:
  - *"Thanks!"*
  - *"Yes that worked."*
  - *"Fix your website."*
