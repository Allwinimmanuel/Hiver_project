# Milestone 4 Data Inspection Report

## Cleaned Dataset (`data/processed/amazonhelp_clean.csv`)
- **Rows**: 168785
- **Columns**: conversation_id, turn_index, customer_tweet_id, customer_author_id, customer_text, customer_created_at, support_tweet_id, support_text, support_created_at, support_text_len, is_long_response, conversation_turns
- **Missing Values**:
  - None
- **Intent Labels Exist?**: No

## Golden Set (`data/gold/amazonhelp_golden_set.csv`)
- **Rows**: 250
- **Columns**: example_id, customer_tweet_id, customer_text, intent, labeling_notes, source_conversation_id, difficulty, text_len
- **Missing Values**:
  - None

## Recommended Training Strategy
Since supervised intent labels do not exist in the cleaned dataset, we must employ a **weak-supervision labeling pipeline**. We will write deterministic keyword and phrase-based rules to assign intents (and confidence scores) to the `customer_text` in `amazonhelp_clean.csv`. This weakly-labeled data will then serve as the training set for our Logistic Regression classifier.