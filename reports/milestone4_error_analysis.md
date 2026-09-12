# Milestone 4 Error Analysis

**Total Errors**: 60 / 250

## Top Confusion Pairs
| True Intent | Predicted Intent | Count |
|---|---|---|
| delivery_issue | other_unknown | 17 |
| account_billing | other_unknown | 5 |
| order_status_tracking | other_unknown | 5 |
| product_defect | other_unknown | 4 |
| refund_return | other_unknown | 4 |
| other_unknown | order_status_tracking | 4 |
| customer_service_escalation | other_unknown | 3 |
| other_unknown | prime_membership | 3 |
| other_unknown | delivery_issue | 2 |
| delivery_issue | refund_return | 2 |

## Errors by True Intent (Recall Failures)
- `delivery_issue`: 22
- `other_unknown`: 13
- `order_status_tracking`: 7
- `refund_return`: 5
- `account_billing`: 5
- `product_defect`: 5
- `customer_service_escalation`: 3

## Errors by Predicted Intent (Precision Failures)
- `other_unknown`: 38
- `order_status_tracking`: 5
- `prime_membership`: 5
- `customer_service_escalation`: 4
- `delivery_issue`: 3
- `refund_return`: 3
- `account_billing`: 2

## Specific Problem Areas
- **Over-prediction of `other_unknown`**: The model incorrectly guessed `other_unknown` 38 times. This is likely because the weak supervision training data heavily skewed towards `other_unknown` (making it the default "safe" guess), combined with noisy Twitter language that doesn't strongly match any TF-IDF vocabulary for specific intents.

## Sample Misclassifications
### Example GS-003
- **Customer Text**: @115850 @115821 Courier is showing false status, no delivery attempted, I am waiting in office, no call no attempt tracking 5180735251790
- **True Intent**: `delivery_issue`
- **Predicted Intent**: `order_status_tracking` (Conf: 0.4575)

### Example GS-004
- **Customer Text**: @119625 @58582 Where is Hindi subtitle even Marathi is working for this
- **True Intent**: `other_unknown`
- **Predicted Intent**: `order_status_tracking` (Conf: 0.3671)

### Example GS-010
- **Customer Text**: @115830 - I have a problem with a merchant, claim resolution is not suitable and no return email address provided for additional help.
- **True Intent**: `refund_return`
- **Predicted Intent**: `other_unknown` (Conf: 0.4219)

### Example GS-011
- **Customer Text**: @117634 The new super awesome ipad kindle app has completely screwed up my page view, all text left and down off the bottom of the screen. Thanks.
- **True Intent**: `customer_service_escalation`
- **Predicted Intent**: `other_unknown` (Conf: 0.5739)

### Example GS-013
- **Customer Text**: The worst part is that was my bday present and now all I have are two empty @21428 boxes and NO WATCH. #IAmBeyondShocked @115821 #IWantMyBirthdayPresent !!!
- **True Intent**: `other_unknown`
- **Predicted Intent**: `customer_service_escalation` (Conf: 0.3424)

### Example GS-016
- **Customer Text**: @AmazonHelp And yes, I try those steps every single time but at this point it shouldn't be on ME to try and track down a package your driver consistently misdelivers.
- **True Intent**: `delivery_issue`
- **Predicted Intent**: `other_unknown` (Conf: 0.4989)

### Example GS-039
- **Customer Text**: I got an Amazon account with 250$ in the account, but the account is fucking locked, how do I fix this!!!!!!!!!!!! I can't make any purchases @AmazonHelp
- **True Intent**: `account_billing`
- **Predicted Intent**: `other_unknown` (Conf: 0.4353)

### Example GS-043
- **Customer Text**: @115850 @AmazonHelp SBI 10% Cashback offer is valid for prime membership account if i want to buy MOTO G5 S PLUS.(India)
- **True Intent**: `other_unknown`
- **Predicted Intent**: `prime_membership` (Conf: 0.8678)

### Example GS-058
- **Customer Text**: @AmazonHelp I have accidentally ordered same book twice can I return the same once I receive it??
- **True Intent**: `refund_return`
- **Predicted Intent**: `other_unknown` (Conf: 0.5946)

### Example GS-060
- **Customer Text**: @115850 @AmazonHelp Still no response from any team regarding the issue, just to show on Twitter that they care they reply to tweets here but later didn't bother to even call once.
- **True Intent**: `customer_service_escalation`
- **Predicted Intent**: `other_unknown` (Conf: 0.4772)

### Example GS-061
- **Customer Text**: @AmazonHelp tonight. Now called back and they can’t reach the driver to get him to come to our home!!! DISGUSTING customer service 😡😡😡😡😡😡😡
- **True Intent**: `delivery_issue`
- **Predicted Intent**: `other_unknown` (Conf: 0.4122)

### Example GS-062
- **Customer Text**: @115821 it really shouldn't take a month and a half to get a pack of soothers. Especially when it was a 2 day shipping deal
- **True Intent**: `order_status_tracking`
- **Predicted Intent**: `other_unknown` (Conf: 0.5518)

### Example GS-068
- **Customer Text**: @115821 YOU BLOCKED IT AND I WANT TO GET RID OF IT BCZ OF THE SUBSCRIPTION BUT I CAN’T ! Help me deactivate it !
- **True Intent**: `account_billing`
- **Predicted Intent**: `other_unknown` (Conf: 0.318)

### Example GS-069
- **Customer Text**: @AmazonHelp I asked if Battlefront 2 Deluxe editions where going to be shipped out in time to arrive by the 14th. My order still says delivery date pending even tho it comes out in 2 days.
- **True Intent**: `other_unknown`
- **Predicted Intent**: `order_status_tracking` (Conf: 0.6401)

### Example GS-070
- **Customer Text**: @AmazonHelp hey guys, my delivery said it would arrive before 9pm today and they tried delivering it at 12:43 this afternoon, it wasn’t left with my neighbours and I’m told I have to wait another day, any chance I can get it redelivered today ? Thank you x
- **True Intent**: `delivery_issue`
- **Predicted Intent**: `other_unknown` (Conf: 0.5339)

## Likely Reasons for Failure
- **Weak Supervision Noise**: The model learned the flawed regex rules from the training data rather than true intent.
- **Noisy Twitter Language**: Abbreviations, typos, and fragmented sentences dilute TF-IDF signals.
- **Class Imbalance**: Despite undersampling, the overwhelming majority of Twitter data is `other_unknown` noise.
- **Multilingual Messages**: Non-English texts confuse the English-centric TF-IDF vocabulary.
- **Context Dependence**: Short tweets (e.g. 'Yes thanks') lack the semantic keywords needed by a bag-of-words model.