# Data

This project uses historical customer-support conversations from the
Customer Support on Twitter dataset.

## Selected brand

AmazonHelp.

## Dataset used by the agent

`amazonhelp_cleaned_with_intents.csv`

The file contains:

- `cleaned_customer_message`
- `cleaned_amazon_response`
- `Intent`

The dataset contains:

- 397 manually labelled examples used for intent classification
- 116,979 historical customer-response cases used for retrieval
- 117,376 rows in total

## Intent labels

The manually labelled examples use the following taxonomy:

- `delivery_issue`
- `delivery_courier`
- `order_change_cancel`
- `item_problem`
- `refund_payment`
- `product_information`
- `prime`
- `technical`
- `Alexa`
- `support_followup`
- `other`

## Data preprocessing

Customer messages and AmazonHelp responses were cleaned by:

1. decoding HTML entities
2. removing URLs
3. removing Twitter mentions
4. removing AmazonHelp agent signatures
5. converting hashtags to normal words
6. removing non-alphanumeric characters
7. normalizing whitespace
8. converting text to lowercase
9. removing unusable empty rows
10. removing exact duplicate customer-response pairs

The project was scoped to English-language support conversations.

## Dataset attribution

The underlying data comes from the
Customer Support on Twitter dataset.

The original dataset should be obtained from its official/source
location if redistribution of the cleaned CSV is not permitted.

Do not commit API keys, credentials, or other secrets to this
repository.