# AmazonHelp AI Support Agent

An AI customer-support agent built from historical AmazonHelp conversations on Twitter.

The system performs three tasks:

1. Classifies an incoming customer message into one of 11 support intents.
2. Retrieves historically similar AmazonHelp conversations and uses them to draft a grounded response.
3. Decides whether the request should be automatically handled or escalated to a human, with a reason.

---

# Quick Start

## Installation

Clone the repository and create a virtual environment.

### Windows

```bash
git lfs install
git clone https://github.com/shreyash729/hiver.git
cd hiver

python -m venv .venv
.venv\Scripts\activate

pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### Linux / macOS

```bash
git lfs install
git clone https://github.com/shreyash729/hiver.git
cd hiver

python3 -m venv .venv
source .venv/bin/activate

pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## API Key

The response-generation, escalation, and response-evaluation components use Groq.

get it from: ```https://console.groq.com/keys```



## Run the Agent

The main interactive demo is:

```bash
python chatbot.py
```

Enter a customer message when prompted.

The agent displays:

- predicted intent and confidence;
- top 5 historical AmazonHelp cases;
- similarity scores;
- generated response;
- AUTO_HANDLE / ESCALATE decision;
- escalation reason;
- LLM judge result.

Enter `-1` to exit.

## Run Evaluation

The repository contains three evaluation scripts.

### 1. Intent Classification

```bash
python evaluation/evaluate_classifier.py
```

Evaluates the intent classifier and reports the classification metrics and baselines.

### 2. Historical Retrieval

```bash
python evaluation/evaluate_retrieval.py
```

Evaluates retrieval using intent-consistency:

```text
Hit@1
Hit@3
Hit@5
Hit@10
```

The script uses the cached classifier and historical embeddings when available.

### 3. Response Generation + LLM Judge

```bash
python evaluation/evaluate_responses.py
```

This requires `GROQ_API_KEY`.

It generates responses for held-out examples, evaluates them with the LLM judge, and saves the generated evaluation results to:

```text
results/response_evaluation_generated.csv
```

## Precomputed Artifacts

The repository includes:

```text
artifacts/intent_model.joblib
artifacts/historical_embeddings.npy
```

These cached artifacts allow the interactive agent, retrieval evaluation, and response evaluation to avoid regenerating the full set of 116,979 historical embeddings.

The classification evaluator intentionally retrains the classifier so that the reported classification benchmark can be reproduced independently.

---

## 1. Problem

The goal of this project is to build an AI support agent for one brand from the Customer Support on Twitter dataset.

The agent should:

- understand the customer's intent;
- use the brand's historical support behavior to generate a response;
- avoid inventing unsupported policies or actions;
- recognize when the issue requires customer/account-specific investigation;
- escalate such cases instead of pretending they can be resolved automatically.

I selected **AmazonHelp** as the target brand.

The project is intended as a support-assistance and triage prototype, not as a fully autonomous Amazon customer-service system.

---

# 2. Dataset

## Source

The project uses:

**Customer Support on Twitter**

Dataset:
`thoughtvector/customer-support-on-twitter`

The dataset contains approximately 3 million tweets across multiple brands and contains real, noisy Twitter customer-support conversations.

The original dataset contains fields including:

- `tweet_id`
- `author_id`
- `inbound`
- `created_at`
- `text`
- `response_tweet_id`
- `in_response_to_tweet_id`

---

# 3. Brand Selection

The dataset contains conversations from multiple companies, so the first step was deciding which brand to build the support agent for.

I compared the two candidate brands considered for this project:

| Brand | Brand tweets | Tweets with `response_tweet_id` | Tweets responding to another tweet |
|---|---:|---:|---:|
| AmazonHelp | 154,011 | 77,386 | 153,514 |
| AppleSupport | 99,735 | 29,269 | 99,599 |

I selected **AmazonHelp** because it provided substantially more response-linked support data, giving the response-generation component a larger historical corpus to retrieve from.

The objective was not simply to choose the largest brand, but to choose a brand with enough historical support responses to make retrieval-based grounding useful.

---

# 4. Constructing Customer → Brand Response Pairs

The raw dataset contains Twitter conversation relationships rather than a ready-made customer-support dataset.

For this project, I extracted direct customer → AmazonHelp response pairs.

Conceptually:

```text
Customer tweet
      |
      | response_tweet_id
      v
AmazonHelp response
```

For every usable customer tweet, the corresponding AmazonHelp response was collected.

The resulting pair structure was:

```text
customer_tweet_id
customer_id
customer_created_at
customer_message
amazon_tweet_id
amazon_created_at
amazon_response
language
```

This makes each row a historical example of:

```text
Customer problem → Actual AmazonHelp response
```

This structure is useful for both:

- intent modelling;
- retrieval-based response generation.

---

# 5. Data Cleaning and Preprocessing

The original Twitter data is noisy and contains information that is not useful for semantic intent classification.

The preprocessing pipeline was therefore:

```text
Raw Twitter dataset
       |
       v
Select AmazonHelp
       |
       v
Construct customer → AmazonHelp pairs
       |
       v
Remove unnecessary original columns
       |
       v
Clean customer messages
       |
       v
Clean Amazon responses
       |
       v
Filter to English conversations
       |
       v
Remove unusable rows
       |
       v
Remove exact duplicate customer-response pairs
       |
       v
Final retrieval/training dataset
```

---

## 5. Text cleaning

Twitter messages contain a significant amount of noise such as:

- URLs;
- `@mentions`;
- Twitter-specific artifacts;
- punctuation/special characters;
- inconsistent whitespace;
- inconsistent capitalization.

Regular expressions were used to clean the messages.

The cleaning process includes:

1. Removing URLs.
2. Removing Twitter `@mentions`.
3. Removing unwanted Twitter artifacts/special characters.
4. Normalizing text to lowercase.
5. Normalizing whitespace.
6. Removing unusable/empty messages.

For example:

```text
Original:

@AmazonHelp My package is STILL delayed!!! 
https://example.com #amazon

After cleaning:

my package is still delayed
```

The objective of cleaning was not to aggressively rewrite the language, but to remove noise that could interfere with semantic similarity and intent classification.

---

# 6. Language Filtering

The AmazonHelp data contains conversations in multiple languages/scripts.

A language/script analysis of AmazonHelp tweets showed that approximately:

- 94.4% were Latin-script;
- 5.6% contained Japanese/CJK characters.

Because this project focuses on an English-language customer-support agent, I restricted the modelling dataset to English conversations.

This keeps the intent taxonomy and evaluation consistent with the scope of the project.

* ALL EDA STEPS ARE IN NOTEBOOK ``` Notebooks/prepare_amazonhelp_data.ipynb```
* ```collab link: ```https://colab.research.google.com/drive/1dzpz4EoXlFiRx0h31VZa9MgJsNngiP-d?usp=sharing
---

# 7. Final Dataset

After cleaning, filtering and deduplication, the modelling dataset contains:

**117,376 customer-response pairs**

with the following final columns:

```text
cleaned_customer_message
cleaned_amazon_response
Intent
```

The dataset consists of:

- **397 labelled examples** (Golden evaluation set)
- **116,979 historical examples**

The 397 labelled examples were manually assigned to support intents.

The 116,979 remaining historical conversations are used as the retrieval corpus.

Importantly, I did not deduplicate only on customer message text.

The same customer message can legitimately receive different responses depending on context, so duplicate handling was performed on the customer-response pair rather than simply deleting repeated customer messages.

---

# 8. Intent Taxonomy

I created an 11-class intent taxonomy based on recurring support patterns observed in the AmazonHelp data.

| Intent | Description |
|---|---|
| `delivery_issue` | Package is late, missing, tracking is delayed/inaccurate, or customer asks where the package is |
| `delivery_courier` | Problem specifically involving the driver, courier, carrier, or delivery behavior |
| `order_change_cancel` | Customer wants to cancel, modify, change, or reorder an existing order |
| `item_problem` | Item is damaged, defective, wrong, missing, or otherwise has a product-specific problem |
| `refund_payment` | Refunds, charges, payment problems, payment methods, Amazon Pay, or money-related issues |
| `product_information` | Product specifications, features, price, availability, listing, or warranty questions |
| `prime` | Prime membership, subscription, benefits, or Prime-specific services |
| `technical` | Website, app, Kindle, Prime Video, login, streaming, or software/service problems |
| `Alexa` | Alexa/Echo/voice-assistant related issues |
| `support_followup` | Existing support case, callback, email, update, or unresolved support interaction |
| `other` | Messages that do not reasonably fit another category |

---

# 9. Train/Test Split

The 397 labelled examples were divided using an 80/20 stratified split.

```text
Training examples: 317
Test examples:      80
```

Stratification was used so that the distribution of the 11 intents was preserved as much as possible between the training and test sets.

The test set was kept separate from model fitting.

---

# 10. Intent Classification

Several approaches were considered.

## 10.1 Baseline 1 — Majority Class

The simplest possible baseline always predicts the most frequent class.

The most frequent intent in the labelled data is:

```text
delivery_issue
```

This gives:

**22.5% accuracy**

---

## 10.2 Baseline 2 — TF-IDF + Logistic Regression

The second baseline uses:

```text
TF-IDF
   |
   v
1-gram + 2-gram features
   |
   v
Logistic Regression
```

This achieves:

**42.5% accuracy**

---

## 10.3 Final Classifier — BGE + Logistic Regression

The final classifier uses:

```text
Customer message
       |
       v
BGE embedding
       |
       v
Logistic Regression
       |
       v
Predicted intent
```

The embedding model used is:

```text
BAAI/bge-base-en-v1.5
```

The classifier is:

```text
LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)
```

The final classifier achieves:

**71.25% accuracy**

with:

```text
Macro F1:    0.634
Weighted F1: 0.722
```

---

# 11. Classification Results

| Model | Accuracy |
|---|---:|
| Majority-class baseline | 22.5% |
| TF-IDF + Logistic Regression | 40.0% |
| **BGE + Logistic Regression** | **71.25%** |

The BGE classifier improves over:

- the majority baseline by **48.75 percentage points**;
- the TF-IDF baseline by **28.75 percentage points**.

---

# 12. Historical Retrieval

After predicting the intent, the system retrieves historically similar AmazonHelp conversations.

The retrieval pipeline is:

```text
Incoming customer message
          |
          v
BGE embedding
          |
          v
Cosine similarity against
116,979 historical messages
          |
          v
Top-K historical cases
```

The historical customer messages are embedded using:

```text
BAAI/bge-base-en-v1.5
```

The embeddings are normalized, so their dot product corresponds to cosine similarity.

The top matching historical conversations are then supplied to the response-generation model.

---

# 13. Retrieval Evaluation

Retrieval was evaluated on the held-out test examples.

The retrieval metric checks whether the retrieved historical cases contain examples whose intent is consistent with the correct test intent.

Results using BGE cosine retrieval:

| Metric | Result |
|---|---:|
| Hit@1 | 65.0% |
| Hit@3 | 77.5% |
| Hit@5 | 82.5% |
| Hit@10 | 90.0% |

### Important interpretation

These numbers should not be interpreted as human-judged semantic relevance.

The historical corpus contains model-predicted intents, so this experiment measures **intent consistency of retrieved cases**.

---

# 14. Reranker Experiment

I also tested a cross-encoder reranker:

```text
cross-encoder/ms-marco-MiniLM-L-6-v2
```

Results:

| Metric | BGE | BGE + Reranker |
|---|---:|---:|
| Hit@1 | 65.0% | 60.0% |
| Hit@3 | 77.5% | 78.75% |
| Hit@5 | 82.5% | 83.75% |
| Hit@10 | 90.0% | 90.0% |

The reranker slightly improved Hit@3 and Hit@5 but reduced Hit@1 and did not improve Hit@10.

Therefore the final system uses the simpler:

```text
BGE + cosine similarity
```

without a reranker.

---

# 15. Response Generation

The response generator uses a Groq-hosted LLM through LangChain.

The model used during development was:

```text
qwen/qwen3.8-27b
```

The generation pipeline is:

```text
Customer message
       |
       v
BGE retrieval
       |
       v
Top historical AmazonHelp responses
       |
       v
LLM
       |
       v
Customer-facing response
```

The LLM is explicitly instructed to:

- use historical Amazon responses as evidence;
- avoid inventing policies;
- avoid inventing refunds or timelines;
- avoid making guarantees;
- avoid claiming to have checked an account/order;
- ask the customer to contact Amazon support when the evidence is insufficient;
- remain concise and professional.

For example:

```text
Customer:
hey i havent recive my pacakge

Predicted intent:
delivery_issue

Generated response:
I am sorry to hear that you have not received your package.
To help us look into this, please share your order details here
so we can check the status and assist you further.
```

---

# 16. Escalation Decision

The system separately decides whether the issue can be safely handled automatically.

The escalation component receives:

```text
Customer message
Predicted intent
Retrieved historical support evidence
```

It returns:

```json
{
  "decision": "AUTO_HANDLE",
  "reason": "..."
}
```

or:

```json
{
  "decision": "ESCALATE",
  "reason": "..."
}
```

The system is instructed to escalate when resolving the issue requires:

- customer-specific order investigation;
- account access;
- payment-specific investigation;
- information not present in the conversation;
- human intervention.

For example, for:

```text
hey i havent recive my pacakge
```

the system returned:

```json
{
  "decision": "ESCALATE",
  "reason": "The customer reports a missing package, which requires checking specific order status and tracking information. The historical evidence shows that agents needed to access account details or ask for specific identifiers to investigate, indicating that a generic response without account access is insufficient."
}
```

This is deliberately conservative: the system should not claim that an issue has been resolved when it does not have access to the customer's order/account.

---

# 17. Complete Agent Pipeline

The final system works as follows:

```text
                  Customer Message
                         |
                         v
                BGE Sentence Embedding
                         |
             +-----------+-----------+
             |                       |
             v                       v
       Intent Classifier        Similarity Search
       BGE + Logistic          BGE + Cosine
        Regression                  |
             |                      |
             v                      v
       Predicted Intent        Top-5 Historical
                              AmazonHelp Cases
                                     |
                         +-----------+-----------+
                         |                       |
                         v                       v
                 Response Generator       Escalation Agent
                      Groq LLM                 Groq LLM
                         |                       |
                         v                       v
                  Draft Response         Decision + Reason
                         |
                         v
                    LLM Judge
```

---

# 18. Response Evaluation

An LLM-as-judge was used to evaluate generated responses.

Each response was scored from 1–5 on:

### 1. Resolution relevance

Does the response address the customer's actual problem?

### 2. Grounding

Are the claims and guidance supported by the historical support evidence?

### 3. Helpfulness

Does the response provide a useful and appropriate next step?

### 4. Unsupported claims

Does the response avoid unsupported claims, invented policies, guarantees, or pretending to access the customer's account/order?

A higher score means the response is safer with respect to unsupported claims.

### 5. Overall

Overall quality of the support response.

---

# 19. LLM Judge Results

The response generation pipeline was evaluated on 10 held-out examples.

Average scores:

| Criterion | Average |
|---|---:|
| Resolution relevance | 4.0 / 5 |
| Grounding | 4.7 / 5 |
| Helpfulness | 3.8 / 5 |
| Unsupported claims | 5.0 / 5 |
| Overall | 4.0 / 5 |

The strongest result was grounding and avoidance of unsupported claims.

---

# 20. Human vs LLM Judge Agreement

To validate the automated judge, the same 10 responses were independently evaluated by a human using the overall 1–5 scale.

Results:

```text
Exact agreement: 90.0%

Spearman correlation: 0.8845

p-value: 0.0007
```

This provides initial evidence that the automated judge is aligned with human evaluation.

However, the sample contains only 10 examples, so this should not be interpreted as a definitive reliability study.

One disagreement was particularly useful: the judge gave a technical-support response a 5, while the human evaluator gave it a 4 because the response stated that Amazon was "currently aware" of technical problems, which was not clearly supported by the retrieved evidence.

This demonstrates why human spot-checking remains useful even when automated judge agreement is high.

---

# 21. Failure Analysis

The final classifier misclassified:

```text
23 / 80
```

test examples.

The main failure modes were:

## Failure Mode 1 — Vague complaints

Examples:

```text
your customer services are appalling
```

and:

```text
do something fast as your service is really bad
```

These messages do not provide a concrete product, order, payment, delivery, or support action.

### Hypothesis

When the message contains little information about the underlying problem, the embedding representation cannot reliably distinguish `other` from `support_followup` or another specific intent.

---

## Failure Mode 2 — Delivery vs courier overlap

Example:

```text
delivery prsn refused to do delivery at floor ...
```

The correct intent was:

```text
delivery_courier
```

but the classifier predicted:

```text
order_change_cancel
```

Another example involved a customer being required to pick up a package without receiving a notice or pickup location.

### Hypothesis

`delivery_issue` and `delivery_courier` contain highly overlapping vocabulary.

The distinction depends on whether the problem concerns:

```text
package status
```

or:

```text
driver/courier behavior
```

which is not always obvious from a short tweet.

---

## Failure Mode 3 — Keyword-driven predictions

Example:

```text
so that s 3 late prime deliveries in the last few weeks
what do i pay my prime membership for again
```

The correct intent was:

```text
delivery_issue
```

but the model predicted:

```text
prime
```

### Hypothesis

The distinctive keyword `prime` can dominate the semantic representation even though the customer's actual problem is repeated late delivery.

The same issue can happen with strongly associated entities such as Alexa.

---

## Failure Mode 4 — Mixed-intent messages

Example:

```text
i have been asking for refund as i cancelled it you are talking
about product delivery delay
```

The correct intent was:

```text
refund_payment
```

but the model predicted:

```text
delivery_issue
```

Another example combined returning an item with the fact that the item had never arrived.

### Hypothesis

Real customer-support messages frequently contain multiple issues.

The current taxonomy forces every message into exactly one intent, so the definition of the "primary" intent can become ambiguous.

---

## Failure Mode 5 — Closely related transactional intents

Errors also occurred between:

```text
product_information
item_problem
refund_payment
order_change_cancel
```

These categories naturally overlap in e-commerce conversations.

For example, customers can simultaneously discuss:

```text
product
    +
order
    +
exchange
    +
payment
```

### Hypothesis

The current single-label taxonomy does not always represent the multi-intent nature of real customer-support requests.

---

# 22. What Is Misleading About My Headline Number?

The headline classification result is:

# 71.25% accuracy

This number is useful, but it should not be interpreted as:

> "The AI support agent correctly handles 71.25% of customers."

The number measures only **intent classification accuracy**.

It does not measure:

- whether the generated response is correct;
- whether retrieval is actually useful to a human;
- whether the escalation decision is correct;
- whether the customer would be satisfied.

There are additional limitations.

### Small test set

The classifier was evaluated on only:

```text
80 examples
```

so the measured accuracy is sensitive to individual examples.

### Class imbalance

The number of examples differs considerably between intents.

For example:

```text
delivery_issue:        18 test examples
other:                 12
support_followup:      11
delivery_courier:       8
refund_payment:         8
technical:              7
...
Alexa:                  1
```

Therefore, the result does not imply equal performance across all intents.

### Rare-intent performance

The model achieved:

```text
Alexa:
precision = 0
recall    = 0
F1        = 0
```

but the class had only one test example.

Similarly, `order_change_cancel` had only two test examples.

### Retrieval metric limitation

The retrieval Hit@K experiment measures intent consistency of retrieved cases rather than human-judged relevance.

### Response evaluation limitation

Only 10 generated responses were evaluated using the LLM judge and human comparison.

Therefore:

> 71.25% is a headline number for the intent classifier, not a reliability claim for the entire support agent.

---

# 23. What I Would Do With One More Week

## 1. Build a larger and cleaner golden evaluation set

Increase the number of independently labelled examples and keep evaluation data completely separated from model development.

For ambiguous examples, use multiple annotators and measure inter-annotator agreement.

## 2. Improve the taxonomy

The current failure modes suggest that some boundaries need refinement.

For example:

```text
Delivery
├── Delivery status/problem
└── Courier/driver problem

Order
├── Change/cancel
├── Refund/payment
└── Item problem
```

A hierarchical classifier could make these distinctions easier.

## 3. Improve retrieval evaluation

Instead of evaluating retrieval primarily through intent consistency, manually evaluate whether the retrieved historical response is actually useful for answering the new customer message.

## 4. Build a dedicated escalation evaluation set

Human-label examples as:

```text
AUTO_HANDLE
ESCALATE
```

and evaluate the escalation component independently.

I would pay particular attention to the false-auto-handle rate because incorrectly auto-handling a case that requires account investigation is potentially more problematic than unnecessarily escalating it.

## 5. Evaluate responses at larger scale

Run response generation and human/LLM evaluation over a larger held-out set, especially for examples where retrieval similarity is low or the intent classifier is uncertain.

---

# 24. Key Design Decisions

1. **AmazonHelp was selected** because it had substantially more response-linked support data than AppleSupport.

2. **Direct customer → AmazonHelp response pairs were used** instead of modelling the complete Twitter thread.

3. **Twitter-specific noise was removed** before semantic modelling.

4. **The project was restricted to English-language support** to keep the scope consistent.

5. **Customer messages were not deduplicated solely by text**, because the same message can legitimately receive different responses.

6. **The intent taxonomy was created from observed AmazonHelp support patterns** rather than importing a generic taxonomy.

7. **A stratified train/test split was used** to preserve intent distribution.

8. **BGE and MiniLM were compared** rather than assuming a particular embedding model was best.

9. **BGE + Logistic Regression was selected** because it achieved the strongest classification result.

10. **The full historical corpus was used for retrieval** rather than only the labelled examples.

11. **A cross-encoder reranker was tested and rejected** because it reduced Hit@1 without improving Hit@10.

12. **Low-confidence pseudo-labels were not used for retraining** because the confidence distribution on the historical corpus was weak.

13. **LLM-based escalation was used** instead of building another supervised classifier because the escalation decision depends on whether available evidence is sufficient for safe resolution.

14. **Escalation returns both a decision and a reason** so the system does not produce an unexplained routing decision.

15. **The LLM judge was compared against human scores** rather than relying only on automated evaluation.

---


# 30. Limitations

This project is a prototype rather than a production customer-support system.

The main limitations are:

- relatively small labelled evaluation/development set.
- very small test support for some intents.
- single-label intent classification for inherently multi-intent customer messages.
- retrieval evaluation based primarily on intent consistency rather than human relevance.
- small response-evaluation sample.
- no direct access to Amazon customer accounts or orders.
- escalation evaluation is not yet backed by a separately human-labelled escalation benchmark.

These limitations are intentionally reported because the objective is to understand where the system can and cannot be trusted.

---

# 31. Conclusion

The final system combines supervised intent classification, semantic retrieval, grounded response generation, and conservative escalation.

The most important result is that BGE + Logistic Regression substantially outperforms both the majority-class and TF-IDF baselines:

```text
22.5%  →  42.5%  →  71.25%
Majority   TF-IDF     BGE + LR
```

The retrieval layer provides historically similar AmazonHelp examples to ground response generation, while the escalation component prevents the system from pretending to resolve issues that require customer-specific account or order information.

The main remaining weaknesses are rare intents, ambiguous/multi-intent messages, and the limited size of the response/evaluation benchmark.

The next iteration should therefore focus less on adding model complexity and more on improving the evaluation set, refining intent boundaries, and independently validating escalation quality.

---

# 32. Source / Attribution

Primary dataset:

**Customer Support on Twitter — `thoughtvector/customer-support-on-twitter`**

Models/libraries used:

- `BAAI/bge-base-en-v1.5` — sentence embeddings
- `sentence-transformers` — embedding inference
- `scikit-learn` — Logistic Regression, TF-IDF and evaluation
- `cross-encoder/ms-marco-MiniLM-L-6-v2` — reranker experiment
- Groq / `qwen/qwen3.8-27b` — response generation, escalation, and LLM evaluation
- LangChain — LLM integration

No external intent taxonomy was used. The 11 support intents were defined from the AmazonHelp data used in this project.
