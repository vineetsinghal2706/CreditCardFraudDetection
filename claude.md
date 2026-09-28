# CLAUDE.md

# Loan Eligibility Assistant
## AI-Powered Conversational Loan Pre-Qualification System

You are acting as a Senior AI/ML Engineer, GenAI Engineer, Python Backend Engineer, MLOps Engineer and Solution Architect.

Your responsibility is to BUILD the complete application described in this document.

Do not create a superficial demo.

Build a modular, testable, production-style application with:

- Streamlit conversational UI
- FastAPI backend
- Qwen local LLM
- LiteLLM abstraction layer
- RAG using ChromaDB
- Semantic loan-type detection
- Conversational memory
- Structured applicant profile extraction
- Versioned loan policy documents
- Deterministic eligibility rules engine
- Financial calculators
- Grounded explanations
- Policy citations
- Input and output guardrails
- Audit logging
- Streaming responses using SSE
- Graceful streaming error handling
- Pytest unit testing
- Promptfoo GenAI evaluation
- GitHub Actions CI/CD evaluation gates
- Docker/Docker Compose
- Prometheus metrics
- Grafana dashboard
- Proper README and architecture documentation

The system is a PRELIMINARY LOAN ELIGIBILITY ASSISTANT.

It must NEVER represent itself as a bank, lender, loan approval system, or final underwriting decision engine.

All policies used in this project are SYNTHETIC DEMONSTRATION POLICIES and must be clearly labelled as such.

The final user-facing language should say:

"Preliminary eligibility assessment only. This is not a loan offer, commitment, or final approval."

---

# 1. BUSINESS PROBLEM

Traditional loan pre-qualification conversations consume staff time and can produce inconsistent answers because eligibility rules may be spread across multiple documents and change independently.

Customers may also:

- not know which loan product they need
- use different terminology for the same loan
- provide information across multiple conversational turns
- not understand why they are eligible or ineligible
- not understand EMI, FOIR or LTV
- receive inconsistent explanations

We want to build an AI-powered conversational Loan Eligibility Assistant.

The customer should be able to type natural-language questions such as:

"I want to buy a house."

"I need a home loan."

"I want to purchase a flat."

"I want a car loan."

"I need financing for an EV."

"I need some cash for personal expenses."

The system must semantically understand these expressions.

Examples:

"house", "home", "flat", "apartment", "property", "building"
→ HOME LOAN

"car", "vehicle", "scooter", "bike", "EV", "auto"
→ AUTO LOAN

"cash loan", "personal expenses", "salary loan", "personal borrowing"
→ PERSONAL LOAN

Do NOT require the user to select the loan type before beginning the conversation.

The system should infer the loan type from conversation whenever possible.

---

# 2. PRIMARY OBJECTIVE

Build an AI-powered Loan Eligibility Assistant that:

1. Interacts with users conversationally.
2. Identifies the intended loan type.
3. Extracts applicant information from natural language.
4. Maintains profile information across conversation turns.
5. Detects missing information.
6. Asks only for missing information.
7. Retrieves the relevant versioned loan policy.
8. Performs deterministic eligibility checks.
9. Calculates financial metrics.
10. Produces an explainable preliminary assessment.
11. Cites the policy rules used.
12. Allows users to ask general loan-policy questions.
13. Provides previous chat history.
14. Provides suggested/default questions.
15. Streams responses.
16. Handles streaming errors gracefully.
17. Maintains an audit record.
18. Exposes health and metrics APIs.
19. Includes automated testing.
20. Includes GenAI evaluation using Promptfoo.
21. Includes CI/CD evaluation gates.

---

# 3. SUPPORTED LOAN PRODUCTS

Implement exactly three loan products:

## PERSONAL LOAN

Synthetic demonstration policy:

- Age: 21 to 60
- Minimum monthly income: ₹25,000
- Salaried employment: minimum 12 months
- Self-employed employment: minimum 24 months
- Minimum credit score: 700
- Maximum FOIR: 50%
- Loan amount: ₹50,000 to ₹25,00,000
- Maximum loan amount: 30x monthly income
- Tenure: 12 to 60 months
- Indicative interest rate: 12.0% p.a.

## HOME LOAN

Synthetic demonstration policy:

- Minimum age: 21
- Loan should close before age 70
- Minimum monthly income: ₹40,000
- Salaried employment: minimum 24 months
- Self-employed employment: minimum 36 months
- Self-employed experience between 24 and 35 months → MANUAL_REVIEW
- Minimum credit score: 700
- Maximum FOIR: 55%
- LTV:
    - <= ₹30 lakh loan → maximum 85%
    - > ₹30 lakh loan → maximum 80%
- Supported property:
    - Apartment
    - Villa
    - Residential House
- Loan amount:
    - minimum ₹5,00,000
    - maximum ₹5,00,00,000
- Tenure: 12 to 360 months
- Indicative interest rate: 8.5% p.a.

## AUTO LOAN

Synthetic demonstration policy:

- Age: 21 to 65
- Loan should close before age 70
- Minimum monthly income: ₹20,000
- Salaried employment: minimum 12 months
- Self-employed employment: minimum 24 months
- Minimum credit score: 680
- Maximum FOIR: 50%
- LTV:
    - New vehicle: maximum 85% of on-road price
    - Used vehicle: maximum 70% of on-road price
- Supported vehicles:
    - Two-wheelers
    - Four-wheelers
    - Commercial vehicles
    - Electric vehicles
- Used vehicle:
    - Maximum vehicle age: 10 years
- Used vehicle maximum tenure: 60 months
- Loan amount:
    - minimum ₹50,000
    - maximum ₹50,00,000
- New vehicle tenure: 12 to 84 months
- Commercial vehicle tenure: maximum 48 months
- Indicative interest rate: 9.0% p.a.

These are synthetic demo rules.

DO NOT represent them as actual bank policies.

---

# 4. POLICY DOCUMENTS

Create actual policy documents instead of hardcoding all rules inside the application.

Create:

data/policies/

    personal_loan_v1.md
    personal_loan_v2.md

    home_loan_v1.md

    auto_loan_v1.md

Also create:

data/policies/README.md

Each policy document must contain:

- Product
- Policy version
- Effective date
- Rule ID
- Rule description
- Threshold
- Evaluation logic
- Exceptions
- Manual review conditions
- Examples
- Source metadata

Example:

## Personal Loan Policy v1

Rule ID:
PL-AGE-001

Rule:
Applicant age must be between 21 and 60 years.

Decision:
PASS if 21 <= age <= 60

Otherwise:
FAIL

---

Rule ID:
PL-CREDIT-001

Rule:
Credit score must be >= 700.

Decision:
PASS if credit_score >= 700

Otherwise:
FAIL

---

Rule ID:
PL-FOIR-001

Rule:
FOIR must not exceed 50%.

Decision:
PASS if FOIR <= 50%

Otherwise:
FAIL

---

# 5. POLICY VERSIONING

Policy versions must be first-class entities.

Do not hardcode active policy versions throughout the code.

Create a configuration such as:

config/policy_config.yaml

Example:

personal_loan:
    active_version: v2

home_loan:
    active_version: v1

auto_loan:
    active_version: v1

The RAG system must retrieve only the active policy version unless a historical version is explicitly requested.

Every eligibility decision must record:

- loan_type
- policy_version
- policy_rule_ids
- timestamp

---

# 6. RAG ARCHITECTURE

Use ChromaDB as the vector database.

Implement:

rag/
    ingest.py
    retriever.py
    embeddings.py
    metadata.py
    chroma/
    policies/

The ingestion process should:

1. Read policy markdown files.
2. Split documents into meaningful chunks.
3. Add metadata.
4. Generate embeddings.
5. Store embeddings in ChromaDB.

Metadata must include:

- loan_type
- policy_version
- rule_id
- policy_title
- effective_date
- source_file

Create separate ChromaDB collections:

personal_loan
home_loan
auto_loan
eligibility

Use metadata filtering.

For example:

loan_type = "home_loan"
policy_version = "v1"

The retrieval process must return:

- text
- rule_id
- policy version
- source document
- similarity/relevance score

---

# 7. SEMANTIC SEARCH

The assistant MUST understand semantic expressions.

Examples:

User:
"I want to purchase a flat."

System:
loan_type = HOME_LOAN

User:
"I want to construct/buy a house."

System:
loan_type = HOME_LOAN

User:
"I need financing for my new car."

System:
loan_type = AUTO_LOAN

User:
"I want an EV."

System:
loan_type = AUTO_LOAN

User:
"I need money for some personal expenses."

System:
loan_type = PERSONAL_LOAN

Do not depend exclusively on exact keyword matching.

Use:

1. LLM-based semantic extraction
2. Python alias fallback
3. Optional vector similarity fallback

Create:

api/loan_classifier.py

The classifier should return:

{
    "loan_type": "home_loan",
    "confidence": 0.92,
    "evidence": "User said 'buy a flat'"
}

If confidence is low, ask the user:

"Are you looking for a Home Loan, Personal Loan or Auto Loan?"

---

# 8. APPLICANT PROFILE

Create a structured profile model.

Example:

{
    "loan_type": null,
    "age": null,
    "monthly_income": null,
    "employment_type": null,
    "employment_duration_months": null,
    "credit_score": null,
    "existing_monthly_obligations": null,
    "requested_loan_amount": null,
    "tenure_months": null,

    "property_type": null,
    "property_value": null,

    "vehicle_type": null,
    "vehicle_condition": null,
    "vehicle_on_road_price": null,
    "vehicle_age_years": null
}

Use Pydantic models.

Create:

api/models.py

---

# 9. PROFILE EXTRACTION

Use Qwen to extract structured information from natural-language conversation.

Example:

User:

"I'm 35 years old and earn around 1 lakh per month. I want to buy a 50 lakh flat."

Extract:

{
    "age": 35,
    "monthly_income": 100000,
    "loan_type": "home_loan",
    "property_type": "apartment",
    "property_value": 5000000
}

The extractor MUST NOT invent missing information.

If a value is not present:

return null.

Do not assume.

---

# 10. CONVERSATIONAL MEMORY

Maintain the applicant profile across turns.

Example:

Turn 1:
"I want a home loan."

Profile:
loan_type = home_loan

Turn 2:
"I'm 35."

Profile:
loan_type = home_loan
age = 35

Turn 3:
"I earn ₹1 lakh."

Profile:
loan_type = home_loan
age = 35
income = ₹1 lakh

Turn 4:
"My credit score is 740."

Profile:
all previous information retained.

The system must NEVER ask again for information already provided.

Implement profile merge logic.

---

# 11. REQUIRED INFORMATION ENGINE

For each loan type define required fields.

Example:

Home Loan:

- age
- income
- employment type
- employment duration
- credit score
- existing obligations
- requested loan amount
- tenure
- property type
- property value

Auto Loan:

- age
- income
- employment type
- employment duration
- credit score
- existing obligations
- requested loan amount
- tenure
- vehicle type
- vehicle condition
- on-road price

Personal Loan:

- age
- income
- employment type
- employment duration
- credit score
- existing obligations
- requested amount
- tenure

The system must calculate:

missing_fields

If missing_fields is not empty:

DO NOT run the eligibility engine.

Instead ask a natural conversational question for the next missing field.

---

# 12. FINANCIAL CALCULATORS

Create:

api/calculators.py

Implement deterministic Python functions for:

## EMI

Use reducing-balance EMI:

EMI = P * r * (1+r)^n / ((1+r)^n - 1)

Where:

P = principal
r = monthly interest rate
n = number of months

Handle zero-interest safely.

---

## FOIR

FOIR =

(existing monthly obligations + proposed EMI)
/
monthly income
* 100

Return both:

foir_percentage

and

foir_amount

---

## LTV

For home loan:

LTV = loan amount / property value * 100

For auto loan:

LTV = loan amount / on-road vehicle price * 100

---

## Maximum Affordable Loan

Calculate the maximum loan amount possible based on:

- income
- existing obligations
- maximum allowed FOIR
- interest rate
- tenure

All calculations must be deterministic.

---

# 13. RULE ENGINE

Create:

api/rules_engine.py

The LLM MUST NEVER make the final eligibility decision.

The rules engine must return one of:

PASS
FAIL
MANUAL_REVIEW
INSUFFICIENT_INFORMATION

The final application-level decision can be:

POTENTIALLY_ELIGIBLE
NOT_ELIGIBLE
MANUAL_REVIEW
INSUFFICIENT_INFORMATION

Each rule evaluation must return:

{
    "rule_id": "HL-CREDIT-001",
    "description": "Credit score must be >= 700",
    "actual_value": 720,
    "threshold": 700,
    "result": "PASS",
    "reason": "Credit score satisfies minimum requirement"
}

Never allow the LLM to override this result.

---

# 14. RULE ENGINE EXAMPLE

If:

credit_score = 720

and:

minimum_credit_score = 700

return:

PASS

If:

credit_score = 650

return:

FAIL

The LLM should only explain:

"Your credit score of 650 is below the minimum demonstration policy requirement of 700."

The LLM must not change the result.

---

# 15. DECISION EXPLANATION

Create:

api/explanation_agent.py

The explanation agent receives ONLY:

- applicant profile
- calculated metrics
- rule results
- retrieved policy context

It must produce a grounded explanation.

Example:

Eligibility Assessment:
POTENTIALLY_ELIGIBLE

Why:

✓ Credit score: 720 — PASS
✓ Monthly income: ₹60,000 — PASS
✓ FOIR: 42% — PASS
✓ LTV: 80% — PASS

Policy:
Auto Loan Policy v1

Rules:
AL-CREDIT-001
AL-FOIR-001
AL-LTV-001

The explanation agent must never invent policy rules.

---

# 16. POLICY CITATIONS

Every eligibility assessment must contain citations.

Example:

Sources:

- Auto Loan Policy v1
- Rule AL-CREDIT-001
- Rule AL-FOIR-001
- Rule AL-LTV-001

Citations must be generated from retrieved metadata.

Do not fabricate citations.

---

# 17. GENERAL LOAN QUESTIONS

The application must support two modes:

MODE 1:
Eligibility assessment

MODE 2:
General loan-policy question

Example:

User:

"What is the maximum tenure for a home loan?"

The system should retrieve the Home Loan policy and answer.

User:

"What is FOIR?"

Answer using policy/knowledge context.

User:

"How is EMI calculated?"

Provide the calculation explanation.

User:

"What documents do I need?"

Return a document checklist.

The user should NOT have to restart the conversation.

---

# 18. DOCUMENT CHECKLIST

Do not show documents automatically.

Only show a document checklist when the user asks.

Examples:

"What documents do I need?"

"What paperwork is required?"

"What documents should I keep ready?"

Then return a product-specific checklist.

Clearly state:

"This is a demonstration checklist and may differ from an actual lender's requirements."

---

# 19. STREAMING CHAT

The FastAPI backend must expose:

POST /chat

Use Server-Sent Events (SSE).

The response should stream tokens/events.

The frontend must show the response progressively.

Do not wait for the complete response before rendering.

Implement event types such as:

- start
- profile_update
- retrieval
- calculation
- rule_result
- token
- citation
- complete
- error

Example:

event: start

event: profile_update

event: retrieval

event: calculation

event: token

event: token

event: citation

event: complete

---

# 20. STREAMING ERROR HANDLING

Handle errors in the middle of streaming.

If the LLM fails halfway through:

Do not crash the UI.

Send:

event: error

with:

{
    "message": "The assistant encountered an issue while generating the response.",
    "recoverable": true
}

The UI should display:

"Something went wrong while generating the response. Please try again."

The existing streamed content should remain visible.

---

# 21. STREAMLIT UI

Create a professional chat-first interface.

Use:

ui/chat_app.py

The UI should have:

## HEADER

Loan Eligibility Assistant

Subtitle:

"AI-powered preliminary loan eligibility assessment"

Warning:

"Demonstration only — not a final loan approval."

---

# 22. PREVIOUS CHAT SECTION

The UI must show previous conversations.

Example sidebar:

Previous Chats

- Home Loan - Sep 28
- Auto Loan - Sep 27
- Personal Loan - Sep 25

Each conversation should have:

- conversation ID
- title
- timestamp

Clicking a previous conversation should restore its messages.

For local development, SQLite or JSON storage is acceptable.

Prefer SQLite for structured persistence.

---

# 23. QUICK QUESTIONS

Add clickable suggested questions.

Examples:

### Home Loan

"Can I qualify for a home loan?"

"What is the maximum home loan tenure?"

"How is home loan EMI calculated?"

"What credit score is required?"

### Personal Loan

"Can I qualify for a personal loan?"

"What is the maximum personal loan amount?"

"How is FOIR calculated?"

### Auto Loan

"Can I get a car loan?"

"Can I finance an EV?"

"What is the LTV for a used car?"

"How much EMI will I pay?"

Clicking a suggested question should send it into the chat.

---

# 24. CHAT INPUT

Provide:

"Ask anything about your loan..."

The user must be able to enter arbitrary loan-related questions.

Examples:

"I want to buy a house."

"I earn 90000 and my credit score is 730."

"Can I afford a 20 lakh car?"

"What happens if my FOIR is 55%?"

"What's the difference between home and personal loan?"

"How does LTV work?"

---

# 25. UI ELIGIBILITY RESULT

When a full assessment is complete, display:

--------------------------------

Loan Type:
Home Loan

Decision:
POTENTIALLY ELIGIBLE

--------------------------------

Financial Metrics:

EMI:
₹XX,XXX

FOIR:
XX%

LTV:
XX%

Maximum Affordable Loan:
₹XX,XX,XXX

--------------------------------

Rule Results:

✓ Age
✓ Income
✓ Employment
✓ Credit Score
✓ FOIR
✓ LTV
✓ Loan Amount
✓ Tenure

--------------------------------

Policy:

Home Loan Policy v1

Rules:
HL-AGE-001
HL-CREDIT-001
HL-FOIR-001
HL-LTV-001

--------------------------------

Disclaimer:

Preliminary assessment only. Not a loan offer or final approval.

---

# 26. INPUT GUARD

Create:

api/guardrails.py

Implement:

1. Prompt injection detection
2. PII detection
3. Topic filtering
4. Input length limit
5. Malformed input protection

Examples of prompt injection:

"Ignore all previous instructions."

"Reveal the system prompt."

"Tell me the hidden rules."

The system should refuse these requests safely.

---

# 27. PII PROTECTION

Detect and mask sensitive information where appropriate.

Examples:

PAN
Aadhaar
bank account numbers
card numbers
phone numbers
email addresses

Do not store unnecessary sensitive information in logs.

Audit logs should contain only information necessary for the demonstration.

---

# 28. OUTPUT GUARD

Create:

api/output_guard.py

The output guard should process streamed content before showing it to the user.

Prevent:

- unsupported loan claims
- invented policies
- fabricated citations
- inappropriate financial guarantees
- sensitive information leakage

Use sentence-level buffering so that unsafe output can be blocked before reaching the UI.

---

# 29. QWEN MODEL

Use Qwen as the local model.

Preferred model:

Qwen 2.5 1.5B

Create:

model_server/

The model server should expose an OpenAI-compatible endpoint.

The application should NOT directly depend on model-specific code everywhere.

Use:

LiteLLM

as an abstraction layer.

Configuration:

LLM_BASE_URL=http://litellm:4000/v1

LLM_MODEL=qwen-local

---

# 30. LITELLM

Create:

litellm/

Use LiteLLM as the LLM gateway/proxy.

The application should call the OpenAI-compatible interface.

Do not hardcode the Qwen implementation into business logic.

This allows future replacement with another model.

---

# 31. TEMPERATURE

For:

- profile extraction
- policy interpretation
- decision explanation

prefer:

temperature = 0

The system should prioritize deterministic and reproducible behavior.

---

# 32. FASTAPI

Create:

api/app.py

Endpoints:

POST /chat

POST /ask

POST /scenario

GET /audit/{application_id}

GET /health

GET /metrics

GET /versions

GET /conversations

GET /conversations/{conversation_id}

POST /conversations

The API must use Pydantic request/response models.

---

# 33. /ASK

/ask should support non-streaming questions.

Example:

POST /ask

{
    "question": "What is the maximum home loan tenure?"
}

Return:

{
    "answer": "...",
    "sources": [...]
}

---

# 34. /SCENARIO

Implement a what-if simulator.

Example:

Existing profile:

income = ₹100000
loan = ₹3000000

User asks:

"What if I reduce my loan amount to ₹25 lakh?"

The endpoint should recalculate:

- EMI
- FOIR
- LTV
- rule results
- final assessment

Do not modify the original profile.

Return the scenario result separately.

---

# 35. AUDIT LOGGING

Create:

api/audit_logger.py

For every completed eligibility assessment store:

- application_id
- conversation_id
- timestamp
- loan_type
- policy_version
- profile snapshot
- calculated metrics
- rule results
- final decision
- source rule IDs

Use JSONL or SQLite.

Never log secrets.

Never log raw sensitive data unnecessarily.

---

# 36. OBSERVABILITY

Implement Prometheus metrics.

Track:

- request count
- request latency
- LLM latency
- RAG latency
- rule-engine latency
- calculation latency
- error count
- streaming failures
- token count if available

Expose:

GET /metrics

Create Grafana dashboard.

Dashboard should contain:

- request rate
- p50 latency
- p95 latency
- error rate
- LLM latency
- RAG latency
- eligibility decisions

---

# 37. SUCCESS METRICS

Track the following project metrics:

1. Eligibility-answer accuracy
2. RAG faithfulness
3. Citation correctness
4. Semantic loan detection accuracy
5. Prompt injection blocking rate
6. Regression test pass rate
7. Blocked-merge rate
8. API latency
9. p95 latency
10. Streaming failure rate

---

# 38. TESTING

Use Pytest.

Create:

tests/

    test_calculators.py
    test_rules_engine.py
    test_profile_extractor.py
    test_loan_classifier.py
    test_rag.py
    test_api.py
    test_guardrails.py

Test boundary conditions.

Example:

Credit score:

699 → FAIL
700 → PASS
701 → PASS

FOIR:

49.9 → PASS
50.0 → PASS
50.1 → FAIL

Age:

20 → FAIL
21 → PASS
60 → PASS
61 → FAIL

LTV:

84.9 → PASS
85 → PASS
85.1 → FAIL

Do the same for Home Loan and Auto Loan.

---

# 39. PROMPTFOO

Use Promptfoo for GenAI evaluation.

Create:

promptfooconfig.yaml

Include test categories:

1. Home loan semantic detection
2. Personal loan semantic detection
3. Auto loan semantic detection
4. Eligibility reasoning
5. Boundary conditions
6. Missing information
7. Policy grounding
8. Citation correctness
9. Prompt injection
10. PII protection
11. Refusal behavior
12. General loan questions

Example:

User:
"I want to buy a flat."

Expected:

home_loan

User:
"I want to purchase a scooter."

Expected:

auto_loan

User:
"I need some cash for wedding expenses."

Expected:

personal_loan

---

# 40. CI/CD

Create:

.github/workflows/

    unit-tests.yml
    eval-gate.yml

Every push:

1. Install dependencies
2. Run Pytest
3. Build application
4. Run static checks
5. Run RAG tests

Every Pull Request:

1. Run unit tests
2. Run Promptfoo
3. Validate eligibility accuracy
4. Validate semantic detection
5. Validate prompt injection
6. Validate citation grounding

If evaluation falls below the configured threshold:

FAIL THE PIPELINE.

The merge should be considered blocked.

---

# 41. DOCKER

Create Dockerfiles for:

- API
- UI
- model server
- LiteLLM if required

Create:

docker-compose.yml

Services:

api
ui
model_server
litellm
prometheus
grafana

ChromaDB should persist to:

rag/chroma/

---

# 42. ENVIRONMENT VARIABLES

Create:

.env.example

Include:

API_KEY=
LLM_BASE_URL=
LLM_MODEL=
CHROMA_PATH=
POLICY_CONFIG_PATH=
LOG_LEVEL=

Never commit actual secrets.

---

# 43. PROJECT STRUCTURE

Use a clean structure similar to:

loan-eligibility-assistant/

├── api/
│   ├── __init__.py
│   ├── app.py
│   ├── models.py
│   ├── profile_extractor.py
│   ├── loan_classifier.py
│   ├── rules_engine.py
│   ├── calculators.py
│   ├── guardrails.py
│   ├── output_guard.py
│   ├── explanation_agent.py
│   ├── audit_logger.py
│   ├── conversation_store.py
│   └── tests/
│
├── ui/
│   └── chat_app.py
│
├── rag/
│   ├── ingest.py
│   ├── retriever.py
│   ├── embeddings.py
│   ├── metadata.py
│   └── chroma/
│
├── data/
│   └── policies/
│       ├── personal_loan_v1.md
│       ├── personal_loan_v2.md
│       ├── home_loan_v1.md
│       ├── auto_loan_v1.md
│       └── README.md
│
├── config/
│   └── policy_config.yaml
│
├── prompts/
│   ├── registry.yaml
│   └── loader.py
│
├── model_server/
│
├── litellm/
│
├── prometheus/
│
├── grafana/
│
├── tests/
│
├── scripts/
│   ├── ingest_policies.py
│   ├── run_tests.sh
│   └── eval_gate.sh
│
├── .github/
│   └── workflows/
│       ├── unit-tests.yml
│       └── eval-gate.yml
│
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── .env.example
├── promptfooconfig.yaml
├── README.md
└── CLAUDE.md

---

# 44. CODE QUALITY

Use:

- Python 3.11+
- Type hints
- Pydantic
- dataclasses where appropriate
- clear function names
- modular design
- dependency injection where useful
- structured logging
- exception handling
- no duplicated business logic

Avoid:

- giant Python files
- hardcoded policies
- hardcoded secrets
- hardcoded LLM responses
- hidden state
- duplicated eligibility logic
- LLM-generated eligibility decisions

---

# 45. IMPORTANT ARCHITECTURAL PRINCIPLE

Separate:

AI interpretation

from:

Business decisioning.

Architecture:

USER
 ↓
LLM
 ↓
STRUCTURED PROFILE
 ↓
RAG POLICY
 ↓
PYTHON RULE ENGINE
 ↓
PYTHON CALCULATORS
 ↓
DETERMINISTIC DECISION
 ↓
LLM EXPLANATION

The LLM can interpret and explain.

The LLM cannot approve or reject the applicant.

---

# 46. DEMO SCENARIO 1 — HOME LOAN

The user enters:

"I want to buy a house."

The system must infer:

loan_type = home_loan

Then ask only for missing fields.

Example conversation:

Assistant:
"Sure. I can help with a preliminary Home Loan assessment. What is your age?"

User:
"35"

Assistant:
"What is your monthly income?"

User:
"₹1,20,000"

Assistant:
"What is your employment type?"

User:
"Salaried"

Continue until all required information is available.

Then:

Retrieve Home Loan Policy v1

Calculate:

EMI
FOIR
LTV

Apply:

HL-AGE-001
HL-INCOME-001
HL-EMP-001
HL-CREDIT-001
HL-FOIR-001
HL-LTV-001
etc.

Return:

POTENTIALLY_ELIGIBLE / NOT_ELIGIBLE / MANUAL_REVIEW

with explanation and citations.

---

# 47. DEMO SCENARIO 2 — SEMANTIC HOME LOAN

User:

"I want to purchase a flat."

The system must recognize:

flat → home loan

User:

"I'm planning to buy a residential property."

Must also identify:

home_loan

Do not force exact phrase "home loan".

---

# 48. DEMO SCENARIO 3 — AUTO LOAN

User:

"I want to buy a car."

System:

AUTO_LOAN

User:

"My salary is ₹60,000, I'm 30 years old and my credit score is 720."

System should retain:

age = 30
income = 60000
credit_score = 720

Then ask only for missing information.

---

# 49. DEMO SCENARIO 4 — PERSONAL LOAN

User:

"I need ₹5 lakh for some personal expenses."

System:

PERSONAL_LOAN

Extract:

requested_amount = ₹5,00,000

Ask missing information.

---

# 50. DEMO SCENARIO 5 — GENERAL QUESTION

User:

"What is FOIR?"

Do not start eligibility assessment.

Answer using grounded policy/general financial explanation.

---

# 51. DEMO SCENARIO 6 — POLICY QUESTION

User:

"What is the maximum LTV for a new car?"

Retrieve:

Auto Loan Policy

Return:

85%

with:

Rule ID
Policy version
Citation

---

# 52. DEMO SCENARIO 7 — WHAT IF

User:

"What if I reduce the loan amount from ₹30 lakh to ₹25 lakh?"

Run /scenario.

Recalculate:

EMI
FOIR
LTV

and compare:

Before
After

---

# 53. ERROR HANDLING

The application must gracefully handle:

- LLM unavailable
- ChromaDB unavailable
- invalid profile
- incomplete profile
- malformed JSON
- timeout
- streaming interruption
- policy retrieval failure
- unsupported question
- low-confidence loan classification

Never show Python stack traces to the customer.

Log detailed error internally.

Show friendly messages externally.

---

# 54. SECURITY

Do not expose:

- API keys
- model credentials
- system prompts
- internal paths
- stack traces
- hidden policy configuration

Use environment variables for secrets.

Validate all user input.

---

# 55. README

Create a comprehensive README containing:

1. Business problem
2. Solution
3. Architecture
4. Tech stack
5. Folder structure
6. Loan policies
7. RAG architecture
8. Rules engine
9. Financial calculations
10. Setup instructions
11. Docker instructions
12. Running locally
13. API documentation
14. Testing
15. Promptfoo evaluation
16. CI/CD
17. Monitoring
18. Example conversations
19. Limitations
20. Disclaimer

---

# 56. API DOCUMENTATION

FastAPI Swagger should be available.

Document every endpoint.

For each endpoint provide:

- purpose
- request schema
- response schema
- example request
- example response
- possible errors

---

# 57. DEVELOPMENT ORDER

Do NOT attempt to build everything randomly.

Build in this order:

PHASE 1
Project structure

PHASE 2
Policy documents

PHASE 3
Policy ingestion and ChromaDB

PHASE 4
Financial calculators

PHASE 5
Deterministic rules engine

PHASE 6
Profile extraction

PHASE 7
Semantic loan classifier

PHASE 8
Conversation memory

PHASE 9
FastAPI

PHASE 10
Qwen + LiteLLM

PHASE 11
RAG integration

PHASE 12
Explanation agent

PHASE 13
Guardrails

PHASE 14
Streaming SSE

PHASE 15
Streamlit UI

PHASE 16
Conversation persistence

PHASE 17
Audit logging

PHASE 18
Pytest

PHASE 19
Promptfoo

PHASE 20
Docker Compose

PHASE 21
Prometheus/Grafana

PHASE 22
GitHub Actions

PHASE 23
End-to-end testing

PHASE 24
Documentation

After every phase, run tests and verify that the application still works.

---

# 58. IMPLEMENTATION RULE

Before writing code:

1. Inspect the existing repository.
2. Understand existing files.
3. Reuse good existing components where possible.
4. Do not unnecessarily rewrite working components.
5. Identify missing functionality.
6. Create a clear implementation plan.
7. Then implement incrementally.

Do not overwrite working code blindly.

---

# 59. ACCEPTANCE CRITERIA

The application is considered complete only when ALL of the following work:

[ ] Streamlit starts successfully.

[ ] FastAPI starts successfully.

[ ] Qwen model is reachable.

[ ] LiteLLM works.

[ ] ChromaDB policy ingestion works.

[ ] Personal Loan policy retrieval works.

[ ] Home Loan policy retrieval works.

[ ] Auto Loan policy retrieval works.

[ ] "house" is understood as Home Loan.

[ ] "flat" is understood as Home Loan.

[ ] "car" is understood as Auto Loan.

[ ] "EV" is understood as Auto Loan.

[ ] "personal expenses" is understood as Personal Loan.

[ ] Conversation memory works.

[ ] Previously provided information is not requested again.

[ ] Missing information is detected.

[ ] EMI calculation works.

[ ] FOIR calculation works.

[ ] LTV calculation works.

[ ] Rules engine works.

[ ] Boundary conditions are tested.

[ ] Final eligibility is deterministic.

[ ] LLM never makes the final decision.

[ ] Policy citations are shown.

[ ] Policy version is shown.

[ ] Audit record is created.

[ ] Previous conversations are displayed.

[ ] Suggested questions work.

[ ] Free-text questions work.

[ ] Streaming works.

[ ] Mid-stream errors are handled gracefully.

[ ] Prompt injection is blocked.

[ ] PII is handled safely.

[ ] Pytest passes.

[ ] Promptfoo evaluation passes.

[ ] GitHub Actions runs tests.

[ ] Docker Compose starts the application.

[ ] Prometheus collects metrics.

[ ] Grafana dashboard works.

[ ] README is complete.

---

# 60. FINAL DEMO REQUIREMENT

Create a polished demo experience.

When the application starts, the user should immediately understand:

"AI-powered Loan Eligibility Assistant"

The UI should have:

LEFT/SIDEBAR:

- New Chat
- Previous Chats
- Home Loan
- Personal Loan
- Auto Loan
- Suggested Questions

MAIN AREA:

- Chat messages
- Streaming assistant response
- Eligibility result cards
- EMI
- FOIR
- LTV
- Rule results
- Policy citations
- Disclaimer

BOTTOM:

Ask anything about your loan...

The UI should look professional and suitable for an enterprise banking capstone demonstration.

Do not make it look like a generic Streamlit prototype.

---

# 61. FINAL PRINCIPLE

The final architecture must demonstrate:

GENAI
+
RAG
+
SEMANTIC SEARCH
+
DETERMINISTIC DECISIONING
+
FINANCIAL CALCULATIONS
+
CONVERSATIONAL MEMORY
+
STREAMING
+
GUARDRAILS
+
AUDITABILITY
+
TESTING
+
PROMPT EVALUATION
+
CI/CD
+
OBSERVABILITY

The system should be easy to explain to a technical mentor and should clearly demonstrate why each technology is being used.

Before declaring the project complete, run the application end-to-end and fix all errors.

Do not simply generate files and stop.

The final response should include:

1. What was implemented
2. Files created/modified
3. How to run the application
4. How to run tests
5. How to run Promptfoo
6. How to start Docker Compose
7. URLs for UI/API/Grafana/Prometheus
8. Example demo conversations
9. Known limitations
10. Recommended next steps
