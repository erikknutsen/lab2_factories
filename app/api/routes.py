from fastapi import APIRouter, Body, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List
from app.services.email_topic_inference import EmailTopicInferenceService
from app.dataclasses import Email
import os
import json

router = APIRouter()

class EmailRequest(BaseModel):
    subject: str
    body: str

class EmailWithTopicRequest(BaseModel):
    subject: str
    body: str
    topic: str

class EmailClassificationResponse(BaseModel):
    predicted_topic: str
    topic_scores: Dict[str, float]
    features: Dict[str, Any]
    available_topics: List[str]

class EmailAddResponse(BaseModel):
    message: str
    email_id: int

@router.post("/emails/classify", response_model=EmailClassificationResponse)
async def classify_email(request: EmailRequest, strategy: str = "topics"):
    """
    Classify an email using one of two strategies:

    Parameters
    ----------
    request : EmailRequest
        JSON body with:
          - subject: str
          - body: str
    strategy : str, optional (default="topics")
        - "topics"  : (default) Use existing topic-description similarity provided by
                      EmailTopicInferenceService(). This is the original behavior.
        - "emails"  : Use the ground_truth label of the most similar *stored* email
                      (nearest neighbor by simple token overlap). Falls back to "topics"
                      if there are no labeled stored emails or no overlap.

    Behavior
    --------
    1) Always run the original classifier first (topic-description similarity) so we
       have a baseline prediction and scores.
    2) If strategy == "emails":
         - Load data/emails.json (a list of dicts like:
             {"id": 1, "subject": "...", "body": "...", "ground_truth": "travel"} )
         - Consider only emails that have a non-empty "ground_truth".
         - Tokenize both the incoming request and each stored email with a very simple
           lowercase alphanumeric tokenizer (regex [a-z0-9]+). No stemming/stopwords.
         - Compute token overlap size between the request tokens and each stored email.
         - Pick the stored email with the largest overlap; if one exists, override the
           predicted topic with that email's ground_truth. Also attach a small trace in
           "features.nearest_email" so you can show this choice in the UI or logs.
       If there is no labeled email or overlap, silently keep the baseline result.

    Notes / Constraints
    -------------------
    - This is intentionally *minimal* and keeps the original interface intact.
    - Token overlap is a crude similarity heuristic; it's here for clarity, not accuracy.
    - File path "data/emails.json" is relative to the working directory. Run uvicorn
      from the repo root (as in the README) so this path resolves correctly.
    - Ground truth values should match your topic keys (e.g., "travel", "billing", etc.).
    """
    try:
        # 1) Run the original inference to get the baseline (unchanged behavior)
        inference_service = EmailTopicInferenceService()
        email = Email(subject=request.subject, body=request.body)
        result = inference_service.classify_email(email)

        # Unpack baseline outputs
        predicted = result["predicted_topic"]       # baseline predicted topic
        topic_scores = result["topic_scores"]       # per-topic similarity scores
        features = dict(result.get("features", {})) # copy; we'll append trace info if needed
        available_topics = result["available_topics"]

        # 2) Optional nearest-neighbor override based on stored emails (very small, local logic)
        if strategy == "emails":
            import os, json, re

            # Tokenize the incoming request once
            q_text = f"{request.subject} {request.body}".lower()
            q_tokens = set(re.findall(r"[a-z0-9]+", q_text))

            best_overlap = 0
            best = None
            path = "data/emails.json"

            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    stored = json.load(f) or []

                # Only consider stored emails that have a ground_truth label
                for rec in stored:
                    gt = rec.get("ground_truth")
                    if not gt:
                        continue

                    # Tokenize the stored email
                    s_text = f"{rec.get('subject','')} {rec.get('body','')}".lower()
                    s_tokens = set(re.findall(r"[a-z0-9]+", s_text))

                    # Overlap size is our simple similarity proxy
                    overlap = len(q_tokens & s_tokens)
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best = rec

            # If we found a labeled neighbor with any overlap, override the prediction
            if best:
                predicted = best["ground_truth"]
                # Record a tiny trace so it's visible in the response for demos/screenshots
                features["nearest_email"] = {
                    "id": best.get("id"),
                    "overlap_tokens": best_overlap
                }
            # If we didn't find anything, we keep the original "topics" prediction.

        # Final response matches the original schema
        return EmailClassificationResponse(
            predicted_topic=predicted,
            topic_scores=topic_scores,
            features=features,
            available_topics=available_topics,
        )
    except Exception as e:
        # Standardized error response (kept from original)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/topics")
async def topics():
    """Get available email topics"""
    inference_service = EmailTopicInferenceService()
    info = inference_service.get_pipeline_info()
    return {"topics": info["available_topics"]}

@router.get("/pipeline/info") 
async def pipeline_info():
    inference_service = EmailTopicInferenceService()
    return inference_service.get_pipeline_info()

# --- HW Step 2: POST /topics ---

TOPICS_FILE = "data/topic_keywords.json"

@router.post("/topics", summary="Create topic")
async def add_topic(payload: dict = Body(...)):
    # expected JSON: {"name":"...","description":"...","keywords":[...]}
    name = (payload.get("name") or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="Missing 'name'.")

    description = (payload.get("description") or "").strip()

    data = {}
    if os.path.exists(TOPICS_FILE):
        with open(TOPICS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)  # expects a dict: { "topic": "description", ... }

    # store as a dictionary
    data[name] = {"description": description}


    with open(TOPICS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return {"status": "ok", "topic": name, "description": description, "saved": True}

# --- HW Step 3: POST /emails ---

from typing import Optional 

EMAILS_FILE = "data/emails.json"


class EmailStoreRequest(BaseModel):
    subject: str
    body: str
    ground_truth: Optional[str] = None  # optional label (topic)

@router.post("/emails", response_model=EmailAddResponse, summary="Store an email with optional ground truth")
async def store_email(request: EmailStoreRequest):
    # load current list (or start fresh)
    emails = []
    if os.path.exists(EMAILS_FILE):
        with open(EMAILS_FILE, "r", encoding="utf-8") as f:
            emails = json.load(f)

    # assign a simple incremental id
    next_id = (max((e.get("id", 0) for e in emails), default=0) + 1) if emails else 1

    record = {
        "id": next_id,
        "subject": request.subject,
        "body": request.body,
    }
    if request.ground_truth:
        record["ground_truth"] = request.ground_truth

    emails.append(record)

    # save back to file
    os.makedirs(os.path.dirname(EMAILS_FILE), exist_ok=True)
    with open(EMAILS_FILE, "w", encoding="utf-8") as f:
        json.dump(emails, f, indent=2, ensure_ascii=False)

    return EmailAddResponse(message="Email stored", email_id=next_id)

# TODO: LAB ASSIGNMENT - Part 2 of 2  
# Create a GET endpoint at "/features" that returns information about all feature generators
# available in the system.
#
# Requirements:
# 1. Create a GET endpoint at "/features"
# 2. Import FeatureGeneratorFactory from app.features.factory
# 3. Use FeatureGeneratorFactory.get_available_generators() to get generator info
# 4. Return a JSON response with the available generators and their feature names
# 5. Handle any exceptions with appropriate HTTP error responses
#
# Expected response format:
# {
#   "available_generators": [
#     {
#       "name": "spam",
#       "features": ["has_spam_words"]
#     },
#     ...
#   ]
# }
#
# Hint: Look at the existing endpoints above for patterns on error handling
# Hint: You may need to instantiate generators to get their feature names

