import os
import pprint
from dotenv import load_dotenv 
import trafilatura
from transformers import pipeline
from langchain_community.utilities import GoogleSerperAPIWrapper

load_dotenv()

os.environ["SERPER_API_KEY"] = os.getenv("SERPER_API_KEY")
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")



# =========================
# CONFIG
# =========================

import trafilatura
import torch
from transformers import BartForConditionalGeneration, BartTokenizer

# =========================
# CONFIG
# =========================

MODEL_NAME = "facebook/bart-large-cnn"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MIN_ARTICLE_LENGTH = 500
MAX_ARTICLE_CHARS = 3500


# =========================
# LOAD MODEL (SAFE)
# =========================

print("[INFO] Loading BART model...")

tokenizer = BartTokenizer.from_pretrained(MODEL_NAME)
model = BartForConditionalGeneration.from_pretrained(MODEL_NAME)
model.to(DEVICE)
model.eval()

print(f"[INFO] Model loaded on {DEVICE}")


# =========================
# ARTICLE EXTRACTION
# =========================

def extract_article(url: str) -> str | None:
    try:
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return None

        text = trafilatura.extract(
            downloaded,
            include_comments=False,
            include_tables=False,
            no_fallback=True
        )

        if not text or len(text) < MIN_ARTICLE_LENGTH:
            return None

        return text[:MAX_ARTICLE_CHARS]

    except Exception as e:
        print(f"[ERROR] Failed to extract {url}: {e}")
        return None


# =========================
# SUMMARIZATION (NO PIPELINE)
# =========================

def summarize_article(text: str) -> str | None:
    try:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            max_length=1024,
            truncation=True
        ).to(DEVICE)

        summary_ids = model.generate(
            inputs["input_ids"],
            num_beams=4,
            max_length=130,
            min_length=60,
            length_penalty=2.0,
            early_stopping=True
        )

        return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    except Exception as e:
        print(f"[ERROR] Summarization failed: {e}")
        return None


# =========================
# BUILD FEED ITEM
# =========================

def build_feed_item(item: dict) -> dict | None:
    title = item.get("title", "").strip()
    url = item.get("link")

    if not title or not url:
        return None

    article_text = extract_article(url)
    if not article_text:
        return None

    summary = summarize_article(article_text)
    if not summary:
        return None

    return {
        "title": title,
        "summary": summary,
        "url": url,
        "source": item.get("source"),
        "date": item.get("date"),
    }


# =========================
# MAIN PIPELINE
# =========================

def process_serper_results(serper_response: dict) -> list[dict]:
    feed = []

    for item in serper_response.get("news", []):
        feed_item = build_feed_item(item)
        if feed_item:
            feed.append(feed_item)

    return feed


# =========================
# TEST
# =========================

if __name__ == "__main__":
    
    search = GoogleSerperAPIWrapper(type="news",gl="in")
    res=search.results("Rohit sharma")



    feed = process_serper_results(res)

    for item in feed:
        print("\n================================")
        print(f"TITLE   : {item['title']}")
        print(f"SOURCE  : {item['source']}")
        print(f"DATE    : {item['date']}")
        print(f"SUMMARY : {item['summary']}")
        print(f"URL     : {item['url']}")





