import os
import argparse
import requests
from datetime import date, timedelta
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Load API key from environment variable
KEY = os.getenv('FINNHUB_KEY')
if not KEY:
    raise ValueError("FINNHUB_KEY not found in environment variables. Make sure it's set in your .env file.")

def fetch_and_save(output_dir="data/raw_news"):
    os.makedirs(output_dir, exist_ok=True)
    
    # Example: get the last 7 days company news for AAPL (Apple)
    to_date = date.today()
    from_date = to_date - timedelta(days=7)
    url = "https://finnhub.io/api/v1/company-news"
    params = {
        "symbol": "AAPL",
        "from": from_date.isoformat(),
        "to": to_date.isoformat(),
        "token": KEY,
    }

    print(f"Fetching news with API key: {KEY[:4]}...{KEY[-4:]}")  # Show first/last 4 chars of key
    resp = requests.get(url, params=params)
    try:
        resp.raise_for_status()
    except requests.exceptions.HTTPError as e:
        print(f"Error fetching data: {e}")
        print(f"Response content: {resp.text}")
        raise


    articles = resp.json()
    
    for idx, a in enumerate(articles):
        doc = {
            "source": a.get("source", "finnhub"),
            "headline": a.get("headline"),
            "url": a.get("url"),
            "datetime": a.get("datetime"),
            "summary": a.get("summary") or "",
        }
        
        # Save each article as a separate text file
        filename = f"article_{idx}_{a.get('datetime')}.txt"
        filepath = os.path.join(output_dir, filename)
        
        content = [
            f"Title: {doc['headline']}",
            f"Source: {doc['source']}",
            f"Date: {doc['datetime']}",
            "",
            f"Summary: {doc['summary']}",
            "",
            f"URL: {doc['url']}"
        ]
        
        with open(filepath, "w", encoding="utf-8") as fh:
            fh.write("\n".join(content))
    
    print(f"Saved {len(articles)} articles to {output_dir}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/raw_news", help="Output directory")
    args = parser.parse_args()
    fetch_and_save(args.out)

if __name__ == "__main__":
    main()
