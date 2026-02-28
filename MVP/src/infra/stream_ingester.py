import os
import threading
import time
from typing import Any, Dict

import httpx


class StreamIngester:
    """
    v1.1 Live Reddit Ingestion Hook.
    Pulls real-time controversial comments from public subreddits
    and pushes them to the local OSKAR `/analyze` endpoint.
    """

    def __init__(self, api_url: str = "http://127.0.0.1:8000/analyze"):
        self.api_url = api_url
        self.is_running = False
        self._thread = None
        self.client = httpx.Client(timeout=30.0)
        self.seen_ids = set()

        # We fetch comments from heavily debated subreddits to test OSKAR's limits
        self.reddit_url = "https://www.reddit.com/r/politics+worldnews+conspiracy+technology/comments.json?limit=15"
        self.headers = {"User-Agent": "OSKAR-LiveStream-Bot/1.0 (Real-Time Moderation Analysis)"}

    def fetch_reddit_comments(self):
        """Fetch the latest comments from Reddit."""
        try:
            response = self.client.get(self.reddit_url, headers=self.headers)
            if response.status_code == 200:
                data = response.json()
                comments = []
                for child in data.get("data", {}).get("children", []):
                    c_data = child.get("data", {})
                    c_id = c_data.get("id")
                    if c_id and c_id not in self.seen_ids:
                        self.seen_ids.add(c_id)

                        # Keep seen_ids from growing infinitely
                        if len(self.seen_ids) > 1000:
                            self.seen_ids.clear()

                        # Filter out very short comments
                        text = c_data.get("body", "")
                        if len(text) > 30 and "I am a bot" not in text:
                            comments.append(
                                {
                                    "user_id": c_data.get("author", "unknown_redditor"),
                                    "text": text,
                                    "social_context": {
                                        "subreddit": c_data.get("subreddit_name_prefixed", ""),
                                        "score": c_data.get("score", 0),
                                        "is_submitter": c_data.get("is_submitter", False),
                                    },
                                }
                            )
                return comments
            else:
                print(f"[StreamIngester] Reddit API Error: {response.status_code}")
                return []
        except Exception as e:
            print(f"[StreamIngester] Failed to fetch Reddit: {e}")
            return []

    def _ingest_loop(self):
        print("[StreamIngester] Started Reddit Live Ingestion loop...")
        while self.is_running:
            comments = self.fetch_reddit_comments()

            for comment in comments:
                if not self.is_running:
                    break

                try:
                    # Post to OSKAR API using key loaded from environment
                    api_key = os.getenv("OSKAR_API_KEY", "")
                    headers = {"X-API-Key": api_key}
                    response = self.client.post(self.api_url, json=comment, headers=headers)

                    if response.status_code == 200:
                        subreddit = comment["social_context"].get("subreddit", "unknown")
                        print(
                            f"[StreamIngester] Injected {subreddit} post from {comment['user_id']}"
                        )
                    else:
                        print(
                            f"[StreamIngester] API Error: {response.status_code} - {response.text}"
                        )

                except Exception as e:
                    print(f"[StreamIngester] Connection Error: {e}")

                # Sleep between individual posts to create a steady stream on the dashboard
                time.sleep(4.0)

            # Sleep before fetching the next batch from Reddit to respect API rate limits (1 req / sec max)
            time.sleep(15.0)

    def start(self):
        if not self.is_running:
            self.is_running = True
            self._thread = threading.Thread(target=self._ingest_loop, daemon=True)
            self._thread.start()

    def stop(self):
        self.is_running = False
        if self._thread:
            self._thread.join(timeout=2.0)


# Global singleton
stream_ingester = StreamIngester()
