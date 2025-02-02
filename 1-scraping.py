import json
import os
import praw
import prawcore
import time
from datetime import datetime, timezone
from typing import List, Dict, Optional

class RedditScraper:
    """A class to scrape posts and comments from Reddit using PRAW."""

    def __init__(self, config_file: str, subreddits: List[str], query: str, limit: int, output_file: str):
        """
        Initialize the RedditScraper with API credentials, subreddits, and query parameter.

        :param config_file: Path to the config file containing Reddit API credentials.
        :param subreddits: List of subreddit names to scrape.
        :param query: Query string to search within subreddits.
        :param limit: Number of posts to scrape per subreddit.
        :param output_file: Path to the output JSON file.
        """
        self.config_file = config_file
        self.subreddits = subreddits
        self.query = query
        self.limit = limit
        self.output_file = output_file
        self.reddit = self._initialize_reddit_client()
        self.seen_posts = set()
        self.posts = []
        self.post_counter = 0
        self.comment_counter = 0

    def _initialize_reddit_client(self) -> praw.Reddit:
        """Load API credentials from the config file and initialize the Reddit client."""
        if not os.path.exists(self.config_file):
            raise FileNotFoundError(f"Config file {self.config_file} not found.")
        
        with open(self.config_file) as file:
            config = json.load(file)
        
        required_keys = ['client_id', 'client_secret', 'user_agent', 'username', 'password']
        if not all(key in config for key in required_keys):
            raise KeyError("API credentials not found in the config file or incomplete.")
        
        return praw.Reddit(
            client_id=config['client_id'],
            client_secret=config['client_secret'],
            user_agent=config['user_agent'],
            username=config['username'],
            password=config['password']
        )
    
    def scrape(self):
        """Main method to scrape all specified subreddits."""
        for subreddit_name in self.subreddits:
            use_query = subreddit_name != 'queensgambit'
            self._scrape_subreddit(subreddit_name, use_query)
    
    def _scrape_subreddit(self, subreddit_name: str, use_query: bool):
        """Scrape posts and comments from a specific subreddit."""
        subreddit = self.reddit.subreddit(subreddit_name)
        subreddit_posts = subreddit.search(self.query, limit=self.limit, sort='new') if use_query else subreddit.new(limit=self.limit)

        for post in subreddit_posts:
            try:
                if self._fetch_post(post, subreddit_name):
                    self._fetch_comments(post, subreddit_name)
                
                print(
                    f"Processed {self.post_counter} posts and {self.comment_counter} comments "
                    f"(Total: {len(self.seen_posts)}). Subreddit: {subreddit_name}.",
                    end='\r'
                )
            except prawcore.exceptions.TooManyRequests as e:
                print(f"Rate limit error occurred: {e}. Retrying in 60 seconds...")
                time.sleep(60)
            time.sleep(2)  # Avoid hitting Reddit API rate limit
    
    def _fetch_post(self, post, subreddit_name) -> bool:
        """Process a Reddit post."""
        post_identifier = post.selftext.strip().lower()
        if post_identifier in self.seen_posts:
            return False
        
        self.seen_posts.add(post_identifier)
        created_time = datetime.fromtimestamp(post.created_utc, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')

        post_data = {
            "post_id": post.id,
            "title": post.title,
            "text": post.selftext,
            "created_time": created_time,
            "url": post.url,
            "num_comments": post.num_comments,
            "subreddit": subreddit_name,
            "comments": []
        }
        self.posts.append(post_data)
        self.post_counter += 1

        # Save updated data to JSON
        self._save_to_json()
        return True
    
    def _fetch_comments(self, post, subreddit_name):
        """Fetch comments for a specific Reddit post."""
        post.comments.replace_more(limit=0)  # Flatten comments
        for comment in post.comments.list():
            comment_identifier = comment.body.strip().lower()
            if comment_identifier in self.seen_posts:
                continue

            self.seen_posts.add(comment_identifier)
            created_time = datetime.fromtimestamp(comment.created_utc, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')

            comment_data = {
                "post_id": post.id,
                "comment_id": comment.id,
                "text": comment.body,
                "created_time": created_time,
                "subreddit": subreddit_name
            }
            self.posts[-1]['comments'].append(comment_data)
            self.comment_counter += 1

            # Save updated data to JSON
            self._save_to_json()
    
    def _save_to_json(self):
        """Save scraped data to a JSON file."""
        with open(self.output_file, 'w') as file:
            json.dump(self.posts, file, indent=4)

if __name__ == "__main__":
    scraper = RedditScraper(
        config_file="config.json",
        subreddits=['queensgambit', 'netflix', 'NetflixBestOf', 'television', 'TvShows'],
        query="Queen's Gambit",
        limit=15000,
        output_file="./data/reddit_data.json"
    )
    scraper.scrape()
