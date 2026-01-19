# Scraper module for sentiment project
import snscrape.modules.twitter as sntwitter
import pandas as pd

def scrape(keyword, limit=500):
    tweets = []
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(keyword).get_items()):
        if i >= limit:
            break
        tweets.append([tweet.date, tweet.content])
    df = pd.DataFrame(tweets, columns=["date", "text"])
    df.to_csv("data/raw/tweets.csv", index=False)

if __name__ == "__main__":
    scrape("AI")
