"""
Data collection pipeline.
Runs newsapi_collector.py, reddit_collector.py, rss_collector.py, and
hackernews_collector.py in sequence. Each one independently appends
to its own JSON file (newsapi_data.json, reddit_data.json,
rss_data.json, hackernews_data.json).

Run: python -m collectors.run_pipeline
"""
from . import newsapi_collector
# from . import reddit_collector
from . import rss_collector
from . import hackernews_collector


def main():
    print("=" * 50)
    print("Step 1/4: NewsAPI collector")
    print("=" * 50)
    newsapi_collector.main()

    # print("\n" + "=" * 50)
    # print("Step 2/4: Reddit collector")
    # print("=" * 50)
    # reddit_collector.main()

    print("\n" + "=" * 50)
    print("Step 2/4: RSS collector")
    print("=" * 50)
    rss_collector.main()

    print("\n" + "=" * 50)
    print("Step 3/4: Hacker News collector")
    print("=" * 50)
    hackernews_collector.main()

    print("\nPipeline complete. Check newsapi_data.json, rss_data.json, hackernews_data.json")


if __name__ == "__main__":
    main()