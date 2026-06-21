"""
Data collection pipeline.
Runs newsapi_collector.py, reddit_collector.py, rss_collector.py, and
hackernews_collector.py in sequence. Each one independently appends
to its own JSON file (newsapi_data.json, reddit_data.json,
rss_data.json, hackernews_data.json).

Run: python -m collectors.run_pipeline
"""
from collectors import run_pipeline
from preprocess import clean, sentiment, chunks
from storage import store


def main():
    print("=" * 50)
    print("COLLECTION STARTING NOW !! ")
    print("=" * 50)
    run_pipeline.main()
    
    print("\n" + "=" * 50)
    print("COLLECTION IS COMPLETE !! ")
    print("=" * 50)
    clean.main()

    print("\n" + "=" * 50)
    print("CLEANING IS COMPLETE !! ")
    print("=" * 50)
    sentiment.main()

    print("\n" + "=" * 50)
    print("SENTIMENT ANALYSIS IS COMPLETE !! ")
    print("=" * 50)
    chunks.main()

    print("\n" + "=" * 50)
    print("CHUNKING IS COMPLETE !! ")
    print("=" * 50)
    store.main()

    print("\n" + "=" * 50)
    print("STORING IS COMPLETE !! ")
    print("=" * 50)


    print("\nPipeline complete. Check chunks.json present in DATA / CLEANED")


if __name__ == "__main__":
    main()