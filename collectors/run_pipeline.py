# """
# Data collection pipeline.
# Runs newsapi_collector.py, reddit_collector.py, rss_collector.py, and
# hackernews_collector.py in sequence. Each one independently appends
# to its own JSON file (newsapi_data.json, reddit_data.json,
# rss_data.json, hackernews_data.json).

# Run: python -m collectors.run_pipeline
# """
# from . import newsapi_collector
# # from . import reddit_collector
# from . import rss_collector
# from . import hackernews_collector


# def main():
#     print("=" * 50)
#     print("Step 1/4: NewsAPI collector")
#     print("=" * 50)
#     newsapi_collector.main()

#     # print("\n" + "=" * 50)
#     # print("Step 2/4: Reddit collector")
#     # print("=" * 50)
#     # reddit_collector.main()

#     print("\n" + "=" * 50)
#     print("Step 2/4: RSS collector")
#     print("=" * 50)
#     rss_collector.main()

#     print("\n" + "=" * 50)
#     print("Step 3/4: Hacker News collector")
#     print("=" * 50)
#     hackernews_collector.main()

#     print("\nPipeline complete. Check newsapi_data.json, rss_data.json, hackernews_data.json")


# if __name__ == "__main__":
#     main()



import json
from datetime import datetime, timezone
 
from config.paths import DATA_DIR
 
from . import newsapi_collector
# from . import reddit_collector
from . import rss_collector
from . import hackernews_collector
 
PIPELINE_META_FILE = DATA_DIR / "pipeline_meta.json"
 
 
def record_pipeline_run():
    """Write the last-successful-collection timestamp for the dashboard."""
    meta = {"last_run": datetime.now(timezone.utc).isoformat()}
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(PIPELINE_META_FILE, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(f"Recorded pipeline run time -> {PIPELINE_META_FILE}")
    except OSError as e:
        # don't fail the whole pipeline just because the stamp didn't write
        print(f"Warning: could not write pipeline_meta.json: {e}")
 
 
def main():
    print("=" * 50)
    print("Step 1/3: NewsAPI collector")
    print("=" * 50)
    newsapi_collector.main()
 
    # print("\n" + "=" * 50)
    # print("Step x: Reddit collector")
    # print("=" * 50)
    # reddit_collector.main()
 
    print("\n" + "=" * 50)
    print("Step 2/3: RSS collector")
    print("=" * 50)
    rss_collector.main()
 
    print("\n" + "=" * 50)
    print("Step 3/3: Hacker News collector")
    print("=" * 50)
    hackernews_collector.main()
 
    # all collectors finished -> stamp the run time
    record_pipeline_run()
 
    print("\nPipeline complete. Check newsapi_data.json, rss_data.json, hackernews_data.json")
 
 
if __name__ == "__main__":
    main()
 