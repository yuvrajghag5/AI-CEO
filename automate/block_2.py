

from rag import rag
from agent import agent
def main():
    

    print("\n" + "=" * 50)
    print("RAG STARTING NOW !! ")
    print("=" * 50)
    rag.main()

    print("\n" + "=" * 50)
    print("RAG IS COMPLETE !! ")
    print("=" * 50)
    agent.main()

    print("\n" + "=" * 50)
    print("AGENT IS COMPLETE !! ")
    print("=" * 50)



    print("\nPipeline 2 complete. Check ceo_report.json present in DATA / EVIDENCE")


if __name__ == "__main__":
    main()