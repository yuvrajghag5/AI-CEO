
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langgraph.prebuilt import create_react_agent
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
 
from config.settings import MODEL
from agent.tools import ALL_TOOLS
 
_agent = None  # lazily built once, reused across questions
 
 
def _build_chat_model():
    print(f"Loading {MODEL} for the agent loop ... (first run can take a while)")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        torch_dtype="auto",
        device_map="auto",
    )
 
    text_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=600,
        do_sample=False,  # deterministic tool-routing; final briefing step still uses outlines separately
    )
 
    llm = HuggingFacePipeline(pipeline=text_pipeline)
    chat_model = ChatHuggingFace(llm=llm, tokenizer=tokenizer)
    return chat_model
 
 
def get_agent():
    global _agent
    if _agent is None:
        chat_model = _build_chat_model()
        _agent = create_react_agent(chat_model, tools=ALL_TOOLS)
    return _agent
 
 
def ask(question: str) -> dict:
    """
    Runs the full reason -> act -> observe loop for one question.
    Returns a dict with the final answer text and the list of tool
    calls that were actually executed (useful for the dashboard to
    show a "tool trace" alongside the answer, proving the agent is
    really calling tools and not just hallucinating).
    """
    agent = get_agent()
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
 
    messages = result["messages"]
    final_answer = messages[-1].content
 
    tool_calls_made = []
    for msg in messages:
        calls = getattr(msg, "tool_calls", None)
        if calls:
            for call in calls:
                tool_calls_made.append({"name": call.get("name"), "args": call.get("args")})
 
    return {
        "question": question,
        "answer": final_answer,
        "tool_calls": tool_calls_made,
    }
 
 
if __name__ == "__main__":
    # quick manual test, mirrors test_tool_calling.py but with the real loop
    test_questions = [
        "What risks is NVIDIA currently facing?",
        "What new growth opportunities does NVIDIA have?",
        "What is the capital of France?",
    ]
    for q in test_questions:
        print("=" * 70)
        print(f"QUESTION: {q}")
        print("=" * 70)
        result = ask(q)
        print(f"\nTool calls made: {result['tool_calls']}")
        print(f"\nFINAL ANSWER:\n{result['answer']}\n")
 