
import chromadb
from transformers import AutoModelForCausalLM, AutoTokenizer
from rag.prompt import RAG_PROMPT
from config.settings import TOP_K, MODEL, MAX_NEW_TOKENS_RAG, NO_REPEAT_NGRAM_SIZE, REPETITION_PENALTY, DO_SAMPLE, TOP_P, TEMPERATURE
from config.paths import VECTORDB

CHROMA_DIR = VECTORDB / "./chroma_db"
COLLECTION_NAME = "ai_ceo_documents"
MODEL_NAME = MODEL   
TOP_K = TOP_K
MAX_NEW_TOKENS = MAX_NEW_TOKENS_RAG
NO_REPEAT_NGRAM_SIZE = NO_REPEAT_NGRAM_SIZE
REPETITION_PENALTY = REPETITION_PENALTY
DO_SAMPLE = DO_SAMPLE
TOP_P = TOP_P
TEMPERATURE = TEMPERATURE

memory = []
def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_or_create_collection(COLLECTION_NAME)
 
 
def load_model():
    print(f"Loading {MODEL_NAME} ... (this can take a while on first run)")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype="auto",
        device_map="auto",
    )
    return tokenizer, model
 
 
def retrieve_context(collection, question, top_k=TOP_K):
    results = collection.query(query_texts=[question], n_results=top_k)
 
    chunks = []
    for i in range(len(results["ids"][0])):
        meta = results["metadatas"][0][i]
        text = results["documents"][0][i]
        chunks.append(f"[{meta.get('source')} | {meta.get('title')}]\n{text}")
 
    return "\n\n---\n\n".join(chunks)
 
 
def format_history(memory, max_turns=3):
    """Use only the last few turns to keep the prompt short."""
    if not memory:
        return "(no previous conversation)"
    recent = memory[-max_turns:]
    return "\n".join(f"Q: {turn['question']}\nA: {turn['answer']}" for turn in recent)
 
 
def generate_response(prompt_text, tokenizer, model):
    messages = [{"role": "user", "content": prompt_text}]
    chat_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
 
    inputs = tokenizer(chat_text, return_tensors="pt").to(model.device)
 
    output_ids = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample= DO_SAMPLE,
        temperature = TEMPERATURE,
        top_p = TOP_P,
        repetition_penalty=REPETITION_PENALTY,
        no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
        pad_token_id=tokenizer.eos_token_id,
    )
 
    generated_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True)
 
 
def answer_question(question, tokenizer, model, collection):
    context = retrieve_context(collection, question)
    history = format_history(memory)
 
    prompt_text = RAG_PROMPT.format(history=history, context=context, question=question)
    answer = generate_response(prompt_text, tokenizer, model).strip()
 
    memory.append({"question": question, "answer": answer})
    return answer
 
 
def main():
    collection = get_collection()
    if collection.count() == 0:
        print("ChromaDB collection is empty. Run store.py first.")
        return
 
    tokenizer, model = load_model()
 
    print("RAG chat ready. Type 'exit' to quit.\n")
    while True:
        question = "If you were the CEO of NVIDIA today, what would you do next and why?"   #input("You: ").strip()
        if question.lower() in ("exit", "quit"):
            break
        if not question:
            continue
 
        answer = answer_question(question, tokenizer, model, collection)
        print(f"\nAgent: {answer}\n")
 
 
if __name__ == "__main__":
    main()