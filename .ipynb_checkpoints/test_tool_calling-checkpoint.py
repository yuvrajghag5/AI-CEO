
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
 
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
 
# Minimal fake versions of your real tools, just for this test.
# Real implementations live in agent/tools.py once this test passes.
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_risks",
            "description": "Find strategic risks facing NVIDIA. Use when the user asks about threats, regulatory issues, competition, or downside scenarios.",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "Optional topic to narrow the search"}
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_opportunities",
            "description": "Find strategic opportunities for NVIDIA. Use when the user asks about growth, new markets, or partnerships.",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "Optional topic to narrow the search"}
                },
                "required": [],
            },
        },
    },
]
 
TEST_QUESTIONS = [
    "What risks is NVIDIA currently facing?",
    "What new growth opportunities does NVIDIA have?",
    "What is the capital of France?",  # should call NO tool — sanity check
]
 
 
def main():
    print(f"Loading {MODEL_NAME} ... this can take a while on first run.\n")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype="auto",
        device_map="auto",
    )
 
    for question in TEST_QUESTIONS:
        print("=" * 70)
        print(f"QUESTION: {question}")
        print("=" * 70)
 
        messages = [{"role": "user", "content": question}]
 
        try:
            chat_text = tokenizer.apply_chat_template(
                messages,
                tools=TOOLS,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception as e:
            print(f"\n  FAILED: tokenizer.apply_chat_template() does not support tools=. Error: {e}")
            print("  -> This model/tokenizer version does not support native tool calling.")
            print("  -> Stop here: either upgrade transformers, or pick a different model.\n")
            return
 
        inputs = tokenizer(chat_text, return_tensors="pt").to(model.device)
 
        output_ids = model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=False,  # deterministic for this test
            pad_token_id=tokenizer.eos_token_id,
        )
 
        generated_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        raw_output = tokenizer.decode(generated_tokens, skip_special_tokens=False)
 
        print("\nRAW MODEL OUTPUT:")
        print(raw_output)
 
        # Heuristic check: does it look like a structured tool call?
        looks_like_tool_call = "tool_call" in raw_output.lower() or '"name"' in raw_output
        print(f"\n  Looks like a structured tool call: {looks_like_tool_call}")
        print()
 
    print("=" * 70)
    print("DONE. Read the RAW MODEL OUTPUT blocks above:")
    print("  - For the risk/opportunity questions: did it emit something")
    print("    like <tool_call>{\"name\": \"get_risks\", \"arguments\": {}}</tool_call>")
    print("    or similar structured JSON, rather than just answering in prose?")
    print("  - For the France question: did it correctly NOT call a tool?")
    print("If yes to both -> Mistral v0.3 + plain transformers works for your agent.")
    print("If it just answers in prose every time, or apply_chat_template failed")
    print("above -> you need a serving layer (vLLM/TGI) or a different model.")
    print("=" * 70)
 
 
if __name__ == "__main__":
    main()
 