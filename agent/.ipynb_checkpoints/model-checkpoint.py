"""
Shared model loader.

Loads Mistral-7B-Instruct-v0.3 ONCE and serves it to both consumers:

  - the agent loop (Box 4) -> needs the raw model + tokenizer to call
    apply_chat_template(tools=...) and model.generate() directly.
  - the CEO briefing (Box 5) -> needs the SAME weights wrapped by
    outlines for schema-constrained generation.

get_base() and get_outlines_model() both build on the one _load_base()
call, so there's only ever one ~14.5GB load on the GPU.

Place at: agent/model.py
"""
from transformers import AutoModelForCausalLM, AutoTokenizer

from config.settings import MODEL

_tokenizer = None
_model = None
_outlines_model = None


def get_base():
    """Return (tokenizer, model), loading once and reusing after that."""
    global _tokenizer, _model
    if _model is None:
        print(f"Loading {MODEL} once ...")
        _tokenizer = AutoTokenizer.from_pretrained(MODEL)
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL,
            torch_dtype="auto",
            device_map="auto",
        )
    return _tokenizer, _model


def get_outlines_model():
    """outlines-wrapped model (SAME weights as get_base()) for
    schema-constrained CEO briefing generation. No second model load."""
    global _outlines_model
    if _outlines_model is None:
        import outlines
        tokenizer, model = get_base()
        _outlines_model = outlines.from_transformers(model, tokenizer)
    return _outlines_model