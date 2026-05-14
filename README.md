# Mini AI Chatbot — Transformer-Powered

A conversational chatbot built on HuggingFace Transformers, comparing
**DialoGPT-medium** (345M) vs **DialoGPT-large** (762M) with configurable
generation parameters and performance benchmarking.

---

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch the interactive chatbot
python chatbot.py

# 3. Run the automated benchmark
python benchmark.py
```

---

## Files

| File            | Purpose                                                  |
|-----------------|----------------------------------------------------------|
| `chatbot.py`    | Interactive chatbot shell with multi-model support       |
| `benchmark.py`  | Automated benchmark comparing both models                |
| `requirements.txt` | Python package dependencies                           |

---

## Interactive commands

| Command              | Description                                   |
|----------------------|-----------------------------------------------|
| `/help`              | Show all commands                             |
| `/history`           | Print conversation history                    |
| `/clear`             | Clear conversation context                    |
| `/config`            | Show current generation settings              |
| `/set <param> <val>` | Change a generation parameter                 |
| `/switch <1\|2>`     | Switch between DialoGPT-medium / large        |
| `/compare`           | Benchmark both models side-by-side            |
| `/stats`             | Show per-model performance statistics         |
| `/quit`              | Exit                                          |

### Tunable parameters via `/set`

| Parameter           | Default | Effect                                              |
|---------------------|---------|-----------------------------------------------------|
| `temperature`       | 0.75    | Higher → more creative/random; lower → more focused |
| `max_new_tokens`    | 150     | Maximum tokens generated per reply                  |
| `top_p`             | 0.92    | Nucleus sampling probability mass                   |
| `top_k`             | 50      | Vocabulary candidates considered per step           |
| `repetition_penalty`| 1.3     | Penalises repeated phrases (>1 = stronger)          |

Example:
```
You: /set temperature 0.5
You: /set max_new_tokens 200
```

---

## Architecture overview

```
User Input
    │
    ▼
ConversationHistory  ──── rolling window (default 6 turns)
    │
    ▼
ChatModel.generate()
    ├── DialoGPT style: concatenate [user<eos>bot<eos>…] tokens → generate
    └── Causal style:   Human:/Assistant: prompt template → generate
    │
    ▼
GenerationConfig  (temperature, top_p, top_k, repetition_penalty, max_new_tokens)
    │
    ▼
HuggingFace AutoModelForCausalLM  (DialoGPT-medium or DialoGPT-large)
    │
    ▼
Decoded response + elapsed time logged
```

---

## Model comparison

| Model              | Params | Avg RAM  | Speed (CPU) | Quality       |
|--------------------|--------|----------|-------------|---------------|
| DialoGPT-medium    | 345M   | ~700 MB  | ~2–5 s/turn | Good          |
| DialoGPT-large     | 762M   | ~1.5 GB  | ~5–12 s/turn| Better        |

**Recommendation:** Use DialoGPT-medium with `temperature=0.75` and
`repetition_penalty=1.3` for everyday use. Upgrade to DialoGPT-large
when response coherence matters more than speed.

---

## Generation parameter guide

| Setting         | Conservative | Balanced | Creative |
|-----------------|-------------|---------|---------|
| temperature     | 0.3         | 0.75    | 1.1     |
| max_new_tokens  | 80          | 120     | 150     |
| repetition_penalty | 1.5      | 1.3     | 1.1     |
| Effect          | Safe, repetitive | Best tradeoff | Diverse, sometimes incoherent |

---

## Adding more models

Edit the `MODELS` dict in `chatbot.py`:

```python
MODELS = {
    "1": ChatModel("microsoft/DialoGPT-medium", "DialoGPT-medium (345M)", style="dialogpt"),
    "2": ChatModel("microsoft/DialoGPT-large",  "DialoGPT-large (762M)",  style="dialogpt"),
    "3": ChatModel("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "TinyLlama (1.1B)", style="causal"),
}
```

Use `style="dialogpt"` for DialoGPT family models and `style="causal"` for
instruction-tuned / chat models that use a Human/Assistant prompt format.
