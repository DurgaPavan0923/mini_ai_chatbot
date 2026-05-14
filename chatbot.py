"""
Mini AI Chatbot — Transformer-powered with multi-model support
Supports: DialoGPT-medium, DialoGPT-large (or fallback models)
Features: conversation history, configurable generation params, model benchmarking
"""

import time
import json
import sys
import textwrap
from dataclasses import dataclass, field, asdict
from typing import Optional
import warnings
warnings.filterwarnings("ignore")

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    set_seed,
)


# ── Generation configuration ────────────────────────────────────────────────

@dataclass
class GenerationConfig:
    max_new_tokens: int = 150
    temperature: float = 0.75
    top_p: float = 0.92
    top_k: int = 50
    repetition_penalty: float = 1.3
    do_sample: bool = True

    def describe(self) -> str:
        return (
            f"max_new_tokens={self.max_new_tokens}, "
            f"temperature={self.temperature}, "
            f"top_p={self.top_p}, "
            f"top_k={self.top_k}, "
            f"repetition_penalty={self.repetition_penalty}"
        )


# ── Conversation history ─────────────────────────────────────────────────────

@dataclass
class Message:
    role: str       # "user" or "assistant"
    content: str
    timestamp: float = field(default_factory=time.time)


class ConversationHistory:
    """Maintains a rolling window of messages for context."""

    def __init__(self, max_turns: int = 6):
        self.messages: list[Message] = []
        self.max_turns = max_turns          # each turn = 1 user + 1 assistant

    def add(self, role: str, content: str):
        self.messages.append(Message(role=role, content=content))
        # Keep only the last max_turns * 2 messages
        self.messages = self.messages[-(self.max_turns * 2):]

    def clear(self):
        self.messages.clear()
        print("  [History cleared]\n")

    def display(self):
        if not self.messages:
            print("  (no history yet)\n")
            return
        for m in self.messages:
            prefix = "You" if m.role == "user" else "Bot"
            print(f"  [{prefix}] {m.content}")
        print()

    def to_dialogpt_input(self, tokenizer) -> torch.Tensor:
        """
        Build the token tensor expected by DialoGPT:
        user_tokens <eos> bot_tokens <eos> user_tokens <eos> …
        """
        ids = []
        for msg in self.messages:
            tokens = tokenizer.encode(msg.content + tokenizer.eos_token)
            ids.extend(tokens)
        return torch.tensor([ids], dtype=torch.long)


# ── Model wrapper ─────────────────────────────────────────────────────────────

class ChatModel:
    """Wraps a HuggingFace causal-LM for conversational use."""

    def __init__(self, model_id: str, display_name: str, style: str = "dialogpt"):
        self.model_id = model_id
        self.display_name = display_name
        self.style = style          # "dialogpt" or "causal"
        self.model = None
        self.tokenizer = None
        self.load_time: float = 0.0
        self.response_times: list[float] = []

    # ── Loading ──────────────────────────────────────────────────────────────

    def load(self):
        print(f"  Loading {self.display_name} …", end=" ", flush=True)
        t0 = time.time()

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.float32,
        )
        self.model.eval()

        # DialoGPT models don't have a pad token by default
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.load_time = time.time() - t0
        params = sum(p.numel() for p in self.model.parameters()) / 1e6
        print(f"done ({params:.0f}M params, {self.load_time:.1f}s)")

    # ── Generation ───────────────────────────────────────────────────────────

    def generate(
        self,
        user_input: str,
        history: ConversationHistory,
        config: GenerationConfig,
    ) -> tuple[str, float]:
        """
        Returns (response_text, elapsed_seconds).
        Adds the user message to history *before* generating,
        then appends the bot reply afterward.
        """
        history.add("user", user_input)

        t0 = time.time()

        if self.style == "dialogpt":
            response = self._generate_dialogpt(history, config)
        else:
            response = self._generate_causal(history, config)

        elapsed = time.time() - t0
        self.response_times.append(elapsed)

        history.add("assistant", response)
        return response, elapsed

    def _generate_dialogpt(
        self, history: ConversationHistory, config: GenerationConfig
    ) -> str:
        input_ids = history.to_dialogpt_input(self.tokenizer)

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
                repetition_penalty=config.repetition_penalty,
                do_sample=config.do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only the newly generated tokens
        new_ids = output_ids[:, input_ids.shape[-1]:]
        response = self.tokenizer.decode(new_ids[0], skip_special_tokens=True).strip()
        return response if response else "(no response)"

    def _generate_causal(
        self, history: ConversationHistory, config: GenerationConfig
    ) -> str:
        # Build a simple prompt from recent history
        prompt_lines = []
        for m in history.messages[:-1]:           # exclude the last user msg (added separately)
            tag = "Human" if m.role == "user" else "Assistant"
            prompt_lines.append(f"{tag}: {m.content}")
        # Add current user message
        last_user = history.messages[-1]
        prompt_lines.append(f"Human: {last_user.content}")
        prompt_lines.append("Assistant:")
        prompt = "\n".join(prompt_lines)

        inputs = self.tokenizer(prompt, return_tensors="pt")

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
                repetition_penalty=config.repetition_penalty,
                do_sample=config.do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        new_ids = output_ids[:, inputs["input_ids"].shape[-1]:]
        response = self.tokenizer.decode(new_ids[0], skip_special_tokens=True).strip()
        # Stop at the next "Human:" marker if the model generated one
        if "Human:" in response:
            response = response.split("Human:")[0].strip()
        return response if response else "(no response)"

    # ── Stats ────────────────────────────────────────────────────────────────

    def avg_response_time(self) -> float:
        return sum(self.response_times) / len(self.response_times) if self.response_times else 0.0

    def stats(self) -> dict:
        return {
            "model": self.display_name,
            "model_id": self.model_id,
            "load_time_s": round(self.load_time, 2),
            "num_responses": len(self.response_times),
            "avg_response_time_s": round(self.avg_response_time(), 2),
            "min_response_time_s": round(min(self.response_times), 2) if self.response_times else 0,
            "max_response_time_s": round(max(self.response_times), 2) if self.response_times else 0,
        }


# ── Chatbot shell ─────────────────────────────────────────────────────────────

MODELS = {
    "1": ChatModel(
        "microsoft/DialoGPT-medium",
        "DialoGPT-medium (345M)",
        style="dialogpt",
    ),
    "2": ChatModel(
        "microsoft/DialoGPT-large",
        "DialoGPT-large (762M)",
        style="dialogpt",
    ),
}

COMMANDS = {
    "/help":    "Show this help text",
    "/history": "Display conversation history",
    "/clear":   "Clear conversation history",
    "/config":  "Show current generation settings",
    "/set":     "Set a parameter  e.g. /set temperature 0.9",
    "/switch":  "Switch model  e.g. /switch 2",
    "/compare": "Run benchmark comparison between models",
    "/stats":   "Show model performance statistics",
    "/quit":    "Exit the chatbot",
}


class ChatbotShell:

    def __init__(self):
        self.active_key = "1"
        self.history = ConversationHistory(max_turns=6)
        self.config = GenerationConfig()
        self._loaded: set[str] = set()

    # ── Helpers ──────────────────────────────────────────────────────────────

    @property
    def active_model(self) -> ChatModel:
        return MODELS[self.active_key]

    def ensure_loaded(self, key: str):
        if key not in self._loaded:
            MODELS[key].load()
            self._loaded.add(key)

    def _wrap(self, text: str, width: int = 78) -> str:
        return "\n".join(textwrap.wrap(text, width=width))

    # ── Commands ─────────────────────────────────────────────────────────────

    def cmd_help(self, _args):
        print("\n  ╔══ Commands ══════════════════════════════════════╗")
        for cmd, desc in COMMANDS.items():
            print(f"  ║  {cmd:<12} {desc}")
        print("  ╚══════════════════════════════════════════════════╝\n")

    def cmd_history(self, _args):
        print("\n── Conversation history ──────────────────────────────")
        self.history.display()

    def cmd_clear(self, _args):
        self.history.clear()

    def cmd_config(self, _args):
        print(f"\n  {self.config.describe()}\n")

    def cmd_set(self, args):
        if len(args) != 2:
            print("  Usage: /set <param> <value>\n")
            return
        param, raw = args
        mapping = {
            "temperature": ("temperature", float),
            "max_new_tokens": ("max_new_tokens", int),
            "top_p": ("top_p", float),
            "top_k": ("top_k", int),
            "repetition_penalty": ("repetition_penalty", float),
        }
        if param not in mapping:
            print(f"  Unknown param. Choose from: {', '.join(mapping)}\n")
            return
        attr, cast = mapping[param]
        try:
            setattr(self.config, attr, cast(raw))
            print(f"  ✓ {param} = {getattr(self.config, attr)}\n")
        except ValueError:
            print(f"  ✗ Invalid value for {param}\n")

    def cmd_switch(self, args):
        key = args[0] if args else ""
        if key not in MODELS:
            print(f"  Available: {', '.join(f'{k}={m.display_name}' for k, m in MODELS.items())}\n")
            return
        self.active_key = key
        self.ensure_loaded(key)
        self.history.clear()
        print(f"  Switched to {self.active_model.display_name}\n")

    def cmd_compare(self, _args):
        """Run the same set of prompts through both models and print a table."""
        prompts = [
            "What is artificial intelligence?",
            "Tell me a short joke.",
            "How do computers work?",
        ]
        results: dict[str, list] = {k: [] for k in MODELS}

        print("\n  ╔══ Benchmark comparison ══════════════════════════╗")
        for key, model in MODELS.items():
            self.ensure_loaded(key)
            print(f"\n  ▶ Model: {model.display_name}")
            for prompt in prompts:
                h = ConversationHistory(max_turns=1)
                resp, elapsed = model.generate(prompt, h, self.config)
                results[key].append(elapsed)
                short = resp[:80] + "…" if len(resp) > 80 else resp
                print(f"    Q: {prompt[:60]}")
                print(f"    A: {short}")
                print(f"    ⏱  {elapsed:.2f}s\n")

        # Summary table
        print("  ┌─────────────────────────────┬──────────┬──────────┬──────────┐")
        print("  │ Model                       │  Avg (s) │  Min (s) │  Max (s) │")
        print("  ├─────────────────────────────┼──────────┼──────────┼──────────┤")
        for key, model in MODELS.items():
            times = results[key]
            avg = sum(times) / len(times)
            print(f"  │ {model.display_name:<27} │ {avg:>8.2f} │ {min(times):>8.2f} │ {max(times):>8.2f} │")
        print("  └─────────────────────────────┴──────────┴──────────┴──────────┘\n")

    def cmd_stats(self, _args):
        print("\n  ╔══ Performance statistics ════════════════════════╗")
        for key, model in MODELS.items():
            if key not in self._loaded:
                print(f"  ▷ {model.display_name}: not loaded yet")
                continue
            s = model.stats()
            print(f"  ▶ {s['model']}")
            print(f"    Load time      : {s['load_time_s']}s")
            print(f"    Responses      : {s['num_responses']}")
            print(f"    Avg resp time  : {s['avg_response_time_s']}s")
            print(f"    Min / Max      : {s['min_response_time_s']}s / {s['max_response_time_s']}s")
        print()

    def cmd_quit(self, _args):
        print("\n  Goodbye! 👋\n")
        sys.exit(0)

    # ── Dispatch ─────────────────────────────────────────────────────────────

    DISPATCH = {
        "/help":    cmd_help,
        "/history": cmd_history,
        "/clear":   cmd_clear,
        "/config":  cmd_config,
        "/set":     cmd_set,
        "/switch":  cmd_switch,
        "/compare": cmd_compare,
        "/stats":   cmd_stats,
        "/quit":    cmd_quit,
        "/exit":    cmd_quit,
    }

    def handle(self, raw: str):
        raw = raw.strip()
        if not raw:
            return
        if raw.startswith("/"):
            parts = raw.split()
            cmd, args = parts[0].lower(), parts[1:]
            if cmd in self.DISPATCH:
                self.DISPATCH[cmd](self, args)
            else:
                print(f"  Unknown command: {cmd}  (type /help)\n")
        else:
            self.ensure_loaded(self.active_key)
            print(f"\n  🤖 {self.active_model.display_name} is thinking…", end="\r")
            response, elapsed = self.active_model.generate(raw, self.history, self.config)
            print(f"  🤖 [{elapsed:.2f}s] {self._wrap(response)}\n")

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self):
        print("""
╔══════════════════════════════════════════════════════════════╗
║          Mini AI Chatbot  ·  Powered by HuggingFace          ║
╠══════════════════════════════════════════════════════════════╣
║  Models:  1 = DialoGPT-medium  |  2 = DialoGPT-large        ║
║  Type /help for commands, /compare to benchmark models       ║
╚══════════════════════════════════════════════════════════════╝
""")
        # Pre-load the default model
        self.ensure_loaded(self.active_key)
        print(f"\n  Active model: {self.active_model.display_name}")
        print(f"  Config: {self.config.describe()}\n")

        while True:
            try:
                user_input = input("You: ")
            except (EOFError, KeyboardInterrupt):
                self.cmd_quit(None)
            self.handle(user_input)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ChatbotShell().run()
