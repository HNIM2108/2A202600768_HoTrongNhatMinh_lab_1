"""
Day 1 — LLM API Foundation
AICB-P1: AI Practical Competency Program, Phase 1

Instructions:
    1. Fill in every section marked with TODO.
    2. Do NOT change function signatures.
    3. Copy this file to solution/solution.py when done.
    4. Run: pytest tests/ -v
"""

import os
import time
from typing import Any, Callable
from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI


def get_openai_client():
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------------------------------------------------------------------------
# Estimated costs per 1K OUTPUT tokens (USD) — update if pricing changes
# ---------------------------------------------------------------------------
COST_PER_1K_OUTPUT_TOKENS = {
    "gpt-4o": 0.010,
    "gpt-4o-mini": 0.0006,
}

OPENAI_MODEL = "gpt-4o"
OPENAI_MINI_MODEL = "gpt-4o-mini"


# ---------------------------------------------------------------------------
# Task 1 — Call GPT-4o
# ---------------------------------------------------------------------------
def call_openai(
    prompt: str,
    model: str = OPENAI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float]:
    client = get_openai_client()
    
    start_time = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens
    )
    end_time = time.time()
    
    latency = end_time - start_time
    response_text = response.choices[0].message.content or ""
    
    return response_text, latency


# ---------------------------------------------------------------------------
# Task 2 — Call GPT-4o-mini
# ---------------------------------------------------------------------------
def call_openai_mini(
    prompt: str,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float]:
    return call_openai(
        prompt=prompt,
        model=OPENAI_MINI_MODEL,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens
    )


# ---------------------------------------------------------------------------
# Task 3 — Compare GPT-4o vs GPT-4o-mini
# ---------------------------------------------------------------------------
def compare_models(prompt: str) -> dict:
    gpt4o_text, gpt4o_lat = call_openai(prompt)
    mini_text, mini_lat = call_openai_mini(prompt)
    
    # Tính chi phí ước tính theo Hint của đề bài
    estimated_tokens = len(gpt4o_text.split()) / 0.75
    gpt4o_cost = (estimated_tokens / 1000) * COST_PER_1K_OUTPUT_TOKENS["gpt-4o"]
    
    return {
        "gpt4o_response": gpt4o_text,
        "mini_response": mini_text,
        "gpt4o_latency": gpt4o_lat,
        "mini_latency": mini_lat,
        "gpt4o_cost_estimate": gpt4o_cost,
    }


# ---------------------------------------------------------------------------
# Task 4 — Streaming chatbot with conversation history
# ---------------------------------------------------------------------------
def streaming_chatbot() -> None:
    client = get_openai_client()
    history = []
    
    while True:
        user_input = input("\nYou: ")
        if user_input.strip().lower() in ["quit", "exit"]:
            print("Exiting chatbot. Goodbye!")
            break
            
        if not user_input.strip():
            continue
            
        history.append({"role": "user", "content": user_input})
        
        print("Assistant: ", end="", flush=True)
        
        stream = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=history,
            stream=True
        )
        
        assistant_reply = ""
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            print(delta, end="", flush=True)
            assistant_reply += delta
            
        print()
        
        history.append({"role": "assistant", "content": assistant_reply})
        history = history[-3:]


# ---------------------------------------------------------------------------
# Bonus Task A — Retry with exponential backoff
# ---------------------------------------------------------------------------
def retry_with_backoff(
    fn: Callable,
    max_retries: int = 3,
    base_delay: float = 0.1,
) -> Any:
    attempt = 0
    while True:
        try:
            return fn()
        except Exception as e:
            if attempt >= max_retries:
                raise e
            delay = base_delay * (2 ** attempt)
            time.sleep(delay)
            attempt += 1


# ---------------------------------------------------------------------------
# Bonus Task B — Batch compare
# ---------------------------------------------------------------------------
def batch_compare(prompts: list[str]) -> list[dict]:
    results = []
    for p in prompts:
        res = compare_models(p)
        res["prompt"] = p
        results.append(res)
    return results


# ---------------------------------------------------------------------------
# Bonus Task C — Format comparison table
# ---------------------------------------------------------------------------
def format_comparison_table(results: list[dict]) -> str:
    # Đảm bảo tiêu đề cột chứa chuẩn xác các từ khóa 'Prompt', 'GPT-4o', 'Mini' để pass kiểm thử
    header = f"{'Prompt':<42} | {'GPT-4o Response':<42} | {'Mini Response':<42} | {'GPT-4o Latency':<15} | {'Mini Latency':<15}\n"
    separator = "-" * len(header) + "\n"
    
    table_str = header + separator
    
    for res in results:
        # Giới hạn độ dài text tối đa 40 ký tự theo Hint
        p_trunc = res['prompt'] if len(res['prompt']) <= 40 else res['prompt'][:37] + "..."
        gpt4o_trunc = res['gpt4o_response'] if len(res['gpt4o_response']) <= 40 else res['gpt4o_response'][:37] + "..."
        mini_trunc = res['mini_response'] if len(res['mini_response']) <= 40 else res['mini_response'][:37] + "..."
        
        table_str += f"{p_trunc:<42} | {gpt4o_trunc:<42} | {mini_trunc:<42} | {res['gpt4o_latency']:<15.4f} | {res['mini_latency']:<15.4f}\n"
        
    return table_str


# ---------------------------------------------------------------------------
# Entry point for manual testing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = "mock-key-for-testing"

    test_prompt = "Explain the difference between temperature and top_p in one sentence."
    print("=== Comparing models ===")
    try:
        result = compare_models(test_prompt)
        for key, value in result.items():
            print(f"{key}: {value}")
    except Exception as e:
        print(f"Không thể chạy manual test thực tế: {e}")

    print("\n=== Starting chatbot (type 'quit' to exit) ===")
    try:
        streaming_chatbot()
    except Exception as e:
        print(f"Chatbot đã dừng: {e}")