#!/usr/bin/env python3
"""
WeDLM with Jacobi iteration example.

This script demonstrates using WeDLM with Jacobi-enabled decoding
for more deterministic and potentially faster parallel decoding.

Usage:
    python example_jacobi.py --model tencent/WeDLM-8B-Instruct
    python example_jacobi.py --model tencent/WeDLM-8B-Instruct --compare  # Compare with standard
"""

import argparse
import time
from transformers import AutoTokenizer

from wedlm import LLM, SamplingParams


def run_generation(llm, tokenizer, prompt, method_name=""):
    """Run generation and print results with timing."""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    start_time = time.perf_counter()
    outputs = llm.generate([text], SamplingParams(temperature=0.0, max_tokens=512))
    end_time = time.perf_counter()

    result = outputs[0]
    elapsed = end_time - start_time
    num_tokens = len(result["token_ids"])
    tps = num_tokens / elapsed if elapsed > 0 else 0

    print(f"\n{'='*60}")
    print(f"Method: {method_name}")
    print(f"{'='*60}")
    print(f"Prompt: {prompt[:100]}...")
    print(f"\nResponse: {result['text'][:500]}...")
    print(f"\n{'='*60}")
    print(f"  Generated tokens: {num_tokens}")
    print(f"  Time elapsed: {elapsed:.2f} seconds")
    print(f"  Speed: {tps:.2f} tokens/s")
    if "stats" in result:
        stats = result["stats"]
        print(f"  Decode forwards: {stats.get('decode_forwards', 'N/A')}")
        print(f"  Tokens per forward: {stats.get('tokens_per_forward', 'N/A'):.2f}")
    print(f"{'='*60}\n")

    return result, elapsed, tps


def main():
    parser = argparse.ArgumentParser(description="WeDLM Jacobi Example")
    parser.add_argument(
        "--model",
        type=str,
        default="tencent/WeDLM-8B-Instruct",
        help="Model path or HuggingFace model ID",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare Jacobi vs standard decoding",
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=3,
        help="Max retries on mismatch for Jacobi",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for Jacobi (for reproducibility)",
    )
    args = parser.parse_args()

    # Test prompt
    prompt = """A store sells apples for $2 each and oranges for $3 each.
Tom buys 5 apples and 4 oranges. How much does Tom spend in total?
Show your step-by-step reasoning."""

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    if args.compare:
        # Run both standard and Jacobi for comparison
        print("\n" + "="*60)
        print("COMPARISON: Standard WeDLM vs Jacobi WeDLM")
        print("="*60)

        # Standard WeDLM
        print("\nInitializing Standard WeDLM...")
        llm_standard = LLM(model=args.model, use_jacobi=False)
        _, time_standard, tps_standard = run_generation(
            llm_standard, tokenizer, prompt, "Standard WeDLM"
        )
        del llm_standard

        # Jacobi WeDLM
        print("\nInitializing Jacobi WeDLM...")
        llm_jacobi = LLM(
            model=args.model,
            use_jacobi=True,
            jacobi_max_retries=args.max_retries,
            jacobi_seed=args.seed,
        )
        _, time_jacobi, tps_jacobi = run_generation(
            llm_jacobi, tokenizer, prompt, "Jacobi WeDLM"
        )

        # Summary
        print("\n" + "="*60)
        print("COMPARISON SUMMARY")
        print("="*60)
        print(f"Standard WeDLM: {tps_standard:.2f} tokens/s ({time_standard:.2f}s)")
        print(f"Jacobi WeDLM:   {tps_jacobi:.2f} tokens/s ({time_jacobi:.2f}s)")
        speedup = tps_jacobi / tps_standard if tps_standard > 0 else 0
        print(f"Speedup: {speedup:.2f}x")
        print("="*60 + "\n")

    else:
        # Run Jacobi only
        print("\nInitializing WeDLM with Jacobi decoding...")
        llm = LLM(
            model=args.model,
            use_jacobi=True,
            jacobi_max_retries=args.max_retries,
            jacobi_seed=args.seed,
        )

        run_generation(llm, tokenizer, prompt, "Jacobi WeDLM")


if __name__ == "__main__":
    main()
