"""
Multi-Turn Conversation Stress Test Script using EvalScope

This script demonstrates how to run a stress test with simulated multi-turn conversations.
It uses the random_multi_turn dataset plugin to generate conversation history.

Usage:
    # Simple test
    python run_multi_turn_benchmark.py --simple

    # Custom test
    python run_multi_turn_benchmark.py --url http://127.0.0.1:8801/v1/chat/completions \\
        --model Qwen2.5-0.5B-Instruct \\
        --parallel 1 5 10 \\
        --number 10 50 100 \\
        --turns 3

    # With custom tokenizer
    python run_multi_turn_benchmark.py \\
        --tokenizer Qwen/Qwen2.5-0.5B-Instruct \\
        --turns 5
"""
import sys
import os

# Add current directory to path for custom plugin import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import and register the custom dataset plugin (import triggers registration)
import custom_multi_turn_dataset  # noqa: F401

from evalscope.perf.arguments import Arguments
from evalscope.perf.main import run_perf_benchmark


def run_simple_test():
    """Run a simple multi-turn test with minimal configuration."""

    args = Arguments(
        model='Qwen2.5-0.5B-Instruct',
        url='http://127.0.0.1:8801/v1/chat/completions',
        api='openai',
        dataset='random_multi_turn',
        tokenizer_path='Qwen/Qwen2.5-0.5B-Instruct',
        parallel=1,
        number=5,
        min_prompt_length=64,
        max_prompt_length=256,
        max_tokens=128,
        stream=True,
        extra_args={'num_turns': 3},
        debug=True,
    )

    print("=" * 60)
    print("Running Simple Multi-Turn Stress Test")
    print("=" * 60)
    run_perf_benchmark(args)


def run_full_test(url: str, model: str, tokenizer: str, parallel: list, number: list, turns: int):
    """Run a full multi-turn stress test with custom configuration."""

    args = Arguments(
        model=model,
        url=url,
        api='openai',
        dataset='random_multi_turn',
        tokenizer_path=tokenizer,
        parallel=parallel,
        number=number,
        min_prompt_length=128,
        max_prompt_length=512,
        prefix_length=0,
        max_tokens=256,
        min_tokens=64,
        temperature=0.7,
        top_p=0.9,
        stream=True,
        extra_args={'num_turns': turns},
        debug=True,
    )

    print("=" * 60)
    print("Multi-Turn Conversation Stress Test Configuration:")
    print(f"  Model: {model}")
    print(f"  URL: {url}")
    print(f"  Tokenizer: {tokenizer}")
    print(f"  Dataset: random_multi_turn")
    print(f"  Concurrency levels: {parallel}")
    print(f"  Requests per level: {number}")
    print(f"  Turns per conversation: {turns}")
    print("=" * 60)

    run_perf_benchmark(args)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Multi-Turn Conversation Stress Test for EvalScope',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Simple test with default settings
  python run_multi_turn_benchmark.py --simple

  # Custom test with 5 turns per conversation
  python run_multi_turn_benchmark.py --turns 5

  # Multiple concurrency levels
  python run_multi_turn_benchmark.py --parallel 1 10 50 --number 10 100 500
        """
    )

    parser.add_argument('--simple', action='store_true',
                        help='Run a simple test with default settings')
    parser.add_argument('--model', type=str, default='Qwen2.5-0.5B-Instruct',
                        help='Model name (default: Qwen2.5-0.5B-Instruct)')
    parser.add_argument('--url', type=str, default='http://127.0.0.1:8801/v1/chat/completions',
                        help='API endpoint URL')
    parser.add_argument('--tokenizer', type=str, default='Qwen/Qwen2.5-0.5B-Instruct',
                        help='Tokenizer path or model ID')
    parser.add_argument('--parallel', type=int, nargs='+', default=[1, 5, 10],
                        help='Concurrency levels to test (space-separated)')
    parser.add_argument('--number', type=int, nargs='+', default=[10, 50, 100],
                        help='Number of requests per concurrency level')
    parser.add_argument('--turns', type=int, default=3,
                        help='Number of turns per conversation (default: 3)')
    parser.add_argument('--min-prompt', type=int, default=128,
                        help='Minimum prompt length per turn')
    parser.add_argument('--max-prompt', type=int, default=512,
                        help='Maximum prompt length per turn')
    parser.add_argument('--max-tokens', type=int, default=256,
                        help='Maximum output tokens')

    parsed_args = parser.parse_args()

    if parsed_args.simple:
        run_simple_test()
    else:
        run_full_test(
            url=parsed_args.url,
            model=parsed_args.model,
            tokenizer=parsed_args.tokenizer,
            parallel=parsed_args.parallel,
            number=parsed_args.number,
            turns=parsed_args.turns,
        )


if __name__ == '__main__':
    main()
