"""
Random Multi-Turn Dataset Plugin for EvalScope Perf Benchmark

This plugin generates random multi-turn conversation data for stress testing.

Usage:
    # Command line
    evalscope perf \
        --dataset random_multi_turn \
        --tokenizer-path Qwen/Qwen2.5-0.5B-Instruct \
        --extra-args '{"num_turns": 3}'
"""
import numpy as np
from typing import Dict, Iterator, List

from evalscope.perf.arguments import Arguments
from evalscope.perf.plugin.datasets.base import DatasetPluginBase
from evalscope.perf.plugin.registry import register_dataset


@register_dataset('random_multi_turn')
class RandomMultiTurnDatasetPlugin(DatasetPluginBase):
    """Generate random multi-turn conversation data for benchmarking.

    Each request contains a multi-turn conversation history ending with a user message.
    The simulator generates mock assistant responses for previous turns.

    Configuration via extra_args:
        - num_turns: Number of turns per conversation (default: 3)
    """

    def __init__(self, query_parameters: Arguments):
        assert query_parameters.tokenizer_path, \
            'Tokenizer path is required for random_multi_turn, please provide it with `--tokenizer-path`.'
        super().__init__(query_parameters)

        self.prefix_length = self.query_parameters.prefix_length
        self.prefix_ids = self.get_random_inputs(self.prefix_length)
        self.template_len = self.get_template_len()
        self.number = self.query_parameters.number or 1

        # Get num_turns from extra_args
        self.num_turns = 3  # Default 3 turns
        if self.query_parameters.extra_args and 'num_turns' in self.query_parameters.extra_args:
            self.num_turns = int(self.query_parameters.extra_args['num_turns'])

    def build_messages(self) -> Iterator[List[Dict]]:
        """Build multi-turn conversation messages.

        Yields:
            Iterator[List[Dict]]: A list of messages representing a multi-turn conversation.
        """
        min_prompt_length = self.query_parameters.min_prompt_length
        max_prompt_length = self.query_parameters.max_prompt_length

        if self.query_parameters.apply_chat_template:
            min_prompt_length = max(10, min_prompt_length - self.template_len)
            max_prompt_length = max_prompt_length - self.template_len + 1

        assert max_prompt_length >= min_prompt_length, \
            'max_prompt_length should be greater than or equal to min_prompt_length.'

        vocab_size = self.tokenizer.vocab_size

        for _ in range(self.number):
            messages = []

            # Generate multi-turn conversation
            for turn_idx in range(self.num_turns):
                # Calculate length for this turn
                # First turn uses full range, subsequent turns use smaller range
                if turn_idx == 0:
                    turn_len = np.random.randint(min_prompt_length, max_prompt_length)
                else:
                    # Subsequent turns: shorter prompts to simulate follow-up questions
                    turn_min = max(10, min_prompt_length // 2)
                    turn_max = max(turn_min + 1, min_prompt_length)
                    turn_len = np.random.randint(turn_min, turn_max)

                # Generate random token sequence
                offset = np.random.randint(0, vocab_size)
                inner_seq = ((offset + turn_idx + np.arange(turn_len)) % vocab_size).tolist()
                token_sequence = self.prefix_ids + inner_seq
                prompt = self.tokenizer.decode(token_sequence)

                # Re-encode to ensure correct token count
                re_encoded = self.tokenizer.encode(prompt, add_special_tokens=False)[:self.prefix_length + turn_len]
                prompt = self.tokenizer.decode(re_encoded)

                # Add user message
                messages.append({'role': 'user', 'content': prompt})

                # For all turns except the last, add a mock assistant response
                # This simulates the conversation history
                if turn_idx < self.num_turns - 1:
                    assistant_response = self._generate_mock_response(turn_idx)
                    messages.append({'role': 'assistant', 'content': assistant_response})

            yield messages

    def _generate_mock_response(self, turn_idx: int) -> str:
        """Generate a mock assistant response for conversation history.

        Args:
            turn_idx: The turn index in the conversation.

        Returns:
            str: A mock response string.
        """
        vocab_size = self.tokenizer.vocab_size
        # Response length varies by turn (shorter for later turns)
        resp_len = np.random.randint(50, 150 - turn_idx * 20)
        resp_tokens = np.random.randint(0, vocab_size, size=max(20, resp_len)).tolist()
        return self.tokenizer.decode(resp_tokens)

    def get_random_inputs(self, length: int) -> List[int]:
        """Generate random input token IDs.

        Args:
            length: Number of tokens to generate.

        Returns:
            List[int]: List of random token IDs.
        """
        if length <= 0:
            return []
        return np.random.randint(0, self.tokenizer.vocab_size, size=length).tolist()

    def get_template_len(self) -> int:
        """Get the length of chat template overhead.

        Returns:
            int: Number of tokens added by chat template.
        """
        empty_message = [self.create_message(text='')]
        template = self.tokenizer.apply_chat_template(empty_message, tokenize=True, add_generation_prompt=True)
        return len(template)
