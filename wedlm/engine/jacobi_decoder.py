# Copyright 2025 Tencent wechat. All rights reserved.
# Modified to add Jacobi iteration support.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# ...

"""
Jacobi-enabled WeDLM Decoder module.

This module extends the standard WeDLM decoder with Jacobi iteration support:
- Uses JacobiSampler for deterministic token selection
- Handles mismatch detection and re-masking
- Tracks convergence across iterations

The decoding flow is:
1. initialize_states() - Create WeDLMState and JacobiState for new sequences
2. prepare_decode_inputs() - Prepare tensors for model forward pass
3. [ModelRunner runs the model]
4. process_decode_outputs_jacobi() - Process with Jacobi iteration logic
"""

import torch
from dataclasses import dataclass
from typing import List, Tuple, Optional

from wedlm.engine.sequence import Sequence, WeDLMState
from wedlm.engine.jacobi_sampler import JacobiSampler
from wedlm.engine.wedlm_decoder import WeDLMDecoder, DecodeContext, PreparedDecodeInputs


class JacobiWeDLMDecoder(WeDLMDecoder):
    """Jacobi-enabled WeDLM Decoder.

    Extends WeDLMDecoder with Jacobi iteration support for more
    deterministic and potentially faster convergence.

    Key differences from standard decoder:
    1. Uses JacobiSampler instead of standard Sampler
    2. Implements mismatch detection and re-masking
    3. Tracks convergence state per sequence
    """

    def __init__(
        self,
        mask_token_id: int,
        block_size: int,
        wedlm_window_size: int,
        vocab_size: int = 152064,
        max_retries: int = 3,
        seed: Optional[int] = None,
    ):
        """Initialize the Jacobi WeDLM decoder.

        Args:
            mask_token_id: Token ID used for mask positions
            block_size: KV cache block size
            wedlm_window_size: Maximum size of the sliding window
            vocab_size: Vocabulary size for Gumbel noise
            max_retries: Maximum retries on mismatch
            seed: Random seed for reproducibility
        """
        # Initialize Jacobi sampler
        self.jacobi_sampler = JacobiSampler(
            max_window_size=wedlm_window_size,
            vocab_size=vocab_size,
            max_retries=max_retries,
            seed=seed,
        )

        # Call parent init with Jacobi sampler
        # Note: We override process_decode_outputs to use Jacobi logic
        self.mask_token_id = mask_token_id
        self.block_size = block_size
        self.wedlm_window_size = wedlm_window_size

    def _handle_mismatch(
        self,
        seq: Sequence,
        state: WeDLMState,
        mismatch_pos: int
    ) -> None:
        """Handle mismatch by re-masking from mismatch position.

        When a previously filled position changes value, we re-mask
        from that position onwards and retry.

        Args:
            seq: Sequence being processed
            state: Current WeDLM state (modified in place)
            mismatch_pos: Position where mismatch occurred
        """
        # Re-mask from mismatch position to end of window
        for i in range(mismatch_pos, len(state.window_tokens)):
            state.window_tokens[i] = self.mask_token_id
            state.window_mask_flags[i] = True

    def process_decode_outputs_jacobi(
        self,
        seqs: List[Sequence],
        prepared: PreparedDecodeInputs,
        logits: torch.Tensor,
    ) -> List[Optional[List[int]]]:
        """Process model outputs with Jacobi iteration logic.

        Extends the standard process_decode_outputs with:
        - Jacobi token selection using fixed Gumbel noise
        - Mismatch detection and re-masking
        - Convergence tracking

        Args:
            seqs: All sequences (for result indexing)
            prepared: Prepared decode inputs
            logits: Model output logits

        Returns:
            List of generated tokens for each sequence (None if no tokens or mismatch)
        """
        step_results: List[Optional[List[int]]] = [None for _ in seqs]

        active_seqs = prepared.active_seqs
        active_states = prepared.active_states
        active_indices = prepared.active_indices
        per_seq_num_non_mask = prepared.per_seq_num_non_mask

        # Calculate row offsets
        row_offsets = []
        total_rows = 0
        for state in active_states:
            row_offsets.append(total_rows)
            total_rows += len(state.window_tokens)

        for j, (seq, state, orig_idx) in enumerate(
            zip(active_seqs, active_states, active_indices)
        ):
            window_size = len(state.window_tokens)
            num_non_mask = per_seq_num_non_mask[j]
            row_start = row_offsets[j]

            seq_logits = logits[row_start : row_start + window_size]

            # Find prefix tokens to prune
            mask_indices = [
                i for i, flag in enumerate(state.window_mask_flags) if flag
            ]
            prune_count = mask_indices[0] if mask_indices else window_size

            # Process pruned tokens (same as standard)
            pruned_tokens = self._process_pruned_tokens(seq, state, prune_count)

            # Process mask positions with Jacobi
            remaining_mask_indices = [
                i for i, flag in enumerate(state.window_mask_flags) if flag
            ]

            if remaining_mask_indices and not state.is_finished:
                mask_logits = seq_logits[num_non_mask:]

                if mask_logits.size(0) > 0:
                    # Use Jacobi sampler
                    fill_indices, token_ids, has_mismatch, mismatch_pos = \
                        self.jacobi_sampler.process_mask_positions_jacobi(
                            seq_id=orig_idx,
                            mask_logits=mask_logits,
                            remaining_mask_indices=remaining_mask_indices,
                            window_tokens=state.window_tokens,
                            window_mask_flags=state.window_mask_flags,
                            temperature=seq.temperature,
                            entropy_threshold=seq.wedlm_entropy_threshold,
                            pos_penalty_factor=seq.wedlm_pos_penalty_factor,
                            top_p=seq.top_p,
                            top_k=seq.top_k,
                        )

                    # Handle mismatch
                    if has_mismatch and mismatch_pos is not None:
                        self._handle_mismatch(seq, state, mismatch_pos)
                        # Don't return tokens on mismatch - will retry
                        step_results[orig_idx] = None
                        continue

                    # Fill selected positions
                    for k, token_id in zip(fill_indices, token_ids):
                        if k < len(remaining_mask_indices):
                            target_pos = remaining_mask_indices[k]
                            if target_pos < len(state.window_tokens):
                                state.window_tokens[target_pos] = token_id
                                state.window_mask_flags[target_pos] = False

            step_results[orig_idx] = pruned_tokens if pruned_tokens else None

        return step_results

    def reset_jacobi_state(self, seq_id: int) -> None:
        """Reset Jacobi state for a finished sequence.

        Args:
            seq_id: Sequence identifier
        """
        self.jacobi_sampler.reset_state(seq_id)

    def reset_all_jacobi_states(self) -> None:
        """Reset all Jacobi states."""
        self.jacobi_sampler.reset_all_states()


def create_jacobi_decoder(
    mask_token_id: int,
    block_size: int,
    wedlm_window_size: int,
    vocab_size: int = 152064,
    max_retries: int = 3,
    seed: Optional[int] = None,
) -> JacobiWeDLMDecoder:
    """Factory function to create a Jacobi-enabled WeDLM decoder.

    Args:
        mask_token_id: Token ID for mask
        block_size: KV cache block size
        wedlm_window_size: Sliding window size
        vocab_size: Vocabulary size
        max_retries: Max retries on mismatch
        seed: Random seed

    Returns:
        Configured JacobiWeDLMDecoder instance
    """
    return JacobiWeDLMDecoder(
        mask_token_id=mask_token_id,
        block_size=block_size,
        wedlm_window_size=wedlm_window_size,
        vocab_size=vocab_size,
        max_retries=max_retries,
        seed=seed,
    )
