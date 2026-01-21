# Copyright 2025 Tencent wechat. All rights reserved.
# Modified to add Jacobi iteration support.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Jacobi-enabled Sampler module for WeDLM decoding.

This module extends the standard WeDLM sampler with Jacobi iteration support:
- Fixed Gumbel noise generation for deterministic token selection
- Mismatch detection between iterations
- Convergence tracking

Key modifications from standard sampler:
1. Fixed Gumbel noise - generated once and reused throughout decoding
2. Modified token selection - argmax(logits + temp * gumbel_noise)
3. Mismatch detection - track when previously selected tokens change
"""

import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass


@dataclass
class JacobiState:
    """State for Jacobi iteration tracking.

    Attributes:
        gumbel_noise: Fixed Gumbel noise for deterministic selection
        prev_tokens: Previously selected tokens (for mismatch detection)
        iteration: Current iteration count
        converged: Whether the sequence has converged
    """
    gumbel_noise: Optional[torch.Tensor] = None
    prev_tokens: Optional[torch.Tensor] = None
    iteration: int = 0
    converged: bool = False
    retry_count: int = 0


class JacobiSampler:
    """Jacobi-enabled sampler for WeDLM decoding.

    This class extends the standard WeDLM sampler with Jacobi iteration:
    - Uses fixed Gumbel noise for deterministic token selection
    - Detects mismatches between iterations
    - Tracks convergence

    The key insight: with fixed Gumbel noise, the same logits will always
    produce the same token selection, enabling convergence detection.
    """

    def __init__(
        self,
        max_window_size: int = 64,
        vocab_size: int = 152064,  # Qwen vocab size
        max_retries: int = 3,
        seed: Optional[int] = None,
    ):
        """Initialize the Jacobi Sampler.

        Args:
            max_window_size: Maximum sliding window size
            vocab_size: Vocabulary size for Gumbel noise generation
            max_retries: Maximum retries on mismatch
            seed: Random seed for reproducibility
        """
        self.max_window_size = max_window_size
        self.vocab_size = vocab_size
        self.max_retries = max_retries
        self.seed = seed

        # Per-sequence Jacobi states
        self.states: Dict[int, JacobiState] = {}

    def _generate_gumbel_noise(
        self,
        shape: Tuple[int, ...],
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Generate Gumbel(0, 1) noise.

        Uses the inverse CDF method: G = -log(-log(U)) where U ~ Uniform(0,1)

        Args:
            shape: Shape of the noise tensor
            device: Device to create tensor on
            dtype: Data type of the tensor

        Returns:
            Gumbel noise tensor
        """
        uniform = torch.rand(shape, device=device, dtype=dtype)
        uniform = torch.clamp(uniform, min=1e-10, max=1.0 - 1e-10)
        return -torch.log(-torch.log(uniform))

    def init_jacobi_state(
        self,
        seq_id: int,
        window_size: int,
        device: torch.device,
    ) -> JacobiState:
        """Initialize Jacobi state for a sequence.

        Args:
            seq_id: Unique sequence identifier
            window_size: Current window size
            device: Device for tensor creation

        Returns:
            Initialized JacobiState
        """
        if self.seed is not None:
            torch.manual_seed(self.seed + seq_id)

        # Generate fixed Gumbel noise for this sequence
        gumbel_noise = self._generate_gumbel_noise(
            (self.max_window_size, self.vocab_size),
            device=device,
        )

        state = JacobiState(
            gumbel_noise=gumbel_noise,
            prev_tokens=None,
            iteration=0,
            converged=False,
            retry_count=0,
        )

        self.states[seq_id] = state
        return state

    def get_or_init_state(
        self,
        seq_id: int,
        window_size: int,
        device: torch.device,
    ) -> JacobiState:
        """Get existing state or initialize new one.

        Args:
            seq_id: Unique sequence identifier
            window_size: Current window size
            device: Device for tensor creation

        Returns:
            JacobiState for this sequence
        """
        if seq_id not in self.states:
            return self.init_jacobi_state(seq_id, window_size, device)
        return self.states[seq_id]

    def compute_entropy(self, logits: torch.Tensor) -> torch.Tensor:
        """Compute entropy for each position's probability distribution.

        Args:
            logits: Raw logits from the model, shape [num_positions, vocab_size]

        Returns:
            Entropy values for each position, shape [num_positions]
        """
        return torch.distributions.Categorical(logits=logits).entropy()

    def _apply_top_k(
        self,
        logits: torch.Tensor,
        top_k: int
    ) -> torch.Tensor:
        """Apply top-k filtering to logits."""
        if top_k <= 0:
            return logits

        top_k = min(top_k, logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k, dim=-1).values[..., -1, None]
        logits = logits.masked_fill(indices_to_remove, float('-inf'))
        return logits

    def _apply_top_p(
        self,
        logits: torch.Tensor,
        top_p: float
    ) -> torch.Tensor:
        """Apply top-p (nucleus) filtering to logits."""
        if top_p >= 1.0:
            return logits

        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False

        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=-1, index=sorted_indices, src=sorted_indices_to_remove
        )
        logits = logits.masked_fill(indices_to_remove, float('-inf'))
        return logits

    def sample_tokens_jacobi(
        self,
        logits: torch.Tensor,
        gumbel_noise: torch.Tensor,
        temperature: float,
        top_p: float = 1.0,
        top_k: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample tokens using fixed Gumbel noise for Jacobi iteration.

        Instead of fresh randomness, uses pre-generated Gumbel noise:
        token = argmax(logits + temperature * gumbel_noise)

        This ensures deterministic selection given the same logits.

        Args:
            logits: Raw logits, shape [num_positions, vocab_size]
            gumbel_noise: Fixed Gumbel noise, shape [num_positions, vocab_size]
            temperature: Sampling temperature. 0 means greedy decoding.
            top_p: Nucleus sampling threshold.
            top_k: Top-k sampling parameter.

        Returns:
            Tuple of (sampled_ids, greedy_ids)
        """
        # Compute greedy predictions
        probs = F.softmax(logits, dim=-1)
        greedy_ids = probs.argmax(dim=-1)

        if temperature == 0:
            return greedy_ids, greedy_ids

        # Apply filtering
        filtered_logits = self._apply_top_k(logits, top_k)
        filtered_logits = self._apply_top_p(filtered_logits, top_p)

        # Jacobi selection: argmax(logits + temp * gumbel_noise)
        # Use float64 for numerical stability
        logits_f64 = filtered_logits.to(torch.float64)
        noise_slice = gumbel_noise[:logits.size(0), :].to(torch.float64)
        perturbed = logits_f64 + temperature * noise_slice
        sampled_ids = torch.argmax(perturbed, dim=-1)

        return sampled_ids, greedy_ids

    def detect_mismatch(
        self,
        current_tokens: torch.Tensor,
        prev_tokens: Optional[torch.Tensor],
        mask_flags: List[bool],
    ) -> Tuple[bool, Optional[int]]:
        """Detect mismatch between current and previous token selections.

        A mismatch occurs when a previously non-masked position changes value.

        Args:
            current_tokens: Currently selected tokens
            prev_tokens: Previously selected tokens (None if first iteration)
            mask_flags: Which positions are still masked

        Returns:
            Tuple of (has_mismatch, first_mismatch_position)
        """
        if prev_tokens is None:
            return False, None

        # Check non-masked positions for changes
        for i, is_mask in enumerate(mask_flags):
            if not is_mask:  # Position was previously filled
                if i < len(current_tokens) and i < len(prev_tokens):
                    if current_tokens[i] != prev_tokens[i]:
                        return True, i

        return False, None

    def select_positions_to_fill(
        self,
        entropy: torch.Tensor,
        remaining_mask_indices: List[int],
        entropy_threshold: Optional[float],
        pos_penalty_factor: float
    ) -> List[int]:
        """Select mask positions to fill using entropy-based parallel decoding.

        Same as standard WeDLM sampler.
        """
        device = entropy.device

        mask_indices_tensor = torch.tensor(
            remaining_mask_indices, device=device, dtype=torch.float
        )

        base_pos = mask_indices_tensor[0]
        distances = mask_indices_tensor - base_pos
        position_penalty = distances * pos_penalty_factor

        adjusted_entropy = entropy + position_penalty

        if entropy_threshold is not None:
            candidates = (adjusted_entropy < entropy_threshold).nonzero(as_tuple=True)[0]
            if candidates.numel() > 0:
                return candidates.tolist()

        return [int(adjusted_entropy.argmin().item())]

    def process_mask_positions_jacobi(
        self,
        seq_id: int,
        mask_logits: torch.Tensor,
        remaining_mask_indices: List[int],
        window_tokens: List[int],
        window_mask_flags: List[bool],
        temperature: float,
        entropy_threshold: Optional[float],
        pos_penalty_factor: float,
        top_p: float = 1.0,
        top_k: int = 0,
    ) -> Tuple[List[int], List[int], bool, Optional[int]]:
        """Process mask positions with Jacobi iteration support.

        Extends the standard process_mask_positions with:
        - Fixed Gumbel noise for token selection
        - Mismatch detection
        - Retry mechanism

        Args:
            seq_id: Unique sequence identifier
            mask_logits: Logits for mask positions only
            remaining_mask_indices: Window indices of remaining mask positions
            window_tokens: Current window tokens
            window_mask_flags: Which positions are masked
            temperature: Sampling temperature
            entropy_threshold: Threshold for parallel position selection
            pos_penalty_factor: Position penalty factor
            top_p: Nucleus sampling threshold
            top_k: Top-k sampling parameter

        Returns:
            Tuple of (fill_indices, token_ids, has_mismatch, mismatch_pos)
        """
        if mask_logits.size(0) == 0:
            return [], [], False, None

        device = mask_logits.device

        # Get or initialize Jacobi state
        state = self.get_or_init_state(seq_id, len(window_tokens), device)

        # Get Gumbel noise slice for mask positions
        gumbel_slice = state.gumbel_noise[:mask_logits.size(0), :]

        # Step 1: Compute entropy for position selection
        entropy = self.compute_entropy(mask_logits)

        # Step 2: Sample tokens using fixed Gumbel noise
        sampled_ids, _ = self.sample_tokens_jacobi(
            mask_logits, gumbel_slice, temperature, top_p, top_k
        )

        # Step 3: Detect mismatch with previous iteration
        current_full = torch.tensor(window_tokens, device=device)
        has_mismatch, mismatch_pos = self.detect_mismatch(
            current_full, state.prev_tokens, window_mask_flags
        )

        # Step 4: Handle mismatch
        if has_mismatch and state.retry_count < self.max_retries:
            state.retry_count += 1
            # Return mismatch info for caller to handle re-masking
            return [], [], True, mismatch_pos

        # Step 5: Select positions to fill
        fill_indices = self.select_positions_to_fill(
            entropy,
            remaining_mask_indices,
            entropy_threshold,
            pos_penalty_factor
        )

        # Step 6: Get token IDs for selected positions
        token_ids = [int(sampled_ids[k].item()) for k in fill_indices]

        # Update state for next iteration
        state.prev_tokens = current_full.clone()
        state.iteration += 1
        state.retry_count = 0  # Reset retry count on successful step

        return fill_indices, token_ids, False, None

    def reset_state(self, seq_id: int) -> None:
        """Reset Jacobi state for a sequence.

        Call this when a sequence is finished or restarted.

        Args:
            seq_id: Sequence identifier to reset
        """
        if seq_id in self.states:
            del self.states[seq_id]

    def reset_all_states(self) -> None:
        """Reset all Jacobi states."""
        self.states.clear()
