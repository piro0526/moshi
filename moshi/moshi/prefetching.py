# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Speech prefetching module for predicting future user and system speech.

This module provides functionality to generate multiple candidate futures
for both user speech and system responses, enabling speculative execution
and look-ahead prediction in dialogue systems.
"""

import logging
from dataclasses import dataclass, field

import torch

from .models.lm import LMGen, LMModel
from .models.compression import MimiModel

logger = logging.getLogger(__name__)


@dataclass
class PrefetchingConfig:
    """Configuration for speech prefetching.

    Args:
        num_candidates: Number of future candidates to generate.
        prediction_seconds: How far ahead to predict (in seconds).
        temperature_user: Temperature for sampling user speech variations.
        temperature_system: Temperature for sampling system speech.
        top_k_user: Top-k sampling parameter for user speech.
        top_k_system: Top-k sampling parameter for system speech.
        beam_search: If True, use beam search; else use independent sampling.
        device: Device to run inference on.
        use_silence_for_user: If True, use silence codes for future user input.
            Otherwise, sample continuations from the model.
    """
    num_candidates: int = 5
    prediction_seconds: float = 1.0
    temperature_user: float = 0.8
    temperature_system: float = 0.7
    top_k_user: int = 250
    top_k_system: int = 250
    beam_search: bool = False
    device: str = "cuda"
    use_silence_for_user: bool = True


@dataclass
class PrefetchingResult:
    """Result of prefetching operation containing multiple candidate futures.

    Attributes:
        user_codes: User speech codes [num_candidates, num_codebooks, num_timesteps]
        system_codes: System speech codes [num_candidates, num_codebooks, num_timesteps]
        user_audio: User audio waveforms [num_candidates, 1, audio_length]
        system_audio: System audio waveforms [num_candidates, 1, audio_length]
        text_tokens: Text tokens [num_candidates, num_timesteps]
        log_probs: Log probabilities for ranking [num_candidates]
        prediction_steps: Number of timesteps predicted
        frame_rate: Frame rate of the codec (Hz)
    """
    user_codes: torch.Tensor
    system_codes: torch.Tensor
    user_audio: torch.Tensor
    system_audio: torch.Tensor
    text_tokens: torch.Tensor
    log_probs: torch.Tensor
    prediction_steps: int
    frame_rate: float


@dataclass
class _Hypothesis:
    """Internal hypothesis tracking during beam search.

    Attributes:
        user_codes: Accumulated user speech codes
        system_codes: Accumulated system speech codes
        text_tokens: Accumulated text tokens
        log_prob: Cumulative log probability
    """
    user_codes: list[torch.Tensor] = field(default_factory=list)
    system_codes: list[torch.Tensor] = field(default_factory=list)
    text_tokens: list[torch.Tensor] = field(default_factory=list)
    log_prob: float = 0.0

    def clone(self) -> "_Hypothesis":
        """Create a deep copy of this hypothesis."""
        return _Hypothesis(
            user_codes=self.user_codes.copy(),
            system_codes=self.system_codes.copy(),
            text_tokens=self.text_tokens.copy(),
            log_prob=self.log_prob,
        )


class PrefetchingModule:
    """Module for prefetching future user and system speech.

    This module uses a streaming language model (Moshi) and audio codec (Mimi)
    to generate multiple candidate futures for both user speech continuation
    and system responses.

    Args:
        mimi: Audio codec model for encoding/decoding speech
        lm: Language model for generating speech tokens
        config: Configuration for prefetching behavior

    Example:
        >>> from moshi.models import loaders
        >>> from moshi.prefetching import PrefetchingModule, PrefetchingConfig
        >>>
        >>> mimi = loaders.get_mimi("path/to/mimi.safetensors", device="cuda")
        >>> lm = loaders.get_moshi_lm("path/to/moshi.safetensors", device="cuda")
        >>>
        >>> config = PrefetchingConfig(num_candidates=5, prediction_seconds=1.0)
        >>> prefetcher = PrefetchingModule(mimi, lm, config)
        >>>
        >>> user_audio = torch.randn(1, 1, 24000, device="cuda")
        >>> result = prefetcher.prefetch(user_audio)
        >>> print(f"Generated {result.user_audio.shape[0]} candidates")
    """

    def __init__(
        self,
        mimi: MimiModel,
        lm: LMModel,
        config: PrefetchingConfig,
    ):
        self.mimi = mimi
        self.lm = lm
        self.config = config
        self.device = torch.device(config.device)

        # Create LMGen wrapper for generation
        self.lm_gen = LMGen(
            lm,
            use_sampling=True,
            temp=config.temperature_system,
            temp_text=config.temperature_system,
            top_k=config.top_k_system,
            top_k_text=config.top_k_system,
        )

        logger.info(
            f"PrefetchingModule initialized: {config.num_candidates} candidates, "
            f"{config.prediction_seconds}s ahead, beam_search={config.beam_search}"
        )

    @torch.no_grad()
    def prefetch(self, user_audio: torch.Tensor) -> PrefetchingResult:
        """Prefetch future user and system speech from current user audio.

        Args:
            user_audio: User audio tensor [1, channels, audio_length].
                Must be a multiple of frame_size (1920 samples at 24kHz).

        Returns:
            PrefetchingResult containing multiple candidate futures.

        Raises:
            ValueError: If audio length is not a multiple of frame_size.
        """
        # Validate input
        if user_audio.shape[-1] % self.mimi.frame_size != 0:
            raise ValueError(
                f"Audio length {user_audio.shape[-1]} must be a multiple of "
                f"frame_size {self.mimi.frame_size}"
            )

        # Encode user audio to codes
        user_codes = self.mimi.encode(user_audio)  # [1, K, T]

        # Compute number of future steps to predict
        prediction_steps = int(self.config.prediction_seconds * self.mimi.frame_rate)

        logger.debug(
            f"Prefetching {prediction_steps} steps "
            f"({self.config.prediction_seconds}s at {self.mimi.frame_rate}Hz)"
        )

        # Generate candidates
        if self.config.beam_search:
            hypotheses = self._generate_with_beam_search(user_codes, prediction_steps)
        else:
            hypotheses = self._generate_with_independent_sampling(
                user_codes, prediction_steps
            )

        # Convert hypotheses to result format
        return self._hypotheses_to_result(hypotheses)

    def _generate_with_independent_sampling(
        self,
        user_codes: torch.Tensor,
        prediction_steps: int,
    ) -> list[_Hypothesis]:
        """Generate candidates using independent sampling (parallel candidates).

        Args:
            user_codes: Input user codes [1, K, T]
            prediction_steps: Number of future steps to predict

        Returns:
            List of hypotheses, one per candidate
        """
        hypotheses = []

        for candidate_idx in range(self.config.num_candidates):
            logger.debug(f"Generating candidate {candidate_idx + 1}/{self.config.num_candidates}")
            hypothesis = self._generate_single_candidate(user_codes, prediction_steps)
            hypotheses.append(hypothesis)

        # Sort by log probability (descending)
        hypotheses.sort(key=lambda h: h.log_prob, reverse=True)
        return hypotheses

    def _generate_single_candidate(
        self,
        user_codes: torch.Tensor,
        prediction_steps: int,
    ) -> _Hypothesis:
        """Generate a single candidate future.

        Args:
            user_codes: Input user codes [1, K, T]
            prediction_steps: Number of future steps to predict

        Returns:
            A single hypothesis with predicted codes
        """
        hypothesis = _Hypothesis()
        cumulative_log_prob = 0.0

        # Enter streaming mode for the LM
        with self.lm_gen.streaming(batch_size=1):
            # First, process the existing user codes to build up context
            B, K, T = user_codes.shape
            for t in range(T):
                codes_t = user_codes[:, :, t:t + 1]  # [1, K, 1]
                tokens = self.lm_gen.step(codes_t)
                # tokens will be None for initial steps due to delays

            # Now generate future predictions
            for step in range(prediction_steps):
                # Generate user input for next step
                if self.config.use_silence_for_user:
                    # Use silence/zero codes for future user input
                    next_user_codes = self._generate_silence_codes()
                else:
                    # Sample continuation from user speech distribution
                    next_user_codes = self._sample_user_continuation()

                # Get system response to this user input
                tokens = self.lm_gen.step(next_user_codes)

                if tokens is not None:
                    # tokens shape: [1, dep_q+1, 1] where dep_q=8
                    # tokens[:, 0] is text, tokens[:, 1:] is system audio codes
                    text_token = tokens[:, 0, 0]  # [1]
                    system_audio_codes = tokens[:, 1:, 0]  # [1, dep_q]

                    # Store the predictions
                    hypothesis.user_codes.append(next_user_codes[:, :, 0])  # [1, K]
                    hypothesis.system_codes.append(system_audio_codes)  # [1, dep_q]
                    hypothesis.text_tokens.append(text_token)  # [1]

                    # For log prob tracking, we would need access to logits
                    # For now, use a placeholder (uniform probability)
                    cumulative_log_prob += 0.0

        hypothesis.log_prob = cumulative_log_prob
        return hypothesis

    def _generate_with_beam_search(
        self,
        user_codes: torch.Tensor,
        prediction_steps: int,
    ) -> list[_Hypothesis]:
        """Generate candidates using beam search.

        Args:
            user_codes: Input user codes [1, K, T]
            prediction_steps: Number of future steps to predict

        Returns:
            List of top-k hypotheses from beam search
        """
        # Initialize beam with a single hypothesis
        beam = [_Hypothesis()]
        beam_size = self.config.num_candidates

        # Process input codes to build context (shared across all beam items)
        with self.lm_gen.streaming(batch_size=1):
            B, K, T = user_codes.shape
            for t in range(T):
                codes_t = user_codes[:, :, t:t + 1]
                self.lm_gen.step(codes_t)

            # Now do beam search for future steps
            for step in range(prediction_steps):
                candidates = []

                # Expand each hypothesis in the beam
                for hyp in beam:
                    # For each hypothesis, generate multiple next steps
                    for _ in range(beam_size):
                        new_hyp = hyp.clone()

                        # Generate next step
                        if self.config.use_silence_for_user:
                            next_user_codes = self._generate_silence_codes()
                        else:
                            next_user_codes = self._sample_user_continuation()

                        tokens = self.lm_gen.step(next_user_codes)

                        if tokens is not None:
                            text_token = tokens[:, 0, 0]
                            system_audio_codes = tokens[:, 1:, 0]

                            new_hyp.user_codes.append(next_user_codes[:, :, 0])
                            new_hyp.system_codes.append(system_audio_codes)
                            new_hyp.text_tokens.append(text_token)

                            candidates.append(new_hyp)

                # Keep top-k candidates
                candidates.sort(key=lambda h: h.log_prob, reverse=True)
                beam = candidates[:beam_size]

        return beam

    def _generate_silence_codes(self) -> torch.Tensor:
        """Generate silence/zero codes for user input.

        Returns:
            Silence codes [1, K, 1] where K is number of codebooks
        """
        # Use zero_token_id to indicate silence/no input
        K = self.lm.num_audio_codebooks
        silence_codes = torch.full(
            (1, K, 1),
            self.lm.zero_token_id,
            device=self.device,
            dtype=torch.long,
        )
        return silence_codes

    def _sample_user_continuation(self) -> torch.Tensor:
        """Sample a continuation for user speech.

        This is a placeholder that currently returns silence.
        In a full implementation, this would sample from a learned
        distribution of user continuations.

        Returns:
            Sampled user codes [1, K, 1]
        """
        # TODO: Implement actual user speech sampling
        # For now, return silence as a conservative default
        return self._generate_silence_codes()

    def _hypotheses_to_result(self, hypotheses: list[_Hypothesis]) -> PrefetchingResult:
        """Convert list of hypotheses to PrefetchingResult.

        Args:
            hypotheses: List of hypotheses to convert

        Returns:
            PrefetchingResult with all candidates
        """
        # Stack user codes: [num_candidates, K, T]
        user_codes_list = []
        system_codes_list = []
        text_tokens_list = []
        log_probs = torch.tensor(
            [h.log_prob for h in hypotheses],
            device=self.device,
            dtype=torch.float,
        )

        for hyp in hypotheses:
            if len(hyp.user_codes) > 0:
                # Stack along time dimension
                user_codes = torch.stack(hyp.user_codes, dim=-1)  # [1, K, T]
                system_codes = torch.stack(hyp.system_codes, dim=-1)  # [1, dep_q, T]
                text_tokens = torch.stack(hyp.text_tokens, dim=-1)  # [1, T]

                user_codes_list.append(user_codes[0])  # [K, T]
                system_codes_list.append(system_codes[0])  # [dep_q, T]
                text_tokens_list.append(text_tokens[0])  # [T]
            else:
                # Empty hypothesis (shouldn't happen, but handle gracefully)
                K = self.lm.num_audio_codebooks
                dep_q = self.lm.dep_q
                user_codes_list.append(torch.zeros(K, 0, device=self.device, dtype=torch.long))
                system_codes_list.append(torch.zeros(dep_q, 0, device=self.device, dtype=torch.long))
                text_tokens_list.append(torch.zeros(0, device=self.device, dtype=torch.long))

        # Stack all candidates
        user_codes = torch.stack(user_codes_list, dim=0)  # [num_candidates, K, T]
        system_codes = torch.stack(system_codes_list, dim=0)  # [num_candidates, dep_q, T]
        text_tokens = torch.stack(text_tokens_list, dim=0)  # [num_candidates, T]

        # Decode to audio
        user_audio = self._decode_to_audio(user_codes)
        system_audio = self._decode_to_audio(system_codes)

        prediction_steps = user_codes.shape[-1]

        return PrefetchingResult(
            user_codes=user_codes,
            system_codes=system_codes,
            user_audio=user_audio,
            system_audio=system_audio,
            text_tokens=text_tokens,
            log_probs=log_probs,
            prediction_steps=prediction_steps,
            frame_rate=self.mimi.frame_rate,
        )

    def _decode_to_audio(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode audio codes to waveforms.

        Args:
            codes: Audio codes [num_candidates, K, T]

        Returns:
            Audio waveforms [num_candidates, 1, audio_length]
        """
        if codes.shape[-1] == 0:
            # Empty codes, return empty audio
            num_candidates = codes.shape[0]
            return torch.zeros(
                num_candidates, 1, 0,
                device=self.device,
                dtype=torch.float,
            )

        # Decode each candidate separately
        audio_list = []
        for i in range(codes.shape[0]):
            codes_i = codes[i:i + 1]  # [1, K, T]
            audio_i = self.mimi.decode(codes_i)  # [1, 1, audio_length]
            audio_list.append(audio_i)

        # Stack all candidates
        audio = torch.cat(audio_list, dim=0)  # [num_candidates, 1, audio_length]
        return audio
