# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the speech prefetching module."""

import pytest
import torch

from moshi.prefetching import PrefetchingModule, PrefetchingConfig, PrefetchingResult
from moshi.models import loaders


@pytest.fixture
def device():
    """Get device for testing (CPU for CI, CUDA if available)."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def mimi_model(device):
    """Create a Mimi model for testing."""
    # Create uninitialized model for testing
    mimi = loaders.get_mimi(None, device=device, num_codebooks=8)
    mimi.eval()
    return mimi


@pytest.fixture
def lm_model(device):
    """Create a language model for testing."""
    # Create uninitialized model for testing
    lm = loaders.get_moshi_lm(None, device=device, dtype=torch.float32)
    lm.eval()
    return lm


@pytest.fixture
def prefetcher(mimi_model, lm_model, device):
    """Create a PrefetchingModule for testing."""
    config = PrefetchingConfig(
        num_candidates=3,
        prediction_seconds=0.5,
        device=device,
    )
    return PrefetchingModule(mimi_model, lm_model, config)


class TestPrefetchingConfig:
    """Tests for PrefetchingConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = PrefetchingConfig()
        assert config.num_candidates == 5
        assert config.prediction_seconds == 1.0
        assert config.temperature_user == 0.8
        assert config.temperature_system == 0.7
        assert config.top_k_user == 250
        assert config.top_k_system == 250
        assert config.beam_search is False
        assert config.device == "cuda"
        assert config.use_silence_for_user is True

    def test_custom_config(self):
        """Test custom configuration values."""
        config = PrefetchingConfig(
            num_candidates=10,
            prediction_seconds=2.0,
            temperature_user=0.9,
            beam_search=True,
            device="cpu",
        )
        assert config.num_candidates == 10
        assert config.prediction_seconds == 2.0
        assert config.temperature_user == 0.9
        assert config.beam_search is True
        assert config.device == "cpu"


class TestPrefetchingModule:
    """Tests for PrefetchingModule."""

    def test_initialization(self, mimi_model, lm_model, device):
        """Test module initialization."""
        config = PrefetchingConfig(device=device)
        prefetcher = PrefetchingModule(mimi_model, lm_model, config)

        assert prefetcher.mimi is mimi_model
        assert prefetcher.lm is lm_model
        assert prefetcher.config is config
        assert prefetcher.lm_gen is not None

    def test_single_frame_prediction(self, prefetcher, device):
        """Test prediction for a single audio frame."""
        # Single frame: 80ms at 24kHz = 1920 samples
        user_audio = torch.randn(1, 1, 1920, device=device)

        result = prefetcher.prefetch(user_audio)

        # Check result structure
        assert isinstance(result, PrefetchingResult)
        assert result.user_codes.device.type == device
        assert result.system_codes.device.type == device
        assert result.user_audio.device.type == device
        assert result.system_audio.device.type == device

        # Check shapes
        num_candidates = prefetcher.config.num_candidates
        assert result.user_codes.shape[0] == num_candidates
        assert result.system_codes.shape[0] == num_candidates
        assert result.user_audio.shape[0] == num_candidates
        assert result.system_audio.shape[0] == num_candidates
        assert result.text_tokens.shape[0] == num_candidates
        assert result.log_probs.shape[0] == num_candidates

        # Check codebook dimensions
        assert result.user_codes.shape[1] == prefetcher.lm.num_audio_codebooks
        assert result.system_codes.shape[1] == prefetcher.lm.dep_q

        # Check audio dimensions
        assert result.user_audio.shape[1] == 1  # Mono
        assert result.system_audio.shape[1] == 1

    def test_multi_frame_prediction(self, prefetcher, device):
        """Test prediction for multiple audio frames."""
        # 10 frames: 800ms at 24kHz = 19200 samples
        user_audio = torch.randn(1, 1, 19200, device=device)

        result = prefetcher.prefetch(user_audio)

        # Check that we got predictions
        assert result.prediction_steps > 0
        assert result.user_codes.shape[-1] == result.prediction_steps
        assert result.system_codes.shape[-1] == result.prediction_steps
        assert result.text_tokens.shape[-1] == result.prediction_steps

    def test_prediction_duration(self, mimi_model, lm_model, device):
        """Test that prediction duration matches configuration."""
        config = PrefetchingConfig(
            num_candidates=2,
            prediction_seconds=1.0,
            device=device,
        )
        prefetcher = PrefetchingModule(mimi_model, lm_model, config)

        user_audio = torch.randn(1, 1, 1920, device=device)
        result = prefetcher.prefetch(user_audio)

        # At 12.5 Hz frame rate, 1 second ≈ 12-13 frames
        expected_steps = int(1.0 * mimi_model.frame_rate)
        assert result.prediction_steps == expected_steps

    def test_probability_ranking(self, prefetcher, device):
        """Test that candidates are ranked by log probability."""
        user_audio = torch.randn(1, 1, 1920, device=device)
        result = prefetcher.prefetch(user_audio)

        # Log probabilities should be in descending order
        # (best candidates first)
        log_probs = result.log_probs.cpu().numpy()
        assert all(log_probs[i] >= log_probs[i + 1] for i in range(len(log_probs) - 1))

    def test_invalid_audio_length(self, prefetcher, device):
        """Test that invalid audio length raises error."""
        # Audio length not a multiple of frame_size (1920)
        user_audio = torch.randn(1, 1, 1000, device=device)

        with pytest.raises(ValueError, match="must be a multiple of frame_size"):
            prefetcher.prefetch(user_audio)

    def test_beam_search_mode(self, mimi_model, lm_model, device):
        """Test prefetching with beam search enabled."""
        config = PrefetchingConfig(
            num_candidates=3,
            prediction_seconds=0.5,
            beam_search=True,
            device=device,
        )
        prefetcher = PrefetchingModule(mimi_model, lm_model, config)

        user_audio = torch.randn(1, 1, 1920, device=device)
        result = prefetcher.prefetch(user_audio)

        # Should still produce valid results
        assert result.user_codes.shape[0] == config.num_candidates
        assert result.prediction_steps > 0

    def test_independent_sampling_mode(self, mimi_model, lm_model, device):
        """Test prefetching with independent sampling."""
        config = PrefetchingConfig(
            num_candidates=3,
            prediction_seconds=0.5,
            beam_search=False,
            device=device,
        )
        prefetcher = PrefetchingModule(mimi_model, lm_model, config)

        user_audio = torch.randn(1, 1, 1920, device=device)
        result = prefetcher.prefetch(user_audio)

        # Should produce valid results
        assert result.user_codes.shape[0] == config.num_candidates
        assert result.prediction_steps > 0

    def test_silence_codes_generation(self, prefetcher):
        """Test generation of silence codes."""
        silence_codes = prefetcher._generate_silence_codes()

        # Check shape
        assert silence_codes.shape == (1, prefetcher.lm.num_audio_codebooks, 1)

        # Check all codes are zero_token_id
        assert torch.all(silence_codes == prefetcher.lm.zero_token_id)

    def test_different_num_candidates(self, mimi_model, lm_model, device):
        """Test with different numbers of candidates."""
        for num_candidates in [1, 3, 5]:
            config = PrefetchingConfig(
                num_candidates=num_candidates,
                prediction_seconds=0.5,
                device=device,
            )
            prefetcher = PrefetchingModule(mimi_model, lm_model, config)

            user_audio = torch.randn(1, 1, 1920, device=device)
            result = prefetcher.prefetch(user_audio)

            assert result.user_codes.shape[0] == num_candidates
            assert result.log_probs.shape[0] == num_candidates

    def test_result_metadata(self, prefetcher, device):
        """Test that result metadata is correct."""
        user_audio = torch.randn(1, 1, 1920, device=device)
        result = prefetcher.prefetch(user_audio)

        # Check metadata fields
        assert isinstance(result.prediction_steps, int)
        assert result.prediction_steps > 0
        assert isinstance(result.frame_rate, float)
        assert result.frame_rate == prefetcher.mimi.frame_rate

    def test_audio_reconstruction(self, prefetcher, device):
        """Test that decoded audio has reasonable properties."""
        user_audio = torch.randn(1, 1, 1920, device=device)
        result = prefetcher.prefetch(user_audio)

        # Check that audio is not all zeros (actually generated)
        assert not torch.all(result.user_audio == 0)
        assert not torch.all(result.system_audio == 0)

        # Check audio length matches prediction steps and frame rate
        expected_audio_length = result.prediction_steps * prefetcher.mimi.frame_size
        assert result.user_audio.shape[-1] == expected_audio_length
        assert result.system_audio.shape[-1] == expected_audio_length

    def test_deterministic_with_same_seed(self, prefetcher, device):
        """Test that results are deterministic with same random seed."""
        user_audio = torch.randn(1, 1, 1920, device=device)

        torch.manual_seed(42)
        result1 = prefetcher.prefetch(user_audio)

        torch.manual_seed(42)
        result2 = prefetcher.prefetch(user_audio)

        # Results should be identical with same seed
        torch.testing.assert_close(result1.log_probs, result2.log_probs)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_cuda_execution(self, mimi_model, lm_model):
        """Test execution on CUDA device."""
        config = PrefetchingConfig(
            num_candidates=2,
            prediction_seconds=0.5,
            device="cuda",
        )
        # Move models to CUDA
        mimi_model = mimi_model.cuda()
        lm_model = lm_model.cuda()

        prefetcher = PrefetchingModule(mimi_model, lm_model, config)

        user_audio = torch.randn(1, 1, 1920, device="cuda")
        result = prefetcher.prefetch(user_audio)

        # Check all tensors are on CUDA
        assert result.user_codes.device.type == "cuda"
        assert result.system_codes.device.type == "cuda"
        assert result.user_audio.device.type == "cuda"
        assert result.system_audio.device.type == "cuda"


class TestPrefetchingResult:
    """Tests for PrefetchingResult."""

    def test_result_creation(self, device):
        """Test creating a PrefetchingResult."""
        result = PrefetchingResult(
            user_codes=torch.zeros(5, 8, 10, device=device, dtype=torch.long),
            system_codes=torch.zeros(5, 8, 10, device=device, dtype=torch.long),
            user_audio=torch.zeros(5, 1, 1920 * 10, device=device),
            system_audio=torch.zeros(5, 1, 1920 * 10, device=device),
            text_tokens=torch.zeros(5, 10, device=device, dtype=torch.long),
            log_probs=torch.tensor([0.0, -0.5, -1.0, -1.5, -2.0], device=device),
            prediction_steps=10,
            frame_rate=12.5,
        )

        assert result.user_codes.shape == (5, 8, 10)
        assert result.system_codes.shape == (5, 8, 10)
        assert result.prediction_steps == 10
        assert result.frame_rate == 12.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
