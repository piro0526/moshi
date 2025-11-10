# Implementation Plan: User Speech Prefetching Module

## Task Summary
Implement a Python module that predicts future user speech and system responses given current user speech audio. The module should generate multiple candidate futures with configurable prediction duration and number of candidates.

## Architecture Overview

### Key Components Identified
1. **Mimi (Audio Codec)**: `moshi/moshi/models/compression.py`
   - Encodes 24kHz audio → codes at 12.5Hz frame rate
   - Frame size: 1920 samples (80ms at 24kHz)
   - 8 codebooks used by Moshi

2. **Moshi LM**: `moshi/moshi/models/lm.py`
   - Models two audio streams: user + system
   - LMGen class handles streaming generation
   - Supports sampling with temperature and top-k

3. **Sampling**: `moshi/moshi/utils/sampling.py`
   - Supports top-k, top-p, and temperature-based sampling
   - Can generate multiple candidates via multinomial sampling

## Implementation Plan

### Module Location
Create new file: `moshi/moshi/prefetching.py`

### Module Structure

```python
# Main components:
1. PrefetchingConfig (dataclass)
   - num_candidates: int
   - prediction_seconds: float
   - temperature_user: float (for sampling user speech)
   - temperature_system: float (for sampling system speech)
   - top_k_user: int
   - top_k_system: int
   - beam_search: bool (use beam search vs independent sampling)

2. PrefetchingState (dataclass)
   - Stores internal state for streaming prefetching
   - Cached LM state
   - Mimi streaming state

3. PrefetchingModule (main class)
   - __init__(mimi: MimiModel, lm: LMModel, config: PrefetchingConfig)
   - prefetch(user_audio: torch.Tensor) -> PrefetchingResult
   - prefetch_streaming(user_codes: torch.Tensor) -> PrefetchingResult
```

### Implementation Details

#### 1. PrefetchingConfig
```python
@dataclass
class PrefetchingConfig:
    num_candidates: int = 5
    prediction_seconds: float = 1.0  # How far ahead to predict
    temperature_user: float = 0.8    # For user speech sampling
    temperature_system: float = 0.7  # For system speech sampling
    top_k_user: int = 250
    top_k_system: int = 250
    beam_search: bool = False  # If True, use beam search; else parallel sampling
    device: str = "cuda"
```

#### 2. PrefetchingResult
```python
@dataclass
class PrefetchingResult:
    # [num_candidates, num_codebooks, num_timesteps]
    user_codes: torch.Tensor
    system_codes: torch.Tensor

    # [num_candidates, 1, audio_length]
    user_audio: torch.Tensor
    system_audio: torch.Tensor

    # [num_candidates, num_timesteps]
    text_tokens: torch.Tensor

    # [num_candidates] - log probabilities for ranking
    log_probs: torch.Tensor

    # Metadata
    prediction_steps: int
    frame_rate: float
```

#### 3. Core Algorithm

**Approach: Beam Search with Speculation**

For speculative prediction, we need to:
1. Clone the current LM streaming state N times (N=num_candidates)
2. For each future timestep:
   - Generate system response for current user input
   - Sample multiple hypotheses for next user utterance
   - Continue generation for K future timesteps
3. Track log probabilities for ranking candidates

**Key Challenge**: Predicting future USER speech
- The model was trained to generate system speech given user input
- For prefetching, we need to speculate on future user turns
- Solutions:
  a. Use the model's ability to predict both streams
  b. Sample from the user codebook distribution (currently zero_token)
  c. Create a "silence" continuation hypothesis + sampling-based alternatives

#### 4. Implementation Steps

**Step 1: Basic Prefetching (Single Step Ahead)**
- File: `moshi/moshi/prefetching.py`
- Implement core class structure
- Support single-step prediction with multiple candidates
- Test with simple scenarios

**Step 2: Multi-Step Prediction**
- Extend to K steps ahead (K = prediction_seconds * frame_rate)
- Implement state management for parallel hypotheses
- Handle streaming state correctly

**Step 3: User Speech Speculation**
Key insight from `lm.py:669-712`: The LMGen._step() method:
- Takes `input_tokens` (user stream codes)
- Generates system stream codes + text

For speculation:
- Option A: Assume continuation of last known user pattern
- Option B: Sample from a "turn-taking" model
- Option C: Generate silence codes for user stream
- **Recommended**: Start with Option C, then add Option B

**Step 4: Beam Search/Ranking**
- Implement log probability tracking
- Rank candidates by cumulative log probability
- Prune low-probability branches

**Step 5: Integration & Testing**
- Create test in `moshi/tests/test_prefetching.py`
- Test with different audio inputs
- Validate output shapes and distributions

### Code Structure

```
moshi/moshi/prefetching.py
├── PrefetchingConfig (dataclass)
├── PrefetchingResult (dataclass)
├── PrefetchingState (dataclass)
└── PrefetchingModule (class)
    ├── __init__(mimi, lm, config)
    ├── prefetch(user_audio) -> PrefetchingResult
    ├── _prepare_streaming_states(batch_size) -> list[LMGenState]
    ├── _speculate_user_continuation(current_codes) -> torch.Tensor
    ├── _generate_candidates_single_step() -> list[Hypothesis]
    ├── _generate_candidates_multi_step() -> list[Hypothesis]
    └── _decode_to_audio(codes) -> torch.Tensor
```

### Critical Implementation Notes

1. **Frame Size Constraints**
   - User audio MUST be multiple of frame_size (1920 samples)
   - From `compression.py:361-365`: streaming mode requires exact frame multiples

2. **Streaming State Management**
   - Each candidate needs its own LM streaming state
   - Use `lm_gen.streaming(batch_size)` context manager
   - Clone states using `copy.deepcopy()` or by tracking offsets

3. **Code Shape**
   - Input codes: `[B, K, T]` where K=num_codebooks, T=timesteps
   - Moshi expects 1+n_q codebooks: 1 text + 8 audio (from user) + 8 audio (from system)

4. **Zero Tokens and Delays**
   - From `lm.py:269`: zero_token_id = -1 (indicates no input)
   - From `loaders.py:118`: delays = [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1]
   - Must handle delayed alignment correctly

5. **Sampling for Multiple Candidates**
   - Modify `sample_token()` in `utils/sampling.py` to support `num_samples > 1`
   - Or: Call sampling multiple times with different random seeds
   - Track log probabilities: `log_prob = torch.log_softmax(logits / temp, dim=-1)`

### Testing Strategy

**Test 1: Single Frame Prediction**
```python
# Input: 80ms of user speech
# Output: 5 candidates for next 80ms (user + system)
user_audio = torch.randn(1, 1, 1920)  # Single frame
result = prefetcher.prefetch(user_audio)
assert result.user_codes.shape == (5, 8, 1)  # 5 candidates, 8 codebooks, 1 frame
```

**Test 2: Multi-Frame Prediction**
```python
# Input: 1 second of user speech
# Output: 5 candidates for next 1 second
user_audio = torch.randn(1, 1, 24000)
config = PrefetchingConfig(num_candidates=5, prediction_seconds=1.0)
result = prefetcher.prefetch(user_audio)
# At 12.5 Hz frame rate, 1 second = 12-13 frames
assert result.user_codes.shape[2] in [12, 13]
```

**Test 3: Probability Ranking**
```python
# Verify candidates are ranked by log probability
result = prefetcher.prefetch(user_audio)
assert result.log_probs.shape == (5,)
assert torch.all(result.log_probs[:-1] >= result.log_probs[1:])  # Descending
```

### Dependencies

- Existing: `torch`, `moshi.models.LMModel`, `moshi.models.MimiModel`, `moshi.models.LMGen`
- May need: `copy` (for state cloning), `dataclasses`, `typing`

### Potential Extensions

1. **Turn-Taking Model**: Learn when user is likely to speak vs stay silent
2. **Context-Aware Prediction**: Use conversation history for better prediction
3. **Adaptive Branching**: Dynamically adjust num_candidates based on uncertainty
4. **Multi-Modal Input**: Incorporate text or other modalities

### Files to Create/Modify

**New Files:**
1. `moshi/moshi/prefetching.py` - Main implementation
2. `moshi/tests/test_prefetching.py` - Unit tests

**Modified Files:**
1. `moshi/moshi/__init__.py` - Add import for PrefetchingModule
2. `moshi/README.md` - Add usage documentation (optional)

### Example API Usage

```python
from moshi.models import loaders
from moshi.prefetching import PrefetchingModule, PrefetchingConfig

# Load models
mimi = loaders.get_mimi("path/to/mimi.safetensors")
lm = loaders.get_moshi_lm("path/to/moshi.safetensors")

# Configure prefetching
config = PrefetchingConfig(
    num_candidates=5,
    prediction_seconds=1.0,
    temperature_user=0.8,
    temperature_system=0.7,
)

# Create prefetcher
prefetcher = PrefetchingModule(mimi, lm, config)

# Prefetch future speech
user_audio = torch.randn(1, 1, 24000)  # 1 second at 24kHz
result = prefetcher.prefetch(user_audio)

# Access results
print(f"Generated {result.user_audio.shape[0]} candidates")
print(f"Each candidate covers {result.prediction_steps} frames")
print(f"Top candidate log prob: {result.log_probs[0]}")

# Decode best candidate
best_user = result.user_audio[0]  # Shape: [1, audio_length]
best_system = result.system_audio[0]
```

## Implementation Priority

**Phase 1 (MVP)**:
- Basic class structure
- Single-step prefetching
- Silence-based user continuation
- Independent sampling (no beam search)

**Phase 2**:
- Multi-step prediction
- Beam search support
- Probability ranking

**Phase 3**:
- Streaming API
- Performance optimization
- Advanced user continuation strategies

## Risk Assessment

**Risk 1: User Speech Speculation Accuracy**
- Mitigation: Start with simple strategies (silence, continuation)
- Alternative: Train a separate model for user turn prediction

**Risk 2: State Management Complexity**
- Mitigation: Use batch processing where possible
- Thoroughly test state isolation between candidates

**Risk 3: Computational Cost**
- Issue: N candidates × K timesteps = N*K forward passes
- Mitigation: Implement batch processing, use CUDA graphs

**Risk 4: Memory Usage**
- Issue: Storing N streaming states
- Mitigation: Implement lazy state cloning, clear old states

## Success Criteria

1. ✅ Module successfully generates multiple candidate futures
2. ✅ Output shapes match specification
3. ✅ Candidates are diverse (different audio/text outputs)
4. ✅ Candidates are ranked by probability
5. ✅ Tests pass with >95% coverage
6. ✅ Performance: <100ms for 5 candidates, 1 second ahead (on GPU)
