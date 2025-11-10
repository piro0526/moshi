# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Moshi is a speech-text foundation model for real-time full-duplex dialogue. The repository contains three separate inference implementations plus a web client:

- **PyTorch** (`moshi/`): For research and experimentation
- **MLX** (`moshi_mlx/`): For on-device inference on macOS and iPhone
- **Rust** (`rust/`): Production-ready implementation with Python bindings (`rustymimi`)
- **Web Client** (`client/`): Browser-based UI used in the live demo

## Core Architecture

### Mimi (Audio Codec)
- Streaming neural audio codec: 24 kHz audio → 1.1 kbps at 12.5 Hz frame rate
- 80ms latency (frame size)
- Transformers in both encoder and decoder
- Located in: `moshi/moshi/models/compression.py` (PyTorch), `rust/moshi-core/src/mimi.rs` (Rust)

### Moshi (Language Model)
- 7B parameter Temporal Transformer models two audio streams simultaneously:
  - User's speech (from audio input)
  - Moshi's speech (sampled from model output)
- Predicts text tokens as "inner monologue" for improved generation quality
- Small Depth Transformer handles inter-codebook dependencies per timestep
- Achieves 160ms theoretical latency (200ms practical on L4 GPU)
- Located in: `moshi/moshi/models/lm.py` (PyTorch), `rust/moshi-core/src/lm_generate.rs` (Rust)

## Common Development Commands

### PyTorch Implementation

**Installation:**
```bash
cd moshi
pip install -e '.[dev]'
```

**Run server:**
```bash
python -m moshi.server [--gradio-tunnel] [--hf-repo kyutai/moshika-pytorch-bf16]
# Access web UI at http://localhost:8998
```

**Run client:**
```bash
python -m moshi.client [--url URL_TO_GRADIO]
```

**Testing and linting (from moshi/ directory):**
```bash
cd moshi
pytest tests              # Run tests
pytest tests/test_lm.py   # Run specific test
flake8                    # Lint check
pyright                   # Type check
```

### MLX Implementation (macOS)

**Installation:**
```bash
cd moshi_mlx
pip install -e '.[dev]'
```

**Run local inference:**
```bash
python -m moshi_mlx.local -q 4  # 4-bit quantization
python -m moshi_mlx.local -q 8  # 8-bit quantization
python -m moshi_mlx.local_web   # Web UI at http://localhost:8998
```

**Testing and linting (from moshi_mlx/ directory):**
```bash
cd moshi_mlx
uvx ruff check        # Lint check
uvx ruff format       # Format check
pyright               # Type check
```

### Rust Implementation

**Build and run server (from rust/ directory):**
```bash
cd rust
cargo run --features cuda --bin moshi-backend -r -- --config moshi-backend/config.json standalone
# For macOS: --features metal
# For quantized model: use config-q8.json
# Access web UI at https://localhost:8998
```

**Run CLI client:**
```bash
cd rust
cargo run --bin moshi-cli -r -- tui --host localhost
```

**Testing and linting:**
```bash
cd rust
cargo check   # Quick check
cargo clippy  # Linter
cargo test    # Run tests
cargo fmt     # Format code
```

**Build rustymimi Python bindings:**
```bash
pip install maturin
maturin dev -r -m rust/mimi-pyo3/Cargo.toml
```

### Web Client

**Build (requires npm):**
```bash
cd client
npm install
npm run build
# Output in client/dist/
```

### Pre-commit Hooks

The repository uses pre-commit hooks for automated quality checks:

```bash
# From repository root, after installing dev dependencies
pre-commit install
pre-commit run --all-files  # Run all hooks manually
```

Hooks include:
- `flake8` and `pyright` on `moshi/`
- `pytest` tests on `moshi/`
- `ruff check` and `ruff format` on `moshi_mlx/`
- `pyright` on `moshi_mlx/`

## Key Code Locations

### PyTorch
- **Models**: `moshi/moshi/models/` - `compression.py` (Mimi), `lm.py` (Moshi LM), `loaders.py` (model loading)
- **Server/Client**: `moshi/moshi/server.py`, `moshi/moshi/client.py`
- **Inference**: `moshi/moshi/run_inference.py`, `moshi/moshi/run_tts.py`
- **Modules**: `moshi/moshi/modules/` - Core building blocks (SEANet, transformers, etc.)
- **Tests**: `moshi/tests/`

### MLX
- **Package**: `moshi_mlx/moshi_mlx/` - MLX-specific implementations
- **Entry points**: `local.py` (CLI), `local_web.py` (web server)

### Rust
- **Core library**: `rust/moshi-core/` - Main implementation
- **Backend server**: `rust/moshi-backend/` - Server binary
- **CLI client**: `rust/moshi-cli/` - Terminal UI client
- **Python bindings**: `rust/mimi-pyo3/` - rustymimi package
- **Standalone server**: `rust/moshi-server/` - Alternative server implementation

## Testing Approach

- **PyTorch**: Tests in `moshi/tests/`, run with `pytest` from `moshi/` directory
- **Rust**: Standard `cargo test` from `rust/` directory
- **Integration tests**: Streaming tests in `scripts/` (e.g., `mimi_streaming_test.py`, `moshi_benchmark.py`)

## Model Selection

Models are hosted on HuggingFace with different quantizations:
- **Moshika** (female voice): Various repos for PyTorch (bf16, int8), MLX (int4, int8, bf16), Rust/Candle (int8, bf16)
- **Moshiko** (male voice): Same format variants as Moshika
- Use `--hf-repo` flag to select different models (e.g., `kyutai/moshika-pytorch-bf16`)
- Config files for Rust: `rust/moshi-backend/config.json` (bf16) and `config-q8.json` (int8)

## Important Notes

- **GPU Requirements**: PyTorch requires ~24GB VRAM for bf16 models (no quantization support yet). MLX supports 4-bit/8-bit quantization for Mac. Rust supports int8 quantization with CUDA.
- **Frame Size**: Mimi uses fixed 1920-sample frames (80ms at 24kHz). Always feed multiples of frame size when streaming.
- **Streaming Context**: Both implementations support streaming mode via context managers (`with model.streaming(batch_size)`)
- **Licenses**: Python code is MIT, Rust code is Apache 2.0, model weights are CC-BY 4.0
- **Contributing**: This is a research implementation. Bug fixes welcome, but feature PRs and refactorings are generally not accepted. Requires CLA acceptance.
