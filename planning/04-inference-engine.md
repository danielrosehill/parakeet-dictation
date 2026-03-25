# Inference Engine — Decision Log

## Current Engine: sherpa-onnx

The app currently uses **sherpa-onnx** as its inference engine. This is a C++ library
with Python bindings that runs ONNX-format ASR models. It handles:

- Audio input → feature extraction → model inference → text output
- VAD (Voice Activity Detection) via Silero VAD
- Streaming (OnlineRecognizer) and offline (OfflineRecognizer) modes
- Endpoint detection for streaming models

## AMD GPU Acceleration

### The Challenge

sherpa-onnx uses ONNX Runtime under the hood. ONNX Runtime supports:

- **CPU** (default, always works)
- **CUDA** (NVIDIA only)
- **ROCm** (AMD, but limited GPU support)
- **DirectML** (Windows only)

Our GPU is an AMD RX 7700 XT / 7800 XT (RDNA 3, Navi 32). ROCm support for RDNA 3
has improved but is not as mature as RDNA 2 or MI-series. Needs testing.

### Options

1. **ONNX Runtime + ROCm EP**: If our GPU is supported, this gives GPU acceleration
   through the existing sherpa-onnx stack with minimal code changes.

2. **ONNX Runtime + Vulkan**: More universal AMD support but potentially slower than
   ROCm for compute workloads.

3. **CPU-only (current)**: Already fast enough for real-time on modern CPUs with 4-8
   threads. The 0.6B models run at ~30x real-time on CPU, which means a 1-second
   utterance is transcribed in ~33ms.

### Decision

**Start with CPU.** The current CPU performance is already well within real-time
requirements. GPU acceleration is a future optimization, not a blocker. The 0.6B
transducer models are efficient enough that CPU inference adds negligible latency
for dictation-length utterances.

If/when we want GPU acceleration:
1. Test ROCm availability: `rocminfo` and check if ONNX Runtime ROCm EP loads
2. If ROCm works, it's a one-line change in the recognizer builder (`provider="rocm"`)
3. If not, evaluate Vulkan compute path

## Should We Switch Engines?

### Considered Alternatives

| Engine | Pros | Cons |
|--------|------|------|
| **faster-whisper** (CTranslate2) | Fast GPU inference | CUDA-only, Whisper architecture only |
| **NeMo** (NVIDIA native) | Runs Parakeet/Nemotron at full fidelity | CUDA-only, massive PyTorch dependency |
| **HF Transformers** | Most flexible, any model | Heavy, slow, CUDA-biased |
| **whisper.cpp** | Vulkan support | Whisper architecture only |

### Verdict

**Stay with sherpa-onnx.** It's the right engine because:

1. Supports the model architectures we need (transducer, CTC, streaming)
2. Lightweight dependency (single pip install)
3. Handles VAD, endpointing, and streaming natively
4. ONNX format means future AMD GPU paths (ROCm, Vulkan) are available
5. Active development with regular new model support
6. The alternative engines are either CUDA-locked or limited to Whisper architecture

The only scenario where we'd switch is if a new engine appears that offers significantly
better AMD GPU acceleration for transducer models. Watch for:
- ONNX Runtime Vulkan EP improvements
- Any AMD-native ASR inference library
- sherpa-onnx adding ROCm/Vulkan support directly
