# Planning Decisions & Research Notes

## Project Objective

Lightweight, local-friendly dictation tool for Ubuntu based on NVIDIA Parakeet (via sherpa-onnx). Focus on direct text entry anywhere, at any text cursor. No cloud dependencies, optimized for lower-resource hardware (laptops without GPU).

---

## Why Parakeet / sherpa-onnx

- The vast majority of speech-to-text tools are just Whisper wrappers. Whisper is great, but for low-latency, on-the-fly text input, other models have generally shown better results.
- **Punctuation restoration** is the key differentiator. Parakeet (NeMo) models include built-in punctuation and capitalization, which is critical for dictation (vs transcription).
- **Moonshine** (by Useful Sensors) — very promising new Chinese product, but lacks punctuation restoration. Ruled out for this reason.
- **In-Sight** (on Hugging Face) — specifically designed for ASR at the edge, but was not found to be very accurate in testing.
- **sherpa-onnx** is the runtime — it provides a clean Python API over ONNX-exported NeMo models, handles VAD, streaming, and works purely on CPU.

## Language Support per Model

- **Parakeet TDT 0.6B** — Supports 25 European languages but the sherpa-onnx `from_transducer()` API has **no language parameter**. The model auto-detects the spoken language. There is no way to force/hint a specific language. If it occasionally outputs wrong-language fragments, `rule_fsts` (finite state transducers for constraining output tokens) may help, but that's an advanced configuration.
- **Canary 180M Flash** — Supports EN, ES, DE, FR with **explicit `src_lang` and `tgt_lang` parameters** via `from_nemo_canary()`. Setting `src_lang="en"` deterministically constrains recognition to English. This is the only model with a proper language selector. Setting `src_lang != tgt_lang` enables speech translation (e.g., speak Spanish, get English text).
- **Nemotron Streaming 0.6B** — English only. No language parameter needed or available.

**Recommendation**: For users who want deterministic single-language recognition, Canary with an explicit language setting is the most reliable. Parakeet generally auto-detects well for English speakers but cannot be locked to a single language via the API.

## VAD

Using **TEN VAD** (ten-vad) — switched from Silero VAD in v1.0.0.

- **TEN VAD** (~306 KB native library, bundled in pip package, Apache 2.0). No separate model download needed. High performance. Uses hop_size=256, threshold=0.5. Requires `libc++1` on Linux.
- **Silero VAD** (previous, removed) — 2.3 MB ONNX model, required separate download. Worked well but heavier.

## GPU Acceleration Question

- This laptop: AMD Ryzen 7 5700U, no discrete GPU — CPU only.
- Home workstation: AMD GPU (not NVIDIA).
- Parakeet/NeMo models are NVIDIA models exported to ONNX. sherpa-onnx uses ONNX Runtime which supports:
  - CPU (works everywhere)
  - CUDA (NVIDIA GPUs only)
  - ROCm (AMD GPUs — but sherpa-onnx may not ship ROCm builds)
  - DirectML (Windows)
- **Ideal**: a single app/model that uses GPU when available (NVIDIA or AMD) and falls back to CPU. Need to investigate whether sherpa-onnx supports ROCm provider for AMD GPUs, or if an alternative ONNX runtime build is needed.

## Cloud Fallback Consideration

- If a cloud/API fallback were ever added, **Deepgram** has shown the best results for real-time dictation use cases.
- User's day-to-day voice transcription uses **Gemini multimodal** — sends audio + a text formatting instruction together (e.g., "this is an email, this is formal"). This combined audio+instruction approach is more logical at the cloud level and produces better formatted output.
- That Gemini-based app is the AI-Transcription-Notepad at ~/repos/github/AI-Transcription-Notepad.

## Multimodal Local Model Direction

- The Gemini multimodal approach (audio + formatting instruction) would be ideal to replicate locally.
- This is significantly more challenging than pure ASR — requires a multimodal LLM, not just a speech model.
- **Voxtral Mini 3B** (Mistral, released ~Feb 2026, real-time version ~Jul 2025): https://huggingface.co/mistralai/Voxtral-Mini-3B-2507
  - 3B parameter multimodal model with audio understanding
  - Could potentially handle audio + instruction prompting locally
  - Question: can it run on this hardware class? 3B is borderline for CPU-only inference, may need quantization.
  - This is a research direction, not a near-term feature.

## Wayland Keyboard Entry

The hardest part of local dictation on modern Linux is not the models — it's getting text into applications on Wayland:
- **ydotool** is the primary method. Requires uinput access (`/dev/uinput` permissions, user in `input` group, udev rule).
- Older ydotool (0.1.x, in Ubuntu repos) works without a daemon but prints a warning. Newer versions (1.x) require `ydotoold` daemon.
- **xdotool** works on X11 but NOT on native Wayland windows.
- Some Wayland compositors support `wtype` as an alternative.
- This is the main installation friction point and should be well-documented.

## Whisper vs Specialized ASR

For the record: Whisper is excellent for batch transcription of recordings, podcasts, meetings, etc. But for real-time dictation where latency matters and text must appear as you speak with proper punctuation, specialized models (Parakeet/NeMo, Canary) running through optimized runtimes (sherpa-onnx) have been more practical. The streaming Nemotron model in particular enables true real-time output, which Whisper cannot do natively.
