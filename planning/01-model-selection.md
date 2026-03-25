# Model Selection — Decision Log

## Hardware Context

- **GPU**: AMD Radeon RX 7700 XT / 7800 XT (Navi 32), 12GB VRAM
- **GPU compute**: ROCm (if supported), Vulkan fallback, ONNX Runtime
- **No CUDA** — rules out CUDA-native engines (faster-whisper CTranslate2, NeMo native)

## Design Philosophy: Desktop-Class, Not Edge

Many local ASR projects optimize for the smallest possible model — running on Raspberry Pi,
phones, or embedded devices. That's the wrong target for this project.

We have a **desktop workstation with 12GB VRAM and a modern CPU**. We should use that power
to get the best possible transcription quality, not chase the smallest footprint. Specifically:

- **Don't sacrifice accuracy for size.** A 600MB model that runs in 200ms on our hardware is
  better than a 50MB model that runs in 150ms but makes more errors.
- **Don't optimize for cold-start time.** The app runs as a persistent tray service — the model
  loads once and stays in memory.
- **Do optimize for per-utterance latency.** The time between "user stops speaking" and "text
  appears" is what matters. On our hardware, even large models can hit sub-second here.
- **GPU acceleration matters.** The model should be runnable with AMD GPU compute (ROCm or
  Vulkan via ONNX Runtime) — not just CPU. This is where the 12GB VRAM pays off.

The sweet spot is the largest model that gives near-instant inference on our hardware.
That's currently the 0.6B parameter class (Parakeet TDT, Nemotron).

## Models Evaluated

### NVIDIA NeMo Models (Currently Used)

#### Parakeet TDT 0.6B v3 — PRIMARY MODEL ★

- **Architecture**: Transducer (TDT — Token-and-Duration Transducer)
- **Params**: 600M | **ONNX size**: 639 MB (int8)
- **Languages**: 25 European languages
- **Streaming**: No (offline, VAD-segmented)
- **Punctuation**: Yes, built-in
- **Runtime**: sherpa-onnx via ONNX
- **Suitability**: Excellent. Best accuracy in the 0.6B class. The transducer architecture
  is purpose-built for real-time. VAD-segmented means it waits for a pause, then transcribes
  the segment very fast. On our hardware this gives sub-second latency after speech ends.
- **Status**: Currently the default model in the app.

#### Nemotron Streaming 0.6B — STREAMING MODEL ★

- **Architecture**: Transducer (streaming variant)
- **Params**: 600M | **ONNX size**: 631 MB (int8)
- **Languages**: English only
- **Streaming**: Yes, true frame-by-frame streaming
- **Punctuation**: Yes, built-in
- **Runtime**: sherpa-onnx via ONNX
- **Suitability**: Best option for true streaming with partial results. Supports endpoint
  detection and partial hypothesis revision — the key ingredients for "Deepgram feel."
  English-only limitation is the main drawback.
- **Status**: Currently available in the app as the "streaming" profile.

#### Canary 180M Flash

- **Architecture**: CTC (Conformer-CTC)
- **Params**: 180M | **ONNX size**: 198 MB (int8)
- **Languages**: EN/ES/DE/FR
- **Streaming**: No
- **Punctuation**: Yes
- **Runtime**: sherpa-onnx via ONNX
- **Suitability**: Good lightweight fallback. CTC architecture is simpler — no language
  model component, so accuracy is lower than transducer models. With 12GB VRAM available,
  there's no reason to prefer this over Parakeet TDT for accuracy-sensitive dictation.
  Useful as a "laptop mode" option.
- **Status**: Currently available in the app as the "laptop" profile.

### Models Investigated But Not Selected

#### Qwen3-ASR-0.6B (Alibaba)
- **Source**: https://huggingface.co/Qwen/Qwen3-ASR-0.6B
- **Architecture**: Unclear from docs — built on Qwen3-Omni foundation model
- **Params**: 900M (despite the "0.6B" name)
- **Languages**: 52 languages and 22 Chinese dialects — impressive multilingual coverage
- **Streaming**: Yes (but only via vLLM backend)
- **Punctuation**: Appears to be included
- **Runtime**: Requires `qwen_asr` custom library + PyTorch. vLLM for streaming.
- **Suitability**: Interesting multilingual option. However, the streaming mode requires
  vLLM (heavy, CUDA-focused). No ONNX export available. Can't run through sherpa-onnx.
  The dependency on vLLM for streaming makes it impractical for a lightweight desktop app.
  Would need a complete engine rewrite.
- **Verdict**: Not viable for current architecture. Worth revisiting if ONNX exports appear.

#### Voxtral Mini 4B Realtime (Mistral)
- **Source**: https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602
- **Architecture**: Custom streaming — causal audio encoder (~970M) + LLM decoder (~3.4B,
  based on Ministral-3B). Not transducer, not CTC, not traditional encoder-decoder.
  Uses sliding window attention for "infinite" streaming.
- **Params**: 4B total
- **Languages**: 13 (AR, DE, EN, ES, FR, HI, IT, NL, PT, ZH, JA, KO, RU)
- **Streaming**: Yes, natively. Configurable delay 80ms-2400ms. 480ms recommended.
- **Punctuation**: Yes
- **Runtime**: HuggingFace Transformers or vLLM. CUDA-focused.
- **Suitability**: Very promising architecture — purpose-built for real-time with
  configurable latency tradeoff. The 480ms sweet spot is exactly the "semi-streaming"
  behavior we want. However: 4B params is large (needs ≥16GB VRAM per their docs),
  requires CUDA (vLLM), no ONNX export. Can't run on our AMD GPU without significant
  porting effort.
- **Verdict**: Architecture is the closest to what we want conceptually. Too heavy and
  CUDA-locked for our setup. Worth watching — if ONNX or ROCm support appears, this
  would be a strong candidate. The configurable delay parameter is exactly the "Deepgram
  feel" tuning knob.

#### Granite 4.0 1B Speech (IBM)
- **Source**: https://huggingface.co/ibm-granite/granite-4.0-1b-speech
- **Architecture**: Encoder-decoder. CTC-trained conformer encoder (16 blocks) →
  q-former projector → Granite-4.0-1b LLM decoder. Essentially an audio-conditioned LLM.
- **Params**: 1B
- **Languages**: 6 (EN, FR, DE, ES, PT, JA) + translation capabilities
- **Streaming**: No — encoder-decoder, processes full segments
- **Punctuation**: Yes (LLM-quality output)
- **Runtime**: HuggingFace Transformers + PyTorch. CUDA expected.
- **Suitability**: Impressive accuracy (5.52% WER, best in class for <2B) and very fast
  (280x RTF). But it's an encoder-decoder — same fundamental problem as Whisper. Must
  process the entire audio segment before emitting text. Also supports keyword biasing
  (useful for names/acronyms), which is a unique feature. No ONNX export.
- **Verdict**: Wrong architecture for real-time streaming. The keyword biasing feature
  is interesting for future reference. If we ever need a high-accuracy offline-only mode
  with translation, this would be worth revisiting.

#### Moonshine (Useful Sensors)
- **Source**: https://huggingface.co/UsefulSensors/moonshine
- **Architecture**: Sequence-to-sequence encoder-decoder (like Whisper, but smaller)
- **Params**: 27M (tiny) / 61M (base)
- **Languages**: English only
- **Streaming**: Not truly streaming — optimized for short chunks but still seq2seq
- **Punctuation**: Unclear
- **Runtime**: Keras (PyTorch/TF/JAX backends). sherpa-onnx has community ONNX exports.
- **Suitability**: Designed to replace Whisper on edge devices. At 27-61M params, these
  are far too small for our desktop use case — we have the headroom for 10-20x larger
  models. Also encoder-decoder architecture, so same fundamental streaming limitation
  as Whisper. Known to hallucinate and produce repetitive text.
- **Verdict**: Too small, wrong architecture. This is the opposite of our design philosophy —
  it's optimized for Raspberry Pi, not desktop workstations. The hallucination issue is
  a dealbreaker for dictation.

#### SenseVoice (FunAudioLLM)
- **Architecture**: Encoder-only with CTC-like output
- **Params**: Small (exact size varies)
- **Languages**: 50+ languages
- **Streaming**: No
- **Suitability**: Interesting for multilingual but not streaming-capable. Small model
  size is again optimizing for the wrong target given our hardware.
- **Verdict**: Not a fit for real-time dictation.

## Summary Table

| Model | Params | Architecture | Streaming | Languages | ONNX | AMD GPU | Verdict |
|-------|--------|-------------|-----------|-----------|------|---------|---------|
| **Parakeet TDT 0.6B** | 600M | Transducer | No (fast offline) | 25 | Yes | Via ONNX RT | **★ Primary** |
| **Nemotron Streaming 0.6B** | 600M | Transducer | Yes | EN | Yes | Via ONNX RT | **★ Streaming** |
| Canary 180M Flash | 180M | CTC | No | 4 | Yes | Via ONNX RT | Lightweight fallback |
| Qwen3-ASR-0.6B | 900M | LLM-based | vLLM only | 52 | No | No | Blocked on runtime |
| Voxtral Mini 4B | 4B | Custom causal | Yes | 13 | No | No | Best arch, CUDA-locked |
| Granite 4.0 1B | 1B | Enc-dec + LLM | No | 6 | No | No | Wrong arch |
| Moonshine | 27-61M | Enc-dec (seq2seq) | No | EN | Community | Possible | Too small, hallucinations |
| SenseVoice | Small | Encoder+CTC | No | 50+ | No | Unknown | Not streaming |

## Current Decision

**Primary model: Parakeet TDT 0.6B** — best accuracy-to-speed ratio for our hardware.
The transducer architecture is correct for real-time dictation.

**Streaming model: Nemotron Streaming 0.6B** — for true streaming with partial-overwrite.

## What Would Change This Decision

1. **Voxtral gets ONNX export or ROCm support** — its configurable-delay streaming
   architecture is exactly what we want. At 4B it might be too large, but worth testing.
2. **Qwen3-ASR gets ONNX export** — 52 languages with streaming would be compelling.
3. **A new NVIDIA NeMo transducer >0.6B** — if a 1-2B transducer appears with ONNX
   export, it would likely be more accurate and still fast enough on our hardware.
4. **ONNX Runtime ROCm EP matures for RDNA 3** — would unlock GPU acceleration for
   existing models, potentially making larger models viable.

## Models NOT Worth Pursuing

| Model | Why Not |
|-------|---------|
| Whisper (any size) | Wrong architecture for real-time |
| Moonshine | Wrong architecture, too small, hallucination issues |
| wav2vec2 | No punctuation, different runtime |
| Vosk | Separate engine, worse accuracy |
| Whisper large-v3 via faster-whisper | CUDA-only (CTranslate2), wrong architecture anyway |
| NeMo native (PyTorch) | CUDA-only, heavy dependency chain |
