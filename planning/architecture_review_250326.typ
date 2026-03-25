#set text(font: "IBM Plex Sans", size: 10pt)
#set page(margin: (x: 2.5cm, y: 2.5cm))
#set par(justify: true, leading: 0.65em)
#set heading(numbering: "1.1")

#align(center)[
  #text(size: 20pt, weight: "bold")[Parakeet Dictation]
  #v(0.3em)
  #text(size: 14pt)[Architecture Review]
  #v(0.3em)
  #text(size: 10pt, fill: luma(100))[26 March 2025 · Planning Document]
]

#v(1.5em)

= Executive Summary

Parakeet Dictation is a local, on-device voice typing application for Linux. This document captures the architectural decisions behind model selection, inference engine choice, and the streaming/text-injection approach. The goal is to achieve dictation quality comparable to cloud services like Deepgram Nova while running entirely on the user's hardware.

*Hardware target:* Desktop workstation with AMD Radeon RX 7700 XT / 7800 XT (12 GB VRAM), modern multi-core CPU. Not an embedded or edge device.

= Design Philosophy

Many local ASR projects optimize for the smallest possible footprint --- running on Raspberry Pi, phones, or embedded devices. That is the wrong target for this project.

With 12 GB of VRAM and a modern CPU, we should use that power to get the best transcription quality, not chase the smallest model. Specifically:

- *Don't sacrifice accuracy for size.* A 600 MB model that infers in 200 ms is better than a 50 MB model that infers in 150 ms but makes more errors.
- *Don't optimize for cold start.* The app runs as a persistent tray service --- the model loads once and stays resident.
- *Do optimize for per-utterance latency.* The time between "user stops speaking" and "text appears" is what matters.
- *GPU acceleration matters.* AMD compute (ROCm or Vulkan via ONNX Runtime) should be leveraged where possible.

The sweet spot is the largest model that gives near-instant inference on our hardware --- currently the 0.6 B parameter class.

= ASR Architecture Primer

All speech-to-text models are ASR, but there are fundamentally different architectures with very different real-time properties.

== Encoder-Decoder (Seq2Seq)

#block(inset: (left: 1em))[
  `[Audio] → Encoder → [hidden states] → Decoder → [text tokens]`
]

The encoder processes the *entire* audio segment into a fixed representation; the decoder then generates text auto-regressively. Must wait for the full utterance before starting. Examples: Whisper, Granite Speech, SeamlessM4T.

*Not suitable for real-time dictation.* No incremental output, no built-in endpointing. "Streaming Whisper" implementations are hacks that re-run the model on growing windows.

== CTC (Connectionist Temporal Classification)

#block(inset: (left: 1em))[
  `[Audio frame] → Encoder → [one label per frame] → CTC decode → [text]`
]

Processes audio frame-by-frame --- each frame independently predicts a token. Very fast, simple decode step. However, each prediction is independent (no language model), so punctuation and casing are harder. Example: Canary 180 M Flash.

*Usable for real-time, but accuracy ceiling is lower than transducers.*

== Transducer (RNN-T / TDT) --- The Right Architecture

#block(inset: (left: 1em))[
  `[Audio frame] → Encoder → ┐`\
  `                           ├→ Joint Network → [token or blank]`\
  `[Previous tokens] → Predictor → ┘`
]

Processes audio frame-by-frame *and* has a predictor (language model) conditioned on previously emitted tokens. This means it uses linguistic context to resolve ambiguities. Emits tokens incrementally. The joint network decides at each frame: emit a token, or emit blank (wait for more audio).

TDT (Token-and-Duration Transducer) is NVIDIA's improvement that also predicts token duration.

*This is what we want.* Frame-by-frame processing with linguistic context, incremental output, sentence boundary inference, and built-in punctuation.

= Models Evaluated

#let check = sym.checkmark
#let cross = sym.times

== Selected Models

#table(
  columns: (1.5fr, 0.8fr, 1fr, 0.6fr, 0.8fr, 0.5fr),
  inset: 6pt,
  align: (left, center, center, center, center, center),
  table.header(
    [*Model*], [*Params*], [*Architecture*], [*Stream*], [*Languages*], [*ONNX*],
  ),
  [Parakeet TDT 0.6B ★], [600 M], [Transducer (TDT)], [No#super[1]], [25], [#check],
  [Nemotron Streaming 0.6B ★], [600 M], [Transducer], [Yes], [EN], [#check],
  [Canary 180 M Flash], [180 M], [CTC], [No], [4], [#check],
)

#text(size: 8pt)[#super[1] Offline / VAD-segmented, but sub-second latency on our hardware.]

*Parakeet TDT 0.6B v3* is the primary model --- best accuracy in the 0.6 B class, 25 European languages, built-in punctuation and capitalization.

*Nemotron Streaming 0.6B* is the streaming model --- true frame-by-frame streaming with endpoint detection and partial hypothesis revision. English only.

== Models Investigated But Not Selected

#table(
  columns: (1.3fr, 0.6fr, 1fr, 0.5fr, 2fr),
  inset: 6pt,
  align: (left, center, center, center, left),
  table.header(
    [*Model*], [*Params*], [*Architecture*], [*ONNX*], [*Why Not*],
  ),
  [Voxtral Mini 4B\ (Mistral)], [4 B], [Custom causal\ encoder + LLM], [#cross], [Best streaming arch conceptually (configurable 80--2400 ms delay). But 4 B params, CUDA-locked (vLLM), no ONNX. Too heavy for our GPU.],
  [Qwen3-ASR-0.6B\ (Alibaba)], [900 M], [LLM-based], [#cross], [52 languages with streaming --- impressive. But streaming requires vLLM (CUDA). No ONNX export. Can't run through sherpa-onnx.],
  [Granite 4.0 1B\ (IBM)], [1 B], [Encoder-decoder\ + LLM], [#cross], [Great accuracy (5.52% WER), unique keyword biasing. But encoder-decoder --- same problem as Whisper. No streaming. CUDA expected.],
  [Moonshine\ (Useful Sensors)], [27--61 M], [Encoder-decoder\ (seq2seq)], [Partial], [Too small for desktop. Wrong architecture. Known hallucination and repetition issues. Optimized for Raspberry Pi.],
  [SenseVoice\ (FunAudioLLM)], [Small], [Encoder + CTC], [#cross], [50+ languages but not streaming. Too small for our hardware target.],
  [Whisper\ (OpenAI)], [25 M--1.5 B], [Encoder-decoder], [Yes], [Wrong architecture for real-time. No incremental output, no endpointing. The canonical example of what we're moving away from.],
)

== What Would Change This Decision

+ *Voxtral gets ONNX export or ROCm support* --- its configurable-delay streaming is exactly what we want.
+ *Qwen3-ASR gets ONNX export* --- 52 languages with streaming would be compelling.
+ *A new NeMo transducer \>0.6 B* appears with ONNX --- more accuracy, still fast enough.
+ *ONNX Runtime ROCm EP matures for RDNA 3* --- unlocks GPU acceleration for existing models.

= Inference Engine

== Why sherpa-onnx

The app uses *sherpa-onnx* as its inference engine --- a C++ library with Python bindings that runs ONNX-format ASR models.

*Reasons for selection:*
- Supports transducer, CTC, and streaming architectures natively
- Handles VAD, endpointing, and partial results out of the box
- Lightweight single dependency (`pip install sherpa-onnx`)
- ONNX format keeps future AMD GPU paths open (ROCm, Vulkan via ONNX Runtime)
- Active development with regular new model support

== Alternatives Considered

#table(
  columns: (1.2fr, 1fr, 2fr),
  inset: 6pt,
  align: (left, center, left),
  table.header(
    [*Engine*], [*GPU Support*], [*Why Not*],
  ),
  [faster-whisper (CTranslate2)], [CUDA only], [Whisper architecture only. CUDA-locked.],
  [NeMo native (PyTorch)], [CUDA only], [Runs Parakeet at full fidelity but massive dependency chain. CUDA-locked.],
  [HF Transformers], [CUDA-biased], [Most flexible but heaviest. Overkill for a tray app.],
  [whisper.cpp], [CUDA / Vulkan], [Whisper architecture only.],
)

== AMD GPU Path

Current status: *CPU-only*. The 0.6 B models run at \~30× real-time on CPU (a 1-second utterance transcribes in \~33 ms), which is already well within real-time requirements.

Future GPU acceleration options:
- ONNX Runtime + ROCm execution provider (RDNA 3 support improving but not mature)
- ONNX Runtime + Vulkan (more universal AMD support, potentially slower)
- Direct ROCm if sherpa-onnx adds support

CPU performance is not a blocker. GPU is a future optimization.

= Streaming & Text Injection

== The "Deepgram Feel"

The gold standard for dictation UX is *semi-streaming* --- the model buffers just enough context to:

+ Resolve hesitations (drop "um", "uh")
+ Infer sentence boundaries and add punctuation
+ Occasionally revise previously emitted text when more context clarifies meaning

This requires a transducer model with endpoint detection, partial result emission, and the ability to revise partial hypotheses. Parakeet TDT and Nemotron Streaming both have this capability at the model level.

== Partial-Overwrite Typing

The implementation gap is in the text injection layer. The target behavior:

+ As partials arrive, type them into the active window via ydotool (Wayland) or xdotool (X11)
+ When a revised partial arrives, send backspace keystrokes to erase the old partial, then type the new one
+ When endpoint is detected, commit the final text and stop revising

This is toggleable --- users who find the visual rewriting distracting can disable it and fall back to the current behavior (type only on final endpoint).

== The Enter Key Problem

If transcribed text contains a newline character and gets injected into a chat window or terminal, it could inadvertently submit a message or execute a command. The text injection layer must *never* send Enter / Return. All `\\n` and `\\r` must be stripped or converted to spaces before injection.

== Key Risks

#table(
  columns: (1.2fr, 2.5fr),
  inset: 6pt,
  align: (left, left),
  table.header(
    [*Risk*], [*Mitigation*],
  ),
  [Backspace races with user typing], [Only overwrite while model is actively emitting; freeze on endpoint],
  [ydotool latency causes flicker], [Batch backspace + type into single call if possible],
  [Partial revision too aggressive], [Make partial-overwrite optional; tune silence thresholds],
  [Enter/newline in output], [Strip all newlines unconditionally before injection],
)

= Properties to Watch in New Models

When evaluating future ASR models for this project:

#table(
  columns: (1.2fr, 2.5fr),
  inset: 6pt,
  align: (left, left),
  table.header(
    [*Property*], [*Why It Matters*],
  ),
  [Streaming capable], [Text must appear as user speaks --- needs transducer or streaming CTC],
  [Built-in punctuation], [Dictation needs periods, commas --- model must be trained with them],
  [Endpointing], [Must detect sentence boundaries via configurable silence thresholds],
  [Partial hypothesis revision], [Correcting early guesses with more context improves accuracy],
  [ONNX export], [Required for our inference engine (sherpa-onnx)],
  [Frame-by-frame processing], [Enables real-time without buffering hacks],
  [0.3--1 B parameters], [Sweet spot: fast enough for real-time, large enough for accuracy],
)

#v(2em)
#line(length: 100%, stroke: 0.5pt + luma(180))
#text(size: 8pt, fill: luma(120))[
  Generated 26 March 2025 · Parakeet Dictation project planning
]
