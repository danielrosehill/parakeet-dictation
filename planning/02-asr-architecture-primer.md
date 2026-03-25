# ASR Architecture Primer — What We're Looking For

## This Is Still ASR, But a Specific Kind

All speech-to-text models are "ASR" (Automatic Speech Recognition), but there are
fundamentally different architectures with very different properties. Understanding
the distinctions is key to evaluating new models as the space evolves.

## The Three Main ASR Architectures

### 1. Encoder-Decoder (Seq2Seq) — e.g. Whisper, Canary (full)

```
[Audio] → Encoder → [hidden states] → Decoder → [text tokens one at a time]
```

- The encoder processes the **entire audio segment** into a fixed representation
- The decoder then generates text auto-regressively (one token at a time)
- Must wait for full utterance before starting — **inherently non-streaming**
- Excellent for transcription of recordings, podcasts, meetings
- Whisper, Google USM, and Meta's SeamlessM4T use this

**Why it's wrong for us:** You speak a sentence, wait, then all the text appears at once.
No partial feedback. No way to type text as the user speaks.

### 2. CTC (Connectionist Temporal Classification) — e.g. Canary Flash, Conformer-CTC

```
[Audio frame] → Encoder → [one label per frame] → CTC decode → [text]
```

- Processes audio **frame-by-frame** — each frame independently predicts a token
- Very fast inference, simple decode step
- Can technically stream (emit tokens per frame)
- **Weakness:** each frame's prediction is independent — no language model, so it can't
  use future context to correct past predictions. Punctuation and casing are harder.
- Good for keyword spotting, command recognition, lightweight ASR

**Relevance to us:** Usable for real-time, but accuracy ceiling is lower than transducers.
The Canary 180M Flash model uses this — good as a lightweight fallback.

### 3. Transducer (RNN-T / TDT) — e.g. Parakeet TDT, Nemotron Streaming ★

```
[Audio frame] → Encoder → ┐
                           ├→ Joint Network → [token or blank]
[Previous tokens] → Predictor → ┘
```

- Processes audio **frame-by-frame** like CTC
- But also has a **predictor** (language model) that conditions on previously emitted tokens
- This means it can use linguistic context: "I went to the ___" helps predict "store" vs "stir"
- Emits tokens incrementally as audio arrives — **true streaming**
- The "joint network" decides at each frame: emit a token, or emit blank (wait for more audio)
- **TDT (Token-and-Duration Transducer)** is NVIDIA's improvement that also predicts
  token duration, improving accuracy further

**Why this is what we want:** Frame-by-frame processing with linguistic context.
Text appears as you speak. The model can infer sentence boundaries, add punctuation,
and resolve ambiguities using both acoustic and language information.

## The Key Properties to Look For

When evaluating new ASR models for this project, check for:

| Property | Why It Matters | What To Look For |
|----------|---------------|------------------|
| **Streaming capable** | Text must appear as user speaks | Transducer or streaming CTC architecture |
| **Built-in punctuation** | Dictation needs periods, commas | Model trained with punctuation in vocabulary |
| **Endpointing** | Must detect when a sentence ends | Configurable silence thresholds, endpoint detection |
| **Partial hypothesis revision** | Correcting early guesses improves accuracy | Model supports emitting and revising partial results |
| **ONNX export available** | Needed for our inference engine (sherpa-onnx) | Published ONNX weights on HuggingFace |
| **Frame-by-frame processing** | Enables real-time without buffering hacks | Not encoder-decoder; should be transducer or CTC |
| **Parameter count 0.3B-1B** | Sweet spot for desktop GPU | Small enough for fast inference, large enough for accuracy |

## The "Deepgram Feel" — Semi-Streaming

The gold standard for dictation UX (as seen in Deepgram Nova) is what might be called
**semi-streaming** or **look-ahead streaming**:

1. Text appears quickly as you speak (low latency partials)
2. The model buffers just enough context to resolve ambiguities
3. Previously emitted text may be **revised** when more context clarifies it
4. Hesitations ("um", "uh") are dropped
5. Sentence boundaries and punctuation are inferred from prosody + silence

This requires a transducer model with:
- Endpoint detection (configurable silence thresholds)
- Partial result emission
- The ability to revise/rewrite partial hypotheses

Parakeet TDT and Nemotron Streaming both have this capability at the model level.
The implementation gap is in the **text injection layer** — the app needs to support
backspacing out old partials and retyping corrected text.

## Watching the Space

As new models appear, the key questions are:

1. **Is it a transducer or encoder-decoder?** If encoder-decoder, skip it for this use case.
2. **Does it have ONNX exports?** If not, it's locked to PyTorch/CUDA and likely can't
   run on our AMD GPU setup.
3. **Does it include punctuation in training?** Many academic ASR models don't — they
   output lowercase unpunctuated text, which is useless for dictation.
4. **What's the parameter count?** Under 100M is probably too small for good accuracy.
   Over 2B is probably too slow for real-time on our hardware. 0.3B-1B is the sweet spot.
5. **Does it support partial revision?** This is the feature that separates "demo-quality"
   streaming from "production-quality" dictation.

The NVIDIA NeMo ecosystem (Parakeet, Nemotron, Canary) is currently the strongest
open-source option for this specific niche. Keep an eye on it, plus any new entrants
from Meta (who have strong ASR research) or Google (who may open-source USM variants).
