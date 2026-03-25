# Streaming Behavior & Text Injection — Design Notes

## The Problem

Current app behavior:
- **Offline mode (Parakeet TDT):** VAD detects speech → waits for silence → transcribes
  entire segment → types final text. Works well but has perceptible delay.
- **Streaming mode (Nemotron):** Partials shown in status bar only. Final text typed
  after endpoint detection. No text appears in the active window until utterance is complete.

Neither mode gives the "Deepgram feel" where text flows into the document as you speak.

## Target Behavior: Partial-Overwrite Typing

The goal is to type partial transcriptions into the active window and revise them as
the model refines its hypothesis:

```
User speaks: "I think we should consider the..."

Time 0.3s: types "I"
Time 0.5s: types " think"
Time 0.8s: types " we should"
Time 1.0s: types " consider"         ← partial, might change
Time 1.2s: backspaces "consider", types "consider the"  ← revised
Time 1.8s: [silence detected] → commit, stop revising
```

### Implementation Approach

1. Track the **last typed partial** (character count)
2. When a new partial arrives:
   - Send N backspace keystrokes (via ydotool/xdotool) to erase the old partial
   - Type the new partial
3. When endpoint is detected:
   - The current text becomes final — stop tracking for backspace
   - Add trailing space
   - Reset partial tracker

### Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Backspace races with user typing | Deletes user's own text | Only overwrite while model is actively emitting; freeze on endpoint |
| ydotool latency | Visible flicker during rewrites | Batch backspace+type into single ydotool call if possible |
| Application doesn't accept synthetic input | Text doesn't appear | Already a known issue; ydotool works on Wayland, xdotool on X11 |
| Partial revision is too aggressive | Text constantly changes, distracting | Make partial-overwrite optional; tune silence thresholds to reduce revisions |
| Enter/newline in transcribed text | Could trigger form submission or send a message | Strip all newlines; this is dictation, not formatting |

### Configuration

Add to AppConfig:
- `partial_overwrite: bool = True` — enable/disable the rewrite behavior
- Can be toggled in settings if users find it distracting

### The Enter Key Problem

A critical robustness concern: if the model transcribes something that gets injected as
an Enter keystroke, it could submit a chat message, execute a command, etc. The text
injection layer must **never** send Enter/Return. All newlines must be stripped or
converted to spaces before injection.

Current `TextTyper.type_text()` already calls `.strip()`, but this only strips
leading/trailing whitespace. Need to also replace internal `\n` and `\r`.

## Decision

**Yes, implement partial-overwrite typing.** The accuracy improvement from allowing the
model to revise partial hypotheses is significant — it's the difference between getting
"um" and hesitations in the output vs. clean sentences. Make it toggleable in settings
for users who find the visual rewriting distracting.
