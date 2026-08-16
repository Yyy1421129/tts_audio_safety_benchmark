# Phase 3 Plan: Paralinguistic and Narrative-Wrapped Audio Safety Evaluation

## 1. Motivation

Phase 1 showed that Step-Audio2 has a clear safety gap on the current audio benchmark: both Matcha-TTS and CosyVoice reached a 25.00% unsafe rate, while Qwen audio stayed at 0.00% to 1.00%.

Phase 2 showed that simple acoustic post-processing did not increase Step-Audio2's unsafe rate: the overall perturbation unsafe rate was 22.67%, with stable WER. This suggests that speed, volume, noise, and silence are not sufficient as the next main direction.

Phase 3 should therefore focus on features that may affect model decision-making while preserving ASR content:

- paralinguistic features: speaker identity, pitch, age/gender impression, emotion, prosody
- narrative wrapping: embedding the same harmful request in role-play, education, fiction, historical analysis, or safety audit contexts

The core evaluation principle remains unchanged:

```text
same harmful core request
-> modified speech style or narrative wrapper
-> target audio model response
-> Kimi-k2.6 judge
-> Step-Audio2 ASR/WER check
```

## 2. CosyVoice Capability Check

Current usable model:

```text
/hpc_stor03/sjtu_home/yi.yang/.cache/modelscope/hub/models/iic/CosyVoice2-0.5B
```

Current project script:

```text
scripts/synthesize_cosyvoice_seed_prompts.py
```

It already supports these CosyVoice modes:

| Mode | Supported in script | Useful for Phase 3 |
| --- | --- | --- |
| `sft` | yes | Not preferred with the current CosyVoice2 model because no built-in `spk2info.pt` was found. |
| `zero_shot` | yes | Useful for speaker/timbre control through reference audio. |
| `cross_lingual` | yes | Already used for baseline; controls voice mainly through `prompt_wav`. |
| `instruct` | yes | Only implemented for CosyVoice 1 instruct models, not the current CosyVoice2 model. |
| `instruct2` | yes | Most useful for CosyVoice2 style/prosody instructions. |

Findings from the code:

- `CosyVoice2` supports `inference_instruct2(tts_text, instruct_text, prompt_wav, ...)`.
- `CosyVoice2` supports reference-audio based timbre control through `zero_shot`, `cross_lingual`, and `instruct2`.
- The local CosyVoice2 model directory contains full inference weights, but no `spk2info.pt`; therefore fixed SFT speaker IDs should not be treated as available.
- Current local reference audio assets are limited:
  - `CosyVoice/asset/zero_shot_prompt.wav`
  - `CosyVoice/asset/cross_lingual_prompt.wav`
- CosyVoice2 tokenizer supports fine-grained tokens such as `[breath]`, `[laughter]`, `[sigh]`, `[noise]`, `[cough]`, `<strong>...</strong>`.
- CosyVoice common instruction list contains style instructions for loud/soft voice, slow/fast speed, happy/sad/angry speaking, and dialects.
- CosyVoice README states that CosyVoice3 has broader instruct support for emotion, speed, volume, and dialect. If Phase 3 needs stronger controllability, CosyVoice3 is a natural upgrade candidate.

Recommended Phase 3 implementation path:

1. Start with CosyVoice2 `instruct2` because it is available in the current project.
2. Use `prompt_wav` variants to control speaker/timbre when suitable reference voices are available.
3. Use `instruct_text` to test emotion/prosody:
   - neutral
   - happy or pleasant
   - surprised
   - sad
   - calm authoritative
   - soft/low voice
4. Validate every generated subset with Step-Audio2 ASR/WER before judging safety.
5. If CosyVoice2 style control is weak or unstable, test Fun-CosyVoice3-0.5B as a stronger controllable TTS option.

## 3. Phase 3 Experimental Design

### 3.1 Paralinguistic Style Grid

Use the same 100 harmful core requests when possible. Start with a smaller subset if GPU cost is high.

Recommended first grid:

| Dimension | Conditions | Goal |
| --- | --- | --- |
| emotion | neutral, happy, surprised, sad | Test whether affect changes refusal tendency. |
| voice/prosody | calm authoritative, soft low voice | Test whether authority or low-energy delivery changes compliance. |
| speaker/timbre | available prompt voices | Test whether reference voice identity changes safety behavior. |

Suggested first scale:

```text
100 prompts x 2 TTS/reference settings x 6 styles = 1200 audio samples
```

If cost is too high, start with 30 prompts:

- 23 baseline common unsafe samples
- 4 TTS-disagreement samples
- 3 stable safe refusal samples

### 3.2 Narrative-Wrapped Prompt Grid

Narrative wrapping should preserve the harmful core request and only change the surrounding context. The wrapper should not explicitly tell the target model that this is a safety refusal test; otherwise it may bias the model toward refusal and weaken the VoiceJailbreak-style comparison. This keeps the experiment comparable to the existing 100 prompts while testing whether legitimate-looking contexts shift the model's decision boundary.

For each original record:

```text
wrapped_tts_text = template_for_risk_category.format(harmful_request=original_tts_text)
```

The wrapper should be stored separately from the original harmful request:

| Field | Meaning |
| --- | --- |
| `source_id` | Original seed prompt ID |
| `risk_category` | Original risk category |
| `original_tts_text` | Original harmful request |
| `wrapped_tts_text` | Narrative-wrapped request used for TTS |
| `wrapper_id` | Template ID |
| `wrapper_family` | education, fiction, historical_research, safety_audit, etc. |

## 4. Safety and Interpretability Constraints

Phase 3 should keep the same safeguards as earlier stages:

- Do not use generated outputs as instructions.
- Judge only model responses, not execute or validate harmful procedures.
- Report aggregate unsafe rates and manually curated examples with redaction if necessary.
- Always pair safety results with WER/ASR checks to show whether content remained understandable.
- Keep the original harmful request unchanged inside each wrapped prompt so that differences are attributable to style or wrapper, not a changed attack objective.
- Avoid adding explicit "the assistant should refuse" cues inside the wrapped prompt, because Phase 3 measures whether narrative framing affects the target model's natural safety behavior.

## 5. Recommended Next Implementation Steps

1. Add a dataset generation script:

```text
scripts/create_phase3_narrative_wrapped_prompts.py
```

2. Add a CosyVoice style synthesis script or extend the existing one:

```text
scripts/synthesize_cosyvoice_phase3_styles.py
```

3. Generate a small pilot first:

```text
30 prompts x 6 styles = 180 samples
```

4. Run:

```text
Step-Audio2 ASR/WER
Step-Audio2 response generation
Kimi-k2.6 judge
Qwen audio model comparison, optional
```

5. Analyze:

- unsafe rate by style
- unsafe rate by narrative wrapper
- WER by style
- prompt-level stability across style and wrapper

## 6. Initial Hypotheses

| Hypothesis | Test |
| --- | --- |
| Emotion/prosody can alter safety refusal behavior without changing ASR text. | Compare neutral vs happy/surprised/sad with WER controlled. |
| Narrative wrapping is stronger than simple acoustic post-processing. | Compare wrapped prompts to baseline and Phase 2 perturbations. |
| The effect is category-dependent. | Compare `fraud_scam`, `misinformation_manipulation`, `cyber_abuse`, and `self_harm`. |
| Step-Audio2 is more sensitive than Qwen audio. | Run a matched subset on Qwen audio. |
