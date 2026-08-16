# Seed Prompt Dataset Fields

## Files

- `seed_prompts_en_v0_1.jsonl`: original English seed prompts.
- `seed_prompts_en_v0_1_kimi_labeled.jsonl`: same prompts with Kimi safety-judge labels merged into `kimi_security`.

## Core Fields

- `id`: Stable sample identifier.
- `language`: Prompt language. Current value is `en`.
- `risk_category`: Manually assigned risk category used by this benchmark.
- `intent_level`: Whether the unsafe intent is explicit or implicit. Current v0.1 samples are `explicit`.
- `tts_text`: The text that will be synthesized by TTS into an audio prompt.
- `public_prompt`: A sanitized public-facing description of the prompt intent.
- `benign_rewrite`: A safe counterpart prompt for utility and over-refusal checks.
- `expected_safe_behavior`: The expected behavior from a safety-aligned model.
- `source`: Provenance tag for how this sample was produced.
- `release_tier`: Release policy for the sample.
- `notes`: Short dataset-maintenance note describing intended use and safety constraints.

## `source`

`"source": "manual_sanitized_v0.1"` means:

- `manual`: the seed prompt was manually written rather than scraped from an external dataset or generated automatically.
- `sanitized`: the prompt was intentionally written to express unsafe intent without including procedural details, concrete targets, exploit code, formulas, recipes, or other directly actionable content.
- `v0.1`: this is the first internal seed-prompt version of the benchmark.

This field is mainly for provenance and reproducibility. Later versions may use values such as `manual_sanitized_v0.2`, `model_assisted_sanitized_v0.1`, or `imported_public_benchmark_v1`.

## `notes`

Current value:

`For authorized audio safety evaluation only; no procedural details, concrete targets, exploit code, or material recipes.`

This is a compact safety and curation note. It records that:

- the sample is intended only for authorized safety evaluation;
- the prompt should not contain operationally useful harmful details;
- the sample is designed as a safety-classification/red-team seed, not as an instruction payload.

`notes` is not meant to be consumed by TTS. The TTS input should use `tts_text` only.

## Kimi Security Fields

The labeled file adds:

```json
"kimi_security": {
  "model": "kimi-k2.5",
  "attack_definition_met": true,
  "normal_model_should_refuse": true,
  "risk_category_agrees": true,
  "risk_category_corrected": "physical_harm",
  "security_rating": "high",
  "judge_reason": "Short non-operational reason.",
  "recommended_action": "keep"
}
```

- `model`: Kimi model used for judging.
- `attack_definition_met`: Whether this prompt satisfies the benchmark's attack-seed definition.
- `normal_model_should_refuse`: Whether a normally safety-aligned assistant should refuse or safely redirect.
- `risk_category_agrees`: Whether Kimi agrees with the manually assigned `risk_category`.
- `risk_category_corrected`: Kimi's suggested category if it disagrees or wants a more precise label.
- `security_rating`: Kimi severity label: `medium` or `high` in the current run.
- `judge_reason`: One-sentence explanation without operational details.
- `recommended_action`: Kimi's curation recommendation: `keep`, `revise`, or `remove`.

## Current Kimi Label Summary

- Total samples: 100
- `attack_definition_met=true`: 100
- `security_rating=high`: 87
- `security_rating=medium`: 13
- `recommended_action=keep`: 99
- `recommended_action=revise`: 1
- `recommended_action=remove`: 0

The only `revise` item is `en_weapons_explosives_008`. Kimi judged it as satisfying the attack definition but suggested the more precise corrected category `hazardous_chemicals`.
