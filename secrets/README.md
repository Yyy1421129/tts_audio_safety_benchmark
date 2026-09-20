# Secrets (do not commit)

API keys and tokens live only on your machine / rental box.

1. Copy the examples below into local files (already gitignored via `secrets/`):
   - `secrets/kimi_api_key.txt`
   - `secrets/sjtu_models_api_key.txt`
2. `chmod 600 secrets/*_api_key.txt`
3. **Rotate any key that was ever committed or pasted in chat.**

Historical note: older commits may still contain key files. Rotating provider-side keys is required even after `git rm --cached`.
