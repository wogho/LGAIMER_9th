# REF4 126 Colab handoff

- Experiment: `REF4-JM-R-RESIDUAL-STRICT-GPU-126`
- Execution split: local preparation/audit/package; Colab T4 strict-forward training
- Google Drive remote root: user-provided shared folder
- Remote subdirectory: `REF4_126/`
- Upload policy: public competition data/code only; immutable copies; no remote deletion
- Primary anchor: official-best 113A
- Candidate correction: Regular-only shallow CatBoost residual; Futures must remain exact identity
- Promotion: strict 2023/2024 forward gains, pitcher-cluster bootstrap, deployable feature parity

Large files are accompanied by SHA-256 checksums. Colab outputs and checkpoints must be written under `REF4_126/results/` and `REF4_126/checkpoints/`.

## Colab launch

1. Add the shared `LG aimer` folder as a shortcut under `My Drive`.
2. Open `REF4_126_T4.ipynb` in Colab and select a T4 runtime.
3. Keep `DRIVE_ROOT_TEXT=/content/drive/MyDrive/LG aimer/REF4_126`.
4. Run all cells once. On disconnect, reconnect and rerun with `RESUME=True`.

The runner reads the frozen 113A strict OOF from `anchor/strict_113A/oof_predictions.csv`. It never reads `test.csv`; it creates no submission ZIP unless a later, separate production step is explicitly approved.
