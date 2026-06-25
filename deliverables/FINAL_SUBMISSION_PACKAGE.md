# Final Submission Package — Hitvika Pathak

## Status: Ready to submit

All code, experiments, reports, and screenshots are complete. The repository is live on GitHub.

## GitHub repository

```
https://github.com/Hitvikapathak/adversarial-nids-cicids2017
```

## Files to attach in IITK application

| # | Item | File |
|---|------|------|
| 1 | Main report (PDF) | `deliverables/Adversarial_Robustness_CIC_IDS2017_Project_Report_FINAL.pdf` |
| 2 | One-page summary (DOCX) | `deliverables/Adversarial_NIDS_One_Page_Summary_IITK_BCyber.docx` |
| 3 | GitHub URL | `https://github.com/Hitvikapathak/adversarial-nids-cicids2017` |
| 4 | Demo video (optional) | Record using `docs/DEMO_RECORDING_GUIDE.md`, upload unlisted to YouTube |

## Key result (for quick reference)

Random Forest: **97.5%** clean accuracy → **0.0%** attack detection under transferred PGD.

MLP after adversarial training: **97.0%** PGD detection rate.

## Reproduce in one command

```bash
pip install -r requirements.txt
python run.py
```