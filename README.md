# Evaluating and Enhancing Adversarial Robustness of ML Models for NIDS (CIC-IDS2017)

**Applicant:** Hitvika Pathak  
**Programme:** IIT Kanpur B.Cyber  
**Repository:** https://github.com/hitvika/adversarial-nids-cicids2017

## Project Goal

Analyze vulnerability of ML-based network intrusion detection to adversarial attacks and implement defenses with practical security relevance.

## Repository Structure

```text
adversarial-nids-cicids2017/
├── data/
│   ├── raw/                 # CIC-IDS2017 CSV (auto-downloaded)
│   └── processed/           # processed splits + profile
├── src/
│   ├── config.py
│   ├── data.py
│   ├── models.py
│   ├── attacks.py
│   ├── defenses.py
│   ├── evaluation.py
│   ├── visualization.py
│   └── pipeline.py
├── notebooks/
├── results/
│   ├── screenshots/
│   └── *.png, *.csv, metrics.json
├── docs/
│   ├── personal_reflection.md
│   └── demo_video_script.md
├── scripts/
│   ├── generate_report.py
│   └── generate_one_page_summary.py
├── run.py
└── requirements.txt
```

## Quick Start

```bash
pip install -r requirements.txt
python run.py
python scripts/generate_report.py
python scripts/generate_one_page_summary.py
```

## Reproducibility

- Seed: `42`
- Split: `72/8/20` train/val/test (stratified)
- Primary epsilon: `0.05` (L∞)
- PGD: `20` steps, step size `0.01`

## Key Findings (from latest run)

See `results/metrics.json` and `results/summary_table.csv`.

## Deliverables

1. Main report: `Adversarial_Robustness_CIC_IDS2017_Project_Report_FINAL.docx`
2. One-page summary: `Adversarial_NIDS_One_Page_Summary_IITK_BCyber.docx`
3. Figures: confusion matrices + epsilon curve in `results/`
4. Terminal screenshot: `results/screenshots/terminal_output.png`
5. Personal reflection: `docs/personal_reflection.md`
6. Demo script: `docs/demo_video_script.md`

## License

Academic portfolio project for IIT Kanpur B.Cyber application.