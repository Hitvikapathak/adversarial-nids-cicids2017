# IITK B.Cyber Submission Checklist

## Included artifacts

- [x] Main report (DOCX): `Adversarial_Robustness_CIC_IDS2017_Project_Report_FINAL.docx`
- [x] Main report (PDF): `Adversarial_Robustness_CIC_IDS2017_Project_Report_FINAL.pdf`
- [x] One-page summary (DOCX): `Adversarial_NIDS_One_Page_Summary_IITK_BCyber.docx`
- [x] GitHub repository: https://github.com/Hitvikapathak/adversarial-nids-cicids2017
- [x] Experiment screenshots: `results/screenshots/terminal_output.png`
- [x] Result graphs: confusion matrices + epsilon curve in `results/`
- [x] Personal reflection: `docs/personal_reflection.md`
- [x] Demo video script: `docs/demo_video_script.md`

## Submission steps

1. Upload **PDF report** to IITK application portal.
2. Upload **one-page summary DOCX** to IITK application portal.
3. Paste **GitHub URL**: `https://github.com/Hitvikapathak/adversarial-nids-cicids2017`
4. *(Optional)* Record 2-minute demo using `docs/DEMO_RECORDING_GUIDE.md` and add YouTube link.

## Reproduce

```bash
python run.py
python scripts/generate_report.py
python scripts/generate_one_page_summary.py
```