"""Project configuration and reproducibility constants."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
SCREENSHOTS_DIR = RESULTS_DIR / "screenshots"
MODELS_DIR = PROJECT_ROOT / "models"
DOCS_DIR = PROJECT_ROOT / "docs"

RANDOM_SEED = 42
EPSILONS = [0.01, 0.05, 0.1]
PRIMARY_EPSILON = 0.05
PGD_STEPS = 20
PGD_STEP_SIZE = 0.01
MAX_SAMPLES_PER_CLASS = 2500
TEST_SIZE = 0.2
VAL_SIZE = 0.1
ATTACK_EVAL_SAMPLES = 200
TOP_FEATURES = 30

DATASET_URL = (
    "https://huggingface.co/datasets/c01dsnap/CIC-IDS2017/resolve/main/"
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv"
)
DATASET_FILENAME = "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv"

DROP_COLUMNS = [
    "Flow ID",
    "Source IP",
    "Destination IP",
    "Timestamp",
    "Label",
]

PROJECT_TITLE = (
    "Evaluating and Enhancing Adversarial Robustness of Machine Learning Models "
    "for Network Intrusion Detection using the CIC-IDS2017 Dataset"
)