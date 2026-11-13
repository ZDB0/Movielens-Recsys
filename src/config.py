from pathlib import Path



# Корінь проєкту
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Шляхи до директорій та файлів
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "ml-1m"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

RATINGS_FILE = RAW_DATA_DIR / "ratings.dat"
MOVIES_FILE = RAW_DATA_DIR / "movies.dat"
USERS_FILE = RAW_DATA_DIR / "users.dat"

TRAIN_MATRIX_PATH = PROCESSED_DATA_DIR / "train_matrix.npz"
TEST_MATRIX_PATH = PROCESSED_DATA_DIR / "test_matrix.npz"
MAPPINGS_PATH = PROCESSED_DATA_DIR / "id_mappings.json"
MOVIES_CLEAN_PATH = PROCESSED_DATA_DIR / "movies.csv"
METRICS_SUMMARY_PATH = MODELS_DIR / "metrics_summary.json"

# Параметри підготовки даних
TEST_RATIO = 0.2
RANDOM_STATE = 42

# Гіперпараметри матричної факторизації та метрик
N_FACTORS = 64
ALS_REGULARIZATION = 0.05
ALS_ITERATIONS = 20
ALS_ALPHA = 15.0
TOP_K = 10