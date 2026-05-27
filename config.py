MODEL_NAME = "google/gemma-2-2b-it"
ANTHROPIC_MODEL = "claude-sonnet-4-20250514"

# Paths
DATA_DIR = "data/"
ACTIVATIONS_DIR = "activations/"
MODELS_DIR = "models/"
OUTPUTS_DIR = "outputs/"

# Dataset
N_CANDIDATES = 150       # prompts to generate before filtering
N_TARGET = 100           # verified examples to keep (50 per class)
MAX_NEW_TOKENS = 200     # max tokens for model response generation

# Probe
CV_FOLDS = 5             # cross-validation folds for logistic regression

# Steering
STEERING_SCALES = [0.01, 0.05, 0.1]   # three scales to test
N_STEERING_TEST = 20                   # prompts to evaluate sycophancy rate
