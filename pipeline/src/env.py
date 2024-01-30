import os


# For logger
LOGLEVEL = int(os.getenv("LOGLEVEL", "10"))
LOGFILE_PATH = os.getenv("LOGFILE_PATH", "./runtime.log")

CONFIG_PATH = os.getenv("CONFIG_PATH", "../setting.yml")
RESULT_DIR = os.getenv("RESULT_DIR", "../results")
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "2024"))

DEVICE = os.getenv("DEVICE", "cuda")
