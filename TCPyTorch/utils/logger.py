import os
import yaml
import logging
from glob import glob

LOGLEVEL = int(os.getenv("LOGLEVEL", "20"))
LOGFMT = "%(asctime)s [%(name)s | %(levelname)s]: %(message)s"
DATEFMT = "%Y-%m-%d %H:%M:%S"
LABEL = yaml.safe_load(open("/train.yml")).get("experiment_label", "noLabel")
OUTPUT_DIR = f"/results/{int(len(glob('/results/*'))+1)}_{LABEL}"
os.mkdir(OUTPUT_DIR)
logging.basicConfig(format=LOGFMT, datefmt=DATEFMT, level=LOGLEVEL)


def get_logger(logger_name: str, log_filepath=f"{OUTPUT_DIR}/output.log") -> logging.Logger:
    logger = logging.getLogger(logger_name)
    logger.setLevel(LOGLEVEL)
    if log_filepath:
        handler = logging.FileHandler(filename=log_filepath)
        formatter = logging.Formatter(fmt=LOGFMT, datefmt=DATEFMT)
        handler.setFormatter(fmt=formatter)
        logger.addHandler(handler)
    return logger
