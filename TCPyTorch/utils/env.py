import yaml
import os
from glob import glob


DEVICE = os.getenv("DEVICE")

LOGLEVEL = int(os.getenv("LOGLEVEL", "20"))

LABEL = yaml.safe_load(open("/train.yml")).get("experiment_label", "noLabel")

OUTPUT_DIR = f"/results/{int(len(glob('/results/*'))+1)}_{LABEL}"
os.mkdir(OUTPUT_DIR)
