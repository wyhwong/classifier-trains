import os


# For logger
LOGLEVEL = int(os.getenv("LOGLEVEL", "20"))
LOGFILEPATH = os.getenv("LOGFILEPATH", "./runtime.log")

CONFIGPATH = os.getenv("CONFIGPATH", "./config.yml")
RESULTDIR = os.getenv("RESULTDIR", "./results")
RANDOMSEED = int(os.getenv("RANDOMSEED", "2024"))

DEVICE = os.getenv("DEVICE", "cuda")
