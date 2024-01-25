import os


# For logger
LOGLEVEL = int(os.getenv("LOGLEVEL", "20"))
LOGFILEPATH = os.getenv("LOGFILEPATH", "./runtime.log")
DEVICE = os.getenv("DEVICE", "cuda")
