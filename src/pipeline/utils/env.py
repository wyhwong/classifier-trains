import os
import zoneinfo


# For logging
STREAMING_LOG_LEVEL = int(os.getenv("STREAMING_LOG_LEVEL", "30"))
FILE_LOG_LEVEL = int(os.getenv("FILE_LOG_LEVEL", "10"))
LOG_FILEPATH = os.getenv("LOG_FILEPATH", "./runtime.log")
LOG_FMT = os.getenv("LOG_FMT", "%(asctime)s [%(name)s | %(levelname)s]: %(message)s")
LOG_DATEFMT = os.getenv("LOG_DATEFMT", "%Y-%m-%dT%H:%M:%SZ")
TZ = zoneinfo.ZoneInfo(os.getenv("TZ") or "UTC")
