import os

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
DEFAULT_THRESHOLDS = os.getenv("THRESHOLDS_PATH", os.path.join(os.path.dirname(__file__), "..", "thresholds.yml"))
