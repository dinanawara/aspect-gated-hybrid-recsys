"""
Configuration module - loads environment variables and provides settings.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "./output"))
MODELS_DIR = Path(os.getenv("MODELS_DIR", "./models"))

# Create directories if they don't exist
OUTPUT_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "")

# Ollama settings
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")

# LLM settings
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "256"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.0"))

# Processing settings
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))
NUM_WORKERS = int(os.getenv("NUM_WORKERS", "4"))

# Filtering thresholds
MIN_ITEM_INTERACTIONS = int(os.getenv("MIN_ITEM_INTERACTIONS", "20"))
MIN_USER_INTERACTIONS = int(os.getenv("MIN_USER_INTERACTIONS", "5"))
MIN_ITEM_RATINGS_FOR_VARIANCE = int(os.getenv("MIN_ITEM_RATINGS_FOR_VARIANCE", "50"))
TOP_VARIANCE_ITEMS = int(os.getenv("TOP_VARIANCE_ITEMS", "1500"))

# Dissonance detection keywords
LOGISTICS_KEYWORDS = [
    "shipping", "delivery", "arrived", "packaging", "box", "seller",
    "package", "shipped", "courier", "fedex", "ups", "usps", "amazon"
]

INCENTIVIZED_KEYWORDS = [
    "coupon", "discount", "free", "received for", "in exchange",
    "promo", "promotional", "gifted", "sample", "complimentary"
]

UNCERTAINTY_KEYWORDS = [
    "haven't used", "just received", "first impression", "will update",
    "too early", "just got", "haven't tried", "update later"
]

POSITIVE_WORDS = [
    "works", "great", "love", "perfect", "excellent", "amazing",
    "fantastic", "wonderful", "best", "awesome", "good quality"
]

NEGATIVE_WORDS = [
    "broke", "stopped", "returned", "refund", "defective", "broken",
    "failed", "terrible", "worst", "garbage", "waste", "junk"
]
