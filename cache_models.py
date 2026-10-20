import logging
from transformers import VitsModel, AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cache_models")

LANGUAGES = ["amh", "orm", "tir", "som"]

def main():
    for lang in LANGUAGES:
        model_name = f"facebook/mms-tts-{lang}"
        logger.info(f"Pre-caching {model_name} into image...")
        AutoTokenizer.from_pretrained(model_name)
        VitsModel.from_pretrained(model_name)
    logger.info("All MMS models successfully cached into Docker image.")

if __name__ == "__main__":
    main()
