"""Entry point for the background ingestion worker."""
import logging
from bookrag.ingestion.worker import run_worker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

if __name__ == "__main__":
    run_worker()
