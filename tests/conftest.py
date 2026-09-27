import json
from pathlib import Path

import pytest

DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def sample_docs() -> list[str]:
    """About 150 short documents of public-domain text (Alice in Wonderland)."""
    with open(DATA_DIR / "sample_docs.jsonl", encoding="utf-8") as f:
        return [json.loads(line)["text"] for line in f]


@pytest.fixture(scope="session")
def small_tokenizer(sample_docs):
    from gptlab.data.tokenizer import train_tokenizer

    return train_tokenizer(sample_docs, vocab_size=512)
