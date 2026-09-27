import random

from gptlab.data.tokenizer import EOT, Tokenizer, bytes_per_token, train_tokenizer

TRICKY_TEXTS = [
    "",
    " ",
    "Hello, world!",
    "  two leading spaces and trailing  ",
    "tabs\tand\nnew\nlines\r\nwindows endings\r\n",
    "Émile a mangé des crêpes à Noël.",
    "中文字符和日本語のテキスト",
    "Emoji: 🤖🚀👍🏽 and flags 🇯🇵",
    "Math: ∑ x² ≥ 0, π ≈ 3.14159",
    "עברית ועربية",
    "Numbers 1234567890 and 3.14e-10",
    "A literal <|endoftext|> inside normal text",
    "​zero width‍ joiner",
    "code: def f(x):\n    return x ** 2\n",
]


def test_round_trip_on_sample_documents(small_tokenizer, sample_docs):
    for doc in sample_docs:
        assert small_tokenizer.decode(small_tokenizer.encode(doc)) == doc


def test_round_trip_on_tricky_text(small_tokenizer):
    for text in TRICKY_TEXTS:
        assert small_tokenizer.decode(small_tokenizer.encode(text)) == text


def test_round_trip_on_random_unicode(small_tokenizer):
    rng = random.Random(0)
    for _ in range(200):
        chars = []
        for _ in range(rng.randint(1, 40)):
            cp = rng.choice([rng.randint(32, 126), rng.randint(0xA0, 0x2FFF), rng.randint(0x1F300, 0x1F6FF)])
            chars.append(chr(cp))
        text = "".join(chars)
        assert small_tokenizer.decode(small_tokenizer.encode(text)) == text


def test_vocab_size_and_end_of_text_token(small_tokenizer):
    assert small_tokenizer.vocab_size == 512
    assert small_tokenizer.eot_id == 0
    # A literal "<|endoftext|>" in text must NOT become the special token.
    ids = small_tokenizer.encode("before " + EOT + " after")
    assert small_tokenizer.eot_id not in ids
    # But decoding the special id gives the marker back.
    assert small_tokenizer.decode([small_tokenizer.eot_id]) == EOT


def test_token_bytes_match_utf8_length(small_tokenizer, sample_docs):
    table = small_tokenizer.token_bytes()
    assert table[small_tokenizer.eot_id] == 0
    for text in TRICKY_TEXTS + sample_docs[:20]:
        ids = small_tokenizer.encode(text)
        assert int(table[ids].sum()) == len(text.encode("utf-8"))


def test_save_and_load_give_identical_encodings(small_tokenizer, sample_docs, tmp_path):
    path = tmp_path / "tok.json"
    small_tokenizer.save(path)
    loaded = Tokenizer.load(path)
    assert loaded.vocab_size == small_tokenizer.vocab_size
    for doc in sample_docs[:10] + TRICKY_TEXTS:
        assert loaded.encode(doc) == small_tokenizer.encode(doc)


def test_bigger_vocab_compresses_better(sample_docs):
    small = train_tokenizer(sample_docs, vocab_size=300)
    big = train_tokenizer(sample_docs, vocab_size=1000)
    assert bytes_per_token(big, sample_docs) > bytes_per_token(small, sample_docs) > 1.0
