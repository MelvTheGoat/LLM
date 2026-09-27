from gptlab.data.clean import (
    CleanConfig,
    CleaningStats,
    Deduper,
    clean_document,
    doc_hash,
    is_val,
    normalize,
)

GOOD = "This is a normal paragraph about plants and sunlight. " * 10


def test_normalize():
    assert normalize("  a\r\nb  \r\n\n\n\nc\t\n") == "a\nb\n\nc"
    # "e" + combining acute accent becomes the single character "é" (NFC).
    assert normalize("café") == "café"


def test_rejection_reasons():
    cfg = CleanConfig()
    assert clean_document(GOOD, cfg)[1] is None
    assert clean_document("too short", cfg)[1] == "too_short"
    assert clean_document("a" * (cfg.max_chars + 1), cfg)[1] == "too_long"
    broken = GOOD + "�" * 10
    assert clean_document(broken, cfg)[1] == "bad_characters"
    numbers = "12 34 56 78 90 | " * 30
    assert clean_document(numbers, cfg)[1] == "few_letters"
    # Non-Latin letters count as letters.
    assert clean_document("这是一个关于植物和阳光的普通段落。" * 20, cfg)[1] is None


def test_exact_dedup_after_normalization():
    cfg = CleanConfig()
    base = "A line about the water cycle and rain.\n" * 10
    # Same text with Windows line endings and trailing spaces: a duplicate.
    variant = base.replace("\n", "   \r\n")
    different = base + "One more line."
    d = Deduper()
    assert not d.is_duplicate(doc_hash(clean_document(base, cfg)[0]))
    assert d.is_duplicate(doc_hash(clean_document(variant, cfg)[0]))
    assert not d.is_duplicate(doc_hash(clean_document(different, cfg)[0]))
    assert len(d) == 2


def test_split_is_deterministic_and_close_to_the_fraction():
    hashes = [doc_hash(f"document number {i}") for i in range(20000)]
    val = [is_val(h, 0.05) for h in hashes]
    assert val == [is_val(h, 0.05) for h in hashes]
    assert 0.04 < sum(val) / len(val) < 0.06
    assert not any(is_val(h, 0.0) for h in hashes)


def test_stats_keep_counts_and_examples():
    s = CleaningStats(max_examples=2)
    for i in range(5):
        s.add_input(10, changed=i % 2 == 0)
        s.add_removed("too_short", f"doc {i}")
    out = s.to_dict()
    assert out["docs_in"] == 5
    assert out["docs_changed_by_normalize"] == 3
    assert out["removed_docs"] == {"too_short": 5}
    assert out["examples"]["too_short"] == ["doc 0", "doc 1"]
