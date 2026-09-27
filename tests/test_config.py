import pytest

from gptlab.config import (
    ConfigError,
    ModelConfig,
    RunConfig,
    config_hash,
    load_config,
    save_config,
)


def write(tmp_path, text):
    p = tmp_path / "run.yaml"
    p.write_text(text)
    return p


def test_minimal_config_uses_defaults(tmp_path):
    cfg = load_config(write(tmp_path, "name: tiny\ntrain:\n  max_steps: 10\n"))
    assert cfg.name == "tiny"
    assert cfg.train.max_steps == 10
    assert cfg.model.d_model == ModelConfig().d_model


def test_unknown_key_is_an_error_with_suggestion(tmp_path):
    path = write(tmp_path, "name: tiny\ntrain:\n  max_steps: 10\n  warmup_step: 5\n")
    with pytest.raises(ConfigError, match="did you mean train.warmup_steps"):
        load_config(path)


def test_numbers_written_without_a_dot_are_accepted(tmp_path):
    # PyYAML reads 6e-4 as a string. The loader must still treat it as a float.
    path = write(tmp_path, "name: t\ntrain:\n  tokens: 2_000_000\n  lr: 6e-4\n")
    cfg = load_config(path)
    assert cfg.train.lr == pytest.approx(6e-4)
    assert cfg.train.tokens == 2_000_000


def test_bad_values_are_rejected(tmp_path):
    with pytest.raises(ConfigError, match="pos_encoding"):
        load_config(write(tmp_path, "name: t\ntrain: {max_steps: 1}\nmodel: {pos_encoding: alibi}\n"))
    with pytest.raises(ConfigError, match="divisible"):
        load_config(write(tmp_path, "name: t\ntrain: {max_steps: 1}\nmodel: {d_model: 100, n_head: 3}\n"))
    with pytest.raises(ConfigError, match="exactly one"):
        load_config(write(tmp_path, "name: t\ntrain: {max_steps: 1, tokens: 100}\n"))
    with pytest.raises(ConfigError, match="true or false"):
        load_config(write(tmp_path, "name: t\ntrain: {max_steps: 1, compile: 'yes'}\n"))


def test_overrides(tmp_path):
    path = write(tmp_path, "name: t\ntrain:\n  max_steps: 10\n")
    cfg = load_config(path, ["train.lr=1e-3", "model.n_layer=2", "model.qk_norm=true"])
    assert cfg.train.lr == pytest.approx(1e-3)
    assert cfg.model.n_layer == 2
    assert cfg.model.qk_norm is True


def test_saved_config_reloads_identically(tmp_path):
    cfg = load_config(write(tmp_path, "name: t\ntrain: {max_steps: 3}\nmodel: {mlp: gelu}\n"))
    out = tmp_path / "saved.yaml"
    save_config(cfg, out)
    again = load_config(out)
    assert again == cfg
    assert config_hash(again) == config_hash(cfg)


def test_hash_ignores_notes_but_not_settings():
    a = RunConfig(name="x")
    b = RunConfig(name="x", notes="a comment")
    c = RunConfig(name="x")
    c.train.lr = 1e-3
    assert config_hash(a) == config_hash(b)
    assert config_hash(a) != config_hash(c)


def test_swiglu_hidden_size_keeps_params_close_to_gelu():
    for d in [128, 256, 384, 512, 768]:
        gelu = ModelConfig(d_model=d, mlp="gelu")
        swiglu = ModelConfig(d_model=d, mlp="swiglu")
        gelu_params = 2 * d * gelu.resolved_mlp_hidden()
        swiglu_params = 3 * d * swiglu.resolved_mlp_hidden()
        assert abs(swiglu_params - gelu_params) / gelu_params < 0.05
        assert swiglu.resolved_mlp_hidden() % 32 == 0
