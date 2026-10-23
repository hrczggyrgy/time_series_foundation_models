"""Unit tests for registry model availability."""

import importlib.util

from forecasting.registry import ModelRegistry


def test_optional_model_entries_are_hidden_without_their_runtime_modules(monkeypatch):
    registry = ModelRegistry()
    original_find_spec = importlib.util.find_spec

    def find_spec(module_name, *args, **kwargs):
        if module_name in {"timesfm3", "tirex2", "chronos", "uni2ts"}:
            return None
        return original_find_spec(module_name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", find_spec)
    available = {entry.model_key for entry in registry.list_available()}

    assert not {"timesfm_3", "tirex_2", "chronos", "moirai"} & available


def test_optional_model_entries_are_registered_with_runtime_requirements():
    registry = ModelRegistry()

    assert registry.get_entry("timesfm_3").required_module == "timesfm3"
    assert registry.get_entry("tirex_2").required_module == "tirex2"
    assert registry.get_entry("chronos").required_module == "chronos"
    assert registry.get_entry("moirai").required_module == "uni2ts"


def test_stl_is_registered_as_the_only_builtin_baseline():
    registry = ModelRegistry()

    assert registry.get_entry("stl") is not None
    assert registry.get_entry("stl").adapter_class == "forecasting.adapters.stl.STLAdapter"
    assert registry.get_entry("naive") is None
    assert registry.get_entry("seasonal_naive") is None


def test_deployment_profile_hides_models_from_other_profiles(monkeypatch):
    registry = ModelRegistry()
    monkeypatch.setenv("TSFM_MODEL_PROFILE", "moirai")
    monkeypatch.setattr(importlib.util, "find_spec", lambda module_name: object())

    available = {entry.model_key for entry in registry.list_available()}

    assert "moirai" in available
    assert "chronos" in available
    assert "tirex_2" not in available
    assert "timesfm_3" not in available
    assert "timesfm_2_5" not in available
    assert "t0_gamma" not in available
