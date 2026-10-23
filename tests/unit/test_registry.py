"""Unit tests for registry model availability."""

from forecasting.registry import ModelRegistry


def test_experimental_registry_entries_are_hidden_by_default():
    registry = ModelRegistry()

    default_models = {entry.model_key for entry in registry.list_available()}
    all_models = {
        entry.model_key
        for entry in registry.list_available(include_experimental=True)
    }

    assert "timesfm_3" not in default_models
    assert "timesfm_3" in all_models
    assert default_models < all_models
