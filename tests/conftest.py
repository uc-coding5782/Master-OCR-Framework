"""Shared pytest fixtures."""

import pytest

from ocr_framework.config.schema import FrameworkConfig


@pytest.fixture
def framework_config() -> FrameworkConfig:
    """Return a default framework configuration for tests."""
    return FrameworkConfig()
