"""Named configuration profile identifiers."""

from __future__ import annotations

from pathlib import Path

from ocr_framework.config.loader import load_config
from ocr_framework.config.schema import FrameworkConfig
from ocr_framework.exceptions import ConfigurationError

PROFILE_NAMES: tuple[str, ...] = ("default", "document", "receipt", "handwriting", "batch")


def get_profile(name: str) -> FrameworkConfig:
    """Return a configuration object for the given profile name.

    Args:
        name: Profile name (e.g., 'default', 'document', 'receipt').

    Returns:
        A FrameworkConfig object loaded from the corresponding YAML file.

    Raises:
        ConfigurationError: If the profile is unknown or the file cannot be loaded.
    """
    if name not in PROFILE_NAMES:
        raise KeyError(f"Unknown profile: {name}")

    # Determine the config directory
    # First, try relative to the current file
    config_dir = Path(__file__).parent.parent.parent / "configs"

    # If that doesn't exist, try relative to the current working directory
    if not config_dir.exists():
        config_dir = Path.cwd() / "configs"

    config_path = config_dir / f"{name}.yaml"

    if not config_path.exists():
        # Fall back to default configuration if profile file doesn't exist
        from ocr_framework.config.defaults import default_config

        config = default_config()
        config.profile = name
        return config

    try:
        config = load_config(config_path)
        config.profile = name
        return config
    except ConfigurationError:
        raise
    except Exception as exc:
        raise ConfigurationError(f"Failed to load profile '{name}': {exc}") from exc
