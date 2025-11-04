"""Test Streamlit configuration for dynamic module loading fix."""

import sys
from pathlib import Path

import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_streamlit_config_exists():
    """Verify that .streamlit/config.toml exists."""
    project_root = Path(__file__).parent.parent
    config_path = project_root / ".streamlit" / "config.toml"
    assert config_path.exists(), "Streamlit config file should exist"


def test_streamlit_version():
    """Verify that Streamlit version is compatible (< 1.40 to avoid module loading bugs)."""
    import streamlit

    version = streamlit.__version__
    major, minor = map(int, version.split(".")[:2])

    # Verify version is 1.39.0 or compatible version that doesn't have dynamic module loading issues
    assert (major, minor) < (1, 40), f"Streamlit version {version} may have dynamic module loading issues"


def test_streamlit_config_content():
    """Verify that config.toml has required settings."""
    project_root = Path(__file__).parent.parent
    config_path = project_root / ".streamlit" / "config.toml"

    if not config_path.exists():
        pytest.skip("Config file not found")

    content = config_path.read_text()

    # Check for key settings that help with module loading
    assert "[server]" in content, "Server section should exist"
    assert "enableCORS" in content, "CORS setting should be present"
    assert "headless" in content, "Headless setting should be present"


def test_gui_app_imports():
    """Verify that gui_app.py can be imported without errors."""
    try:
        from src import gui_app

        # Check that key components are present
        assert hasattr(gui_app, "build_combined_figure"), "build_combined_figure should exist"
        assert hasattr(gui_app, "load_prediction_model"), "load_prediction_model should exist"
    except ImportError as e:
        pytest.fail(f"Failed to import gui_app: {e}")
