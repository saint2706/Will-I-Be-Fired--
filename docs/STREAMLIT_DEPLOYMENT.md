# Streamlit Deployment Guide

## Overview

This document explains the Streamlit deployment configuration for the "Will I Be Fired?" application and the fix for dynamic module loading errors.

## Dynamic Module Loading Issue

### Problem

When deploying to Streamlit Cloud with Streamlit 1.50.0, the following errors occurred:

```
TypeError: error loading dynamically imported module: 
https://mlba-angad.streamlit.app/~/+/static/js/createDownloadLinkElement.ZaXNnPK4.js

TypeError: error loading dynamically imported module: 
https://mlba-angad.streamlit.app/~/+/static/js/withFullScreenWrapper.C3561XxJ.js
```

These errors appear when:
- Using `st.download_button()` (triggers `createDownloadLinkElement`)
- Using `st.plotly_chart()` with fullscreen capability (triggers `withFullScreenWrapper`)

### Root Cause

Streamlit 1.50.0 introduced a regression with dynamic ES module loading on Streamlit Cloud. The browser fails to load JavaScript modules required for certain interactive components.

### Solution

The fix involves two changes:

1. **Downgrade Streamlit Version**: Downgrade from `1.50.0` to `1.39.0`, which is stable and doesn't have the module loading bug.

2. **Add Streamlit Configuration**: Create `.streamlit/config.toml` with proper server settings for deployment.

## Configuration Details

### `.streamlit/config.toml`

```toml
[server]
# Enable CORS to allow cross-origin requests for dynamically loaded modules
enableCORS = true

# Enable XSRF protection
enableXsrfProtection = true

# Set the base URL path for the application
# This helps with proper module resolution in deployed environments
headless = true

[browser]
# Ensure the browser can load dynamic modules
gatherUsageStats = false

[theme]
# Optional: Define a consistent theme
primaryColor = "#F63366"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"
```

### Dependency Versions

- **Before**: `streamlit==1.50.0`
- **After**: `streamlit==1.39.0`

Updated in both:
- `requirements.txt`
- `pyproject.toml`

## Deployment Steps

### Local Testing

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run src/gui_app.py
```

### Streamlit Cloud Deployment

1. Ensure `.streamlit/config.toml` is committed to the repository
2. Ensure `requirements.txt` specifies `streamlit==1.39.0`
3. Deploy to Streamlit Cloud as usual
4. The app should now load without module loading errors

## Components Affected

The fix resolves issues with these Streamlit components:

- **`st.download_button()`**: Used in the batch upload tab for downloading prediction reports
- **`st.plotly_chart()`**: Used for displaying model metrics and tenure risk visualizations
- **Fullscreen capabilities**: All Plotly charts can now be viewed in fullscreen mode

## Verification

Run the test suite to verify the fix:

```bash
pytest tests/test_streamlit_config.py -v
```

This will verify:
- ✅ `.streamlit/config.toml` exists
- ✅ Streamlit version is < 1.40
- ✅ Configuration contains required settings
- ✅ GUI app imports without errors

## Alternative Solutions

If you need features from Streamlit 1.50.0+, monitor the Streamlit GitHub repository for fixes to the module loading issue:

- [Streamlit GitHub Issues](https://github.com/streamlit/streamlit/issues)
- Search for: "dynamic module loading" or "error loading dynamically imported module"

Once the issue is resolved in a newer version, you can upgrade by:

1. Updating the version in `requirements.txt` and `pyproject.toml`
2. Testing locally
3. Updating the test in `tests/test_streamlit_config.py` to accept the new version

## References

- [Streamlit Configuration Documentation](https://docs.streamlit.io/library/advanced-features/configuration)
- [Streamlit Deployment Guide](https://docs.streamlit.io/streamlit-community-cloud/deploy-your-app)
- Project Issue: Dynamic module loading errors on Streamlit Cloud
