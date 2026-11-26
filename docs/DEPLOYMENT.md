# Streamlit Community Cloud deployment

## Current GitHub layout

This project is currently stored inside the repository at:

```text
Heart_Disease_Classification_Project/heart_disease_classification/
```

When creating the app in [Streamlit Community Cloud](https://share.streamlit.io/), use:

| Setting | Value |
|---|---|
| Repository | `khalilDev21/heart_disease_classification` |
| Branch | `main` |
| Main file path | `Heart_Disease_Classification_Project/heart_disease_classification/app/streamlit_app.py` |
| Python version | `3.14` |

The app needs `models/best_model.joblib`. Keep it in the project folder in GitHub.

## Dependencies and theme

- `app/requirements.txt` pins the runtime packages to the saved model's versions.
- The root `requirements.txt` is for training and explainability scripts.
- For the theme to load with this nested entrypoint, place `.streamlit/config.toml` at the **top level of the GitHub repository**. The local theme file is in `.streamlit/config.toml` within the project folder.

## Updating the deployed app

After changing the app locally, upload the updated `app/streamlit_app.py` to the matching nested GitHub folder and commit it. Also upload the theme file to the repository-level `.streamlit/config.toml` if it is not there already. Streamlit Community Cloud will redeploy after the repository changes.

For details, see the [Community Cloud deployment guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy).
