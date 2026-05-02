# Koyeb Deployment Guide

This guide explains how to deploy the Study Bot FastAPI app on Koyeb.

## Project Files Needed

Make sure these files are in your GitHub repository:

- `main.py`
- `requirements.txt`
- `static/index.html`
- `.gitignore`

Do not upload `.env`. It contains secret values. Your `.gitignore` already
keeps `.env` out of Git.

## Step 1: Push Code To GitHub

Open a terminal in this project folder and run:

```bash
git add .
git commit -m "Add Koyeb deployment guide"
git push
```

## Step 2: Create A Koyeb App

1. Go to Koyeb.
2. Create a new app.
3. Choose GitHub as the deployment method.
4. Select your Study Bot repository.
5. Choose Buildpack as the builder.

Koyeb will install the packages from `requirements.txt`.

## Step 3: Add The Run Command

Use this run command:

```bash
uvicorn main:app --host 0.0.0.0
```

This starts the FastAPI app and makes it available to Koyeb.

## Step 4: Add Environment Variables

In the Koyeb app settings, add these environment variables:

```text
GROQ_API_KEY=your_groq_api_key
MONGODB_URI=your_mongodb_connection_string
```

Use the same values from your local `.env` file.

## Step 5: Deploy

Click Deploy and wait for the build to finish.

After deployment, Koyeb gives you a public URL. Open that URL in your browser
to use the Study Bot online.

## Step 6: Test The App

Ask a study question in the deployed app.

If the bot does not answer, check:

- `GROQ_API_KEY` is correct.
- `MONGODB_URI` is correct.
- MongoDB allows connections from Koyeb.
- Koyeb deployment logs do not show errors.

## Useful Link

Koyeb FastAPI guide:

```text
https://www.koyeb.com/docs/deploy/fastapi
```
