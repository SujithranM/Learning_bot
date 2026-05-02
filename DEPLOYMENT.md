# Railway Deployment Guide

This guide explains how to deploy the Study Bot FastAPI app on Railway.

## Project Files Needed

Make sure these files are in your GitHub repository:

- `main.py`
- `requirements.txt`
- `railway.json`
- `static/index.html`
- `.gitignore`

Do not upload `.env`. It contains secret values. Your `.gitignore` already
keeps `.env` out of Git.

## Step 1: Push Code To GitHub

Open a terminal in this project folder and run:

```bash
git add .
git commit -m "Add Railway deployment setup"
git push
```

## Step 2: Create Railway Project

1. Go to Railway.
2. Click New Project.
3. Choose Deploy from GitHub repo.
4. Select your Study Bot repository.
5. Wait for Railway to build the project.

Railway will install the packages from `requirements.txt`.

## Step 3: Check Start Command

The project has a `railway.json` file with this start command:

```bash
uvicorn main:app --host 0.0.0.0 --port $PORT
```

Railway gives your app a port through the `$PORT` variable, so do not replace
`$PORT` with `8000`.

## Step 4: Add Environment Variables

In Railway:

1. Open your project.
2. Open the service.
3. Go to Variables.
4. Add these variables:

```text
GROQ_API_KEY=your_groq_api_key
MONGODB_URI=your_mongodb_connection_string
```

Use the same values from your local `.env` file.

## Step 5: Generate Public URL

1. Open your Railway service.
2. Go to Settings.
3. Find Networking.
4. Click Generate Domain.

Railway will create a public website link for your app.

## Step 6: Test The App

Open the generated Railway URL in your browser.

Ask a study question. If the bot does not answer, check:

- `GROQ_API_KEY` is correct.
- `MONGODB_URI` is correct.
- MongoDB allows connections from Railway.
- Railway deployment logs do not show errors.
