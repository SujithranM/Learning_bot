# PythonAnywhere Deployment Guide

This guide explains how to deploy the Study Bot FastAPI app on PythonAnywhere.

## Important Note

This project now uses SQLite for chat history.

SQLite stores data in a local file named:

```text
chat_history.db
```

That means you do not need MongoDB, PostgreSQL, or any external database for the
free PythonAnywhere deployment.

## Project Files Needed

Make sure these files are in your GitHub repository:

- `main.py`
- `requirements.txt`
- `static/index.html`
- `.gitignore`

Do not upload `.env`. It contains secret values. Your `.gitignore` already
keeps `.env` out of Git.

Also do not upload `chat_history.db`. It is created automatically when the app
runs.

## Step 1: Push Code To GitHub

Open a terminal in this project folder and run:

```bash
git add .
git commit -m "Use SQLite for PythonAnywhere deployment"
git push
```

## Step 2: Create PythonAnywhere Account

1. Go to PythonAnywhere.
2. Create a free account.
3. Open a Bash console.

Your website URL will look like this:

```text
https://YOURUSERNAME.pythonanywhere.com
```

Replace `YOURUSERNAME` with your PythonAnywhere username.

## Step 3: Download Your Project

In the PythonAnywhere Bash console, run:

```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/YOUR_REPOSITORY_NAME.git
cd YOUR_REPOSITORY_NAME
```

Replace the GitHub URL parts with your real GitHub username and repository name.

## Step 4: Create A Virtual Environment

Run:

```bash
mkvirtualenv studybot --python=python3.10
pip install -r requirements.txt
pip install --upgrade pythonanywhere
```

## Step 5: Create The `.env` File On PythonAnywhere

Create a `.env` file inside your project folder on PythonAnywhere.

Add:

```text
GROQ_API_KEY=your_groq_api_key
```

Use the same Groq key from your local `.env` file.

Remember: do not push `.env` to GitHub.

## Step 6: Create The FastAPI Website

PythonAnywhere uses an ASGI command for FastAPI apps.

Run this command, but replace `YOURUSERNAME` and `YOUR_REPOSITORY_NAME`:

```bash
pa website create --domain YOURUSERNAME.pythonanywhere.com --command '/home/YOURUSERNAME/.virtualenvs/studybot/bin/uvicorn --app-dir /home/YOURUSERNAME/YOUR_REPOSITORY_NAME --uds ${DOMAIN_SOCKET} main:app'
```

## Step 7: Open Your Website

Go to:

```text
https://YOURUSERNAME.pythonanywhere.com
```

Ask a study question to test the app.

## Step 8: If You Change Code Later

In the PythonAnywhere Bash console:

```bash
cd YOUR_REPOSITORY_NAME
git pull
pa website reload --domain YOURUSERNAME.pythonanywhere.com
```

## Troubleshooting

If the web page does not load:

- Check the PythonAnywhere error log.
- Make sure the ASGI command has the correct username and project folder.
- Make sure dependencies installed successfully.

If the bot does not answer:

- Check `GROQ_API_KEY`.
- Check the PythonAnywhere error log.
- Make sure `api.groq.com` is allowed on your PythonAnywhere account.

## Useful Links

PythonAnywhere FastAPI/ASGI guide:

```text
https://help.pythonanywhere.com/pages/ASGICommandLine/
```

PythonAnywhere free account limits:

```text
https://help.pythonanywhere.com/pages/FreeAccountsFeatures/
```
