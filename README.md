# Study Bot

Study Bot is a simple AI-powered academic assistant. Students can ask study
questions in a web page, and the bot replies with clear, orderly answers.

The project uses:

- FastAPI for the backend
- Groq through LangChain for AI answers
- SQLite for chat history
- HTML, CSS, and JavaScript for the frontend

## Features

- Ask study-related questions
- Get structured answers with headings, steps, bullet points, or tables
- Store chat history for each user
- Run locally with Uvicorn
- Deploy on PythonAnywhere

## Project Structure

```text
.
|-- main.py
|-- requirements.txt
|-- DEPLOYMENT.md
|-- static/
|   `-- index.html
`-- .gitignore
```

## Environment Variables

Create a `.env` file in the project folder.

Add your Groq API key:

```text
GROQ_API_KEY=your_groq_api_key
```

Do not upload `.env` to GitHub.

## Local Setup

Create and activate a virtual environment:

```bash
python -m venv venv
```

On Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the app:

```bash
uvicorn main:app --host 127.0.0.1 --port 8000
```

Open this URL in your browser:

```text
http://127.0.0.1:8000
```

## SQLite Database

The app uses SQLite to save chat history.

When the app starts, it automatically creates:

```text
chat_history.db
```

This file is ignored by Git because it is local data.

## Main API Route

The frontend sends questions to:

```text
POST /chat
```

Example request body:

```json
{
  "user_id": "user123",
  "question": "What is photosynthesis?"
}
```

Example response:

```json
{
  "response": "The bot answer appears here."
}
```

## Deployment

This project is prepared for PythonAnywhere deployment.

See:

```text
DEPLOYMENT.md
```

## Notes

- `.env` stores secret values and should not be committed.
- `chat_history.db` is created automatically and should not be committed.
- The app is designed for study-related questions only.
