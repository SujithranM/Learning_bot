from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from motor.motor_asyncio import AsyncIOMotorClient
from datetime import datetime, timezone
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import unicodedata

# Load secret values from the .env file.
# Example: GROQ_API_KEY and MONGODB_URI are stored there instead of inside code.
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
mongo_uri = os.getenv("MONGODB_URI")

#print("A",groq_api_key is not None)
#print("B", mongo_uri is not None)


# Connect to MongoDB.
# The app stores every user's questions and bot answers in this collection.
client = AsyncIOMotorClient(mongo_uri)
db = client["Genie"]
collection = db["users"]

# Create the FastAPI app.
app = FastAPI()

# This class describes the JSON data that the frontend sends to /chat.
# Example:
# {
#   "user_id": "user123",
#   "question": "What is photosynthesis?"
# }
class ChatRequest(BaseModel):
    user_id: str
    question: str

# Allow the frontend page to call this backend API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
    
)

# Let FastAPI serve files from the static folder.
# This makes /static/index.html and other frontend files available in the browser.
app.mount("/static", StaticFiles(directory="static"), name="static")

# This prompt tells the AI how it should behave.
# The system message gives rules, the history keeps previous chat messages,
# and the user message contains the new question.
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", 
         "You are Study Bot, an AI-powered academic assistant. "
         "Your role is to help students with study-related questions only. "
         "Always answer in a clear, orderly Markdown format. "
         "Do not reply as one long paragraph. "
         "Use this structure when it fits the question: "
         "## Short Answer, ## Explanation, ## Steps or Key Points, and ## Example. "
         "Use numbered steps for processes, bullet points for lists, and tables for comparisons. "
         "Keep paragraphs short, with no more than three sentences each. "
         "Use plain ASCII characters only. Avoid emoji, special numbering symbols, and typographic dashes or quotes."
        ),
        ("placeholder", "{history}"),
        ("user", "{question}")
    ]
)

# Create the Groq chat model.
# This is the AI model that will generate the Study Bot's answers.
llm = ChatGroq(
    api_key=groq_api_key,
    model="openai/gpt-oss-20b"
)

# Combine the prompt and the model into one chain.
# Later, we call this chain with the user's question.
chain = prompt | llm


def normalize_answer(text: str) -> str:
    """Convert special characters into simple ASCII characters."""
    replacements = {
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2026": "...",
    }

    normalized = unicodedata.normalize("NFKC", text)
    for source, target in replacements.items():
        normalized = normalized.replace(source, target)

    return normalized.encode("ascii", "ignore").decode("ascii")


async def get_history(user_id):
    """Get all previous messages for one user from MongoDB."""
    cursor = collection.find({"user_id": user_id}).sort("timestamp", 1)
    history = []

    # LangChain expects history as pairs like ("user", "hello").
    async for chat in cursor:
        history.append((chat["role"], chat["message"]))

    return history

@app.get("/")
def home():
    # Show the main web page when the user opens http://127.0.0.1:8000/
    return FileResponse("static/index.html")

@app.post("/chat")
async def chat(request: ChatRequest):
    # Step 1: Get the old conversation for this user.
    history = await get_history(request.user_id)

    # Step 2: Ask the AI model for an answer.
    # ainvoke is used because this route is async.
    response = await chain.ainvoke({"history": history, "question": request.question})

    # Step 3: Clean the answer so the frontend gets simple readable text.
    response_text = normalize_answer(response.content)

    # Step 4: Save both the user's question and the bot's answer in MongoDB.
    await collection.insert_many([
        {
            "user_id": request.user_id,
            "role": "user",
            "message": request.question,
            "timestamp": datetime.now(timezone.utc)
        },
        {
            "user_id": request.user_id,
            "role": "assistant",
            "message": response_text,
            "timestamp": datetime.now(timezone.utc)
        }
    ])
    
    # Step 5: Send the bot answer back to the frontend as JSON.
    return {"response" : response_text}
