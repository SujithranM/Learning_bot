from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from pymongo import MongoClient
from datetime import datetime, timezone
from fastapi import FastAPI
from pydantic import BaseModel

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
mongo_uri = os.getenv("MONGODB_URI")

#print("A",groq_api_key is not None)
#print("B", mongo_uri is not None)


client = MongoClient(mongo_uri)
db = client["Genie"]
collection = db["users"]

app = FastAPI()

class ChatRequest(BaseModel):
    user_id: str
    question: str

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
    
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", 
         "You are Study Bot, an AI-powered academic assistant. "
         "Your role is to help students with study-related questions only. "
         "Provide clear explanations."
        ),
        ("placeholder", "{history}"),
        ("user", "{question}")
    ]
)

llm = ChatGroq(
    api_key=groq_api_key,
    model="openai/gpt-oss-20b"
)

chain = prompt | llm


def get_history(user_id):
    chats = collection.find({"user_id": user_id}).sort("timestamp", 1)
    history = []

    for chat in chats:
        history.append((chat["role"], chat["message"]))
    return history

@app.get("/")
def home():
     return {"message": "Welcome to the Learrning Chatbot API!"}

@app.post("/chat")
def chat(request: ChatRequest):
    history = get_history(request.user_id)
    response = chain.invoke({"history": history, "question": request.question})

    collection.insert_one({
        "user_id": request.user_id,
        "role":"user",
        "message":request.question,
        "timestamp": datetime.now(timezone.utc)
        })

    collection.insert_one({
        "user_id": request.user_id,
        "role":"assistant",
        "message":response.content,
        "timestamp": datetime.now(timezone.utc)
        })
    
    return {"response" : response.content}




