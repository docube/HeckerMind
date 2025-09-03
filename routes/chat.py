# app/routes/chat.py
# Handles chat requests across all documents using OpenAI's chat completion API.

from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse
from app.services.vectorstore_loader import get_relevant_documents
from app.config.settings import get_settings
from openai import AsyncOpenAI

router = APIRouter()
settings = get_settings()

# Initialize OpenAI Client
openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

async def stream_chat_response(question: str):
    """
    Generator that streams OpenAI's chat completion token-by-token.
    """
    try:
        # 1. Retrieve relevant documents across all vectorstores
        relevant_docs = get_relevant_documents(question)

        if not relevant_docs:
            yield "Sorry, no documents were found related to your question."
            return

        # 2. Build context string
        context_text = "\n\n".join(doc.page_content for doc in relevant_docs)

        # 3. Send context + question to OpenAI
        response = await openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an intelligent HR data assistant. You are given structured employee information extracted from multiple related tables in Excel sheets. Each table holds different but connected data about employees, such as:\n\n1. Core Employee Details (name, ID, contact info, job title, department, etc.)\n2. Previous Employers (company names, positions held, duration, references)\n3. Next of Kin (names, relationship, contact details)\n4. Dependents (children, spouse, other dependents with their details)\n5. References (names, relationship, contact info)\n\nYour task is to:\n- Understand that all tables are linked by an employee's unique identifier.\n- Combine data across tables to answer user queries with context.\n- If the query is broad (e.g., 'show me employee history'), summarize across all relevant tables.\n- If the query is specific (e.g., 'who is John Doe’s next of kin?'), retrieve the relevant information and present it clearly.\n- Maintain awareness of data relationships: e.g., an employee may have multiple dependents, or multiple previous employers.\n- Always clarify missing or ambiguous data instead of guessing.\n- Respond in a structured, professional, and concise manner.\n\nGoal: Provide context-aware, human-like answers that go beyond raw retrieval and instead show reasoning about the employee’s overall record."
},
                {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {question}"}
            ],
            temperature=0.0,
            stream=True
        )

        # 4. Stream tokens
        async for chunk in response:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    except Exception as e:
        yield f"An error occurred: {str(e)}"

@router.post("/chat")
async def chat(request: Request):
    """
    POST endpoint to chat across all documents.
    Accepts JSON body with 'question' field.
    """
    body = await request.json()
    question = body.get("question")

    if not question:
        raise HTTPException(status_code=400, detail="Question is required in the request body.")

    return StreamingResponse(stream_chat_response(question), media_type="text/event-stream")
