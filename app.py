from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

from agent import ask  # reuse your existing agent logic


class QuestionRequest(BaseModel):
    question: str


class AnswerResponse(BaseModel):
    question: str
    answer: str
    sources: List[str]


app = FastAPI(title="Policy AI Agent")


@app.post("/ask", response_model=AnswerResponse)
def ask_endpoint(payload: QuestionRequest):
    """
    Simple API endpoint that takes a question and returns
    the agent's answer plus the source documents.
    """
    result = ask(payload.question)
    # result already has keys: question, answer, sources
    return AnswerResponse(**result)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
