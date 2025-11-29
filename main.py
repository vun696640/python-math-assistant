from fastapi import FastAPI
from pydantic import BaseModel
import math
import os

from dotenv import load_dotenv
from openai import OpenAI

# --- Load .env & tạo OpenAI client ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise RuntimeError("Thiếu OPENAI_API_KEY trong .env")

client = OpenAI(api_key=OPENAI_API_KEY)

# --- FastAPI app ---
app = FastAPI()

# Request cho máy tính tự xử lý bằng Python
class MathRequest(BaseModel):
    expression: str  # ví dụ: "1+2*3" hoặc "sqrt(9)"

# Request cho OpenAI giải / giải thích
class AICalcRequest(BaseModel):
    question: str  # ví dụ: "Giải phương trình x^2 - 5x + 6 = 0"

@app.get("/")
def root():
    return {"message": "Math API is running"}

# ----------------- 1. Máy tự tính (không AI) -----------------
@app.post("/calculate")
def calculate(req: MathRequest):
    try:
        # chỉ cho phép các hàm trong math, tránh eval bậy bạ
        allowed_names = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}
        allowed_names["__builtins__"] = {}

        result = eval(req.expression, allowed_names)
        return {"expression": req.expression, "result": result}
    except Exception as e:
        return {"error": str(e)}

# ----------------- 2. Nhờ OpenAI giải / giải thích -----------------
@app.post("/ai-calculate")
def ai_calculate(req: AICalcRequest):
    """
    Gửi câu hỏi/biểu thức cho OpenAI để:
    - giải bài
    - giải thích từng bước
    """
    prompt = f"""
    You are a math tutor. Solve the following problem step by step.
    Show reasoning clearly, but keep it concise and suitable for a grade 9 student.
    Answer in Vietnamese.
    
    Problem: {req.question}
    """

    try:
        completion = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": "You are a helpful math tutor."},
                {"role": "user", "content": prompt},
            ],
        )

        answer = completion.choices[0].message.content
        return {
            "question": req.question,
            "answer": answer
        }
    except Exception as e:
        return {"error": str(e)}
