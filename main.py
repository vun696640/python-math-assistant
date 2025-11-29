from fastapi import FastAPI
from pydantic import BaseModel
import math

app = FastAPI()

class MathRequest(BaseModel):
    expression: str  # ví dụ: "1+2*3" hoặc "sqrt(9)"

@app.get("/")
def root():
    return {"message": "Math API is running"}

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
