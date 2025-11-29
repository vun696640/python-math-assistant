import os
from typing import List, Dict, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pypdf import PdfReader
from openai import OpenAI
import json


# ========================
#  CONFIG
# ========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHUAN_DAU_RA_FILE = os.path.join(BASE_DIR, "chuan_dau_ra")

app = FastAPI(title="Math AI Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI()


# ========================
#  DATA MODELS
# ========================
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]

class ChatResponse(BaseModel):
    reply: str
    standards: List[str]

class Standard(BaseModel):
    code: str
    name: str
    description: str

class TestCase(BaseModel):
    input: str
    expected_standards: List[str]

class TestGenerationResponse(BaseModel):
    tests: List[TestCase]


# ========================
#  LOAD CHUẨN ĐẦU RA (T1–T15)
# ========================
def parse_chuan_dau_ra_md(path: str) -> Dict[str, Standard]:
    if not os.path.exists(path):
        return {}

    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    standards = {}
    code = None
    name = ""
    buf = []

    for line in lines:
        s = line.strip()
        if s.startswith("## "):
            # save previous
            if code:
                standards[code] = Standard(
                    code=code,
                    name=name,
                    description="\n".join(buf).strip()
                )
            # parse new
            after = s[3:].strip()
            if "–" in after:
                code = after.split("–")[0].strip()
                name = after.split("–")[1].strip()
            else:
                parts = after.split()
                code = parts[0]
                name = parts[0]
            buf = []
        else:
            if code:
                buf.append(line.rstrip())

    if code:
        standards[code] = Standard(
            code=code,
            name=name,
            description="\n".join(buf).strip()
        )

    return standards


STANDARDS = parse_chuan_dau_ra_md(CHUAN_DAU_RA_FILE)


# ========================
#  SIMPLE KEYWORD DETECTOR
# ========================
def detect_standards_from_text(text: str) -> List[str]:
    t = text.lower()
    found = []

    KEYWORDS = {
        "T1": ["ucln", "bcnn", "lũy thừa", "số mũ", "luy thua"],
        "T2": ["phần trăm", "%", "tỉ lệ", "giam gia"],
        "T3": ["góc", "tam giác", "chứng minh", "tam giac"],
        "T4": ["căn bậc hai", "sqrt"],
        "T5": ["phương trình bậc nhất"],
        "T6": ["hệ phương trình"],
        "T7": ["bất phương trình"],
        "T8": ["hàm số", "đồ thị", "graph"],
        "T9": ["tam giác vuông", "pytago"],
        "T10": ["đường tròn", "cung", "góc nội tiếp"],
        "T11": ["tiếp tuyến"],
        "T12": ["hình trụ", "hình nón", "hình cầu"],
        "T13": ["tần số", "bảng", "thống kê"],
        "T14": ["xác suất"],
        "T15": ["bài toán thực tế", "thực tế"]
    }

    for code, keywords in KEYWORDS.items():
        if any(k in t for k in keywords):
            found.append(code)

    if not found:
        found.append("T1")

    # Chỉ giữ chuẩn có thật trong file chuẩn đầu ra
    return [x for x in found if x in STANDARDS]


# ========================
#  API: HEALTH
# ========================
@app.get("/api/health")
def api_health():
    return {
        "status": "ok",
        "standards_loaded": list(STANDARDS.keys())
    }


# ========================
#  API: CHAT
# ========================
@app.post("/chat/message", response_model=ChatResponse)
async def chat_message(req: ChatRequest):
    if not os.environ.get("OPENAI_API_KEY"):
        raise HTTPException(500, "Thiếu OPENAI_API_KEY")

    standards_text = "\n".join(
        f"{s.code}: {s.name}\n{s.description}\n"
        for s in STANDARDS.values()
    )

    system_prompt = (
        "Bạn là trợ lý Toán lớp 9. "
        "Giải thích rõ ràng, chi tiết, từng bước. "
        "Bám sát chuẩn đầu ra T1–T15.\n\n"
        f"{standards_text}"
    )

    messages = [{"role": "system", "content": system_prompt}]
    messages += [{"role": m.role, "content": m.content} for m in req.messages]

    try:
        completion = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages
        )
        reply = completion.choices[0].message.content.strip()
    except Exception as e:
        raise HTTPException(500, f"OpenAI error: {e}")

    # detect standards
    last_user_msg = ""
    for m in reversed(req.messages):
        if m.role == "user":
            last_user_msg = m.content
            break

    detected = detect_standards_from_text(last_user_msg)

    return ChatResponse(reply=reply, standards=detected)


# ========================
#  API: TEST GENERATOR (AI TỰ SINH TEST)
# ========================
@app.get("/api/generate_tests", response_model=TestGenerationResponse)
async def generate_tests():
    standards_text = "\n".join(
        f"{s.code}: {s.name}\n{s.description}\n"
        for s in STANDARDS.values()
    )

    prompt = f"""
    Hãy tạo 5 bài kiểm tra (test case) cho Toán 9 dựa trên các chuẩn đầu ra sau:

    {standards_text}

    Yêu cầu:
    - Không lặp lại ví dụ trong chuẩn đầu ra
    - Mỗi test có dạng:
        {{
            "input": "...",
            "expected_standards": ["T..."]
        }}
    - Chỉ xuất JSON duy nhất với cấu trúc:
      {{
        "tests": [ ... ]
      }}
    """

    try:
        completion = client.chat.completions.create(
            model="gpt-4.1",
            messages=[{"role": "user", "content": prompt}]
        )
        raw = completion.choices[0].message.content
        data = json.loads(raw)
        return data
    except Exception as e:
        raise HTTPException(500, f"OpenAI error in test generation: {e}")


# ========================
#  API: FILE PARSE
# ========================
@app.post("/file/parse")
async def parse_file(file: UploadFile = File(...)):
    name = file.filename.lower()

    try:
        if name.endswith(".pdf"):
            reader = PdfReader(file.file)
            pages = [page.extract_text() or "" for page in reader.pages]
            content = "\n\n".join(pages)
        else:
            raw = await file.read()
            content = raw.decode("utf-8", errors="ignore")

        if len(content) > 8000:
            content = content[:8000]

        return {"filename": name, "content": content}

    except Exception as e:
        raise HTTPException(400, f"Không đọc được file: {e}")


# ========================
#  ROOT (CHO RENDER HEALTH CHECK)
# ========================
@app.get("/")
def root():
    return {"message": "Math AI Assistant is running!"}


# ========================
#  STATIC (UI)
# ========================
app.mount("/web", StaticFiles(directory=".", html=True), name="static")
