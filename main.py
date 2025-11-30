import os
import json
import random
from typing import List, Dict

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pypdf import PdfReader
from openai import OpenAI

# ========================
#  PATHS & CONFIG
# ========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _find_chuan_dau_ra_file() -> str:
    """
    Tìm file chuẩn đầu ra: ưu tiên 'chuan_dau_ra.md', nếu không có thì 'chuan_dau_ra'.
    """
    candidates = ["chuan_dau_ra.md", "chuan_dau_ra"]
    for name in candidates:
        path = os.path.join(BASE_DIR, name)
        if os.path.exists(path):
            return path
    # fallback: cứ trả về .md, nhưng parse sẽ dùng bộ mặc định
    return os.path.join(BASE_DIR, candidates[0])


CHUAN_DAU_RA_FILE = _find_chuan_dau_ra_file()
INDEX_FILE = os.path.join(BASE_DIR, "index.html")

app = FastAPI(title="K9 Math AI Assistant")

# Serve static (jsPDF/pdfMake/font, …)
static_dir = os.path.join(BASE_DIR, "static")
if os.path.isdir(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

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


class ClassifyRequest(BaseModel):
    text: str


class ClassifyResponse(BaseModel):
    standards: List[str]


class QuizQuestion(BaseModel):
    id: int
    text: str
    options: List[str]
    correct_index: int
    standards: List[str]


class QuizResponse(BaseModel):
    questions: List[QuizQuestion]


# ========================
#  CHUẨN ĐẦU RA T1–T15
# ========================
def default_standards() -> Dict[str, Standard]:
    names = {
        "T1": "Số & lũy thừa",
        "T2": "Tỉ lệ & phần trăm",
        "T3": "Góc & tam giác",
        "T4": "Căn bậc hai",
        "T5": "Phương trình bậc nhất",
        "T6": "Hệ phương trình",
        "T7": "Bất phương trình",
        "T8": "Hàm số & đồ thị",
        "T9": "Tam giác vuông & định lý Pythagore",
        "T10": "Đường tròn",
        "T11": "Tiếp tuyến",
        "T12": "Hình trụ, nón, cầu",
        "T13": "Thống kê",
        "T14": "Xác suất",
        "T15": "Bài toán thực tế",
    }
    return {
        code: Standard(
            code=code,
            name=name,
            description=f"Chuẩn {code} – {name}. (Mặc định, dùng khi thiếu file chuan_dau_ra.)",
        )
        for code, name in names.items()
    }


def parse_chuan_dau_ra_md(path: str) -> Dict[str, Standard]:
    """
    Đọc file chuẩn đầu ra dạng markdown (## T1 – ...).
    Nếu file không tồn tại hoặc rỗng, trả về bộ mặc định.
    Không spam WARN nữa.
    """
    if not os.path.exists(path):
        return default_standards()

    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    standards: Dict[str, Standard] = {}
    code = None
    name = ""
    buf: List[str] = []

    for line in lines:
        s = line.strip()
        if s.startswith("## "):
            # Lưu block cũ
            if code:
                standards[code] = Standard(
                    code=code,
                    name=name,
                    description="\n".join(buf).strip(),
                )
            after = s[3:].strip()
            if "–" in after:
                parts = after.split("–", 1)
                code = parts[0].strip()
                name = parts[1].strip()
            else:
                parts = after.split()
                code = parts[0].strip()
                name = parts[0].strip()
            buf = []
        else:
            if code:
                buf.append(line.rstrip())

    if code:
        standards[code] = Standard(
            code=code,
            name=name,
            description="\n".join(buf).strip(),
        )

    if not standards:
        return default_standards()
    return standards


def load_chuan_dau_ra_raw(path: str, standards: Dict[str, Standard]) -> str:
    """
    Raw text để nhét vào prompt AI. Nếu có file thì dùng file, không thì build từ dict.
    """
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    parts = ["# Chuẩn đầu ra Toán 9 (mặc định)"]
    for s in standards.values():
        parts.append(f"## {s.code} – {s.name}")
        parts.append(s.description)
    return "\n\n".join(parts)


STANDARDS: Dict[str, Standard] = parse_chuan_dau_ra_md(CHUAN_DAU_RA_FILE)
CHUAN_DAU_RA_TEXT: str = load_chuan_dau_ra_raw(CHUAN_DAU_RA_FILE, STANDARDS)

# ========================
#  DETECT CHUẨN TỪ TEXT
# ========================
def detect_standards_from_text(text: str) -> List[str]:
    t = text.lower()
    found: List[str] = []

    KEYWORDS = {
        "T1": ["ucln", "bcnn", "lũy thừa", "số mũ", "luy thua"],
        "T2": ["phần trăm", "%", "tỉ lệ", "tỷ lệ", "giảm giá"],
        "T3": ["góc", "tam giác", "chứng minh", "tam giac"],
        "T4": ["căn bậc hai", "căn bậc 2", "sqrt"],
        "T5": ["phương trình bậc nhất", "pt bậc nhất"],
        "T6": ["hệ phương trình", "hpt"],
        "T7": ["bất phương trình"],
        "T8": ["hàm số", "đồ thị", "graph"],
        "T9": ["tam giác vuông", "pytago", "pythagore"],
        "T10": ["đường tròn", "cung", "góc nội tiếp"],
        "T11": ["tiếp tuyến"],
        "T12": ["hình trụ", "hình nón", "hình cầu"],
        "T13": ["tần số", "bảng tần số", "thống kê"],
        "T14": ["xác suất"],
        "T15": ["bài toán thực tế", "thực tế", "bối cảnh thực tế"],
    }

    for code, keywords in KEYWORDS.items():
        if any(k in t for k in keywords):
            found.append(code)

    if not found:
        found.append("T1")

    # lọc theo bộ STANDARDS đang có
    return [x for x in found if x in STANDARDS]


# ========================
#  BASIC ROUTES
# ========================
@app.get("/", response_class=HTMLResponse)
def root():
    if os.path.exists(INDEX_FILE):
        with open(INDEX_FILE, "r", encoding="utf-8") as f:
            return f.read()
    return HTMLResponse(
        "<h1>K9 Math AI Assistant</h1><p>Không tìm thấy index.html</p>",
        status_code=500,
    )


@app.get("/api/health")
def api_health():
    return {
        "status": "ok",
        "standards_loaded": list(STANDARDS.keys()),
        "has_chuan_dau_ra_file": os.path.exists(CHUAN_DAU_RA_FILE),
    }


# ========================
#  CHAT – GIẢI TOÁN (luôn trả 200, không ném 500)
# ========================
@app.post("/chat/message", response_model=ChatResponse)
async def chat_message(req: ChatRequest):
    standards_text = "\n".join(
        f"{s.code}: {s.name}\n{s.description}\n" for s in STANDARDS.values()
    )

    system_prompt = (
        "Bạn là Trợ lý Toán 9 AI dành cho học sinh.\n"
        "- Giải bài tập rõ ràng, từng bước, đúng chương trình Toán 9.\n"
        "- Bám sát bộ chuẩn T1–T15 do giáo viên cung cấp.\n"
        "- Dùng LaTeX với $...$ (inline) và $$...$$ (block) cho công thức.\n"
        "- Sau khi giải xong, hãy:\n"
        "  • Nói ngắn gọn học sinh vừa ôn lại nội dung gì (có thể nhắc T1–T15 nếu phù hợp).\n"
        "  • Gợi ý 1–3 hoạt động học tiếp theo.\n"
        "  • Có thể gợi ý rằng bạn có thể tạo quiz luyện tập.\n"
        "- Trình bày gọn, không thừa dòng trắng.\n\n"
        "Dưới đây là tóm tắt các chuẩn T1–T15:\n"
        f"{standards_text}"
    )

    messages = [{"role": "system", "content": system_prompt}]
    messages += [{"role": m.role, "content": m.content} for m in req.messages]

    # lấy msg user cuối để detect chuẩn
    last_user_msg = ""
    for m in reversed(req.messages):
        if m.role == "user":
            last_user_msg = m.content
            break

    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
        )
        reply = completion.choices[0].message.content.strip()
    except Exception as e:
        # Bất kể lỗi gì (key, network, model, ...) vẫn trả 200 với reply là thông báo lỗi
        reply = (
            "Hiện tại server gặp lỗi khi gọi mô hình AI nên mình tạm thời "
            "không giải được bài toán này.\n\n"
            "Người quản trị có thể kiểm tra lại cấu hình OPENAI_API_KEY, "
            "model, hoặc kết nối mạng của server.\n"
            f"(Chi tiết kỹ thuật: {e})"
        )

    detected = detect_standards_from_text(last_user_msg)
    return ChatResponse(reply=reply, standards=detected)


# ========================
#  CLASSIFY (DEV)
# ========================
@app.post("/api/classify", response_model=ClassifyResponse)
async def api_classify(req: ClassifyRequest):
    detected = detect_standards_from_text(req.text)
    return ClassifyResponse(standards=detected)


# ========================
#  QUIZ SINH BỞI AI
# ========================
def generate_quiz_with_ai(mode: str, n: int = 10) -> List[QuizQuestion]:
    """
    Gọi OpenAI để sinh n câu hỏi trắc nghiệm dựa trên CHUẨN ĐẦU RA.
    mode: 'input' (đầu vào), 'output' (đầu ra / củng cố).
    """
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("Thiếu OPENAI_API_KEY, không sinh quiz AI được.")

    # Giới hạn độ dài chuẩn đầu ra cho gọn prompt
    chuan_text = CHUAN_DAU_RA_TEXT
    if len(chuan_text) > 12000:
        chuan_text = chuan_text[:12000]

    difficulty_note = (
        "Tạo câu hỏi ở mức cơ bản–trung bình, phù hợp kiểm tra đầu vào."
        if mode == "input"
        else "Tạo câu hỏi ở mức vận dụng–nâng cao nhẹ, phù hợp kiểm tra đầu ra, củng cố kiến thức."
    )

    system_prompt = (
        "Bạn là giáo viên Toán 9, chuyên soạn đề trắc nghiệm bám sát chuẩn đầu ra T1–T15.\n"
        "Bạn sẽ sinh ra bộ câu hỏi trắc nghiệm bốn lựa chọn A–D, đúng chương trình, rõ ràng, không đánh đố.\n"
    )

    user_prompt = f"""
Chuẩn đầu ra Toán 9:

{chuan_text}

Nhiệm vụ:

- {difficulty_note}
- Mỗi câu hỏi là 1 bài trắc nghiệm 4 lựa chọn A, B, C, D.
- Mỗi câu gắn với 1–3 chuẩn trong số T1–T15 (ví dụ ["T1"], ["T3","T9"]...).
- Tuyệt đối không dùng kiến thức ngoài chương trình Toán 9.

Yêu cầu xuất:

Trả về *DUY NHẤT* một JSON hợp lệ theo mẫu:

{{
  "questions": [
    {{
      "id": 1,
      "text": "Nội dung câu hỏi...",
      "options": ["Phương án A", "Phương án B", "Phương án C", "Phương án D"],
      "correct_index": 0,
      "standards": ["T1","T2"]
    }},
    ...
  ]
}}

- Sinh đúng {n} câu hỏi.
- "correct_index" là số 0–3 tương ứng A–D.
- "standards" chỉ bao gồm các mã trong: {list(STANDARDS.keys())}.
"""

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    raw = completion.choices[0].message.content.strip()

    # Bóc JSON
    try:
        if "```" in raw:
            raw = raw.split("```", 2)[1]
            raw = raw.replace("json", "", 1).strip()
        data = json.loads(raw)
    except Exception as e:
        raise RuntimeError(f"Lỗi parse JSON quiz từ OpenAI: {e}")

    questions_data = data.get("questions", [])
    questions: List[QuizQuestion] = []
    seen_ids = set()

    for i, q in enumerate(questions_data):
        try:
            q_id = int(q.get("id") or (i + 1))
            if q_id in seen_ids:
                q_id = max(seen_ids) + 1
            seen_ids.add(q_id)

            text = str(q.get("text", "")).strip()
            options = [str(x) for x in (q.get("options") or [])]
            if len(options) != 4:
                continue
            ci = int(q.get("correct_index", 0))
            if ci < 0 or ci > 3:
                continue
            standards = [s for s in (q.get("standards") or []) if s in STANDARDS]

            if not text:
                continue

            questions.append(
                QuizQuestion(
                    id=q_id,
                    text=text,
                    options=options,
                    correct_index=ci,
                    standards=standards or ["T1"],
                )
            )
        except Exception:
            continue

    if not questions:
        raise RuntimeError("Không sinh được câu hỏi hợp lệ nào từ OpenAI.")
    if len(questions) > n:
        questions = questions[:n]

    return questions


@app.get("/api/input_quiz", response_model=QuizResponse)
async def api_input_quiz():
    try:
        questions = generate_quiz_with_ai("input", n=10)
        return QuizResponse(questions=questions)
    except Exception as e:
        raise HTTPException(500, f"Lỗi sinh đề kiểm tra đầu vào bằng AI: {e}")


@app.get("/api/output_quiz", response_model=QuizResponse)
async def api_output_quiz():
    try:
        questions = generate_quiz_with_ai("output", n=10)
        return QuizResponse(questions=questions)
    except Exception as e:
        raise HTTPException(500, f"Lỗi sinh đề kiểm tra đầu ra bằng AI: {e}")


# ========================
#  FILE PARSE (PDF/TXT)
# ========================
@app.post("/file/parse")
async def parse_file(file: UploadFile = File(...)):
    name = file.filename

    try:
        if name.lower().endswith(".pdf"):
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
