import os
from typing import List, Dict

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from pypdf import PdfReader
from openai import OpenAI

# ========================
#  CONFIG
# ========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHUAN_DAU_RA_FILE = os.path.join(BASE_DIR, "chuan_dau_ra.md")
INDEX_FILE = os.path.join(BASE_DIR, "index.html")

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


# ---- Bài kiểm tra đầu vào ----
class PlacementQuestion(BaseModel):
    id: int
    text: str
    options: List[str]
    correct_index: int  # 0 = A, 1 = B, ...
    standards: List[str]


class PlacementQuizResponse(BaseModel):
    questions: List[PlacementQuestion]


# ---- API classify offline (dùng cho dev nếu cần) ----
class ClassifyRequest(BaseModel):
    text: str


class ClassifyResponse(BaseModel):
    standards: List[str]


# ========================
#  FALLBACK T1–T15
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
            description=f"Chuẩn {code} – {name} (mặc định, dùng khi thiếu file chuan_dau_ra.md).",
        )
        for code, name in names.items()
    }


# ========================
#  LOAD CHUẨN ĐẦU RA
# ========================
def parse_chuan_dau_ra_md(path: str) -> Dict[str, Standard]:
    if not os.path.exists(path):
        print(
            f"[WARN] Không tìm thấy file chuẩn đầu ra: {path} – dùng fallback mặc định."
        )
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
            if code:
                standards[code] = Standard(
                    code=code, name=name, description="\n".join(buf).strip()
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
            code=code, name=name, description="\n".join(buf).strip()
        )

    if not standards:
        print("[WARN] File chuẩn đầu ra rỗng – dùng fallback mặc định.")
        standards = default_standards()

    print(f"[INFO] Loaded {len(standards)} standards: {list(standards.keys())}")
    return standards


STANDARDS = parse_chuan_dau_ra_md(CHUAN_DAU_RA_FILE)


# ========================
#  SIMPLE KEYWORD DETECTOR
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

    return [x for x in found if x in STANDARDS]


# ========================
#  API: HEALTH
# ========================
@app.get("/api/health")
def api_health():
    return {
        "status": "ok",
        "standards_loaded": list(STANDARDS.keys()),
    }


# ========================
#  API: CHAT – trợ lý dạy kèm AI
# ========================
@app.post("/chat/message", response_model=ChatResponse)
async def chat_message(req: ChatRequest):
    if not os.environ.get("OPENAI_API_KEY"):
        raise HTTPException(500, "Thiếu OPENAI_API_KEY")

    standards_text = "\n".join(
        f"{s.code}: {s.name}\n{s.description}\n" for s in STANDARDS.values()
    )

    system_prompt = (
        "Bạn là Trợ lý Toán 9 AI cho học sinh.\n"
        "- Giải chi tiết, từng bước, đúng chương trình Toán 9.\n"
        "- Bám sát chuẩn đầu ra T1–T15.\n"
        "- Sau khi giải xong, hãy nêu ngắn gọn: học sinh vừa ôn lại kiến thức nào (chuẩn T mấy), "
        "và gợi ý 1–3 hoạt động học tập tiếp theo.\n"
        "- Có thể gợi ý rằng bạn sẽ tạo quiz luyện tập nếu học sinh muốn.\n"
        "- Dùng LaTeX với $...$ (inline) và $$...$$ (block) để hiển thị công thức.\n"
        "- Trình bày gọn, không thừa dòng trắng.\n\n"
        "Tóm tắt chuẩn đầu ra:\n"
        f"{standards_text}"
    )

    messages = [{"role": "system", "content": system_prompt}]
    messages += [{"role": m.role, "content": m.content} for m in req.messages]

    try:
        completion = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
        )
        reply = completion.choices[0].message.content.strip()
    except Exception as e:
        raise HTTPException(500, f"OpenAI error: {e}")

    last_user_msg = ""
    for m in reversed(req.messages):
        if m.role == "user":
            last_user_msg = m.content
            break

    detected = detect_standards_from_text(last_user_msg)

    return ChatResponse(reply=reply, standards=detected)


# ========================
#  API: CLASSIFY OFFLINE (dev)
# ========================
@app.post("/api/classify", response_model=ClassifyResponse)
async def api_classify(req: ClassifyRequest):
    detected = detect_standards_from_text(req.text)
    return ClassifyResponse(standards=detected)


# ========================
#  API: BÀI KIỂM TRA ĐẦU VÀO (PLACEMENT QUIZ)
# ========================
@app.get("/api/placement_quiz", response_model=PlacementQuizResponse)
async def get_placement_quiz():
    """
    Trả về 10 câu trắc nghiệm đầu vào.
    Mỗi câu có 4 đáp án A–D, gắn với một vài chuẩn T1–T15.
    Grading sẽ làm ở frontend.
    """
    questions: List[PlacementQuestion] = [
        # 1 – T1
        PlacementQuestion(
            id=1,
            text="Giá trị của 2^3 · 2^2 là:",
            options=["2^5", "2^6", "10", "32"],
            correct_index=3,  # 32
            standards=["T1"],
        ),
        # 2 – T2
        PlacementQuestion(
            id=2,
            text="Một cửa hàng giảm giá 20% cho chiếc áo giá 250 000 đồng. Giá sau giảm là:",
            options=["200 000 đồng", "225 000 đồng", "230 000 đồng", "240 000 đồng"],
            correct_index=1,
            standards=["T2"],
        ),
        # 3 – T3
        PlacementQuestion(
            id=3,
            text="Trong tam giác, tổng số đo ba góc luôn bằng:",
            options=["90°", "120°", "180°", "360°"],
            correct_index=2,
            standards=["T3"],
        ),
        # 4 – T4
        PlacementQuestion(
            id=4,
            text="Căn bậc hai số học của 81 là:",
            options=["±9", "9", "–9", "8"],
            correct_index=1,
            standards=["T4"],
        ),
        # 5 – T5
        PlacementQuestion(
            id=5,
            text="Nghiệm của phương trình 2x – 5 = 9 là:",
            options=["x = 2", "x = 3", "x = 5", "x = 7"],
            correct_index=3,
            standards=["T5"],
        ),
        # 6 – T6
        PlacementQuestion(
            id=6,
            text="Hệ nào sau đây là hệ phương trình bậc nhất hai ẩn?",
            options=[
                "x^2 + y = 1; x + y^2 = 2",
                "x + y = 3; 2x – y = 1",
                "x^2 + y^2 = 4; xy = 1",
                "x – y^2 = 0; x + y = 1",
            ],
            correct_index=1,
            standards=["T6"],
        ),
        # 7 – T7
        PlacementQuestion(
            id=7,
            text="Bất phương trình 3x – 2 > 4 tương đương với:",
            options=["x > 2", "x > 3", "x > 4", "x > 6"],
            correct_index=0,
            standards=["T7"],
        ),
        # 8 – T8
        PlacementQuestion(
            id=8,
            text="Đồ thị của hàm số y = 2x + 1 là:",
            options=[
                "Một đoạn thẳng",
                "Một đường tròn",
                "Một parabol",
                "Một đường thẳng",
            ],
            correct_index=3,
            standards=["T8"],
        ),
        # 9 – T9
        PlacementQuestion(
            id=9,
            text="Trong tam giác vuông, định lý Pytago phát biểu:",
            options=[
                "Hai cạnh góc vuông có tổng bằng cạnh huyền",
                "Bình phương cạnh huyền bằng tổng bình phương hai cạnh góc vuông",
                "Bình phương cạnh huyền bằng hiệu bình phương hai cạnh góc vuông",
                "Chu vi tam giác vuông luôn bằng 180°",
            ],
            correct_index=1,
            standards=["T3", "T9"],
        ),
        # 10 – T12, T15
        PlacementQuestion(
            id=10,
            text=(
                "Một bồn nước hình trụ có bán kính đáy 0,5 m và chiều cao 1,2 m. "
                "Thể tích gần đúng của bồn (làm tròn 1 chữ số thập phân, π ≈ 3,14) là:"
            ),
            options=["0,9 m³", "0,8 m³", "1,0 m³", "3,1 m³"],
            correct_index=0,
            standards=["T12", "T15"],
        ),
    ]

    # lọc lại standards chỉ lấy những mã tồn tại
    for q in questions:
        q.standards = [s for s in q.standards if s in STANDARDS]

    return PlacementQuizResponse(questions=questions)


# ========================
#  API: FILE PARSE
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


# ========================
#  ROOT: SERVE UI
# ========================
@app.get("/", response_class=HTMLResponse)
def root():
    if os.path.exists(INDEX_FILE):
        with open(INDEX_FILE, "r", encoding="utf-8") as f:
            return f.read()
    return HTMLResponse(
        "<h1>Math AI Assistant</h1><p>Không tìm thấy index.html</p>",
        status_code=500,
    )
