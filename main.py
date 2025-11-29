import os
import json
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


# ========================
#  FALLBACK T1–T15 (nếu không có file .md)
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
        "T12": "Hình trụ, hình nón, hình cầu",
        "T13": "Thống kê",
        "T14": "Xác suất",
        "T15": "Bài toán thực tế",
    }
    return {
        code: Standard(
            code=code,
            name=name,
            description=f"Chuẩn {code} – {name} (mô tả mặc định dùng khi thiếu file chuan_dau_ra.md).",
        )
        for code, name in names.items()
    }


# ========================
#  LOAD CHUẨN ĐẦU RA
# ========================
def parse_chuan_dau_ra_md(path: str) -> Dict[str, Standard]:
    if not os.path.exists(path):
        print(f"[WARN] Không tìm thấy file chuẩn đầu ra: {path} – dùng fallback mặc định.")
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
        "Bạn là Trợ lý Toán 9 AI dành cho học sinh (một gia sư AI chứ không chỉ là chatbot).\n"
        "Nguyên tắc:\n"
        "1) Giải chi tiết, từng bước, đúng chương trình Toán 9.\n"
        "2) Bám sát chuẩn đầu ra T1–T15.\n"
        "3) Sau khi giải xong, phải:\n"
        "   - Tóm tắt ngắn gọn: Học sinh vừa ôn lại/đặt trọng tâm vào kiến thức gì.\n"
        "   - Đề xuất 1–3 hoạt động học tập tiếp theo (vd: làm thêm dạng nào, nhắc lại công thức nào).\n"
        "   - Gợi ý rằng bạn có thể tạo một quiz luyện tập nếu học sinh muốn.\n"
        "4) Khi viết công thức, dùng LaTeX với $...$ cho inline và $$...$$ cho công thức đứng riêng.\n"
        "5) Trình bày gọn gàng, không chèn quá nhiều dòng trống.\n\n"
        "Dưới đây là mô tả tóm tắt các chuẩn đầu ra:\n\n"
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
#  API: GENERATE TESTS – TẠO BỘ TEST OFFLINE, KHÔNG GỌI OPENAI
# ========================
@app.get("/api/generate_tests", response_model=TestGenerationResponse)
async def generate_tests():
    """
    Trả về 10 test case cố định để kiểm thử bộ phân loại T1–T15.
    Không phụ thuộc OpenAI → không bao giờ lỗi 500.
    """
    tests: List[TestCase] = [
        # T1
        TestCase(
            input="Tính UCLN và BCNN của 18 và 24.",
            expected_standards=["T1"],
        ),
        # T2
        TestCase(
            input="Một cửa hàng giảm giá 20% cho một chiếc áo giá 250 000 đồng. Hỏi sau giảm giá, chiếc áo còn bao nhiêu tiền?",
            expected_standards=["T2"],
        ),
        # T3
        TestCase(
            input="Trong tam giác ABC, biết tổng ba góc của tam giác bằng 180°. Hãy chứng minh điều đó.",
            expected_standards=["T3"],
        ),
        # T4
        TestCase(
            input="Tính căn bậc hai của 49 và 81.",
            expected_standards=["T4"],
        ),
        # T5
        TestCase(
            input="Giải phương trình bậc nhất: 2x - 5 = 9.",
            expected_standards=["T5"],
        ),
        # T6
        TestCase(
            input="Giải hệ phương trình: x + y = 5, x - y = 1.",
            expected_standards=["T6"],
        ),
        # T7
        TestCase(
            input="Giải bất phương trình: 3x - 2 > 4.",
            expected_standards=["T7"],
        ),
        # T8
        TestCase(
            input="Cho hàm số y = 2x + 1. Hãy vẽ đồ thị hàm số này trên hệ trục tọa độ.",
            expected_standards=["T8"],
        ),
        # T9 + T3
        TestCase(
            input="Trong tam giác vuông ABC tại A, hãy áp dụng định lý Pytago để tính độ dài cạnh còn lại.",
            expected_standards=["T3", "T9"],
        ),
        # T15 (bài toán thực tế + phần trăm)
        TestCase(
            input="Một bồn nước hình trụ được dùng trong thực tế để chứa nước cho gia đình. Bồn có bán kính đáy 0,6 m và chiều cao 1,5 m. Tính thể tích bồn và cho biết nếu dùng hết 70% lượng nước trong bồn thì còn lại bao nhiêu mét khối nước.",
            expected_standards=["T12", "T2", "T15"],
        ),
    ]

    # Đảm bảo chỉ dùng những mã có trong STANDARDS (phòng khi file chuẩn đầu ra bị sửa)
    filtered_tests: List[TestCase] = []
    for t in tests:
        valid = [c for c in t.expected_standards if c in STANDARDS]
        filtered_tests.append(TestCase(input=t.input, expected_standards=valid))

    return TestGenerationResponse(tests=filtered_tests)


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
