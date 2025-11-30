import os
import random
import json
from typing import List, Dict

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pypdf import PdfReader
from openai import OpenAI

# ========================
#  PATH / CONFIG
# ========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_FILE = os.path.join(BASE_DIR, "index.html")


def _detect_chuan_file() -> str:
    """
    Thử tìm file chuẩn đầu ra với vài tên phổ biến.
    Ưu tiên:
      1. chuan_dau_ra.md
      2. chuan-dau-ra.md
      3. chuan-dau-ra.txt
    """
    candidates = [
        os.path.join(BASE_DIR, "chuan_dau_ra.md"),
        os.path.join(BASE_DIR, "chuan-dau-ra.md"),
        os.path.join(BASE_DIR, "chuan-dau-ra.txt"),
    ]
    for p in candidates:
        if os.path.exists(p):
            print(f"[INFO] Đã tìm thấy file chuẩn đầu ra: {p}")
            return p
    # Không có file nào → trả về đường dẫn mặc định, nhưng parse() sẽ dùng fallback
    print(
        "[WARN] Không tìm thấy file chuẩn đầu ra nào (chuan_dau_ra.md / chuan-dau-ra.md). "
        "Sử dụng bộ chuẩn mặc định T1–T15."
    )
    return candidates[0]


CHUAN_DAU_RA_FILE = _detect_chuan_file()

app = FastAPI(title="Math AI Assistant")

# phục vụ /static (jspdf + font)
static_dir = os.path.join(BASE_DIR, "static")
if os.path.isdir(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
else:
    print(f"[WARN] Không tìm thấy thư mục static: {static_dir}")

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


class ClassifyRequest(BaseModel):
    text: str


class ClassifyResponse(BaseModel):
    standards: List[str]


class QuizQuestion(BaseModel):
    id: int
    text: str
    options: List[str]
    correct_index: int  # 0 = A, 1 = B, ...
    standards: List[str]


class QuizResponse(BaseModel):
    questions: List[QuizQuestion]


# ========================
#  DEFAULT T1–T15
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
            description=f"Chuẩn {code} – {name} (mặc định, dùng khi thiếu file chuan_dau_ra).",
        )
        for code, name in names.items()
    }


# ========================
#  LOAD CHUẨN ĐẦU RA
# ========================
def parse_chuan_dau_ra_md(path: str) -> Dict[str, Standard]:
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
            # lưu chuẩn trước
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
        print("[WARN] File chuẩn đầu ra tồn tại nhưng rỗng – dùng fallback mặc định.")
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
        "T1": ["ucln", "bcnn", "lũy thừa", "luy thua", "số mũ", "so mu"],
        "T2": ["phần trăm", "%", "tỉ lệ", "tỷ lệ", "giảm giá"],
        "T3": ["góc", "tam giác", "tam giac", "chứng minh"],
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
#  HEALTH
# ========================
@app.get("/api/health")
def api_health():
    return {"status": "ok", "standards_loaded": list(STANDARDS.keys())}


# ========================
#  CHAT – TRỢ LÝ TOÁN
# ========================
@app.post("/chat/message", response_model=ChatResponse)
async def chat_message(req: ChatRequest):
    if not os.environ.get("OPENAI_API_KEY"):
        raise HTTPException(500, "Thiếu OPENAI_API_KEY")

    standards_text = "\n".join(
        f"{s.code}: {s.name}\n{s.description}\n" for s in STANDARDS.values()
    )

    system_prompt = (
        "Bạn là Trợ lý Toán 9 AI dành cho học sinh.\n"
        "- Giải bài tập rõ ràng, từng bước, đúng kiến thức Toán 9.\n"
        "- Bám sát bộ chuẩn T1–T15 do giáo viên cung cấp.\n"
        "- Dùng LaTeX với $...$ (inline) và $$...$$ (block) để hiển thị công thức.\n"
        "- Sau khi giải xong, hãy:\n"
        "  • Nói ngắn gọn học sinh vừa ôn lại nội dung gì (có thể nhắc T1–T15 nếu phù hợp).\n"
        "  • Gợi ý 1–3 hoạt động học tiếp theo.\n"
        "  • Có thể gợi ý rằng bạn có thể tạo quiz luyện tập nếu học sinh muốn.\n"
        "- Trình bày gọn, không thừa dòng trắng.\n"
        "\n"
        "Dưới đây là tóm tắt các chuẩn T1–T15:\n"
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
#  CLASSIFY – DEV
# ========================
@app.post("/api/classify", response_model=ClassifyResponse)
async def api_classify(req: ClassifyRequest):
    detected = detect_standards_from_text(req.text)
    return ClassifyResponse(standards=detected)


# ========================
#  QUIZ – FALLBACK NGÂN HÀNG CÂU HỎI
# (dùng khi OpenAI lỗi / không có API key)
# ========================
INPUT_QUESTION_BANK: List[QuizQuestion] = [
    QuizQuestion(
        id=1,
        text="Kết quả của 2^3 · 2^2, viết dưới dạng lũy thừa của 2, là:",
        options=["2^5", "2^6", "10", "32"],
        correct_index=0,
        standards=["T1"],
    ),
    QuizQuestion(
        id=2,
        text="Một cửa hàng giảm giá 20% cho chiếc áo giá 250 000 đồng. Giá sau giảm là:",
        options=["200 000 đồng", "225 000 đồng", "230 000 đồng", "240 000 đồng"],
        correct_index=1,
        standards=["T2"],
    ),
    QuizQuestion(
        id=3,
        text="Trong tam giác, tổng số đo ba góc luôn bằng:",
        options=["90°", "120°", "180°", "360°"],
        correct_index=2,
        standards=["T3"],
    ),
]

OUTPUT_QUESTION_BANK: List[QuizQuestion] = [
    QuizQuestion(
        id=101,
        text="Kết quả của 5^2 · 5^(−1) là:",
        options=["5", "25", "1", "1/5"],
        correct_index=0,
        standards=["T1"],
    ),
    QuizQuestion(
        id=102,
        text="Một lớp có 30 học sinh, trong đó 40% là nữ. Số học sinh nữ là:",
        options=["10", "12", "14", "16"],
        correct_index=1,
        standards=["T2"],
    ),
]


def _pick_random_questions(bank: List[QuizQuestion], n: int = 10) -> List[QuizQuestion]:
    if not bank:
        return []
    if len(bank) <= n:
        chosen = bank[:]
    else:
        chosen = random.sample(bank, n)

    result: List[QuizQuestion] = []
    for q in chosen:
        filtered_standards = [s for s in q.standards if s in STANDARDS]
        result.append(
            QuizQuestion(
                id=q.id,
                text=q.text,
                options=list(q.options),
                correct_index=q.correct_index,
                standards=filtered_standards,
            )
        )
    return result


# ========================
#  QUIZ – GENERATE BẰNG OPENAI
# ========================
def _generate_quiz_with_openai(mode: str, n: int = 10) -> List[QuizQuestion]:
    """
    mode: 'input' (dễ hơn, đầu vào) hoặc 'output' (khó hơn, đầu ra / luyện tập).
    Trả về list QuizQuestion; nếu lỗi → raise, để caller fallback.
    """
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("No OPENAI_API_KEY")

    # mô tả chuẩn T1–T15 cho prompt
    standards_text = "\n".join(
        f"{s.code}: {s.name}\n{s.description}" for s in STANDARDS.values()
    )

    difficulty = (
        "mức độ cơ bản đến trung bình, phù hợp kiểm tra đầu vào."
        if mode == "input"
        else "mức độ từ trung bình đến vận dụng, có thể lồng ghép bối cảnh thực tế đơn giản."
    )

    system_prompt = (
        "Bạn là giáo viên Toán 9, tạo câu hỏi trắc nghiệm 4 lựa chọn A–D.\n"
        "Mỗi câu hỏi phải được gắn với 1–3 chuẩn T1–T15.\n"
        "Xuất ra **DUY NHẤT** một JSON hợp lệ theo cấu trúc:\n"
        "{\n"
        '  "questions": [\n'
        "    {\n"
        '      "text": "câu hỏi...",\n'
        '      "options": ["A...", "B...", "C...", "D..."],\n'
        '      "correct_index": 0,\n'
        '      "standards": ["T1","T3"]\n'
        "    }, ...\n"
        "  ]\n"
        "}\n"
        "Không giải thích thêm, không thêm markdown, chỉ trả JSON.\n"
        "\n"
        "Tóm tắt chuẩn T1–T15:\n"
        f"{standards_text}\n"
    )

    user_prompt = (
        f"Hãy tạo {n} câu hỏi trắc nghiệm Toán 9 ({difficulty}) cho mode '{mode}'. "
        "Mỗi câu hỏi 4 đáp án, đúng duy nhất 1 đáp án."
    )

    completion = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.4,
    )
    raw = completion.choices[0].message.content

    try:
        data = json.loads(raw)
    except Exception as e:
        raise RuntimeError(f"JSON parse error: {e} – content: {raw!r}") from e

    questions_raw = data.get("questions", [])
    questions: List[QuizQuestion] = []
    next_id_base = 1000 if mode == "output" else 1

    for i, q in enumerate(questions_raw):
        text = str(q.get("text", "")).strip()
        options = q.get("options", [])
        if len(options) != 4:
            continue
        correct_index = int(q.get("correct_index", 0))
        if not (0 <= correct_index <= 3):
            continue
        standards_codes = [
            s for s in q.get("standards", []) if s in STANDARDS
        ] or ["T1"]

        questions.append(
            QuizQuestion(
                id=next_id_base + i,
                text=text,
                options=[str(o) for o in options],
                correct_index=correct_index,
                standards=standards_codes,
            )
        )

    if not questions:
        raise RuntimeError("Không sinh được câu hỏi hợp lệ nào từ OpenAI.")

    return questions[:n]


@app.get("/api/input_quiz", response_model=QuizResponse)
async def api_input_quiz():
    # Ưu tiên: sinh bằng OpenAI
    try:
        questions = _generate_quiz_with_openai(mode="input", n=10)
        return QuizResponse(questions=questions)
    except Exception as e:
        print(f"[WARN] Lỗi sinh quiz đầu vào bằng OpenAI, dùng fallback: {e}")
        questions = _pick_random_questions(INPUT_QUESTION_BANK, n=10)
        return QuizResponse(questions=questions)


@app.get("/api/output_quiz", response_model=QuizResponse)
async def api_output_quiz():
    # Ưu tiên: sinh bằng OpenAI
    try:
        questions = _generate_quiz_with_openai(mode="output", n=10)
        return QuizResponse(questions=questions)
    except Exception as e:
        print(f"[WARN] Lỗi sinh quiz đầu ra bằng OpenAI, dùng fallback: {e}")
        questions = _pick_random_questions(OUTPUT_QUESTION_BANK, n=10)
        return QuizResponse(questions=questions)


# ========================
#  FILE PARSE (TXT/PDF)
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
