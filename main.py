# main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Literal
from datetime import datetime, timedelta
from io import BytesIO
import json
import os

from openai import OpenAI  # pip install openai
from reportlab.lib.pagesizes import A4  # pip install reportlab
from reportlab.pdfgen import canvas
from pypdf import PdfReader  # pip install pypdf

# ============================================================
# CẤU HÌNH CƠ BẢN – THEO ĐÚNG MÔ TẢ DỰ ÁN “TRỢ LÝ TOÁN AI” :contentReference[oaicite:1]{index=1}
# - KHÔNG LƯU DỮ LIỆU LÂU DÀI
# - MỌI PHÂN TÍCH CHỈ TỒN TẠI TRONG REQUEST/PHIÊN
# - HỖ TRỢ: test đầu vào/đầu ra, phân tích lỗi, lộ trình học, giải bài, chat + phân loại T1–T15
# ============================================================

app = FastAPI(
    title="K9 Math AI Assistant",
    description="Trợ lý Toán AI cho học sinh lớp 9 – không lưu dữ liệu lâu dài.",
    version="2.0.0",
)

# OpenAI client – đọc từ biến môi trường OPENAI_API_KEY
client = OpenAI()  # đảm bảo đã setx OPENAI_API_KEY "sk-..."

# ============================================================
# CHUẨN ĐẦU RA & NGÂN HÀNG CÂU HỎI (T1–T15)
# ============================================================

class Standard(BaseModel):
    code: str  # Ví dụ: "T1"
    name: str  # Tên chuẩn
    description: str  # Mô tả chuẩn theo file chuan_dau_ra


class Question(BaseModel):
    id: str
    text: str
    options: List[str]
    correct_index: int
    standard_code: str  # T1..T15
    topic: str  # tên chủ đề, vd: "Đại số – Phương trình bậc nhất"


# Bạn có thể lưu chuẩn và câu hỏi trong 1 file JSON.
# Ví dụ file chuan_dau_ra_questions.json:
# {
#   "standards": [
#     {"code": "T1", "name": "Phương trình bậc nhất", "description": "..."},
#     ...
#   ],
#   "questions": [
#     {
#       "id": "Q1",
#       "text": "Giải phương trình 2x+5=15...",
#       "options": ["3","5","7","10"],
#       "correct_index": 0,
#       "standard_code": "T1",
#       "topic": "Đại số – Phương trình bậc nhất"
#     },
#     ...
#   ]
# }

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
QUESTION_FILE = os.path.join(BASE_DIR, "chuan_dau_ra_questions.json")

STANDARDS: Dict[str, Standard] = {}
QUESTION_BANK: List[Question] = []


def load_questions_from_file() -> None:
    """Đọc chuẩn đầu ra & câu hỏi từ file JSON. Nếu không có thì dùng demo."""
    global STANDARDS, QUESTION_BANK
    if os.path.exists(QUESTION_FILE):
        with open(QUESTION_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        stds = data.get("standards", [])
        STANDARDS = {s["code"]: Standard(**s) for s in stds}

        q_list = data.get("questions", [])
        QUESTION_BANK = [Question(**q) for q in q_list]
    else:
        # DEMO fallback: 3 chuẩn T1–T3. Bạn sửa thành T1–T15 thật theo file chuan_dau_ra của bạn.
        demo_standards = [
            Standard(code="T1", name="Phương trình bậc nhất", description="Giải phương trình bậc nhất một ẩn."),
            Standard(code="T2", name="Tỉ lệ & phần trăm", description="Giải bài toán liên quan đến phần trăm."),
            Standard(code="T3", name="Góc & tam giác", description="Tính số đo các góc trong tam giác."),
        ]
        STANDARDS = {s.code: s for s in demo_standards}

        QUESTION_BANK = [
            Question(
                id="Q1",
                text="Giải phương trình: 2x + 5 = 15. Giá trị của x là:",
                options=["3", "5", "7", "10"],
                correct_index=0,
                standard_code="T1",
                topic="Đại số – Phương trình bậc nhất",
            ),
            Question(
                id="Q2",
                text="Một lớp có 40 học sinh, trong đó 60% là nữ. Số học sinh nữ là:",
                options=["18", "20", "22", "24"],
                correct_index=3,
                standard_code="T2",
                topic="Đại số – Tỉ lệ & phần trăm",
            ),
            Question(
                id="Q3",
                text="Trong tam giác, hai góc lần lượt là 60° và 45°. Góc còn lại là:",
                options=["45°", "60°", "75°", "90°"],
                correct_index=2,
                standard_code="T3",
                topic="Hình học – Tam giác & góc",
            ),
        ]


load_questions_from_file()

# ============================================================
# MODELS CHUNG
# ============================================================

class EntryTestRequest(BaseModel):
    num_questions: int = Field(10, ge=3, le=30)
    standards: Optional[List[str]] = None  # Ví dụ: ["T1","T2"] – nếu None thì trộn tất cả


class UserAnswer(BaseModel):
    question_id: str
    answer_index: Optional[int] = None


class StandardResult(BaseModel):
    standard_code: str
    standard_name: str
    correct: int
    total: int
    mastery_rate: float  # 0–1


class EntryTestSubmission(BaseModel):
    answers: List[UserAnswer]


class EntryTestAnalysis(BaseModel):
    standard_results: List[StandardResult]
    global_comment: str
    weak_standards: List[str]  # danh sách code T1..T15 nên ưu tiên


class StudyPlanRequest(BaseModel):
    standard_results: List[StandardResult]
    days: int = Field(7, ge=1, le=30)
    minutes_per_day: int = Field(30, ge=15, le=120)


class StudyTask(BaseModel):
    day_index: int
    standard_code: str
    topic: str
    activity_type: str
    description: str
    estimated_minutes: int


class StudyPlan(BaseModel):
    start_date: datetime
    tasks: List[StudyTask]


class ExerciseAnalysisRequest(BaseModel):
    problem_text: str
    student_answer: Optional[str] = None
    language: str = "vi"  # "vi" hoặc "en"


class ExerciseAnalysisResponse(BaseModel):
    detected_standard_codes: List[str]  # vd ["T1","T3"]
    needed_knowledge: str
    explanation: str
    step_by_step_hint: str


class SessionReportRequest(BaseModel):
    nickname: Optional[str] = None
    entry_analysis: EntryTestAnalysis
    study_plan: StudyPlan
    notes: Optional[str] = None


# Chat models – cho UI giống ChatGPT
class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]


class ChatResponse(BaseModel):
    reply: str
    standards: List[str]  # ví dụ ["T1","T4"] – chuẩn liên quan trong cuộc trò chuyện


# ============================================================
# HELPER: ĐOÁN CHUẨN T1–T15 TỪ TEXT (GỢI Ý CHO EXERCISE & CHAT)
# ============================================================

def guess_standards_from_text(text: str) -> List[str]:
    """Đoán sơ bộ chuẩn T1–T15 dựa vào từ khóa (fallback nếu AI không dùng)."""
    t = text.lower()
    tags: List[str] = []
    if "phương trình" in t or "giải phương trình" in t:
        tags.append("T1")
    if "phần trăm" in t or "%" in t or "tỉ lệ" in t:
        tags.append("T2")
    if "tam giác" in t or "góc" in t:
        tags.append("T3")
    # TODO: khi có đủ chuẩn T4..T15, thêm rule ở đây

    # Lọc những chuẩn thực sự tồn tại
    tags = [code for code in tags if code in STANDARDS]
    # Nếu không đoán được gì thì trả empty list
    return list(dict.fromkeys(tags))  # remove dup


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():
    return {
        "message": "Math AI Assistant is running",
        "note": "Hệ thống không lưu dữ liệu lâu dài. Mọi phân tích chỉ tồn tại trong từng request/p phiên.",
        "standards_loaded": list(STANDARDS.keys()),
    }

# ============================================================
# 1. TẠO ĐỀ TEST ĐẦU VÀO / ĐẦU RA TỪ CHUẨN T1–T15
# ============================================================

@app.post("/entry-test/generate", response_model=List[Question])
def generate_entry_test(req: EntryTestRequest):
    """
    Sinh đề kiểm tra dựa trên câu hỏi gắn với chuẩn đầu ra (T1–T15).
    Test chỉ phục vụ phân tích trong phiên, không lưu kết quả trên server.
    """
    import random

    if not QUESTION_BANK:
        raise HTTPException(status_code=500, detail="QUESTION_BANK trống – hãy cấu hình chuan_dau_ra_questions.json.")

    if req.standards:
        valid_codes = [s for s in req.standards if s in STANDARDS]
        if not valid_codes:
            valid_codes = list(STANDARDS.keys())
        filtered = [q for q in QUESTION_BANK if q.standard_code in valid_codes]
    else:
        filtered = QUESTION_BANK[:]

    if not filtered:
        raise HTTPException(status_code=500, detail="Không tìm thấy câu hỏi tương ứng với các chuẩn đã chọn.")

    n = min(req.num_questions, len(filtered))
    selected = random.sample(filtered, n)
    # KHÔNG gửi đáp án đúng ra client; ở đây Question có correct_index,
    # nhưng frontend chỉ nên dùng text + options. Bạn có thể tạo model riêng nếu cần ẩn.
    return selected


@app.post("/entry-test/analyze", response_model=EntryTestAnalysis)
def analyze_entry_test(submission: EntryTestSubmission):
    """
    Nhận câu trả lời, phân tích theo từng chuẩn đầu ra (T1–T15).
    Không lưu điểm hay câu trả lời trên server (purely stateless).
    """
    q_map: Dict[str, Question] = {q.id: q for q in QUESTION_BANK}
    stats: Dict[str, Dict[str, int]] = {}

    for ans in submission.answers:
        q = q_map.get(ans.question_id)
        if not q:
            continue
        code = q.standard_code
        if code not in stats:
            stats[code] = {"correct": 0, "total": 0}
        stats[code]["total"] += 1
        if ans.answer_index is not None and ans.answer_index == q.correct_index:
            stats[code]["correct"] += 1

    standard_results: List[StandardResult] = []
    weak: List[str] = []

    for code, s in stats.items():
        total = s["total"]
        correct = s["correct"]
        mastery = correct / total if total > 0 else 0.0
        standard = STANDARDS.get(code, Standard(code=code, name=code, description=""))
        standard_results.append(
            StandardResult(
                standard_code=code,
                standard_name=standard.name,
                correct=correct,
                total=total,
                mastery_rate=round(mastery, 2),
            )
        )
        if mastery < 0.7:
            weak.append(code)

    if not standard_results:
        comment = "Chưa đủ dữ liệu để phân tích. Hãy làm lại bài test với nhiều câu hơn."
    else:
        comment = (
            "Bạn đã hoàn thành bài test. Các chuẩn có tỷ lệ đúng dưới 70% là những chuẩn nên ưu tiên ôn tập."
        )

    return EntryTestAnalysis(
        standard_results=standard_results,
        global_comment=comment,
        weak_standards=weak,
    )

# ============================================================
# 2. TẠO LỘ TRÌNH HỌC THEO CHUẨN YẾU (SPACED REPETITION TRONG PHIÊN)
# ============================================================

@app.post("/study-plan", response_model=StudyPlan)
def create_study_plan(req: StudyPlanRequest):
    import math

    start = datetime.now()
    tasks: List[StudyTask] = []

    if not req.standard_results:
        tasks.append(
            StudyTask(
                day_index=0,
                standard_code="T0",
                topic="Tổng quan Toán 9",
                activity_type="định hướng",
                description="Ôn tập tổng quan các kiến thức quan trọng của Toán 9.",
                estimated_minutes=req.minutes_per_day,
            )
        )
        return StudyPlan(start_date=start, tasks=tasks)

    # Sắp xếp chuẩn theo độ thành thạo tăng dần (yếu trước)
    sorted_results = sorted(req.standard_results, key=lambda r: r.mastery_rate)

    for day in range(req.days):
        r = sorted_results[day % len(sorted_results)]
        std = STANDARDS.get(r.standard_code, Standard(code=r.standard_code, name=r.standard_name, description=""))
        total = req.minutes_per_day

        tasks.append(
            StudyTask(
                day_index=day,
                standard_code=r.standard_code,
                topic=std.name,
                activity_type="ôn lý thuyết",
                description=f"Đọc lại lý thuyết và ví dụ mẫu liên quan đến chuẩn {r.standard_code}: {std.name}.",
                estimated_minutes=int(total * 0.3),
            )
        )
        tasks.append(
            StudyTask(
                day_index=day,
                standard_code=r.standard_code,
                topic=std.name,
                activity_type="làm bài tập",
                description=f"Làm 4–6 bài tập luyện tập tập trung vào chuẩn {r.standard_code}.",
                estimated_minutes=int(total * 0.5),
            )
        )
        tasks.append(
            StudyTask(
                day_index=day,
                standard_code=r.standard_code,
                topic=std.name,
                activity_type="ôn nhanh (spaced repetition)",
                description=f"Làm lại các câu hỏi tương tự những câu bạn đã sai ở chuẩn {r.standard_code}.",
                estimated_minutes=total - int(total * 0.3) - int(total * 0.5),
            )
        )

    return StudyPlan(start_date=start, tasks=tasks)

# ============================================================
# 3. GIẢI THÍCH BÀI TẬP BẰNG AI + GẮN CHUẨN T1–T15
# ============================================================

async def call_openai_for_exercise(req: ExerciseAnalysisRequest, default_tags: List[str]) -> ExerciseAnalysisResponse:
    """
    Gọi OpenAI để giải thích bài toán + yêu cầu phân loại chuẩn T1–T15.
    Không lưu nội dung trên server (chỉ truyền qua OpenAI và trả về).
    """
    lang = req.language.lower()
    tags_str = ", ".join(sorted(STANDARDS.keys()))

    if lang == "en":
        system_prompt = (
            "You are a Grade 9 math tutor for Vinschool students in Vietnam. "
            "Explain clearly step by step, and classify each problem into 1–3 learning outcomes labeled T1–T15."
        )
        user_prompt = f"""
Problem:
{req.problem_text}

Student's answer (may be empty):
{req.student_answer or "(no answer)"}

Available outcome codes: {tags_str}.
1) Briefly explain what knowledge is needed.
2) Provide a full step-by-step solution.
3) Provide a shorter hint-only version (no final answer).
4) Decide which outcome codes (T1..T15) are most relevant to this problem.

At the end, write on a separate line:
TAGS: T1,T3 (for example).
"""
    else:
        system_prompt = (
            "Bạn là trợ lý Toán lớp 9 cho học sinh Vinschool. "
            "Giải thích ngắn gọn, dễ hiểu, từng bước; đồng thời phân loại bài toán theo các chuẩn đầu ra T1–T15."
        )
        user_prompt = f"""
Bài toán:
{req.problem_text}

Bài làm của học sinh (nếu có):
{req.student_answer or "(học sinh chưa làm)"}

Các chuẩn đầu ra có thể dùng: {tags_str}.

Yêu cầu:
1) Giải thích kiến thức cần có để làm bài này.
2) Trình bày lời giải từng bước, rõ ràng.
3) Viết một phiên bản chỉ gợi ý (không ghi đáp số).
4) Chọn 1–3 chuẩn T1..T15 phù hợp nhất với bài toán này.

Ở cuối câu trả lời, hãy viết trên một dòng riêng:
TAGS: T1,T3 (ví dụ).
"""

    completion = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.4,
    )
    content = completion.choices[0].message.content or ""

    # Tách TAGS
    detected_tags = default_tags[:]
    for line in content.splitlines()[::-1]:  # duyệt từ cuối lên
        if line.strip().upper().startswith("TAGS:"):
            raw = line.split(":", 1)[1]
            codes = [c.strip().upper() for c in raw.replace(" ", "").split(",") if c.strip()]
            detected_tags = [c for c in codes if c in STANDARDS]
            # bỏ dòng TAGS khỏi phần giải thích
            content = "\n".join(l for l in content.splitlines() if l != line)
            break

    # Tách thành 3 phần thô (không cần quá hoàn hảo)
    needed = content
    explanation = content
    hint = content
    parts = content.split("3.")
    if len(parts) == 2:
        explanation_part = parts[0]
        hint = "3." + parts[1]
        subparts = explanation_part.split("2.")
        if len(subparts) == 2:
            needed = subparts[0]
            explanation = "2." + subparts[1]

    return ExerciseAnalysisResponse(
        detected_standard_codes=detected_tags,
        needed_knowledge=needed.strip(),
        explanation=explanation.strip(),
        step_by_step_hint=hint.strip(),
    )


@app.post("/exercise/analyze", response_model=ExerciseAnalysisResponse)
async def analyze_exercise(req: ExerciseAnalysisRequest):
    default_tags = guess_standards_from_text(req.problem_text)
    return await call_openai_for_exercise(req, default_tags)

# ============================================================
# 4. XUẤT BÁO CÁO PDF TRONG PHIÊN (KHÔNG LƯU LẠI)
# ============================================================

def build_pdf_bytes(report: SessionReportRequest) -> bytes:
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    w, h = A4
    y = h - 50

    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, y, "BÁO CÁO PHIÊN HỌC TOÁN AI")
    y -= 30

    c.setFont("Helvetica", 11)
    nickname = report.nickname or "Học sinh"
    c.drawString(50, y, f"Học sinh: {nickname}")
    y -= 20
    c.drawString(50, y, f"Thời gian tạo: {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    y -= 30

    # Kết quả theo chuẩn
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "1. Kết quả bài test theo chuẩn đầu ra:")
    y -= 20
    c.setFont("Helvetica", 11)
    for r in report.entry_analysis.standard_results:
        line = f"- {r.standard_code} ({r.standard_name}): {r.correct}/{r.total} câu đúng (~{int(r.mastery_rate*100)}%)"
        c.drawString(60, y, line)
        y -= 15
        if y < 80:
            c.showPage()
            y = h - 50

    y -= 10
    c.drawString(50, y, "Nhận xét tổng quát:")
    y -= 15
    for line in report.entry_analysis.global_comment.splitlines():
        c.drawString(60, y, line)
        y -= 15
        if y < 80:
            c.showPage()
            y = h - 50

    # Lộ trình học
    y -= 10
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "2. Lộ trình học gợi ý:")
    y -= 20
    c.setFont("Helvetica", 11)
    for task in report.study_plan.tasks:
        day_str = (report.study_plan.start_date + timedelta(days=task.day_index)).strftime("%d/%m")
        std = f"{task.standard_code}"
        line = f"Ngày {day_str} – {std} – {task.topic} – {task.activity_type} (~{task.estimated_minutes} phút)"
        c.drawString(60, y, line)
        y -= 15
        if y < 80:
            c.showPage()
            y = h - 50

    # Ghi chú
    if report.notes:
        y -= 10
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, y, "3. Ghi chú thêm:")
        y -= 20
        c.setFont("Helvetica", 11)
        for line in report.notes.splitlines():
            c.drawString(60, y, line)
            y -= 15
            if y < 80:
                c.showPage()
                y = h - 50

    c.showPage()
    c.save()
    buf.seek(0)
    return buf.read()


@app.post("/session/report/pdf")
def export_session_report(report: SessionReportRequest):
    pdf_bytes = build_pdf_bytes(report)
    return StreamingResponse(
        BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={
            "Content-Disposition": 'attachment; filename="math_ai_session_report.pdf"'
        },
    )

# ============================================================
# 5. FILE UPLOAD – ĐỌC/QUÉT FILE ĐỂ DÙNG TRONG CHAT
# ============================================================

@app.post("/file/parse")
async def parse_file(file: UploadFile = File(...)):
    """
    Đọc nội dung file (txt / pdf) để client đưa vào đoạn chat.
    KHÔNG lưu file lại trên server.
    """
    content = ""
    if file.content_type in ("text/plain", "application/json"):
        raw = await file.read()
        content = raw.decode("utf-8", errors="ignore")
    elif file.content_type == "application/pdf" or file.filename.lower().endswith(".pdf"):
        raw = await file.read()
        reader = PdfReader(BytesIO(raw))
        pages_text = []
        for page in reader.pages:
            pages_text.append(page.extract_text() or "")
        content = "\n".join(pages_text)
    else:
        raise HTTPException(status_code=400, detail=f"Không hỗ trợ loại file: {file.content_type}")

    # Giới hạn độ dài để an toàn
    if len(content) > 8000:
        content = content[:8000] + "\n[...đã cắt bớt nội dung vì quá dài...]"

    return {"filename": file.filename, "content": content}

# ============================================================
# 6. CHAT KIỂU CHATGPT + PHÂN LOẠI T1–T15
# ============================================================

@app.post("/chat/message", response_model=ChatResponse)
async def chat_message(req: ChatRequest):
    """
    Nhận toàn bộ history messages từ client (stateless),
    trả lời như ChatGPT + phân loại cuộc trò chuyện vào 1–3 chuẩn T1–T15.
    """
    if not req.messages:
        raise HTTPException(status_code=400, detail="messages không được rỗng")

    # Chuẩn T1–T15 + mô tả dạng ngắn để AI biết
    standards_desc = "\n".join(
        f"{code}: {std.name}" for code, std in STANDARDS.items()
    ) or "T1..T15: các chuẩn Toán 9."

    system_prompt = (
        "Bạn là trợ lý Toán AI cho học sinh lớp 9 Vinschool. "
        "Bạn trả lời ngắn gọn, rõ ràng, tập trung vào giải thích bản chất, tránh giải hộ toàn bộ nếu không cần. "
        "Sau khi trả lời, hãy tự suy nghĩ xem cuộc trò chuyện này liên quan đến chuẩn đầu ra nào (T1–T15). "
        "Các chuẩn (tóm tắt):\n"
        + standards_desc
        + "\n"
        "Ở cuối câu trả lời, trên 1 dòng riêng, hãy viết: TAGS: T1,T3 (ví dụ)."
    )

    messages = [{"role": "system", "content": system_prompt}]
    for m in req.messages:
        messages.append({"role": m.role, "content": m.content})

    completion = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages,
        temperature=0.5,
    )
    full_reply = completion.choices[0].message.content or ""

    # Parse TAGS
    tags: List[str] = []
    for line in full_reply.splitlines()[::-1]:
        if line.strip().upper().startswith("TAGS:"):
            raw = line.split(":", 1)[1]
            tags = [c.strip().upper() for c in raw.replace(" ", "").split(",") if c.strip()]
            tags = [c for c in tags if c in STANDARDS]
            full_reply = "\n".join(l for l in full_reply.splitlines() if l != line)
            break

    if not tags:
        # fallback: đoán từ toàn bộ hội thoại
        text_all = "\n".join(m.content for m in req.messages)
        tags = guess_standards_from_text(text_all)

    return ChatResponse(reply=full_reply.strip(), standards=tags)
