from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Literal
from datetime import datetime, timedelta
from io import BytesIO
import os
import json

from openai import OpenAI           # pip install openai
from reportlab.lib.pagesizes import A4   # pip install reportlab
from reportlab.pdfgen import canvas
from pypdf import PdfReader         # pip install pypdf


# ============================================================
# CẤU HÌNH CHUNG
# ============================================================

app = FastAPI(
    title="K9 Math AI Assistant",
    description="Trợ lý Toán AI lớp 9 – AI For Good – không lưu dữ liệu lâu dài.",
    version="2.0.0",
)

client = OpenAI()  # Đọc OPENAI_API_KEY từ biến môi trường

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHUAN_DAU_RA_FILE = os.path.join(BASE_DIR, "chuan_dau_ra")


# ============================================================
# MÔ HÌNH CHUẨN ĐẦU RA & CÂU HỎI
# ============================================================

class Standard(BaseModel):
    code: str      # "T1"
    name: str      # "Số và Lũy Thừa (Đại số)"
    description: str


class QuestionInternal(BaseModel):
    id: str
    text: str
    options: List[str]
    correct_index: int
    standard_code: str   # "T1".."T15"


class QuestionPublic(BaseModel):
    id: str
    text: str
    options: List[str]
    standard_code: str
    standard_name: str


class EntryTestRequest(BaseModel):
    num_questions: int = Field(10, ge=3, le=30)
    standards: Optional[List[str]] = None  # nếu None thì dùng tất cả T1..T15


class UserAnswer(BaseModel):
    question_id: str
    answer_index: Optional[int] = None


class StandardResult(BaseModel):
    standard_code: str
    standard_name: str
    correct: int
    total: int
    mastery_rate: float  # 0..1


class EntryTestSubmission(BaseModel):
    answers: List[UserAnswer]


class EntryTestAnalysis(BaseModel):
    standard_results: List[StandardResult]
    global_comment: str
    weak_standards: List[str]  # các code T1..T15 nên ưu tiên


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
    detected_standard_codes: List[str]
    needed_knowledge: str
    explanation: str
    step_by_step_hint: str


class SessionReportRequest(BaseModel):
    nickname: Optional[str] = None
    entry_analysis: EntryTestAnalysis
    study_plan: StudyPlan
    notes: Optional[str] = None


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]


class ChatResponse(BaseModel):
    reply: str
    standards: List[str]  # các chuẩn T1..T15 liên quan


# ============================================================
# ĐỌC FILE `chuan_dau_ra` (MARKDOWN) THÀNH T1..T15
# ============================================================

def parse_chuan_dau_ra_md(path: str) -> Dict[str, Standard]:
    """
    Parse file Markdown 'chuan_dau_ra' thành dict { 'T1': Standard(...), ... }.
    Định dạng mong đợi (giống file trên GitHub):
        ## T1 – Số và Lũy Thừa (Đại số)
        ...
        ## T2 – ...
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Không tìm thấy file {path}")

    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    standards: Dict[str, Standard] = {}
    current_code: Optional[str] = None
    current_name: str = ""
    current_text: List[str] = []

    for line in lines:
        s = line.strip()

        # Bắt đầu chuẩn mới: dòng dạng "## T1 – ..." (có thể có/dư khoảng trắng)
        if s.startswith("## "):
            after = s[3:].strip()   # ví dụ "T1 – Số và Lũy Thừa..."
            if not after.upper().startswith("T"):
                # ví dụ "# Chuẩn đầu ra Toán 9..." sẽ bị bỏ qua
                # hoặc "## YÊU CẦU ĐẠT ĐƯỢC" cũng bỏ
                pass
            else:
                # Lưu chuẩn cũ nếu có
                if current_code:
                    standards[current_code] = Standard(
                        code=current_code,
                        name=current_name,
                        description="\n".join(current_text).strip()
                    )

                # Lấy code: "T1" hoặc "T10"
                head = after.split("–", 1)[0].strip()   # "T1" hoặc "T10 ..."
                code = head.split()[0]                  # "T1"
                current_code = code

                # Lấy tên chuẩn (sau dấu –)
                if "–" in after:
                    current_name = after.split("–", 1)[1].strip()
                else:
                    current_name = current_code

                current_text = []
                continue

        # Nếu đang ở trong 1 chuẩn, gom text mô tả
        if current_code:
            current_text.append(line.rstrip())

    # Lưu chuẩn cuối cùng
    if current_code:
        standards[current_code] = Standard(
            code=current_code,
            name=current_name,
            description="\n".join(current_text).strip()
        )

    return standards

STANDARDS: Dict[str, Standard] = parse_chuan_dau_ra_md(CHUAN_DAU_RA_FILE)

# QUESTION_STORE: lưu tạm câu hỏi vừa sinh ra trong bộ nhớ để chấm điểm
QUESTION_STORE: Dict[str, QuestionInternal] = {}


# ============================================================
# HELPER: ĐOÁN CHUẨN TỪ TEXT
# ============================================================

def guess_standards_from_text(text: str) -> List[str]:
    t = text.lower()
    tags: List[str] = []

    if "phương trình" in t or "phuong trinh" in t:
        tags.append("T1")
    if "phần trăm" in t or "%" in t or "phan tram" in t or "tỉ lệ" in t or "ti le" in t:
        tags.append("T2")
    if "tam giác" in t or "tam giac" in t or "góc" in t or "goc" in t:
        tags.append("T3")

    # Có thể thêm rule cho T4..T15 sau

    tags = [c for c in tags if c in STANDARDS]
    # loại trùng
    return list(dict.fromkeys(tags))


# ============================================================
# ROOT
# ============================================================

@app.get("/api/health")
def root():
    return {
        "message": "Math AI Assistant is running",
        "standards_loaded": list(STANDARDS.keys()),
        "note": "Không lưu thông tin cá nhân hay bài làm trên server. Câu hỏi sinh động theo từng request.",
    }
    
from fastapi.staticfiles import StaticFiles

# Serve index.html và các file tĩnh ở thư mục hiện tại (cùng chỗ với main.py)
app.mount("/", StaticFiles(directory=".", html=True), name="static")


# ============================================================
# 1. OPTION A: SINH ĐỀ TỪ CHUẨN T1–T15 BẰNG OPENAI
# ============================================================

@app.post("/entry-test/generate", response_model=List[QuestionPublic])
async def generate_entry_test(req: EntryTestRequest):
    """
    Tạo đề kiểm tra bằng OpenAI, bám sát chuẩn đầu ra T1–T15.
    Không sử dụng ngân hàng câu hỏi cố định.
    """
    if not STANDARDS:
        raise HTTPException(status_code=500, detail="Không load được chuẩn từ file chuan_dau_ra.")

    # Chọn chuẩn cần dùng
    if req.standards:
        codes = [c for c in req.standards if c in STANDARDS]
        if not codes:
            codes = list(STANDARDS.keys())
    else:
        codes = list(STANDARDS.keys())

    # Context chuẩn cho OpenAI
    standards_text = "\n\n".join(
        f"{code}: {STANDARDS[code].name}\n{STANDARDS[code].description}"
        for code in codes
    )

    user_prompt = f"""
Bạn là giáo viên Toán 9. Hãy tạo {req.num_questions} câu hỏi trắc nghiệm 4 lựa chọn
theo chương trình Toán 9 Việt Nam, gắn với các chuẩn T1–T15 sau:

{standards_text}

YÊU CẦU:
- Mỗi câu tập trung chủ yếu vào 1 chuẩn (standard_code).
- Nội dung rõ ràng, dùng ký hiệu toán học cơ bản.
- options là danh sách 4 phương án; correct_index là chỉ số của đáp án đúng (0,1,2,3).
- Trả lời BẰNG JSON THUẦN theo đúng cấu trúc:

[
  {{
    "id": "Q1",
    "text": "Nội dung câu hỏi...",
    "options": ["A", "B", "C", "D"],
    "correct_index": 1,
    "standard_code": "T1"
  }},
  ...
]
Không thêm chú thích hay giải thích bên ngoài JSON.
"""

    completion = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "Bạn là giáo viên Toán lớp 9, thiết kế đề kiểm tra theo chuẩn đầu ra."},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
    )

    raw = completion.choices[0].message.content or ""

    try:
        data = json.loads(raw)
        questions_internal = [QuestionInternal(**q) for q in data]
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi parse JSON từ OpenAI: {e}. Nội dung nhận được: {raw[:500]}"
        )

    # Cập nhật QUESTION_STORE (trong bộ nhớ) – không chứa thông tin cá nhân
    QUESTION_STORE.clear()
    for q in questions_internal:
        QUESTION_STORE[q.id] = q

    # Trả bản public (không lộ đáp án, chỉ kèm tên chuẩn)
    result: List[QuestionPublic] = []
    for q in questions_internal:
        std = STANDARDS.get(q.standard_code)
        if not std:
            continue
        result.append(
            QuestionPublic(
                id=q.id,
                text=q.text,
                options=q.options,
                standard_code=q.standard_code,
                standard_name=std.name,
            )
        )

    return result


# ============================================================
# 2. PHÂN TÍCH KẾT QUẢ ĐỀ – THEO CHUẨN ĐẦU RA
# ============================================================

@app.post("/entry-test/analyze", response_model=EntryTestAnalysis)
def analyze_entry_test(sub: EntryTestSubmission):
    """
    Chấm điểm theo chuẩn T1–T15 dựa trên QUESTION_STORE (các câu hỏi vừa sinh).
    Không lưu bài làm hay tên học sinh.
    """
    if not QUESTION_STORE:
        raise HTTPException(
            status_code=400,
            detail="Không có đề trong bộ nhớ. Hãy gọi /entry-test/generate trước khi phân tích."
        )

    stats: Dict[str, Dict[str, int]] = {}
    for ans in sub.answers:
        q = QUESTION_STORE.get(ans.question_id)
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
        std = STANDARDS.get(code, Standard(code=code, name=code, description=""))
        standard_results.append(
            StandardResult(
                standard_code=code,
                standard_name=std.name,
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
        comment = "Các chuẩn có tỷ lệ đúng dưới 70% là những chuẩn bạn nên ưu tiên ôn tập."

    return EntryTestAnalysis(
        standard_results=standard_results,
        global_comment=comment,
        weak_standards=weak,
    )


# ============================================================
# 3. TẠO LỘ TRÌNH HỌC (SPACED REPETITION)
# ============================================================

@app.post("/study-plan", response_model=StudyPlan)
def create_study_plan(req: StudyPlanRequest):
    start = datetime.now()
    tasks: List[StudyTask] = []

    if not req.standard_results:
        tasks.append(
            StudyTask(
                day_index=0,
                standard_code="T0",
                topic="Tổng quan Toán 9",
                activity_type="định hướng",
                description="Ôn tập tổng quan các kiến thức trọng tâm của Toán 9.",
                estimated_minutes=req.minutes_per_day,
            )
        )
        return StudyPlan(start_date=start, tasks=tasks)

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
                description=f"Làm lại những dạng bài mà trước đó bạn đã làm sai liên quan tới chuẩn {r.standard_code}.",
                estimated_minutes=total - int(total * 0.3) - int(total * 0.5),
            )
        )

    return StudyPlan(start_date=start, tasks=tasks)


# ============================================================
# 4. GIẢI THÍCH BÀI TẬP BẰNG AI + GẮN CHUẨN
# ============================================================

async def call_openai_for_exercise(req: ExerciseAnalysisRequest, default_tags: List[str]) -> ExerciseAnalysisResponse:
    tags_str = ", ".join(sorted(STANDARDS.keys())) or "T1..T15"

    if req.language.lower() == "en":
        system_prompt = (
            "You are a Grade 9 math tutor for Vietnamese students. "
            "Explain clearly and concisely, step by step, and classify the problem into learning outcomes T1–T15."
        )
        user_prompt = f"""
Problem:
{req.problem_text}

Student answer (may be empty):
{req.student_answer or "(no answer)"}

Available outcome codes: {tags_str}

1) Briefly explain the required knowledge.
2) Provide a full step-by-step solution.
3) Provide a shorter hint-only version (no final answer).
4) Choose 1–3 outcome codes (T1..T15) related to this problem.

At the end, on a separate line, write:
TAGS: T1,T3  (for example).
"""
    else:
        system_prompt = (
            "Bạn là trợ lý Toán lớp 9 cho học sinh Vinschool. "
            "Giải thích dễ hiểu, từng bước, và phân loại bài toán theo chuẩn T1–T15."
        )
        user_prompt = f"""
Bài toán:
{req.problem_text}

Bài làm của học sinh (nếu có):
{req.student_answer or "(học sinh chưa làm)"}

Các chuẩn đầu ra có thể dùng: {tags_str}.

1) Giải thích kiến thức cần có để làm bài này.
2) Trình bày lời giải từng bước, rõ ràng.
3) Viết phiên bản chỉ gợi ý (không ghi đáp số).
4) Chọn 1–3 chuẩn T1..T15 phù hợp.

Ở cuối câu trả lời, trên 1 dòng riêng, hãy viết:
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

    # Tách TAGS ở cuối
    tags = default_tags[:]
    lines = content.splitlines()
    for line in reversed(lines):
        if line.strip().upper().startswith("TAGS:"):
            raw = line.split(":", 1)[1]
            codes = [c.strip().upper() for c in raw.replace(" ", "").split(",") if c.strip()]
            tags = [c for c in codes if c in STANDARDS]
            content = "\n".join(l for l in lines if l != line)
            break

    # Tách đại khái 3 phần
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
        detected_standard_codes=tags,
        needed_knowledge=needed.strip(),
        explanation=explanation.strip(),
        step_by_step_hint=hint.strip(),
    )


@app.post("/exercise/analyze", response_model=ExerciseAnalysisResponse)
async def analyze_exercise(req: ExerciseAnalysisRequest):
    default_tags = guess_standards_from_text(req.problem_text)
    return await call_openai_for_exercise(req, default_tags)


# ============================================================
# 5. XUẤT BÁO CÁO PDF (CHỈ DỰA TRÊN DATA CLIENT GỬI)
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
        line = f"Ngày {day_str} – {task.standard_code} – {task.topic} – {task.activity_type} (~{task.estimated_minutes} phút)"
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
        headers={"Content-Disposition": 'attachment; filename="math_ai_session_report.pdf"'},
    )


# ============================================================
# 6. FILE UPLOAD – ĐỌC NỘI DUNG ĐỂ ĐƯA VÀO CHAT
# ============================================================

@app.post("/file/parse")
async def parse_file(file: UploadFile = File(...)):
    """
    Đọc nội dung file (txt / json / pdf) để client đưa vào đoạn chat.
    Không lưu file trên server.
    """
    content = ""
    if file.content_type in ("text/plain", "application/json"):
        raw = await file.read()
        content = raw.decode("utf-8", errors="ignore")
    elif file.content_type == "application/pdf" or file.filename.lower().endswith(".pdf"):
        raw = await file.read()
        reader = PdfReader(BytesIO(raw))
        texts = [(page.extract_text() or "") for page in reader.pages]
        content = "\n".join(texts)
    else:
        raise HTTPException(status_code=400, detail=f"Không hỗ trợ loại file: {file.content_type}")

    if len(content) > 8000:
        content = content[:8000] + "\n[...đã cắt bớt nội dung vì quá dài...]"

    return {"filename": file.filename, "content": content}


# ============================================================
# 7. CHAT KIỂU CHATGPT + PHÂN LOẠI T1–T15
# ============================================================

@app.post("/chat/message", response_model=ChatResponse)
async def chat_message(req: ChatRequest):
    if not req.messages:
        raise HTTPException(status_code=400, detail="messages không được rỗng")

    standards_desc = "\n".join(
        f"{code}: {std.name}" for code, std in STANDARDS.items()
    ) or "T1..T15: các chuẩn Toán 9."

    system_prompt = (
        "Bạn là trợ lý Toán AI cho học sinh lớp 9 Vinschool. "
        "Bạn giải thích rõ ràng, ưu tiên giúp học sinh hiểu cách làm. "
        "Sau câu trả lời, hãy tự suy nghĩ xem cuộc hội thoại liên quan đến những chuẩn nào trong T1–T15.\n"
        "Tóm tắt chuẩn:\n" + standards_desc +
        "\nỞ cuối câu trả lời, trên một dòng riêng, hãy viết: TAGS: T1,T3 (ví dụ)."
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

    tags: List[str] = []
    lines = full_reply.splitlines()
    for line in reversed(lines):
        if line.strip().upper().startswith("TAGS:"):
            raw = line.split(":", 1)[1]
            tags = [c.strip().upper() for c in raw.replace(" ", "").split(",") if c.strip()]
            tags = [c for c in tags if c in STANDARDS]
            full_reply = "\n".join(l for l in lines if l != line)
            break

    if not tags:
        # fallback: đoán từ toàn bộ nội dung hội thoại
        all_text = "\n".join(m.content for m in req.messages)
        tags = guess_standards_from_text(all_text)

    return ChatResponse(reply=full_reply.strip(), standards=tags)
