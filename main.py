from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime, timedelta
from io import BytesIO

import random

from openai import OpenAI  # pip install openai
from reportlab.lib.pagesizes import A4  # pip install reportlab
from reportlab.pdfgen import canvas

# ======== CẤU HÌNH CƠ BẢN ========

app = FastAPI(
    title="K9 Math AI Assistant",
    description="Trợ lý Toán AI cho học sinh lớp 9 – không lưu dữ liệu lâu dài.",
    version="1.0.0",
)

# OpenAI client (dùng biến môi trường OPENAI_API_KEY)
client = OpenAI()  # đảm bảo đã set:  setx OPENAI_API_KEY "sk-......"


# ======== MÔ HÌNH DỮ LIỆU ========

class Question(BaseModel):
    id: str
    topic: str
    standard: str  # Chuẩn/kiến thức
    text: str
    options: Optional[List[str]] = None


class _InternalQuestion(Question):
    # Thêm đáp án đúng để dùng nội bộ, không trả cho client
    correct_option: Optional[int] = None


class EntryTestRequest(BaseModel):
    num_questions: int = Field(10, ge=3, le=30)
    topics: Optional[List[str]] = None  # nếu None thì trộn tất cả chủ đề


class UserAnswer(BaseModel):
    question_id: str
    answer_index: Optional[int] = None  # với trắc nghiệm
    free_answer: Optional[str] = None   # nếu sau này có tự luận


class EntryTestSubmission(BaseModel):
    answers: List[UserAnswer]


class TopicResult(BaseModel):
    topic: str
    correct: int
    total: int
    mastery_rate: float  # 0–1


class EntryTestAnalysis(BaseModel):
    topic_results: List[TopicResult]
    global_comment: str
    suggested_topics: List[str]


class StudyPlanRequest(BaseModel):
    topic_results: List[TopicResult]
    days: int = Field(7, ge=1, le=30)
    minutes_per_day: int = Field(30, ge=15, le=120)


class StudyTask(BaseModel):
    day_index: int
    topic: str
    activity_type: str  # "ôn lý thuyết", "làm bài tập", v.v.
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
    detected_topic: str
    needed_knowledge: str
    explanation: str
    step_by_step_hint: str


class SessionReportRequest(BaseModel):
    # Tất cả dữ liệu phiên học đều do client gửi lên,
    # server KHÔNG đọc lại từ bất kỳ nơi nào khác
    nickname: Optional[str] = None
    entry_analysis: EntryTestAnalysis
    study_plan: StudyPlan
    notes: Optional[str] = None


# ======== NGÂN HÀNG CÂU HỎI ĐƠN GIẢN (DEMO) ========

QUESTION_BANK: List[_InternalQuestion] = [
    _InternalQuestion(
        id="Q1",
        topic="Đại số - Phương trình bậc nhất",
        standard="Giải phương trình bậc nhất một ẩn",
        text="Giải phương trình: 2x + 5 = 15. Giá trị của x là:",
        options=["3", "5", "7", "10"],
        correct_option=0,
    ),
    _InternalQuestion(
        id="Q2",
        topic="Đại số - Hệ thức tỉ lệ",
        standard="Tính tỉ lệ và phần trăm",
        text="Một lớp có 40 học sinh, trong đó 60% là nữ. Số học sinh nữ là:",
        options=["18", "20", "22", "24"],
        correct_option=3,
    ),
    _InternalQuestion(
        id="Q3",
        topic="Hình học - Góc",
        standard="Tính số đo góc trong tam giác",
        text="Trong tam giác, hai góc lần lượt là 60° và 45°. Góc còn lại là:",
        options=["45°", "60°", "75°", "90°"],
        correct_option=2,
    ),
    _InternalQuestion(
        id="Q4",
        topic="Hình học - Đường tròn",
        standard="Tính chu vi đường tròn",
        text="Chu vi đường tròn bán kính 5 cm (lấy π ≈ 3,14) là:",
        options=["10π", "25π", "31,4 cm", "15,7 cm"],
        correct_option=2,
    ),
    _InternalQuestion(
        id="Q5",
        topic="Thống kê",
        standard="Tính số trung bình cộng",
        text="Điểm kiểm tra Toán của bạn là 6, 7, 8, 9. Điểm trung bình là:",
        options=["7", "7,5", "8", "8,5"],
        correct_option=1,
    ),
    # Bạn có thể tự bổ sung thêm câu hỏi vào đây
]


def hide_answers(q: _InternalQuestion) -> Question:
    """Chỉ trả câu hỏi, không trả đáp án đúng."""
    return Question(
        id=q.id,
        topic=q.topic,
        standard=q.standard,
        text=q.text,
        options=q.options,
    )


# ======== API CƠ BẢN ========

@app.get("/")
def root():
    return {
        "message": "Math AI Assistant is running",
        "note": "Hệ thống không lưu dữ liệu người dùng lâu dài. "
                "Mọi phân tích chỉ tồn tại trong từng request / phiên.",
    }


# ======== 1. TẠO ĐỀ TEST ĐẦU VÀO / ĐẦU RA ========

@app.post("/entry-test/generate", response_model=List[Question])
def generate_entry_test(req: EntryTestRequest):
    """
    Sinh ngẫu nhiên đề test từ ngân hàng câu hỏi nội bộ.
    Không lưu bài làm hay điểm số.
    """
    # Lọc theo topic nếu người dùng chọn
    if req.topics:
        filtered = [q for q in QUESTION_BANK if q.topic in req.topics]
        if not filtered:
            filtered = QUESTION_BANK[:]
    else:
        filtered = QUESTION_BANK[:]

    num = min(req.num_questions, len(filtered))
    selected = random.sample(filtered, num)

    return [hide_answers(q) for q in selected]


# ======== 2. PHÂN TÍCH KẾT QUẢ TEST (LỖI SAI, ĐIỂM YẾU) ========

@app.post("/entry-test/analyze", response_model=EntryTestAnalysis)
def analyze_entry_test(submission: EntryTestSubmission):
    """
    Nhận danh sách câu trả lời, phân tích theo từng chủ đề.
    Không lưu bất kỳ điểm số nào trên server.
    """
    # Tạo map id -> question
    q_map: Dict[str, _InternalQuestion] = {q.id: q for q in QUESTION_BANK}

    topic_stats: Dict[str, Dict[str, int]] = {}

    for ans in submission.answers:
        q = q_map.get(ans.question_id)
        if not q:
            continue

        topic = q.topic
        if topic not in topic_stats:
            topic_stats[topic] = {"correct": 0, "total": 0}

        topic_stats[topic]["total"] += 1

        is_correct = False
        if q.options is not None and ans.answer_index is not None:
            is_correct = (ans.answer_index == q.correct_option)

        if is_correct:
            topic_stats[topic]["correct"] += 1

    topic_results: List[TopicResult] = []
    weak_topics: List[str] = []

    for topic, stats in topic_stats.items():
        total = stats["total"]
        correct = stats["correct"]
        mastery = correct / total if total > 0 else 0.0

        topic_results.append(
            TopicResult(
                topic=topic,
                correct=correct,
                total=total,
                mastery_rate=round(mastery, 2),
            )
        )

        if mastery < 0.7:  # <70% coi là yếu
            weak_topics.append(topic)

    if not topic_results:
        comment = (
            "Chưa đủ dữ liệu để phân tích. Hãy làm lại bài test với nhiều câu hơn."
        )
    else:
        comment = (
            "Bạn đã hoàn thành bài test. Các chủ đề có tỷ lệ đúng thấp hơn 70% "
            "nên được ưu tiên ôn tập trước."
        )

    return EntryTestAnalysis(
        topic_results=topic_results,
        global_comment=comment,
        suggested_topics=weak_topics,
    )


# ======== 3. TẠO LỘ TRÌNH HỌC / LỊCH HỌC TRONG PHIÊN ========

@app.post("/study-plan", response_model=StudyPlan)
def create_study_plan(req: StudyPlanRequest):
    """
    Tạo lộ trình học theo nguyên lý lặp lại ngắt quãng trong phạm vi 1 phiên.
    Client có thể xuất ra PDF và tự lưu, server không lưu.
    """
    # Sắp xếp chủ đề từ yếu -> mạnh
    sorted_topics = sorted(req.topic_results, key=lambda t: t.mastery_rate)

    start = datetime.now()
    tasks: List[StudyTask] = []

    if not sorted_topics:
        # Không có dữ liệu, tạo 1 nhiệm vụ mặc định
        tasks.append(
            StudyTask(
                day_index=0,
                topic="Tổng quan",
                activity_type="định hướng",
                description="Ôn lại các khái niệm cơ bản của Toán 9.",
                estimated_minutes=req.minutes_per_day,
            )
        )
    else:
        # Lặp lại các topic yếu nhiều hơn
        day_idx = 0
        for day in range(req.days):
            topic_info = sorted_topics[day % len(sorted_topics)]
            topic = topic_info.topic

            # Chia nhỏ thời gian trong ngày: 50% bài tập, 30% lý thuyết, 20% ôn nhanh
            total = req.minutes_per_day
            tasks.append(
                StudyTask(
                    day_index=day,
                    topic=topic,
                    activity_type="ôn lý thuyết",
                    description=f"Đọc lại lý thuyết và ví dụ mẫu về chủ đề: {topic}.",
                    estimated_minutes=int(total * 0.3),
                )
            )
            tasks.append(
                StudyTask(
                    day_index=day,
                    topic=topic,
                    activity_type="làm bài tập",
                    description=f"Làm 4–6 bài tập cơ bản và 1–2 bài nâng cao về {topic}.",
                    estimated_minutes=int(total * 0.5),
                )
            )
            tasks.append(
                StudyTask(
                    day_index=day,
                    topic=topic,
                    activity_type="ôn nhanh (spaced repetition)",
                    description=f"Làm lại 2–3 câu mà trước đó bạn đã sai ở chủ đề {topic}.",
                    estimated_minutes=total - int(total * 0.3) - int(total * 0.5),
                )
            )
            day_idx += 1

    return StudyPlan(
        start_date=start,
        tasks=tasks,
    )


# ======== 4. PHÂN TÍCH & GIẢI THÍCH BÀI TẬP BẰNG AI ========

async def call_openai_for_exercise(req: ExerciseAnalysisRequest, topic_guess: str) -> ExerciseAnalysisResponse:
    """
    Gọi OpenAI để giải thích bài toán. Tất cả dữ liệu chỉ dùng trong request này,
    không lưu ở đâu khác.
    """
    lang = req.language.lower()

    if lang == "en":
        system_prompt = (
            "You are a Grade 9 math tutor for Vietnamese students. "
            "Explain concepts clearly, step by step, in simple English."
        )
        user_prompt = f"""
Problem:
{req.problem_text}

Student's answer (may be empty):
{req.student_answer or "(no answer)"}

Detected topic: {topic_guess}.

1. Briefly explain what knowledge is needed to solve this.
2. Give a step-by-step solution.
3. Then rewrite a shorter 'hint-only' version (no final answer), so the student can try again.
"""
    else:
        # Mặc định tiếng Việt
        system_prompt = (
            "Bạn là trợ lý Toán lớp 9 cho học sinh Vinschool. "
            "Bạn giải thích ngắn gọn, dễ hiểu, ưu tiên giúp học sinh hiểu bản chất."
        )
        user_prompt = f"""
Bài toán:
{req.problem_text}

Bài làm (nếu có) của học sinh:
{req.student_answer or "(học sinh chưa làm)"}

Chủ đề dự đoán: {topic_guess}.

Yêu cầu:
1. Giải thích ngắn gọn kiến thức cần có để làm bài này.
2. Trình bày lời giải từng bước, rõ ràng.
3. Viết lại một phiên bản chỉ gợi ý (không ghi đáp số), để học sinh có thể tự làm lại.
"""

    completion = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.4,
    )

    content = completion.choices[0].message.content

    # Đơn giản: tách 3 phần theo số thứ tự (nếu không tách được thì dùng chung)
    needed_knowledge = content
    explanation = content
    hint = content

    # Bạn có thể parse thông minh hơn sau; demo tách thô bằng split
    parts = content.split("3.")
    if len(parts) == 2:
        explanation_part = parts[0]
        hint = "3." + parts[1]
        # tiếp tục tách 1. / 2.
        subparts = explanation_part.split("2.")
        if len(subparts) == 2:
            needed_knowledge = subparts[0]
            explanation = "2." + subparts[1]

    return ExerciseAnalysisResponse(
        detected_topic=topic_guess,
        needed_knowledge=needed_knowledge.strip(),
        explanation=explanation.strip(),
        step_by_step_hint=hint.strip(),
    )


def simple_topic_guess(text: str) -> str:
    """Đoán chủ đề rất thô sơ dựa vào từ khóa."""
    t = text.lower()
    if "phương trình" in t or "giải phương trình" in t or "x +" in t:
        return "Đại số - Phương trình"
    if "tam giác" in t or "góc" in t:
        return "Hình học - Tam giác & góc"
    if "đường tròn" in t or "bán kính" in t or "chu vi" in t:
        return "Hình học - Đường tròn"
    if "%" in t or "phần trăm" in t:
        return "Đại số - Tỉ lệ & phần trăm"
    if "trung bình" in t:
        return "Thống kê - Trung bình cộng"
    return "Tổng quan Toán 9"


@app.post("/exercise/analyze", response_model=ExerciseAnalysisResponse)
async def analyze_exercise(req: ExerciseAnalysisRequest):
    """
    Phân tích 1 bài toán: đoán chủ đề + nhờ OpenAI giải thích.
    Server chỉ dùng nội dung bài toán trong request hiện tại, không lưu file/bài làm.
    """
    topic_guess = simple_topic_guess(req.problem_text)
    return await call_openai_for_exercise(req, topic_guess)


# ======== 5. XUẤT BÁO CÁO PDF (TRONG PHIÊN) ========

def build_pdf_bytes(report: SessionReportRequest) -> bytes:
    """
    Tạo PDF hoàn toàn trong bộ nhớ. Không ghi ra đĩa, không lưu lại.
    """
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    y = height - 50

    title = "BÁO CÁO PHIÊN HỌC TOÁN AI"
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, y, title)
    y -= 30

    c.setFont("Helvetica", 11)
    nickname = report.nickname or "Học sinh"
    c.drawString(50, y, f"Học sinh: {nickname}")
    y -= 20
    c.drawString(50, y, f"Thời gian tạo báo cáo: {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    y -= 30

    # Kết quả theo chủ đề
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "1. Kết quả bài test theo chủ đề:")
    y -= 20
    c.setFont("Helvetica", 11)

    for tr in report.entry_analysis.topic_results:
        line = f"- {tr.topic}: {tr.correct}/{tr.total} câu đúng (≈ {int(tr.mastery_rate*100)}%)"
        c.drawString(60, y, line)
        y -= 15
        if y < 80:
            c.showPage()
            y = height - 50

    y -= 10
    c.drawString(50, y, "Nhận xét tổng quát:")
    y -= 15
    for line in report.entry_analysis.global_comment.split("\n"):
        c.drawString(60, y, line)
        y -= 15
        if y < 80:
            c.showPage()
            y = height - 50

    # Lộ trình học
    y -= 10
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "2. Lộ trình học gợi ý:")
    y -= 20
    c.setFont("Helvetica", 11)

    for task in report.study_plan.tasks:
        day_str = (report.study_plan.start_date + timedelta(days=task.day_index)).strftime("%d/%m")
        text = f"Ngày {day_str} - {task.topic} - {task.activity_type} (~{task.estimated_minutes} phút)"
        c.drawString(60, y, text)
        y -= 15
        if y < 80:
            c.showPage()
            y = height - 50

    # Ghi chú thêm
    if report.notes:
        y -= 10
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, y, "3. Ghi chú thêm:")
        y -= 20
        c.setFont("Helvetica", 11)
        for line in report.notes.split("\n"):
            c.drawString(60, y, line)
            y -= 15
            if y < 80:
                c.showPage()
                y = height - 50

    c.showPage()
    c.save()

    buffer.seek(0)
    return buffer.read()


@app.post("/session/report/pdf")
def export_session_report(report: SessionReportRequest):
    """
    Tạo file PDF báo cáo **chỉ dựa trên dữ liệu client gửi lên**.
    Server không lưu bản sao nào sau khi trả về.
    """
    pdf_bytes = build_pdf_bytes(report)
    buffer = BytesIO(pdf_bytes)

    return StreamingResponse(
        buffer,
        media_type="application/pdf",
        headers={
            "Content-Disposition": 'attachment; filename="math_ai_session_report.pdf"'
        },
    )
