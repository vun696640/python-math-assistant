import os
from typing import List, Dict, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from pydantic import BaseModel
from pypdf import PdfReader
from openai import OpenAI


# ====== CẤU HÌNH CƠ BẢN ======
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHUAN_DAU_RA_FILE = os.path.join(BASE_DIR, "chuan_dau_ra")

app = FastAPI(title="K9 Math AI Assistant")

# Cho phép gọi từ web (frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # nếu muốn chặt hơn thì set đúng domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OpenAI client (đọc key từ env OPENAI_API_KEY)
client = OpenAI()


# ====== MODEL DỮ LIỆU ======
class ChatMessage(BaseModel):
    role: str  # "user" hoặc "assistant"
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


# ====== ĐỌC FILE CHUẨN ĐẦU RA (T1–T15) ======
def parse_chuan_dau_ra_md(path: str) -> Dict[str, Standard]:
    """
    Parse file Markdown 'chuan_dau_ra' thành dict { 'T1': Standard(...), ... }.
    Định dạng mong đợi giống file trong repo, ví dụ:

        ## T1 – Số và Lũy Thừa (Đại số)
        #### 1. Kiến thức (Normal)
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


try:
    STANDARDS: Dict[str, Standard] = parse_chuan_dau_ra_md(CHUAN_DAU_RA_FILE)
except Exception as e:
    print("Lỗi load chuẩn đầu ra:", e)
    STANDARDS = {}


# ====== HÀM PHÂN LOẠI CHUẨN T1–T15 ĐƠN GIẢN ======
def detect_standards_from_text(text: str) -> List[str]:
    """
    Phân loại cực đơn giản: dò keyword → list mã chuẩn (T1..T15).
    Sau này muốn thông minh hơn có thể gọi OpenAI riêng để gán chuẩn.
    """
    t = text.lower()
    found: List[str] = []

    # Ví dụ mapping đơn giản
    if any(k in t for k in ["lũy thừa", "số mũ", "ước chung", "uoc chung", "ucln", "bcnn"]):
        found.append("T1")
    if any(k in t for k in ["phần trăm", "%", "tỷ lệ", "ti le", "tỉ lệ"]):
        found.append("T2")
    if any(k in t for k in ["góc", "tam giác", "tam giac", "chứng minh góc", "chung minh goc"]):
        found.append("T3")
    # Nếu không tìm thấy gì → tạm gán T1 để luôn có ít nhất 1 chuẩn
    if not found:
        found.append("T1")

    # Chỉ giữ những chuẩn thực sự tồn tại trong file
    found = [code for code in found if code in STANDARDS]
    return list(dict.fromkeys(found))  # loại trùng


# ====== API: KIỂM TRA SERVER ======
@app.get("/api/health")
def health():
    return {
        "message": "Math AI Assistant is running",
        "standards_loaded": list(STANDARDS.keys()),
        "note": "Không lưu thông tin cá nhân hay bài làm trên server.",
    }


# ====== API: CHAT /chat/message ======
@app.post("/chat/message", response_model=ChatResponse)
async def chat_message(req: ChatRequest):
    if not os.environ.get("OPENAI_API_KEY"):
        raise HTTPException(
            status_code=500,
            detail="OPENAI_API_KEY chưa được cấu hình trên server (Render).",
        )

    # Ghép lịch sử chat thành dạng cho OpenAI
    messages_for_model = []

    # System message: giải thích vai trò & đưa chuẩn đầu ra vào context
    standards_text = "\n".join(
        f"{s.code}: {s.name}\n{s.description}\n"
        for s in STANDARDS.values()
    ) or "Chưa load được chuẩn đầu ra."

    system_prompt = (
        "Bạn là trợ lý Toán 9 bằng tiếng Việt cho dự án AI For Good.\n"
        "Nhiệm vụ:\n"
        "- Giải thích chi tiết, từng bước, phù hợp học sinh lớp 9.\n"
        "- Chỉ hỗ trợ kiến thức trong chương trình Toán 9.\n"
        "- Luôn phân tích cẩn thận, tránh chỉ đưa đáp án.\n"
        "- Bám sát các chuẩn đầu ra T1–T15 dưới đây:\n\n"
        f"{standards_text}\n\n"
        "Sau khi trả lời, hệ thống backend sẽ tự gán T1–T15, nên bạn không cần ghi mã chuẩn trong câu trả lời."
    )

    messages_for_model.append({"role": "system", "content": system_prompt})

    for m in req.messages:
        role = "assistant" if m.role == "assistant" else "user"
        messages_for_model.append({"role": role, "content": m.content})

    try:
        completion = client.chat.completions.create(
            model="gpt-4.1-mini",  # có thể đổi sang model khác nếu muốn
            messages=messages_for_model,
        )
        reply_text = completion.choices[0].message.content.strip()
    except Exception as e:
        print("OpenAI error:", e)
        raise HTTPException(status_code=500, detail=f"Lỗi gọi OpenAI: {e}")

    # Lấy chuẩn đầu ra từ câu hỏi cuối cùng của học sinh
    # (nếu muốn tinh hơn có thể dùng cả lịch sử)
    last_user_msg = ""
    for m in reversed(req.messages):
        if m.role == "user":
            last_user_msg = m.content
            break

    detected = detect_standards_from_text(last_user_msg)

    return ChatResponse(reply=reply_text, standards=detected)


# ====== API: ĐỌC FILE /file/parse ======
@app.post("/file/parse")
async def parse_file(file: UploadFile = File(...)):
    """
    Nhận file bài tập (PDF/TXT/MD/JSON), trích nội dung text để đưa vào chat.
    """
    filename = file.filename
    try:
        if filename.lower().endswith(".pdf"):
            reader = PdfReader(file.file)
            texts = [page.extract_text() or "" for page in reader.pages]
            content = "\n\n".join(texts)
        else:
            # Các loại file text đơn giản
            content_bytes = await file.read()
            try:
                content = content_bytes.decode("utf-8")
            except UnicodeDecodeError:
                content = content_bytes.decode("latin-1", errors="ignore")

        # Giới hạn độ dài để tránh gửi quá nhiều lên model
        if len(content) > 8000:
            content = content[:8000]

        return {"filename": filename, "content": content}
    except Exception as e:
        print("Error parsing file:", e)
        raise HTTPException(status_code=400, detail=f"Không đọc được file: {e}")


# ====== SERVE FRONTEND (index.html) ======
# Đảm bảo index.html nằm cùng thư mục với main.py
app.mount("/", StaticFiles(directory=".", html=True), name="static")
