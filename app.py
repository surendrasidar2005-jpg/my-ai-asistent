"""
MyAssistant v2 — Production Grade AI Assistant
- Groq API: llama-3.3-70b-versatile (best free model)
- Streaming responses
- PDF analysis, Web search, File manager
- Render.com / Railway ready
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
import httpx, os, json, time, asyncio
from pathlib import Path
from collections import defaultdict

# ── Optional deps ───────────────────────────────────────────────────────────
try:
    import pdfplumber
    PDF_OK = True
except ImportError:
    PDF_OK = False

try:
    from duckduckgo_search import DDGS
    SEARCH_OK = True
except ImportError:
    SEARCH_OK = False

# ── App ─────────────────────────────────────────────────────────────────────
app = FastAPI(title="MyAssistant v2", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# ── Config ──────────────────────────────────────────────────────────────────
WORKSPACE    = Path("workspace"); WORKSPACE.mkdir(exist_ok=True)
GROQ_KEY     = os.environ.get("GROQ_API_KEY", "")
ACCESS_PASS  = os.environ.get("ACCESS_PASSWORD", "")
PORT         = int(os.environ.get("PORT", 8000))

# Model priority list — falls back automatically
GROQ_MODELS = [
    "llama-3.3-70b-versatile",   # Best quality, free
    "llama-3.1-70b-versatile",   # Fallback
    "llama-3.1-8b-instant",      # Fast fallback
]
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

# ── Rate limiting (simple in-memory) ────────────────────────────────────────
rate_store: dict = defaultdict(list)

def rate_limit_ok(ip: str, max_req=30, window=60) -> bool:
    now = time.time()
    rate_store[ip] = [t for t in rate_store[ip] if now - t < window]
    if len(rate_store[ip]) >= max_req:
        return False
    rate_store[ip].append(now)
    return True

# ── System Prompt (carefully engineered) ────────────────────────────────────
SYSTEM_PROMPT = """You are MyAssistant, an expert AI assistant for students and professionals, especially those in technical fields like mining engineering.

## Your Capabilities
- Answer questions deeply and accurately in Hindi or English (match user's language)
- Analyze documents and PDFs with precision
- Help with research, calculations, writing, and problem-solving
- Explain complex topics clearly with examples

## Response Quality Rules
1. Always give thorough, accurate, well-structured answers
2. Use markdown formatting: **bold**, bullet points, code blocks, tables
3. For calculations or technical topics, show step-by-step reasoning
4. If you don't know something, say so honestly
5. For Hindi questions, reply in Hindi; for English, reply in English; for mixed, use mixed

## ABSOLUTE SAFETY RULES (NEVER VIOLATE)
1. NEVER assist with payment systems, UPI, banking, financial transfers, or transaction-related tasks
2. NEVER access, modify, or reference files outside the workspace directory
3. NEVER execute arbitrary code or system commands
4. NEVER share API keys, passwords, or sensitive credentials
5. NEVER generate harmful, illegal, or unethical content

## Context Handling
- When a PDF/document context is provided, analyze it carefully and answer based on its content
- Always cite which part of the document your answer comes from when analyzing PDFs"""

# ── Safety system ────────────────────────────────────────────────────────────
BLOCKED = [
    "upi","paytm","phonepe","gpay","google pay","amazon pay","bhim",
    "credit card","debit card","card number","account number","ifsc","cvv","pin",
    "bank transfer","send money","neft","rtgs","imps","net banking",
    "otp for payment","wallet transfer","razorpay","stripe","paypal transfer"
]

def is_safe(text: str) -> bool:
    t = text.lower()
    return not any(k in t for k in BLOCKED)

def auth_ok(pw: str) -> bool:
    return (not ACCESS_PASS) or (pw == ACCESS_PASS)

# ── Schemas ──────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    history: list = []
    context: str = ""
    password: str = ""
    stream: bool = True

    @validator("message")
    def msg_not_empty(cls, v):
        if not v.strip():
            raise ValueError("Message empty hai")
        if len(v) > 8000:
            raise ValueError("Message bahut lamba hai (max 8000 chars)")
        return v.strip()

class FileRequest(BaseModel):
    filename: str
    content: str = ""
    action: str  # read | write | delete

class SearchRequest(BaseModel):
    query: str
    password: str = ""

# ── Groq streaming call ──────────────────────────────────────────────────────
async def groq_stream(messages: list, model_idx=0):
    """Generator that yields SSE chunks from Groq"""
    model = GROQ_MODELS[min(model_idx, len(GROQ_MODELS)-1)]
    async with httpx.AsyncClient(timeout=90) as client:
        async with client.stream(
            "POST", GROQ_URL,
            headers={"Authorization": f"Bearer {GROQ_KEY}", "Content-Type": "application/json"},
            json={"model": model, "messages": messages,
                  "temperature": 0.65, "max_tokens": 2048, "stream": True}
        ) as resp:
            if resp.status_code == 429 and model_idx < len(GROQ_MODELS)-1:
                async for chunk in groq_stream(messages, model_idx+1):
                    yield chunk
                return
            if resp.status_code != 200:
                body = await resp.aread()
                yield f"data: {json.dumps({'error': f'Groq error {resp.status_code}'})}\n\n"
                return

            async for line in resp.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        yield "data: [DONE]\n\n"
                        return
                    try:
                        chunk = json.loads(data)
                        delta = chunk["choices"][0]["delta"].get("content", "")
                        if delta:
                            yield f"data: {json.dumps({'token': delta})}\n\n"
                    except Exception:
                        continue

async def groq_sync(messages: list) -> str:
    """Non-streaming call for internal use"""
    async with httpx.AsyncClient(timeout=60) as client:
        for model in GROQ_MODELS:
            r = await client.post(
                GROQ_URL,
                headers={"Authorization": f"Bearer {GROQ_KEY}", "Content-Type": "application/json"},
                json={"model": model, "messages": messages, "temperature": 0.65, "max_tokens": 2048}
            )
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"]
            if r.status_code != 429:
                raise HTTPException(500, f"Groq error: {r.text[:200]}")
    raise HTTPException(503, "Groq quota exceeded. Thodi der baad try karo.")

# ── Routes ───────────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return HTMLResponse(open("static/index.html", encoding="utf-8").read())

@app.get("/status")
async def status():
    return {
        "ok": bool(GROQ_KEY),
        "model": GROQ_MODELS[0],
        "all_models": GROQ_MODELS,
        "pdf": PDF_OK,
        "search": SEARCH_OK,
        "password_protected": bool(ACCESS_PASS),
        "version": "2.0.0"
    }

@app.post("/chat")
async def chat(req: ChatRequest, request: Request):
    ip = request.client.host
    if not rate_limit_ok(ip):
        raise HTTPException(429, "Bahut zyada requests! Ek minute baad try karo.")
    if not auth_ok(req.password):
        raise HTTPException(401, "Password galat hai!")
    if not GROQ_KEY:
        raise HTTPException(503, "GROQ_API_KEY environment variable set nahi hai!")
    if not is_safe(req.message):
        raise HTTPException(403, "⚠️ Payment/banking related requests blocked hain!")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Inject PDF context smartly
    if req.context.strip():
        messages.append({
            "role": "system",
            "content": f"## Document Context (User ne ye document upload kiya hai):\n\n{req.context[:4000]}\n\nIs document ke baare mein accurately answer karo."
        })

    # Last 12 messages history
    for h in req.history[-12:]:
        if h.get("role") in ("user", "assistant") and h.get("content"):
            messages.append({"role": h["role"], "content": h["content"]})

    messages.append({"role": "user", "content": req.message})

    if req.stream:
        return StreamingResponse(
            groq_stream(messages),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
        )
    else:
        reply = await groq_sync(messages)
        return {"response": reply}

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    if not PDF_OK:
        raise HTTPException(501, "Server pe pdfplumber install nahi hai")
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Sirf .pdf files allowed hain")
    if file.size and file.size > 20 * 1024 * 1024:
        raise HTTPException(413, "PDF 20MB se chhota hona chahiye")

    raw = await file.read()
    safe = Path(file.filename).name
    path = WORKSPACE / safe
    path.write_bytes(raw)

    pages_text = []
    try:
        with pdfplumber.open(path) as pdf:
            total = len(pdf.pages)
            for i, page in enumerate(pdf.pages[:25]):
                t = page.extract_text()
                if t and t.strip():
                    pages_text.append(f"[Page {i+1}]\n{t.strip()}")
    except Exception as e:
        raise HTTPException(500, f"PDF parse error: {e}")

    if not pages_text:
        raise HTTPException(422, "PDF se text extract nahi hua. Scanned PDF ho sakta hai.")

    full = "\n\n".join(pages_text)
    return {
        "filename": safe,
        "total_pages": total,
        "extracted_pages": len(pages_text),
        "text": full[:10000],
        "preview": full[:400],
        "word_count": len(full.split())
    }

@app.post("/search")
async def search(req: SearchRequest):
    if not auth_ok(req.password):
        raise HTTPException(401, "Password galat hai!")
    if not SEARCH_OK:
        raise HTTPException(501, "duckduckgo-search install nahi hai")
    if not is_safe(req.query):
        raise HTTPException(403, "Query blocked hai")
    if not req.query.strip():
        raise HTTPException(400, "Query empty hai")

    try:
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(req.query.strip(), max_results=6):
                results.append({
                    "title":   r.get("title", ""),
                    "snippet": r.get("body", ""),
                    "url":     r.get("href", "")
                })
        return {"results": results, "query": req.query}
    except Exception as e:
        raise HTTPException(500, f"Search failed: {e}")

@app.get("/files")
async def list_files():
    exts = {".txt":"📝",".md":"📋",".py":"🐍",".json":"📦",
            ".csv":"📊",".pdf":"📄",".html":"🌐",".js":"⚡"}
    files = []
    for f in sorted(WORKSPACE.iterdir()):
        if f.is_file():
            files.append({
                "name": f.name, "size": f.stat().st_size,
                "ext": f.suffix, "icon": exts.get(f.suffix, "📄"),
                "modified": int(f.stat().st_mtime)
            })
    return {"files": files}

@app.post("/file")
async def handle_file(req: FileRequest):
    safe = Path(req.filename).name          # strip any path traversal
    path = WORKSPACE / safe

    if req.action == "read":
        if not path.exists():
            raise HTTPException(404, f"'{safe}' nahi mila")
        return {"filename": safe, "content": path.read_text(encoding="utf-8", errors="replace")}

    elif req.action == "write":
        if not safe:
            raise HTTPException(400, "Filename empty hai")
        if not is_safe(req.content):
            raise HTTPException(403, "Content mein restricted keywords hain")
        path.write_text(req.content, encoding="utf-8")
        return {"ok": True, "message": f"✅ '{safe}' save ho gayi!"}

    elif req.action == "delete":
        if path.exists():
            path.unlink()
        return {"ok": True, "message": f"🗑️ '{safe}' delete ho gayi"}

    raise HTTPException(400, "Action: 'read' | 'write' | 'delete'")

# ── Static + Boot ────────────────────────────────────────────────────────────
app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=PORT, reload=False)
