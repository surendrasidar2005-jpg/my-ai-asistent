"""
NanoBot AI Assistant
====================
True agentic AI - ReAct pattern (no Groq tool API - 100% reliable)
One command → thinks → acts → reports done
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx, os, json, time, asyncio, math, re
from pathlib import Path
from collections import defaultdict

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

# ── Config ───────────────────────────────────────────────────────────────────
app = FastAPI(title="NanoBot")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

WORKSPACE   = Path("workspace"); WORKSPACE.mkdir(exist_ok=True)
GROQ_KEY    = os.environ.get("GROQ_API_KEY", "")
ACCESS_PASS = os.environ.get("ACCESS_PASSWORD", "")
PORT        = int(os.environ.get("PORT", 8000))
GROQ_URL    = "https://api.groq.com/openai/v1/chat/completions"
MODELS      = [
    "llama-3.3-70b-versatile",
    "llama-3.1-70b-versatile",
    "llama-3.1-8b-instant",
]

# ── Safety ───────────────────────────────────────────────────────────────────
BLOCKED = [
    "upi","paytm","phonepe","gpay","google pay","amazon pay","bhim",
    "credit card","debit card","card number","account number","ifsc","cvv",
    "bank transfer","send money","neft","rtgs","imps","net banking",
    "otp","wallet transfer","razorpay","stripe","paypal","payment gateway",
    "bank account","financial transfer","wire transfer"
]

def is_safe(text: str) -> bool:
    t = text.lower()
    return not any(k in t for k in BLOCKED)

def auth_ok(pw: str) -> bool:
    return (not ACCESS_PASS) or (pw == ACCESS_PASS)

def get_ip(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for")
    if fwd: return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"

# ── Rate limit ────────────────────────────────────────────────────────────────
rate_store: dict = defaultdict(list)
def rate_ok(ip: str) -> bool:
    now = time.time()
    rate_store[ip] = [t for t in rate_store[ip] if now - t < 60]
    if len(rate_store[ip]) >= 25: return False
    rate_store[ip].append(now)
    return True

# ── Tool Definitions ──────────────────────────────────────────────────────────
TOOLS_DOC = """
Available tools (use EXACTLY this JSON format):

{"action": "web_search", "args": {"query": "search query here"}}
{"action": "write_file", "args": {"filename": "name.txt", "content": "file content here"}}
{"action": "read_file", "args": {"filename": "name.txt"}}
{"action": "append_file", "args": {"filename": "name.txt", "content": "content to add"}}
{"action": "list_files", "args": {}}
{"action": "delete_file", "args": {"filename": "name.txt"}}
{"action": "calculator", "args": {"expression": "2 * sqrt(144)"}}
{"action": "summarize_pdf", "args": {"filename": "uploaded.pdf"}}
{"action": "done", "args": {"message": "Final answer or summary of what was accomplished"}}
"""

SYSTEM_PROMPT = f"""You are NanoBot, a powerful AI assistant that autonomously completes tasks.

{TOOLS_DOC}

## HOW TO WORK (VERY IMPORTANT):
You work in a loop. Each response must be EITHER:

1. A tool call in this EXACT format (one JSON per line, nothing else):
THOUGHT: [your reasoning about what to do next]
ACTION: {{"action": "tool_name", "args": {{...}}}}

2. OR a final answer when task is done:
THOUGHT: Task is complete
ACTION: {{"action": "done", "args": {{"message": "..."}}}}

## RULES:
- ALWAYS use THOUGHT: before ACTION:
- NEVER skip steps - do them one by one
- For multi-step tasks: search THEN save THEN confirm
- Be thorough - if user says search and save, do BOTH
- Respond in the SAME language as user (Hindi/English)
- After completing ALL steps, use "done" action with full summary
- NEVER fabricate search results - actually use web_search tool

## PAYMENT SAFETY:
NEVER help with UPI, banking, payments, or financial transfers.

## EXAMPLES:
User: "Search python tutorial and save to python.txt"
THOUGHT: I need to search for python tutorial, then save results to file
ACTION: {{"action": "web_search", "args": {{"query": "python tutorial for beginners"}}}}

[After getting results]
THOUGHT: Got search results, now saving to file
ACTION: {{"action": "write_file", "args": {{"filename": "python.txt", "content": "Python Tutorial\\n..."}}}}

[After saving]
THOUGHT: Both tasks done - searched and saved
ACTION: {{"action": "done", "args": {{"message": "✅ Done! Searched Python tutorial and saved to python.txt"}}}}
"""

# ── Tool Executor ─────────────────────────────────────────────────────────────
async def run_tool(action: str, args: dict) -> str:
    try:
        if action == "web_search":
            query = args.get("query", "")
            if not is_safe(query):
                return "❌ Blocked: Payment related search not allowed."
            if not SEARCH_OK:
                return "❌ Search not available."
            results = []
            try:
                ddgs = DDGS()
                for r in ddgs.text(query, max_results=6):
                    results.append(
                        f"**{r.get('title','')}**\n"
                        f"{r.get('body','')}\n"
                        f"URL: {r.get('href','')}"
                    )
            except Exception as e:
                return f"Search error: {e}"
            return "\n\n".join(results) if results else "No results found."

        elif action == "write_file":
            fname   = Path(args["filename"]).name
            content = args.get("content", "")
            if not is_safe(content):
                return "❌ Blocked: Payment related content."
            (WORKSPACE / fname).write_text(content, encoding="utf-8")
            return f"✅ File '{fname}' saved ({len(content)} chars, {len(content.splitlines())} lines)"

        elif action == "append_file":
            fname   = Path(args["filename"]).name
            content = args.get("content", "")
            path    = WORKSPACE / fname
            existing = path.read_text(encoding="utf-8") if path.exists() else ""
            path.write_text(existing + "\n" + content, encoding="utf-8")
            return f"✅ Appended to '{fname}'"

        elif action == "read_file":
            fname = Path(args["filename"]).name
            path  = WORKSPACE / fname
            if not path.exists():
                files = [f.name for f in WORKSPACE.iterdir() if f.is_file()]
                return f"❌ File '{fname}' not found. Available: {files}"
            content = path.read_text(encoding="utf-8", errors="replace")
            return f"File '{fname}' content:\n\n{content}"

        elif action == "list_files":
            files = sorted([f.name for f in WORKSPACE.iterdir() if f.is_file()])
            if not files:
                return "No files in workspace yet."
            sizes = {f.name: f.stat().st_size for f in WORKSPACE.iterdir() if f.is_file()}
            return "Files:\n" + "\n".join(f"- {f} ({sizes[f]} bytes)" for f in files)

        elif action == "delete_file":
            fname = Path(args["filename"]).name
            path  = WORKSPACE / fname
            if path.exists():
                path.unlink()
                return f"✅ '{fname}' deleted"
            return f"❌ File '{fname}' not found"

        elif action == "calculator":
            expr = args.get("expression", "")
            safe_ns = {
                "__builtins__": {},
                "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos,
                "tan": math.tan,   "log": math.log, "log10": math.log10,
                "pi": math.pi,     "e": math.e,     "abs": abs,
                "round": round,    "pow": pow,       "floor": math.floor,
                "ceil": math.ceil, "exp": math.exp,  "degrees": math.degrees,
                "radians": math.radians, "factorial": math.factorial,
                "hypot": math.hypot, "asin": math.asin, "acos": math.acos,
                "atan": math.atan, "atan2": math.atan2,
            }
            result = eval(expr, safe_ns)
            return f"🧮 {expr} = **{result}**"

        elif action == "summarize_pdf":
            fname = Path(args.get("filename", "")).name
            path  = WORKSPACE / fname
            if not path.exists():
                return f"❌ PDF '{fname}' not found. Upload it first."
            if not PDF_OK:
                return "❌ PDF support not available."
            pages = []
            with pdfplumber.open(path) as pdf:
                for i, pg in enumerate(pdf.pages[:20]):
                    t = pg.extract_text()
                    if t: pages.append(f"[Page {i+1}]\n{t.strip()}")
            text = "\n\n".join(pages)
            return f"PDF '{fname}' content:\n\n{text[:6000]}"

        elif action == "done":
            return args.get("message", "Task completed.")

        else:
            return f"❌ Unknown tool: {action}"

    except Exception as e:
        return f"❌ Tool error: {e}"

# ── Parse model output ────────────────────────────────────────────────────────
def parse_action(text: str):
    """Extract THOUGHT and ACTION from model response."""
    thought = ""
    action_data = None

    # Extract THOUGHT
    thought_match = re.search(r'THOUGHT:\s*(.+?)(?=ACTION:|$)', text, re.DOTALL | re.IGNORECASE)
    if thought_match:
        thought = thought_match.group(1).strip()

    # Extract ACTION JSON
    action_match = re.search(r'ACTION:\s*(\{.+?\})', text, re.DOTALL | re.IGNORECASE)
    if action_match:
        try:
            action_data = json.loads(action_match.group(1))
        except json.JSONDecodeError:
            # Try to find any JSON in the text
            json_match = re.search(r'\{[^{}]*"action"[^{}]*\}', text)
            if json_match:
                try:
                    action_data = json.loads(json_match.group())
                except:
                    pass

    return thought, action_data

# ── Groq API call ─────────────────────────────────────────────────────────────
async def call_groq(client: httpx.AsyncClient, messages: list, model: str) -> str:
    for m in ([model] + [x for x in MODELS if x != model]):
        try:
            resp = await client.post(
                GROQ_URL,
                headers={"Authorization": f"Bearer {GROQ_KEY}", "Content-Type": "application/json"},
                json={"model": m, "messages": messages, "temperature": 0.3, "max_tokens": 1500},
                timeout=60
            )
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"]
        except Exception:
            continue
    return None

# ── Main Agent Stream ─────────────────────────────────────────────────────────
async def agent_stream(user_msg: str, history: list):
    MAX_STEPS = 10
    messages  = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Add history
    for h in history[-8:]:
        if h.get("role") in ("user", "assistant") and h.get("content"):
            messages.append({"role": h["role"], "content": str(h["content"])})

    messages.append({"role": "user", "content": user_msg})

    async with httpx.AsyncClient() as client:
        for step in range(MAX_STEPS):
            # Call model
            yield f"data: {json.dumps({'type':'thinking', 'text':'🤔 Soch raha hoon...'})}\n\n"

            response = await call_groq(client, messages, MODELS[0])
            if not response:
                yield f"data: {json.dumps({'type':'error', 'text':'❌ Groq se connect nahi hua. API key check karo.'})}\n\n"
                yield "data: [DONE]\n\n"
                return

            thought, action_data = parse_action(response)

            # Show thought
            if thought:
                yield f"data: {json.dumps({'type':'thought', 'text':f'💭 {thought}'})}\n\n"
                await asyncio.sleep(0.1)

            # No action found — treat as final answer
            if not action_data:
                words = response.split(" ")
                for i, w in enumerate(words):
                    yield f"data: {json.dumps({'type':'token', 'text':w+(' ' if i<len(words)-1 else '')})}\n\n"
                    await asyncio.sleep(0.01)
                yield "data: [DONE]\n\n"
                return

            action = action_data.get("action", "")
            args   = action_data.get("args", {})

            # Done action
            if action == "done":
                final = args.get("message", "✅ Kaam ho gaya!")
                words = final.split(" ")
                for i, w in enumerate(words):
                    yield f"data: {json.dumps({'type':'token', 'text':w+(' ' if i<len(words)-1 else '')})}\n\n"
                    await asyncio.sleep(0.01)
                yield "data: [DONE]\n\n"
                return

            # Execute tool
            tool_names = {
                "web_search":   f"🔍 Searching: {args.get('query','...')}",
                "write_file":   f"💾 Writing file: {args.get('filename','...')}",
                "read_file":    f"📖 Reading: {args.get('filename','...')}",
                "append_file":  f"✏️ Appending to: {args.get('filename','...')}",
                "list_files":   "📂 Listing files...",
                "delete_file":  f"🗑️ Deleting: {args.get('filename','...')}",
                "calculator":   f"🧮 Calculating: {args.get('expression','...')}",
                "summarize_pdf":f"📄 Reading PDF: {args.get('filename','...')}",
            }
            display = tool_names.get(action, f"🔧 {action}")
            yield f"data: {json.dumps({'type':'tool', 'text':display})}\n\n"
            await asyncio.sleep(0.2)

            result = await run_tool(action, args)

            # Show short result preview
            preview = result[:150] + "..." if len(result) > 150 else result
            yield f"data: {json.dumps({'type':'result', 'text':f'📋 {preview}'})}\n\n"
            await asyncio.sleep(0.1)

            # Feed result back to model
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": f"Tool result:\n{result[:3000]}\n\nContinue with the task."})

        # Max steps reached
        yield f"data: {json.dumps({'type':'token', 'text':'⚠️ Task bahut bada tha, steps khatam ho gaye. Jo hua uska summary: task partially complete.'})}\n\n"
        yield "data: [DONE]\n\n"

# ── Schemas ───────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message:  str
    history:  list = []
    password: str  = ""

# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return HTMLResponse(open("static/index.html", encoding="utf-8").read())

@app.get("/status")
async def status():
    return {
        "ok":      bool(GROQ_KEY),
        "version": "NanoBot-1.0",
        "pdf":     PDF_OK,
        "search":  SEARCH_OK,
        "tools":   ["web_search","write_file","read_file","append_file",
                    "list_files","delete_file","calculator","summarize_pdf"]
    }

@app.post("/chat")
async def chat(req: ChatRequest, request: Request):
    if not rate_ok(get_ip(request)):
        raise HTTPException(429, "Zyada requests! 1 min baad try karo.")
    if not auth_ok(req.password):
        raise HTTPException(401, "Password galat hai!")
    if not GROQ_KEY:
        raise HTTPException(503, "GROQ_API_KEY set nahi! Render Environment mein add karo.")
    if not is_safe(req.message):
        raise HTTPException(403, "⚠️ Payment/banking requests blocked!")
    return StreamingResponse(
        agent_stream(req.message, req.history),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    if not PDF_OK:
        raise HTTPException(501, "pdfplumber install nahi hai")
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Sirf PDF allowed")
    raw  = await file.read()
    name = Path(file.filename).name
    path = WORKSPACE / name
    path.write_bytes(raw)
    pages = []
    with pdfplumber.open(path) as pdf:
        total = len(pdf.pages)
        for i, pg in enumerate(pdf.pages[:25]):
            t = pg.extract_text()
            if t: pages.append(f"[Page {i+1}]\n{t.strip()}")
    full = "\n\n".join(pages)
    return {
        "filename":        name,
        "total_pages":     total,
        "extracted_pages": len(pages),
        "text":            full[:10000],
        "preview":         full[:300],
        "word_count":      len(full.split())
    }

@app.get("/files")
async def list_files():
    icons = {".txt":"📝",".md":"📋",".py":"🐍",".json":"📦",".csv":"📊",".pdf":"📄",".html":"🌐"}
    return {"files": [
        {"name": f.name, "size": f.stat().st_size, "icon": icons.get(f.suffix,"📄")}
        for f in sorted(WORKSPACE.iterdir()) if f.is_file()
    ]}

@app.get("/file/{filename}")
async def get_file(filename: str):
    path = WORKSPACE / Path(filename).name
    if not path.exists():
        raise HTTPException(404, "File not found")
    return {"content": path.read_text(encoding="utf-8", errors="replace")}

@app.delete("/file/{filename}")
async def del_file(filename: str):
    path = WORKSPACE / Path(filename).name
    if path.exists():
        path.unlink()
        return {"ok": True}
    raise HTTPException(404, "File not found")

app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=PORT, reload=False)
