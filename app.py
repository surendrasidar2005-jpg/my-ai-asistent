"""
MyAssistant Agent v3
True AI Agent with tool calling loop
- Web search, File R/W, PDF, Calculator, Code runner
- Payment completely blocked
- Agentic loop: thinks → picks tool → executes → thinks again → done
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx, os, json, time, asyncio, math, subprocess, traceback
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

# ── App ─────────────────────────────────────────────────────────────────────
app = FastAPI(title="MyAssistant Agent v3")

# ── Fix 4: CORS Restriction ──────────────────────────────────────────────────
ALLOWED_ORIGINS = [
    "https://my-ai-asistent.onrender.com",
    "http://localhost:8000"
]
app.add_middleware(
    CORSMiddleware, 
    allow_origins=ALLOWED_ORIGINS, 
    allow_methods=["POST", "GET"], 
    allow_headers=["*"]
)

WORKSPACE   = Path("workspace"); WORKSPACE.mkdir(exist_ok=True)
GROQ_KEY    = os.environ.get("GROQ_API_KEY", "")
ACCESS_PASS = os.environ.get("ACCESS_PASSWORD", "")
PORT        = int(os.environ.get("PORT", 8000))
GROQ_URL    = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL  = "llama-3.3-70b-versatile"

# ── Safety ──────────────────────────────────────────────────────────────────
BLOCKED = [
    "upi","paytm","phonepe","gpay","google pay","amazon pay","bhim",
    "credit card","debit card","card number","account number","ifsc","cvv",
    "bank transfer","send money","neft","rtgs","imps","net banking",
    "otp for payment","wallet transfer","razorpay","stripe","paypal",
    "payment gateway","transaction","bank account","financial transfer"
]

def is_safe(text: str) -> bool:
    t = text.lower()
    return not any(k in t for k in BLOCKED)

def auth_ok(pw: str) -> bool:
    return (not ACCESS_PASS) or (pw == ACCESS_PASS)

# ── Fix 1: Memory Helpers ────────────────────────────────────────────────────
MEMORY_FILE = WORKSPACE / "memory.json"

def load_memory() -> list:
    if not MEMORY_FILE.exists():
        return []
    try:
        return json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
    except:
        return []

def save_memory(facts: list):
    old = load_memory()
    new = old + facts
    # Keep max 20 entries, delete oldest
    new = new[-20:]
    MEMORY_FILE.write_text(json.dumps(new, indent=2, ensure_ascii=False), encoding="utf-8")

async def extract_facts(user_msg: str, ai_msg: str):
    """Hidden LLM call to extract key facts for memory"""
    if not GROQ_KEY: return
    prompt = f"Extract 1-2 important facts (names, preferences, results) from this chat. User: {user_msg}. AI: {ai_msg}. Return as JSON array of strings: [\"fact1\", \"fact2\"]. If no new facts, return []"
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(GROQ_URL, headers={"Authorization": f"Bearer {GROQ_KEY}"}, json={
                "model": "llama-3.1-8b-instant",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "response_format": {"type": "json_object"}
            })
            data = resp.json()
            facts = json.loads(data["choices"][0]["message"]["content"]).get("facts", [])
            if facts: save_memory(facts)
        except:
            pass

# ── TOOLS DEFINITION (what agent can use) ───────────────────────────────────
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current information. Use when user asks to find, search, or look up something online.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query to look up"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Create or overwrite a file in workspace with given content. Use when user asks to save, write, or create a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string", "description": "File name (e.g. notes.txt, report.md)"},
                    "content":  {"type": "string", "description": "Full content to write to the file"}
                },
                "required": ["filename", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read contents of a file from workspace. Use when user asks to read, open, or view a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string", "description": "File name to read"}
                },
                "required": ["filename"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List all files in workspace. Use when user asks what files exist.",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a mathematical expression. Use for any calculation, formula, or math problem.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Math expression to evaluate (e.g. '2 * (3 + 4)', 'sqrt(144)', 'sin(45 * pi / 180)')"}
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "append_file",
            "description": "Append content to end of existing file. Use when user wants to add to a file without overwriting.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string", "description": "File name to append to"},
                    "content":  {"type": "string", "description": "Content to append"}
                },
                "required": ["filename", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "delete_file",
            "description": "Delete a file from workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string", "description": "File name to delete"}
                },
                "required": ["filename"]
            }
        }
    }
]

# ── Tool Executor ────────────────────────────────────────────────────────────
async def execute_tool(name: str, args: dict) -> str:
    try:
        if name == "web_search":
            query = args.get("query", "")
            if not is_safe(query):
                return "❌ BLOCKED: Payment related search blocked."
            if not SEARCH_OK:
                return "Search unavailable (duckduckgo-search not installed)"
            results = []
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=5):
                    results.append(f"Title: {r.get('title','')}\nSnippet: {r.get('body','')}\nURL: {r.get('href','')}")
            return "\n\n---\n\n".join(results) if results else "No results found."

        elif name == "write_file":
            fname   = Path(args["filename"]).name
            content = args.get("content", "")
            if not is_safe(content):
                return "❌ BLOCKED: Payment related content blocked."
            (WORKSPACE / fname).write_text(content, encoding="utf-8")
            return f"✅ File '{fname}' successfully saved! ({len(content)} characters)"

        elif name == "append_file":
            fname   = Path(args["filename"]).name
            content = args.get("content", "")
            if not is_safe(content):
                return "❌ BLOCKED: Payment related content blocked."
            path = WORKSPACE / fname
            existing = path.read_text(encoding="utf-8") if path.exists() else ""
            path.write_text(existing + "\n" + content, encoding="utf-8")
            return f"✅ Content appended to '{fname}'"

        elif name == "read_file":
            fname = Path(args["filename"]).name
            path  = WORKSPACE / fname
            if not path.exists():
                return f"❌ File '{fname}' not found. Use list_files to see available files."
            return path.read_text(encoding="utf-8", errors="replace")

        elif name == "list_files":
            files = [f.name for f in WORKSPACE.iterdir() if f.is_file()]
            return "Files in workspace:\n" + "\n".join(f"- {f}" for f in files) if files else "No files in workspace yet."

        elif name == "calculator":
            expr = args.get("expression", "")
            safe_globals = {
                "__builtins__": {},
                "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos,
                "tan": math.tan, "log": math.log, "log10": math.log10,
                "pi": math.pi, "e": math.e, "abs": abs,
                "round": round, "pow": pow, "floor": math.floor,
                "ceil": math.ceil, "factorial": math.factorial
            }
            result = eval(expr, safe_globals)
            return f"Result: {result}"

        elif name == "delete_file":
            fname = Path(args["filename"]).name
            path  = WORKSPACE / fname
            if path.exists():
                path.unlink()
                return f"✅ File '{fname}' deleted."
            return f"❌ File '{fname}' not found."

        else:
            return f"Unknown tool: {name}"

    except Exception as e:
        return f"❌ Tool error: {str(e)}"

# ── System Prompt ────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are MyAssistant, a powerful AI Agent. You can autonomously complete multi-step tasks using your tools.

## Your Tools
- **web_search**: Search internet for current information
- **write_file**: Create/save files in workspace
- **read_file**: Read existing files
- **append_file**: Add content to existing files
- **list_files**: See all files in workspace
- **calculator**: Solve any math/calculation
- **delete_file**: Remove files

## Agent Behavior Rules
1. When given a task, THINK first about what steps are needed
2. Use tools autonomously — don't ask for permission for each step
3. Chain tools together: search → summarize → save to file (all in one go!)
4. After completing a task, give a clear summary of what you did
5. If something fails, try an alternative approach
6. Respond in Hindi or English based on what the user uses

## ABSOLUTE SAFETY RULES (NEVER VIOLATE)
1. NEVER assist with payment systems, UPI, banking, or financial transfers
2. NEVER access files outside workspace directory
3. NEVER run system commands or access OS
4. NEVER share API keys or sensitive data

## Examples of what you can do autonomously:
- "Mining ke baare mein search karo aur notes.txt mein save karo" → searches + saves automatically
- "Calculate blast radius using formula" → calculates step by step
- "Meri sabhi files list karo aur summary do" → lists + summarizes
- "Search for rock mechanics and create a study guide" → searches + writes structured file"""

# ── Rate limiting ─────────────────────────────────────────────────────────────
rate_store: dict = defaultdict(list)
def rate_ok(ip: str) -> bool:
    now = time.time()
    rate_store[ip] = [t for t in rate_store[ip] if now - t < 60]
    if len(rate_store[ip]) >= 20: return False
    rate_store[ip].append(now)
    return True

# ── Schemas ───────────────────────────────────────────────────────────────────
class AgentRequest(BaseModel):
    message: str
    history: list = []
    password: str = ""

# ── Agentic SSE Stream ────────────────────────────────────────────────────────
async def agent_stream(messages: list):
    """
    True agentic loop:
    1. Call LLM
    2. If tool_calls → execute tools → feed results back → loop
    3. If final text → stream to user
    """
    MAX_ITERATIONS = 8
    iteration = 0

    async with httpx.AsyncClient(timeout=120) as client:
        while iteration < MAX_ITERATIONS:
            iteration += 1

            # Call Groq
            payload = {
                "model": GROQ_MODEL,
                "messages": messages,
                "tools": TOOLS,
                "tool_choice": "auto",
                "temperature": 0.4,
                "max_tokens": 2048
            }

            try:
                resp = await client.post(
                    GROQ_URL,
                    headers={"Authorization": f"Bearer {GROQ_KEY}", "Content-Type": "application/json"},
                    json=payload
                )
            except Exception as e:
                yield f"data: {json.dumps({'type':'error','text':f'Connection error: {e}'})}\n\n"
                return

            if resp.status_code != 200:
                err = resp.text[:300]
                yield f"data: {json.dumps({'type':'error','text':f'Groq error {resp.status_code}: {err}'})}\n\n"
                return

            data      = resp.json()
            choice    = data["choices"][0]
            message   = choice["message"]
            finish    = choice["finish_reason"]

            # Add assistant response to history
            messages.append(message)

            # ── Tool calls ───────────────────────────────────────────────
            if finish == "tool_calls" and message.get("tool_calls"):
                tool_results = []

                for tc in message["tool_calls"]:
                    tool_name = tc["function"]["name"]
                    try:
                        tool_args = json.loads(tc["function"]["arguments"])
                    except:
                        tool_args = {}

                    # Tell user what agent is doing
                    tool_display = {
                        "web_search":  f"🔍 Searching: {tool_args.get('query','')}",
                        "write_file":  f"💾 Saving file: {tool_args.get('filename','')}",
                        "read_file":   f"📖 Reading file: {tool_args.get('filename','')}",
                        "list_files":  "📂 Listing workspace files...",
                        "calculator":  f"🧮 Calculating: {tool_args.get('expression','')}",
                        "append_file": f"✏️ Appending to: {tool_args.get('filename','')}",
                        "delete_file": f"🗑️ Deleting: {tool_args.get('filename','')}"
                    }.get(tool_name, f"🔧 Using tool: {tool_name}")

                    yield f"data: {json.dumps({'type':'tool','text':tool_display})}\n\n"
                    await asyncio.sleep(0.1)

                    # Fix 3: Tool Timeout
                    try:
                        result = await asyncio.wait_for(execute_tool(tool_name, tool_args), timeout=10.0)
                    except asyncio.TimeoutError:
                        result = "❌ Tool timed out after 10 seconds, trying next step."

                    tool_results.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": result[:3000]  # limit result size
                    })

                # Feed all tool results back
                messages.extend(tool_results)
                continue  # loop again

            # ── Final response ────────────────────────────────────────────
            if finish in ("stop", "length") or message.get("content"):
                content = message.get("content", "")
                if content:
                    # Fix 1: Trigger background memory extraction
                    user_msg = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
                    asyncio.create_task(extract_facts(user_msg, content))

                    # Stream token by token
                    words = content.split(" ")
                    for i, word in enumerate(words):
                        chunk = word + (" " if i < len(words)-1 else "")
                        yield f"data: {json.dumps({'type':'token','text':chunk})}\n\n"
                        await asyncio.sleep(0.01)
                yield "data: [DONE]\n\n"
                return

            # Safety: no more tool calls and no content
            yield "data: [DONE]\n\n"
            return

    yield f"data: {json.dumps({'type':'error','text':'Agent max iterations reached.'})}\n\n"
    yield "data: [DONE]\n\n"

# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return HTMLResponse(open("index.html", encoding="utf-8").read())

# ── Fix 2: Health Check ──────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": time.time(), "msg": "Server is warm and ready"}

@app.get("/status")
async def status():
    return {
        "ok": bool(GROQ_KEY), "model": GROQ_MODEL,
        "pdf": PDF_OK, "search": SEARCH_OK,
        "password_protected": bool(ACCESS_PASS),
        "version": "3.0.0-agent",
        "tools": [t["function"]["name"] for t in TOOLS]
    }

@app.post("/agent")
async def agent(req: AgentRequest, request: Request):
    if not rate_ok(request.client.host):
        raise HTTPException(429, "Bahut zyada requests! 1 min baad try karo.")
    if not auth_ok(req.password):
        raise HTTPException(401, "Password galat hai!")
    if not GROQ_KEY:
        raise HTTPException(503, "GROQ_API_KEY set nahi hai!")
    if not is_safe(req.message):
        raise HTTPException(403, "⚠️ Payment/banking related requests blocked!")

    # Fix 1: Inject Memory Context
    memory = load_memory()
    mem_ctx = "\nPrevious context: " + " | ".join(memory) if memory else ""
    full_prompt = SYSTEM_PROMPT + mem_ctx

    messages = [{"role": "system", "content": full_prompt}]
    for h in req.history[-10:]:
        if h.get("role") in ("user", "assistant") and h.get("content"):
            messages.append({"role": h["role"], "content": h["content"]})
    messages.append({"role": "user", "content": req.message})

    return StreamingResponse(
        agent_stream(messages),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    if not PDF_OK:
        raise HTTPException(501, "pdfplumber nahi hai")
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Sirf PDF allowed")
    raw  = await file.read()
    safe = Path(file.filename).name
    path = WORKSPACE / safe
    path.write_bytes(raw)
    pages = []
    with pdfplumber.open(path) as pdf:
        total = len(pdf.pages)
        for i, page in enumerate(pdf.pages[:25]):
            t = page.extract_text()
            if t: pages.append(t.strip())
    
    full_text = "\n\n".join(pages)
    
    # Fix 5: PDF Chunking (Max 2000 chars per chunk, max 5 chunks)
    CHUNK_SIZE = 2000
    all_chunks = [full_text[i:i + CHUNK_SIZE] for i in range(0, len(full_text), CHUNK_SIZE)]
    chunks = all_chunks[:5]
    processed_text = "\n\n--- Next Chunk ---\n\n".join(chunks)
    
    note = f" (Showing analysis of first {len(chunks)} chunks/pages)"
    
    return {
        "filename": safe, 
        "total_pages": total,
        "text": processed_text + note, 
        "preview": processed_text[:400],
        "word_count": len(processed_text.split()),
        "note": note
    }

@app.get("/files")
async def list_files_api():
    icons = {".txt":"📝",".md":"📋",".py":"🐍",".json":"📦",".csv":"📊",".pdf":"📄"}
    return {"files": [
        {"name": f.name, "size": f.stat().st_size,
         "icon": icons.get(f.suffix, "📄"), "ext": f.suffix}
        for f in sorted(WORKSPACE.iterdir()) if f.is_file()
    ]}

@app.get("/file/{filename}")
async def get_file(filename: str):
    path = WORKSPACE / Path(filename).name
    if not path.exists():
        raise HTTPException(404, "File not found")
    return {"content": path.read_text(encoding="utf-8", errors="replace")}

app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import uvicorn
    # Fix 2: Warm startup message
    print("Server is warm and ready")
    uvicorn.run("app:app", host="0.0.0.0", port=PORT, reload=False)
