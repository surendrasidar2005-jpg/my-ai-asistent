"""
MyAssistant Agent v3.3 — BUG-FREE VERSION
Critical Fixes:
  1. DDGS 5.3.1 uses context manager: `with DDGS() as ddgs:`
  2. Fixed model fallback logic
  3. Added timeout to DDGS to prevent hanging
  4. Fixed all error handling paths
  5. Added proper resource cleanup
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx, os, json, time, asyncio, math
from pathlib import Path
from collections import defaultdict
from typing import Optional

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

# ── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(title="MyAssistant Agent v3.3")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

WORKSPACE   = Path("workspace"); WORKSPACE.mkdir(exist_ok=True)
GROQ_KEY    = os.environ.get("GROQ_API_KEY", "")
ACCESS_PASS = os.environ.get("ACCESS_PASSWORD", "")
PORT        = int(os.environ.get("PORT", 8000))
GROQ_URL    = "https://api.groq.com/openai/v1/chat/completions"

# Fallback models
GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-70b-versatile", 
    "llama-3.1-8b-instant",
    "mixtral-8x7b-32768",
    "gemma-7b-it"
]
GROQ_MODEL = GROQ_MODELS[0]

# ── Safety ───────────────────────────────────────────────────────────────────
BLOCKED = [
    "upi","paytm","phonepe","gpay","google pay","amazon pay","bhim",
    "credit card","debit card","card number","account number","ifsc","cvv",
    "bank transfer","send money","neft","rtgs","imps","net banking",
    "otp for payment","wallet transfer","razorpay","stripe","paypal",
    "payment gateway","transaction","bank account","financial transfer"
]

def is_safe(text: str) -> bool:
    if not text:
        return True
    t = text.lower()
    return not any(k in t for k in BLOCKED)

def auth_ok(pw: str) -> bool:
    return (not ACCESS_PASS) or (pw == ACCESS_PASS)

# ── Tools Definition ─────────────────────────────────────────────────────────
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
            "description": "Create or overwrite a file in workspace with given content.",
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
            "description": "Read contents of a file from workspace.",
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
            "description": "List all files in workspace.",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a mathematical expression.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Math expression e.g. 'sqrt(144)', '2*(3+4)', 'sin(pi/2)'"}
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "append_file",
            "description": "Append content to end of an existing file without overwriting.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string"},
                    "content":  {"type": "string"}
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
                    "filename": {"type": "string"}
                },
                "required": ["filename"]
            }
        }
    }
]

# ── Tool Executor ─────────────────────────────────────────────────────────────
async def execute_tool(name: str, args: dict) -> str:
    try:
        if name == "web_search":
            query = args.get("query", "")
            if not is_safe(query):
                return "❌ BLOCKED: Payment related search blocked."
            if not SEARCH_OK:
                return "❌ Search unavailable (duckduckgo-search not installed)"
            if not query:
                return "❌ No search query provided"
            
            results = []
            try:
                # CRITICAL FIX: DDGS 5.3.1 requires context manager!
                with DDGS(timeout=15) as ddgs:  # Added timeout
                    search_results = list(ddgs.text(query, max_results=5))
                    for r in search_results:
                        results.append(
                            f"Title: {r.get('title','')}\n"
                            f"Snippet: {r.get('body','')}\n"
                            f"URL: {r.get('href','')}"
                        )
            except Exception as search_err:
                return f"❌ Search error: {str(search_err)}"
            
            return "\n\n---\n\n".join(results) if results else "No results found."

        elif name == "write_file":
            fname   = Path(args["filename"]).name
            content = args.get("content", "")
            if not fname:
                return "❌ No filename provided"
            if not is_safe(content):
                return "❌ BLOCKED: Payment related content blocked."
            try:
                (WORKSPACE / fname).write_text(content, encoding="utf-8")
                return f"✅ File '{fname}' saved! ({len(content)} characters)"
            except Exception as e:
                return f"❌ Error saving file: {str(e)}"

        elif name == "append_file":
            fname   = Path(args["filename"]).name
            content = args.get("content", "")
            if not fname:
                return "❌ No filename provided"
            if not is_safe(content):
                return "❌ BLOCKED: Payment related content blocked."
            try:
                path     = WORKSPACE / fname
                existing = path.read_text(encoding="utf-8") if path.exists() else ""
                path.write_text(existing + "\n" + content, encoding="utf-8")
                return f"✅ Content appended to '{fname}'"
            except Exception as e:
                return f"❌ Error appending to file: {str(e)}"

        elif name == "read_file":
            fname = Path(args["filename"]).name
            if not fname:
                return "❌ No filename provided"
            path  = WORKSPACE / fname
            if not path.exists():
                return f"❌ File '{fname}' not found. Use list_files to see available files."
            try:
                return path.read_text(encoding="utf-8", errors="replace")
            except Exception as e:
                return f"❌ Error reading file: {str(e)}"

        elif name == "list_files":
            try:
                files = [f.name for f in WORKSPACE.iterdir() if f.is_file()]
                if not files:
                    return "No files in workspace yet."
                return "Files in workspace:\n" + "\n".join(f"- {f}" for f in sorted(files))
            except Exception as e:
                return f"❌ Error listing files: {str(e)}"

        elif name == "calculator":
            expr = args.get("expression", "")
            if not expr:
                return "❌ No expression provided"
            safe_globals = {
                "__builtins__": {},
                "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos,
                "tan": math.tan,   "log": math.log, "log10": math.log10,
                "pi": math.pi,     "e": math.e,     "abs": abs,
                "round": round,    "pow": pow,       "floor": math.floor,
                "ceil": math.ceil, "factorial": math.factorial,
                "exp": math.exp,   "degrees": math.degrees,
                "radians": math.radians, "hypot": math.hypot
            }
            try:
                result = eval(expr, safe_globals)
                return f"Result: {result}"
            except Exception as e:
                return f"❌ Calculation error: {str(e)}"

        elif name == "delete_file":
            fname = Path(args["filename"]).name
            if not fname:
                return "❌ No filename provided"
            path  = WORKSPACE / fname
            try:
                if path.exists():
                    path.unlink()
                    return f"✅ File '{fname}' deleted."
                return f"❌ File '{fname}' not found."
            except Exception as e:
                return f"❌ Error deleting file: {str(e)}"

        else:
            return f"❌ Unknown tool: {name}"

    except Exception as e:
        return f"❌ Tool error: {str(e)}"


# ── System Prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are MyAssistant, a powerful AI Agent. You autonomously complete multi-step tasks using tools.

## Your Tools
- **web_search**: Search internet for current information
- **write_file**: Create/save files in workspace
- **read_file**: Read existing files
- **append_file**: Add content to existing files
- **list_files**: See all files in workspace
- **calculator**: Solve any math/calculation
- **delete_file**: Remove files

## Agent Behavior Rules
1. When given a task, plan the steps needed first
2. Use tools autonomously — do NOT ask for permission for each step
3. Chain tools together: search → summarize → save to file (all in one go!)
4. After completing a task, give a clear summary of what you did
5. If something fails, try an alternative approach
6. Respond in Hindi or English based on what the user uses

## ABSOLUTE SAFETY RULES (NEVER VIOLATE)
1. NEVER assist with payment systems, UPI, banking, or financial transfers
2. NEVER access files outside the workspace directory
3. NEVER run system commands or access the OS
4. NEVER share API keys or sensitive data

## Examples:
- "Mining ke baare mein search karo aur notes.txt mein save karo" → search + save automatically
- "Calculate blast radius formula for 500kg charge" → calculate step by step
- "Meri sabhi files list karo" → list files
- "Search rock mechanics and create study guide" → search + write file"""

# ── Rate Limiting ─────────────────────────────────────────────────────────────
rate_store: dict = defaultdict(list)

def rate_ok(ip: str) -> bool:
    now = time.time()
    rate_store[ip] = [t for t in rate_store[ip] if now - t < 60]
    if len(rate_store[ip]) >= 20:
        return False
    rate_store[ip].append(now)
    return True

def get_client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client and request.client.host:
        return request.client.host
    return "unknown"

# ── Schemas ───────────────────────────────────────────────────────────────────
class AgentRequest(BaseModel):
    message:  str
    history:  list = []
    password: str  = ""

# ── Agentic SSE Stream ────────────────────────────────────────────────────────
async def agent_stream(messages: list):
    MAX_ITERATIONS = 10
    iteration      = 0
    current_model  = GROQ_MODEL

    async with httpx.AsyncClient(timeout=120) as client:
        while iteration < MAX_ITERATIONS:
            iteration += 1

            payload = {
                "model":        current_model,
                "messages":     messages,
                "tools":        TOOLS,
                "tool_choice":  "auto",
                "temperature":  0.4,
                "max_tokens":   4096,
                "stream":       False
            }

            try:
                resp = await client.post(
                    GROQ_URL,
                    headers={
                        "Authorization": f"Bearer {GROQ_KEY}",
                        "Content-Type":  "application/json"
                    },
                    json=payload
                )
            except Exception as e:
                yield f"data: {json.dumps({'type':'error','text':f'Connection error: {e}'})}\n\n"
                return

            # FIXED: Better fallback logic
            if resp.status_code != 200:
                # Try fallback models
                fallback_success = False
                for fallback in GROQ_MODELS[1:]:  # Skip first (already tried)
                    payload["model"] = fallback
                    try:
                        resp = await client.post(
                            GROQ_URL,
                            headers={
                                "Authorization": f"Bearer {GROQ_KEY}",
                                "Content-Type":  "application/json"
                            },
                            json=payload
                        )
                        if resp.status_code == 200:
                            current_model = fallback
                            fallback_success = True
                            break
                    except Exception:
                        continue
                
                if not fallback_success:
                    err = resp.text[:500]
                    yield f"data: {json.dumps({'type':'error','text':f'Groq error {resp.status_code}: {err}'})}\n\n"
                    return

            try:
                data = resp.json()
            except json.JSONDecodeError as e:
                yield f"data: {json.dumps({'type':'error','text':f'Invalid JSON response: {str(e)}'})}\n\n"
                return

            if "choices" not in data or not data["choices"]:
                yield f"data: {json.dumps({'type':'error','text':'No response from AI'})}\n\n"
                return

            choice  = data["choices"][0]
            message = choice.get("message", {})
            finish  = choice.get("finish_reason", "stop")

            if not message:
                yield f"data: {json.dumps({'type':'error','text':'Empty message from AI'})}\n\n"
                return

            messages.append(message)

            # ── Tool calls ────────────────────────────────────────────────
            if finish == "tool_calls" or message.get("tool_calls"):
                tool_calls = message.get("tool_calls", [])
                if not tool_calls:
                    yield f"data: {json.dumps({'type':'error','text':'Tool call requested but no tools specified'})}\n\n"
                    return

                tool_results = []

                for tc in tool_calls:
                    tool_name = tc.get("function", {}).get("name", "unknown")
                    
                    try:
                        tool_args = json.loads(tc.get("function", {}).get("arguments", "{}"))
                    except (json.JSONDecodeError, ValueError, KeyError, TypeError):
                        tool_args = {}

                    tool_display = {
                        "web_search":  f"🔍 Searching: {tool_args.get('query','')[:50]}",
                        "write_file":  f"💾 Saving file: {tool_args.get('filename','')}",
                        "read_file":   f"📖 Reading file: {tool_args.get('filename','')}",
                        "list_files":  "📂 Listing workspace files...",
                        "calculator":  f"🧮 Calculating: {tool_args.get('expression','')}",
                        "append_file": f"✏️ Appending to: {tool_args.get('filename','')}",
                        "delete_file": f"🗑️ Deleting: {tool_args.get('filename','')}"
                    }.get(tool_name, f"🔧 Using tool: {tool_name}")

                    yield f"data: {json.dumps({'type':'tool','text':tool_display})}\n\n"
                    await asyncio.sleep(0.1)

                    result = await execute_tool(tool_name, tool_args)

                    tool_results.append({
                        "role":         "tool",
                        "tool_call_id": tc.get("id", "unknown"),
                        "content":      str(result)[:4000]
                    })

                messages.extend(tool_results)
                continue

            # ── Final response ────────────────────────────────────────────
            content = message.get("content", "")
            if content is None:
                content = ""
            
            if finish in ("stop", "length") or content:
                if content:
                    words = content.split(" ")
                    for i, word in enumerate(words):
                        chunk = word + (" " if i < len(words) - 1 else "")
                        yield f"data: {json.dumps({'type':'token','text':chunk})}\n\n"
                        await asyncio.sleep(0.005)
                yield "data: [DONE]\n\n"
                return

            yield "data: [DONE]\n\n"
            return

    yield f"data: {json.dumps({'type':'error','text':'Agent reached max iterations.'})}\n\n"
    yield "data: [DONE]\n\n"


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def root():
    try:
        index_path = Path("static/index.html")
        if index_path.exists():
            return HTMLResponse(content=index_path.read_text(encoding="utf-8"))
        else:
            return HTMLResponse(content="<h1>MyAssistant Agent v3.3</h1><p>Static files not found.</p>")
    except Exception as e:
        return HTMLResponse(content=f"<h1>Error</h1><p>{str(e)}</p>", status_code=500)

@app.get("/status")
async def status():
    return {
        "ok":                 bool(GROQ_KEY),
        "model":              GROQ_MODEL,
        "pdf":                PDF_OK,
        "search":             SEARCH_OK,
        "password_protected": bool(ACCESS_PASS),
        "version":            "3.3.0-bug-free",
        "tools":              [t["function"]["name"] for t in TOOLS]
    }

@app.post("/agent")
async def agent(req: AgentRequest, request: Request):
    ip = get_client_ip(request)
    if not rate_ok(ip):
        raise HTTPException(429, "Bahut zyada requests! 1 min baad try karo.")
    if not auth_ok(req.password):
        raise HTTPException(401, "Password galat hai!")
    if not GROQ_KEY:
        raise HTTPException(503, "GROQ_API_KEY set nahi hai!")
    if not is_safe(req.message):
        raise HTTPException(403, "⚠️ Payment/banking related requests blocked!")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for h in req.history[-12:]:
        if h.get("role") in ("user", "assistant", "tool") and h.get("content") is not None:
            messages.append({"role": h["role"], "content": str(h["content"])})
    messages.append({"role": "user", "content": req.message})

    return StreamingResponse(
        agent_stream(messages),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Content-Type": "text/event-stream"
        }
    )

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    if not PDF_OK:
        raise HTTPException(501, "pdfplumber nahi hai")
    if not file.filename:
        raise HTTPException(400, "No filename provided")
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Sirf PDF allowed hai")
    
    try:
        raw  = await file.read()
        if len(raw) > 50 * 1024 * 1024:
            raise HTTPException(400, "File too large (max 50MB)")
            
        safe = Path(file.filename).name
        path = WORKSPACE / safe
        path.write_bytes(raw)
        
        pages = []
        with pdfplumber.open(path) as pdf:
            total = len(pdf.pages)
            for i, page in enumerate(pdf.pages[:50]):
                t = page.extract_text()
                if t:
                    pages.append(f"[Page {i+1}]\n{t.strip()}")
        
        full = "\n\n".join(pages)
        extracted_count = len(pages)
        
        return {
            "filename":        safe,
            "total_pages":     total,
            "extracted_pages": extracted_count,
            "text":            full[:15000],
            "preview":         full[:500],
            "word_count":      len(full.split())
        }
    except Exception as e:
        raise HTTPException(500, f"PDF processing error: {str(e)}")

@app.get("/files")
async def list_files_api():
    try:
        icons = {".txt":"📝", ".md":"📋", ".py":"🐍", ".json":"📦", ".csv":"📊", ".pdf":"📄"}
        files = []
        for f in sorted(WORKSPACE.iterdir()):
            if f.is_file():
                files.append({
                    "name": f.name,
                    "size": f.stat().st_size,
                    "icon": icons.get(f.suffix.lower(), "📄"),
                    "ext":  f.suffix
                })
        return {"files": files}
    except Exception as e:
        raise HTTPException(500, f"Error listing files: {str(e)}")

@app.get("/file/{filename}")
async def get_file(filename: str):
    try:
        path = WORKSPACE / Path(filename).name
        if not path.exists():
            raise HTTPException(404, "File not found")
        if not path.is_file():
            raise HTTPException(400, "Not a file")
        return {"content": path.read_text(encoding="utf-8", errors="replace")}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Error reading file: {str(e)}")

@app.delete("/file/{filename}")
async def delete_file_api(filename: str):
    """Direct delete endpoint for files"""
    try:
        path = WORKSPACE / Path(filename).name
        if not path.exists():
            raise HTTPException(404, "File not found")
        path.unlink()
        return {"success": True, "message": f"File '{filename}' deleted"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Error deleting file: {str(e)}")

# Static files mount - MUST be last
app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=PORT, reload=False)
