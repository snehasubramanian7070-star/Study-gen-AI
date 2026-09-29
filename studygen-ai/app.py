"""
StudyGen AI – Smart Study Generator
====================================
A Flask web application powered by the Groq API.
Implements two AI agents:
  1. Study Material Agent  – transforms notes/files/images into study aids
  2. Study Planner & Evaluation Agent – personalized plans + MCQ quizzes

Run:  python app.py
URL:  http://127.0.0.1:5000
"""

import os
import io
import json
import base64
import sqlite3
import datetime
import traceback

from flask import (
    Flask, request, jsonify, session,
    render_template_string, redirect, url_for
)
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Optional heavy dependencies – gracefully degrade when absent
# ---------------------------------------------------------------------------
try:
    import groq as groq_lib          # official Groq Python client
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "studygen-dev-secret-key-change-me")

GROQ_API_KEY      = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL        = os.getenv("GROQ_MODEL", "llama3-70b-8192")
GROQ_VISION_MODEL = os.getenv("GROQ_VISION_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
GROQ_STT_MODEL    = os.getenv("GROQ_STT_MODEL", "whisper-large-v3-turbo")

MAX_UPLOAD_MB     = 10
MAX_UPLOAD_BYTES  = MAX_UPLOAD_MB * 1024 * 1024
ALLOWED_TEXT_EXT  = {".txt", ".pdf"}
ALLOWED_IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".gif"}

DB_PATH = "studygen.db"

# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------
def get_db():
    """Return a SQLite connection (row_factory = sqlite3.Row)."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create tables if they do not already exist."""
    conn = get_db()
    c = conn.cursor()
    c.executescript("""
        CREATE TABLE IF NOT EXISTS quiz_results (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            subject     TEXT,
            score       INTEGER,
            total       INTEGER,
            percentage  REAL,
            weak_topics TEXT,
            taken_at    TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS study_plans (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            subject       TEXT,
            exam_date     TEXT,
            hours_per_day INTEGER,
            prep_level    TEXT,
            plan_json     TEXT,
            created_at    TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS study_tasks (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            plan_id     INTEGER,
            day_label   TEXT,
            task        TEXT,
            completed   INTEGER DEFAULT 0,
            created_at  TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS milestones (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            title       TEXT,
            description TEXT,
            achieved_at TEXT DEFAULT (datetime('now'))
        );
    """)
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Groq helpers
# ---------------------------------------------------------------------------
def get_groq_client():
    """Return an initialised Groq client or None if unavailable."""
    if not GROQ_AVAILABLE:
        return None
    if not GROQ_API_KEY:
        return None
    return groq_lib.Groq(api_key=GROQ_API_KEY)


def generate_response(system_prompt: str, user_message: str,
                      model: str | None = None,
                      temperature: float = 0.7,
                      max_tokens: int = 4096) -> dict:
    """
    Central function to call the Groq chat-completions API.
    Returns {"success": True/False, "content": "...", "error": "..."}
    """
    client = get_groq_client()
    if client is None:
        if not GROQ_API_KEY:
            return {"success": False,
                    "error": "GROQ_API_KEY is not set. "
                             "Please add it to your .env file and restart."}
        return {"success": False,
                "error": "Groq Python library not installed. "
                         "Run: pip install groq"}

    chosen_model = model or GROQ_MODEL
    try:
        completion = client.chat.completions.create(
            model=chosen_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_message},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = completion.choices[0].message.content
        return {"success": True, "content": content}
    except Exception as exc:
        err = str(exc)
        app.logger.error("Groq API error: %s", traceback.format_exc())
        return {"success": False, "error": f"Groq API error: {err}"}


def generate_vision_response(system_prompt: str, user_text: str,
                              image_b64: str, mime_type: str = "image/jpeg") -> dict:
    """
    Call the Groq vision model with an inline base-64 image.
    """
    client = get_groq_client()
    if client is None:
        return {"success": False,
                "error": "Groq client unavailable – check API key and library."}
    try:
        completion = client.chat.completions.create(
            model=GROQ_VISION_MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": f"{system_prompt}\n\n{user_text}"},
                    {"type": "image_url",
                     "image_url": {"url": f"data:{mime_type};base64,{image_b64}"}},
                ],
            }],
            temperature=0.5,
            max_tokens=4096,
        )
        content = completion.choices[0].message.content
        return {"success": True, "content": content}
    except Exception as exc:
        app.logger.error("Groq vision error: %s", traceback.format_exc())
        return {"success": False, "error": f"Groq vision error: {str(exc)}"}


def transcribe_audio(audio_bytes: bytes, filename: str = "audio.webm") -> dict:
    """Transcribe audio using Groq Whisper STT."""
    client = get_groq_client()
    if client is None:
        return {"success": False, "error": "Groq client unavailable."}
    try:
        transcription = client.audio.transcriptions.create(
            model=GROQ_STT_MODEL,
            file=(filename, io.BytesIO(audio_bytes)),
        )
        return {"success": True, "content": transcription.text}
    except Exception as exc:
        app.logger.error("Groq STT error: %s", traceback.format_exc())
        return {"success": False, "error": f"Speech-to-text error: {str(exc)}"}


# ---------------------------------------------------------------------------
# File extraction helpers
# ---------------------------------------------------------------------------
def extract_text_from_file(file_storage) -> dict:
    """Extract text from an uploaded TXT or PDF file."""
    filename  = file_storage.filename or ""
    ext       = os.path.splitext(filename)[1].lower()
    raw_bytes = file_storage.read()

    if len(raw_bytes) > MAX_UPLOAD_BYTES:
        return {"success": False,
                "error": f"File exceeds {MAX_UPLOAD_MB} MB limit."}

    if ext == ".txt":
        try:
            return {"success": True, "text": raw_bytes.decode("utf-8", errors="replace")}
        except Exception as exc:
            return {"success": False, "error": f"Could not read text file: {exc}"}

    if ext == ".pdf":
        if not PDF_AVAILABLE:
            return {"success": False,
                    "error": "PyPDF2 not installed. Run: pip install PyPDF2"}
        try:
            reader = PyPDF2.PdfReader(io.BytesIO(raw_bytes))
            pages  = [page.extract_text() or "" for page in reader.pages]
            text   = "\n".join(pages).strip()
            if not text:
                return {"success": False,
                        "error": "Could not extract text from PDF "
                                 "(scanned/image-only PDFs are not supported via text extraction)."}
            return {"success": True, "text": text}
        except Exception as exc:
            return {"success": False, "error": f"PDF parse error: {exc}"}

    return {"success": False,
            "error": f"Unsupported file type '{ext}'. "
                     f"Allowed: {', '.join(ALLOWED_TEXT_EXT)}"}


# ---------------------------------------------------------------------------
# Agent 1 – Study Material Agent
# ---------------------------------------------------------------------------
STUDY_MATERIAL_SYSTEM = """You are StudyGen AI's Study Material Agent – an expert educational assistant.
Your job is to help students understand their study material.
Always base your answers on the student's provided material when available.
If the answer is not covered by the provided material, clearly say so, then provide a general explanation.
Keep explanations student-friendly, clear, and structured.
Use bullet points, numbered lists, headings, and examples where appropriate."""


def study_material_agent(action: str, material: str, query: str = "") -> dict:
    """
    Route a study-material action to the Groq API.
    action: one of summary | keypoints | flashcards | concepts |
                      explain | conceptmap | qa
    """
    prompts = {
        "summary": (
            "Generate a concise, well-structured summary of the provided study material. "
            "Use clear headings and bullet points. Highlight the most important ideas."
        ),
        "keypoints": (
            "Extract the 10-15 most important key points from the study material. "
            "Format each as a short, memorable bullet point."
        ),
        "flashcards": (
            "Generate 10 Q&A flashcards from the study material. "
            "Format each exactly as:\nQ: [question]\nA: [answer]\n"
            "Make questions test understanding, not just recall."
        ),
        "concepts": (
            "Identify the most important concepts and topics in the study material. "
            "For each concept, give a one-sentence description."
        ),
        "explain": (
            f"Explain this topic in simple, student-friendly language: '{query}'. "
            "Use analogies and examples. Assume the student is seeing this for the first time."
        ),
        "conceptmap": (
            "Create a structured concept map / revision outline for this material. "
            "Use a hierarchical outline with main topics, subtopics, and key relationships. "
            "Format as indented text or numbered outline."
        ),
        "qa": (
            f"Answer this question based on the study material provided: '{query}'. "
            "If the answer is directly in the material, quote the relevant part. "
            "If not in the material, say 'This is not directly covered in your material' "
            "and provide a general answer."
        ),
    }

    if action not in prompts:
        return {"success": False, "error": f"Unknown action: {action}"}

    instruction = prompts[action]

    if material.strip():
        user_msg = (
            f"STUDY MATERIAL:\n{material[:8000]}\n\n"
            f"TASK: {instruction}"
        )
    else:
        if action in ("explain", "qa"):
            user_msg = f"TASK: {instruction}"
        else:
            return {"success": False,
                    "error": "Please provide study material first."}

    return generate_response(STUDY_MATERIAL_SYSTEM, user_msg)


def study_material_agent_image(user_query: str, image_b64: str, mime_type: str) -> dict:
    """Process a study image with the vision model."""
    system = (
        "You are StudyGen AI's Study Material Agent with vision capability. "
        "Analyze the provided study image (textbook page, diagram, notes, etc.) "
        "and respond to the student's query. Extract all visible text and diagrams. "
        "Be thorough and educational."
    )
    if not user_query.strip():
        user_query = (
            "Analyze this study material image. Extract all text, explain the content, "
            "identify key concepts, and provide a summary."
        )
    return generate_vision_response(system, user_query, image_b64, mime_type)


# ---------------------------------------------------------------------------
# Agent 2 – Study Planner & Evaluation Agent
# ---------------------------------------------------------------------------
PLANNER_SYSTEM = """You are StudyGen AI's Study Planner & Evaluation Agent.
You help students create personalized study plans and evaluate their exam readiness.
Be practical, specific, and encouraging. Structure plans clearly day-by-day.
When generating quizzes, produce well-formed JSON only – no extra commentary."""


def planner_agent_generate_plan(subject: str, exam_date: str,
                                 hours_per_day: int, prep_level: str,
                                 syllabus: str, important_topics: str) -> dict:
    """Generate a personalized day-by-day study plan."""
    today    = datetime.date.today()
    try:
        exam_dt  = datetime.datetime.strptime(exam_date, "%Y-%m-%d").date()
        days_left = max((exam_dt - today).days, 1)
    except ValueError:
        days_left = 14  # fallback

    user_msg = f"""
Subject: {subject}
Exam Date: {exam_date} ({days_left} days from today)
Study Hours Per Day: {hours_per_day}
Current Preparation Level: {prep_level}
Syllabus / Topics: {syllabus}
Important / Weak Topics: {important_topics or 'None specified'}

Generate a detailed day-by-day study plan for all {days_left} days (or up to 30 days if more).
For each day, specify:
- Day number and date
- Topics to cover
- Study activities (reading, practice problems, revision)
- Time allocation per topic
- Priority level (High/Medium/Low)

Also include:
- A recommended daily routine
- Revision days in the final week
- Mock test days if time permits
- Key focus areas based on preparation level

Format clearly with headings. Be specific and actionable.
"""
    return generate_response(PLANNER_SYSTEM, user_msg, temperature=0.5, max_tokens=4096)


def planner_agent_generate_quiz(subject: str, material: str,
                                 num_questions: int = 10) -> dict:
    """
    Generate MCQ quiz questions.
    Returns JSON array of question objects.
    """
    user_msg = f"""
Subject: {subject}
Study Material: {material[:6000] if material else 'General knowledge about ' + subject}

Generate exactly {num_questions} multiple-choice questions (MCQs) as a JSON array.
Each object must have exactly these keys:
  "question"    : the question text
  "options"     : array of exactly 4 option strings (e.g. ["A) ...", "B) ...", "C) ...", "D) ..."])
  "answer"      : the correct option string (must exactly match one of the options)
  "explanation" : a brief explanation of why the answer is correct
  "topic"       : the topic/concept this question tests

Return ONLY the JSON array. No markdown, no commentary, no ```json fences.
"""
    result = generate_response(PLANNER_SYSTEM, user_msg, temperature=0.4, max_tokens=3000)
    if not result["success"]:
        return result

    # Try to parse JSON from the response
    content = result["content"].strip()
    # Strip possible markdown fences
    if content.startswith("```"):
        lines = content.split("\n")
        content = "\n".join(lines[1:-1]) if len(lines) > 2 else content

    try:
        questions = json.loads(content)
        if not isinstance(questions, list):
            raise ValueError("Expected a JSON array")
        return {"success": True, "questions": questions}
    except (json.JSONDecodeError, ValueError) as exc:
        app.logger.warning("Quiz JSON parse error: %s\nRaw: %s", exc, content[:500])
        # Attempt to find JSON array in the text
        import re
        match = re.search(r'\[.*\]', content, re.DOTALL)
        if match:
            try:
                questions = json.loads(match.group())
                return {"success": True, "questions": questions}
            except Exception:
                pass
        return {"success": False,
                "error": "Could not parse quiz questions from AI response. "
                         "Please try again.",
                "raw": content[:1000]}


def planner_agent_evaluate_quiz(questions: list, answers: dict) -> dict:
    """
    Evaluate submitted quiz answers.
    questions: list of question dicts
    answers:   dict of {"0": "selected option", "1": ...}
    Returns score, percentage, weak topics, recommendations.
    """
    score        = 0
    total        = len(questions)
    results      = []
    weak_topics  = []

    for i, q in enumerate(questions):
        selected  = answers.get(str(i), "")
        correct   = q.get("answer", "")
        is_correct = selected.strip() == correct.strip()
        if is_correct:
            score += 1
        else:
            topic = q.get("topic", "General")
            if topic not in weak_topics:
                weak_topics.append(topic)

        results.append({
            "question":    q.get("question", ""),
            "options":     q.get("options", []),
            "selected":    selected,
            "correct":     correct,
            "is_correct":  is_correct,
            "explanation": q.get("explanation", ""),
            "topic":       q.get("topic", ""),
        })

    percentage = round((score / total) * 100, 1) if total > 0 else 0

    # Generate AI recommendations
    rec_prompt = (
        f"A student scored {score}/{total} ({percentage}%) on a quiz about '{answers.get('subject', 'the subject')}'. "
        f"Weak topics: {', '.join(weak_topics) if weak_topics else 'None – all correct!'}.\n\n"
        "Provide:\n"
        "1. A brief performance assessment (2-3 sentences)\n"
        "2. 3-5 specific revision recommendations\n"
        "3. A simple performance forecast (study readiness, NOT exam prediction)\n"
        "4. Focus alerts for topics needing attention\n"
        "Be encouraging but honest. Keep it concise."
    )
    rec_result = generate_response(PLANNER_SYSTEM, rec_prompt, temperature=0.6, max_tokens=800)

    return {
        "score":           score,
        "total":           total,
        "percentage":      percentage,
        "results":         results,
        "weak_topics":     weak_topics,
        "recommendations": rec_result.get("content", "") if rec_result["success"] else "",
    }


# ---------------------------------------------------------------------------
# Simple Orchestrator
# ---------------------------------------------------------------------------
def orchestrate(agent: str, action: str, payload: dict) -> dict:
    """
    Route requests to the correct agent.
    agent: 'study_material' | 'planner'
    action: agent-specific action string
    payload: dict of required parameters
    """
    if agent == "study_material":
        if action == "image":
            return study_material_agent_image(
                payload.get("query", ""),
                payload.get("image_b64", ""),
                payload.get("mime_type", "image/jpeg"),
            )
        return study_material_agent(
            action,
            payload.get("material", ""),
            payload.get("query",    ""),
        )

    if agent == "planner":
        if action == "generate_plan":
            return planner_agent_generate_plan(
                payload.get("subject",          ""),
                payload.get("exam_date",        ""),
                int(payload.get("hours_per_day", 3)),
                payload.get("prep_level",       "Beginner"),
                payload.get("syllabus",         ""),
                payload.get("important_topics", ""),
            )
        if action == "generate_quiz":
            return planner_agent_generate_quiz(
                payload.get("subject",  ""),
                payload.get("material", ""),
                int(payload.get("num_questions", 10)),
            )

    return {"success": False, "error": f"Unknown agent/action: {agent}/{action}"}


# ---------------------------------------------------------------------------
# HTML Templates (render_template_string approach)
# ---------------------------------------------------------------------------

# ---- Shared CSS & Nav -------------------------------------------------------
SHARED_CSS = """
<style>
  :root {
    --primary:   #3b82f6;
    --primary-d: #2563eb;
    --accent:    #7c3aed;
    --success:   #10b981;
    --warning:   #f59e0b;
    --danger:    #ef4444;
    --light-bg:  #f8fafc;
    --card-bg:   #ffffff;
    --border:    #e2e8f0;
    --text:      #1e293b;
    --muted:     #64748b;
    --lavender:  #ede9fe;
    --blue-soft: #eff6ff;
    --green-soft:#ecfdf5;
  }
  body { background: var(--light-bg); color: var(--text);
         font-family: -apple-system,"Segoe UI",system-ui,sans-serif;
         font-size: 15px; line-height: 1.65; }
  .navbar { background: #ffffff; border-bottom: 1px solid var(--border);
            box-shadow: 0 1px 3px rgba(0,0,0,.06); }
  .navbar-brand { font-weight: 700; font-size: 1.25rem; color: var(--primary) !important; }
  .nav-link { color: var(--text) !important; font-weight: 500; transition: color .2s; }
  .nav-link:hover, .nav-link.active { color: var(--primary) !important; }
  .card { border: 1px solid var(--border); border-radius: 14px;
          background: var(--card-bg); box-shadow: 0 1px 4px rgba(0,0,0,.04); }
  .card:hover { box-shadow: 0 4px 16px rgba(59,130,246,.08); }
  .btn-primary { background: var(--primary); border-color: var(--primary);
                 border-radius: 8px; font-weight: 600; transition: all .2s; }
  .btn-primary:hover { background: var(--primary-d); border-color: var(--primary-d);
                       transform: translateY(-1px); }
  .btn-outline-primary { border-color: var(--primary); color: var(--primary);
                         border-radius: 8px; font-weight: 600; }
  .btn-outline-primary:hover { background: var(--primary); color:#fff; }
  .badge-soft-primary { background: var(--blue-soft); color: var(--primary);
                        border-radius:6px; padding:3px 10px; font-size:.8rem; font-weight:600; }
  .badge-soft-purple  { background: var(--lavender); color: var(--accent);
                        border-radius:6px; padding:3px 10px; font-size:.8rem; font-weight:600; }
  .badge-soft-green   { background: var(--green-soft); color: var(--success);
                        border-radius:6px; padding:3px 10px; font-size:.8rem; font-weight:600; }
  .section-title { font-weight: 700; font-size: 1.5rem; color: var(--text); }
  .section-sub   { color: var(--muted); font-size: .95rem; }
  textarea.form-control, input.form-control, select.form-control {
    border-radius: 10px; border-color: var(--border); font-size:.93rem; }
  textarea.form-control:focus, input.form-control:focus, select.form-control:focus {
    border-color: var(--primary); box-shadow: 0 0 0 3px rgba(59,130,246,.12); }
  .ai-response-card { background: var(--blue-soft); border: 1px solid #bfdbfe;
                      border-radius: 12px; padding: 1.25rem 1.5rem; }
  .ai-response-card pre { white-space: pre-wrap; font-family: inherit; margin:0; }
  .spinner-wrap { display: none; text-align: center; padding: 2rem; }
  .spinner-wrap.active { display: block; }
  .hero-section { background: linear-gradient(135deg, #eff6ff 0%, #f5f3ff 100%);
                  border-radius: 20px; padding: 3rem 2rem; margin-bottom: 2rem; }
  .feature-icon { width: 48px; height: 48px; border-radius: 12px;
                  display: flex; align-items:center; justify-content:center;
                  font-size: 1.4rem; margin-bottom: .75rem; }
  .progress { height: 8px; border-radius: 4px; }
  .quiz-option { cursor: pointer; border-radius: 10px; padding: .6rem 1rem;
                 border: 2px solid var(--border); margin-bottom: .5rem;
                 transition: all .18s; background: #fff; }
  .quiz-option:hover   { border-color: var(--primary); background: var(--blue-soft); }
  .quiz-option.selected { border-color: var(--primary); background: var(--blue-soft); }
  .quiz-option.correct  { border-color: var(--success); background: var(--green-soft); color:#065f46; }
  .quiz-option.wrong    { border-color: var(--danger); background: #fef2f2; color:#991b1b; }
  .milestone-badge { display:inline-flex; align-items:center; gap:.4rem;
                     background: var(--green-soft); color:#065f46;
                     border-radius: 20px; padding: 4px 14px; font-size:.85rem; font-weight:600; }
  .toast-container { position: fixed; top: 1rem; right: 1rem; z-index: 9999; }
  footer { margin-top: 4rem; padding: 1.5rem 0; border-top: 1px solid var(--border);
           color: var(--muted); font-size: .85rem; text-align: center; }
  @media(max-width:768px){ .hero-section{ padding:2rem 1rem; } }
</style>
"""

def nav_html(active="home"):
    pages = [
        ("home",      "/",           "🏠 Home"),
        ("assistant", "/assistant",  "📚 Study Assistant"),
        ("quiz",      "/quiz",       "🧪 Quiz"),
        ("planner",   "/planner",    "📅 Study Planner"),
        ("dashboard", "/dashboard",  "📊 Dashboard"),
        ("about",     "/about",      "ℹ️ About"),
    ]
    items = ""
    for key, href, label in pages:
        cls = "nav-link active" if key == active else "nav-link"
        items += f'<li class="nav-item"><a class="{cls}" href="{href}">{label}</a></li>\n'
    return f"""
<nav class="navbar navbar-expand-lg sticky-top">
  <div class="container">
    <a class="navbar-brand" href="/">
      <span style="color:var(--primary)">📖</span> StudyGen AI
    </a>
    <button class="navbar-toggler" type="button" data-bs-toggle="collapse"
            data-bs-target="#navmenu">
      <span class="navbar-toggler-icon"></span>
    </button>
    <div class="collapse navbar-collapse" id="navmenu">
      <ul class="navbar-nav ms-auto gap-1">{items}</ul>
    </div>
  </div>
</nav>"""

TOAST_JS = """
function showToast(msg, type='info'){
  const colors = {info:'#3b82f6', success:'#10b981', error:'#ef4444', warning:'#f59e0b'};
  const div = document.createElement('div');
  div.className='alert mb-2 shadow-sm';
  div.style.cssText=`background:${colors[type]||colors.info};color:#fff;border-radius:10px;
    padding:.7rem 1.2rem;font-weight:500;min-width:260px;`;
  div.textContent=msg;
  document.getElementById('toast-container').appendChild(div);
  setTimeout(()=>div.remove(), 4500);
}
"""

def page_shell(title, active, body, extra_js=""):
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>{title} – StudyGen AI</title>
  <link rel="stylesheet"
        href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css">
  {SHARED_CSS}
</head>
<body>
{nav_html(active)}
<div id="toast-container" class="toast-container"></div>
<div class="container py-4">
{body}
</div>
<footer>
  <p>StudyGen AI &copy; 2025 &mdash; Powered by <strong>Groq API</strong> &amp;
     Open-Source LLMs &mdash; Built with Flask &amp; Bootstrap 5</p>
</footer>
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"></script>
<script>
{TOAST_JS}
{extra_js}
</script>
</body>
</html>"""


# ===========================================================================
# PAGE: Home
# ===========================================================================
HOME_BODY = """
<div class="hero-section text-center mb-4">
  <span style="font-size:3rem">📖</span>
  <h1 class="fw-bold mt-2" style="font-size:2.4rem">StudyGen AI</h1>
  <p class="lead text-muted mb-4">
    Your AI-powered Smart Study Generator.<br>
    Transform notes, textbooks, and lecture material into structured learning experiences.
  </p>
  <a href="/assistant" class="btn btn-primary btn-lg px-5 me-2">
    🚀 Start Studying
  </a>
  <a href="/planner" class="btn btn-outline-primary btn-lg px-4">
    📅 Plan My Study
  </a>
</div>

<div class="row g-4 mb-4">
  <div class="col-md-4">
    <div class="card p-4 h-100">
      <div class="feature-icon bg-blue-soft" style="background:var(--blue-soft)">📝</div>
      <h5 class="fw-bold">Study Material Agent</h5>
      <p class="text-muted">
        Paste notes, upload PDFs/TXT files, or even a photo of your textbook.
        Get summaries, key points, flashcards, concept maps, and simple explanations instantly.
      </p>
      <a href="/assistant" class="btn btn-outline-primary btn-sm mt-auto">Open Assistant →</a>
    </div>
  </div>
  <div class="col-md-4">
    <div class="card p-4 h-100">
      <div class="feature-icon" style="background:var(--lavender)">🧪</div>
      <h5 class="fw-bold">MCQ Quiz & Evaluation</h5>
      <p class="text-muted">
        Generate practice quizzes from your material. Submit answers, get your score,
        see explanations, and identify weak topics to focus revision.
      </p>
      <a href="/quiz" class="btn btn-outline-primary btn-sm mt-auto">Take a Quiz →</a>
    </div>
  </div>
  <div class="col-md-4">
    <div class="card p-4 h-100">
      <div class="feature-icon" style="background:var(--green-soft)">📅</div>
      <h5 class="fw-bold">Personalized Study Planner</h5>
      <p class="text-muted">
        Enter your exam date, syllabus, and daily study hours.
        Get a day-by-day personalized study plan with topic prioritization.
      </p>
      <a href="/planner" class="btn btn-outline-primary btn-sm mt-auto">Create Plan →</a>
    </div>
  </div>
</div>

<div class="row g-4 mb-4">
  <div class="col-md-6">
    <div class="card p-4 h-100">
      <h5 class="fw-bold">🎯 What Problem Does It Solve?</h5>
      <ul class="text-muted mt-2">
        <li>Students struggle to organize large amounts of study material</li>
        <li>Identifying what to study first before exams is stressful</li>
        <li>Self-testing and identifying weak areas is time-consuming</li>
        <li>Getting personalized guidance without a tutor is difficult</li>
      </ul>
      <p class="text-muted">
        <strong>StudyGen AI</strong> solves all of this with two intelligent agents
        working together to guide you from raw notes to exam readiness.
      </p>
    </div>
  </div>
  <div class="col-md-6">
    <div class="card p-4 h-100">
      <h5 class="fw-bold">⚡ AI Workflow</h5>
      <div class="mt-2" style="font-size:.93rem; color:var(--muted)">
        <div class="d-flex align-items-center gap-2 mb-1">
          <span class="badge-soft-primary">1</span> Upload notes / image / speak
        </div>
        <div class="d-flex align-items-center gap-2 mb-1">
          <span class="badge-soft-primary">2</span> Flask backend + Orchestrator
        </div>
        <div class="d-flex align-items-center gap-2 mb-1">
          <span class="badge-soft-purple">3</span> Study Material Agent <em>or</em> Planner Agent
        </div>
        <div class="d-flex align-items-center gap-2 mb-1">
          <span class="badge-soft-primary">4</span> Groq API → Open-source LLM
        </div>
        <div class="d-flex align-items-center gap-2">
          <span class="badge-soft-green">5</span> Structured result → Your dashboard
        </div>
      </div>
    </div>
  </div>
</div>

<div class="card p-4 text-center" style="background:linear-gradient(135deg,#eff6ff,#f5f3ff)">
  <h5 class="fw-bold">📊 Track Your Progress</h5>
  <p class="text-muted mb-3">
    Quiz scores, weak topics, study milestones, and upcoming tasks — all in one place.
  </p>
  <a href="/dashboard" class="btn btn-primary">View Dashboard →</a>
</div>
"""

@app.route("/")
def home():
    return page_shell("Home", "home", HOME_BODY)


# ===========================================================================
# PAGE: Study Assistant
# ===========================================================================
ASSISTANT_BODY = """
<div class="row mb-3">
  <div class="col">
    <h2 class="section-title">📚 Study Assistant</h2>
    <p class="section-sub">
      Paste your notes, upload a file, or upload a textbook image.
      Then choose what you want the AI to do.
    </p>
  </div>
</div>

<div class="row g-4">
  <!-- Left: Input Panel -->
  <div class="col-lg-5">
    <div class="card p-4 mb-3">
      <h6 class="fw-bold mb-3">📋 Study Material Input</h6>

      <!-- Tabs -->
      <ul class="nav nav-tabs mb-3" id="inputTabs">
        <li class="nav-item">
          <button class="nav-link active" data-bs-toggle="tab" data-bs-target="#tab-text">
            ✏️ Text
          </button>
        </li>
        <li class="nav-item">
          <button class="nav-link" data-bs-toggle="tab" data-bs-target="#tab-file">
            📄 File
          </button>
        </li>
        <li class="nav-item">
          <button class="nav-link" data-bs-toggle="tab" data-bs-target="#tab-image">
            🖼️ Image
          </button>
        </li>
        <li class="nav-item">
          <button class="nav-link" data-bs-toggle="tab" data-bs-target="#tab-voice">
            🎙️ Voice
          </button>
        </li>
      </ul>

      <div class="tab-content">
        <!-- Text Tab -->
        <div class="tab-pane fade show active" id="tab-text">
          <textarea id="materialText" class="form-control" rows="8"
            placeholder="Paste your notes, textbook content, or any study material here…"></textarea>
          <button class="btn btn-sm btn-outline-secondary mt-2"
                  onclick="clearMaterial()">Clear</button>
        </div>

        <!-- File Tab -->
        <div class="tab-pane fade" id="tab-file">
          <div class="border rounded-3 p-3 text-center" style="border-style:dashed!important">
            <p class="text-muted mb-2">Upload a TXT or PDF file (max 10 MB)</p>
            <input type="file" id="fileInput" class="form-control mb-2"
                   accept=".txt,.pdf">
            <button class="btn btn-primary btn-sm" onclick="uploadFile()">
              📤 Upload & Extract
            </button>
          </div>
          <div id="fileStatus" class="mt-2"></div>
        </div>

        <!-- Image Tab -->
        <div class="tab-pane fade" id="tab-image">
          <div class="border rounded-3 p-3 text-center" style="border-style:dashed!important">
            <p class="text-muted mb-2">Upload a textbook / notes image (PNG, JPG, WEBP)</p>
            <input type="file" id="imageInput" class="form-control mb-2"
                   accept=".png,.jpg,.jpeg,.webp">
            <input type="text" id="imageQuery" class="form-control mb-2 mt-1"
                   placeholder="What do you want to know about this image? (optional)">
            <button class="btn btn-primary btn-sm" onclick="analyzeImage()">
              🔍 Analyze Image
            </button>
          </div>
          <div id="imageStatus" class="mt-2"></div>
        </div>

        <!-- Voice Tab -->
        <div class="tab-pane fade" id="tab-voice">
          <div class="text-center p-3">
            <p class="text-muted mb-3">
              Click Record to capture a voice query or study question.
              Browser speech recognition is used where supported.
            </p>
            <button class="btn btn-primary mb-2" id="voiceBtn" onclick="startVoice()">
              🎙️ Start Recording
            </button>
            <button class="btn btn-secondary mb-2 d-none" id="stopVoiceBtn"
                    onclick="stopVoice()">
              ⏹ Stop Recording
            </button>
            <p id="voiceStatus" class="text-muted small"></p>
            <textarea id="voiceText" class="form-control mt-2" rows="3"
              placeholder="Transcribed voice query appears here…"></textarea>
            <button class="btn btn-outline-primary btn-sm mt-2"
                    onclick="useVoiceText()">
              ✅ Use as Query
            </button>
          </div>
        </div>
      </div>
    </div>

    <!-- Quick Actions -->
    <div class="card p-4">
      <h6 class="fw-bold mb-3">⚡ Quick Actions</h6>
      <div class="d-grid gap-2">
        <button class="btn btn-outline-primary" onclick="runAction('summary')">
          📝 Generate Summary
        </button>
        <button class="btn btn-outline-primary" onclick="runAction('keypoints')">
          🔑 Extract Key Points
        </button>
        <button class="btn btn-outline-primary" onclick="runAction('flashcards')">
          🃏 Generate Flashcards
        </button>
        <button class="btn btn-outline-primary" onclick="runAction('concepts')">
          💡 Identify Key Concepts
        </button>
        <button class="btn btn-outline-primary" onclick="runAction('conceptmap')">
          🗺️ Create Concept Map
        </button>
      </div>
    </div>
  </div>

  <!-- Right: Q&A + Response -->
  <div class="col-lg-7">
    <!-- Q&A -->
    <div class="card p-4 mb-3">
      <h6 class="fw-bold mb-3">❓ Ask a Question / Get Explanation</h6>
      <textarea id="queryInput" class="form-control mb-2" rows="3"
        placeholder="Ask anything about your material, or request an explanation of a concept…">
      </textarea>
      <div class="d-flex gap-2">
        <button class="btn btn-primary flex-fill" onclick="runAction('qa')">
          💬 Answer Question
        </button>
        <button class="btn btn-outline-primary flex-fill" onclick="runAction('explain')">
          🧠 Explain Topic
        </button>
      </div>
    </div>

    <!-- Spinner -->
    <div id="spinner" class="spinner-wrap">
      <div class="spinner-border text-primary" style="width:2.5rem;height:2.5rem"></div>
      <p class="text-muted mt-2">AI is thinking…</p>
    </div>

    <!-- AI Response -->
    <div id="responseArea" class="d-none">
      <div class="d-flex justify-content-between align-items-center mb-2">
        <h6 class="fw-bold mb-0" id="responseTitle">AI Response</h6>
        <span class="badge-soft-primary" id="actionBadge"></span>
      </div>
      <div class="ai-response-card">
        <pre id="responseText" style="font-size:.9rem"></pre>
      </div>
      <div class="mt-2 text-end">
        <button class="btn btn-sm btn-outline-secondary" onclick="copyResponse()">
          📋 Copy
        </button>
        <button class="btn btn-sm btn-outline-secondary ms-1" onclick="clearResponse()">
          🗑️ Clear
        </button>
      </div>
    </div>

    <!-- Empty state -->
    <div id="emptyState" class="text-center py-5 text-muted">
      <div style="font-size:3rem">🤖</div>
      <p class="mt-2">AI responses will appear here.<br>
         Provide study material and choose an action to get started.</p>
    </div>
  </div>
</div>
"""

ASSISTANT_JS = r"""
let studyMaterial = "";

function getMaterial() {
  return document.getElementById('materialText').value.trim();
}
function clearMaterial() {
  document.getElementById('materialText').value = '';
  studyMaterial = '';
}
function showSpinner(show) {
  document.getElementById('spinner').classList.toggle('active', show);
  document.getElementById('emptyState').classList.toggle('d-none', show);
  if (show) document.getElementById('responseArea').classList.add('d-none');
}
function showResponse(title, badge, text) {
  document.getElementById('responseTitle').textContent = title;
  document.getElementById('actionBadge').textContent = badge;
  document.getElementById('responseText').textContent = text;
  document.getElementById('responseArea').classList.remove('d-none');
  document.getElementById('emptyState').classList.add('d-none');
  document.getElementById('spinner').classList.remove('active');
}
function clearResponse() {
  document.getElementById('responseArea').classList.add('d-none');
  document.getElementById('emptyState').classList.remove('d-none');
}
function copyResponse() {
  const t = document.getElementById('responseText').textContent;
  navigator.clipboard.writeText(t).then(() => showToast('Copied!', 'success'));
}

const ACTION_LABELS = {
  summary:'Summary', keypoints:'Key Points', flashcards:'Flashcards',
  concepts:'Key Concepts', conceptmap:'Concept Map', qa:'Q&A Answer', explain:'Explanation'
};

function runAction(action) {
  const material = getMaterial();
  const query    = document.getElementById('queryInput').value.trim();
  if (!material && !['qa','explain'].includes(action)) {
    showToast('Please provide study material first.', 'warning'); return;
  }
  if (['qa','explain'].includes(action) && !query) {
    showToast('Please enter a question or topic.', 'warning'); return;
  }
  showSpinner(true);
  fetch('/api/study-material', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({action, material, query})
  })
  .then(r => r.json())
  .then(data => {
    showSpinner(false);
    if (data.success) {
      showResponse(ACTION_LABELS[action] || 'AI Response', '📖 Study Material Agent', data.content);
    } else {
      showToast('Error: ' + data.error, 'error');
      document.getElementById('spinner').classList.remove('active');
      document.getElementById('emptyState').classList.remove('d-none');
    }
  })
  .catch(e => {
    showSpinner(false);
    showToast('Network error: ' + e.message, 'error');
    document.getElementById('emptyState').classList.remove('d-none');
  });
}

function uploadFile() {
  const f = document.getElementById('fileInput').files[0];
  if (!f) { showToast('Please select a file first.','warning'); return; }
  const fd = new FormData();
  fd.append('file', f);
  document.getElementById('fileStatus').innerHTML =
    '<span class="text-primary">Extracting text…</span>';
  fetch('/api/upload-file', {method:'POST', body:fd})
  .then(r=>r.json())
  .then(data=>{
    if(data.success){
      document.getElementById('materialText').value = data.text;
      // Switch to text tab
      document.querySelector('[data-bs-target="#tab-text"]').click();
      document.getElementById('fileStatus').innerHTML =
        '<span class="text-success">✅ Text extracted successfully!</span>';
      showToast('File uploaded and text extracted!', 'success');
    } else {
      document.getElementById('fileStatus').innerHTML =
        `<span class="text-danger">❌ ${data.error}</span>`;
      showToast(data.error, 'error');
    }
  })
  .catch(e=>{
    document.getElementById('fileStatus').innerHTML =
      `<span class="text-danger">❌ ${e.message}</span>`;
  });
}

function analyzeImage() {
  const f = document.getElementById('imageInput').files[0];
  if (!f) { showToast('Please select an image first.','warning'); return; }
  const query = document.getElementById('imageQuery').value.trim();
  const reader = new FileReader();
  reader.onload = function(e) {
    const b64 = e.target.result.split(',')[1];
    const mime = f.type || 'image/jpeg';
    showSpinner(true);
    document.getElementById('imageStatus').innerHTML =
      '<span class="text-primary">Analyzing image with vision AI…</span>';
    fetch('/api/analyze-image', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({image_b64:b64, mime_type:mime, query})
    })
    .then(r=>r.json())
    .then(data=>{
      showSpinner(false);
      if(data.success){
        showResponse('Image Analysis','🖼️ Vision Model', data.content);
        document.getElementById('imageStatus').innerHTML =
          '<span class="text-success">✅ Analysis complete</span>';
      } else {
        showToast('Image error: '+data.error,'error');
        document.getElementById('emptyState').classList.remove('d-none');
        document.getElementById('imageStatus').innerHTML =
          `<span class="text-danger">❌ ${data.error}</span>`;
      }
    });
  };
  reader.readAsDataURL(f);
}

// Voice input using Web Speech API
let recognition = null;
let mediaRecorder = null;
let audioChunks = [];

function startVoice() {
  if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    recognition = new SR();
    recognition.continuous = false;
    recognition.interimResults = true;
    recognition.lang = 'en-US';
    recognition.onstart = () => {
      document.getElementById('voiceStatus').textContent = '🔴 Recording… speak now';
      document.getElementById('voiceBtn').classList.add('d-none');
      document.getElementById('stopVoiceBtn').classList.remove('d-none');
    };
    recognition.onresult = (e) => {
      let transcript = '';
      for (let i=0; i<e.results.length; i++) transcript += e.results[i][0].transcript;
      document.getElementById('voiceText').value = transcript;
    };
    recognition.onerror = (e) => {
      document.getElementById('voiceStatus').textContent = 'Error: ' + e.error;
      resetVoiceBtns();
    };
    recognition.onend = () => {
      document.getElementById('voiceStatus').textContent = '✅ Recording stopped';
      resetVoiceBtns();
    };
    recognition.start();
  } else {
    // Fallback: record and send to Groq STT
    startMediaRecording();
  }
}
function stopVoice() {
  if (recognition) { recognition.stop(); recognition = null; }
  if (mediaRecorder && mediaRecorder.state !== 'inactive') { mediaRecorder.stop(); }
  resetVoiceBtns();
}
function resetVoiceBtns() {
  document.getElementById('voiceBtn').classList.remove('d-none');
  document.getElementById('stopVoiceBtn').classList.add('d-none');
}
function startMediaRecording() {
  navigator.mediaDevices.getUserMedia({audio:true})
  .then(stream => {
    audioChunks = [];
    mediaRecorder = new MediaRecorder(stream);
    mediaRecorder.ondataavailable = e => audioChunks.push(e.data);
    mediaRecorder.onstop = () => {
      const blob = new Blob(audioChunks, {type:'audio/webm'});
      const fd = new FormData();
      fd.append('audio', blob, 'recording.webm');
      document.getElementById('voiceStatus').textContent = '⏳ Transcribing with Groq Whisper…';
      fetch('/api/transcribe-audio', {method:'POST', body:fd})
      .then(r=>r.json())
      .then(data=>{
        if(data.success){
          document.getElementById('voiceText').value = data.content;
          document.getElementById('voiceStatus').textContent = '✅ Transcription complete';
        } else {
          document.getElementById('voiceStatus').textContent = 'Error: '+data.error;
        }
      });
    };
    mediaRecorder.start();
    document.getElementById('voiceStatus').textContent = '🔴 Recording (media recorder)…';
    document.getElementById('voiceBtn').classList.add('d-none');
    document.getElementById('stopVoiceBtn').classList.remove('d-none');
  })
  .catch(e => {
    document.getElementById('voiceStatus').textContent =
      'Microphone access denied or not available: ' + e.message;
  });
}
function useVoiceText() {
  const t = document.getElementById('voiceText').value.trim();
  if (!t) { showToast('No voice text to use.','warning'); return; }
  document.getElementById('queryInput').value = t;
  showToast('Voice query added to question field!','success');
}
"""


@app.route("/assistant")
def assistant():
    body = ASSISTANT_BODY
    return page_shell("Study Assistant", "assistant", body, ASSISTANT_JS)


# ===========================================================================
# PAGE: Quiz
# ===========================================================================
QUIZ_BODY = """
<div class="row mb-3">
  <div class="col">
    <h2 class="section-title">🧪 Practice Quiz</h2>
    <p class="section-sub">Generate MCQ quizzes from your study material and evaluate your knowledge.</p>
  </div>
</div>

<div id="quizSetup">
  <div class="card p-4 mb-4">
    <h6 class="fw-bold mb-3">⚙️ Quiz Settings</h6>
    <div class="row g-3">
      <div class="col-md-6">
        <label class="form-label fw-semibold">Subject / Topic</label>
        <input type="text" id="quizSubject" class="form-control"
               placeholder="e.g. Python Programming, Organic Chemistry">
      </div>
      <div class="col-md-3">
        <label class="form-label fw-semibold">Number of Questions</label>
        <select id="numQuestions" class="form-control">
          <option value="5">5 Questions</option>
          <option value="10" selected>10 Questions</option>
          <option value="15">15 Questions</option>
        </select>
      </div>
      <div class="col-md-3">
        <label class="form-label fw-semibold">&nbsp;</label>
        <button class="btn btn-primary w-100 d-block" onclick="generateQuiz()">
          ⚡ Generate Quiz
        </button>
      </div>
    </div>
    <div class="mt-3">
      <label class="form-label fw-semibold">
        Study Material (optional – for targeted questions)
      </label>
      <textarea id="quizMaterial" class="form-control" rows="4"
        placeholder="Paste relevant notes here for more targeted questions (optional)…">
      </textarea>
    </div>
  </div>

  <div id="quizSpinner" class="spinner-wrap">
    <div class="spinner-border text-primary" style="width:2.5rem;height:2.5rem"></div>
    <p class="text-muted mt-2">Generating quiz questions…</p>
  </div>
</div>

<!-- Quiz Questions (hidden until generated) -->
<div id="quizContainer" class="d-none">
  <div class="d-flex justify-content-between align-items-center mb-3">
    <h5 class="fw-bold mb-0" id="quizTitle">Quiz</h5>
    <span class="badge-soft-purple" id="quizProgress">0 / 0 answered</span>
  </div>
  <div id="questionsContainer"></div>
  <div class="text-center mt-4">
    <button class="btn btn-primary btn-lg px-5" onclick="submitQuiz()">
      📤 Submit Quiz
    </button>
    <button class="btn btn-outline-secondary btn-lg px-4 ms-2" onclick="resetQuiz()">
      🔄 New Quiz
    </button>
  </div>
</div>

<!-- Results (hidden until submitted) -->
<div id="resultsContainer" class="d-none">
  <div class="card p-4 mb-4" id="scoreCard">
    <div class="text-center mb-3">
      <h4 class="fw-bold">Quiz Results</h4>
      <div style="font-size:3.5rem;font-weight:800;color:var(--primary)" id="scoreDisplay">
        0%
      </div>
      <p class="text-muted" id="scoreDetail">0 / 0 correct</p>
    </div>
    <div class="progress mb-3" style="height:12px">
      <div class="progress-bar" id="scoreBar" style="width:0%"></div>
    </div>
    <div id="weakTopicsSection" class="d-none">
      <h6 class="fw-bold">⚠️ Weak Topics</h6>
      <div id="weakTopicsList" class="mb-2"></div>
    </div>
  </div>

  <div id="recommendationsCard" class="card p-4 mb-4 d-none">
    <h6 class="fw-bold mb-2">💡 AI Recommendations</h6>
    <pre id="recommendationsText"
         style="white-space:pre-wrap;font-family:inherit;font-size:.9rem;color:var(--muted)">
    </pre>
  </div>

  <h5 class="fw-bold mb-3">📋 Detailed Results</h5>
  <div id="detailedResults"></div>
  <div class="text-center mt-4">
    <button class="btn btn-primary px-4 me-2" onclick="resetQuiz()">🔄 New Quiz</button>
    <button class="btn btn-outline-primary px-4" onclick="saveQuizResults()">
      💾 Save to Dashboard
    </button>
  </div>
</div>
"""

QUIZ_JS = r"""
let quizQuestions = [];
let userAnswers   = {};
let currentSubject = '';
let savedEvaluation = null;

function generateQuiz() {
  const subject = document.getElementById('quizSubject').value.trim();
  const material = document.getElementById('quizMaterial').value.trim();
  const num     = document.getElementById('numQuestions').value;
  if (!subject) { showToast('Please enter a subject/topic.','warning'); return; }

  currentSubject = subject;
  document.getElementById('quizSpinner').classList.add('active');

  fetch('/api/generate-quiz', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({subject, material, num_questions:parseInt(num)})
  })
  .then(r=>r.json())
  .then(data=>{
    document.getElementById('quizSpinner').classList.remove('active');
    if(data.success){
      quizQuestions = data.questions;
      userAnswers   = {};
      renderQuiz();
    } else {
      showToast('Quiz generation failed: '+data.error,'error');
    }
  })
  .catch(e=>{
    document.getElementById('quizSpinner').classList.remove('active');
    showToast('Error: '+e.message,'error');
  });
}

function renderQuiz() {
  document.getElementById('quizSetup').classList.add('d-none');
  document.getElementById('resultsContainer').classList.add('d-none');
  document.getElementById('quizContainer').classList.remove('d-none');
  document.getElementById('quizTitle').textContent =
    '📝 ' + currentSubject + ' Quiz (' + quizQuestions.length + ' Questions)';
  updateProgress();

  const container = document.getElementById('questionsContainer');
  container.innerHTML = '';
  quizQuestions.forEach((q,i) => {
    const card = document.createElement('div');
    card.className = 'card p-4 mb-3';
    card.id = 'q-card-'+i;
    const topicBadge = q.topic
      ? `<span class="badge-soft-primary" style="font-size:.75rem">${q.topic}</span>` : '';
    let optHtml = q.options.map(opt => `
      <div class="quiz-option" onclick="selectAnswer(${i}, '${escQ(opt)}')"
           id="opt-${i}-${escId(opt)}">
        ${opt}
      </div>`).join('');
    card.innerHTML = `
      <div class="d-flex justify-content-between mb-2">
        <span class="fw-semibold">Q${i+1}. ${q.question}</span>
        ${topicBadge}
      </div>
      <div id="options-${i}">${optHtml}</div>
    `;
    container.appendChild(card);
  });
}

function escQ(s){ return s.replace(/'/g,"&#39;"); }
function escId(s){ return s.replace(/[^a-zA-Z0-9]/g,'_'); }

function selectAnswer(qIdx, option) {
  userAnswers[qIdx] = option;
  // Clear previously selected
  const opts = document.querySelectorAll(`#options-${qIdx} .quiz-option`);
  opts.forEach(o => o.classList.remove('selected'));
  // Mark selected
  opts.forEach(o => {
    if(o.textContent.trim() === option.trim()) o.classList.add('selected');
  });
  updateProgress();
}

function updateProgress() {
  const answered = Object.keys(userAnswers).length;
  document.getElementById('quizProgress').textContent =
    answered + ' / ' + quizQuestions.length + ' answered';
}

function submitQuiz() {
  if(Object.keys(userAnswers).length < quizQuestions.length){
    if(!confirm('You have unanswered questions. Submit anyway?')) return;
  }
  const strAnswers = {};
  Object.keys(userAnswers).forEach(k => strAnswers[String(k)] = userAnswers[k]);
  strAnswers['subject'] = currentSubject;

  fetch('/api/evaluate-quiz', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({questions:quizQuestions, answers:strAnswers})
  })
  .then(r=>r.json())
  .then(data=>{
    if(data.success){
      savedEvaluation = data;
      renderResults(data);
    } else {
      showToast('Evaluation error: '+data.error,'error');
    }
  });
}

function renderResults(data) {
  document.getElementById('quizContainer').classList.add('d-none');
  document.getElementById('resultsContainer').classList.remove('d-none');
  document.getElementById('scoreDisplay').textContent = data.percentage + '%';
  document.getElementById('scoreDetail').textContent =
    data.score + ' / ' + data.total + ' correct';
  document.getElementById('scoreBar').style.width = data.percentage + '%';

  const bar = document.getElementById('scoreBar');
  if(data.percentage >= 70) bar.style.background = 'var(--success)';
  else if(data.percentage >= 40) bar.style.background = 'var(--warning)';
  else bar.style.background = 'var(--danger)';

  const scoreCard = document.getElementById('scoreCard');
  if(data.percentage >= 70) scoreCard.style.background='var(--green-soft)';
  else if(data.percentage>=40) scoreCard.style.background='#fffbeb';
  else scoreCard.style.background='#fef2f2';

  if(data.weak_topics && data.weak_topics.length > 0){
    document.getElementById('weakTopicsSection').classList.remove('d-none');
    document.getElementById('weakTopicsList').innerHTML =
      data.weak_topics.map(t=>
        `<span class="badge bg-warning text-dark me-1 mb-1">${t}</span>`
      ).join('');
  }

  if(data.recommendations){
    document.getElementById('recommendationsCard').classList.remove('d-none');
    document.getElementById('recommendationsText').textContent = data.recommendations;
  }

  const dr = document.getElementById('detailedResults');
  dr.innerHTML = '';
  data.results.forEach((r,i)=>{
    const div = document.createElement('div');
    div.className = 'card p-3 mb-2';
    const icon = r.is_correct ? '✅' : '❌';
    const bg   = r.is_correct ? 'var(--green-soft)' : '#fef2f2';
    const optsHtml = r.options.map(o=>{
      let cls = '';
      const isCorrect  = o.trim()===r.correct.trim();
      const isSelected = o.trim()===r.selected.trim();
      if(isCorrect)  cls='quiz-option correct';
      else if(isSelected && !isCorrect) cls='quiz-option wrong';
      else cls='quiz-option';
      return `<div class="${cls}" style="cursor:default">${o}</div>`;
    }).join('');
    div.style.background = bg;
    div.innerHTML = `
      <div class="fw-semibold mb-2">${icon} Q${i+1}. ${r.question}</div>
      ${optsHtml}
      ${!r.is_correct ? `
        <div class="mt-2 p-2 rounded" style="background:#fff;border-left:3px solid var(--primary)">
          <strong>📖 Explanation:</strong> ${r.explanation}
        </div>` : ''}
    `;
    dr.appendChild(div);
  });
}

function resetQuiz() {
  quizQuestions = [];
  userAnswers   = {};
  savedEvaluation = null;
  document.getElementById('quizContainer').classList.add('d-none');
  document.getElementById('resultsContainer').classList.add('d-none');
  document.getElementById('quizSetup').classList.remove('d-none');
  document.getElementById('quizSpinner').classList.remove('active');
}

function saveQuizResults() {
  if(!savedEvaluation) return;
  fetch('/api/save-quiz', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({
      subject: currentSubject,
      score: savedEvaluation.score,
      total: savedEvaluation.total,
      percentage: savedEvaluation.percentage,
      weak_topics: savedEvaluation.weak_topics
    })
  })
  .then(r=>r.json())
  .then(data=>{
    if(data.success) showToast('Results saved to dashboard!','success');
    else showToast('Save error: '+data.error,'error');
  });
}
"""

@app.route("/quiz")
def quiz():
    return page_shell("Quiz", "quiz", QUIZ_BODY, QUIZ_JS)


# ===========================================================================
# PAGE: Study Planner
# ===========================================================================
PLANNER_BODY = """
<div class="row mb-3">
  <div class="col">
    <h2 class="section-title">📅 Study Planner</h2>
    <p class="section-sub">
      Enter your exam details and get a personalized day-by-day study plan.
    </p>
  </div>
</div>

<div class="row g-4">
  <div class="col-lg-5">
    <div class="card p-4">
      <h6 class="fw-bold mb-3">📋 Your Study Details</h6>
      <div class="mb-3">
        <label class="form-label fw-semibold">Subject / Course</label>
        <input type="text" id="planSubject" class="form-control"
               placeholder="e.g. Data Structures, Biology, History">
      </div>
      <div class="mb-3">
        <label class="form-label fw-semibold">Exam Date</label>
        <input type="date" id="planExamDate" class="form-control">
      </div>
      <div class="mb-3">
        <label class="form-label fw-semibold">Study Hours Available Per Day</label>
        <select id="planHours" class="form-control">
          <option value="1">1 hour</option>
          <option value="2">2 hours</option>
          <option value="3" selected>3 hours</option>
          <option value="4">4 hours</option>
          <option value="5">5 hours</option>
          <option value="6">6+ hours</option>
        </select>
      </div>
      <div class="mb-3">
        <label class="form-label fw-semibold">Current Preparation Level</label>
        <select id="planLevel" class="form-control">
          <option value="Beginner">Beginner – Just starting</option>
          <option value="Intermediate" selected>Intermediate – Know basics</option>
          <option value="Advanced">Advanced – Mostly prepared</option>
        </select>
      </div>
      <div class="mb-3">
        <label class="form-label fw-semibold">Syllabus / Topics to Cover</label>
        <textarea id="planSyllabus" class="form-control" rows="5"
          placeholder="List all topics, chapters, or units you need to study…">
        </textarea>
      </div>
      <div class="mb-3">
        <label class="form-label fw-semibold">Important / Weak Topics (optional)</label>
        <input type="text" id="planImportant" class="form-control"
               placeholder="Topics you find difficult or that carry more marks">
      </div>
      <button class="btn btn-primary w-100" onclick="generatePlan()">
        🚀 Generate My Study Plan
      </button>
    </div>
  </div>

  <div class="col-lg-7">
    <div id="planSpinner" class="spinner-wrap">
      <div class="spinner-border text-primary" style="width:2.5rem;height:2.5rem"></div>
      <p class="text-muted mt-2">Creating your personalized study plan…</p>
    </div>

    <div id="planEmpty" class="text-center py-5 text-muted">
      <div style="font-size:3rem">📅</div>
      <p class="mt-2">Your personalized study plan will appear here.<br>
         Fill in the details and click Generate.</p>
    </div>

    <div id="planResult" class="d-none">
      <div class="d-flex justify-content-between align-items-center mb-3">
        <h5 class="fw-bold mb-0">Your Study Plan</h5>
        <div>
          <button class="btn btn-sm btn-outline-primary me-1" onclick="savePlan()">
            💾 Save Plan
          </button>
          <button class="btn btn-sm btn-outline-secondary" onclick="clearPlan()">
            🗑️ Clear
          </button>
        </div>
      </div>
      <div class="ai-response-card">
        <pre id="planText" style="white-space:pre-wrap;font-family:inherit;font-size:.9rem"></pre>
      </div>
    </div>
  </div>
</div>
"""

PLANNER_JS = r"""
let lastPlanData = null;

function generatePlan() {
  const subject   = document.getElementById('planSubject').value.trim();
  const examDate  = document.getElementById('planExamDate').value;
  const hours     = document.getElementById('planHours').value;
  const level     = document.getElementById('planLevel').value;
  const syllabus  = document.getElementById('planSyllabus').value.trim();
  const important = document.getElementById('planImportant').value.trim();

  if (!subject)  { showToast('Please enter a subject.','warning'); return; }
  if (!examDate) { showToast('Please select an exam date.','warning'); return; }
  if (!syllabus) { showToast('Please enter your syllabus/topics.','warning'); return; }

  lastPlanData = {subject, exam_date:examDate, hours_per_day:parseInt(hours),
                  prep_level:level, syllabus, important_topics:important};

  document.getElementById('planSpinner').classList.add('active');
  document.getElementById('planEmpty').classList.add('d-none');
  document.getElementById('planResult').classList.add('d-none');

  fetch('/api/generate-plan', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify(lastPlanData)
  })
  .then(r=>r.json())
  .then(data=>{
    document.getElementById('planSpinner').classList.remove('active');
    if(data.success){
      document.getElementById('planText').textContent = data.content;
      document.getElementById('planResult').classList.remove('d-none');
    } else {
      showToast('Plan generation failed: '+data.error,'error');
      document.getElementById('planEmpty').classList.remove('d-none');
    }
  })
  .catch(e=>{
    document.getElementById('planSpinner').classList.remove('active');
    showToast('Error: '+e.message,'error');
    document.getElementById('planEmpty').classList.remove('d-none');
  });
}

function savePlan() {
  if(!lastPlanData) return;
  const planText = document.getElementById('planText').textContent;
  fetch('/api/save-plan', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({...lastPlanData, plan_text: planText})
  })
  .then(r=>r.json())
  .then(data=>{
    if(data.success) showToast('Study plan saved to dashboard!','success');
    else showToast('Save error: '+data.error,'error');
  });
}

function clearPlan() {
  document.getElementById('planResult').classList.add('d-none');
  document.getElementById('planEmpty').classList.remove('d-none');
  lastPlanData = null;
}
"""

@app.route("/planner")
def planner():
    return page_shell("Study Planner", "planner", PLANNER_BODY, PLANNER_JS)


# ===========================================================================
# PAGE: Dashboard
# ===========================================================================
def get_dashboard_data():
    """Pull stats from SQLite for the dashboard."""
    conn = get_db()
    c    = conn.cursor()

    # Quiz stats
    quizzes = c.execute("""
        SELECT subject, score, total, percentage, weak_topics, taken_at
        FROM quiz_results ORDER BY taken_at DESC LIMIT 20
    """).fetchall()

    avg_pct = c.execute(
        "SELECT AVG(percentage) FROM quiz_results"
    ).fetchone()[0] or 0

    # Study plans
    plans = c.execute("""
        SELECT subject, exam_date, hours_per_day, prep_level, created_at
        FROM study_plans ORDER BY created_at DESC LIMIT 5
    """).fetchall()

    # Tasks
    tasks = c.execute("""
        SELECT t.task, t.completed, t.day_label, p.subject
        FROM study_tasks t JOIN study_plans p ON t.plan_id = p.id
        ORDER BY t.created_at DESC LIMIT 20
    """).fetchall()

    # Weak topics aggregated
    all_weak = []
    for q in quizzes:
        raw = q["weak_topics"] or ""
        if raw:
            try:
                items = json.loads(raw)
                all_weak.extend(items)
            except Exception:
                all_weak.extend([x.strip() for x in raw.split(",") if x.strip()])

    weak_count: dict = {}
    for t in all_weak:
        weak_count[t] = weak_count.get(t, 0) + 1
    top_weak = sorted(weak_count.items(), key=lambda x: x[1], reverse=True)[:6]

    # Milestones
    milestones = c.execute(
        "SELECT title, description, achieved_at FROM milestones ORDER BY achieved_at DESC LIMIT 5"
    ).fetchall()

    conn.close()
    return {
        "quizzes":   [dict(q) for q in quizzes],
        "avg_pct":   round(avg_pct, 1),
        "plans":     [dict(p) for p in plans],
        "tasks":     [dict(t) for t in tasks],
        "top_weak":  top_weak,
        "milestones":[dict(m) for m in milestones],
    }


@app.route("/dashboard")
def dashboard():
    data = get_dashboard_data()

    quiz_rows = ""
    for q in data["quizzes"]:
        pct   = q.get("percentage", 0)
        color = "success" if pct >= 70 else ("warning" if pct >= 40 else "danger")
        quiz_rows += f"""
        <tr>
          <td>{q.get('subject','—')}</td>
          <td>{q.get('score',0)}/{q.get('total',0)}</td>
          <td><span class="badge bg-{color}">{pct}%</span></td>
          <td class="text-muted small">{q.get('taken_at','')[:16]}</td>
        </tr>"""
    if not quiz_rows:
        quiz_rows = """<tr><td colspan="4" class="text-center text-muted py-3">
            No quizzes taken yet. <a href="/quiz">Take your first quiz →</a></td></tr>"""

    weak_badges = ""
    for topic, cnt in data["top_weak"]:
        weak_badges += f'<span class="badge bg-warning text-dark me-1 mb-1">{topic} ({cnt}✗)</span>'
    if not weak_badges:
        weak_badges = '<span class="text-muted">No weak topics identified yet.</span>'

    plan_rows = ""
    for p in data["plans"]:
        plan_rows += f"""
        <div class="d-flex justify-content-between align-items-center p-2
                    border-bottom">
          <div>
            <strong>{p.get('subject','—')}</strong>
            <span class="text-muted small ms-2">Exam: {p.get('exam_date','')}</span>
          </div>
          <span class="badge-soft-primary">{p.get('hours_per_day',0)}h/day</span>
        </div>"""
    if not plan_rows:
        plan_rows = """<div class="text-center text-muted py-3">
          No plans saved yet. <a href="/planner">Create a study plan →</a></div>"""

    milestone_html = ""
    for m in data["milestones"]:
        milestone_html += f'<span class="milestone-badge me-2 mb-2">🏆 {m.get("title","")}</span>'
    if not milestone_html:
        milestone_html = '<span class="text-muted">Complete quizzes and save plans to earn milestones.</span>'

    # Chart data for last 10 quizzes
    chart_labels = json.dumps([q.get("subject","Quiz")[:12] for q in reversed(data["quizzes"][-10:])])
    chart_scores = json.dumps([q.get("percentage", 0) for q in reversed(data["quizzes"][-10:])])

    body = f"""
<div class="row mb-3">
  <div class="col">
    <h2 class="section-title">📊 Study Dashboard</h2>
    <p class="section-sub">Track your learning progress, quiz performance, and study milestones.</p>
  </div>
</div>

<!-- KPI Cards -->
<div class="row g-3 mb-4">
  <div class="col-md-3 col-6">
    <div class="card p-3 text-center">
      <div style="font-size:2rem;font-weight:800;color:var(--primary)">
        {len(data['quizzes'])}
      </div>
      <div class="text-muted small">Quizzes Taken</div>
    </div>
  </div>
  <div class="col-md-3 col-6">
    <div class="card p-3 text-center">
      <div style="font-size:2rem;font-weight:800;color:var(--success)">
        {data['avg_pct']}%
      </div>
      <div class="text-muted small">Average Score</div>
    </div>
  </div>
  <div class="col-md-3 col-6">
    <div class="card p-3 text-center">
      <div style="font-size:2rem;font-weight:800;color:var(--accent)">
        {len(data['plans'])}
      </div>
      <div class="text-muted small">Study Plans</div>
    </div>
  </div>
  <div class="col-md-3 col-6">
    <div class="card p-3 text-center">
      <div style="font-size:2rem;font-weight:800;color:var(--warning)">
        {len(data['top_weak'])}
      </div>
      <div class="text-muted small">Weak Topics</div>
    </div>
  </div>
</div>

<!-- Overall Progress Bar -->
<div class="card p-4 mb-4">
  <h6 class="fw-bold mb-3">📈 Overall Performance</h6>
  <div class="d-flex justify-content-between mb-1">
    <span class="text-muted small">Average Quiz Score</span>
    <span class="fw-semibold">{data['avg_pct']}%</span>
  </div>
  <div class="progress mb-3">
    <div class="progress-bar bg-primary" style="width:{data['avg_pct']}%"></div>
  </div>
  <canvas id="quizChart" height="120"></canvas>
</div>

<div class="row g-4 mb-4">
  <!-- Weak Topics -->
  <div class="col-md-6">
    <div class="card p-4 h-100">
      <h6 class="fw-bold mb-3">⚠️ Weak Topics (Needs Revision)</h6>
      <div>{weak_badges}</div>
      {'<div class="mt-3"><a href="/quiz" class="btn btn-sm btn-outline-primary">Practice Quiz →</a></div>' if data['top_weak'] else ''}
    </div>
  </div>
  <!-- Milestones -->
  <div class="col-md-6">
    <div class="card p-4 h-100">
      <h6 class="fw-bold mb-3">🏆 Learning Milestones</h6>
      <div>{milestone_html}</div>
    </div>
  </div>
</div>

<!-- Recent Quizzes -->
<div class="card p-4 mb-4">
  <h6 class="fw-bold mb-3">🧪 Recent Quiz Activity</h6>
  <div class="table-responsive">
    <table class="table table-hover mb-0">
      <thead><tr>
        <th>Subject</th><th>Score</th><th>Percentage</th><th>Date</th>
      </tr></thead>
      <tbody>{quiz_rows}</tbody>
    </table>
  </div>
</div>

<!-- Study Plans -->
<div class="card p-4 mb-4">
  <h6 class="fw-bold mb-3">📅 Saved Study Plans</h6>
  {plan_rows}
  <div class="mt-3">
    <a href="/planner" class="btn btn-sm btn-outline-primary">+ New Study Plan</a>
  </div>
</div>
"""

    dash_js = f"""
const labels = {chart_labels};
const scores = {chart_scores};
if(labels.length > 0){{
  const ctx = document.getElementById('quizChart').getContext('2d');
  new Chart(ctx, {{
    type: 'bar',
    data: {{
      labels: labels,
      datasets: [{{
        label: 'Quiz Score (%)',
        data: scores,
        backgroundColor: scores.map(s =>
          s>=70 ? 'rgba(16,185,129,.7)' : s>=40 ? 'rgba(245,158,11,.7)' : 'rgba(239,68,68,.7)'
        ),
        borderRadius: 6,
      }}]
    }},
    options: {{
      responsive: true,
      plugins: {{ legend: {{ display:false }} }},
      scales: {{
        y: {{ beginAtZero:true, max:100,
              grid: {{ color:'rgba(0,0,0,.06)' }} }},
        x: {{ grid: {{ display:false }} }}
      }}
    }}
  }});
}}
"""
    # Inject Chart.js CDN before closing
    chart_script = """
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.3/dist/chart.umd.min.js"></script>
"""
    html = page_shell("Dashboard", "dashboard", body, dash_js)
    # Insert Chart.js before </head>
    html = html.replace("</head>", chart_script + "</head>")
    return html


# ===========================================================================
# PAGE: About
# ===========================================================================
ABOUT_BODY = """
<div class="row mb-3">
  <div class="col">
    <h2 class="section-title">ℹ️ About StudyGen AI</h2>
    <p class="section-sub">
      An AI-powered Smart Study Generator for students, built with Flask and the Groq API.
    </p>
  </div>
</div>

<div class="row g-4">
  <div class="col-lg-8">
    <div class="card p-4 mb-4">
      <h5 class="fw-bold">🎯 Problem Statement</h5>
      <p>
        Modern students face information overload. Textbooks, lecture notes, PDFs, and
        online resources pile up, making it hard to identify what to study, how deeply,
        and in what order – especially under exam pressure.
      </p>
      <p>
        <strong>StudyGen AI</strong> is a Smart Study Generator that helps students:
      </p>
      <ul>
        <li>Organize and understand large amounts of study material</li>
        <li>Identify important concepts and weak areas automatically</li>
        <li>Practice with AI-generated MCQ quizzes and get instant feedback</li>
        <li>Create personalized day-by-day exam preparation plans</li>
        <li>Track study progress and receive AI-driven revision guidance</li>
      </ul>
    </div>

    <div class="card p-4 mb-4">
      <h5 class="fw-bold">🤖 Two-Agent Architecture</h5>
      <div class="row g-3">
        <div class="col-md-6">
          <div class="p-3 rounded-3" style="background:var(--blue-soft);border:1px solid #bfdbfe">
            <h6 class="fw-bold" style="color:var(--primary)">📝 Study Material Agent</h6>
            <ul class="small text-muted mb-0">
              <li>Processes text, PDFs, TXT files</li>
              <li>Analyzes textbook/notes images (vision AI)</li>
              <li>Generates summaries and key points</li>
              <li>Creates Q&amp;A flashcards</li>
              <li>Identifies important concepts</li>
              <li>Explains difficult topics simply</li>
              <li>Builds structured concept maps</li>
              <li>Answers questions from your material</li>
            </ul>
          </div>
        </div>
        <div class="col-md-6">
          <div class="p-3 rounded-3" style="background:var(--lavender);border:1px solid #c4b5fd">
            <h6 class="fw-bold" style="color:var(--accent)">📅 Study Planner &amp; Evaluation Agent</h6>
            <ul class="small text-muted mb-0">
              <li>Generates personalized study plans</li>
              <li>Day-by-day topic scheduling</li>
              <li>Prioritizes weak and important topics</li>
              <li>Creates MCQ practice quizzes</li>
              <li>Evaluates quiz performance</li>
              <li>Identifies weak areas</li>
              <li>Provides revision recommendations</li>
              <li>Gives performance forecasts</li>
            </ul>
          </div>
        </div>
      </div>
    </div>

    <div class="card p-4 mb-4">
      <h5 class="fw-bold">⚡ AI Workflow</h5>
      <div class="p-3 rounded-3" style="background:var(--light-bg);font-size:.9rem">
        <pre style="white-space:pre-wrap;font-family:inherit;color:var(--text)">
Study Resource / Voice Query / Image
        ↓
Flask Backend (app.py)
        ↓
Simple Orchestrator (orchestrate function)
        ↓
Study Material Agent   OR   Study Planner & Evaluation Agent
        ↓
Groq API (generate_response / generate_vision_response)
        ↓
Open-source / Open-weight LLM (LLaMA 3, Qwen Vision, Whisper)
        ↓
Structured AI Result (JSON / plain text)
        ↓
Frontend (Bootstrap 5 + Vanilla JS)
        ↓
Student Dashboard / Learning Progress (SQLite)
        </pre>
      </div>
    </div>
  </div>

  <div class="col-lg-4">
    <div class="card p-4 mb-4">
      <h6 class="fw-bold">⚡ Powered By</h6>
      <ul class="list-unstyled mt-2">
        <li class="mb-2">
          <span class="badge-soft-primary me-2">Groq API</span>
          Ultra-fast LLM inference
        </li>
        <li class="mb-2">
          <span class="badge-soft-purple me-2">LLaMA 3</span>
          Text generation (llama3-70b-8192)
        </li>
        <li class="mb-2">
          <span class="badge-soft-primary me-2">Llama 4 Scout</span>
          Vision / image analysis
        </li>
        <li class="mb-2">
          <span class="badge-soft-green me-2">Whisper v3</span>
          Speech-to-text (STT)
        </li>
        <li class="mb-2">
          <span class="badge-soft-primary me-2">Flask</span>
          Python web framework
        </li>
        <li class="mb-2">
          <span class="badge-soft-purple me-2">SQLite</span>
          Local progress storage
        </li>
        <li class="mb-2">
          <span class="badge-soft-green me-2">Bootstrap 5</span>
          Responsive UI
        </li>
      </ul>
    </div>

    <div class="card p-4 mb-4">
      <h6 class="fw-bold">🌐 Multimodal Input</h6>
      <p class="text-muted small">
        StudyGen AI accepts study material in multiple formats:
      </p>
      <ul class="text-muted small">
        <li><strong>Text:</strong> Paste notes directly</li>
        <li><strong>Files:</strong> Upload TXT or PDF</li>
        <li><strong>Images:</strong> Upload textbook/notes photos → analyzed by a vision LLM</li>
        <li><strong>Voice:</strong> Browser speech recognition or Groq Whisper STT</li>
      </ul>
    </div>

    <div class="card p-4">
      <h6 class="fw-bold">🔒 Privacy & API</h6>
      <p class="text-muted small">
        Your Groq API key is stored locally in a <code>.env</code> file and never shared.
        Study data is stored in a local SQLite database (<code>studygen.db</code>).
        No data is sent to any server except the Groq API for AI inference.
      </p>
    </div>
  </div>
</div>
"""

@app.route("/about")
def about():
    return page_shell("About", "about", ABOUT_BODY)


# ===========================================================================
# API Routes
# ===========================================================================

# ---- Study Material ---------------------------------------------------------
@app.route("/api/study-material", methods=["POST"])
def api_study_material():
    data     = request.get_json(silent=True) or {}
    action   = data.get("action", "").strip()
    material = data.get("material", "").strip()
    query    = data.get("query", "").strip()

    if not action:
        return jsonify({"success": False, "error": "Missing action."})

    result = orchestrate("study_material", action,
                         {"material": material, "query": query})
    return jsonify(result)


@app.route("/api/upload-file", methods=["POST"])
def api_upload_file():
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file uploaded."})
    f   = request.files["file"]
    ext = os.path.splitext(f.filename or "")[1].lower()
    if ext not in ALLOWED_TEXT_EXT:
        return jsonify({"success": False,
                        "error": f"Unsupported file type '{ext}'. "
                                 f"Allowed: {', '.join(ALLOWED_TEXT_EXT)}"})
    result = extract_text_from_file(f)
    return jsonify(result)


@app.route("/api/analyze-image", methods=["POST"])
def api_analyze_image():
    data      = request.get_json(silent=True) or {}
    image_b64 = data.get("image_b64", "")
    mime_type = data.get("mime_type", "image/jpeg")
    query     = data.get("query", "")

    if not image_b64:
        return jsonify({"success": False, "error": "No image data received."})

    # Basic size check (base64 is ~4/3 of original)
    if len(image_b64) > MAX_UPLOAD_BYTES * 1.4:
        return jsonify({"success": False,
                        "error": f"Image too large (max {MAX_UPLOAD_MB} MB)."})

    result = orchestrate("study_material", "image",
                         {"image_b64": image_b64,
                          "mime_type": mime_type,
                          "query": query})
    return jsonify(result)


@app.route("/api/transcribe-audio", methods=["POST"])
def api_transcribe_audio():
    if "audio" not in request.files:
        return jsonify({"success": False, "error": "No audio file uploaded."})
    audio_file = request.files["audio"]
    audio_bytes = audio_file.read()
    if len(audio_bytes) > MAX_UPLOAD_BYTES:
        return jsonify({"success": False,
                        "error": f"Audio file too large (max {MAX_UPLOAD_MB} MB)."})
    result = transcribe_audio(audio_bytes, audio_file.filename or "audio.webm")
    return jsonify(result)


# ---- Quiz -------------------------------------------------------------------
@app.route("/api/generate-quiz", methods=["POST"])
def api_generate_quiz():
    data     = request.get_json(silent=True) or {}
    subject  = data.get("subject", "").strip()
    material = data.get("material", "").strip()
    num_q    = int(data.get("num_questions", 10))

    if not subject:
        return jsonify({"success": False, "error": "Subject is required."})
    if num_q < 1 or num_q > 20:
        num_q = 10

    result = orchestrate("planner", "generate_quiz",
                         {"subject": subject,
                          "material": material,
                          "num_questions": num_q})
    return jsonify(result)


@app.route("/api/evaluate-quiz", methods=["POST"])
def api_evaluate_quiz():
    data      = request.get_json(silent=True) or {}
    questions = data.get("questions", [])
    answers   = data.get("answers", {})

    if not questions:
        return jsonify({"success": False, "error": "No questions to evaluate."})

    evaluation = planner_agent_evaluate_quiz(questions, answers)
    return jsonify({"success": True, **evaluation})


@app.route("/api/save-quiz", methods=["POST"])
def api_save_quiz():
    data       = request.get_json(silent=True) or {}
    subject    = data.get("subject", "Unknown")
    score      = data.get("score", 0)
    total      = data.get("total", 0)
    percentage = data.get("percentage", 0)
    weak_list  = data.get("weak_topics", [])

    conn = get_db()
    try:
        conn.execute("""
            INSERT INTO quiz_results (subject, score, total, percentage, weak_topics)
            VALUES (?,?,?,?,?)
        """, (subject, score, total, percentage, json.dumps(weak_list)))

        # Award milestone for first perfect score or high score
        if percentage == 100:
            conn.execute(
                "INSERT INTO milestones (title, description) VALUES (?,?)",
                ("Perfect Score!", f"100% on {subject} quiz")
            )
        elif percentage >= 80:
            conn.execute(
                "INSERT INTO milestones (title, description) VALUES (?,?)",
                ("High Achiever", f"{percentage}% on {subject} quiz")
            )
        conn.commit()
    except Exception as exc:
        conn.close()
        return jsonify({"success": False, "error": str(exc)})
    conn.close()
    return jsonify({"success": True})


# ---- Planner ----------------------------------------------------------------
@app.route("/api/generate-plan", methods=["POST"])
def api_generate_plan():
    data = request.get_json(silent=True) or {}
    required = ["subject", "exam_date", "hours_per_day", "prep_level", "syllabus"]
    for field in required:
        if not data.get(field):
            return jsonify({"success": False,
                            "error": f"Missing required field: {field}"})

    result = orchestrate("planner", "generate_plan", data)
    return jsonify(result)


@app.route("/api/save-plan", methods=["POST"])
def api_save_plan():
    data = request.get_json(silent=True) or {}
    conn = get_db()
    try:
        cur = conn.execute("""
            INSERT INTO study_plans
              (subject, exam_date, hours_per_day, prep_level, plan_json)
            VALUES (?,?,?,?,?)
        """, (
            data.get("subject"),
            data.get("exam_date"),
            data.get("hours_per_day", 3),
            data.get("prep_level"),
            data.get("plan_text", ""),
        ))
        plan_id = cur.lastrowid
        # Award milestone for first plan
        plans_count = conn.execute("SELECT COUNT(*) FROM study_plans").fetchone()[0]
        if plans_count == 1:
            conn.execute(
                "INSERT INTO milestones (title, description) VALUES (?,?)",
                ("Study Planner", f"Created first study plan for {data.get('subject')}")
            )
        conn.commit()
    except Exception as exc:
        conn.close()
        return jsonify({"success": False, "error": str(exc)})
    conn.close()
    return jsonify({"success": True, "plan_id": plan_id})


@app.route("/api/toggle-task", methods=["POST"])
def api_toggle_task():
    data    = request.get_json(silent=True) or {}
    task_id = data.get("task_id")
    done    = data.get("completed", False)
    conn    = get_db()
    conn.execute("UPDATE study_tasks SET completed=? WHERE id=?",
                 (1 if done else 0, task_id))
    conn.commit()
    conn.close()
    return jsonify({"success": True})


# ---- Health check -----------------------------------------------------------
@app.route("/api/health")
def api_health():
    return jsonify({
        "status":         "ok",
        "groq_available": GROQ_AVAILABLE,
        "api_key_set":    bool(GROQ_API_KEY),
        "pdf_available":  PDF_AVAILABLE,
        "model":          GROQ_MODEL,
        "vision_model":   GROQ_VISION_MODEL,
    })


# ===========================================================================
# App entry point
# ===========================================================================
if __name__ == "__main__":
    init_db()
    print("=" * 60)
    print("  StudyGen AI – Smart Study Generator")
    print("=" * 60)
    if not GROQ_API_KEY:
        print("  ⚠  WARNING: GROQ_API_KEY is not set.")
        print("  Copy .env.example to .env and add your key.")
    else:
        print(f"  ✅ Groq API key detected.")
    print(f"  📖 Text model   : {GROQ_MODEL}")
    print(f"  🖼️  Vision model : {GROQ_VISION_MODEL}")
    print(f"  🎙️  STT model    : {GROQ_STT_MODEL}")
    print(f"  💾 Database      : {DB_PATH}")
    print("  🌐 URL           : http://127.0.0.1:5000")
    print("=" * 60)
    app.run(debug=True, host="127.0.0.1", port=5000)
