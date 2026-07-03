
import os
import re
import json
import uuid
import sqlite3
from pathlib import Path
from datetime import datetime
 
import streamlit as st
from dotenv import load_dotenv
from openai import AzureOpenAI
 
load_dotenv()
 
# --------------------------------------------------------------------------- #
# Storage + config
# --------------------------------------------------------------------------- #
DATA_DIR = Path(__file__).with_name("lecture_data")
VIDEO_DIR = DATA_DIR / "videos"
INDEX_FILE = DATA_DIR / "index.json"
DB_PATH = DATA_DIR / "school.db"          # <-- preference-system database lives alongside lecture data
VIDEO_DIR.mkdir(parents=True, exist_ok=True)
 
SUBJECTS = ["Mathematics", "Physics", "Chemistry", "Biology", "Computer Science",
            "English", "Economics", "Accounting", "Islamiyat", "Pakistan Studies", "Urdu"]
 
VIDEO_TYPES = ["mp4", "mov", "webm", "m4v"]
 
# --------------------------------------------------------------------------- #
# Model registry — same wiring as the other course apps
# --------------------------------------------------------------------------- #
OPENAI_EP = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
OPENAI_KEY = os.environ.get("AZURE_OPENAI_API_KEY", "")
FOUNDRY_EP = os.environ.get("AZURE_FOUNDRY_ENDPOINT", "")
FOUNDRY_KEY = os.environ.get("AZURE_FOUNDRY_API_KEY", "")
API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21")
 
MODELS = {
    "GPT-5.5": (os.environ.get("MODEL_GPT55_DEPLOYMENT", "gpt-5-5"), OPENAI_EP, OPENAI_KEY),
    "DeepSeek-V4-Pro": (os.environ.get("MODEL_DEEPSEEK_V4_PRO_DEPLOYMENT", "ds-v4pro"), FOUNDRY_EP, FOUNDRY_KEY),
    "Grok-4.3": (os.environ.get("MODEL_GROK43_DEPLOYMENT", "xai-grok43"), FOUNDRY_EP, FOUNDRY_KEY),
    "Mistral-Medium-3.5": (os.environ.get("MODEL_MISTRAL_MEDIUM_35_DEPLOYMENT", "mstr-med35"), FOUNDRY_EP, FOUNDRY_KEY),
}
 
 
def ai_ready(model):
    return bool(MODELS[model][1] and MODELS[model][2])
 
 
# --------------------------------------------------------------------------- #
# Data helpers (lecture index — unchanged)
# --------------------------------------------------------------------------- #
def load_index() -> list:
    if INDEX_FILE.exists():
        try:
            return json.loads(INDEX_FILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []
    return []
 
 
def save_index(items: list) -> None:
    INDEX_FILE.write_text(json.dumps(items, indent=2), encoding="utf-8")
 
 
def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)[:60]
 
 
def add_lecture(title, subject, description, notes, uploaded_file) -> None:
    lid = uuid.uuid4().hex[:10]
    ext = Path(uploaded_file.name).suffix.lower() or ".mp4"
    fname = f"{lid}_{safe_name(Path(uploaded_file.name).stem)}{ext}"
    (VIDEO_DIR / fname).write_bytes(uploaded_file.getbuffer())
    items = load_index()
    items.append({
        "id": lid, "title": title.strip(), "subject": subject,
        "description": description.strip(), "notes": notes.strip(),
        "video": fname, "uploaded_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    })
    save_index(items)
 
 
def delete_lecture(lid: str) -> None:
    items = load_index()
    for it in items:
        if it["id"] == lid:
            try:
                (VIDEO_DIR / it["video"]).unlink(missing_ok=True)
            except OSError:
                pass
    save_index([it for it in items if it["id"] != lid])
 
 
def parse_json(text):
    if not text:
        return None
    cleaned = re.sub(r"```(?:json)?", "", text).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        m = re.search(r"(\{.*\}|\[.*\])", cleaned, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                return None
    return None
 
 
# --------------------------------------------------------------------------- #
# AI helpers
# --------------------------------------------------------------------------- #
def _call(model, prompt, system, max_tokens=700):
    deployment, endpoint, key = MODELS[model]
    if not endpoint or not key:
        return {"ok": False, "text": f"⚠️ {model}: AI is not configured in this environment."}
    try:
        client = AzureOpenAI(api_key=key, azure_endpoint=endpoint, api_version=API_VERSION)
        resp = client.chat.completions.create(
            model=deployment,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": prompt}],
            temperature=1,
            max_completion_tokens=max_tokens,
        )
        return {"ok": True, "text": resp.choices[0].message.content or ""}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "text": f"⚠️ Could not reach {model}: {exc}"}
 
 
def _lecture_context(lec) -> str:
    return (f"Lecture: {lec['title']} ({lec['subject']})\n"
            f"Description: {lec.get('description','')}\n"
            f"Notes:\n{lec.get('notes','') or '(no notes provided)'}")
 
 
def ask_tutor(model, lec, question):
    system = ("You are a friendly O-Level tutor. Answer the student's question about this "
              "lecture using its notes. If the notes don't cover it, use your general "
              "O-Level knowledge but say so. Keep it clear and simple.")
    return _call(model, f"{_lecture_context(lec)}\n\nSTUDENT QUESTION: {question}",
                 system, max_tokens=600)["text"]
 
 
def summarize(model, lec):
    system = "You summarise lessons into clear revision notes for O-Level students."
    prompt = (f"{_lecture_context(lec)}\n\nWrite a revision summary: 5-7 key bullet points "
              "plus one 'exam tip'.")
    return _call(model, prompt, system, max_tokens=600)["text"]
 
 
def make_quiz(model, lec, n=5):
    system = "You write clear O-Level multiple-choice questions with one correct answer."
    prompt = (f"{_lecture_context(lec)}\n\nWrite {n} multiple-choice questions testing this "
              "lecture. Return ONLY JSON: "
              '{"questions":[{"q":"...","options":["a","b","c","d"],"answer_index":0,'
              '"explanation":"why"}]}')
    res = _call(model, prompt, system, max_tokens=1100)
    data = parse_json(res["text"]) if res["ok"] else None
    return data.get("questions") if isinstance(data, dict) else None
 
 
def generate_welcome_note(model, name, subjects):
    """Personalized AI-concierge welcome note shown after sign-up — the app's signature touch."""
    system = ("You are an enthusiastic school orientation concierge who writes short, warm, "
              "personalized welcome notes for new O-Level students. No corporate tone, no "
              "generic filler — make it feel handwritten and specific.")
    prompt = (f"Student name: {name}\n"
              f"Subjects enrolled: {', '.join(subjects) if subjects else 'none yet'}\n\n"
              "Write a short welcome note (3-4 sentences): greet them by name, say something "
              "specific and encouraging about the *combination* of subjects they picked, and "
              "end with one light, motivating line about their first week.")
    res = _call(model, prompt, system, max_tokens=220)
    return res["text"] if res["ok"] else None
 
 
@st.cache_data(show_spinner=False)
def video_bytes(path_str, size):
    return Path(path_str).read_bytes()
 
 
# --------------------------------------------------------------------------- #
# Preference-system database
# --------------------------------------------------------------------------- #
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn
 
 
def create_tables():
    conn = get_conn()
    cursor = conn.cursor()
 
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS students (
            student_id INTEGER PRIMARY KEY AUTOINCREMENT,
            full_name TEXT NOT NULL,
            roll_number TEXT UNIQUE NOT NULL,
            grade_level TEXT,
            email TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
 
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS subjects (
            subject_id INTEGER PRIMARY KEY AUTOINCREMENT,
            subject_name TEXT UNIQUE NOT NULL,
            subject_code TEXT UNIQUE
        );
    """)
 
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS teachers (
            teacher_id INTEGER PRIMARY KEY AUTOINCREMENT,
            full_name TEXT NOT NULL,
            email TEXT,
            max_students INTEGER DEFAULT 30
        );
    """)
 
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS teacher_subjects (
            teacher_id INTEGER NOT NULL,
            subject_id INTEGER NOT NULL,
            PRIMARY KEY (teacher_id, subject_id),
            FOREIGN KEY (teacher_id) REFERENCES teachers(teacher_id) ON DELETE CASCADE,
            FOREIGN KEY (subject_id) REFERENCES subjects(subject_id) ON DELETE CASCADE
        );
    """)
 
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS student_preferences (
            preference_id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER NOT NULL,
            subject_id INTEGER NOT NULL,
            preferred_teacher_id INTEGER,
            priority INTEGER DEFAULT 1,
            submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (student_id) REFERENCES students(student_id) ON DELETE CASCADE,
            FOREIGN KEY (subject_id) REFERENCES subjects(subject_id) ON DELETE CASCADE,
            FOREIGN KEY (preferred_teacher_id) REFERENCES teachers(teacher_id) ON DELETE SET NULL,
            UNIQUE (student_id, subject_id, priority)
        );
    """)
 
    conn.commit()
    conn.close()
 
 
def seed_subjects():
    """Keep the `subjects` table in sync with the app's SUBJECTS list."""
    conn = get_conn()
    cursor = conn.cursor()
    cursor.executemany(
        "INSERT OR IGNORE INTO subjects (subject_name) VALUES (?);",
        [(s,) for s in SUBJECTS]
    )
    conn.commit()
    conn.close()
 
 
def get_student_by_roll(roll_number):
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT student_id, full_name, grade_level, email FROM students WHERE roll_number = ?;",
                   (roll_number,))
    row = cursor.fetchone()
    conn.close()
    return row  # (student_id, full_name, grade_level, email) or None
 
 
def get_student_by_id(student_id):
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT student_id, full_name, roll_number, grade_level, email FROM students WHERE student_id = ?;",
                   (student_id,))
    row = cursor.fetchone()
    conn.close()
    return row  # (student_id, full_name, roll_number, grade_level, email) or None
 
 
def register_student(full_name, roll_number, grade_level, email):
    """Add a new student. Returns the student_id, or None if roll_number already exists."""
    conn = get_conn()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO students (full_name, roll_number, grade_level, email) VALUES (?, ?, ?, ?);",
            (full_name, roll_number, grade_level, email)
        )
        conn.commit()
        return cursor.lastrowid
    except sqlite3.IntegrityError:
        return None
    finally:
        conn.close()
 
 
def register_teacher_for_subjects(full_name, email, subject_names):
    """Create the teacher if needed, then link them to each chosen subject."""
    conn = get_conn()
    cursor = conn.cursor()
 
    cursor.execute("SELECT teacher_id FROM teachers WHERE full_name = ?;", (full_name,))
    row = cursor.fetchone()
    if row:
        teacher_id = row[0]
    else:
        cursor.execute("INSERT INTO teachers (full_name, email) VALUES (?, ?);", (full_name, email))
        teacher_id = cursor.lastrowid
 
    for subject_name in subject_names:
        cursor.execute("SELECT subject_id FROM subjects WHERE subject_name = ?;", (subject_name,))
        srow = cursor.fetchone()
        if srow:
            cursor.execute(
                "INSERT OR IGNORE INTO teacher_subjects (teacher_id, subject_id) VALUES (?, ?);",
                (teacher_id, srow[0])
            )
 
    conn.commit()
    conn.close()
    return teacher_id
 
 
def get_teachers_for_subject(subject_name):
    """List teachers who teach a given subject."""
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT t.teacher_id, t.full_name
        FROM teachers t
        JOIN teacher_subjects ts ON ts.teacher_id = t.teacher_id
        JOIN subjects s ON s.subject_id = ts.subject_id
        WHERE s.subject_name = ?
        ORDER BY t.full_name;
    """, (subject_name,))
    rows = cursor.fetchall()
    conn.close()
    return rows  # list of (teacher_id, full_name)
 
 
def submit_preference(student_id, subject_name, preferred_teacher_id=None, priority=1):
    """Record (or update) a student's subject + preferred teacher choice."""
    conn = get_conn()
    cursor = conn.cursor()
 
    cursor.execute("SELECT subject_id FROM subjects WHERE subject_name = ?;", (subject_name,))
    row = cursor.fetchone()
    if row is None:
        conn.close()
        return False, f"Subject '{subject_name}' not found."
    subject_id = row[0]
 
    try:
        cursor.execute("""
            INSERT INTO student_preferences (student_id, subject_id, preferred_teacher_id, priority)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(student_id, subject_id, priority)
            DO UPDATE SET preferred_teacher_id = excluded.preferred_teacher_id,
                          submitted_at = CURRENT_TIMESTAMP;
        """, (student_id, subject_id, preferred_teacher_id, priority))
        conn.commit()
        return True, "Preference saved."
    except sqlite3.IntegrityError as exc:
        return False, str(exc)
    finally:
        conn.close()
 
 
def view_student_preferences(student_id):
    """Return all preferences for a student, in priority order."""
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT sub.subject_name, t.full_name, sp.priority
        FROM student_preferences sp
        JOIN subjects sub ON sub.subject_id = sp.subject_id
        LEFT JOIN teachers t ON t.teacher_id = sp.preferred_teacher_id
        WHERE sp.student_id = ?
        ORDER BY sp.priority ASC;
    """, (student_id,))
    rows = cursor.fetchall()
    conn.close()
    return rows  # list of (subject_name, teacher_name_or_None, priority)
 
 
def get_preferences_for_teacher(teacher_id):
    """Which students picked this teacher, and for which subject."""
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT s.full_name, s.roll_number, sub.subject_name, sp.priority
        FROM student_preferences sp
        JOIN students s ON s.student_id = sp.student_id
        JOIN subjects sub ON sub.subject_id = sp.subject_id
        WHERE sp.preferred_teacher_id = ?
        ORDER BY sub.subject_name, sp.priority;
    """, (teacher_id,))
    rows = cursor.fetchall()
    conn.close()
    return rows
 
 
# --------------------------------------------------------------------------- #
# UI — Teacher
# --------------------------------------------------------------------------- #
def teacher_view():
    tab_upload, tab_profile = st.tabs(["📤 Upload lecture", "🏫 My subjects & student picks"])
 
    with tab_upload:
        st.subheader("👩‍🏫 Teacher — add a lecture")
        with st.form("upload", clear_on_submit=True):
            title = st.text_input("Lecture title", placeholder="e.g. Photosynthesis — Part 1")
            subject = st.selectbox("Subject", SUBJECTS)
            description = st.text_input("One-line description", placeholder="What is this lesson about?")
            notes = st.text_area("Lecture notes (the AI tutor & quiz use these)", height=180,
                                 placeholder="Paste or write the key notes for this lecture…")
            video = st.file_uploader("Video lecture", type=VIDEO_TYPES)
            submitted = st.form_submit_button("⬆️ Upload lecture", type="primary")
            if submitted:
                if not title.strip() or video is None:
                    st.warning("Please give a title and choose a video file.")
                else:
                    with st.spinner("Saving…"):
                        add_lecture(title, subject, description, notes, video)
                    st.success(f"Uploaded “{title.strip()}” to {subject}.")
 
        items = load_index()
        if items:
            st.markdown("#### Your lectures")
            for it in reversed(items):
                c1, c2 = st.columns([5, 1])
                c1.markdown(f"**{it['title']}** · {it['subject']}  \n"
                            f"<span style='opacity:.7'>{it.get('description','')} · "
                            f"uploaded {it.get('uploaded_at','')}</span>", unsafe_allow_html=True)
                if c2.button("🗑️ Delete", key=f"del_{it['id']}"):
                    delete_lecture(it["id"])
                    st.rerun()
 
    with tab_profile:
        st.subheader("🏫 Register the subjects you teach")
        st.caption("This lets students pick you as their preferred teacher for a subject.")
        with st.form("teacher_profile"):
            t_name = st.text_input("Your full name")
            t_email = st.text_input("Your email")
            t_subjects = st.multiselect("Subjects you teach", SUBJECTS)
            t_submitted = st.form_submit_button("Save my profile", type="primary")
            if t_submitted:
                if not t_name.strip() or not t_subjects:
                    st.warning("Please enter your name and pick at least one subject.")
                else:
                    register_teacher_for_subjects(t_name.strip(), t_email.strip(), t_subjects)
                    st.success(f"Saved! You're now listed for: {', '.join(t_subjects)}.")
 
        st.divider()
        st.markdown("#### See which students picked you")
        conn = get_conn()
        all_teachers = conn.execute("SELECT teacher_id, full_name FROM teachers ORDER BY full_name;").fetchall()
        conn.close()
        if not all_teachers:
            st.info("No teachers registered yet.")
        else:
            names = [t[1] for t in all_teachers]
            picked = st.selectbox("Choose your name", range(len(all_teachers)), format_func=lambda i: names[i])
            teacher_id = all_teachers[picked][0]
            rows = get_preferences_for_teacher(teacher_id)
            if not rows:
                st.info("No students have picked you yet.")
            else:
                for full_name, roll, subject_name, priority in rows:
                    st.write(f"**{full_name}** ({roll}) — {subject_name}, priority {priority}")
 
 
# --------------------------------------------------------------------------- #
# UI — Student: lecture playback
# --------------------------------------------------------------------------- #
def _play_lecture(lec, model):
    st.markdown(f"### {lec['title']}")
    if lec.get("description"):
        st.caption(lec["description"])
    path = VIDEO_DIR / lec["video"]
    if path.exists():
        st.video(video_bytes(str(path), path.stat().st_size))
    else:
        st.error("Video file is missing (it may have been reset on redeploy).")
 
    tab_notes, tab_ask, tab_quiz = st.tabs(["📄 Notes & summary", "💬 Ask the tutor", "🧠 Quiz me"])
 
    with tab_notes:
        st.markdown(lec.get("notes") or "_No notes were added for this lecture._")
        if ai_ready(model) and lec.get("notes"):
            if st.button("✨ Summarise for revision", key=f"sum_{lec['id']}"):
                with st.spinner("Summarising…"):
                    st.session_state[f"summary_{lec['id']}"] = summarize(model, lec)
            if st.session_state.get(f"summary_{lec['id']}"):
                st.info(st.session_state[f"summary_{lec['id']}"])
 
    with tab_ask:
        if not ai_ready(model):
            st.info("The AI tutor isn't configured in this environment.")
        else:
            q = st.text_input("Ask anything about this lecture",
                              key=f"q_{lec['id']}", placeholder="e.g. Why is chlorophyll important?")
            if st.button("Ask", key=f"ask_{lec['id']}", type="primary") and q.strip():
                with st.spinner("Thinking…"):
                    st.session_state[f"ans_{lec['id']}"] = ask_tutor(model, lec, q)
            if st.session_state.get(f"ans_{lec['id']}"):
                st.markdown(st.session_state[f"ans_{lec['id']}"])
 
    with tab_quiz:
        if not ai_ready(model):
            st.info("Quizzes need the AI, which isn't configured here.")
        else:
            _quiz_ui(lec, model)
 
 
def _quiz_ui(lec, model):
    qkey = f"quiz_{lec['id']}"
    if st.button("🎯 Make me a quiz", key=f"mkquiz_{lec['id']}"):
        with st.spinner("Writing your quiz…"):
            st.session_state[qkey] = make_quiz(model, lec)
            st.session_state[f"{qkey}_submitted"] = False
    quiz = st.session_state.get(qkey)
    if not quiz:
        return
    answers = {}
    for i, item in enumerate(quiz):
        st.markdown(f"**Q{i+1}. {item['q']}**")
        answers[i] = st.radio("Pick one", item["options"], index=None,
                              key=f"{qkey}_{i}", label_visibility="collapsed")
    if st.button("Submit answers", key=f"{qkey}_submit", type="primary"):
        st.session_state[f"{qkey}_submitted"] = True
    if st.session_state.get(f"{qkey}_submitted"):
        correct = 0
        for i, item in enumerate(quiz):
            chosen = answers.get(i)
            right = item["options"][item["answer_index"]]
            ok = chosen == right
            correct += int(ok)
            st.markdown(("✅" if ok else "❌") + f" **Q{i+1}** — correct: *{right}*")
            st.caption("💡 " + item.get("explanation", ""))
        st.markdown(f"### Score: {correct}/{len(quiz)}")
        if correct == len(quiz):
            st.balloons()
 
 
# --------------------------------------------------------------------------- #
# UI — Student sign-up wizard
#
# A branded, 3-step onboarding flow: details -> subjects & teachers -> a
# generated "digital enrollment card" + an AI concierge welcome note. This is
# the piece that's meant to stand out from a plain sign-up form.
# --------------------------------------------------------------------------- #
_HERO_CSS = """
<style>
.sh-hero {
    background: linear-gradient(135deg, #6C5CE7 0%, #00B894 100%);
    border-radius: 20px;
    padding: 2.2rem 2rem;
    color: white;
    margin-bottom: 1.4rem;
    box-shadow: 0 10px 30px rgba(108,92,231,0.30);
}
.sh-hero h1 { margin: 0; font-size: 1.9rem; }
.sh-hero p { opacity: .92; margin-top: .5rem; font-size: 1rem; }
.sh-step-pill {
    display: inline-block; padding: .32rem 1rem; border-radius: 999px;
    font-size: .8rem; font-weight: 600; margin-right: .4rem; margin-bottom: .6rem;
}
.sh-step-active { background: #6C5CE7; color: white; }
.sh-step-done   { background: #00B894; color: white; }
.sh-step-todo   { background: #eef0f5; color: #8a8f9c; }
.sh-id-card {
    background: linear-gradient(135deg, #1e1e2f, #2d2d44);
    border-radius: 18px; padding: 1.6rem 1.7rem; color: white;
    max-width: 460px; box-shadow: 0 8px 26px rgba(0,0,0,.28);
}
.sh-id-avatar {
    width: 54px; height: 54px; border-radius: 50%;
    background: linear-gradient(135deg, #6C5CE7, #00B894);
    display: flex; align-items: center; justify-content: center;
    font-weight: 700; font-size: 1.25rem; margin-bottom: .7rem;
}
.sh-id-name { font-size: 1.15rem; font-weight: 700; }
.sh-id-meta { opacity: .7; font-size: .82rem; margin-bottom: .7rem; }
.sh-subject-badge {
    display: inline-block; background: rgba(255,255,255,.12);
    padding: .28rem .75rem; border-radius: 999px; font-size: .78rem;
    margin: .2rem .3rem .2rem 0;
}
.sh-note {
    border-left: 4px solid #6C5CE7; background: rgba(108,92,231,.06);
    padding: .9rem 1.1rem; border-radius: 8px; margin-top: .8rem;
}
</style>
"""
 
_STEP_LABELS = ["1 · Your details", "2 · Subjects & teachers", "3 · Confirm"]
 
 
def _inject_hero_css():
    st.markdown(_HERO_CSS, unsafe_allow_html=True)
 
 
def _render_hero():
    st.markdown(
        """<div class="sh-hero">
        <h1>🚀 Join the Study Hub</h1>
        <p>Set up your profile once — pick your subjects, choose the teachers you vibe with,
        and get a personalized welcome note from our AI concierge.</p>
        </div>""",
        unsafe_allow_html=True,
    )
 
 
def _render_steps(current):
    html = ""
    for i, label in enumerate(_STEP_LABELS, start=1):
        if i == current:
            cls = "sh-step-active"
        elif i < current:
            cls = "sh-step-done"
        else:
            cls = "sh-step-todo"
        html += f'<span class="sh-step-pill {cls}">{label}</span>'
    st.markdown(html, unsafe_allow_html=True)
 
 
def _init_signup_state():
    st.session_state.setdefault("signup_step", 1)
    st.session_state.setdefault("signup_data", {})
    st.session_state.setdefault("editing_prefs", False)
 
 
def _step_details():
    st.subheader("Step 1 — Tell us about you")
    roll = st.text_input("Candidate number", value=st.session_state.signup_data.get("roll", ""),
                         placeholder="e.g. OL-2026-001")
 
    if roll.strip():
        existing = get_student_by_roll(roll.strip())
        if existing:
            st.info(f"Welcome back, **{existing[1]}**! We'll log you straight in — "
                    "no need to re-enter your details.")
            if st.button("Continue as this student →", type="primary"):
                st.session_state.student_id = existing[0]
                st.session_state.student_name = existing[1]
                st.rerun()
            return
 
    name = st.text_input("Full name", value=st.session_state.signup_data.get("name", ""))
    email = st.text_input("Email address", value=st.session_state.signup_data.get("email", ""),
                          placeholder="you@example.com")
    grade = st.text_input("Grade level", value=st.session_state.signup_data.get("grade", ""),
                          placeholder="e.g. O Level Year 2")
 
    _, c2 = st.columns([1, 1])
    if c2.button("Next: Subjects →", type="primary"):
        if not (roll.strip() and name.strip() and email.strip()):
            st.warning("Candidate number, name and email are required.")
        elif "@" not in email:
            st.warning("Please enter a valid email address.")
        else:
            st.session_state.signup_data.update({
                "roll": roll.strip(), "name": name.strip(),
                "email": email.strip(), "grade": grade.strip(),
            })
            st.session_state.signup_step = 2
            st.rerun()
 
 
def _step_subjects():
    st.subheader("Step 2 — Pick your subjects & preferred teachers")
    st.caption("Choose every subject you're taking. For each one you can optionally pick "
              "the teacher you'd like, and rank how important that pick is to you.")
 
    existing_map = st.session_state.signup_data.get("subjects", {})
    chosen = st.multiselect("Which subjects are you taking?", SUBJECTS,
                            default=list(existing_map.keys()))
 
    new_map = {}
    for subj in chosen:
        teachers = get_teachers_for_subject(subj)
        names = ["No preference"] + [t[1] for t in teachers]
        prev = existing_map.get(subj, {})
        default_idx = names.index(prev["teacher_name"]) if prev.get("teacher_name") in names else 0
 
        col1, col2 = st.columns([2, 1])
        with col1:
            t_choice = st.selectbox(f"Preferred teacher — {subj}", names,
                                    index=default_idx, key=f"tch_{subj}")
        with col2:
            pr = st.number_input("Priority", min_value=1, max_value=5,
                                 value=prev.get("priority", 1), step=1, key=f"pr_{subj}")
 
        teacher_id = None
        if t_choice != "No preference":
            teacher_id = next(t[0] for t in teachers if t[1] == t_choice)
        new_map[subj] = {"teacher_id": teacher_id, "teacher_name": t_choice, "priority": int(pr)}
 
    c1, c2 = st.columns([1, 1])
    if c1.button("← Back"):
        if st.session_state.editing_prefs:
            st.session_state.editing_prefs = False
        else:
            st.session_state.signup_step = 1
        st.rerun()
    if c2.button("Next: Review →", type="primary"):
        if not chosen:
            st.warning("Pick at least one subject.")
        else:
            st.session_state.signup_data["subjects"] = new_map
            st.session_state.signup_step = 3
            st.rerun()
 
 
def _step_confirm(model):
    st.subheader("Step 3 — Review & confirm")
    data = st.session_state.signup_data
    subjects = data.get("subjects", {})
 
    initials = "".join(p[0].upper() for p in (data.get("name") or "?").split()[:2]) or "?"
    badges = "".join(
        f'<span class="sh-subject-badge">{s} · {v["teacher_name"]}</span>'
        for s, v in subjects.items()
    )
    st.markdown(f"""
    <div class="sh-id-card">
      <div class="sh-id-avatar">{initials}</div>
      <div class="sh-id-name">{data.get('name', '')}</div>
      <div class="sh-id-meta">{data.get('roll', '')} · {data.get('grade') or '—'} · {data.get('email', '')}</div>
      <div>{badges}</div>
    </div>
    """, unsafe_allow_html=True)
 
    st.write("")
    c1, c2 = st.columns([1, 1])
    if c1.button("← Back"):
        st.session_state.signup_step = 2
        st.rerun()
    if c2.button("✅ Confirm & join", type="primary"):
        if st.session_state.get("student_id"):
            student_id = st.session_state.student_id
        else:
            student_id = register_student(data["name"], data["roll"], data.get("grade", ""), data["email"])
            if student_id is None:
                existing = get_student_by_roll(data["roll"])
                student_id = existing[0] if existing else None
            if student_id is None:
                st.error("That candidate number is already registered. Please double-check it.")
                return
            st.session_state.student_id = student_id
            st.session_state.student_name = data["name"]
 
        for subj, v in subjects.items():
            submit_preference(student_id, subj, v["teacher_id"], v["priority"])
 
        if ai_ready(model) and not st.session_state.get("welcome_note"):
            with st.spinner("Your AI concierge is writing you a welcome note…"):
                note = generate_welcome_note(model, data.get("name", ""), list(subjects.keys()))
            if note:
                st.session_state["welcome_note"] = note
 
        st.session_state.editing_prefs = False
        st.balloons()
        st.rerun()
 
 
def _signup_dashboard():
    st.markdown(f"### 🎉 You're all set, {st.session_state.student_name}!")
 
    if st.session_state.get("welcome_note"):
        st.markdown(f'<div class="sh-note">💬 {st.session_state["welcome_note"]}</div>',
                    unsafe_allow_html=True)
 
    rows = view_student_preferences(st.session_state.student_id)
    if rows:
        st.markdown("#### Your enrolled subjects")
        for subject_name, teacher_name, priority in rows:
            st.write(f"{priority}. **{subject_name}** → {teacher_name or 'No preference'}")
    else:
        st.info("You haven't picked any subjects yet.")
 
    st.divider()
    c1, c2 = st.columns([1, 1])
    if c1.button("➕ Update subjects / teachers"):
        srow = get_student_by_id(st.session_state.student_id)
        if srow:
            st.session_state.signup_data = {
                "name": srow[1], "roll": srow[2], "grade": srow[3] or "", "email": srow[4] or "",
                "subjects": {
                    subj: {"teacher_id": None, "teacher_name": teacher or "No preference", "priority": pr}
                    for subj, teacher, pr in rows
                },
            }
        st.session_state.editing_prefs = True
        st.session_state.signup_step = 2
        st.rerun()
    if c2.button("Switch student"):
        for k in ("student_id", "student_name", "signup_step", "signup_data",
                  "editing_prefs", "welcome_note"):
            st.session_state.pop(k, None)
        st.rerun()
 
 
def student_signup_wizard(model):
    """Entry point for the sign-up / preferences tab."""
    _inject_hero_css()
    _render_hero()
    _init_signup_state()
 
    if st.session_state.get("student_id") and not st.session_state.editing_prefs:
        _signup_dashboard()
        return
 
    step = st.session_state.signup_step
    _render_steps(step)
    st.write("")
 
    if step == 1:
        _step_details()
    elif step == 2:
        _step_subjects()
    elif step == 3:
        _step_confirm(model)
 
 
# --------------------------------------------------------------------------- #
def student_view(model):
    tab_lectures, tab_signup = st.tabs(["🎬 Lectures", "🚀 Sign up & preferences"])
 
    with tab_lectures:
        items = load_index()
        if not items:
            st.info("📭 No lectures yet. Ask your teacher to switch to **Teacher** mode and "
                    "upload one!")
        else:
            subjects = sorted({it["subject"] for it in items})
            subject = st.selectbox("📚 Choose a subject", subjects)
            in_subject = [it for it in items if it["subject"] == subject]
 
            titles = [it["title"] for it in in_subject]
            picked = st.selectbox("🎬 Choose a lecture", range(len(in_subject)),
                                  format_func=lambda i: titles[i])
            st.divider()
            _play_lecture(in_subject[picked], model)
 
    with tab_signup:
        student_signup_wizard(model)
 
 
# --------------------------------------------------------------------------- #
def main():
    st.set_page_config(page_title="O-Level Study Hub", page_icon="🎓", layout="wide")
 
    create_tables()
    seed_subjects()
 
    st.sidebar.title("🎓 O-Level Study Hub")
    role = st.sidebar.radio("I am a…", ["Student", "Teacher"], index=0)
    model = st.sidebar.selectbox("AI tutor model", list(MODELS.keys()))
    if not ai_ready(model):
        st.sidebar.warning("AI features are off (no key set) — video + notes still work.")
    st.sidebar.caption(f"{len(load_index())} lecture(s) available.")
 
    if role == "Teacher":
        st.title("👩‍🏫 Teacher dashboard")
        teacher_view()
    else:
        st.title("🎬 Study time!")
        st.caption("Pick a subject, watch the lecture, and study with your AI tutor.")
        student_view(model)
 
 
if __name__ == "__main__":
    main()
