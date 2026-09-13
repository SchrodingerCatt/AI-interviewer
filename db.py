"""
db.py — SQLite-ზე დაფუძნებული ცენტრალური საცავი.

ორი ცხრილი:
  - interviews:     ერთი მთლიანი ინტერვიუს სესია (ტრანსკრიპტი მთლიანად)
  - opportunities:  ერთი კონკრეტული, დამოუკიდებელი შესაძლებლობა (ერთ interview-ს
                    შეიძლება რამდენიმე opportunity ჰქონდეს)

Review → Edit → Confirm ნაკადი გატარებულია opportunities.status ველით:
  draft → (მომხმარებელმა შეასწორა? → edited) → confirmed

Manager View მხოლოდ 'confirmed' სტატუსის ჩანაწერებს კითხულობს.
"""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "data" / "app.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS interviews (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    department TEXT NOT NULL,
    role TEXT NOT NULL,
    transcript TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'in_progress',
    started_at TEXT NOT NULL,
    completed_at TEXT
);

CREATE TABLE IF NOT EXISTS opportunities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    interview_id INTEGER NOT NULL REFERENCES interviews(id),

    process TEXT,
    problem TEXT,
    current_method TEXT,
    frequency TEXT,
    time_estimate TEXT,
    existing_tools TEXT,
    main_difficulty TEXT,
    desired_outcome TEXT,
    solution_category TEXT,
    ai_relevance TEXT,
    automation_relevance TEXT,
    priority TEXT,

    status TEXT NOT NULL DEFAULT 'draft',
    edited_by_user INTEGER NOT NULL DEFAULT 0,
    confirmed_at TEXT,

    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""

OPPORTUNITY_FIELDS = [
    "process", "problem", "current_method", "frequency", "time_estimate",
    "existing_tools", "main_difficulty", "desired_outcome",
    "solution_category", "ai_relevance", "automation_relevance", "priority",
]


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@contextmanager
def get_conn():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_conn() as conn:
        conn.executescript(SCHEMA)


# ---------------------------------------------------------------------------
# Interviews
# ---------------------------------------------------------------------------

def create_interview(department: str, role: str, transcript: list) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            """INSERT INTO interviews (department, role, transcript, status, started_at, completed_at)
               VALUES (?, ?, ?, 'completed', ?, ?)""",
            (department, role, json.dumps(transcript, ensure_ascii=False), now_iso(), now_iso()),
        )
        return cur.lastrowid


def get_interview(interview_id: int):
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM interviews WHERE id = ?", (interview_id,)).fetchone()
        return dict(row) if row else None


# ---------------------------------------------------------------------------
# Opportunities
# ---------------------------------------------------------------------------

def create_opportunities(interview_id: int, opportunities: list) -> list:
    """ინახავს AI-ის მიერ ამოღებულ opportunity-ებს 'draft' სტატუსით.
    აბრუნებს ჩაწერილ სტრიქონებს (id-ებით), რომ frontend-მა Review ეტაპზე
    შეძლოს მათი რედაქტირება."""
    created = []
    with get_conn() as conn:
        for opp in opportunities:
            ts = now_iso()
            values = [opp.get(f, "უცნობია") for f in OPPORTUNITY_FIELDS]
            cur = conn.execute(
                f"""INSERT INTO opportunities
                    (interview_id, {", ".join(OPPORTUNITY_FIELDS)}, status, edited_by_user, created_at, updated_at)
                    VALUES (?, {", ".join(["?"] * len(OPPORTUNITY_FIELDS))}, 'draft', 0, ?, ?)""",
                [interview_id, *values, ts, ts],
            )
            created.append(cur.lastrowid)

    return [get_opportunity(oid) for oid in created]


def get_opportunity(opportunity_id: int):
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM opportunities WHERE id = ?", (opportunity_id,)).fetchone()
        return dict(row) if row else None


def update_opportunity(opportunity_id: int, fields: dict, confirm: bool = False):
    """ანახლებს opportunity-ის ველებს. თუ ერთი მაინც შეიცვალა AI-ის თავდაპირველ
    მნიშვნელობასთან შედარებით, edited_by_user=1 დაისმის. confirm=True-ს
    შემთხვევაში status='confirmed' და confirmed_at ივსება."""
    existing = get_opportunity(opportunity_id)
    if not existing:
        return None

    updates = {}
    changed = False
    for f in OPPORTUNITY_FIELDS:
        if f in fields and fields[f] != existing.get(f):
            updates[f] = fields[f]
            changed = True

    with get_conn() as conn:
        set_clauses = [f"{f} = ?" for f in updates]
        params = list(updates.values())

        if changed:
            set_clauses.append("edited_by_user = 1")

        set_clauses.append("updated_at = ?")
        params.append(now_iso())

        if confirm:
            set_clauses.append("status = 'confirmed'")
            set_clauses.append("confirmed_at = ?")
            params.append(now_iso())
        elif changed:
            set_clauses.append("status = 'edited'")

        params.append(opportunity_id)
        conn.execute(
            f"UPDATE opportunities SET {', '.join(set_clauses)} WHERE id = ?",
            params,
        )

    return get_opportunity(opportunity_id)


def list_confirmed_opportunities():
    """Manager View-სთვის — ყველა დადასტურებული opportunity, დაკავშირებულ
    დეპარტამენტთან/როლთან ერთად (JOIN interviews-თან)."""
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT o.*, i.department AS department, i.role AS role
               FROM opportunities o
               JOIN interviews i ON i.id = o.interview_id
               WHERE o.status = 'confirmed'
               ORDER BY o.confirmed_at DESC"""
        ).fetchall()
        return [dict(r) for r in rows]


def get_opportunity_with_interview(opportunity_id: int):
    """დეტალური ხედვისთვის — opportunity + მისი სრული ტრანსკრიპტი."""
    opp = get_opportunity(opportunity_id)
    if not opp:
        return None
    interview = get_interview(opp["interview_id"])
    opp["interview"] = interview
    if interview and interview.get("transcript"):
        try:
            opp["interview"]["transcript"] = json.loads(interview["transcript"])
        except (json.JSONDecodeError, TypeError):
            pass
    return opp
