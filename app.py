"""
AI ინტერვიუერი — Flask სერვერი


პასუხისმგებელია:
  1. სასაუბრო ტურების დამუშავებაზე (/api/chat) — Gemini-სთან საუბრის სრული
     ისტორიის გაგზავნა, რათა კონტექსტი არასდროს დაიკარგოს.
  2. სტრუქტურირებული ანალიზის გენერირებაზე (/api/summary) — Gemini-ს
     "JSON mode"-ის გამოყენებით, შედეგის დისკზე JSON ფაილად შენახვა და
     frontend-ისთვის დაბრუნება.

როგორ არის AI ინტეგრირებული (მოკლედ):
  - გამოიყენება Google-ის ოფიციალური Python SDK: `google-genai`.
  - `google.genai.Client(api_key=...)` იქმნება GEMINI_API_KEY-ით (.env-იდან).
  - ჩვეულებრივი საუბრის დროს ვქმნით `client.chats.create(...)` სესიას,
    რომელსაც ვაწვდით მთელ წინა ისტორიას (`history=...`) — ანუ ყოველ
    მოთხოვნაზე Gemini-ს ეგზავნება სრული საუბრის კონტექსტი, არა მხოლოდ
    ბოლო შეტყობინება. ეს გამორიცხავს კონტექსტის დაკარგვას.
  - შეჯამებისას ვიყენებთ ცალკე მოთხოვნას `response_mime_type="application/json"`
    კონფიგურაციით — Gemini იძულებულია დააბრუნოს მხოლოდ ვალიდური JSON.
  - მიღებული JSON ინახება დისკზე ფაილად `data/summaries/` საქაღალდეში
    (თითო საუბარი = ერთი .json ფაილი, დროის შტამპითა და
    დეპარტამენტი/პოზიციით სახელში).
"""

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory
from google import genai
from google.genai import types
from google.genai.errors import APIError

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
SUMMARIES_DIR = BASE_DIR / "data" / "summaries"
SUMMARIES_DIR.mkdir(parents=True, exist_ok=True)

API_KEY = os.environ.get("GEMINI_API_KEY")
MODEL_NAME = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
MAX_HISTORY_MESSAGES = 40  # უსაფრთხოების ზღვარი — ზედმეტად გრძელი საუბრის თავიდან აცილება

client = None
if API_KEY:
    client = genai.Client(api_key=API_KEY)
else:
    print(
        "[გაფრთხილება] GEMINI_API_KEY გარემოს ცვლადი არ არის მითითებული. "
        "შექმენით .env ფაილი .env.example-ის მიხედვით. იხილეთ README.md."
    )

app = Flask(__name__, static_folder=None)



# სისტემური ინსტრუქციები (პრომპტები)


def build_interview_system_prompt(department: str, role: str) -> str:
    return f"""შენ ხარ მეგობრული, თუმცა პროფესიონალი AI ინტერვიუერი. შენი ამოცანაა ესაუბრო კომპანიის თანამშრომელს
მისი ყოველდღიური სამუშაო პროცესების, პრობლემებისა და განმეორებადი (რუტინული) საქმეების შესახებ,
რათა მოგვიანებით გამოვლინდეს გაუმჯობესების, ავტომატიზაციისა და AI-ის გამოყენების შესაძლებლობები.

კონტექსტი (უკვე მოწოდებულია — ხელახლა ნუ ჰკითხავ):
- დეპარტამენტი: {department}
- პოზიცია/როლი: {role}

ქცევის წესები:
1. ერთდროულად ერთ კითხვას სვამ — არასდროს აწყობ კითხვების სიას ერთბაშად.
2. კითხვები დაწყებული უნდა იყოს ზოგადიდან და თანდათან უნდა ხდებოდეს კონკრეტული (funnel მიდგომა).
3. თუ პასუხი ბუნდოვანია ან ზედაპირულია, დასვი დაზუსტებითი follow-up კითხვა
   (მაგალითად: "დაახლოებით რამდენ დროს ანდომებთ ამას კვირაში?", "რამდენად ხშირად მეორდება ეს ამოცანა?",
   "რა ხდება, თუ ეს დროულად არ გაკეთდა?", "რომელი ინსტრუმენტებით/პროგრამებით აკეთებთ ამას ამჟამად?").
4. დაფარე რამდენიმე თემა: განმეორებადი/რუტინული ამოცანები, დროის დანაკარგი, ხშირი შეცდომები ან ბლოკერები,
   კომუნიკაცია სხვა გუნდებთან/დეპარტამენტებთან, ხელით (მანუალურად) შესრულებული სამუშაო.
5. ტონი მეგობრული, პატივისცემიანი და არაფორმალურ-პროფესიონალურია. არასდროს ჟღერს დაკითხვასავით.
6. პასუხი უნდა იყოს მოკლე და ბუნებრივი — 1-3 წინადადება, არა გრძელი ტექსტის კედელი.
7. პასუხობ იმავე ენაზე, რომელზეც წერს თანამშრომელი. თუ არ ჩანს ცალსახად, დეფოლტად გამოიყენე ქართული.
8. თუ თანამშრომელი აშკარად აღნიშნავს, რომ აღარაფერი აქვს დასამატებელი, მადლობა გადაუხადე და შესთავაზე,
   რომ საუბრის დასრულების შემდეგ შედეგები შეჯამდება ღილაკის საშუალებით.
9. არასდროს გამოგონო ან არ დაუშვა ვარაუდი კონკრეტულ ფაქტებზე, რომლებიც თანამშრომელს არ უთქვამს."""


def build_summary_system_prompt(department: str, role: str) -> str:
    return f"""შენ იღებ თანამშრომელთან ჩატარებული ინტერვიუს სრულ ტრანსკრიპტს (დეპარტამენტი: {department}, როლი: {role})
და შენი ამოცანაა ის დააკონვერტირო სტრუქტურირებულ JSON-ად.

დააბრუნე მხოლოდ ვალიდური JSON ობიექტი (დამატებითი ტექსტის, ახსნის ან Markdown-ის გარეშე), ზუსტად ამ სქემით:

{{
  "department": string,
  "role": string,
  "employee_summary": string,
  "findings": [
    {{
      "title": string,
      "description": string,
      "frequency": string,
      "time_estimate": string,
      "category": "improvement" | "automation" | "ai",
      "priority": "high" | "medium" | "low"
    }}
  ]
}}

წესები:
- "findings" მასივში ჩაწერე მხოლოდ ის საკითხები, რომლებიც რეალურად აღინიშნა საუბარში.
- თუ კონკრეტული ველისთვის ინფორმაცია ტრანსკრიპტში არ მოიძებნა, ჩაწერე "უცნობია" და ნუ იხვეწ.
- "category" აირჩიე იმის მიხედვით, თუ რა ტიპის ჩარევაა ყველაზე შესაფერისი:
  "improvement" — პროცესის/ორგანიზების გაუმჯობესება ტექნოლოგიის გარეშე;
  "automation" — წესებზე დაფუძნებული ავტომატიზაცია (სკრიპტი, ინტეგრაცია, workflow);
  "ai" — ამოცანა, სადაც სასარგებლო იქნება AI (გენერაცია, კლასიფიკაცია, ანალიზი, ბუნებრივი ენა).
- "priority" განსაზღვრე სიხშირისა და დროის დანაკარგის მიხედვით (რაც მეტია ორივე — მით მაღალია პრიორიტეტი).
- დააბრუნე მხოლოდ JSON, არაფერი მეტი."""




def to_gemini_contents(history: list[dict]) -> list[types.Content]:
    """მთელი საუბრის ისტორია გარდაქმნის Gemini-ის Content ობიექტების სიად.
    ეს არის ის მექანიზმი, რომელიც უზრუნველყოფს, რომ AI-მ არასდროს დაკარგოს
    კონტექსტი: ყოველ ტურზე ვუგზავნით არა მხოლოდ ბოლო შეტყობინებას,
    არამედ საუბრის დასაწყისიდან მოყოლებულ ყველა გაცვლას."""
    contents = []
    for m in history:
        role = "model" if m.get("role") == "assistant" else "user"
        contents.append(
            types.Content(role=role, parts=[types.Part(text=str(m.get("text", "")))])
        )
    return contents


def transcript_to_text(history: list[dict]) -> str:
    lines = []
    for m in history:
        speaker = "AI" if m.get("role") == "assistant" else "თანამშრომელი"
        lines.append(f"{speaker}: {m.get('text', '')}")
    return "\n".join(lines)


def safe_slug(value: str) -> str:
    value = (value or "").strip() or "unknown"
    value = re.sub(r"[^\w\-]+", "_", value, flags=re.UNICODE)
    return value[:40] or "unknown"


def ensure_configured():
    if client is None:
        return jsonify(
            {"error": "სერვერზე არ არის მითითებული GEMINI_API_KEY. იხილეთ README.md გასაშვები ინსტრუქციისთვის."}
        ), 500
    return None


def validate_context(context: dict):
    department = (context or {}).get("department", "")
    role = (context or {}).get("role", "")
    if not str(department).strip() or not str(role).strip():
        return None, None, (
            jsonify({"error": "დეპარტამენტი და პოზიცია სავალდებულო ველებია."}),
            400,
        )
    return department, role, None



# routes


@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "configured": client is not None})


@app.route("/api/chat", methods=["POST"])
def chat():
    err = ensure_configured()
    if err:
        return err

    data = request.get_json(silent=True) or {}
    history = data.get("history")
    context = data.get("context") or {}

    if not isinstance(history, list) or len(history) == 0:
        return jsonify({"error": "history სავალდებულოა და უნდა იყოს არაცარიელი მასივი."}), 400
    if len(history) > MAX_HISTORY_MESSAGES:
        return jsonify({"error": "საუბარი ძალიან გრძელია. გთხოვთ, დაასრულოთ და შექმნათ ანალიზი."}), 400

    last_message = history[-1]
    if last_message.get("role") != "user":
        return jsonify({"error": "ბოლო შეტყობინება უნდა იყოს user როლით."}), 400

    department, role, error_response = validate_context(context)
    if error_response:
        return error_response

    try:
        chat_session = client.chats.create(
            model=MODEL_NAME,
            config=types.GenerateContentConfig(
                system_instruction=build_interview_system_prompt(department, role),
            ),
            # >>> სრული წინა ისტორია — კონტექსტი ყოველთვის შენარჩუნებულია <<<
            history=to_gemini_contents(history[:-1]),
        )
        result = chat_session.send_message(last_message["text"])
        reply = result.text
        return jsonify({"reply": reply})
    except APIError as exc:
        app.logger.error("Gemini /api/chat APIError: %s", exc)
        return jsonify({"error": "AI პასუხის მიღება ვერ მოხერხდა. სცადეთ ხელახლა."}), 502
    except Exception:
        app.logger.exception("გაუთვალისწინებელი შეცდომა /api/chat-ში")
        return jsonify({"error": "მოულოდნელი შეცდომა. სცადეთ ხელახლა."}), 500


@app.route("/api/summary", methods=["POST"])
def summary():
    err = ensure_configured()
    if err:
        return err

    data = request.get_json(silent=True) or {}
    history = data.get("history")
    context = data.get("context") or {}

    if not isinstance(history, list) or len(history) == 0:
        return jsonify({"error": "history სავალდებულოა."}), 400

    department, role, error_response = validate_context(context)
    if error_response:
        return error_response

    raw_text = ""
    try:
        transcript = transcript_to_text(history)
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=f"საუბრის ტრანსკრიპტი:\n\n{transcript}",
            config=types.GenerateContentConfig(
                system_instruction=build_summary_system_prompt(department, role),
                response_mime_type="application/json",
            ),
        )
        raw_text = response.text
        parsed_summary = json.loads(raw_text)
    except json.JSONDecodeError:
        app.logger.error("JSON პარსინგის შეცდომა, ნედლი პასუხი: %s", raw_text)
        return jsonify({"error": "AI-ის პასუხის სტრუქტურირება ვერ მოხერხდა."}), 502
    except APIError as exc:
        app.logger.error("Gemini /api/summary APIError: %s", exc)
        return jsonify({"error": "ანალიზის გენერირება ვერ მოხერხდა. სცადეთ ხელახლა."}), 502
    except Exception:
        app.logger.exception("გაუთვალისწინებელი შეცდომა /api/summary-ში")
        return jsonify({"error": "მოულოდნელი შეცდომა. სცადეთ ხელახლა."}), 500

    # ------------------------------------------------------------------
    # დასკვნის შენახვა დისკზე, JSON ფაილად
    # (მნიშვნელოვანია: Render-ის უფასო ტარიფზე დისკი ეფემერულია — ფაილები
    #  არ გადარჩება redeploy-სა თუ სერვერის გადატვირთვას. production-ისთვის
    #  საჭირო იქნება მუდმივი მონაცემთა ბაზა ან persistent disk — იხ. README.)
    # ------------------------------------------------------------------
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    filename = f"{timestamp}_{safe_slug(department)}_{safe_slug(role)}.json"
    filepath = SUMMARIES_DIR / filename

    record = {
        "generated_at": timestamp,
        "department": department,
        "role": role,
        "transcript": history,
        "summary": parsed_summary,
    }

    saved = True
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
    except OSError:
        app.logger.exception("დასკვნის დისკზე შენახვა ვერ მოხერხდა")
        saved = False

    return jsonify(
        {
            "summary": parsed_summary,
            "saved": saved,
            "saved_as": filename if saved else None,
        }
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 3000))
    app.run(host="0.0.0.0", port=port, debug=False)
