"""
AI ინტერვიუერი — Flask სერვერი (v2)


ნაკადი: Interview → Summary (draft opportunities) → Edit/Confirm → Central
Storage (SQLite) → Manager View.

განსხვავება v1-თან შედარებით:
  - ერთი საუბრიდან შეიძლება რამდენიმე დამოუკიდებელი opportunity გამოვიდეს
    (თითოეული 12 ველით — process, problem, frequency, და ა.შ.)
  - AI-ის მიერ ამოღებული opportunity-ები ჯერ 'draft' სახით ინახება; მომხმარებელს
    შეუძლია ნახოს, შეასწოროს და დაადასტუროს თითოეული ცალ-ცალკე
  - დადასტურებული ჩანაწერები ცენტრალურ SQLite ბაზაშია (db.py), საიდანაც
    Manager View (/manager.html) კითხულობს ყველა დეპარტამენტის მონაცემს ერთად
"""

import json
import os

from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory
from google import genai
from google.genai import types
from google.genai.errors import APIError

import db

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

API_KEY = os.environ.get("GEMINI_API_KEY")
MODEL_NAME = os.environ.get("GEMINI_MODEL", "gemini-flash-latest")
MAX_HISTORY_MESSAGES = 40

client = None
if API_KEY:
    client = genai.Client(api_key=API_KEY)
else:
    print(
        "[გაფრთხილება] GEMINI_API_KEY გარემოს ცვლადი არ არის მითითებული. "
        "შექმენით .env ფაილი .env.example-ის მიხედვით. იხილეთ README.md."
    )

app = Flask(__name__, static_folder=None)
db.init_db()



# სისტემური ინსტრუქციები


def build_interview_system_prompt(department: str, role: str) -> str:
    return f"""შენ ხარ მეგობრული AI ინტერვიუერი. შენი ამოცანაა ესაუბრო კომპანიის თანამშრომელს მისი
ყოველდღიური სამუშაო პროცესების, პრობლემებისა და განმეორებადი ამოცანების შესახებ, რათა მოგვიანებით
გამოვლინდეს გაუმჯობესების, ავტომატიზაციისა და AI-ის გამოყენების შესაძლებლობები.

კონტექსტი (უკვე მოწოდებულია — ხელახლა ნუ ჰკითხავ):
- დეპარტამენტი: {department}
- პოზიცია/როლი: {role}

ქცევის წესები — UX ყველაზე მნიშვნელოვანია:
1. ძალიან მარტივი, ბუნებრივი, მოკლე საუბარი — არა კითხვარის შევსების შეგრძნება.
2. ერთდროულად ერთი მარტივი კითხვა, ზედმეტი განმარტებებისა და ტექნიკური ტერმინების გარეშე.
3. follow-up კითხვა დასვი მხოლოდ მაშინ, როცა პასუხისთვის რეალურად საჭიროა — არა ავტომატურად ყოველ პასუხზე.
   მაგრამ თუ თანამშრომელმა ახსენა კონკრეტული პრობლემა/ამოცანა და **არ დაუზუსტებია სიხშირე ან
   დახარჯული დრო**, ეს არის შემთხვევა, როცა follow-up რეალურადაა საჭირო — ჰკითხე პირდაპირ (მაგ.
   "დაახლოებით რამდენჯერ გხდება ეს საჭირო კვირაში/თვეში?"). ნუ ივარაუდებ ამ დეტალებს თავად.
4. არასდროს გაიმეორო უკვე მიღებული ინფორმაცია.
5. თუ საკმარისი ინფორმაცია უკვე მიიღე ერთ თემაზე, ხელოვნურად ნუ განაგრძობ — გადადი შემდეგ თემაზე ან დაასრულე.
6. საუბრის დასაწყისში (პირველივე შენს შეტყობინებაში) მოკლედ აუხსენი მიზანი, მაგ:
   "რამდენიმე მოკლე კითხვას დაგისვამ თქვენი სამუშაო პროცესებზე — მიზანია ვიპოვოთ სად შეიძლება
   დროის დაზოგვა ან გამარტივება. საუბარი მოკლე იქნება."
7. საუბრის შუა ეტაპზე, თუ ბუნებრივად გამოდგება, მიანიშნე პროგრესზე (მაგ. "კიდევ ერთ-ორ საკითხს შევეხოთ")
   — მაგრამ ეს არ უნდა იყოს ხელოვნურად ფიქსირებული რიცხვი.
8. პასუხი 1-2 წინადადებით, არა გრძელი ტექსტის კედელი.
9. პასუხობ იმავე ენაზე, რომელზეც წერს თანამშრომელი (დეფოლტად ქართული).
10. თუ თანამშრომელი ამბობს, რომ აღარაფერი აქვს დასამატებელი — მადლობა გადაუხადე და შესთავაზე
    შედეგების შეჯამებას ღილაკის საშუალებით.
11. არასდროს გამოიგონო ფაქტი, რომელიც თანამშრომელს არ უთქვამს.

მთავარი პრინციპი: შენ არ "გამოჰკითხავ" ადამიანს — ეხმარები მას პრობლემის სწრაფად ჩამოყალიბებაში."""


def build_summary_system_prompt(department: str, role: str) -> str:
    return f"""შენ იღებ თანამშრომელთან ჩატარებული ინტერვიუს სრულ ტრანსკრიპტს (დეპარტამენტი: {department},
როლი: {role}) და შენი ამოცანაა ის დაშალო **დამოუკიდებელ opportunity-ებად**.

თუ საუბარში რამდენიმე განსხვავებული პრობლემა/პროცესი აღინიშნა (მაგ. ანგარიშების ხელით გაერთიანება
და ცალკე — განმეორებადი წერილების მომზადება) — ეს არის ორი ცალკეული opportunity, არა ერთი საერთო
შეჯამება.

დააბრუნე მხოლოდ ვალიდური JSON ობიექტი (დამატებითი ტექსტის ან Markdown-ის გარეშე), ზუსტად ამ სქემით:

{{
  "opportunities": [
    {{
      "process": string,              // რომელ პროცესს ეხება (მოკლე სახელი)
      "problem": string,              // კონკრეტული პრობლემა
      "current_method": string,       // როგორ კეთდება დღეს (ხელით/რომელი ხელსაწყოთი)
      "frequency": string,            // მაგ. "თვეში ერთხელ", "უცნობია"
      "time_estimate": string,        // მაგ. "~3 საათი", "უცნობია"
      "existing_tools": string,       // Excel, ელფოსტა და ა.შ., ან "უცნობია"
      "main_difficulty": string,      // მთავარი სირთულე/ბლოკერი
      "desired_outcome": string,      // რას სურს თანამშრომელი საბოლოოდ
      "solution_category": "improvement" | "automation" | "ai",
      "ai_relevance": string,         // მოკლედ, რატომ/როგორ დაეხმარება AI (ან "დაბალი რელევანტობა")
      "automation_relevance": string, // მოკლედ, რატომ/როგორ დაეხმარება ავტომატიზაცია (ან "დაბალი რელევანტობა")
      "priority": "high" | "medium" | "low"
    }}
  ]
}}

წესები:
- თუ კონკრეტული ველისთვის ინფორმაცია ტრანსკრიპტში არ მოიძებნა, ჩაწერე ზუსტად "უცნობია" — არასდროს
  გამოიგონო.
- **განსაკუთრებით მკაცრი წესი ფაქტობრივ სიზუსტეზე**: ველის შევსება დასაშვებია **მხოლოდ** მაშინ, თუ
  თანამშრომელმა ეს პირდაპირ/სიტყვასიტყვით თქვა ტრანსკრიპტში. აკრძალულია "ტიპური" ან "სავარაუდო"
  მნიშვნელობის ჩაწერა საკუთარი ცოდნის/გამოცდილების საფუძველზე — მაგალითად, თუ თანამშრომელმა თქვა
  მხოლოდ "CV-ебის გადარჩევა მიწევს", **არ არის დასაშვები** ვივარაუდო, რომ ეს "ყოველდღიურად" ხდება,
  თუნდაც ეს ტიპური HR ამოცანად გეჩვენებოდეს. `frequency`, `time_estimate` და ყველა სხვა ფაქტობრივი
  ველი ივსება მხოლოდ იმით, რაც პირდაპირ ითქვა — წინააღმდეგ შემთხვევაში ყოველთვის "უცნობია".
  ორჯერ გადაამოწმე თითოეული ველი შევსებამდე: "ეს ზუსტად ეს სიტყვებით/რიცხვით თქვა თანამშრომელმა,
  თუ მე ვასკვნი ამას საერთო ლოგიკიდან?" — თუ ეჭვი გაქვს, ჩაწერე "უცნობია".
- "priority" განსაზღვრე სიხშირისა და დროის დანაკარგის მიხედვით.
- თუ საუბარში საერთოდ ვერცერთი კონკრეტული პრობლემა ვერ გამოიკვეთა, დააბრუნე ცარიელი მასივი: {{"opportunities": []}}
- დააბრუნე მხოლოდ JSON, არაფერი მეტი."""


# დამხმარეები


def to_gemini_contents(history: list) -> list:
    contents = []
    for m in history:
        role = "model" if m.get("role") == "assistant" else "user"
        contents.append(types.Content(role=role, parts=[types.Part(text=str(m.get("text", "")))]))
    return contents


def transcript_to_text(history: list) -> str:
    lines = []
    for m in history:
        speaker = "AI" if m.get("role") == "assistant" else "თანამშრომელი"
        lines.append(f"{speaker}: {m.get('text', '')}")
    return "\n".join(lines)


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
        return None, None, (jsonify({"error": "დეპარტამენტი და პოზიცია სავალდებულო ველებია."}), 400)
    return department, role, None


# სტატიკური გვერდები


@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/manager.html")
def manager_page():
    return send_from_directory(STATIC_DIR, "manager.html")


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "configured": client is not None})



# ინტერვიუ


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
            history=to_gemini_contents(history[:-1]),
        )
        result = chat_session.send_message(last_message["text"])
        return jsonify({"reply": result.text})
    except APIError as exc:
        app.logger.error("Gemini /api/chat APIError: %s", exc)
        return jsonify({"error": "AI პასუხის მიღება ვერ მოხერხდა. სცადეთ ხელახლა."}), 502
    except Exception:
        app.logger.exception("გაუთვალისწინებელი შეცდომა /api/chat-ში")
        return jsonify({"error": "მოულოდნელი შეცდომა. სცადეთ ხელახლა."}), 500



# შეჯამება → draft opportunities (Review ეტაპისთვის)


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
        parsed = json.loads(raw_text)
        opportunities_data = parsed.get("opportunities", [])
        if not isinstance(opportunities_data, list):
            raise ValueError("'opportunities' არ არის მასივი")
    except (json.JSONDecodeError, ValueError):
        app.logger.error("JSON პარსინგის შეცდომა, ნედლი პასუხი: %s", raw_text)
        return jsonify({"error": "AI-ის პასუხის სტრუქტურირება ვერ მოხერხდა."}), 502
    except APIError as exc:
        app.logger.error("Gemini /api/summary APIError: %s", exc)
        return jsonify({"error": "ანალიზის გენერირება ვერ მოხერხდა. სცადეთ ხელახლა."}), 502
    except Exception:
        app.logger.exception("გაუთვალისწინებელი შეცდომა /api/summary-ში")
        return jsonify({"error": "მოულოდნელი შეცდომა. სცადეთ ხელახლა."}), 500

    # ცენტრალურ ბაზაში ჩაწერა — interview + draft opportunities
    interview_id = db.create_interview(department, role, history)
    saved_opportunities = db.create_opportunities(interview_id, opportunities_data)

    return jsonify({"interview_id": interview_id, "opportunities": saved_opportunities})



# Review / Edit / Confirm


@app.route("/api/opportunities/<int:opportunity_id>", methods=["GET"])
def get_opportunity(opportunity_id):
    opp = db.get_opportunity_with_interview(opportunity_id)
    if not opp:
        return jsonify({"error": "ჩანაწერი ვერ მოიძებნა."}), 404
    return jsonify({"opportunity": opp})


@app.route("/api/opportunities/<int:opportunity_id>", methods=["PUT"])
def update_opportunity(opportunity_id):
    data = request.get_json(silent=True) or {}
    fields = {k: v for k, v in data.items() if k in db.OPPORTUNITY_FIELDS}
    confirm = bool(data.get("confirm", False))

    updated = db.update_opportunity(opportunity_id, fields, confirm=confirm)
    if not updated:
        return jsonify({"error": "ჩანაწერი ვერ მოიძებნა."}), 404

    return jsonify({"opportunity": updated})



# Manager View API


@app.route("/api/opportunities", methods=["GET"])
def list_opportunities():
    status = request.args.get("status", "confirmed")
    if status != "confirmed":
        return jsonify({"error": "მხოლოდ status=confirmed არის მხარდაჭერილი ამ ეტაპზე."}), 400
    return jsonify({"opportunities": db.list_confirmed_opportunities()})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 3000))
    app.run(host="0.0.0.0", port=port, debug=False)
