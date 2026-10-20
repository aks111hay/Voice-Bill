"""Voice Bill: speak a Hinglish order -> Gnani Prisma STT -> parse -> bill -> Gnani Timbre TTS."""
import json, os, re
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

load_dotenv()
BASE = Path(__file__).parent
CATALOG = json.loads((BASE / "catalog.json").read_text(encoding="utf-8"))
BY_ID = {c["id"]: c for c in CATALOG}
GNANI = "https://api.vachana.ai"
E = os.getenv
app = FastAPI(title="Voice Bill")

# ---------- Gnani speech ----------
def gnani_headers():
    if not E("GNANI_API_KEY"):
        raise HTTPException(500, "GNANI_API_KEY missing in .env")
    return {"X-API-Key-ID": E("GNANI_API_KEY")}

async def stt(audio: bytes, lang: str) -> str:
    fmt = E("STT_FORMAT", "verbatim")
    data = {"language_code": lang, "format": fmt}
    if fmt == "verbatim" and lang == "en-IN":  # word boosting only works with verbatim
        words = sorted({a for c in CATALOG for a in c["aliases"] if re.fullmatch(r"[A-Za-z]+", a)})[:100]
        data.update(bias_list=json.dumps(words), bias_score="1.0")
    async with httpx.AsyncClient(timeout=60) as cx:
        r = await cx.post(f"{GNANI}/stt/v3", headers=gnani_headers(), data=data,
                          files={"audio_file": ("order.wav", audio, "audio/wav")})
    if r.status_code != 200:
        raise HTTPException(502, f"Gnani STT {r.status_code}: {r.text[:200]}")
    return r.json().get("transcript", "").strip()

async def tts(text: str, style: str) -> bytes:
    english = style == "english"
    body = {"text": text, "model": "timbre-v2.5",
            "voice": E("TTS_VOICE_ENGLISH", "Kaveri") if english else E("TTS_VOICE_HINGLISH", "Poorvi"),
            "language": "en-IN" if english else "hi-en", "speed": 1.0,
            "audio_config": {"sample_rate": 24000, "num_channels": 1, "sample_width": 2,
                             "encoding": "linear_pcm", "container": "wav"}}
    async with httpx.AsyncClient(timeout=60) as cx:
        r = await cx.post(f"{GNANI}/api/v1/tts/inference", headers=gnani_headers(), json=body)
    if r.status_code != 200:
        raise HTTPException(502, f"Gnani TTS {r.status_code}: {r.text[:200]}")
    return r.content

# ---------- Parsing: LLM first, rules as fallback ----------
UNITS = {"kg": "kg", "kilo": "kg", "kilogram": "kg", "किलो": "kg", "g": "g", "gm": "g", "gram": "g", "ग्राम": "g",
         "l": "l", "ltr": "l", "litre": "l", "liter": "l", "लीटर": "l", "ml": "ml", "packet": "packet",
         "pkt": "packet", "पैकेट": "packet", "piece": "piece", "pcs": "piece", "pc": "piece", "dozen": "dozen",
         "darjan": "dozen", "दर्जन": "dozen"}
NUMS = {"ek": 1, "एक": 1, "one": 1, "do": 2, "दो": 2, "two": 2, "teen": 3, "तीन": 3, "three": 3, "char": 4,
        "chaar": 4, "चार": 4, "four": 4, "paanch": 5, "panch": 5, "पांच": 5, "five": 5, "chhe": 6, "six": 6,
        "saat": 7, "seven": 7, "aath": 8, "eight": 8, "nau": 9, "nine": 9, "das": 10, "ten": 10,
        "aadha": .5, "adha": .5, "आधा": .5, "half": .5, "paav": .25, "pav": .25, "quarter": .25,
        "dedh": 1.5, "डेढ़": 1.5, "dhai": 2.5, "dhaai": 2.5, "ढाई": 2.5}
FILLER = {"ka", "ke", "ki", "of", "ko"}
TOK = re.compile(r"[0-9]+(?:\.[0-9]+)?|[A-Za-z\u0900-\u097F]+")

def to_num(t):
    if t in NUMS: return NUMS[t]
    try: return float(t)
    except ValueError: return None

def qty_near(toks, i, step, used):
    """Read '[qty] [unit]' (before) or '[unit] [qty]'-less '[qty] [unit]' (after) next to an item."""
    j, unit, qty = i + step, None, None
    while 0 <= j < len(toks) and toks[j] in FILLER and step < 0: j += step
    for _ in range(2):
        if not (0 <= j < len(toks)) or j in used: break
        t = toks[j]
        if t in UNITS and unit is None: unit = UNITS[t]
        elif to_num(t) is not None and qty is None: qty = to_num(t); used.add(j)
        else: break
        j += step
    return qty, unit

def rule_parse(text):
    toks = TOK.findall(text.lower())
    aliases = sorted(((tuple(TOK.findall(a.lower())), c["id"]) for c in CATALOG for a in c["aliases"]),
                     key=lambda x: -len(x[0]))
    hits, i = [], 0
    while i < len(toks):
        for a, cid in aliases:
            if tuple(toks[i:i + len(a)]) == a:
                hits.append([i, i + len(a), cid, None, None]); i += len(a) - 1; break
        i += 1
    used = set()
    after_first = bool(hits) and hits[0][0] == 0  # "chawal 2 kilo, dal 1 kilo" style
    order = [(1, 1), (-1, 0)] if after_first else [(-1, 0), (1, 1)]
    for step, end in order:
        for h in hits:
            if h[3] is None: h[3], h[4] = qty_near(toks, h[end] - (1 if end else 0) if step > 0 else h[0], step, used)
    return [{"id": h[2], "qty": h[3], "unit": h[4]} for h in hits], []

LLM_PROMPT = """You extract shop orders from Hindi/Hinglish/English voice transcripts.
Catalog (id: spoken names):
{cat}
Return ONLY JSON: {{"items":[{{"id":"<catalog id>","qty":<number>,"unit":"kg|g|l|ml|packet|piece|dozen|null"}}],"unmatched":["words not in catalog"]}}
Rules: use only catalog ids; convert number words (ek=1, do=2, teen=3, aadha=0.5, dedh=1.5, dhai=2.5, paav=0.25); if no quantity heard use null; never compute prices."""

async def llm_parse(text):
    prompt = LLM_PROMPT.format(cat="\n".join(f"{c['id']}: {', '.join(c['aliases'])}" for c in CATALOG))
    async with httpx.AsyncClient(timeout=30) as cx:
        if E("GEMINI_API_KEY"):
            m = E("GEMINI_MODEL", "gemini-2.5-flash")
            r = await cx.post(f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent",
                              params={"key": E("GEMINI_API_KEY")},
                              json={"systemInstruction": {"parts": [{"text": prompt}]},
                                    "contents": [{"role": "user", "parts": [{"text": text}]}],
                                    "generationConfig": {"temperature": 0, "responseMimeType": "application/json"}})
            r.raise_for_status(); raw, name = r.json()["candidates"][0]["content"]["parts"][0]["text"], "gemini"
        else:
            r = await cx.post("https://api.anthropic.com/v1/messages",
                              headers={"x-api-key": E("ANTHROPIC_API_KEY"), "anthropic-version": "2023-06-01"},
                              json={"model": E("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001"), "max_tokens": 600,
                                    "system": prompt, "messages": [{"role": "user", "content": text}]})
            r.raise_for_status(); raw, name = r.json()["content"][0]["text"], "claude"
    d = json.loads(re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M).strip())
    return d.get("items", []), d.get("unmatched", []), name

# ---------- Billing (plain code does the maths, never the LLM) ----------
def convert(qty, spoken, item_unit):
    if qty is None: return 1.0, "quantity not heard, assumed 1"
    if spoken in (None, item_unit): return float(qty), ""
    if (spoken, item_unit) in (("g", "kg"), ("ml", "l")): return qty / 1000, ""
    return float(qty), f"heard '{spoken}', item is sold per {item_unit} - check"

def build_bill(rows):
    merged, items, total = {}, [], Decimal(0)
    for r in rows:
        c = BY_ID.get(r.get("id"))
        if not c: continue
        qty, note = convert(r.get("qty"), r.get("unit"), c["unit"])
        m = merged.setdefault(c["id"], {"qty": 0.0, "note": ""})
        m["qty"] += qty; m["note"] = m["note"] or note
    for cid, m in merged.items():
        c, qty = BY_ID[cid], round(m["qty"], 3)
        line = (Decimal(str(qty)) * Decimal(str(c["price"]))).quantize(Decimal("0.01"), ROUND_HALF_UP)
        total += line
        items.append({"id": cid, "name": c["name"], "hi": c["hi"], "qty": qty, "unit": c["unit"],
                      "unit_price": c["price"], "line_total": float(line), "note": m["note"]})
    return {"items": items, "total": float(total)}

# ---------- Speech text (numbers as words, as Gnani recommends) ----------
ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()

def words(n):
    if n < 20: return ONES[n]
    if n < 100: return TENS[n // 10] + (" " + ONES[n % 10] if n % 10 else "")
    for div, name in ((10**7, "crore"), (10**5, "lakh"), (1000, "thousand"), (100, "hundred")):
        if n >= div: return words(n // div) + f" {name}" + (" " + words(n % div) if n % div else "")

def say_qty(q):
    whole, frac = int(q), round(q - int(q), 3)
    if frac == 0: return words(whole)
    f = {0.5: "half", 0.25: "quarter", 0.75: "three quarter"}.get(frac)
    if f: return f if whole == 0 else f"{words(whole)} and {f}"
    return f"{words(whole)} point " + " ".join(ONES[int(d)] for d in str(frac)[2:])

SAY_UNIT = {"kg": "kilo", "l": "litre", "packet": "packet", "piece": "piece", "dozen": "dozen"}

def speech(bill, style):
    en = style == "english"
    if not bill["items"]: return "I did not catch any items. Please try again." if en else "Koi item samajh nahi aaya. Dobara boliye."
    lines = ", ".join(f"{say_qty(i['qty'])} {SAY_UNIT[i['unit']]} {i['name'] if en else i['hi']}" for i in bill["items"])
    rupees, paise = divmod(round(bill["total"] * 100), 100)
    amt = words(rupees) + (f" rupees and {words(paise)} paise" if paise else " rupees")
    return (f"Your bill is ready. {lines}. Total {amt}. Thank you!" if en
            else f"Aapka bill taiyaar hai. {lines}. Total {amt}. Dhanyavaad!")

# ---------- Routes ----------
class Req(BaseModel):
    items: list[dict]
    style: str = "hinglish"

@app.post("/api/order")
async def order(lang: str = Form("en-IN"), text: str = Form(""), audio: UploadFile | None = File(None)):
    transcript = text.strip()
    if audio is not None:
        transcript = await stt(await audio.read(), lang)
    if not transcript:
        raise HTTPException(400, "No speech heard. Try again closer to the mic.")
    parser, rows, unmatched = "rules", None, []
    if E("GEMINI_API_KEY") or E("ANTHROPIC_API_KEY"):
        try: rows, unmatched, parser = await llm_parse(transcript)
        except Exception as ex: print("LLM parse failed, using rules:", ex)
    if rows is None: rows, unmatched = rule_parse(transcript)
    return {"transcript": transcript, "parser": parser, "unmatched": unmatched, **build_bill(rows)}

@app.post("/api/bill")
def recalc(req: Req): return build_bill(req.items)

@app.post("/api/speak")
async def speak(req: Req):
    return Response(await tts(speech(build_bill(req.items), req.style), req.style), media_type="audio/wav")

@app.get("/api/catalog")
def catalog(): return CATALOG

@app.get("/")
def index(): return FileResponse(BASE / "static" / "index.html")