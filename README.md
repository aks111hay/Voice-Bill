# Voice Bill (Gnani Great Indian AI Internship Challenge)

Speak a Hinglish order -> Gnani Prisma (STT) -> parse items -> bill -> Gnani Timbre (TTS) reads it back.

## Run
1. `pip install -r requirements.txt`
2. `cp .env.example .env` and add `GNANI_API_KEY` (from app.gnani.ai/voice).
   Optional: `GEMINI_API_KEY` or `ANTHROPIC_API_KEY` for smarter parsing. Without one, a rule parser is used.
3. `uvicorn app:app --reload` then open http://localhost:8000 (mic needs localhost or https).

## Notes
- Prices live in `catalog.json`; edit them freely.
- Totals are computed in code, never by the LLM.
- Use synthetic orders in your demo, no real customer data.

## Data and flow
- SQLite (`bill.db`, auto-created, seeded from `catalog.json`) holds prices, stock and saved orders. Delete `bill.db` to reset.
- Flow: speak order -> bill is read back with any unavailable items -> customer says haan/nahi -> saved and stock updated.
- Bread is seeded out of stock and eggs/Maggi low, so you can demo the "not available" messages.