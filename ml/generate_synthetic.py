r"""Generate synthetic resident requests with an LLM (English + Arabic + Egyptian).

Each generated request is written in 3 languages. All three rows share one
group_id, so make_splits.py keeps translations together (no train/val leakage).

Pick ONE provider (PowerShell). With no --provider the script auto-detects:
Azure env vars -> OPENAI_API_KEY -> ANTHROPIC_API_KEY -> local Ollama.

  # Anthropic (Claude)
  pip install anthropic
  $env:ANTHROPIC_API_KEY = "<key>"
  python ml\generate_synthetic.py --provider anthropic --groups-per-category 15

  # OpenAI (or any OpenAI-compatible server via OPENAI_BASE_URL)
  pip install openai
  $env:OPENAI_API_KEY = "<key>"
  python ml\generate_synthetic.py --provider openai --model <model name> --groups-per-category 15

  # Ollama: free, runs on your PC, no key (install from ollama.com first)
  ollama pull qwen2.5:7b
  python ml\generate_synthetic.py --provider ollama --model qwen2.5:7b --groups-per-category 15

  # Azure OpenAI
  $env:AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_DEPLOYMENT

  python ml\generate_synthetic.py --dry-run     # print one prompt, no API call
Output: data/processed/synthetic_pool.csv
"""
import argparse
import json
import math
import os
import random
import re
import time
import uuid

import pandas as pd

from config import PROC, SEED, CATEGORIES, URGENCY
from text_utils import clean_text, detect_lang

SCENARIOS = {
    "plumbing": ["kitchen sink clogged", "water leaking through the bathroom ceiling", "very low water pressure",
                 "toilet will not flush or keeps running", "no hot water (water heater)", "burst pipe, water flooding the floor",
                 "dripping tap", "bad smell coming from a drain", "washing machine drain leaking", "water pump or tank problem"],
    "electricity": ["whole unit has no power", "circuit breaker keeps tripping", "socket not working",
                    "sparks or burning smell from a socket", "light fixture not working", "lights flickering",
                    "exposed wires on a wall", "ceiling fan not working", "balcony or outdoor light out", "power cuts only in one room"],
    "ac": ["AC not cooling", "AC leaking water inside", "AC very noisy", "AC will not turn on / remote not working",
           "AC freezing up with ice", "thermostat not responding", "bad smell from the AC", "AC only cools one room",
           "AC trips the breaker", "heating not working in winter"],
    "carpentry": ["door will not close or lock", "window stuck and will not close", "wardrobe door hinge broken",
                  "kitchen cabinet drawer broken", "wooden floor lifting or swollen", "door handle broken",
                  "balcony door stuck", "roller shutter jammed", "swollen bedroom door", "skirting or door frame damaged"],
    "pest control": ["cockroaches in the kitchen", "ants everywhere", "mice or rats in the unit", "bed bugs",
                     "mosquitoes or flies infestation", "termites in wood", "wasps on the balcony",
                     "fleas after a pet", "rodents seen in the building", "insects coming from the drains"],
    "general": ["repaint a wall", "paint peeling", "crack in the wall or ceiling", "elevator not working",
                "intercom or door buzzer broken", "garbage not collected in the corridor", "stairwell needs cleaning",
                "mold on a wall", "compound gate remote not working", "garden or common-area maintenance request"],
}


def urgency_guide(category):
    """Urgency rules. AC-specific rules only appear in the AC prompt, so other
    categories are not nudged toward writing AC requests."""
    g = """urgency:
- high: safety risk or active damage (flooding, burst pipe, sparks or burning smell, gas smell, no power or no water for the whole unit, stuck elevator with a person inside)
- medium: important thing does not work but no immediate danger (one dead socket, clogged drain, door will not lock, pests seen)
- low: cosmetic or can wait (paint, small cracks, squeaky door, routine cleaning)
Most requests are medium. Use high only for real danger or active damage. About 1 in 5 should be low (cosmetic or can wait).
calibration: how long a problem has lasted does not change urgency by itself. Pest sightings = medium (low for a single ant or mosquito, high only for stings, bites or a large infestation). No hot water = medium. A door that will not lock = medium. Mixed-up or vague requests = medium."""
    if category == "ac":
        g += "\nAC not working = medium (high only with a health risk such as an elderly person or baby in extreme heat). AC noise, smell, remote or thermostat issues = low."
    return g


SYSTEM = (
    "You write realistic maintenance requests that residents of a residential compound in Egypt send to a "
    "property-management Telegram bot. Output valid JSON only."
)


LANG_LINES = {
    "en": '- "en": natural English',
    "ar": '- "ar": plain Modern Standard Arabic, Arabic script',
    "ar_eg": '- "ar_eg": Egyptian colloquial Arabic, Arabic script, the way people write on WhatsApp (not a literal translation)',
}

SHORT = {  # per-category examples of a very short request (illustration only)
    "plumbing": ('"sink is clogged"', '"الحوض مسدود"'),
    "electricity": ('"no power in the kitchen"', '"الكهربا قطعت في المطبخ"'),
    "ac": ('"room is boiling"', '"الاوضة نار"'),
    "carpentry": ('"bedroom door stuck"', '"الباب مش بيقفل"'),
    "pest control": ('"roaches again"', '"فيه صراصير"'),
    "general": ('"elevator is out"', '"الاسانسير عطلان"'),
}

INDIRECT = {  # symptom described without naming the appliance or trade
    "plumbing": "water on the kitchen floor and I can't tell where it comes from",
    "electricity": "half the flat went dark and the fridge stopped humming",
    "ac": "bedroom is boiling even with the thing on",
    "carpentry": "can't get out onto the balcony, something is stuck",
    "pest control": "something small keeps running along the kitchen wall at night",
    "general": "dark patches keep spreading on the bedroom wall",
}


def build_prompt(category, scenarios, n, hints, langs=("en", "ar", "ar_eg"), units=()):
    hint_txt = ""
    if hints:
        hint_txt = "\nReal complaint types seen in public 311 data for this category (for inspiration only): " + "; ".join(hints) + "\n"
    unit_txt = ", ".join(f'"{u}"' for u in units) or '"unit 204", "building C, flat 7"'
    lang_block = "\n".join(LANG_LINES[l] for l in langs)
    if len(langs) > 1:
        lang_intro = "Each request must be written several ways about the same problem, as the same person would say it:"
    else:
        lang_intro = "Each request is written as a resident would type it:"
    en_ex, ar_ex = SHORT.get(category, ('"it is broken"', '"عطلان"'))
    short_example = ar_ex if (len(langs) > 1 or langs[0] != "en") else en_ex
    json_keys = ", ".join(f'"{l}": "..."' for l in langs)
    indirect_ex = INDIRECT.get(category, "it stopped working")
    label = {"ac": "air conditioning"}.get(category, category)
    return f"""Write {n} different resident requests for the category "{label}".
Spread them across these situations: {"; ".join(scenarios)}.{hint_txt}
{lang_intro}
{lang_block}

Make the set realistic and varied:
- vary length: some very short ({short_example}), some detailed, some like a transcribed voice note (run-on, filler words)
- vary tone: polite, neutral, frustrated, worried
- include occasional typos or missing punctuation
- about 30% should mention where they live; when they do, use one of these references (and vary how they write it): {unit_txt}
- the problem must be clearly about {label} as the main issue; never include phone numbers or real names
- in about 40% of requests, describe what they see or feel without naming the appliance or the trade (e.g. "{indirect_ex}")
- do not begin more than 1 in 5 requests with the same word; some may have no greeting at all
- do not copy the example wording above, and never write the category name or the word "category" in a request
- assign urgency by these rules, and vary it realistically across items (do not make them all the same):
{urgency_guide(category)}

Return JSON exactly like:
{{"requests": [{{{json_keys}, "urgency": "low|medium|high"}}]}}"""


TIMEOUT = 900
DEFAULT_MODELS = {"anthropic": "claude-sonnet-5-5", "ollama": "qwen2.5:7b"}


def detect_provider():
    if os.environ.get("AZURE_OPENAI_ENDPOINT"):
        return "azure"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    return "ollama"


def parse_json(text):
    """Extract the JSON object even if the model wrapped it in text or code fences."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)  # Qwen3 reasoning blocks
    i, j = text.find("{"), text.rfind("}")
    if i == -1 or j == -1:
        raise ValueError("no JSON object in reply")
    return json.loads(text[i:j + 1])


def call_llm(provider, model, system, prompt):
    """Return the raw text reply. One function per provider; all lazy imports."""
    if provider == "anthropic":
        import anthropic
        r = anthropic.Anthropic().messages.create(
            model=model, max_tokens=8000, temperature=1.0, system=system,
            messages=[{"role": "user", "content": prompt}])
        return r.content[0].text
    if provider in ("openai", "azure"):
        if provider == "azure":
            from openai import AzureOpenAI
            client = AzureOpenAI(azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
                                 api_key=os.environ["AZURE_OPENAI_API_KEY"],
                                 api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21"))
            model = os.environ["AZURE_OPENAI_DEPLOYMENT"]
        else:
            from openai import OpenAI
            client = OpenAI(base_url=os.environ.get("OPENAI_BASE_URL") or None)
        r = client.chat.completions.create(
            model=model, temperature=1.0, response_format={"type": "json_object"},
            messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}])
        return r.choices[0].message.content
    if provider == "ollama":
        import urllib.request
        host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        body = json.dumps({"model": model, "stream": False, "format": "json", "think": False,
                           "options": {"temperature": 1.0, "num_ctx": 8192},
                           "messages": [{"role": "system", "content": system},
                                        {"role": "user", "content": prompt}]}).encode()
        req = urllib.request.Request(f"{host}/api/chat", body, {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            return json.loads(resp.read())["message"]["content"]
    raise SystemExit(f"Unknown provider: {provider}")


def request_batch(provider, model, category, scenarios, n, hints, langs=("en", "ar", "ar_eg"), units=(), retries=4):
    prompt = build_prompt(category, scenarios, n, hints, langs, units)
    for attempt in range(retries):
        try:
            t0 = time.time()
            items = parse_json(call_llm(provider, model, SYSTEM, prompt))["requests"]
            print(f"  call took {time.time() - t0:.0f}s, got {len(items)} requests")
            return items
        except Exception as e:  # rate limit, bad JSON, timeout, server down
            wait = 2 ** attempt * 2
            print(f"  retry {attempt + 1}/{retries} after error: {type(e).__name__}: {str(e)[:120]} (wait {wait}s)")
            time.sleep(wait)
    return []


# Keyword guard: catches generator drift (AC text under another category, or the
# prompt's category instruction leaking into the request text).
AC_WORDS = re.compile(r"\ba/?c\b|aircon|air[- ]?condition|cooling|thermostat|تكييف|تكيف", re.I)
LEAK = re.compile(r"categor|carpentry|pest control|pest thing|plumbing issue|electrical issue", re.I)
REJECTED = [0]


def rows_from_items(items, category, seed_tag, langs=("en", "ar", "ar_eg")):
    rows = []
    for it in items:
        if not isinstance(it, dict) or it.get("urgency") not in URGENCY:
            continue
        raw = " ".join(str(it.get(k, "")) for k in langs)
        if (category != "ac" and AC_WORDS.search(raw)) or LEAK.search(raw):
            REJECTED[0] += 1  # drop the whole group so translations stay consistent
            continue
        gid = f"syn_{uuid.uuid4().hex[:12]}"
        for key, lang in (("en", "en"), ("ar", "ar"), ("ar_eg", "ar")):
            if key not in langs:
                continue
            text = clean_text(it.get(key, ""))
            if len(text) < 4 or detect_lang(text) != lang:  # drop empty or wrong-language rows
                continue
            rows.append({"group_id": gid, "text": text, "category": category, "urgency": it["urgency"],
                         "language": lang, "source": f"synthetic_{key}", "combo": seed_tag})
    return rows


def load_units(rng, k=6):
    """Sample varied unit references from the mock registry (or invent some)."""
    p = PROC / "mock_units.csv"
    if p.exists():
        df = pd.read_csv(p).sample(k, random_state=rng.randint(0, 10**6))
        styles = [lambda r: r.unit, lambda r: f"unit {r.unit}", lambda r: f"{r.compound}, building {r.building}, {r.unit}"]
        return [rng.choice(styles)(r) for r in df.itertuples()]
    out = []
    for _ in range(k):
        b, f, u = rng.choice("ABCDEF"), rng.randint(1, 8), rng.randint(1, 4)
        out.append(rng.choice([f"{b}-{f}{u:02d}", f"unit {f}{u:02d}", f"building {b}, flat {f}{u:02d}", f"villa {rng.randint(1, 80)}"]))
    return out


def load_hints():
    p = PROC / "train_pool.csv"
    if not p.exists():
        return {}
    df = pd.read_csv(p)
    return {c: sorted(df[df.category == c]["combo"].dropna().unique())[:12] for c in CATEGORIES}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups-per-category", type=int, default=300)
    ap.add_argument("--per-call", type=int, default=None, help="default 15 (APIs) or 5 (ollama)")
    ap.add_argument("--provider", choices=["anthropic", "openai", "azure", "ollama"], default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--seed", type=int, default=SEED, help="change this for each extra batch so scenarios differ")
    ap.add_argument("--languages", default="en,ar,ar_eg", help="comma list from: en, ar, ar_eg (use 'en' on weak local models)")
    ap.add_argument("--timeout", type=int, default=900, help="seconds to wait per call (ollama)")
    ap.add_argument("--output", default=str(PROC / "synthetic_pool.csv"))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    rng = random.Random(a.seed)
    langs = tuple(l.strip() for l in a.languages.split(",") if l.strip())
    bad = set(langs) - set(LANG_LINES)
    if bad or not langs:
        raise SystemExit(f"--languages must be from en, ar, ar_eg (got {a.languages})")
    hints = load_hints()
    global TIMEOUT
    TIMEOUT = a.timeout
    provider = a.provider or detect_provider()
    model = a.model or DEFAULT_MODELS.get(provider)
    if provider == "openai" and not model:
        raise SystemExit("--model is required for --provider openai (the model name you have access to)")
    per_call = a.per_call or (5 if provider == "ollama" else 15)
    print(f"Provider: {provider}  model: {model or os.environ.get('AZURE_OPENAI_DEPLOYMENT')}  per-call: {per_call}")

    if a.dry_run:
        cat = "ac"
        print(build_prompt(cat, rng.sample(SCENARIOS[cat], 4), per_call, hints.get(cat, [])[:5], langs, load_units(rng)))
        return

    all_rows = []
    for cat in CATEGORIES:
        calls = math.ceil(a.groups_per_category / per_call)
        print(f"{cat}: {calls} calls")
        for i in range(calls):
            sc = rng.sample(SCENARIOS[cat], 4)
            items = request_batch(provider, model, cat, sc, per_call, rng.sample(hints.get(cat, []), min(5, len(hints.get(cat, [])))), langs, load_units(rng))
            all_rows += rows_from_items(items, cat, "synthetic|" + "|".join(sc[:2]), langs)
            print(f"  call {i + 1}/{calls}: total rows {len(all_rows)}")
        # save after every category so a crash does not lose work
        pd.DataFrame(all_rows).to_csv(a.output, index=False)

    df = pd.DataFrame(all_rows).drop_duplicates(subset=["text", "category"])
    df.to_csv(a.output, index=False)
    print(f"\nWrote {len(df):,} rows ({df.group_id.nunique():,} groups) -> {a.output}")
    print(f"Rejected by keyword guard: {REJECTED[0]} groups")
    print(df.groupby(["category", "urgency"]).size().unstack(fill_value=0))
    print(df["language"].value_counts())
    print("\nREAD 30 random rows before training:  python -c \"import pandas as pd; print(pd.read_csv(r'%s').sample(30, random_state=1).to_string())\"" % a.output)


if __name__ == "__main__":
    main()