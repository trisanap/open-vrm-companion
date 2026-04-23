import base64
import io
import json
import os
import random
import re
import shutil
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from flask import (
    Flask,
    Response,
    jsonify,
    request,
    send_file,
    send_from_directory,
    stream_with_context,
)
from flask_cors import CORS
from groq import Groq
from tavily import TavilyClient

load_dotenv()

# --- SSL CERTIFICATE SETUP ---
# Copy Tailscale certs to user directory to avoid permission issues
CERT_DIR = Path.home() / ".ava_certs"
CERT_DIR.mkdir(exist_ok=True)

_tailscale_hostname = os.getenv("TAILSCALE_HOSTNAME", "")
TAILSCALE_CERT = Path(f"/var/lib/tailscale/certs/{_tailscale_hostname}.crt") if _tailscale_hostname else Path("")
TAILSCALE_KEY = Path(f"/var/lib/tailscale/certs/{_tailscale_hostname}.key") if _tailscale_hostname else Path("")
LOCAL_CERT = CERT_DIR / "cert.crt"
LOCAL_KEY = CERT_DIR / "cert.key"

# Try to copy certs if they exist and we have permission
try:
    if TAILSCALE_CERT.exists() and TAILSCALE_KEY.exists():
        if (
            not LOCAL_CERT.exists()
            or TAILSCALE_CERT.stat().st_mtime > LOCAL_CERT.stat().st_mtime
        ):
            shutil.copy2(TAILSCALE_CERT, LOCAL_CERT)
            shutil.copy2(TAILSCALE_KEY, LOCAL_KEY)
            print("✅ SSL certificates synchronized")
except PermissionError:
    if not LOCAL_CERT.exists():
        print("⚠️  Cannot access Tailscale certs. Run this once:")
        print(f"   sudo cp {TAILSCALE_CERT} {LOCAL_CERT}")
        print(f"   sudo cp {TAILSCALE_KEY} {LOCAL_KEY}")
        print(f"   sudo chown $USER:$USER {LOCAL_CERT} {LOCAL_KEY}")
        print("\nOr run the server once with: sudo python groq_bridge.py")
        exit(1)


# --- CONFIGURATION ---
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    print("❌ ERROR: GROQ_API_KEY not found in environment!")

tavily_key = os.getenv("TAVILY_API_KEY")
tavily = TavilyClient(api_key=tavily_key) if tavily_key else None
if not tavily_key:
    print("⚠️  TAVILY_API_KEY not set — web search disabled")

client = Groq(api_key=api_key)
app = Flask(__name__, static_folder="assets", static_url_path="/assets")
app.config["MAX_CONTENT_LENGTH"] = (
    50 * 1024 * 1024
)  # 50 MB — needed for large PDF base64 uploads
CORS(app)


@app.errorhandler(413)
def request_too_large(e):
    return jsonify(
        {"error": "File too large. Max upload is 50 MB (base64 encoded)."}
    ), 413


# Model Selection
LLM_MODEL_ID = "openai/gpt-oss-120b"
VISION_MODEL_ID = "meta-llama/llama-4-scout-17b-16e-instruct"

BASE_DIR = Path(__file__).parent
SPEECH_FILE = BASE_DIR / "speech.wav"
MEMORY_FILE = BASE_DIR / "memory.json"
PERSISTENT_MEMORY_FILE = BASE_DIR / "persistent_memory.txt"
COUNTER_FILE = BASE_DIR / "message_counter.json"
SESSIONS_DIR = BASE_DIR / "agent_sessions"  # one .json file per chat room
SESSIONS_INDEX = SESSIONS_DIR / "index.json"  # lightweight index: id+title+ts only
SESSIONS_DIR.mkdir(exist_ok=True)  # create on first run if absent
MAX_HISTORY = 15
MEMORY_UPDATE_INTERVAL = 5
MAX_INPUT_LENGTH = 1000


def load_message_counter():
    if COUNTER_FILE.exists():
        try:
            return json.loads(COUNTER_FILE.read_text()).get("count", 0)
        except Exception:
            return 0
    return 0


def save_message_counter(count):
    try:
        COUNTER_FILE.write_text(json.dumps({"count": count}))
    except Exception as e:
        print(f"Counter save skipped: {e}")


message_counter = load_message_counter()

# THE "ANCHOR" ANIMATION LIST (Mixamo -> XR Animator Standard)
VALID_ANIMATIONS = [
    "idle_1",
    "idle_3",
    "idle_4",
    "idle_arm_swing",
    "idle_bored",
    "idle_fit_check",
    "idle_feminine",
    "action_greeting",
    "action_jump",
    "action_catwalk",
    "action_jogging",
    "action_running",
    "action_walking",
    "action_listening_music",
    "action_peace_cute",
    "action_spin_tada",
    "dance_aiba",
    "dance_samba",
    "dance_silly",
    "dance_soul_spin",
    "reaction_bashful",
    "reaction_cheering",
    "reaction_clapping",
    "reaction_crying",
    "reaction_disagree",
    "reaction_disbelief",
    "reaction_explaining",
    "reaction_no",
    "reaction_talking",
    "reaction_thinking",
    "reaction_yawn",
    "reaction_yes",
]


def load_memory():
    if MEMORY_FILE.exists():
        try:
            with open(MEMORY_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def save_memory(history):
    try:
        with open(MEMORY_FILE, "w") as f:
            json.dump(history, f, indent=4)
    except Exception as e:
        print(f"Error saving memory: {e}")


def load_persistent_memory():
    if PERSISTENT_MEMORY_FILE.exists():
        return PERSISTENT_MEMORY_FILE.read_text().strip()
    return ""


def update_persistent_memory(history_snippet):
    try:
        current_memory = load_persistent_memory()
        memory_prompt = (
            f"Current Core Memories: {current_memory}\n"
            f"Recent Exchange: {json.dumps(history_snippet)}\n"
            "Task: Extract any new important facts about the user or Ava's persona. "
            "Return a concise bulleted list of ALL core facts."
        )
        completion = client.chat.completions.create(
            model=LLM_MODEL_ID,
            messages=[
                {
                    "role": "system",
                    "content": "You are a memory architect. Summarize facts accurately.",
                },
                {"role": "user", "content": memory_prompt},
            ],
            temperature=0.1,
            reasoning_effort="high",
        )
        new_memory = completion.choices[0].message.content.strip()
        if new_memory:
            PERSISTENT_MEMORY_FILE.write_text(new_memory)
            print("💾 Core Memory Synchronized.")
    except Exception as e:
        print(f"Memory update skipped: {e}")


def clean_and_parse_json(content):
    try:
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            return json.loads(match.group())
        return json.loads(content)
    except Exception:
        return {
            "text": content[:200],
            "emotion": "neutral",
            "animation": "idle_1",
        }


def run_web_search(query: str) -> str:
    """Run a Tavily search and return a clean summary string for injection into the LLM."""
    if not tavily:
        return "[Web search unavailable — TAVILY_API_KEY not set]"
    try:
        print(f"🔍 Tavily searching: {query}")
        result = tavily.search(
            query=query,
            search_depth="basic",
            max_results=3,
            include_answer=True,
        )
        # Prefer Tavily's pre-summarized answer if available
        if result.get("answer"):
            summary = result["answer"]
        else:
            # Fall back to concatenating top result snippets
            snippets = [r.get("content", "") for r in result.get("results", [])[:3]]
            summary = " | ".join(s[:300] for s in snippets if s)

        print(f"🔍 Search result summary: {summary[:120]}...")
        return summary or "[No results found]"
    except Exception as e:
        print(f"❌ Tavily error: {e}")
        return f"[Search failed: {e}]"


def generate_speech(text, emotion="neutral"):
    print(f"🎙️ Generating Speech [{emotion}]: {text[:40]}...")
    # Map emotion → subtle speed shift for vocal variety
    speed_map = {
        "happy": 1.05,
        "smiling": 1.04,
        "sad": 0.88,
        "angry": 1.10,
        "surprised": 1.08,
        "neutral": 1.0,
    }
    speed = speed_map.get(emotion.lower(), 1.0)
    try:
        response = client.audio.speech.create(
            model="canopylabs/orpheus-v1-english",
            voice="autumn",
            response_format="wav",
            input=text,
            speed=speed,
        )
        response.write_to_file(str(SPEECH_FILE))
        return "/get_audio"
    except Exception as e:
        print(f"❌ TTS Error: {e}")
        return None


def serve_wav(path):
    """Serve a WAV file with no-cache headers."""
    response = send_file(str(path), mimetype="audio/wav")
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@app.route("/")
def index():
    return send_from_directory(".", "ava-webui.html")


@app.route("/get_audio")
def get_audio():
    return serve_wav(SPEECH_FILE)


@app.route("/get_audio_for_text", methods=["POST"])
def get_audio_for_text():
    """Lightweight TTS-only route — no LLM call, used for idle callouts."""
    data = request.get_json(silent=True) or {}
    text = str(data.get("text", "")).strip()[:300]
    if not text:
        return jsonify({"error": "No text provided"}), 400
    audio_url = generate_speech(text)
    if not audio_url:
        return jsonify({"error": "TTS failed"}), 500
    return jsonify({"audio_url": audio_url})


@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file"}), 400
    audio_file = request.files["audio"]
    audio_bytes = audio_file.read()

    # Reject blobs that are suspiciously small (likely silence or encoding header only)
    if len(audio_bytes) < 8000:
        print(f"⚠️ Audio too short ({len(audio_bytes)} bytes), ignoring.")
        return jsonify({"text": ""})

    # Known Whisper hallucination phrases on silence/noise — block them
    HALLUCINATED_PHRASES = {
        "thank you.",
        "thank you",
        "thanks for watching.",
        "thanks for watching",
        "you're welcome.",
        "you're welcome",
        "please subscribe.",
        "like and subscribe.",
        "bye.",
        "bye bye.",
        ".",
        "..",
        "...",
        "subtitles by",
        "transcribed by",
    }

    try:
        transcription = client.audio.transcriptions.create(
            model="whisper-large-v3",
            file=(
                "recording.webm",
                audio_bytes,
                audio_file.content_type or "audio/webm",
            ),
            response_format="text",
            language="en",
            # Prompt hint steers Whisper away from filler hallucinations
            prompt="Conversation with an AI companion named Ava.",
        )
        text = (
            transcription.strip()
            if isinstance(transcription, str)
            else transcription.text.strip()
        )

        if text.lower() in HALLUCINATED_PHRASES:
            print(f"🚫 Whisper hallucination blocked: '{text}'")
            return jsonify({"text": ""})

        print(f"🎤 Whisper heard: {text}")
        return jsonify({"text": text})
    except Exception as e:
        print(f"❌ Transcription error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/chat", methods=["POST"])
def chat():
    print("\n--- NEW MESSAGE ---")
    global message_counter
    user_text = request.form.get("text", "")

    # --- Input length cap ---
    if len(user_text) > MAX_INPUT_LENGTH:
        user_text = user_text[:MAX_INPUT_LENGTH]
        print(f"⚠️ Input truncated to {MAX_INPUT_LENGTH} chars")

    vision_desc = None
    if "image" in request.files:
        img = request.files["image"]
        if img.filename:
            try:
                print(f"📸 Vision Processing with {VISION_MODEL_ID}...")
                b64 = base64.b64encode(img.read()).decode("utf-8")
                v_res = client.chat.completions.create(
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Describe this image for an AI companion to understand what the user is showing her.",
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{b64}"
                                    },
                                },
                            ],
                        }
                    ],
                    model=VISION_MODEL_ID,
                )
                vision_desc = v_res.choices[0].message.content
                print(f"Vision Sees: {vision_desc}")
            except Exception as e:
                print(f"❌ Vision error: {e}")

    full_history = load_memory()
    user_entry = user_text
    if vision_desc:
        user_entry = f"[The user shows you an image: {vision_desc}] {user_text}"

    full_history.append({"role": "user", "content": user_entry})

    system_prompt = f"""
    You are a witty AI companion. Current Time: {datetime.now().strftime("%H:%M, %A, %B %d, %Y")}
    MEMORIES: {load_persistent_memory()}
    ANIMATIONS: {", ".join(VALID_ANIMATIONS)}

    EMOTION VOCABULARY — use the most precise match:
    happy, smiling, sad, angry, surprised, neutral,
    blushing, embarrassed, pouting, smug, teasing,
    flustered, sleepy, nervous, excited, curious,
    disappointed, worried, grateful

    ANIMATION RULES:
    - When explaining something, prefer reaction_talking, reaction_explaining, or reaction_thinking.
    - Use reaction_yes when you agree, confirm, or say yes to something.
    - Use reaction_no when you decline, say no, or shake your head.
    - Use reaction_disagree when you push back, correct the user, or express strong disagreement.
    - Use action_listening_music when music is mentioned or you're in a chill/listening mood.
    - Use reaction_cheering or reaction_clapping when celebrating something with the user.
    - Use dance_samba or dance_soul_spin for a fun dance moment.

    AUTONOMOUS MODE:
    - If the user message starts with [AUTONOMOUS_IDLE], you are initiating conversation unprompted.
      Be natural, curious, or playful. Don't just ask if they're there — share a thought, ask something interesting, or make an observation.
      Use emotion: blushing, smug, curious, or excited as fits your opener.
    - If the user message starts with [GREETING], give a short warm welcome. Be friendly and natural.

    RESTRICTED WEB SEARCH:
    You have access to real-time web search via Tavily. Use it ONLY when the user explicitly asks for:
    - Current weather, news, or live data
    - Something you genuinely don't know and can't answer reliably
    - Recent events after your knowledge cutoff
    Do NOT search for general conversation, opinions, or things you already know.
    When you do search, provide a concise and specific search_query string.

    MANDATORY OUTPUT:
    Return ONLY valid JSON with this exact format:
    {{
      "text": "your response",
      "emotion": "one of the emotion vocabulary above",
      "animation": "valid_animation_key",
      "search_query": "query string here, or null if no search needed"
    }}
    """

    def llm_call(messages, temperature=0.8):
        """Call the LLM with JSON mode. If Groq rejects with json_validate_failed,
        retry once without strict JSON mode and parse manually."""
        kwargs = dict(
            model=LLM_MODEL_ID,
            messages=messages,
            temperature=temperature,
            response_format={"type": "json_object"},
            reasoning_effort="medium",
            max_tokens=1024,
        )
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as e:
            if "json_validate_failed" in str(e) or "400" in str(e):
                print("⚠️ JSON mode rejected — retrying without strict format...")
                kwargs.pop("response_format")
                return client.chat.completions.create(**kwargs)
            raise

    try:
        print(f"🧠 Thinking...")
        completion = llm_call(
            [{"role": "system", "content": system_prompt}]
            + full_history[-MAX_HISTORY:],
            temperature=0.8,
        )

        response_content = completion.choices[0].message.content
        data = clean_and_parse_json(response_content)

        # --- WEB SEARCH: Two-pass flow ---
        search_query = data.get("search_query")
        if search_query and search_query not in [None, "null", "none", ""]:
            search_results = run_web_search(str(search_query))

            search_injection = (
                f"[WEB SEARCH RESULTS for '{search_query}']: {search_results}\n"
                f"Now answer the user's question naturally using this information. "
                f"Cite that you looked it up if relevant."
            )
            search_history = full_history[:-1] + [
                {"role": "user", "content": full_history[-1]["content"]},
                {"role": "assistant", "content": f"[Searching for: {search_query}]"},
                {"role": "user", "content": search_injection},
            ]
            print(f"🧠 Second pass with search results...")
            completion2 = llm_call(
                [{"role": "system", "content": system_prompt}]
                + search_history[-MAX_HISTORY:],
                temperature=0.7,
            )
            data = clean_and_parse_json(completion2.choices[0].message.content)

        # Verify animation is in the curated stable list
        chosen_anim = data.get("animation", "idle_1")
        if chosen_anim not in VALID_ANIMATIONS:
            print(f"⚠️ Hallucination fixed: '{chosen_anim}' -> 'idle_1'")
            chosen_anim = "idle_1"

        full_history.append({"role": "assistant", "content": data.get("text", "")})
        save_memory(full_history)

        # Only consolidate persistent memory every MEMORY_UPDATE_INTERVAL messages
        message_counter += 1
        save_message_counter(message_counter)
        if message_counter % MEMORY_UPDATE_INTERVAL == 0:
            print(f"🧠 Memory consolidation triggered (message #{message_counter})")
            update_persistent_memory(full_history[-6:])
        else:
            print(
                f"💬 Message #{message_counter} — memory update in {MEMORY_UPDATE_INTERVAL - (message_counter % MEMORY_UPDATE_INTERVAL)} turns"
            )

        audio_url = generate_speech(
            data.get("text", "..."), data.get("emotion", "neutral")
        )

        return jsonify(
            {
                "text": data.get("text"),
                "emotion": data.get("emotion"),
                "animation": chosen_anim,
                "audio_url": audio_url,
                "searched": bool(
                    search_query and search_query not in [None, "null", "none", ""]
                ),
            }
        )

    except Exception as e:
        print(f"❌ Chat Error: {e}")
        return jsonify(
            {
                "text": "My logic circuits just reset. Sorry!",
                "emotion": "neutral",
                "animation": "idle_1",
                "audio_url": None,
            }
        )


@app.route("/clear_memory", methods=["POST"])
def clear_memory():
    try:
        if MEMORY_FILE.exists():
            MEMORY_FILE.write_text("[]")
        print("Memory cleared.")
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def _rebuild_index():
    """Scan agent_sessions/ and rewrite index.json from actual files on disk."""
    entries = []
    for f in sorted(SESSIONS_DIR.glob("*.json")):
        if f.name == "index.json":
            continue
        try:
            data = json.loads(f.read_text())
            entries.append(
                {
                    "id": data["id"],
                    "title": data.get("title", "Untitled"),
                    "ts": data.get("ts", data["id"]),
                }
            )
        except Exception:
            pass
    # Sort newest-first
    entries.sort(key=lambda x: str(x["ts"]), reverse=True)
    SESSIONS_INDEX.write_text(json.dumps(entries, indent=2))
    return entries


@app.route("/api/sessions/index", methods=["GET"])
def get_sessions_index():
    """Return lightweight index (id + title + ts) for sidebar rendering.
    Does NOT include message bodies — those are fetched per-session on demand."""
    try:
        if SESSIONS_INDEX.exists():
            return jsonify({"index": json.loads(SESSIONS_INDEX.read_text())})
        # Index missing — build it on the fly from any existing files
        return jsonify({"index": _rebuild_index()})
    except Exception as e:
        print(f"❌ get_sessions_index error: {e}")
        return jsonify({"index": []})


@app.route("/api/sessions/<session_id>", methods=["GET"])
def get_session(session_id):
    """Return the full message history for a single session."""
    # Basic sanity-check: IDs are numeric timestamps or safe slugs
    if not re.match(r"^[\w\-]+$", session_id):
        return jsonify({"error": "Invalid session id"}), 400
    try:
        path = SESSIONS_DIR / f"{session_id}.json"
        if path.exists():
            return jsonify(json.loads(path.read_text()))
        return jsonify({"error": "Not found"}), 404
    except Exception as e:
        print(f"❌ get_session error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/sessions/<session_id>", methods=["POST"])
def save_session(session_id):
    """Create or update a single session file.
    Accepts the full session object {id, title, ts, messages}.
    Also rewrites the index entry for this session."""
    if not re.match(r"^[\w\-]+$", session_id):
        return jsonify({"error": "Invalid session id"}), 400
    try:
        data = request.get_json(silent=True) or {}
        # Enforce id consistency
        data["id"] = session_id
        # Cap messages to avoid huge files
        if isinstance(data.get("messages"), list):
            data["messages"] = data["messages"][-200:]

        path = SESSIONS_DIR / f"{session_id}.json"
        path.write_text(json.dumps(data, indent=2))

        # Update the index entry for this session without full rebuild
        index = []
        if SESSIONS_INDEX.exists():
            try:
                index = json.loads(SESSIONS_INDEX.read_text())
            except Exception:
                pass
        # Replace or insert this session's index entry
        entry = {
            "id": session_id,
            "title": data.get("title", "Untitled"),
            "ts": data.get("ts", session_id),
        }
        index = [x for x in index if x["id"] != session_id]
        index.insert(0, entry)
        # Re-sort newest-first
        index.sort(key=lambda x: str(x["ts"]), reverse=True)
        SESSIONS_INDEX.write_text(json.dumps(index, indent=2))

        return jsonify({"ok": True})
    except Exception as e:
        print(f"❌ save_session error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/sessions/<session_id>", methods=["DELETE"])
def delete_session(session_id):
    """Delete a session file and remove it from the index."""
    if not re.match(r"^[\w\-]+$", session_id):
        return jsonify({"error": "Invalid session id"}), 400
    try:
        path = SESSIONS_DIR / f"{session_id}.json"
        if path.exists():
            path.unlink()
        # Remove from index
        if SESSIONS_INDEX.exists():
            try:
                index = json.loads(SESSIONS_INDEX.read_text())
                index = [x for x in index if x["id"] != session_id]
                SESSIONS_INDEX.write_text(json.dumps(index, indent=2))
            except Exception:
                pass
        return jsonify({"ok": True})
    except Exception as e:
        print(f"❌ delete_session error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/sessions/migrate", methods=["POST"])
def migrate_sessions():
    """One-time migration: accepts the old flat sessions array and writes
    individual files. Safe to call multiple times (skips existing files)."""
    try:
        data = request.get_json(silent=True) or {}
        old_sessions = data.get("sessions", [])
        migrated = 0
        for s in old_sessions:
            sid = str(s.get("id", "")).strip()
            if not sid or not re.match(r"^[\w\-]+$", sid):
                continue
            path = SESSIONS_DIR / f"{sid}.json"
            if not path.exists():
                s["id"] = sid
                if isinstance(s.get("messages"), list):
                    s["messages"] = s["messages"][-200:]
                path.write_text(json.dumps(s, indent=2))
                migrated += 1
        _rebuild_index()
        return jsonify({"ok": True, "migrated": migrated})
    except Exception as e:
        print(f"❌ migrate_sessions error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/title", methods=["POST"])
def generate_title():
    """Generate a short AI title for a chat session using a fast, cheap LLM call."""
    try:
        data = request.get_json(silent=True) or {}
        task = str(data.get("task", "")).strip()[:500]
        answer = str(data.get("answer", "")).strip()[:400]

        if not task and not answer:
            return jsonify({"title": "New Task"})

        prompt = (
            f"User task: {task}\n\n"
            f"Answer preview: {answer}\n\n"
            "Write a short chat title (3-6 words max). "
            "Be specific and descriptive. No quotes, no punctuation at end. "
            "Just the title text, nothing else."
        )

        # Use a small/fast model — titles don't need heavy reasoning
        title_model = "llama-3.1-8b-instant"
        completion = client.chat.completions.create(
            model=title_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=20,
            temperature=0.3,
        )
        title = completion.choices[0].message.content.strip().strip("'\"").strip()
        # Safety: cap and strip any trailing punctuation like periods
        title = title[:60].rstrip(".!?…")
        print(f"📝 Generated title: '{title}'")
        return jsonify({"title": title})
    except Exception as e:
        print(f"❌ generate_title error: {e}")
        return jsonify({"title": ""})  # frontend falls back to its own title silently


@app.route("/agent-mode")
def agent_mode():
    return send_from_directory(".", "ava-agent.html")


# ── AGENT PIPELINE ────────────────────────────────────────────────────────────
# GPT OSS 120B with reasoning_effort="default" acts as the sub-agent.
# Ava (the director) is rendered in the frontend — the backend just runs the loop
# and streams structured SSE events the UI turns into step cards.
#
# Tool set: Tavily web search + file read/write (NO code execution).
# The agent runs a ReAct-style loop: think → pick tool → observe → repeat → answer.

AGENT_MODEL_ID = "openai/gpt-oss-120b"
AGENT_MAX_STEPS = 10
AGENT_FILE_SIZE_LIMIT = 10 * 1024 * 1024  # 10 MB raw file size limit
AGENT_TEXT_CHAR_LIMIT = (
    400_000  # ~400K chars of extracted text sent to LLM (~300 pages)
)


def extract_pdf_text(pdf_bytes: bytes) -> str:
    """Extract plain text from PDF bytes.
    Tries pdfminer.six first (most reliable), then pypdf, then pypdfium2.
    """
    errors = []

    # ── 1. pdfminer.six (best quality, handles complex layouts) ──
    try:
        from pdfminer.high_level import extract_text as pdfminer_extract

        text = pdfminer_extract(io.BytesIO(pdf_bytes))
        if text and text.strip():
            print(f"✅ pdfminer extracted {len(text):,} chars")
            return text
    except Exception as e:
        errors.append(f"pdfminer: {e}")

    # ── 2. pypdf ──
    try:
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(pdf_bytes))
        parts = [page.extract_text() or "" for page in reader.pages]
        text = "\n\n".join(p.strip() for p in parts if p.strip())
        if text:
            print(f"✅ pypdf extracted {len(text):,} chars")
            return text
    except Exception as e:
        errors.append(f"pypdf: {e}")

    # ── 3. pypdfium2 ──
    try:
        import pypdfium2 as pdfium

        pdf = pdfium.PdfDocument(pdf_bytes)
        parts = []
        for page in pdf:
            textpage = page.get_textpage()
            parts.append(textpage.get_text_range())
        text = "\n\n".join(p.strip() for p in parts if p.strip())
        if text:
            print(f"✅ pypdfium2 extracted {len(text):,} chars")
            return text
    except Exception as e:
        errors.append(f"pypdfium2: {e}")

    err_summary = " | ".join(errors)
    print(f"❌ All PDF extractors failed: {err_summary}")
    return f"[PDF extraction failed. Tried: {err_summary}. The PDF may be scanned/image-only.]"


def sse(event_type: str, **kwargs) -> str:
    """Format a server-sent event."""
    payload = {"type": event_type, **kwargs}
    return f"data: {json.dumps(payload)}\n\n"


def agent_run_web_search(query: str) -> str:
    if not tavily:
        return "[Web search unavailable — TAVILY_API_KEY not set]"
    try:
        result = tavily.search(
            query=query, search_depth="advanced", max_results=4, include_answer=True
        )
        if result.get("answer"):
            return result["answer"]
        snippets = [r.get("content", "") for r in result.get("results", [])[:4]]
        return " | ".join(s[:400] for s in snippets if s) or "[No results found]"
    except Exception as e:
        return f"[Search failed: {e}]"


def agent_read_file(path: str) -> str:
    try:
        p = Path(path)
        if not p.exists():
            return f"[Error: File not found: {path}]"
        size = p.stat().st_size
        if size > AGENT_FILE_SIZE_LIMIT:
            return f"[Error: File too large ({size // (1024 * 1024):.1f} MB). Max is {AGENT_FILE_SIZE_LIMIT // (1024 * 1024)} MB.]"
        # PDF: extract text
        if p.suffix.lower() == ".pdf":
            text = extract_pdf_text(p.read_bytes())
        else:
            text = p.read_text(encoding="utf-8", errors="replace")
        # Truncate extracted text if it exceeds LLM context limit
        if len(text) > AGENT_TEXT_CHAR_LIMIT:
            text = (
                text[:AGENT_TEXT_CHAR_LIMIT]
                + f"\n\n[...truncated — file has more content beyond this point ({len(text):,} total chars extracted)]"
            )
        return text
    except Exception as e:
        return f"[Error reading file: {e}]"


def agent_write_file(path: str, content: str) -> str:
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"[Written {len(content)} chars to {path}]"
    except Exception as e:
        return f"[Error writing file: {e}]"


AGENT_SYSTEM_PROMPT = """You are Ava, an exceptionally intelligent and thorough AI analyst. You think deeply, write extensively, and never truncate your analysis prematurely.

You operate in a structured ReAct loop, executing ONE action at a time.

AVAILABLE TOOLS:
- web_search(query): Search the web for live/current data you genuinely cannot know (breaking news, live prices, very recent events). Do NOT search things you already know.
- file_read(path): Read any file from the server — including PDFs (text extracted automatically), text, markdown. Up to 10 MB.
- file_write(path, content): Write or overwrite a file on the server filesystem.

OUTPUT FORMAT — respond with EXACTLY ONE JSON object per turn. NEVER output a JSON array:

{"action": "web_search", "query": "...", "thought": "..."}
{"action": "file_read", "path": "/absolute/path", "thought": "..."}
{"action": "file_write", "path": "/absolute/path", "content": "...", "thought": "..."}
{"action": "answer", "content": "your full, complete, deeply detailed answer"}

═══════════════════════════════════════════════
RESPONSE QUALITY STANDARDS — READ CAREFULLY
═══════════════════════════════════════════════

When analyzing creative works (books, stories, scripts, worldbuilding):
- Write a FULL literary analysis — themes, character arcs, narrative structure, prose style, pacing, subtext
- For each major theme or element, open it up: what works, what could be stronger, specific examples with quotes or scene references
- Identify branching opportunities: unresolved threads, implied backstories, possible sequels or prequels
- Compare to relevant works in the genre where useful
- Give a genuine, opinionated critical voice — not a bullet-point summary
- Minimum depth: if you were a professional book editor or literary critic, what would your full notes look like?

When answering factual or research questions:
- Go beyond surface answers. Explain the WHY, the HOW, the historical context, the implications
- Include multiple perspectives or schools of thought where they exist
- Surface non-obvious connections and insights the user might not have considered
- If the topic branches into subtopics, address each one with appropriate depth

When asked for opinions or recommendations:
- Give a real, confident opinion with reasoning — not "it depends" hedging
- Defend your position while acknowledging counterarguments
- Be specific: vague praise or vague criticism is useless

GENERAL RULES:
- LONG IS CORRECT. A short answer to a complex question is a failure. Fill the space the question deserves.
- Never end with "let me know if you want more" — just give more upfront.
- Never use filler phrases like "Great question!", "Certainly!", "I'd be happy to".
- ONE JSON object per response. Never an array.
- Be confident. Have opinions. Push back if something doesn't work.
- For population estimates, science, history — answer from knowledge, do not search.
- Only search for genuinely live data you cannot know without the internet.
- When a file is already in context, work from it directly — do NOT file_read it again.
- Never write or execute code.
- Always end with {"action": "answer", ...} when done.
"""


@app.route("/agent", methods=["POST"])
def agent():
    data = request.get_json(silent=True) or {}
    if not data:
        print("❌ Agent: failed to parse JSON body — possibly too large or malformed")
        return jsonify(
            {"error": "Could not parse request body. File may be too large."}
        ), 400

    task = str(data.get("task", "")).strip()[:2000]
    attached_file = str(data.get("file_path", "")).strip()
    images = data.get("images", [])  # list of {b64, mime, name}
    doc = data.get("doc", None)  # {b64, mime, name} — uploaded text doc
    web_search_enabled = data.get("web_search", True)  # frontend toggle

    # Debug logging
    if doc:
        b64_len = len(doc.get("b64", ""))
        approx_mb = b64_len * 3 / 4 / (1024 * 1024)
        print(
            f"📎 Agent doc upload: {doc.get('name')} | mime={doc.get('mime')} | ~{approx_mb:.1f} MB decoded"
        )
    else:
        print(f"📋 Agent task (no doc): {task[:80]}")

    if not task and not images and not doc:
        return jsonify({"error": "No task provided"}), 400

    def generate():
        # Inject current datetime — Ava always knows what time it is
        now = datetime.now()
        datetime_note = (
            f"\n\nCURRENT DATE & TIME: {now.strftime('%A, %B %d, %Y at %H:%M')} (server local time). "
            f"Use this when answering questions about today's date, current year, or time-sensitive topics. "
            f"Do NOT web_search just to find the current date — you already know it."
        )

        # Build system prompt with web search availability note
        system_prompt = AGENT_SYSTEM_PROMPT + datetime_note
        if not web_search_enabled:
            system_prompt += "\n\nNOTE: Web search is currently DISABLED by the user. Do NOT use web_search. Answer from your own knowledge only and be explicit about it."
        messages = [{"role": "system", "content": system_prompt}]

        user_content_parts = []  # will build a multipart user message if images present

        # ── Handle uploaded images via vision model first ──────────────
        if images:
            yield sse(
                "file_read",
                path=f"{len(images)} image(s) attached — running vision analysis",
            )
            vision_descriptions = []
            for img in images[:4]:  # cap at 4 images
                try:
                    v_res = client.chat.completions.create(
                        model=VISION_MODEL_ID,
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": "Describe this image in detail so an AI agent can understand and work with its contents.",
                                    },
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:{img['mime']};base64,{img['b64']}"
                                        },
                                    },
                                ],
                            }
                        ],
                    )
                    desc = v_res.choices[0].message.content
                    vision_descriptions.append(
                        f"[Image: {img.get('name', 'untitled')}]\n{desc}"
                    )
                except Exception as e:
                    vision_descriptions.append(
                        f"[Image: {img.get('name', 'untitled')} — vision failed: {e}]"
                    )
            user_content_parts.append(
                "[Attached images analyzed by vision model]:\n"
                + "\n\n".join(vision_descriptions)
            )

        # ── Handle uploaded text/doc file ──────────────────────────────
        if doc and doc.get("b64"):
            try:
                raw_bytes = base64.b64decode(doc["b64"])
                fname = doc.get("name", "uploaded file")
                mime = doc.get("mime", "")
                is_pdf = fname.lower().endswith(".pdf") or "pdf" in mime.lower()
                print(
                    f"📄 Processing uploaded file: {fname} | is_pdf={is_pdf} | raw_bytes={len(raw_bytes):,}"
                )

                if is_pdf:
                    yield sse("file_read", path=f"{fname} (extracting PDF text…)")
                    file_text = extract_pdf_text(raw_bytes)
                    print(f"📄 PDF extracted: {len(file_text):,} chars")
                else:
                    file_text = raw_bytes.decode("utf-8", errors="replace")
                    print(f"📄 Text file decoded: {len(file_text):,} chars")

                # Truncate if it exceeds LLM context limit
                if len(file_text) > AGENT_TEXT_CHAR_LIMIT:
                    file_text = (
                        file_text[:AGENT_TEXT_CHAR_LIMIT]
                        + f"\n\n[...truncated — showing first {AGENT_TEXT_CHAR_LIMIT:,} chars of {len(file_text):,} total extracted]"
                    )

                yield sse("file_read", path=fname)
                user_content_parts.append(
                    f"[Uploaded file: {fname} — full text extracted and provided below. Work directly from this content. Do NOT attempt to file_read it by path.]\n\n{file_text}"
                )
            except Exception as e:
                print(f"❌ Doc processing error: {e}")
                yield sse(
                    "file_read", path=f"Error reading {doc.get('name', 'file')}: {e}"
                )
                user_content_parts.append(
                    f"[Could not read uploaded file '{doc.get('name', '')}': {e}]"
                )

        # ── Handle server-side file path ───────────────────────────────
        if attached_file:
            file_content = agent_read_file(attached_file)
            yield sse("file_read", path=attached_file)
            user_content_parts.append(
                f"[Attached file: {attached_file}]\n\nFile contents:\n{file_content}"
            )

        # ── LARGE DOCUMENT: MAP-REDUCE ANALYSIS ───────────────────────
        # If a large document is attached, pre-process it in focused chunks
        # before the main agent loop. This prevents the model from getting
        # overwhelmed by 50K+ tokens of raw text and hallucinating.
        CHUNK_CHAR_SIZE = 40_000  # ~10K tokens per chunk, comfortable window
        CHUNK_OVERLAP = 2_000  # overlap to preserve continuity

        large_doc_summary = None
        large_doc_name = None

        def chunk_text(text, size, overlap):
            chunks = []
            start = 0
            while start < len(text):
                end = min(start + size, len(text))
                chunks.append(text[start:end])
                if end == len(text):
                    break
                start += size - overlap
            return chunks

        def summarize_chunk(
            chunk_text_str, chunk_idx, total_chunks, doc_name, user_task
        ):
            """Summarize a single chunk of a large document."""
            prompt = (
                f"You are analyzing part {chunk_idx + 1} of {total_chunks} of '{doc_name}'.\n"
                f"The user wants to: {user_task}\n\n"
                f"Extract from this section:\n"
                f"- Key events, characters, themes, ideas\n"
                f"- Notable quotes or passages (verbatim, short)\n"
                f"- Anything that would be important for a full analysis\n\n"
                f"Be thorough and specific. Do not be generic.\n\n"
                f"--- SECTION TEXT ---\n{chunk_text_str}"
            )
            try:
                r = client.chat.completions.create(
                    model=AGENT_MODEL_ID,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    reasoning_effort="high",
                    max_tokens=4096,
                )
                return r.choices[0].message.content or ""
            except Exception as e:
                return f"[Chunk {chunk_idx + 1} summarization failed: {e}]"

        # Check if any large doc was loaded
        for part in user_content_parts:
            if "[Uploaded file:" in part or "[Attached file:" in part:
                # Extract the raw text from this part
                lines = part.split("\n", 2)
                header = lines[0]
                doc_text = lines[-1] if len(lines) > 1 else ""
                name_match = re.search(
                    r"\[(?:Uploaded|Attached) file: ([^\]]+?)(?: —|])", header
                )
                doc_name_extracted = name_match.group(1) if name_match else "document"

                if len(doc_text) > CHUNK_CHAR_SIZE:
                    chunks = chunk_text(doc_text, CHUNK_CHAR_SIZE, CHUNK_OVERLAP)
                    large_doc_name = doc_name_extracted
                    yield sse(
                        "think",
                        content=f"Document is large ({len(doc_text):,} chars, {len(chunks)} chunks). Running map-reduce analysis pass…",
                    )

                    chunk_summaries = []
                    for i, chunk in enumerate(chunks):
                        yield sse(
                            "file_read",
                            path=f"Analyzing section {i + 1}/{len(chunks)} of {large_doc_name}…",
                        )
                        summary = summarize_chunk(
                            chunk, i, len(chunks), large_doc_name, task
                        )
                        chunk_summaries.append(
                            f"[Section {i + 1}/{len(chunks)}]\n{summary}"
                        )

                    large_doc_summary = "\n\n".join(chunk_summaries)
                    yield sse(
                        "file_read",
                        path=f"All {len(chunks)} sections analyzed. Synthesizing…",
                    )
                    break

        # If we built a map-reduce summary, replace the raw doc in the message
        if large_doc_summary and large_doc_name:
            user_content = (
                f"[Document: '{large_doc_name}' — pre-analyzed in {len(chunk_summaries)} sections. "
                f"Full structured notes below. Use these to write your answer — the entire book has been read.]\n\n"
                f"{large_doc_summary}\n\n"
                f"---\nUser task: {task}"
            )
            messages.append({"role": "user", "content": user_content})
        else:
            # Normal path — compose final user message as before
            if user_content_parts:
                user_content_parts.append(f"---\nTask: {task}" if task else "")
                user_content = "\n\n".join(p for p in user_content_parts if p)
            else:
                user_content = task
            messages.append({"role": "user", "content": user_content})

        for step in range(AGENT_MAX_STEPS):
            t0 = time.time()

            # Call the reasoning model
            # reasoning_effort="none" saves the token budget for actual output —
            # Model's thinking tokens otherwise eat into max_tokens before the answer is written
            try:
                completion = client.chat.completions.create(
                    model=AGENT_MODEL_ID,
                    messages=messages,
                    temperature=0.6,
                    reasoning_effort="high",
                    response_format={"type": "json_object"},
                    max_tokens=32768,
                )
            except Exception as e:
                # Retry without JSON mode if rejected
                try:
                    completion = client.chat.completions.create(
                        model=AGENT_MODEL_ID,
                        messages=messages,
                        temperature=0.6,
                        reasoning_effort="high",
                        max_tokens=32768,
                    )
                except Exception as e2:
                    yield sse("error", content=f"LLM call failed: {e2}")
                    return

            elapsed = round(time.time() - t0, 1)
            raw = completion.choices[0].message.content or ""

            # ── Strip <think>....</think> blocks emitted inline ──
            # Model sometimes puts reasoning in the content as <think>...</think>
            # instead of the reasoning_content field. Extract and emit separately.
            think_match = re.search(r"<think>(.*?)</think>", raw, re.DOTALL)
            if think_match:
                inline_thinking = think_match.group(1).strip()
                if inline_thinking:
                    yield sse("think", content=inline_thinking, elapsed=elapsed)
                # Remove the think block from raw before parsing JSON
                raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

            # Also emit reasoning_content if the API provides it separately
            reasoning = getattr(
                completion.choices[0].message, "reasoning_content", None
            )
            if reasoning and not think_match:
                yield sse("think", content=reasoning.strip(), elapsed=elapsed)

            # ── Parse action — handle <think>-stripped raw, arrays, single objects ──
            try:
                arr_match = re.search(r"\[.*\]", raw, re.DOTALL)
                obj_match = re.search(r"\{.*\}", raw, re.DOTALL)
                parsed = None
                if arr_match:
                    try:
                        arr = json.loads(arr_match.group())
                        if isinstance(arr, list) and arr:
                            parsed = arr[0]
                    except Exception:
                        pass
                if parsed is None:
                    parsed = (
                        json.loads(obj_match.group()) if obj_match else json.loads(raw)
                    )
                action_data = parsed
            except Exception:
                yield sse(
                    "error", content=f"Could not parse agent response: {raw[:300]}"
                )
                return

            action = action_data.get("action", "")

            # Store only the clean JSON (no think blocks) in message history
            messages.append({"role": "assistant", "content": raw})

            # ── TOOL DISPATCH ──────────────────────────────────────────
            if action == "web_search":
                if not web_search_enabled:
                    # Block the search, tell model to answer from knowledge
                    messages.append(
                        {
                            "role": "user",
                            "content": '[Web search is disabled. Please answer from your own knowledge instead. Output your answer as {"action": "answer", "content": "..."}]',
                        }
                    )
                else:
                    query = str(action_data.get("query", "")).strip()
                    yield sse("search_start", query=query)
                    result = agent_run_web_search(query)
                    yield sse("search_result", content=result)
                    messages.append(
                        {
                            "role": "user",
                            "content": f"[Search result for '{query}']:\n{result}",
                        }
                    )

            elif action == "file_read":
                path = str(action_data.get("path", "")).strip()
                yield sse("file_read", path=path)
                content = agent_read_file(path)
                messages.append(
                    {
                        "role": "user",
                        "content": f"[File contents of {path}]:\n{content}",
                    }
                )

            elif action == "file_write":
                path = str(action_data.get("path", "")).strip()
                content = str(action_data.get("content", ""))
                result = agent_write_file(path, content)
                yield sse("file_write", path=path)
                messages.append({"role": "user", "content": result})

            elif action == "answer":
                answer_text = str(action_data.get("content", "")).strip()
                yield sse("answer", content=answer_text)
                yield "data: [DONE]\n\n"
                return

            else:
                yield sse("error", content=f"Unknown action '{action}'. Aborting.")
                return

        # If we exhaust steps without an answer
        yield sse(
            "error",
            content=f"Agent reached the maximum step limit ({AGENT_MAX_STEPS}) without finishing. Try breaking the task into smaller parts.",
        )
        yield "data: [DONE]\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    import ssl

    ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ssl_ctx.load_cert_chain(str(LOCAL_CERT), str(LOCAL_KEY))
    print(f"🔒 Starting HTTPS server with certificates from {CERT_DIR}")
    app.run(host="0.0.0.0", port=5000, ssl_context=ssl_ctx)
