# 🌸 Open VRM Companion

An expressive AI companion with a 3D VRM character, voice, emotions, animations, and an agentic research mode. Built on Python Flask + Three.js, powered by the Groq API (or swap in any OpenAI-compatible backend).

> Inspired by [ChatVRM](https://github.com/pixiv/ChatVRM), [local-chat-vrm](https://github.com/pixiv/local-chat-vrm), [Open-LLM-VTuber](https://github.com/Open-LLM-VTuber/Open-LLM-VTuber), [ChatVRM-LocalAI](https://github.com/camenduru/ChatVRM-LocalAI), and [Realtime Avatar AI Companion](https://github.com/igna-s/Realtime_Avatar_AI_Companion).

![demo](assets/dance.gif)

---

## ✨ Features

- **3D VRM character** rendered with Three.js + `@pixiv/three-vrm` v3 (VRM 1.0 compatible)
- **Emotion expression system** — happy, sad, angry, surprised, blushing, pouting, smug, flustered, sleepy, and more, with per-response fading
- **BVH animations** — idles, dances, reactions, actions (Mixamo → XR Animator pipeline)
- **Voice chat** — Whisper Large v3 STT + Orpheus TTS via Groq API, with hallucination filtering
- **Vision** — show the companion an image and she describes what she sees (Llama 4 Scout)
- **Persistent memory** — short-term conversation history + long-term fact extraction across sessions
- **Autonomous idle callouts** — the companion speaks up on her own when you've been away
- **Web search** — live Tavily search injection into the LLM context
- **Agent mode** (`agent.html`) — multi-step reasoning agent with web search, file read/write, PDF analysis, markdown rendering, and per-session history
- **Mobile compatible** — HTTPS + Tailscale, deferred AudioContext for Android/iOS
- **Swappable backends** — any OpenAI-compatible LLM endpoint works in place of Groq

---

## 🗂️ Project Structure

```
open-vrm-companion/
├── groq_bridge.py          # Flask backend — LLM, TTS, STT, Vision, Agent
├── webui.html          # Main companion chat UI with 3D VRM
├── agent.html          # Agent mode — research, file tasks, markdown output
├── start.sh            # Launch script
├── .env                    # Your API keys (not committed)
├── .env.example            # Template — copy this to .env
├── memory.json             # Short-term conversation history (auto-generated)
├── persistent_memory.txt   # Long-term extracted facts (auto-generated)
├── message_counter.json    # Message count for memory update intervals
├── agent_sessions/         # Per-session agent chat history (auto-generated)
└── assets/
    ├── animations/         # BVH animation files for demo
    ├── Avatar_Sample.vrm   # Sample VRM model (bring your own — see below)
```

---

## 🚀 Quick Start
 
### 1. Clone the repo
 
```bash
git clone https://github.com/trisanap/open-vrm-companion.git
cd open-vrm-companion
```
 
### 2. Create a virtual environment and install dependencies
 
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install flask flask-cors groq python-dotenv tavily-python
```
 
### 3. Set up your environment variables
 
Edit `.env` and fill in your keys:
 
```
GROQ_API_KEY=your_groq_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here      # optional — enables web search
TAILSCALE_HOSTNAME=your-machine.ts.net       # optional — enables HTTPS
```
 
- Get a Groq API key at [console.groq.com](https://console.groq.com)
- Get a Tavily API key at [app.tavily.com](https://app.tavily.com) (free tier available)
### 4. Add your VRM model
 
Place your `.vrm` file in the `assets/` folder and update the model path in `webui.html`:
 
```js
const VRM_DEFAULT = assetPathPrefix + "YourCharacter.vrm";
```
 
> **Don't have a VRM?** Create one for free at [VRoid Studio](https://vroid.com/en/studio) or on [VRoid Studio Steam page](https://store.steampowered.com/app/1486350/VRoid_Studio/). Export as VRM 1.0.
 
### 5. Run the server
 
```bash
# HTTP (local only)
python groq_bridge.py
 
# Or use the launch script
bash start.sh
```
 
Open `http://localhost:5000` in your browser.
 
---
 
## 🔒 HTTPS / Remote Access (Optional)
 
For mobile access or cross-device use, the server supports HTTPS via [Tailscale](https://tailscale.com/):
 
1. Install Tailscale and enable HTTPS certificates for your machine
2. Set `TAILSCALE_HOSTNAME=your-machine.your-tailnet.ts.net` in your `.env`
3. Run the server — it will auto-copy certs to `~/.ava_certs/` and serve over HTTPS
You can then access the companion from any device on your Tailscale network.
 
---
 
## 🎭 Animations
 
Animations are BVH files sourced from [Mixamo](https://www.mixamo.com/) and converted using [XR Animator](https://xr-animator.web.app/). Place BVH files in `assets/animations/`.
 
To add a new animation, update **three places** in `groq_bridge.py` and `webui.html`:
 
1. `VALID_ANIMATIONS` list in `groq_bridge.py`
2. `PANEL_ANIMATIONS` array in `webui.html`
3. `curatedAnimations` array in `webui.html`
---
 
## 🧠 Models Used (Groq API defaults)
 
| Role | Model |
|---|---|
| Chat / Reasoning | `openai/gpt-oss-120b` |
| Vision | `meta-llama/llama-4-scout-17b-16e-instruct` |
| TTS | `canopylabs/orpheus-v1-english` |
| STT | `whisper-large-v3` |
| Title generation | `llama-3.1-8b-instant` |
 
All models are swappable — edit the model ID constants at the top of `groq_bridge.py`.
 
---
 
## 🔧 Customization
 
**Change the companion's persona** — edit the `system_prompt` in the `/chat` route in `groq_bridge.py`.
 
**Change the voice** — update the `voice` parameter in `generate_speech()`. Orpheus voices: `tara`, `leah`, `jess`, `leo`, `dan`, `mia`, `zac`, `zoe`, `autumn`.
 
**Add new emotions** — add blend shape targets to the VRM expression map in `webui.html`.
 
**Swap the LLM backend** — the Groq client is OpenAI-compatible. Point it at Ollama, LM Studio, or any local inference server by changing the `base_url`.
 
---
 
## 📋 Requirements
 
- Python 3.10+
- A modern browser (Chrome recommended for best WebGL/AudioContext support)
- Groq API key (free tier works for development)
- A VRM 1.0 character file
---
 
## 🙏 Credits & Inspirations
 
| Project | What it contributed |
|---|---|
| [ChatVRM / pixiv](https://github.com/pixiv/ChatVRM) | Original VRM + LLM chat concept |
| [local-chat-vrm / pixiv](https://github.com/pixiv/local-chat-vrm) | Local-first VRM architecture |
| [Open-LLM-VTuber](https://github.com/Open-LLM-VTuber/Open-LLM-VTuber) | Open-source VTuber pipeline |
| [ChatVRM-LocalAI](https://github.com/camenduru/ChatVRM-LocalAI) | Local AI backend integration |
| [Realtime Avatar AI Companion](https://github.com/igna-s/Realtime_Avatar_AI_Companion) | Real-time companion interaction model |
| [@pixiv/three-vrm](https://github.com/pixiv/three-vrm) | VRM rendering in Three.js |
| [Mixamo](https://www.mixamo.com/) | Animation source |
| [XR Animator](https://xr-animator.web.app/) | BVH conversion tool |

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

The VRM model file in this repository is just a sample avatar provided by VRoid Studio. Please bring your own, respecting the license of your chosen model.

Animation BVH files converted from Mixamo are subject to [Adobe's Mixamo license](https://www.adobe.com/legal/terms/mixamo-terms-of-service.html) — personal and commercial use permitted, redistribution restrictions apply.
