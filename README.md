# 🤖 Day 25 — Multi-Agent System with CrewAI

Part of my **100-Day AI Engineering Bootcamp** (Day 25) — building one AI project a day to become job-ready.

## 🤔 The Problem I Set Out to Solve

Days 22-24 built one agent juggling multiple tools. Today's shift is architectural: instead of one agent doing everything, I wanted a *team* — agents with defined roles, where each one specializes, hands off its output to the next, and the final result is better than any single agent could produce alone. The test: give it the topic "AI Engineering jobs in India 2026" and see if three agents working sequentially could produce a structured, credible research report.

They did. One real bug stood between the initial code and that result.

## 🪲 The Bug — A Known CrewAI Issue With Groq

The crew started correctly (Task Started, Senior Research Analyst spun up), then immediately failed with:

```
litellm.BadRequestError: GroqException -
'messages.0' : property 'cache_breakpoint' is unsupported
```

After investigating, this turned out to be a documented bug in CrewAI (Issue #5886): `mark_cache_breakpoint()` is called on **all** messages in CrewAI's agent executor, regardless of which LLM provider you're using. The Anthropic adapter correctly strips it. Every other provider — including Groq — receives `cache_breakpoint: true` in the raw message payload, and Groq's API rejects it with a 400 error.

**The fix:** a two-line monkey-patch at the top of `crew.py`:

```python
import crewai.llms.cache as _crewai_cache
_crewai_cache.mark_cache_breakpoint = lambda msg: msg
```

`lambda msg: msg` takes a message and returns it completely unchanged — disabling the cache_breakpoint injection before it ever reaches Groq.

## ⚙️ What It Does

- 🔎 **Researcher** searches the web for verified facts and sources on any topic
- 🧠 **Analyst** reads the research and extracts 3-5 key insights with implications
- ✍️ **Writer** turns the analysis into a structured report (exec summary, findings, implications, conclusion)
- 🔗 Output of each agent flows automatically to the next — no manual wiring
- 📄 Final report rendered in Streamlit

## 🛠️ Tech Stack

- **Python**
- **CrewAI 1.15.1** — multi-agent orchestration framework
- **LiteLLM** — routes CrewAI's LLM calls to Groq
- **Groq API (LLaMA 3.3 70B)** — LLM powering all three agents
- **SerperDev API** — web search for the Researcher (2,500 free searches/month at serper.dev)
- **Streamlit** — UI

## 🚀 How to Run

```bash
git clone https://github.com/PrashikSawant/Multi-Agent-System-with-CrewAI
cd Day25-CrewAI
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
# Add GROQ_API_KEY and SERPER_API_KEY to .env
streamlit run app.py
```

**Heads up:** each run makes 3 sequential LLM calls with web search — expect 1-3 minutes per topic.

## 💡 What I Learned

- Multi-agent is an architectural choice, not just "more tools" — each agent specializes, and role/goal/backstory quality directly affects output quality
- CrewAI's sequential process handles task handoffs automatically, without writing any loop or message-passing code manually
- Framework abstractions save time but also hide bugs — the cache_breakpoint issue would never have appeared in my raw Day 22-24 code
- The monkey-patch is a real, legitimate tool for working around third-party bugs while waiting for an upstream fix

## 🔮 What's Next

**Day 26 — Voice Chatbot**, adding Whisper speech-to-text so you can talk to the AI instead of typing.

## 👤 About Me

I'm Prashik, building one AI project a day for 100 days to become a job-ready AI Engineer. Follow my journey on [LinkedIn](https://www.linkedin.com/in/prashik-sawant-ds).
