# Beginner Guide — `app.py`

This is a step-by-step walkthrough of `app.py`, the Streamlit frontend for **Memorial AI** — a chatbot that talks like a deceased grandfather using stored memories.

---

## Step 1 — Imports & setup (lines 1–18)

```python
import streamlit as st
from dotenv import load_dotenv
from graph import grandpa_graph
from memories import add_memory

load_dotenv()
```

- **`streamlit as st`** — the web UI library. Every `st.something()` call renders a UI element.
- **`load_dotenv()`** — reads your `.env` file and puts the keys (OpenAI, Pinecone, etc.) into environment variables so the rest of the code can find them.
- **`grandpa_graph`** — a LangGraph workflow imported from `graph.py`. This is the "brain": when you send a message, it routes through nodes that retrieve memories and generate a reply.
- **`add_memory`** — a function from `memories.py` that stores a new memory (embeds it and writes to Pinecone).

```python
st.set_page_config(page_title="Memorial AI", page_icon="🕯️", layout="centered")
```
Sets the browser tab title, favicon, and centers the content. Must be the **first** Streamlit call.

---

## Step 2 — CSS styling (lines 21–69)

`st.markdown(..., unsafe_allow_html=True)` injects raw HTML/CSS into the page. `unsafe_allow_html=True` is required because Streamlit escapes HTML by default.

The CSS defines five visual elements:
- `.stApp` — dark navy background (`#0a0f1a`)
- `.user-bubble` — orange chat bubbles, aligned right (your messages)
- `.grandpa-bubble` — dark gray bubbles, aligned left (Grandpa's replies)
- `.avatar-box` — the rounded card at the top showing the avatar
- `.memory-tag` — the small example-memory chips in tab 2

Nothing dynamic here — it's just styling.

---

## Step 3 — Session state (lines 73–77)

```python
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "grandpa", "content": "My dear one… 🙏"}
    ]
```

**Key concept:** Streamlit re-runs the entire script top-to-bottom every time the user interacts. So normal Python variables get reset. `st.session_state` is a dict-like object that **survives reruns** for the lifetime of the browser tab.

This block says: "If this is the very first run, seed the chat with Grandpa's greeting." On every later rerun, the `if` is false and the existing history is preserved.

---

## Step 4 — Avatar header (lines 80–87)

Pure HTML rendered through `st.markdown`. Shows the 👴 emoji, name, dates, and an orange "● Online" indicator. Static content.

---

## Step 5 — Tabs (line 90)

```python
tab_chat, tab_memory = st.tabs(["💬 Chat", "📚 Add Memory"])
```

Creates two tabs. The `with tab_chat:` and `with tab_memory:` blocks below scope everything inside to the respective tab.

---

## Step 6 — Chat tab: render history (lines 95–106)

```python
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="user-bubble">{msg["content"]}</div>', ...)
    else:
        st.markdown(f'<div class="grandpa-bubble">{msg["content"]}</div>', ...)
```

Loops through the stored history and renders each message with the matching CSS class. Right-aligned orange for you, left-aligned dark for Grandpa.

---

## Step 7 — Input form (lines 110–120)

```python
with st.form(key="chat_form", clear_on_submit=True):
    col1, col2 = st.columns([5, 1])
    with col1:
        user_input = st.text_input(...)
    with col2:
        send = st.form_submit_button("Send ➤")
```

**Why a form?** Without it, Streamlit reruns on *every keystroke*. A form batches input — nothing happens until `send` is clicked (or Enter is pressed). `clear_on_submit=True` empties the textbox after sending. `st.columns([5, 1])` gives a 5:1 width split (wide input + small button).

---

## Step 8 — Handle send: the core logic (lines 123–146)

```python
if send and user_input.strip():
    # 1. Append user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    # 2. Invoke the LangGraph workflow
    with st.spinner("Grandpa is thinking…"):
        result = grandpa_graph.invoke({
            "user_message": user_input,
            "relevant_memories": "",
            "reply": ""
        })

    # 3. Append Grandpa's reply
    st.session_state.messages.append({"role": "grandpa", "content": result["reply"]})

    # 4. Force a rerun so new bubbles render
    st.rerun()
```

This is the heart of the app:
1. **Save your message** to session state.
2. **Call `grandpa_graph.invoke(...)`** with an initial state dict. The graph fills `relevant_memories` (via a RAG node that searches Pinecone) and `reply` (the LLM node). The empty strings are placeholders — the graph mutates them.
3. **Save the reply.**
4. **`st.rerun()`** — restarts the script from the top. Since `st.session_state.messages` now has two new entries, the loop in Step 6 renders them.

The `st.spinner` shows a loading indicator only during step 2.

---

## Step 9 — Add Memory tab (lines 151–184)

```python
with st.form(key="memory_form", clear_on_submit=True):
    new_memory = st.text_area("Write a memory", height=100)
    save = st.form_submit_button("💾 Save Memory")

if save and new_memory.strip():
    with st.spinner("Saving memory…"):
        add_memory(new_memory.strip())
    st.success("✅ Memory saved!")
```

Mirrors the chat tab's pattern:
- Show some hardcoded example memory chips for inspiration.
- A form with a textarea + save button.
- On submit, call `add_memory(...)` — which internally embeds the text and upserts it into Pinecone.
- `st.success(...)` shows a green confirmation banner.

No `st.rerun()` here because there's no list to re-render — the success message is enough.

---

## Mental model — the whole flow

```
User types → form submit → append to messages
                ↓
        grandpa_graph.invoke({user_message, ...})
                ↓
        [graph internals: retrieve memories from Pinecone → LLM with persona prompt]
                ↓
        result["reply"] → append to messages → st.rerun() → bubbles re-render
```

The Add Memory tab is a separate write path that feeds the same Pinecone index the chat path reads from.
