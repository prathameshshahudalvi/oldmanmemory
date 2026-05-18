# app.py
# Streamlit frontend for Memorial AI
# Run with: streamlit run app.py

import streamlit as st
from dotenv import load_dotenv
from graph import grandpa_graph
from memories import add_memory

# ── Load API keys ──────────────────────────────────────────────────────────────
load_dotenv()

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Memorial AI",
    page_icon="🕯️",
    layout="centered"
)

# ── Custom CSS — makes it look like the orange device ─────────────────────────
st.markdown("""
<style>
    /* Dark background */
    .stApp { background-color: #0a0f1a; color: #f9fafb; }

    /* Hide default streamlit header */
    #MainMenu, header, footer { visibility: hidden; }

    /* Chat message bubbles */
    .user-bubble {
        background: #f97316;
        color: white;
        padding: 10px 16px;
        border-radius: 18px 18px 4px 18px;
        margin: 6px 0;
        max-width: 75%;
        margin-left: auto;
        text-align: right;
        font-size: 15px;
    }
    .grandpa-bubble {
        background: #1f2937;
        color: #e5e7eb;
        padding: 10px 16px;
        border-radius: 18px 18px 18px 4px;
        margin: 6px 0;
        max-width: 75%;
        font-size: 15px;
        border-left: 3px solid #f97316;
    }
    .avatar-box {
        text-align: center;
        padding: 20px;
        background: #111827;
        border-radius: 20px;
        margin-bottom: 20px;
        border: 1px solid #1f2937;
    }
    .memory-tag {
        background: #1f2937;
        border: 1px solid #374151;
        border-radius: 8px;
        padding: 6px 10px;
        font-size: 13px;
        color: #9ca3af;
        margin: 4px 0;
    }
</style>
""", unsafe_allow_html=True)

# ── Session state (remembers chat history during the session) ──────────────────
# st.session_state works like a dictionary that persists across reruns
if "messages" not in st.session_state:
    st.session_state.messages = [
        # First message from Grandpa when you open the app
        {"role": "grandpa", "content": "My dear one… it's so good to hear from you. I'm right here. What's on your mind? 🙏"}
    ]

# ── Avatar + header ────────────────────────────────────────────────────────────
st.markdown("""
<div class="avatar-box">
    <div style="font-size: 64px;">👴</div>
    <div style="font-size: 20px; font-weight: 700; color: #f9fafb; margin-top: 8px;">Grandpa Wei</div>
    <div style="font-size: 13px; color: #6b7280;">1938 – 2023 · Memorial AI</div>
    <div style="margin-top: 10px; font-size: 12px; color: #f97316;">● Online</div>
</div>
""", unsafe_allow_html=True)

# ── Tabs: Chat | Add Memory ────────────────────────────────────────────────────
tab_chat, tab_memory = st.tabs(["💬 Chat", "📚 Add Memory"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: CHAT
# ══════════════════════════════════════════════════════════════════════════════
with tab_chat:

    # Show all messages in the chat history
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            # User bubble (right side, orange)
            st.markdown(f'<div class="user-bubble">{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            # Grandpa bubble (left side, dark)
            st.markdown(f'<div class="grandpa-bubble">{msg["content"]}</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Input box at the bottom ────────────────────────────────────────────────
    # Use a form so pressing Enter sends the message
    with st.form(key="chat_form", clear_on_submit=True):
        col1, col2 = st.columns([5, 1])

        with col1:
            user_input = st.text_input(
                label="message",
                placeholder="Say something to Grandpa…",
                label_visibility="collapsed"  # hides the label text
            )
        with col2:
            send = st.form_submit_button("Send ➤")

    # ── Handle send ───────────────────────────────────────────────────────────
    if send and user_input.strip():

        # 1. Add user message to chat history
        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })

        # 2. Show a spinner while Grandpa is "thinking"
        with st.spinner("Grandpa is thinking…"):
            result = grandpa_graph.invoke({
                "user_message": user_input,
                "relevant_memories": "",  # filled by rag_node inside graph
                "reply": ""               # filled by llm_node inside graph
            })

        # 3. Add Grandpa's reply to chat history
        st.session_state.messages.append({
            "role": "grandpa",
            "content": result["reply"]
        })

        # 4. Rerun the page to show the new messages
        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: ADD MEMORY
# ══════════════════════════════════════════════════════════════════════════════
with tab_memory:

    st.markdown("#### Add a memory about Grandpa")
    st.markdown(
        "<p style='color:#6b7280; font-size:13px;'>The more memories you add, the better Grandpa can talk.</p>",
        unsafe_allow_html=True
    )

    # Some example memories to inspire the user
    st.markdown("**Example memories:**")
    examples = [
        "Loved fishing at dawn every Sunday morning",
        "Always said: 'Hard work is the only honest road'",
        "Wife's name was Mei. She made the best dumplings.",
    ]
    for ex in examples:
        st.markdown(f'<div class="memory-tag">💭 {ex}</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Memory input form
    with st.form(key="memory_form", clear_on_submit=True):
        new_memory = st.text_area(
            "Write a memory",
            placeholder="e.g. He grew up in Shandong and loved the countryside...",
            height=100
        )
        save = st.form_submit_button("💾 Save Memory")

    # Handle save
    if save and new_memory.strip():
        with st.spinner("Saving memory…"):
            add_memory(new_memory.strip())
        st.success("✅ Memory saved! Grandpa will remember this.")
