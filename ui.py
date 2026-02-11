import streamlit as st
import streamlit.components.v1 as components

def scroll_to_bottom():
    components.html(
        """
               <script>
               const main = window.parent.document.querySelector('section.main');
               if (main) {
                   main.scrollTo({ top: main.scrollHeight, behavior: 'smooth' });
               }
               </script>
               """,
        height=0,
    )


def floating_scroll_button():
    components.html(
        """
        <style>
        /* Hide iframe background space */
        body {
            margin: 0;
            background: transparent;
        }

        #scrollBtn {
            position: fixed;
            bottom: 80px;
            right: 24px;
            z-index: 999999;
            background: #4f46e5;
            color: white;
            border: none;
            border-radius: 50%;
            width: 48px;
            height: 48px;
            font-size: 22px;
            cursor: pointer;
            box-shadow: 0 4px 12px rgba(0,0,0,0.4);
        }
        #scrollBtn:hover { background: #4338ca; }
        </style>

        <button id="scrollBtn" title="Go to bottom">⬇️</button>

        <script>
        const btn = document.getElementById("scrollBtn");
        btn.onclick = function () {
            window.scrollTo({ top: document.body.scrollHeight, behavior: "smooth" });
        };
        </script>
        """,
        height=1,   # 👈 almost zero height, so white box invisible
    )



def render_sidebar(conversations, current_chat_id, on_new_chat, on_select_chat):
    with st.sidebar:
        if st.button("➕ New Chat"):
            on_new_chat()
            st.rerun()

        st.markdown("---")
        st.markdown("### 💬 Your Chats")

        for c in conversations:
            is_active = (c["id"] == current_chat_id)
            label = "👉 " + c["title"] if is_active else c["title"]

            if st.button(label, key=c["id"]):
                on_select_chat(c["id"])
                st.rerun()





        # st.markdown(:)

def render_header():
    st.set_page_config(page_title="C.B.R.I Procurebuddy", page_icon="🤖")
    st.title("🤖 C.B.R.I Purchase Assistant")
    st.caption("Powered by Groq (Llama 3) & GFR Rules")

def render_chat(messages, show_table_callback):
    user_count = 0  # sirf user messages ka counter

    for msg in messages:
        st.markdown("---")  # separator line

        with st.chat_message(msg["role"]):
            if msg["role"] == "user":
                user_count += 1
                # Sirf user ke liye number show karo
                st.markdown(
                    f"<div style='opacity:0.6; font-size:12px;'>#{user_count}</div>",
                    unsafe_allow_html=True
                )

            if msg["content"] == "__TABLE_SHOWN__":
                st.markdown("### 📊 CBRI / CSIR Purchase Process – Cost Slab Wise")
                show_table_callback()
            else:
                st.markdown(msg["content"])

    # Auto scroll to bottom
    scroll_to_bottom()



def render_input(is_busy: bool):
    user_input = st.chat_input(
        "Ask about CSIR/CBRI purchase rules, process, approvals...",
        disabled=is_busy
    )

    return user_input