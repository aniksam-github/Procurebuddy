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

    if st.button("⚙️ Settings"):
        st.session_state.show_settings = True
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

def render_auth_screen(auth_handlers):
    """
    auth_handlers = {
        "login": fn(email, password) -> (ok, result_or_msg),
        "create": fn(email) -> (ok, temp_password_or_msg),
        "reset": fn(email) -> (ok, temp_password_or_msg),
    }
    """

    st.title("🔐 CBRI ProcureBuddy - Login")
    tab = st.radio("Choose option", ["Login", "Create Account", "Reset Password"], horizontal=True)

    if tab == "Login":
        email = st.text_input("Official Email")
        password = st.text_input("Password", type="password")

        if st.button("Login", type="primary"):
            ok, result = auth_handlers["login"](email, password)
            if not ok:
                st.error(result)
            else:
                st.success("Login successful !!!")
                return {"action": "login_success", "user": email, "user_record": result}

    elif tab == "Create Account":
        email = st.text_input("Official Email (CBRI)")

        if st.button("Create Account", type="primary"):
            ok, result = auth_handlers["create"](email)
            if not ok:
                st.error(result)
            else:
                st.success("Account Created!!!")
                st.info(f"Your temporary password is: {result}\n\nPlease login and change it immediately.")

    elif tab == "Reset Password":
        email = st.text_input("Official Email")

        if st.button("Reset Password", type="primary"):
            ok, result = auth_handlers["reset"](email)
            if not ok:
                st.error(result)
            else:
                st.success("Password Reset Successful!")
                st.info(f"Your new temporary password is: {result}\n\nPlease login and change it immediately.")

    return {"action": "none"}


def render_force_change_password():
    st.title("🔑 Change Your Password")

    new_pw = st.text_input("New Password", type="password")
    confirm_pw = st.text_input("Confirm New Password", type="password")

    if st.button("Change Password", type="primary"):
        if not new_pw or not confirm_pw:
            st.error("Please fill both fields.")
        elif new_pw != confirm_pw:
            st.error("Passwords do not match.")
        else:
            return {"action": "change_password", "new_password": new_pw}

    return {"action": "none"}

def render_verify_otp_screen(email):
    st.title("📩 Verify OTP")

    otp = st.text_input("Enter OTP")
    pw1 = st.text_input("New Password", type="password")
    pw2 = st.text_input("Confirm Password", type="password")

    if st.button("Verify & Create Account"):
        if pw1 != pw2:
            st.error("Passwords do not match.")
        else:
            return {
                "action":"verify_otp",
                "email": email,
                "otp": otp,
                "password": pw1
            }

    return {
        "action" : "none"
    }

def render_totp_verify_screen():
    st.title("🔐 Two-Factor Authentication")

    code = st.text_input("Enter 6-digit code from Authenticator App", max_chars=6)
    if st.button("verify", type="primary"):
        if not code or len(code) != 6:
            st.error("Please enter a valid 6-digit code.")
        else:
            return {"action": "verify_totp", "code": code}

    return {
        "action" : "none"
    }


def render_enable_totp_screen(qr_base64):
    st.title("Enable Two-Factor Authentication")
    st.markdown("Scan this QR code in Google/Microsoft Authenticator: ")
    st.image(f"data:image/png;base64, {qr_base64}")
    code = st.text_input("enter 6-dit code to confirm", max_chars=6)

    if st.button("confirm & Enable", type="primary"):
        if not code or len(code) != 6:
            st.error("Enter valid 6-digit code")
        else:
            return {"action" : "confirm_totp", "code":code}

    return {
        "action" : "none"
    }

def render_settings_screen(is_totp_enabled):
    st.title("⚙️ Settings")
    st.markdown("### 🔐 Security")

    if not is_totp_enabled:
        if st.button("Enable 2FA (TOTP)", type="primary"):
            return {
                "action" : "enable_totp"
            }
    else:
        st.success("✅ Two-Factor Authentication is already enabled")

    st.markdown("---")

    if st.button("⬅ Back"):
        return {
            "action" : "back"
        }
    return {
        "action" : "none"
    }