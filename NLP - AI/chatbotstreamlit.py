import streamlit as st
from nltk.chat.util import Chat, reflections

# Chatbot pairs
pairs = [
    [r"(hi|hello|hey)(.*)", 
     ["Hello! 👋 Welcome to QuickBuddy Delivery. How can I help you today?"]],

    [r"my name is (.*)", 
     ["Hi %1! 😊 How can I assist you with your order today?"]],

    [r"(.*)your name\??", 
     ["I'm QuickBuddy Assistant 🤖, here to help you!"]],

    [r"(.*)(track|status)(.*)", 
     ["📦 Track your order in the 'My Orders' section."]],

    [r"(.*)(delivery time|when will it arrive)(.*)", 
     ["⏱️ Delivery usually takes 30–45 minutes."]],

    [r"(.*)(cancel order)(.*)", 
     ["❌ Cancel your order before dispatch."]],

    [r"(.*)(refund|money back)(.*)", 
     ["💰 Refunds take 5–7 days."]],

    [r"(.*)(payment issue|failed payment)(.*)", 
     ["💳 Please retry or use another payment method."]],

    [r"(.*)(complaint|problem|issue)(.*)", 
     ["⚠️ Please describe your issue."]],

    [r"(.*)(offer|discount)(.*)", 
     ["🎁 Check the Offers section in the app."]],

    [r"(.*)", 
     ["🤖 I'm not sure. Our support team will contact you."]]
]

chatbot = Chat(pairs, reflections)

# Page config
st.set_page_config(page_title="QuickBuddy Chatbot", page_icon="🤖")

st.title("🚀 QuickBuddy Delivery Chatbot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input box (chat-style)
user_input = st.chat_input("Type your message...")

if user_input:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Bot response
    response = chatbot.respond(user_input)

    # Show bot response
    st.session_state.messages.append({"role": "assistant", "content": response})

    # Refresh display
    st.rerun()
