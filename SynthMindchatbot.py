import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Initialize Streamlit app
st.title("Welcome to ShopWise Solutions")
st.header("The initialisation of the agent will take a while on the initial load as the tensors are downloaded. Please be patient.")

# Sidebar for model selection and configurations
st.sidebar.header("Model Configuration")
model_name = st.sidebar.text_input("Enter Hugging Face Model Name", value="owiyedouglas/Qwen2.5_finetuned_V1_100")

# Load model and tokenizer
@st.cache_resource
def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model

try:
    tokenizer, model = load_model(model_name)
    st.sidebar.success(f"Model {model_name} loaded successfully!")
except Exception as e:
    st.sidebar.error(f"Failed to load model: {e}")

# Chat interface
st.subheader("Chat Interface")
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# User input
user_query = st.text_input("Your Message:", key="user_input")

# Respond to user query
if st.button("Send") and user_query.strip():
    with st.spinner("Generating response..."):
        try:
            # Tokenize user input
            input_ids = tokenizer.encode(user_query + tokenizer.eos_token, return_tensors="pt")
            
            # Generate model response
            chat_history_ids = input_ids if len(st.session_state.chat_history) == 0 else torch.cat([st.session_state.chat_history[-1], input_ids], dim=-1)
            response_ids = model.generate(chat_history_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
            response = tokenizer.decode(response_ids[:, chat_history_ids.shape[-1]:][0], skip_special_tokens=True)
            
            # Update chat history
            st.session_state.chat_history.append((user_query, response))
        except Exception as e:
            st.error(f"Error in generating response: {e}")

# Display chat history
st.subheader("Chat History")
if st.session_state.chat_history:
    for i, (query, response) in enumerate(st.session_state.chat_history):
        st.markdown(f"**You**: {query}")
        st.markdown(f"**Model**: {response}")
else:
    st.write("No conversation yet. Start chatting!")

# Clear chat history
if st.button("Clear Chat"):
    st.session_state.chat_history = []
    st.success("Chat history cleared!")
