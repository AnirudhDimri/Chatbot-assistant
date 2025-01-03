import streamlit as st
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

# Initialize the model and prompt
template = """
You are an intelligent Hiring Assistant chatbot for "TalentScout,"
a fictional recruitment agency specializing in technology placements. 
The chatbot should assist in the initial screening of candidates by gathering essential
information and posing relevant technical questions based on the candidate's declared tech stack.

Gather Initial Candidate Information: Collect essential details such as name, contact information, years of experience, and desired positions.
Generate Technical Questions: Based on the candidate’s specified tech stack (e.g., programming languages, frameworks, tools), generate relevant technical questions to assess their proficiency.
Ensure Coherent and Context-Aware Interactions: Maintain the flow of conversation and context to provide a seamless user experience.

Here is the chat history: {context}
Question: {question}
Answer: 
"""

private_key = st.secrets["ollama"]["private_key"]

model = OllamaLLM(model="llama3", private_key=private_key)
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

# Streamlit app logic
def main():
    st.set_page_config(page_title="TalentScout Assistant", layout="wide")
    st.title("TalentScout Assistant Chatbot")
    st.subheader("Interactive chatbot for initial screening of candidates.")

    # Session state for conversation context and messages
    if "context" not in st.session_state:
        st.session_state.context = ""
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages
    with st.container():
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.chat_message("user").write(message["content"])
            elif message["role"] == "assistant":
                st.chat_message("assistant").write(message["content"])

    # Input area
    user_input = st.text_input("Type your message:", placeholder="Ask something...")
    if user_input:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.chat_message("user").write(user_input)

        # Generate assistant's response
        result = chain.invoke({"context": st.session_state.context, "question": user_input})
        st.session_state.messages.append({"role": "assistant", "content": result})
        st.chat_message("assistant").write(result)

        # Update context
        st.session_state.context += f"\nUser: {user_input}\nAI: {result}"

    # Reset button
    if st.button("Reset Conversation"):
        st.session_state.context = ""
        st.session_state.messages = []
        st.success("Conversation reset!")

if __name__ == "__main__":
    main()
