from rag_pipeline import query_faiss
from agent import agent_respond
from config import SUPPORT_EMAIL, SUPPORT_PHONE

def check_human_transfer(user_input, confidence_score):
    """Determines if the query should be transferred to a human."""
    trigger_phrases = ["speak to a human", "talk to an agent", "escalate", "real person"]
    
    if any(phrase in user_input.lower() for phrase in trigger_phrases) or confidence_score < 0.7:
        return True

    return False

def chat():
    """Main chatbot function."""
    while True:
        user_input = input("You: ")

        # Query FAISS & Generate AI response
        response, confidence = query_faiss(user_input)

        # Check for human transfer
        if check_human_transfer(user_input, confidence):
            print(f"AI Assistant: I understand you'd like to speak with a human. Contact support at {SUPPORT_EMAIL} or call {SUPPORT_PHONE}.")
        else:
            # Use Gemini API for a refined response
            refined_response = agent_respond(user_input)
            print(f"AI Assistant: {refined_response}")

if __name__ == "__main__":
    chat()
