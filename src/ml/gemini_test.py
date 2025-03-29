import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv(override=True)

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def gemini_representation_model(topic_words):
    model = genai.GenerativeModel('gemini-2.5-pro-exp-03-25')
    prompt = f"Create **a single, concise topic title** (ideally 2-5 words) using the main ideas from these keywords: {', '.join(topic_words)}. **Output only the title.**"
    response = model.generate_content(prompt)
    
    return response.text if response.text else "No representation available"


topic_words = ('henry', 'henry is', 'blacksmith', 'as henry', 'son', 'his', 'of henry', 'blacksmith son', 'henry the', 'henry and')

print(gemini_representation_model(topic_words))