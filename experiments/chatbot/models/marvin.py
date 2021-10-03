import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

start_sequence = "\nMarvin:"
restart_sequence = "\nHuman: "

response = openai.Completion.create(
  engine="curie",
  prompt="The following is a conversation with Marvin, an AI chatbot. Marvin is helpful, creative, clever, very friendly and funny.\n\nHuman: Hello, who are you?\nMarvin: I am Marvin, an AI chatbot. How can I help you today?\nHuman:",
  temperature=0.9,
  max_tokens=150,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0.6,
  stop=["\n", " Human:", "Marvin:"]
)

