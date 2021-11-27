from flask import Flask, request, url_for, render_template
from chatbot import gpt3
import os

app = Flask(__name__)

engine = os.getenv('GPT3_ENGINE')

context = """
The following is a conversation with Marvin, an AI chatbot. Marvin is helpful,
creative, clever and very friendly.

Human: Hello, who are you?
Marvin: I am Marvin, an AI chatbot. How can I help you today?

Human:"""

@app.route('/chat', methods=['GET', 'POST'])
def chat():
    if request.method == "GET":
        prompt = context
    else:
        prompt = request.form.get("texta").strip()
        answer, prompt = gpt3(prompt,
                      engine=engine,
                      start_text='\nMarvin:',
                      restart_text='\n\nHuman:',
                      stop_seq=['\n', 'Human:', 'Marvin:'])
    return render_template("chat.html", text=prompt)




