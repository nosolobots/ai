import openai
import os

openai.api_key = os.getenv('OPENAI_API_KEY')

def gpt3(prompt, engine='curie', response_length=64,
         temperature=0.7, top_p=1.0, freq_penalty=.0, pres_penalty=.0,
         start_text='', restart_text='', stop_seq=[]):

    response = openai.Completion.create(
        prompt=prompt + start_text,
        engine=engine,
        max_tokens=response_length,
        temperature=temperature,
        top_p=top_p,
        frequency_penalty=freq_penalty,
        presence_penalty=pres_penalty,
        stop=stop_seq)

    answer = response.choices[0]['text']
    new_prompt = prompt + start_text + answer + restart_text

    return answer, new_prompt

def chat():
    prompt = """
    The following is a conversation with Marvin, an AI chatbot. Marvin is helpful, creative, clever, very friendly and funny.
    Human: Hello, who are you?
    Marvin: I am Marvin, an AI chatbot. How can I help you today?
    Human:"""

    print(prompt)
    while True:
        prompt += input("")
        answer, prompt = gpt3(prompt,
                              start_text='\nMarvin:',
                              restart_text='\nHuman:',
                              stop_seq=['\n', 'Human:', 'Marvin:'])
        print(answer)

if __name__=='__main__':
    chat()



