import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")


class LLM():
    def __init__(self, context_prompt: str) -> None:
        self.context_prompt = context_prompt
        self.message_history = [{
            "role": "system",
            "content": context_prompt
        }]
        self.token_usage = 0

    def prompt(self, message: str) -> str:
        self.message_history.append({
            "role": "user",
            "content": message
        })
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=self.message_history
        )

        self.message_history.append({
            "role": response.choices[0]['message']['role'],
            "content": response.choices[0]['message']['content']
        })

        self.token_usage += response['usage']['total_tokens']

        return response.choices[0]['message']['content']
