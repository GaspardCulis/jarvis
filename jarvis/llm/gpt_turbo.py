import openai
import os
from jarvis.llm.contexts import JARVIS_CONTEXT
from jarvis.llm.modules.module_registry import ModuleRegistry

openai.api_key = os.getenv("OPENAI_API_KEY")


class LLM():
    def __init__(self) -> None:
        self.message_history = JARVIS_CONTEXT
        self.token_usage = 0

    def prompt(self, message: str | dict):
        if isinstance(message, str):
            self.message_history.append({
                "role": "user",
                "content": message
            })
        else:
            self.message_history.append(message)

        functions = []
        for module in ModuleRegistry.get_instance().modules.values():
            functions.append({
                "name": module.name,
                "description": module.description,
                "parameters": {
                    "type": "object",
                    "properties": module.parameters
                }
            })

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k-0613",
            messages=self.message_history,
            functions=functions,
            function_call="auto",
        )

        self.message_history.append(response.choices[0]['message'])

        self.token_usage += response['usage']['total_tokens']

        return response.choices[0]['message']
