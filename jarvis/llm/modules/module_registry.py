import json

class ModuleRegistry():
    _instance = None

    def __init__(self):
        if ModuleRegistry._instance is not None:
            raise Exception("ModuleRegistry is a singleton!")
        ModuleRegistry._instance = self
        self.modules = {}

    @staticmethod
    def get_instance() -> 'ModuleRegistry':
        if ModuleRegistry._instance is None:
            ModuleRegistry()
        return ModuleRegistry._instance

    def register(self, module: 'LLMModule') -> None:
        self.modules[module.name] = module

    def gpt_function_call(self, function_call) -> str:
        arguments = []
        gpt_arguments = json.loads(function_call["arguments"])
        for value in gpt_arguments.values():
            arguments.append(value)

        print(f"[ModuleRegistry] Calling \"{function_call['name']}\" module with arguments {arguments}")
        try:
            result = self.modules[function_call["name"]].activate(*arguments)
            if len(result) > 500:
                result = result[:500] + "[...] Trimmed output."
            return result
        except:
            print("[WARNING] Failed to call module")
            return "Error while calling "+function_call["name"]+" function."
