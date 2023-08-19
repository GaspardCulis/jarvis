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

    def get_preprompts(self) -> list[dict]:
        out = []
        for module in self.modules.values():
            out = out + module.get_preprompts()
        return out

    def gpt_function_call(self, function_call) -> str:
        try:
            gpt_arguments = json.loads(function_call["arguments"])
        except json.decoder.JSONDecodeError:
            print("[WARNING] Invalid function call JSON arguments: ", function_call["arguments"])
            return "Invalid function call JSON arguments"

        print(f"[ModuleRegistry] Calling \"{function_call['name']}\" module with arguments {gpt_arguments}")
        try:
            result = self.modules[function_call["name"]].activate(gpt_arguments)
            if len(result) > 500:
                result = result[:500] + "[...] Trimmed output."
            return result
        except Exception as e:
            print("[WARNING] Failed to call module: ", e)
            return "Error while calling "+function_call["name"]+" function."
