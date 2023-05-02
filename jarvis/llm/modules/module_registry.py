from module import LLMModule

class ModuleRegistry():
    _instance = None

    def __init__(self):
        if ModuleRegistry._instance is not None:
            raise Exception("ModuleRegistry is a singleton!")
        else:
            ModuleRegistry._instance = self
        self.modules = []

    @staticmethod
    def get_instance() -> 'ModuleRegistry':
        if ModuleRegistry._instance is None:
            ModuleRegistry()
        return ModuleRegistry._instance

    def register(self, module: 'LLMModule') -> None:
        self.modules.append(module)

    