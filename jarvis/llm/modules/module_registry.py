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

    def evaluate(self, message: str) -> str | None:
        for module in self.modules:
            if module.should_activate(message):
                return module.activate(message)
        return None
