from jarvis.llm.gpt_turbo import LLM

llm = LLM("Hello, I'm Jarvis. I'm a chatbot that can help you with your daily tasks. What can I do for you today?")
while True:
    message = input("You: ")
    response = llm.prompt(message)
    print(f"Jarvis: {response}")