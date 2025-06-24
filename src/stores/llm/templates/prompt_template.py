class PromptTemplate:
    def __init__(self):
        self.system_prompt = (
            "You are a highly skilled assistant that helps with answering questions based on the context "
            "provided by the user. Your role is to read and understand the given context, then generate a "
            "concise and accurate answer to the user's question."
        )

    def create_question_prompt(self, question: str, context: str) -> str:
        return f"Question: {question}\nContext: {context}\nAnswer:"

    def create_system_prompt(self) -> str:
        return self.system_prompt

    def create_custom_prompt(self, custom_instruction: str) -> str:
        return f"Instruction: {custom_instruction}\n{self.system_prompt}"

    def format_answer(self, answer: str) -> str:
        return f"Answer: {answer}\nEnd of Answer."
