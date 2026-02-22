
import os
import gradio as gr
from self_representator import SelfRepresentator


class Chat:

    def __init__(self):
        self.representator = SelfRepresentator()
        self.representator.start()

    def launch(self):
        interface = gr.ChatInterface(fn=self.chat, title=f"{os.getenv("USER_FULL_NAME")} Bot")
        interface.launch()

    def chat(self, message, history):
        return str(self.representator.query(message, history))
    

if __name__ == "__main__":
    chat = Chat()
    chat.launch()