import os

import gradio as gr
from google import genai
from google.genai import types
from gradio import ChatMessage
import requests

GEMINI_API_KEY = os.getenv('GEMINI_TOKEN')
GEMINI_MODEL = 'gemini-2.5-flash'
# Only run this block for Gemini Developer API
client = genai.Client(api_key=GEMINI_API_KEY)

file_cache = {}


def chat(message, history, system_message, rag = True):
    prompt = message["text"]
    files = message["files"]
    if not rag:

        content_list = []
        for message_hist in history:
            role = message_hist['role']
            content = message_hist['content']
            history_single_parts = []
            for part in content:
                if 'file' in part:
                    file = part['file']['path']
                    up_file = file_cache[file]
                    history_single_parts.append(types.Part.from_uri(file_uri=up_file[0], mime_type=up_file[1]))
                else:
                    text = part['text']
                    history_single_parts.append(types.Part.from_text(text=text))
            content_list.append(
                types.Content(
                    role=role if role == 'user' else 'model',
                    parts=history_single_parts
                )
            )

        parts = [
            types.Part.from_text(text=prompt),
        ]

        if files:
            for file in files:
                uploaded_file = client.files.upload(file=file)
                file_cache[file] = (uploaded_file.uri, uploaded_file.mime_type)
                parts.append(types.Part.from_uri(file_uri=uploaded_file.uri, mime_type=uploaded_file.mime_type))

        content_list.append(
            types.Content(
                role='user',
                parts=parts
            )
        )

        history.append(
            ChatMessage(
                role="assistant",
                content="",
                metadata={"title": "⏳Thinking:"}
            )
        )

        # Initialize buffers
        thought_buffer = ""
        response_buffer = ""
        thinking_complete = False

        for chunk in client.models.generate_content_stream(
                model=GEMINI_MODEL,
                contents=content_list,
                config=types.GenerateContentConfig(
                    system_instruction=system_message,
                    temperature=0.3,
                    thinking_config=types.ThinkingConfig(include_thoughts=True)
                )
        ):
            # Gemini 2.0 Flash Thinking separa i pensieri dal testo nei parts
            for part in chunk.candidates[0].content.parts:
                if part.thought:
                    thought_buffer += part.text
                    # Durante il pensiero:
                    # - status="pending" mostra lo spinner e tiene aperto l'accordion
                    # - log può contenere il testo del pensiero
                    yield gr.ChatMessage(
                        role="assistant",
                        content="",
                        metadata={
                            "title": "Ragionamento in corso...",
                            "log": thought_buffer,
                            "status": "pending"
                        }
                    )

                elif part.text:
                    response_buffer += part.text
                    # Durante la risposta:
                    # - content contiene la risposta vera e propria (fuori dal blocco)
                    # - status="done" chiude l'accordion del pensiero automaticamente
                    yield [gr.ChatMessage(
                        role="assistant",
                        content="",
                        metadata={
                            "title": "Pensiero completato",
                            "log": thought_buffer,
                            "status": "done"
                        }),
                     gr.ChatMessage(
                        role="assistant",
                        content=response_buffer,

                    )]

    else:
        res = requests.post(f'http://localhost:8443/rest/async/rag_streaming?query={prompt}', stream=True)
        yield ChatMessage(role="assistant", content="")
        response_buffer = ""

        for chunk in res:
            if chunk:
                decoded_chunk = chunk.decode('utf-8')
                response_buffer += decoded_chunk

                yield ChatMessage(
                    role="assistant",
                    content=response_buffer
                )


with gr.Blocks() as demo:
    system_message = gr.Text('Sei un assistente AI', label="Inserisci il system message")
    rag = gr.Checkbox(True, label="Select se vuoi abilitare il RAG")
    gr.ChatInterface(
        chat,
        title="Gemini RAG Chatbot",
        description="Fai una domanda al chatbot_researcher Gemini",
        multimodal=True,  # Abilita il caricamento di file
        additional_inputs=[system_message, rag]
    )

demo.launch()
