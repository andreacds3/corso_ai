import os

import gradio as gr
from google import genai
from google.genai import types
from gradio import ChatMessage


GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
GEMINI_MODEL = 'gemini-2.5-flash'
# Only run this block for Gemini Developer API
client = genai.Client(api_key=GEMINI_API_KEY)

file_cache = {}


def chat(message, history, system_message):
    prompt = message["text"]
    files = message["files"]


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
                thinking_config=types.ThinkingConfig(
                    include_thoughts=True,
                    #thinking_level=types.ThinkingLevel.MEDIUM
                )
            )
    ):

        parts = chunk.candidates[0].content.parts
        current_chunk = parts[0].text

        if not thinking_complete:
            # Complete thought and start response
            thought_buffer += current_chunk
            history[-1] = ChatMessage(
                role="assistant",
                content=thought_buffer,
                metadata={"title": "⏳Thinking:"}
            )

            # Add response message
            history.append(
                ChatMessage(
                    role="assistant",
                    content=""
                )
            )
            thinking_complete = True

        elif thinking_complete:
            # Continue streaming response
            response_buffer += current_chunk
            history[-1] = ChatMessage(
                role="assistant",
                content=response_buffer
            )

        else:
            # Continue streaming thoughts
            thought_buffer += current_chunk
            history[-1] = ChatMessage(
                role="assistant",
                content=thought_buffer,
                metadata={"title": "⏳Thinking: "}
            )

        yield history


with gr.Blocks() as demo:
    system_message = gr.Text('Sei un assistente AI', label="Inserisci il system message")

    gr.ChatInterface(
        chat,
        title="Gemini Thinking Chatbot",
        description="Carica file e osserva il processo di ragionamento di Gemini.",
        multimodal=True,  # Abilita il caricamento di file
        additional_inputs=system_message
    )

demo.launch()
