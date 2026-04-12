from google import genai
from google.genai import types
import os
import json
from typing import List

from pydantic import BaseModel

class TableResult(BaseModel):
    result: bool

class KeyPoints(BaseModel):
    keypoints: List[str]

class Insights(BaseModel):
    insights: List[str]

class Entities(BaseModel):
    entities: List[str]

GEMINI_API_KEY = os.getenv('GEMINI_TOKEN')
GEMINI_MODEL = 'gemini-2.5-flash'
client = genai.Client(api_key=GEMINI_API_KEY)



def ask_gemini_to_reconcile_tables(table_1, table_2):
    system_message = """
    You will be presented with two tables in markdown format. The tables are extracted from a pdf file and converted in markdown format. The tables all covers cyber security related aspects. Given that a table in a pdf document can be splitted in multiple pages, the two tables can indeed be an uniuque table, of which the second is the continuation of the first in a different page.
    \nYou will be presented with two tables and you need to reconcile the tables if they are one the continuation of the other. Carefully analyze the header of the second table, that can actually be a simple row of the first table. 
    ## GUIDELINES
    1. A possible hint that the second table is a continuation of the first is that the number of columns are the same. 
    2. Another possible hint is that the header of the second table has fields that are not particular categories but instances of the columns of the first table.
    3. Presenting objects related to cybersecurity is not an hint because both tables presents cybersecurity aspects.
    4. If the header of the first table is the same as the header of the second table it means that the two tables are separated.
    Output a json  with an unique field called "result" and a boolean value: The value is true if you think that the second table is the continuation of the first, false otherwise. 
    IMPORTANT: DON'T ADD COMMENTS OR BLOCKS, RETURN ONLY THE JSON CONTENT.
    """
    prompt_template = """
    Analyze the following two tables, and evaluate if the second is a continuation of the first.
    TABLES:\n\nTABLE_1:\n{{table_1}}\n\nTABLE_2:\n{{table_2}}
    """

    prompt = prompt_template.replace('{{table_1}}', table_1).replace('{{table_2}}', table_2)
    parts = [
        types.Part.from_text(text=prompt),
    ]
    content_list = [
        types.Content(
            role='user',
            parts=parts
        )
    ]
    result = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=content_list,
        config=types.GenerateContentConfig(
            system_instruction=system_message,
            temperature=0.3,
            response_mime_type='application/json',
            response_schema=TableResult)
    )
    return result.parsed

def ask_gemini_to_describe_table(table):
    system_message = """
    You are presented with a table in markdown format.
    You are asked to provide a clear description in italian language of the table structure, and a clear and explanatory description of each row, one row at a time.
    Don't miss any row and don't use any Header (#,##,###,####)!!\n
    """
    prompt_template = """
    Please describe the following table:\n\n {{table}}\n\n
    """

    prompt = prompt_template.replace('{{table}}', table)
    parts = [
        types.Part.from_text(text=prompt),
    ]
    content_list = [
        types.Content(
            role='user',
            parts=parts
        )
    ]
    result = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=content_list,
            config=types.GenerateContentConfig(
                system_instruction=system_message,
                temperature=0.3)
    )
    return result.text

def ask_gemini_to_extract_keypoints(section: str):
    system_message = """You are a System specialized in extracting valuable keypoints from pdf sections. 
    You are presented with a text in markdown format, and you need to extract  key points from the text: 
    ##GUIDELINES
    1. If the text doesn't have meaningful content, for example if it contains only section headers, footers, titles, links or captions or table identifiers, it's absolutely mandatory to answer with an empty list: 
    2. Each key point SHOULD BE auto-consistent and should contain all the informations needed to understand its content.
    3. Answer in Italian.
    4. Don't miss any general entities present in the text"""
    prompt_template = """Extract the key points from the following text: \n\n{{text}}\n\n"""
    prompt = prompt_template.replace('{{text}}', section)
    parts = [
        types.Part.from_text(text=prompt),
    ]
    content_list = [
        types.Content(
            role='user',
            parts=parts
        )
    ]
    result = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=content_list,
            config=types.GenerateContentConfig(
                system_instruction=system_message,
                temperature=0.3,
                response_mime_type='application/json',
            response_schema=KeyPoints)
    )
    return result.parsed

def ask_gemini_to_extract_insights(keypoints: List[KeyPoints]):
    system_message = """    You are an helpful and skilled analyst, and you are very good in creating valuable insight form a list of given key points extracted from a pdf file. 
    You are presented with a list of key_points and you are asked to select good key points from them, merge and fuse relative key points or cancel key points that doesn't give any useful information., extracting valuable insights from them.
    ## GUIDELINES
    1. If two key points are related, merge them in an unique insight maintaining all the info.
    2. If one keypoint is not auto-explanatory, (for example if it refers to something that is not present in the keypoint), then rephrase it using other key points.
    3. Answer in Italian
    4. Be verbose."""
    prompt_template = """Select, merge and rephrase the following key points: \n\n{{key_points}}\n\n"""
    prompt = prompt_template.replace('{{key_points}}', '\n\n'.join([key for key_list in keypoints for key in key_list.keypoints]))
    parts = [
        types.Part.from_text(text=prompt),
    ]
    content_list = [
        types.Content(
            role='user',
            parts=parts
        )
    ]
    result = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=content_list,
            config=types.GenerateContentConfig(
                system_instruction=system_message,
                temperature=0.3,
                response_mime_type='application/json',
            response_schema=Insights)
    )
    return result.parsed

def ask_gemini_to_extract_abstract(insights: Insights):
    system_message = """    You are an helpful and skilled analyst, and you are very good in creating valuable abstract form a list of given insights extracted from a pdf file. 
    You are presented with a list of insights and you are asked to create an excellent abstract from them in Italian."""
    prompt_template = """Create an abstract from the following insights: \n\n{{insights}}\n\n"""
    prompt = prompt_template.replace('{{insights}}', '\n\n'.join([key for key in insights.insights]))
    parts = [
        types.Part.from_text(text=prompt),
    ]
    content_list = [
        types.Content(
            role='user',
            parts=parts
        )
    ]
    result = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=content_list,
            config=types.GenerateContentConfig(
                system_instruction=system_message,
                temperature=0.3)
    )
    return result.text

def ask_gemini_to_extract_entities(text: str):
    system_message = """    You are an helpful and skilled analyst, and you are very good in extracting entities from a given text. 
    You are presented with a text and you are asked to extract entities from it.
    ## GUIDELINES
    1. Extract only Organization names, Locations and person names
    3. Answer in Italian"""
    prompt_template = """Extract entities from the given text: \n\n{{text}}"""
    prompt = prompt_template.replace('{{text}}', text)
    parts = [
        types.Part.from_text(text=prompt),
    ]
    content_list = [
        types.Content(
            role='user',
            parts=parts
        )
    ]
    result = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=content_list,
            config=types.GenerateContentConfig(
                system_instruction=system_message,
                temperature=0.3,
                response_mime_type='application/json',
            response_schema=Entities)
    )
    return result.parsed

def ask_gemini_to_rewrite_the_query_in_affirmative_way(query: str):
    system_message = """You are an AI language model assistant. Your task is to generate ONLY ONE alternative affirmative version in Italian of the given query to be processed by a RAG system. Don't add any comment or explanation. Just rewrite the query in plain text"""
    prompt_template = """Please rewrite the following query.
    query: {{query}}
    \n\n"""
    prompt = prompt_template.replace('{{query}}', query)
    parts = [
        types.Part.from_text(text=prompt),
    ]
    content_list = [
        types.Content(
            role='user',
            parts=parts
        )
    ]
    result = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=content_list,
            config=types.GenerateContentConfig(
                system_instruction=system_message,
                temperature=0.3)
    )
    return result.text



def ask_gemini_to_answer_query(query: str, documents: List[str]):
    system_message = """You are an excellent AI assitant that answer user queries given relevant documents. The documents are sorted by relevance (higher means more relevant).
    Rispondi solo con affermazioni fattuali basati sui documenti recuperati.
    Cita le tue fonti alla fine di ogni frase utilizzando [1], [2] ecc
    """
    prompt_template = """Answer the given query using the given context documents: \n\nQUERY: {{query}}. \n\nDOCUMENTS: {{documents}}"""
    prompt = prompt_template.replace('{{query}}', query).replace('{{documents}}', '\n\n'.join(documents))
    parts = [
        types.Part.from_text(text=prompt),
    ]
    content_list = [
        types.Content(
            role='user',
            parts=parts
        )
    ]
    result = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=content_list,
            config=types.GenerateContentConfig(
                system_instruction=system_message,
                temperature=0.3)
    )
    return result.text

def ask_gemini_to_answer_query_stream(query: str, documents: List[str]):
    system_message = """You are an excellent AI assitant that answer user queries given relevant documents. The documents are sorted by relevance (higher means more relevant).
    Rispondi solo con affermazioni fattuali basati sui documenti recuperati.
    Cita le tue fonti alla fine di ogni frase utilizzando [1], [2] ecc
    """
    prompt_template = """Answer the given query using the given context documents: \n\nQUERY: {{query}}. \n\nDOCUMENTS: {{documents}}"""
    prompt = prompt_template.replace('{{query}}', query).replace('{{documents}}', '\n\n'.join(documents))
    parts = [
        types.Part.from_text(text=prompt),
    ]
    content_list = [
        types.Content(
            role='user',
            parts=parts
        )
    ]
    for chunk in client.models.generate_content_stream(
            model=GEMINI_MODEL,
            contents=content_list,
            config=types.GenerateContentConfig(
                system_instruction=system_message,
                temperature=0.3)
    ):
        yield chunk.text
