import fastapi

import logging
import os
from contextlib import asynccontextmanager
import base64
import qdrant_client
from qdrant_client.models import PointStruct

from qdrant_client import models
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from marker.converters.pdf import PdfConverter
from marker.converters.table import TableConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered
from typing import List
from fastembed import TextEmbedding, SparseTextEmbedding, LateInteractionTextEmbedding
from fastapi.responses import StreamingResponse

from gemini_utils import (ask_gemini_to_reconcile_tables, ask_gemini_to_describe_table, ask_gemini_to_extract_keypoints, ask_gemini_to_extract_insights,
                          ask_gemini_to_extract_abstract, ask_gemini_to_answer_query_stream,
                          ask_gemini_to_extract_entities, ask_gemini_to_rewrite_the_query_in_affirmative_way, ask_gemini_to_answer_query)
from langchain_text_splitters import MarkdownHeaderTextSplitter
from transformers import AutoModel
from phoenix.otel import register
from opentelemetry.trace import Status, StatusCode
import torch
from marker.config.parser import ConfigParser
phoenix_project_name = "pdf-rag"


API_DESCRIPTION = """
App to recover data from a rag pipeline
"""

class IndexingRequest(BaseModel):
    file_name: str
    file_content: str

COLLECTION_NAME = 'pdf'
phoenix_project_name = "pdf_rag"

# With phoenix, we just need to register to get the tracer provider with the appropriate endpoint.
endpoint = "http://phoenix:6006/v1/traces"
tracer_provider_phoenix = register(project_name=phoenix_project_name, endpoint=endpoint, auto_instrument=True)
tracer = tracer_provider_phoenix.get_tracer(__name__)

def create_app() -> FastAPI:
    """
    Factory method to create the FastAPI app, its routes
    :return: the FastApi app configured
    """
    state_dict = {}

    @asynccontextmanager
    async def app_init(app: FastAPI):
        """
        Initialize the app by creating and initializing the summarizer service
        :param app: the fastapi app
        :return: None
        """
        dense_embedding_model = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2", cache_dir='/tmp/fastembed_cache', providers=["CPUExecutionProvider"])
        bm25_embedding_model = SparseTextEmbedding("Qdrant/bm25", cache_dir='/tmp/fastembed_cache', providers=["CPUExecutionProvider"])
        late_interaction_embedding_model = LateInteractionTextEmbedding("colbert-ir/colbertv2.0", cache_dir='/tmp/fastembed_cache', providers=["CPUExecutionProvider"])
        client = qdrant_client.QdrantClient('http://vector_store:6333', timeout=1000)
        config = {
            "torch_device": "cpu"

        }
        config_parser = ConfigParser(config)
        artifacts = create_model_dict()

        state_dict['pdf_processor'] = PdfConverter(artifact_dict=artifacts, config=config_parser.generate_config_dict())
        state_dict['table_processor'] = TableConverter(artifact_dict=artifacts, config=config_parser.generate_config_dict())
        state_dict['dense_embedding_model'] = dense_embedding_model
        state_dict['bm25_embedding_model'] = bm25_embedding_model
        state_dict['late_interaction_embedding_model'] = late_interaction_embedding_model
        state_dict['qdrant_client'] = client
        model = AutoModel.from_pretrained(
            'jinaai/jina-reranker-v3',
            torch_dtype=torch.bfloat16,  # Fondamentale: dimezza l'uso di VRAM e accelera il calcolo
            trust_remote_code=True,
            attn_implementation="sdpa"
        )
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        model.to(device)
        state_dict['model'] = model


        yield


    # create the app
    app = FastAPI(
        title="RAG",
        description=API_DESCRIPTION,
        version="0.0.1",
        docs_url='/',
        lifespan=app_init
    )



    @app.post('/rest/async/index_pdf_document')
    async def index(document: IndexingRequest):
        """
        Index a pdf document

        :param document: A document object with a *file_name* and a *file_content*

        :return: response: A boolean object representing if the file has been encoded

        """
        #STEP 1: Estrazione Markdown e tabelle
        pdf_processor = state_dict['pdf_processor']
        table_processor = state_dict['table_processor']
        file_location = os.path.join('/tmp', document.file_name)
        with open(file_location, 'wb') as temporary_file:
            temporary_file.write(base64.b64decode(document.file_content))
        pdf_rendered = pdf_processor(file_location)
        table_rendered = table_processor(file_location)
        text, _, images = text_from_rendered(pdf_rendered)
        print("text extracted")
        tables, _, images = text_from_rendered(table_rendered)
        print("tables extracted")

        tables_splitted = tables.split('\n\n')

        #STEP 2: Riconciliazione tabelle
        new_tables = list()
        for index in range(len(tables_splitted) - 1):
            result = ask_gemini_to_reconcile_tables(tables_splitted[index], tables_splitted[index + 1])
            print(result)
            if result.result:
                text = text.replace(tables_splitted[index + 1], '')
                text = text.replace(tables_splitted[index], tables_splitted[index] + '\n' + tables_splitted[index + 1])
                new_tables.append(tables_splitted[index] + '\n' + tables_splitted[index + 1])
            else:
                new_tables.append(tables_splitted[index])

        if not result.result:
            new_tables.append(tables_splitted[index + 1])
        tables_splitted = list(new_tables)
        print("tables merged")

        #STEP 3: descrizione tabelle
        for table in tables_splitted:
            description = ask_gemini_to_describe_table(table)
            text = text.replace(table,
                                f"Quì c'era una tabella che è stata sostituita dalla sua descrizione. \nDescrizione della tabella: {description}")

        #STEP 4: Split in sezioni
        text_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[('#', 1), ('##', 2), ('###', 3), ('####', 4)])
        documents = text_splitter.split_text(text)

        #Step 5: Estrazione PIRAMIDE Knowledge base
        sezioni = [doc.page_content for doc in documents]
        keypoints = []
        for sezione in sezioni:
            keypoints.append(ask_gemini_to_extract_keypoints(sezione))
        print("Keypoint extracted")
        insights = ask_gemini_to_extract_insights(keypoints)
        print("Insights extracted")

        abstract = ask_gemini_to_extract_abstract(insights)
        print("Abstract extracted")

        documenti_da_encodare = sezioni + ['\n\n'.join(key.keypoints) for key in keypoints] + [
            '\n\n'.join(insights.insights)] + [abstract]

        entitized_docs = dict()
        for doc in documenti_da_encodare:
            entitized_docs[doc] = ask_gemini_to_extract_entities(doc).entities
        print("Entities extracted")

        dense_embedding_model = state_dict['dense_embedding_model']
        bm25_embedding_model = state_dict['bm25_embedding_model']
        late_interaction_embedding_model = state_dict['late_interaction_embedding_model']
        dense_embeddings = list(dense_embedding_model.embed(doc for doc in entitized_docs.keys()))
        bm25_embeddings = list(bm25_embedding_model.embed(doc for doc in entitized_docs.keys()))
        late_interaction_embeddings = list(late_interaction_embedding_model.embed(doc for doc in entitized_docs.keys()))
        entities = [value for key, value in entitized_docs.items()]
        qdrant_client = state_dict['qdrant_client']
        points = []
        for idx, (dense_embedding, bm25_embedding, late_interaction_embedding, doc) in enumerate(
                zip(dense_embeddings, bm25_embeddings, late_interaction_embeddings, entitized_docs)):
            point = PointStruct(
                id=idx,
                vector={
                    "dense": dense_embedding,
                    "bm25": bm25_embedding.as_object(),
                    "colbert": late_interaction_embedding,
                },
                payload={"document": doc, "metadata": {"entities": entitized_docs[doc]}}
            )
            points.append(point)
        for point in points:
            operation_info = qdrant_client.upsert(
                collection_name=COLLECTION_NAME,
                points=[point]
            )
        return operation_info

    @app.post('/rest/async/rag')
    async def rag(query: str):


        rewrited_query = rewrite_query(query)
        entities = extract_entities(rewrited_query)
        dense_embeddings, bm25_embeddings, late_interaction_embeddings = encode_query(rewrited_query)
        results = retrieve(query,dense_embeddings, bm25_embeddings, late_interaction_embeddings, entities)
        reranked_result = rerank(query, results)
        answer_gemini = answer(query, reranked_result)
        return answer_gemini

    @app.post('/rest/async/rag_streaming')
    async def rag_streaming(query: str):
        rewrited_query = rewrite_query(query)
        entities = extract_entities(rewrited_query)
        dense_embeddings, bm25_embeddings, late_interaction_embeddings = encode_query(rewrited_query)
        results = retrieve(query, dense_embeddings, bm25_embeddings, late_interaction_embeddings, entities)
        reranked_result = rerank(query, results)
        answer_gemini = answer_stream(query, reranked_result)
        return StreamingResponse(answer_gemini)

    @tracer.chain
    def rewrite_query(query):
        rewrited_query = ask_gemini_to_rewrite_the_query_in_affirmative_way(query)
        return rewrited_query

    @tracer.chain
    def extract_entities(query):
        entities = ask_gemini_to_extract_entities(query).entities
        return entities

    def encode_query(query):
        dense_embedding_model = state_dict['dense_embedding_model']
        bm25_embedding_model = state_dict['bm25_embedding_model']
        late_interaction_embedding_model = state_dict['late_interaction_embedding_model']
        dense_embeddings = list(dense_embedding_model.embed(query))[0]
        bm25_embeddings = list(bm25_embedding_model.embed(query))[0]
        late_interaction_embeddings = list(late_interaction_embedding_model.embed(query))[0]
        return dense_embeddings, bm25_embeddings, late_interaction_embeddings

    def retrieve(query, dense_embeddings, bm25_embeddings, late_interaction_embeddings, entities):
        with tracer.start_as_current_span("retrieving_documents", openinference_span_kind='retriever') as span:
            # Log the event of starting retrieval
            span.add_event("Starting retrieve")
            # Record the input query as an attribute for visibility
            # Phoenix allows you to use span.set_input
            span.set_input(query)
            try:
                prefetch = [
                    models.Prefetch(
                        query=dense_embeddings,
                        using="dense",
                        limit=20,
                    ),
                    models.Prefetch(
                        query=models.SparseVector(**bm25_embeddings.as_object()),
                        using="bm25",
                        limit=20,
                    ),
                ]
                client = state_dict['qdrant_client']
                results = client.query_points(
                    "pdf",
                    prefetch=prefetch,
                    query=late_interaction_embeddings,
                    using="colbert",
                    with_payload=True,
                    limit=10,
                )

                results_no_entity = [res for res in results.points]

                prefetch = [
                    models.Prefetch(
                        query=dense_embeddings,
                        using="dense",
                        limit=20,
                        filter=models.Filter(should=create_should_clause(entities))
                    ),
                    models.Prefetch(
                        query=models.SparseVector(**bm25_embeddings.as_object()),
                        using="bm25",
                        limit=20,
                        filter=models.Filter(should=create_should_clause(entities))
                    ),
                ]

                results = client.query_points(
                    "pdf",
                    prefetch=prefetch,
                    query=late_interaction_embeddings,
                    using="colbert",
                    with_payload=True,
                    limit=10,
                )
                results_entity = [res for res in results.points]

                retrieved_docs = results_entity + results_no_entity
                retrieved_docs_ids = list(set([doc.id for doc in retrieved_docs]))
                # Record details about each retrieved document
                for i, id_ in enumerate(retrieved_docs_ids):
                    span.set_attribute(f"retrieval.documents.{i}.document.id", id_)
                    span.set_attribute(f"retrieval.documents.{i}.document.content", [doc.payload['document'] for doc in retrieved_docs if doc.id == id_][0])
                    #span.set_attribute(f"retrieval.documents.{i}.document.metadata", doc.payload['metadata']['entities'])
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.set_attribute("error.type", type(e).__name__)
                span.set_attribute("error.message", str(e))
                raise

            # Mark the span as successful if no error was raised
            span.set_status(Status(StatusCode.OK))
            return [doc.payload['document'] for doc in retrieved_docs]

    @tracer.chain
    def rerank(query, documents):
        model = state_dict['model']
        model.eval()
        device = next(model.parameters()).device
        print(f"Il modello è su: {device}")

        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                # Limita max_length! Se i tuoi documenti sono lunghi,
                # il tempo cresce in modo quadratico.
                results = model.rerank(
                    query,
                    documents,
                )

        return results[:5]

    @tracer.chain
    def answer(query, documents):
        answer_gemini = ask_gemini_to_answer_query(query, [result['document'] for result in documents])
        return answer_gemini

    @tracer.chain
    def answer_stream(query, documents):
        for chunk in ask_gemini_to_answer_query_stream(query, [result['document'] for result in documents]):
            yield chunk

    def create_should_clause(entities: List[str]):
        should = []
        for entity in entities:
            should.append(models.FieldCondition(key='metadata.entities[]', match=models.MatchText(text=entity)))
        return should

    return app


app = create_app()
