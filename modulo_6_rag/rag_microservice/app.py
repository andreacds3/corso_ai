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
from gemini_utils import (ask_gemini_to_reconcile_tables, ask_gemini_to_describe_table, ask_gemini_to_extract_keypoints, ask_gemini_to_extract_insights,
                          ask_gemini_to_extract_abstract, ask_gemini_to_extract_entities, ask_gemini_to_rewrite_the_query_in_affirmative_way, ask_gemini_to_answer_query)
from langchain_text_splitters import MarkdownHeaderTextSplitter
from transformers import AutoModel

API_DESCRIPTION = """
App to recover data from a rag pipeline
"""

class IndexingRequest(BaseModel):
    file_name: str
    file_content: str

COLLECTION_NAME = 'pdf'


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
        dense_embedding_model = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2", cache_dir='/tmp/fastembed_cache')
        bm25_embedding_model = SparseTextEmbedding("Qdrant/bm25", cache_dir='/tmp/fastembed_cache')
        late_interaction_embedding_model = LateInteractionTextEmbedding("colbert-ir/colbertv2.0", cache_dir='/tmp/fastembed_cache')
        client = qdrant_client.QdrantClient('http://vector_store:6333', timeout=1000)

        state_dict['pdf_processor'] = PdfConverter(artifact_dict=create_model_dict())
        state_dict['table_processor'] = TableConverter(artifact_dict=create_model_dict())
        state_dict['dense_embedding_model'] = dense_embedding_model
        state_dict['bm25_embedding_model'] = bm25_embedding_model
        state_dict['late_interaction_embedding_model'] = late_interaction_embedding_model
        state_dict['qdrant_client'] = client
        model = AutoModel.from_pretrained(
            'jinaai/jina-reranker-v3',
            dtype="auto",
            trust_remote_code=True,
        )
        model.to('cuda')
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
            temporary_file.write(base64.b64decode(document.file.encode()))
        pdf_rendered = pdf_processor(file_location)
        table_rendered = table_processor(file_location)
        text, _, images = text_from_rendered(pdf_rendered)
        tables, _, images = text_from_rendered(table_rendered)
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

        insights = ask_gemini_to_extract_insights(keypoints)

        abstract = ask_gemini_to_extract_abstract(insights)
        documenti_da_encodare = sezioni + ['\n\n'.join(key.keypoints) for key in keypoints] + [
            '\n\n'.join(insights.insights)] + [abstract]

        entitized_docs = dict()
        for doc in documenti_da_encodare:
            entitized_docs[doc] = ask_gemini_to_extract_entities(doc).entities

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

        operation_info = qdrant_client.upsert(
            collection_name=COLLECTION_NAME,
            points=points
        )
        return operation_info

    @app.post('/rest/async/rag')
    async def index(query: str):
        rewrited_query = ask_gemini_to_rewrite_the_query_in_affirmative_way(query)
        entities = ask_gemini_to_extract_entities(rewrited_query).entities
        dense_embedding_model = state_dict['dense_embedding_model']
        bm25_embedding_model = state_dict['bm25_embedding_model']
        late_interaction_embedding_model = state_dict['late_interaction_embedding_model']
        dense_embeddings = list(dense_embedding_model.embed(query))[0]
        bm25_embeddings = list(bm25_embedding_model.embed(query))[0]
        late_interaction_embeddings = list(late_interaction_embedding_model.embed(query))[0]

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

        results = qdrant_client.query_points(
            "pdf",
            prefetch=prefetch,
            query=late_interaction_embeddings,
            using="colbert",
            with_payload=True,
            limit=10,
        )

        results_no_entity = [res.payload['document'] for res in results.points]

        def create_should_clause(entities: List[str]):
            should = []
            for entity in entities:
                should.append(models.FieldCondition(key='metadata.entities[]', match=models.MatchText(text=entity)))
            return should

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

        results = await qdrant_client.query_points(
            "pdf",
            prefetch=prefetch,
            query=late_interaction_embeddings,
            using="colbert",
            with_payload=True,
            limit=10,
        )
        results_entity = [res.payload['document'] for res in results.points]

        results = set(results_entity + results_no_entity)
        model = state_dict['model']
        model.eval()
        results = model.rerank(rewrited_query, list(results))
        reranked_results = results[:5]
        answer = ask_gemini_to_answer_query(rewrited_query, reranked_results)
        return answer


    return app


app = create_app()
