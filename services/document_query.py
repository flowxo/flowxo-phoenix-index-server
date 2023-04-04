from ast import Dict
from typing import Optional

import tiktoken

from datastore.datastore import DataStore
from models.api import DocumentQueryRequest, DocumentQueryResponse, User
from models.models import (DocumentChunkWithScore, DocumentMetadataFilter,
                           DocumentResult, LanguageModelContext, Query)
from services.assemble_context import assemble_chunks_into_context

# Global variables
tokenizer = tiktoken.get_encoding(
    "cl100k_base"
) 

async def execute_document_query(datastore: DataStore, user: User, query_request: DocumentQueryRequest) -> DocumentQueryResponse:
    results = await datastore.query(
        [Query(query = query_request.query, top_k = query_request.top_k)],
        organization_id=user.organization_id,
        index_name=query_request.index_name,
    )
    if len(results) == 0:
        return []
    segments = results[0].results
    context = assemble_chunks_into_context(segments, max_tokens=query_request.max_context_tokens, source_metadata_field=query_request.source_metadata_field) if query_request.include_context else None
    context_token_count = len(tokenizer.encode(context, disallowed_special=())) if context else None
    docs:Dict[str, DocumentResult] = {}
    for segment in segments:
        docid = segment.metadata.document_id
        doc = docs.get(docid, None)
        if doc is None:
            doc = DocumentResult(
                id=docid,
                metadata=segment.metadata,
                index_name=segment.metadata.index_name,
                top_score=segment.score,
                total_score=segment.score,
                segment_count=1
            )
            docs[docid] = doc
        else:
            doc.total_score += segment.score
            doc.segment_count = doc.segment_count + 1
    context = LanguageModelContext(text=context, token_count=context_token_count) if context else None
    result = DocumentQueryResponse(results=list(docs.values()), context=context)
    return result
            
            