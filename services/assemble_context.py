from typing import Dict, List, Optional

import tiktoken

from models.models import (DocumentChunkWithScore, DocumentMetadata,
                           SourceMetadataField)
from services.openai import get_chat_completion

# Global variables
tokenizer = tiktoken.get_encoding(
    "cl100k_base"
) 

def get_source_text(metadata: DocumentMetadata, field: SourceMetadataField):
    if field == SourceMetadataField.url:
        return metadata.url
    elif field == SourceMetadataField.document_id:
        return metadata.document_id
    elif field == SourceMetadataField.title:
        return metadata.title
    elif field == SourceMetadataField.auto:
        return metadata.url or metadata.title or metadata.document_id
    else:
        return None

def assemble_chunks_into_context(query_results: List[DocumentChunkWithScore], max_tokens: Optional[int], source_metadata_field: SourceMetadataField) -> str:
    """
    Takes in a list of document chunks and returns a context string.
    """

    # Initialize an empty dictionary of documents where we will accumulate results until we run out of tokens
    docs: Dict[str, str] = {}
    sources: Dict[str, str] = {}
    # Initialize a counter for the number of tokens
    num_tokens = 0
    max_tokens = 4000 if max_tokens is None else max_tokens

    # Loop until all tokens are consumed
    for chunk in query_results:
        docid = chunk.metadata.document_id
        chunk_text = chunk.text
        chunk_token_count = len(tokenizer.encode(chunk_text, disallowed_special=()))
        # Stop if we have reached the maximum number of tokens
        if (num_tokens + chunk_token_count) > max_tokens:
            break

        num_tokens += chunk_token_count        
        # Add the chunk text to the document
        docs[docid] = docs.get(docid, 'Content: ') + chunk.text + '\n\n'
        # Add the chunk source to the source map
        if source_metadata_field != SourceMetadataField.none and chunk.metadata:
            sources[docid] = get_source_text(chunk.metadata, source_metadata_field)

    if source_metadata_field != SourceMetadataField.none:
        for docid in docs.keys():
            if sources.get(docid):
                docs[docid] += 'Source: ' + sources[docid] + '\n\n'

    # Simply concatenate all the documents together
    context = ("\n###\n").join(docs.values()).strip()
    return context
