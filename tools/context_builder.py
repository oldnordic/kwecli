"""Rank, chunk, and build context window."""

from typing import List, Dict, Any, Optional
import re


def rank_documents(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Rank documents by relevance score."""
    return sorted(documents, key=lambda x: x.get('score', 0), reverse=True)


def chunk_text(text: str, max_chunk_size: int = 1000) -> List[str]:
    """Split text into chunks of specified size."""
    if len(text) <= max_chunk_size:
        return [text]
    
    # Simple chunking by sentences
    sentences = re.split(r'[.!?]+', text)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        if len(current_chunk) + len(sentence) <= max_chunk_size:
            current_chunk += sentence + ". "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks


def build_context_window(documents: List[Dict[str, Any]], max_tokens: int = 4000) -> str:
    """Build context window from ranked documents."""
    ranked_docs = rank_documents(documents)
    
    context_parts = []
    current_length = 0
    
    for doc in ranked_docs:
        title = doc.get('title', 'Unknown')
        content = doc.get('content', '')
        
        # Estimate tokens (rough approximation: 1 token ≈ 4 characters)
        doc_text = f"Title: {title}\nContent: {content}\n\n"
        estimated_tokens = len(doc_text) // 4
        
        if current_length + estimated_tokens <= max_tokens:
            context_parts.append(doc_text)
            current_length += estimated_tokens
        else:
            break
    
    return "".join(context_parts).strip()


def merge_small_chunks(chunks: List[str], min_chunk_size: int = 100) -> List[str]:
    """Merge small chunks to create more meaningful segments."""
    if not chunks:
        return []
    
    merged_chunks = []
    current_chunk = ""
    
    for chunk in chunks:
        if len(current_chunk) + len(chunk) < min_chunk_size:
            current_chunk += " " + chunk if current_chunk else chunk
        else:
            if current_chunk:
                merged_chunks.append(current_chunk.strip())
            current_chunk = chunk
    
    if current_chunk:
        merged_chunks.append(current_chunk.strip())
    
    return merged_chunks


def create_context_summary(documents: List[Dict[str, Any]]) -> str:
    """Create a summary of the context for quick reference."""
    if not documents:
        return "No relevant documents found."
    
    summary_parts = []
    for i, doc in enumerate(documents[:3], 1):  # Top 3 documents
        title = doc.get('title', 'Unknown')
        doc_type = doc.get('type', 'document')
        score = doc.get('score', 0)
        
        summary_parts.append(
            f"{i}. {title} ({doc_type}, score: {score:.2f})"
        )
    
    return f"Found {len(documents)} relevant documents:\n" + "\n".join(summary_parts)
