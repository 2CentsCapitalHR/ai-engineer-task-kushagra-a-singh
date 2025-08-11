#!/usr/bin/env python3
"""
ChromaDB Backend for ADGM RAG System
Clean implementation focusing only on vector storage and retrieval
"""

import json
import logging
import os
import uuid
from typing import Any, Dict, List, Optional

import google.generativeai as genai
from tqdm import tqdm

# Configure logging
logger = logging.getLogger(__name__)

# Load configuration
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
MODEL_NAME = os.getenv("GOOGLE_MODEL", "gemini-2.0-flash")

# Import ADGM reference data
from adgm_rag import REFERENCE_SNIPPETS


class ADGMChromaRAG:
    """ChromaDB backend for ADGM legal document analysis"""

    def __init__(self, persist_directory: str = None):
        """Initialize ChromaDB client and collection"""
        if persist_directory is None:
            from config import CHROMA_DB_DIR

            persist_directory = str(CHROMA_DB_DIR)

        self.persist_directory = persist_directory

        # Initialize ChromaDB
        try:
            import chromadb

            self.client = chromadb.PersistentClient(path=persist_directory)
            logger.info(f"ChromaDB initialized at: {persist_directory}")
        except ImportError:
            logger.error(
                "ChromaDB package not found. Install with: pip install chromadb>=0.4.0"
            )
            raise
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise

        # Get or create collection
        try:
            self.collection = self.client.get_or_create_collection(
                name="adgm_legal_docs",
                metadata={"description": "ADGM legal documents and regulations"},
            )
            logger.info("ADGM document collection ready")
        except Exception as e:
            logger.error(f"Failed to create collection: {e}")
            raise

        # Initialize knowledge base
        self._initialize_knowledge_base()

    def _initialize_knowledge_base(self):
        """Load ADGM reference documents into ChromaDB"""
        try:
            existing_count = self.collection.count()
            if existing_count >= len(REFERENCE_SNIPPETS):
                logger.info(
                    f"Knowledge base already initialized with {existing_count} documents"
                )
                return
        except Exception as e:
            logger.warning(f"Could not check existing documents: {e}")

        logger.info("Initializing ADGM knowledge base in ChromaDB...")

        # Prepare documents
        documents = []
        metadatas = []
        ids = []

        for idx, ref in enumerate(
            tqdm(REFERENCE_SNIPPETS, desc="Processing documents")
        ):
            doc_id = f"adgm_doc_{idx}_{str(uuid.uuid4())[:8]}"
            documents.append(ref["text"])
            metadatas.append(
                {
                    "title": ref["title"],
                    "url": ref["url"],
                    "category": ref["category"],
                    "doc_index": idx,
                }
            )
            ids.append(doc_id)

        # Add to ChromaDB
        try:
            self.collection.add(documents=documents, metadatas=metadatas, ids=ids)
            logger.info(f"Successfully added {len(documents)} documents to ChromaDB")
        except Exception as e:
            logger.error(f"Failed to add documents to ChromaDB: {e}")
            raise

    def retrieve_relevant_snippets(
        self, query: str, top_k: int = 3, category_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant documents using ChromaDB cosine similarity"""
        try:
            # Prepare where clause for category filtering
            where_clause = {"category": category_filter} if category_filter else None

            # Perform similarity search
            results = self.collection.query(
                query_texts=[query], n_results=top_k, where=where_clause
            )

            # Format results
            relevant_snippets = []
            if results and results["documents"] and results["documents"][0]:
                for i, doc_text in enumerate(results["documents"][0]):
                    metadata = (
                        results["metadatas"][0][i] if results["metadatas"] else {}
                    )
                    distance = (
                        results["distances"][0][i] if results["distances"] else 0.0
                    )

                    # Convert distance to similarity score
                    similarity_score = 1.0 - distance

                    snippet = {
                        "title": metadata.get("title", "Unknown"),
                        "url": metadata.get("url", ""),
                        "category": metadata.get("category", "General"),
                        "text": doc_text,
                        "similarity_score": similarity_score,
                        "distance": distance,
                    }
                    relevant_snippets.append(snippet)

            return relevant_snippets

        except Exception as e:
            logger.error(f"Error in ChromaDB retrieval: {e}")
            return []

    def enhanced_gemini_analysis(
        self,
        prompt: str,
        rag_snippets: Optional[List[Dict]] = None,
        temperature: float = 0.1,
    ) -> Dict[str, Any]:
        """Enhanced Gemini analysis with ChromaDB context"""
        try:
            genai.configure(api_key=GEMINI_API_KEY)
            model = genai.GenerativeModel(MODEL_NAME)

            if rag_snippets:
                # Build context from ChromaDB results
                context_parts = []
                for snippet in rag_snippets:
                    similarity = snippet.get("similarity_score", 0.0)
                    context_parts.append(
                        f"Source: {snippet['title']} (Relevance: {similarity:.3f})\n"
                        f"URL: {snippet['url']}\n"
                        f"Content: {snippet['text']}\n"
                    )

                context = "\n" + "=" * 50 + "\n".join(context_parts)

                enhanced_prompt = f"""
                You are an expert ADGM legal compliance analyst. Use the following official ADGM sources:
                
                OFFICIAL ADGM SOURCES (Retrieved via ChromaDB Semantic Search):
                {context}
                
                DOCUMENT CONTENT TO ANALYZE:
                {prompt}
                
                ANALYSIS REQUIREMENTS:
                1. Check jurisdiction compliance (must be ADGM, not UAE Federal/Dubai/Abu Dhabi courts)
                2. Verify required clauses are present and properly worded
                3. Ensure formatting meets ADGM standards
                4. Confirm compliance with current ADGM regulations
                5. Identify ambiguous or non-binding language
                6. Check for outdated legal references
                
                Respond in this exact JSON format:
                {{
                    "red_flag": "Detailed description of compliance issue found, or null if compliant",
                    "law_citation": "Exact ADGM regulation citation (e.g., 'ADGM Companies Regulations 2020, Article X')",
                    "suggestion": "Specific compliant alternative wording or action required",
                    "severity": "High/Medium/Low based on legal and business impact",
                    "category": "jurisdiction/missing_clauses/formatting/compliance/ambiguity/outdated",
                    "confidence": "High/Medium/Low based on analysis certainty",
                    "compliant_clause": "Suggested replacement clause text if applicable"
                }}
                
                Be precise with citations and reference the exact ADGM regulation articles.
                """
            else:
                enhanced_prompt = prompt

            response = model.generate_content(
                enhanced_prompt,
                generation_config={
                    "temperature": temperature,
                    "max_output_tokens": 1024,
                    "top_p": 0.8,
                    "top_k": 40,
                },
            )

            result = response.text.strip()

            # Parse JSON response
            try:
                result = result.replace("```json", "").replace("```", "").strip()
                parsed_result = json.loads(result)

                # Add metadata
                parsed_result["retrieval_method"] = "ChromaDB_cosine_similarity"
                parsed_result["rag_sources_count"] = (
                    len(rag_snippets) if rag_snippets else 0
                )

                return parsed_result

            except json.JSONDecodeError:
                return {
                    "red_flag": result if result else "Analysis completed",
                    "law_citation": "ADGM Companies Regulations 2020",
                    "suggestion": "Review with qualified ADGM legal counsel",
                    "severity": "Medium",
                    "category": "compliance",
                    "confidence": "Low",
                    "retrieval_method": "ChromaDB_cosine_similarity",
                    "rag_sources_count": len(rag_snippets) if rag_snippets else 0,
                }

        except Exception as e:
            logger.error(f"Gemini analysis error: {e}")
            return {
                "red_flag": f"Analysis error: {str(e)}",
                "law_citation": "ADGM Legal Framework - General Requirements",
                "suggestion": "Manual legal review required due to analysis error",
                "severity": "Medium",
                "category": "compliance",
                "confidence": "Low",
                "retrieval_method": "ChromaDB_cosine_similarity",
                "error": str(e),
            }
