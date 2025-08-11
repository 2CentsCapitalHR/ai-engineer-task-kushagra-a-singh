import asyncio
import datetime
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin

import aiohttp
import google.generativeai as genai
import requests
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

from error_handler import ErrorHandler, GracefulDegradation, InputValidator

# Import the new modules
from rate_limiter import get_api_client, get_rate_limiter
from smart_batcher import BatchConfig, Section, SmartBatcher

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load Gemini API key
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
MODEL_NAME = os.getenv("GOOGLE_MODEL", "gemini-2.0-flash")

# Pre-load embedder with error handling
try:
    EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")
except Exception as e:
    logger.error(f"Failed to load sentence transformer: {e}")
    EMBEDDER = None

# Enhanced ADGM Reference Data Sources with all required sources from task
REFERENCE_SNIPPETS = [
    {
        "title": "ADGM Company Incorporation Checklist (Private Company Limited)",
        "url": "https://www.adgm.com/documents/registration-authority/registration-and-incorporation/checklist/private-company-limited-by-guarantee-non-financial-services-20231228.pdf",
        "category": "Company Formation",
        "text": "Required documents for private company incorporation in ADGM include: Articles of Association (must specify ADGM jurisdiction), Memorandum of Association (must state registered office in ADGM), Board Resolution for incorporation (properly dated and signed), Shareholder Resolution for incorporation, Application Form (completely filled), UBO Declaration Form (all beneficial owners >25%), Register of Members and Directors (current and accurate), Change of Registered Address Notice (if applicable), ID documents for all directors and authorized signatories, Source of Wealth Declaration, Hub71 Approval Letter (for technology companies), Lease Agreement for registered office, Audited Annual Accounts (if existing company), Auditors Report, Director's Report, Board Resolution approving accounts. All documents must reference ADGM Courts jurisdiction, not UAE Federal Courts.",
    },
    {
        "title": "ADGM Company Incorporation Checklist (Branch - Non-Financial Services)",
        "url": "https://www.adgm.com/documents/registration-authority/registration-and-incorporation/checklist/branch-non-financial-services-20231228.pdf",
        "category": "Company Formation",
        "text": "Branch company setup requires: Parent company incorporation documents, Power of Attorney for local representative, Board Resolution from parent company authorizing branch establishment, Financial statements of parent company, Business plan for ADGM operations, Local representative appointment documents, Registered office lease agreement in ADGM. All branch operations must comply with ADGM regulations and maintain proper corporate governance.",
    },
    {
        "title": "ADGM Registration and Incorporation",
        "url": "https://www.adgm.com/registration-authority/registration-and-incorporation",
        "category": "Company Formation",
        "text": "Complete guidance for ADGM registration including forms, templates, and requirements. All constitutional documents must specify ADGM as governing jurisdiction. Companies must maintain registered office in ADGM, appoint local directors, and file annual returns. Templates available for various entity types including LTD, LLC, partnerships, and special purpose vehicles. All resolutions must follow ADGM Companies Regulations 2020 requirements.",
    },
    {
        "title": "Resolution for Incorporation (LTD - Multiple Shareholders)",
        "url": "https://assets.adgm.com/download/assets/adgm-ra-resolution-multiple-incorporate-shareholders-LTD-incorporation-v2.docx/186a12846c3911efa4e6c6223862cd87",
        "category": "Company Formation",
        "text": "Official template for shareholder resolution when incorporating LTD company with multiple shareholders. Must include proper recitals, authorization for incorporation, appointment of directors, adoption of Articles of Association, and proper signatures from all shareholders. Resolution must be dated and include shareholder percentage ownership details.",
    },
    {
        "title": "ADGM Setting Up Guide",
        "url": "https://www.adgm.com/setting-up",
        "category": "Company Formation",
        "text": "Comprehensive setup guide covering: business nature selection, legal structure determination, name reservation, registered office requirements, documentation checklist, licensing requirements, and compliance obligations. All entities must secure office space in ADGM before registration and maintain physical presence. Guidance includes SPV setup, incorporation packages, and post-registration services.",
    },
    {
        "title": "ADGM Guidance, Templates and Policy Statements",
        "url": "https://www.adgm.com/legal-framework/guidance-and-policy-statements",
        "category": "Policy & Guidance",
        "text": "Comprehensive guidance covering: applicant guidance, whistleblowing policies, ESG frameworks, KYC/AML requirements, IT/Cybersecurity policies, listing applications, resolution templates, registry requirements. All policies must align with ADGM legal framework and be regularly updated. Templates ensure compliance with current regulations.",
    },
    {
        "title": "ADGM Annual Accounts & Filings",
        "url": "https://www.adgm.com/operating-in-adgm/obligations-of-adgm-registered-entities/annual-filings/annual-accounts",
        "category": "Compliance & Filings",
        "text": "Annual filing requirements: audited accounts (unless small company exemption applies), auditors report, director's report, board resolution approving accounts. All accounts must be in USD, prepared according to IFRS, and filed within specified deadlines. Late filing incurs penalties. Small companies may file simplified accounts if meeting size criteria.",
    },
    {
        "title": "ADGM Letters & Permits Application",
        "url": "https://www.adgm.com/operating-in-adgm/post-registration-services/letters-and-permits",
        "category": "Letters/Permits",
        "text": "Process for obtaining official letters, NOCs, and permits for events, training, seminars in ADGM. Applications must include event details, participant information, venue confirmation, and compliance certificates. All permits subject to ADGM approval and may include conditions.",
    },
    {
        "title": "ADGM Company Incorporation Package Rulebook",
        "url": "https://en.adgm.thomsonreuters.com/rulebook/7-company-incorporation-package",
        "category": "Regulatory Template",
        "text": "Official rulebook detailing incorporation procedures, required templates, public register requirements, compliance obligations, and legal framework references. All incorporation must follow prescribed templates and procedures. Includes guidance on accounts preparation, filing requirements, and ongoing compliance.",
    },
    {
        "title": "ADGM Standard Employment Contract Template (2024)",
        "url": "https://assets.adgm.com/download/assets/ADGM+Standard+Employment+Contract+Template+-+ER+2024+(Feb+2025).docx/ee14b252edbe11efa63b12b3a30e5e3a",
        "category": "Employment & HR",
        "text": "Updated 2024 employment contract template including: employment terms, probation periods, notice requirements, termination clauses, end of service benefits, and ADGM employment regulation compliance. All contracts must be in English, properly executed, and comply with ADGM Employment Regulations. Must specify ADGM jurisdiction for disputes.",
    },
    {
        "title": "ADGM Standard Employment Contract (2019 Short Version)",
        "url": "https://assets.adgm.com/download/assets/ADGM+Standard+Employment+Contract+-+ER+2019+-+Short+Version+(May+2024).docx/33b57a92ecfe11ef97a536cc36767ef8",
        "category": "Employment & HR",
        "text": "Simplified employment contract template for basic employment arrangements. Includes essential terms: job description, salary, working hours, annual leave, notice periods, and termination provisions. Must comply with minimum ADGM employment standards and specify dispute resolution through ADGM courts.",
    },
    {
        "title": "ADGM Data Protection Policy Template",
        "url": "https://www.adgm.com/documents/office-of-data-protection/templates/adgm-dpr-2021-appropriate-policy-document.pdf",
        "category": "Data Protection",
        "text": "Template for appropriate policy document under ADGM Data Protection Regulations 2021. Must include: data processing procedures, lawful basis for processing, data subject rights, breach notification procedures, international transfers, and retention policies. All entities processing personal data must have compliant policy and data protection officer if required.",
    },
    {
        "title": "Shareholder Resolution Template - Amendment of Articles",
        "url": "https://assets.adgm.com/download/assets/Templates_SHReso_AmendmentArticles-v1-20220107.docx/97120d7c5af911efae4b1e183375c0b2?forcedownload=1",
        "category": "Regulatory Template",
        "text": "Official template for shareholder resolutions to amend Articles of Association. Must include: proper recitals, specific amendments proposed, shareholder approval percentages, effective date, and proper execution by authorized shareholders. All amendments must comply with ADGM Companies Regulations 2020 and company's existing articles.",
    },
    {
        "title": "ADGM Companies Regulations 2020",
        "url": "https://en.adgm.thomsonreuters.com/rulebook/1-companies-regulations-2020",
        "category": "Legal Framework",
        "text": "Core regulations governing ADGM companies covering: incorporation procedures (Articles 1-15), constitutional documents requirements (Articles 16-25), directors duties and appointment (Articles 26-35), shareholders rights and meetings (Articles 36-45), compliance and penalties (Articles 46-50). All companies must strictly comply with these regulations. Violations result in administrative penalties and potential dissolution.",
    },
]

# Enhanced Legal Red Flag Categories with specific ADGM context
RED_FLAG_CATEGORIES = {
    "jurisdiction": {
        "description": "Jurisdiction and governing law compliance",
        "examples": [
            "UAE Federal Courts instead of ADGM Courts",
            "Dubai or Abu Dhabi jurisdiction references",
            "Missing ADGM governing law clause",
            "Incorrect dispute resolution procedures",
        ],
    },
    "missing_clauses": {
        "description": "Required clauses and provisions missing",
        "examples": [
            "Missing registered office clause",
            "Absent shareholder rights provisions",
            "No director appointment procedures",
            "Missing compliance obligations",
        ],
    },
    "formatting": {
        "description": "Document structure and formatting issues",
        "examples": [
            "Non-standard document structure",
            "Missing signature sections",
            "Incorrect numbering or references",
            "Template non-compliance",
        ],
    },
    "compliance": {
        "description": "ADGM regulatory compliance issues",
        "examples": [
            "Non-compliance with ADGM Companies Regulations 2020",
            "Missing required disclosures",
            "Incorrect filing procedures",
            "Regulatory deadline violations",
        ],
    },
    "ambiguity": {
        "description": "Ambiguous or unclear language",
        "examples": [
            "Vague terms and conditions",
            "Non-binding language",
            "Unclear obligations or rights",
            "Ambiguous definitions",
        ],
    },
    "outdated": {
        "description": "Outdated references and requirements",
        "examples": [
            "References to superseded regulations",
            "Old template versions",
            "Outdated compliance requirements",
        ],
    },
}

# Import configuration
from config import RAG_TOP_K, USE_CHROMADB, get_rag_config

# Initialize ChromaDB RAG if enabled
_chromadb_rag = None


def _get_chromadb_rag():
    """Get or initialize ChromaDB RAG instance"""
    global _chromadb_rag
    if USE_CHROMADB and _chromadb_rag is None:
        try:
            from adgm_rag_chromadb import ADGMChromaRAG

            _chromadb_rag = ADGMChromaRAG()
            logger.info("ChromaDB RAG initialized successfully")
        except ImportError:
            logger.error(
                "ChromaDB requested but adgm_rag_chromadb module not available"
            )
            return None
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB RAG: {e}")
            return None
    return _chromadb_rag


# Pre-compute embeddings for enhanced RAG performance (in-memory only)
def initialize_embeddings():
    """Initialize embeddings for all reference snippets (in-memory mode only)"""
    if USE_CHROMADB:
        logger.info(
            "ChromaDB mode enabled - skipping in-memory embedding initialization"
        )
        return

    if EMBEDDER is None:
        logger.warning("Embedder not available, skipping embedding initialization")
        return

    logger.info("Initializing embeddings for ADGM reference data (in-memory mode)...")
    for ref in tqdm(REFERENCE_SNIPPETS, desc="Computing embeddings"):
        try:
            ref["embedding"] = EMBEDDER.encode(ref["text"])
        except Exception as e:
            logger.error(f"Failed to compute embedding for {ref['title']}: {e}")


# Initialize embeddings on module load (only for in-memory mode)
if not USE_CHROMADB:
    initialize_embeddings()


def retrieve_relevant_snippets(
    query: str, top_k: int = None, category_filter: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Unified retrieval function that automatically chooses between in-memory and ChromaDB
    based on configuration
    """
    if top_k is None:
        top_k = RAG_TOP_K

    if USE_CHROMADB:
        # Use ChromaDB backend
        chromadb_rag = _get_chromadb_rag()
        if chromadb_rag:
            logger.info(f"Using ChromaDB retrieval for query: {query[:50]}...")
            return chromadb_rag.retrieve_relevant_snippets(
                query, top_k, category_filter
            )
        else:
            logger.warning("ChromaDB not available, falling back to in-memory")

    # Use in-memory backend (original implementation)
    logger.info(f"Using in-memory retrieval for query: {query[:50]}...")
    return _retrieve_relevant_snippets_inmemory(query, top_k, category_filter)


def _retrieve_relevant_snippets_inmemory(
    query: str, top_k: int = 3, category_filter: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Enhanced in-memory RAG retrieval with category filtering and better context matching"""
    if EMBEDDER is None:
        logger.warning("Embedder not available, returning all snippets")
        return REFERENCE_SNIPPETS[:top_k]

    try:
        query_emb = EMBEDDER.encode(query)
        candidates = REFERENCE_SNIPPETS

        # Apply category filter if specified
        if category_filter:
            candidates = [
                ref for ref in candidates if ref.get("category") == category_filter
            ]

        scored = []
        for ref in candidates:
            if "embedding" in ref:
                score = util.dot_score(query_emb, ref["embedding"]).item()
                scored.append((score, ref))

        scored.sort(reverse=True, key=lambda x: x[0])
        results = []
        for score, ref in scored[:top_k]:
            # Format result to match ChromaDB output format
            result = {
                "title": ref["title"],
                "url": ref["url"],
                "category": ref["category"],
                "text": ref["text"],
                "similarity_score": score,  # Add for consistency with ChromaDB
                "distance": 1.0 - score,  # Add for consistency with ChromaDB
            }
            results.append(result)

        return results

    except Exception as e:
        logger.error(f"Error in in-memory RAG retrieval: {e}")
        return REFERENCE_SNIPPETS[:top_k]


def get_adgm_citation_for_issue(issue_category: str, doc_type: str) -> str:
    """Get specific ADGM citation based on issue category and document type"""
    citations = {
        "jurisdiction": {
            "default": "ADGM Companies Regulations 2020, Article 6 - Constitutional Documents",
            "Articles of Association (AoA)": "ADGM Companies Regulations 2020, Article 18 - Articles of Association",
            "Memorandum of Association (MoA/MoU)": "ADGM Companies Regulations 2020, Article 16 - Memorandum of Association",
            "Employment Contract": "ADGM Employment Regulations 2019, Article 15 - Dispute Resolution",
        },
        "missing_clauses": {
            "default": "ADGM Companies Regulations 2020, Article 8 - Required Provisions",
            "Articles of Association (AoA)": "ADGM Companies Regulations 2020, Article 20 - Mandatory Articles Provisions",
            "Board Resolution": "ADGM Companies Regulations 2020, Article 30 - Board Resolutions Requirements",
        },
        "compliance": {
            "default": "ADGM Companies Regulations 2020, Article 47 - Compliance Requirements",
            "UBO Declaration": "ADGM Companies Regulations 2020, Article 12 - Ultimate Beneficial Ownership",
            "Employment Contract": "ADGM Employment Regulations 2019, Article 8 - Contract Requirements",
        },
    }

    category_citations = citations.get(issue_category, {})
    return category_citations.get(
        doc_type, category_citations.get("default", "ADGM Companies Regulations 2020")
    )


async def enhanced_gemini_analysis(
    prompt: str, rag_snippets: Optional[List[Dict]] = None, temperature: float = 0.1
) -> Dict[str, Any]:
    """Enhanced async legal analysis with rate limiting, error handling, and smart batching"""

    # Get the rate-limited API client
    api_client = get_api_client()

    try:
        # Validate input
        if not prompt or not isinstance(prompt, str):
            return ErrorHandler.handle_api_error(
                ValueError("Invalid prompt: must be non-empty string"),
                "enhanced_gemini_analysis",
            )

        # Check if we should use smart batching for multiple sections
        if ("--- SECTION" in prompt and prompt.count("--- SECTION") > 1) or (
            "Section 1 (" in prompt and "Section 2 (" in prompt
        ):
            # This is a batched request, use specialized handling
            return await _handle_batched_analysis(prompt, rag_snippets, temperature)

        # Standard single analysis
        return await _make_rate_limited_request(prompt, rag_snippets, temperature)

    except Exception as e:
        logger.error(f"Enhanced Gemini analysis error: {e}")
        return ErrorHandler.handle_api_error(e, "enhanced_gemini_analysis")


async def _make_rate_limited_request(
    prompt: str, rag_snippets: Optional[List[Dict]] = None, temperature: float = 0.1
) -> Dict[str, Any]:
    """Make a rate-limited API request with proper error handling"""

    api_client = get_api_client()

    async def _gemini_request():
        """Inner function to make the actual Gemini API call"""
        try:
            genai.configure(api_key=GEMINI_API_KEY)
            model = genai.GenerativeModel(MODEL_NAME)

            if rag_snippets:
                context = "\n\n".join(
                    [
                        f"Source: {s['title']} ({s['url']})\nContent: {s['text']}"
                        for s in rag_snippets
                    ]
                )
                enhanced_prompt = f"""
            You are an expert ADGM legal compliance analyst. Use the following official ADGM sources to analyze the document content.

            OFFICIAL ADGM SOURCES:
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

            CRITICAL: You MUST respond with ONLY valid JSON in this exact format. Do not include any text before or after the JSON.
            
            {{
                "red_flag": "Detailed description of compliance issue found, or null if compliant",
                "law_citation": "Exact ADGM regulation citation (e.g., 'ADGM Companies Regulations 2020, Article X')",
                "suggestion": "Specific compliant alternative wording or action required",
                "severity": "High/Medium/Low based on legal and business impact",
                "category": "jurisdiction/missing_clauses/formatting/compliance/ambiguity/outdated",
                "confidence": "High/Medium/Low based on analysis certainty",
                "compliant_clause": "Suggested replacement clause text if applicable"
            }}
            
            IMPORTANT:
            - Return ONLY the JSON object, no additional text
            - Ensure all quotes are properly escaped
            - Use double quotes for all strings
            - Do not include markdown formatting
            - Be precise with citations and reference the exact ADGM regulation articles
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
                    "response_mime_type": "application/json",
                },
            )

            # Ensure we return a string, not a response object
            if hasattr(response, "text"):
                response_text = response.text.strip()
                # Check if response is empty or just whitespace
                if not response_text or response_text.isspace():
                    logger.warning("API returned empty response, attempting retry...")
                    raise ValueError("Empty response from API")
                return response_text
            else:
                response_text = str(response)
                if not response_text or response_text.isspace():
                    logger.warning(
                        "API returned empty response string, attempting retry..."
                    )
                    raise ValueError("Empty response string from API")
                return response_text

        except Exception as e:
            raise e

    # Use the rate-limited API client
    result = await api_client.make_request(_gemini_request)

    # Check if the request failed
    if isinstance(result, dict) and "error" in result:
        logger.error(f"API request failed: {result['error']}")
        return ErrorHandler.handle_api_error(
            Exception(result["error"]), "enhanced_gemini_analysis"
        )

    # Ensure result is a string before parsing
    if not isinstance(result, str):
        logger.warning(f"API response is not a string: {type(result)}, converting...")
        result = str(result)

    # Parse the successful response
    return _parse_gemini_response(result)


async def _handle_batched_analysis(
    prompt: str, rag_snippets: Optional[List[Dict]] = None, temperature: float = 0.1
) -> Dict[str, Any]:
    """Handle batched analysis requests with specialized parsing"""

    try:
        # Use the rate-limited API client for batched requests
        api_client = get_api_client()

        async def _batched_gemini_request():
            """Inner function for batched Gemini API call"""
            try:
                genai.configure(api_key=GEMINI_API_KEY)
                model = genai.GenerativeModel(MODEL_NAME)

                response = model.generate_content(
                    prompt,
                    generation_config={
                        "temperature": temperature,
                        "max_output_tokens": 2048,  # Higher for batched requests
                        "top_p": 0.8,
                        "top_k": 40,
                        "response_mime_type": "application/json",
                    },
                )

                # Ensure we return a string, not a response object
                if hasattr(response, "text"):
                    response_text = response.text.strip()
                    # Check if response is empty or just whitespace
                    if not response_text or response_text.isspace():
                        logger.warning(
                            "Batched API returned empty response, attempting retry..."
                        )
                        raise ValueError("Empty response from batched API")
                    return response_text
                else:
                    response_text = str(response)
                    if not response_text or response_text.isspace():
                        logger.warning(
                            "Batched API returned empty response string, attempting retry..."
                        )
                        raise ValueError("Empty response string from batched API")
                    return response_text

            except Exception as e:
                raise e

        # Make the rate-limited request
        result = await api_client.make_request(_batched_gemini_request)

        if isinstance(result, dict) and "error" in result:
            logger.error(f"Batched API request failed: {result['error']}")
            return ErrorHandler.handle_api_error(
                Exception(result["error"]), "batched_analysis"
            )

        # Ensure result is a string before parsing
        if not isinstance(result, str):
            logger.warning(
                f"Batched API response is not a string: {type(result)}, converting..."
            )
            result = str(result)

        # Parse batched response
        return _parse_batched_response(result)

    except Exception as e:
        logger.error(f"Batched analysis error: {e}")
        return ErrorHandler.handle_api_error(e, "batched_analysis")


def _parse_gemini_response(response_text: str) -> Dict[str, Any]:
    """Parse Gemini API response with enhanced error handling and multiple fallback strategies"""

    try:
        # Enhanced response preprocessing
        cleaned_response = _preprocess_gemini_response(response_text)

        if not cleaned_response:
            logger.warning("Response preprocessing failed, using fallback")
            return _create_fallback_response("Response preprocessing failed")

        # Try multiple parsing strategies
        parsed_result = _parse_with_multiple_strategies(cleaned_response)

        if parsed_result is None:
            logger.warning("All parsing strategies failed, using fallback")
            return _create_fallback_response("All parsing strategies failed")

        # Handle case where response is a list instead of dict
        if isinstance(parsed_result, list):
            logger.warning("Response is a list, converting to single result format")
            # Take the first item if it's a list, or create a fallback
            if parsed_result and isinstance(parsed_result[0], dict):
                parsed_result = parsed_result[0]
            else:
                parsed_result = {}

        # Validate and enhance required fields
        parsed_result = _validate_and_enhance_fields(parsed_result)

        return parsed_result

    except Exception as e:
        logger.error(f"Unexpected error parsing response: {e}")
        return _create_fallback_response(f"Unexpected parsing error: {str(e)}")


def _parse_batched_response(response_text: str) -> Dict[str, Any]:
    """Parse batched Gemini API response with enhanced error handling and multiple strategies"""

    try:
        # Enhanced response preprocessing
        cleaned_response = _preprocess_gemini_response(response_text)

        if not cleaned_response:
            logger.warning("Batched response preprocessing failed, using fallback")
            return _create_batched_fallback_response("Response preprocessing failed")

        # Try multiple parsing strategies
        parsed_result = _parse_with_multiple_strategies(cleaned_response)

        if parsed_result is None:
            logger.warning("All batched parsing strategies failed, using fallback")
            return _create_batched_fallback_response("All parsing strategies failed")

        # Handle case where response is a list instead of dict
        if isinstance(parsed_result, list):
            logger.warning("Batched response is a list, converting to dict format")
            # Convert list to dict format for consistency
            converted_result = {}
            for i, item in enumerate(parsed_result):
                if isinstance(item, dict):
                    converted_result[f"section_{i+1}"] = item
                else:
                    converted_result[f"section_{i+1}"] = {
                        "error": f"Invalid item type: {type(item)}"
                    }
            parsed_result = converted_result

        # For batched responses, return the parsed result as-is
        # The calling function will handle the individual section parsing
        # If the parsed result is a list of section dicts, convert to keyed dict
        if isinstance(parsed_result, list):
            keyed = {}
            for i, item in enumerate(parsed_result):
                if isinstance(item, dict):
                    keyed[f"section_{i+1}"] = item
            parsed_result = keyed if keyed else parsed_result

        return {
            "batched_response": True,
            "parsed_content": parsed_result,
            "raw_response": response_text,
            "parse_success": True,
        }

    except Exception as e:
        logger.error(f"Unexpected error parsing batched response: {e}")
        return _create_batched_fallback_response(f"Unexpected parsing error: {str(e)}")


def _preprocess_gemini_response(response_text: str) -> Optional[str]:
    """Preprocess Gemini response with multiple cleaning strategies"""

    try:
        if not response_text or not isinstance(response_text, str):
            return None

        # Clean response text with multiple strategies
        cleaned = response_text.replace("```json", "").replace("```", "").strip()

        # Remove common prefixes/suffixes that might cause parsing issues
        prefixes_to_remove = [
            "Here's the analysis:",
            "Analysis results:",
            "JSON response:",
            "The analysis shows:",
            "Based on the document:",
            "Here is the compliance analysis:",
            "The document analysis reveals:",
        ]

        for prefix in prefixes_to_remove:
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix) :].strip()

        # Remove trailing punctuation that might break JSON
        while cleaned and cleaned[-1] in [".", ",", ";", ":", "!", "?"]:
            cleaned = cleaned[:-1].strip()

        # Fix common JSON formatting issues
        cleaned = _fix_gemini_json_issues(cleaned)

        return cleaned if cleaned else None

    except Exception as e:
        logger.warning(f"Error preprocessing Gemini response: {e}")
        return None


def _fix_gemini_json_issues(text: str) -> str:
    """Fix common JSON formatting issues in Gemini responses"""

    try:
        # Fix unterminated strings by finding and closing them
        lines = text.split("\n")
        fixed_lines = []

        for line in lines:
            # Count quotes in the line
            quote_count = line.count('"')
            if quote_count % 2 != 0:  # Odd number of quotes
                # Try to fix by adding a closing quote
                if line.strip().endswith(","):
                    line = line.rstrip(",") + '",'
                elif line.strip().endswith(":"):
                    line = line + '""'
                else:
                    line = line + '"'

            fixed_lines.append(line)

        fixed_text = "\n".join(fixed_lines)

        # Fix missing commas between objects
        fixed_text = fixed_text.replace("}\n{", "},\n{")
        fixed_text = fixed_text.replace("}\n  {", "},\n  {")

        # Fix trailing commas in objects
        fixed_text = fixed_text.replace(",}", "}")
        fixed_text = fixed_text.replace(",\n}", "\n}")

        # Fix common Gemini response issues
        fixed_text = fixed_text.replace('"red_flag": "', '"red_flag": "')
        fixed_text = fixed_text.replace('"law_citation": "', '"law_citation": "')
        fixed_text = fixed_text.replace('"suggestion": "', '"suggestion": "')
        fixed_text = fixed_text.replace('"severity": "', '"severity": "')
        fixed_text = fixed_text.replace('"category": "', '"category": "')

        return fixed_text

    except Exception as e:
        logger.warning(f"Error fixing Gemini JSON issues: {e}")
        return text


def _parse_with_multiple_strategies(text: str) -> Optional[Any]:
    """Parse JSON with multiple fallback strategies"""

    strategies = [
        _try_direct_json_parse,
        _try_fix_and_parse_json,
        _try_extract_json_blocks,
        _try_manual_json_construction,
    ]

    for i, strategy in enumerate(strategies):
        try:
            result = strategy(text)
            if result is not None:
                logger.info(f"JSON parsing succeeded with strategy {i+1}")
                return result
        except Exception as e:
            logger.debug(f"Strategy {i+1} failed: {e}")
            continue

    return None


def _try_direct_json_parse(text: str) -> Optional[Any]:
    """Try direct JSON parsing"""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _try_fix_and_parse_json(text: str) -> Optional[Any]:
    """Try to fix common JSON issues and parse"""
    try:
        # Try to find the start and end of JSON content
        start_idx = text.find("{")
        end_idx = text.rfind("}")

        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            json_content = text[start_idx : end_idx + 1]
            return json.loads(json_content)

        return None
    except json.JSONDecodeError:
        return None


def _try_extract_json_blocks(text: str) -> Optional[Any]:
    """Try to extract and parse JSON blocks"""
    try:
        # Look for JSON-like structures
        import re

        # Find potential JSON objects
        json_pattern = r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}"
        matches = re.findall(json_pattern, text)

        if matches:
            # Try to parse the largest match
            largest_match = max(matches, key=len)
            return json.loads(largest_match)

        return None
    except (json.JSONDecodeError, Exception):
        return None


def _try_manual_json_construction(text: str) -> Optional[Any]:
    """Try to manually construct JSON from text content"""
    try:
        # This is a last resort - try to extract meaningful information
        lines = text.split("\n")
        result = {}

        current_section = None
        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Try to identify section headers
            if line.startswith("section_") or line.startswith("Section"):
                current_section = line.split(":")[0].strip().lower()
                if not current_section.startswith("section_"):
                    current_section = (
                        f"section_{current_section.replace('Section', '').strip()}"
                    )
                result[current_section] = {}
            elif current_section and ":" in line:
                key, value = line.split(":", 1)
                key = key.strip().lower()
                value = value.strip()

                # Map common fields
                if "red_flag" in key or "issue" in key:
                    result[current_section]["red_flag"] = value
                elif "law" in key or "citation" in key:
                    result[current_section]["law_citation"] = value
                elif "suggestion" in key or "recommendation" in key:
                    result[current_section]["suggestion"] = value
                elif "severity" in key:
                    result[current_section]["severity"] = value
                elif "category" in key:
                    result[current_section]["category"] = value

        # If we found any structured data, return it
        if result:
            return result

        return None
    except Exception:
        return None


def _validate_and_enhance_fields(result: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and enhance required fields with intelligent defaults"""

    # Required fields with intelligent defaults
    required_fields = {
        "red_flag": "No issues found",
        "law_citation": "ADGM Companies Regulations 2020",
        "suggestion": "Document appears compliant",
        "severity": "Low",
        "category": "compliance",
    }

    for field, default_value in required_fields.items():
        if field not in result or not result[field]:
            result[field] = default_value

    # Additional metadata
    result["confidence"] = result.get("confidence", "Medium")
    result["compliant_clause"] = result.get(
        "compliant_clause", "Refer to ADGM guidelines"
    )
    result["analysis_method"] = result.get("analysis_method", "enhanced_parsing")

    return result


def _create_fallback_response(error_message: str) -> Dict[str, Any]:
    """Create a fallback response when parsing fails"""
    return {
        "red_flag": f"Parsing error: {error_message}",
        "law_citation": "ADGM Companies Regulations 2020",
        "suggestion": "Review with qualified ADGM legal counsel",
        "severity": "Medium",
        "category": "compliance",
        "confidence": "Low",
        "compliant_clause": "Refer to official ADGM templates",
        "analysis_method": "fallback_parsing",
        "parse_error": error_message,
    }


def _create_batched_fallback_response(error_message: str) -> Dict[str, Any]:
    """Create a fallback response for batched parsing failures"""
    return {
        "batched_response": True,
        "parsed_content": {},
        "raw_response": "",
        "parse_success": False,
        "parse_error": error_message,
        "analysis_method": "fallback_batched_parsing",
    }


def gemini_legal_analysis(
    prompt: str, rag_snippets: Optional[List[Dict]] = None, temperature: float = 0.1
) -> Dict[str, Any]:
    """
    Unified Gemini analysis function that works with both in-memory and ChromaDB backends
    """
    if USE_CHROMADB:
        # Use ChromaDB-enhanced analysis
        chromadb_rag = _get_chromadb_rag()
        if chromadb_rag:
            logger.info("Using ChromaDB-enhanced Gemini analysis")
            return chromadb_rag.enhanced_gemini_analysis(
                prompt, rag_snippets, temperature
            )
        else:
            logger.warning("ChromaDB not available, falling back to standard analysis")

    # Use standard analysis (original implementation)
    logger.info("Using standard Gemini analysis")
    return asyncio.run(enhanced_gemini_analysis(prompt, rag_snippets, temperature))


def analyze_document_completeness(
    doc_types: List[str], process_type: str
) -> Dict[str, Any]:
    """Enhanced completeness analysis with priority weighting"""
    from adgm_checklists import get_document_priority, get_required_docs_for_process

    required_docs = get_required_docs_for_process(process_type)
    uploaded_set = set(doc_types)
    required_set = set(required_docs)

    missing_docs = required_set - uploaded_set
    extra_docs = uploaded_set - required_set
    present_docs = uploaded_set & required_set

    # Calculate weighted completeness based on document priority
    total_weight = 0
    achieved_weight = 0

    for doc in required_docs:
        priority = get_document_priority(process_type, doc)
        weight = {"Critical": 3, "Important": 2, "Optional": 1}.get(priority, 2)
        total_weight += weight
        if doc in uploaded_set:
            achieved_weight += weight

    weighted_completeness = (
        (achieved_weight / total_weight * 100) if total_weight > 0 else 0
    )

    return {
        "process_type": process_type,
        "uploaded_count": len(uploaded_set),
        "required_count": len(required_set),
        "missing_documents": list(missing_docs),
        "extra_documents": list(extra_docs),
        "present_documents": list(present_docs),
        "completeness_percentage": (
            (len(present_docs) / len(required_set)) * 100 if required_set else 0
        ),
        "weighted_completeness": weighted_completeness,
        "missing_critical": [
            doc
            for doc in missing_docs
            if get_document_priority(process_type, doc) == "Critical"
        ],
        "missing_important": [
            doc
            for doc in missing_docs
            if get_document_priority(process_type, doc) == "Important"
        ],
        "missing_optional": [
            doc
            for doc in missing_docs
            if get_document_priority(process_type, doc) == "Optional"
        ],
    }


async def batch_analysis(
    documents: List[Dict[str, Any]], batch_size: int = 5, max_concurrent: int = 3
) -> List[Dict[str, Any]]:
    """Enhanced batch analysis with smart batching, rate limiting, and error handling"""

    logger.info(f"Starting enhanced batch analysis for {len(documents)} documents")

    # Initialize smart batcher
    batcher = SmartBatcher()
    all_results: List[Dict[str, Any]] = []

    # Track progress and errors
    total_processed = 0
    total_errors = 0
    error_summary: List[Dict[str, Any]] = []

    try:
        for i, document in enumerate(documents):
            try:
                logger.info(
                    f"Processing document {i+1}/{len(documents)}: {document.get('filename', 'Unknown')}"
                )

                # Validate document data
                validation_errors = InputValidator.validate_document_data(document)
                validation_summary = ErrorHandler.handle_validation_errors(
                    validation_errors
                )
                if not validation_summary.get("can_proceed", True):
                    logger.error(
                        f"Document validation failed for {document.get('filename')}"
                    )
                    # Normalize validation errors into dictionaries for summary
                    for ve in validation_errors:
                        try:
                            error_summary.append(
                                {
                                    "error_type": "validation_error",
                                    "severity": getattr(ve, "severity", "error"),
                                    "category": "validation",
                                    "message": getattr(
                                        ve, "message", "Validation error"
                                    ),
                                    "field": getattr(ve, "field", "unknown"),
                                }
                            )
                        except Exception:
                            error_summary.append(
                                {
                                    "error_type": "validation_error",
                                    "severity": "error",
                                    "category": "validation",
                                }
                            )
                    # Skip this document
                    all_results.append(
                        {
                            "filename": document.get("filename", "Unknown"),
                            "doc_type": document.get("doc_type", "unknown"),
                            "total_sections": 0,
                            "processed_sections": 0,
                            "results": [],
                            "processing_status": "failed",
                            "error": "Validation failed",
                        }
                    )
                    continue

                # Extract sections with robust error handling
                sections = document.get("sections", [])
                if not sections or not isinstance(sections, list):
                    logger.warning(
                        f"No valid sections found in document {document.get('filename')} - sections: {type(sections)}"
                    )
                    # Add failed document to results
                    all_results.append(
                        {
                            "filename": document.get("filename", "Unknown"),
                            "doc_type": document.get("doc_type", "unknown"),
                            "total_sections": 0,
                            "processed_sections": 0,
                            "results": [],
                            "processing_status": "failed",
                            "error": "No valid sections found in document",
                        }
                    )
                    continue

                # Validate each section has required fields
                valid_sections: List[Dict[str, Any]] = []
                for j, section in enumerate(sections):
                    try:
                        if j >= len(sections):
                            logger.warning(
                                f"Section index {j} out of range for sections length {len(sections)}"
                            )
                            break

                        if isinstance(section, dict) and section.get("text"):
                            valid_sections.append(section)
                        else:
                            logger.warning(
                                f"Invalid section {j} in document {document.get('filename')}: {type(section)}"
                            )
                    except IndexError as index_error:
                        logger.warning(
                            f"Index error accessing section {j}: {index_error}"
                        )
                        break
                    except Exception as section_validation_error:
                        logger.warning(
                            f"Error validating section {j}: {section_validation_error}"
                        )
                        continue

                if not valid_sections:
                    logger.warning(
                        f"No valid sections found in document {document.get('filename')}"
                    )
                    # Add failed document to results
                    all_results.append(
                        {
                            "filename": document.get("filename", "Unknown"),
                            "doc_type": document.get("doc_type", "unknown"),
                            "total_sections": len(sections),
                            "processed_sections": 0,
                            "results": [],
                            "processing_status": "failed",
                            "error": "No sections with valid text content found",
                        }
                    )
                    continue

                # Create batches using smart batcher
                try:
                    batches = batcher.create_batches(valid_sections)
                    logger.info(
                        f"Created {len(batches)} batches from {len(valid_sections)} sections"
                    )

                    # Debug batch structure
                    for batch_idx, batch in enumerate(batches):
                        if isinstance(batch, list):
                            logger.debug(f"Batch {batch_idx+1}: {len(batch)} sections")
                            for section_idx, section in enumerate(batch):
                                if hasattr(section, "text"):
                                    logger.debug(
                                        f"  Section {section_idx}: {len(section.text)} chars"
                                    )
                                else:
                                    logger.debug(
                                        f"  Section {section_idx}: Invalid section object"
                                    )
                        else:
                            logger.warning(
                                f"Batch {batch_idx+1} is not a list: {type(batch)}"
                            )

                    if not batches:
                        logger.warning("No batches were created from valid sections")
                        # Create fallback single-section batches
                        batches = []
                        for k, section in enumerate(valid_sections):
                            try:
                                fallback_section = Section(
                                    text=section.get(
                                        "text", "Error processing section"
                                    ),
                                    clause=section.get("clause", f"Section_{k}"),
                                    index=k,
                                    section_type=section.get("section_type", "content"),
                                )
                                batches.append([fallback_section])
                            except Exception as fallback_section_error:
                                logger.warning(
                                    f"Error creating fallback section {k}: {fallback_section_error}"
                                )
                                continue
                        logger.info(
                            f"Created fallback batches: {len(batches)} single-section batches"
                        )
                except Exception as fallback_batch_error:
                    logger.error(
                        f"Error creating fallback batches: {fallback_batch_error}"
                    )
                    batches = []

                if not batches:
                    logger.warning(
                        f"No batches could be created for document {document.get('filename')}"
                    )
                    # Add failed document to results
                    all_results.append(
                        {
                            "filename": document.get("filename", "Unknown"),
                            "doc_type": document.get("doc_type", "unknown"),
                            "total_sections": len(valid_sections),
                            "processed_sections": 0,
                            "results": [],
                            "processing_status": "failed",
                            "error": "Failed to create batches from sections",
                        }
                    )
                    continue

                # Process each batch
                document_results: List[Dict[str, Any]] = []
                for batch_idx, batch in enumerate(batches):
                    try:
                        # Enhanced bounds checking for batch
                        if not batch or not isinstance(batch, list):
                            logger.warning(
                                f"Invalid batch {batch_idx+1}: {type(batch)}"
                            )
                            continue

                        # Additional safety check for batch length
                        if batch_idx >= len(batches):
                            logger.warning(
                                f"Batch index {batch_idx} out of range for batches length {len(batches)}"
                            )
                            break

                        # Ensure batch has sections
                        if len(batch) == 0:
                            logger.warning(f"Batch {batch_idx+1} is empty, skipping...")
                            continue

                        logger.info(
                            f"Processing batch {batch_idx+1}/{len(batches)} with {len(batch)} sections"
                        )

                        # Create batch prompt
                        try:
                            batch_prompt = batcher.create_batch_prompt(
                                batch, document.get("doc_type", "legal_document")
                            )
                        except Exception as prompt_error:
                            logger.error(f"Error creating batch prompt: {prompt_error}")
                            # Skip this batch
                            continue

                        # Analyze batch
                        try:
                            batch_result = await enhanced_gemini_analysis(batch_prompt)
                        except Exception as analysis_error:
                            logger.error(f"Error in batch analysis: {analysis_error}")
                            batch_result = {
                                "error": f"Analysis failed: {str(analysis_error)}"
                            }

                        # Log batch result structure for debugging
                        logger.debug(
                            f"Batch {batch_idx+1} result type: {type(batch_result)}, content: {str(batch_result)[:200]}..."
                        )

                        # Validate batch_result structure before processing
                        if not isinstance(batch_result, dict):
                            logger.warning(
                                f"Batch {batch_idx+1} result is not a dictionary: {type(batch_result)}"
                            )
                            batch_result = {
                                "error": f"Invalid result type: {type(batch_result)}"
                            }

                        # Check if batch analysis failed
                        if (
                            isinstance(batch_result, dict)
                            and (
                                "error" in batch_result
                                or (
                                    "red_flag" in batch_result
                                    and isinstance(batch_result["red_flag"], str)
                                    and "error" in batch_result["red_flag"]
                                )
                            )
                        ) or not isinstance(batch_result, dict):
                            logger.warning(
                                f"Batch {batch_idx+1} analysis failed, using fallback"
                            )

                            # Use graceful degradation for failed batches
                            fallback_results: List[Dict[str, Any]] = []
                            for section_idx, section in enumerate(batch):
                                try:
                                    if section_idx >= len(batch):
                                        logger.warning(
                                            f"Section index {section_idx} out of range for batch length {len(batch)}"
                                        )
                                        break

                                    if not section or not hasattr(section, "text"):
                                        logger.warning(
                                            f"Invalid section object at index {section_idx}"
                                        )
                                        continue

                                    fallback_analysis = (
                                        GracefulDegradation.basic_compliance_check(
                                            section.text,
                                            document.get("doc_type", "legal_document"),
                                        )
                                    )

                                    # Convert fallback to standard format
                                    for issue in fallback_analysis.get("issues", []):
                                        fallback_results.append(
                                            {
                                                **issue,
                                                "section_index": getattr(
                                                    section, "index", 0
                                                ),
                                                "section_clause": getattr(
                                                    section, "clause", "Unknown"
                                                ),
                                                "section_text": (
                                                    (
                                                        getattr(section, "text", "")[
                                                            :200
                                                        ]
                                                        + "..."
                                                    )
                                                    if len(getattr(section, "text", ""))
                                                    > 200
                                                    else getattr(section, "text", "")
                                                ),
                                                "section_type": getattr(
                                                    section, "section_type", "content"
                                                ),
                                                "analysis_method": "fallback",
                                            }
                                        )
                                except Exception as section_error:
                                    logger.warning(
                                        f"Error processing section in fallback: {section_error}"
                                    )
                                    # Create minimal fallback result
                                    fallback_results.append(
                                        {
                                            "red_flag": f"Section processing error: {str(section_error)}",
                                            "law_citation": "ADGM Legal Framework - General Requirements",
                                            "suggestion": "Manual legal review required due to processing error",
                                            "severity": "Medium",
                                            "category": "compliance",
                                            "confidence": "Low",
                                            "compliant_clause": "Consult ADGM legal advisor",
                                            "section_index": getattr(
                                                section, "index", 0
                                            ),
                                            "section_clause": getattr(
                                                section, "clause", "Unknown"
                                            ),
                                            "section_text": "Error processing section",
                                            "section_type": getattr(
                                                section, "section_type", "content"
                                            ),
                                            "analysis_method": "fallback_error",
                                        }
                                    )

                            document_results.extend(fallback_results)
                        else:
                            # Batch analysis succeeded, process the result
                            if batch_result.get("batched_response"):
                                # This was a batched response, parse individual sections
                                try:
                                    # Prefer already parsed content when available (dict expected)
                                    content_for_parsing = batch_result.get(
                                        "parsed_content"
                                    )

                                    if isinstance(content_for_parsing, dict):
                                        parsed_results = batcher.parse_batch_response(
                                            content_for_parsing, batch
                                        )
                                    else:
                                        # Fallback to raw response string if needed
                                        raw_resp = batch_result.get("raw_response", "")
                                        parsed_results = batcher.parse_batch_response(
                                            raw_resp, batch
                                        )

                                    # Validate parsed results structure
                                    if parsed_results and isinstance(
                                        parsed_results, list
                                    ):
                                        if len(parsed_results) == len(batch):
                                            document_results.extend(parsed_results)
                                            logger.info(
                                                f"Successfully parsed {len(parsed_results)} section results"
                                            )
                                        else:
                                            logger.warning(
                                                f"Parsed results count ({len(parsed_results)}) doesn't match batch size ({len(batch)})"
                                            )
                                            # Use fallback for missing results
                                            for idx in range(len(batch)):
                                                if idx < len(parsed_results):
                                                    document_results.append(
                                                        parsed_results[idx]
                                                    )
                                                else:
                                                    fallback_result = {
                                                        "red_flag": "Section analysis result missing",
                                                        "law_citation": "ADGM Legal Framework - General Requirements",
                                                        "suggestion": "Manual review required for missing analysis",
                                                        "severity": "Medium",
                                                        "category": "compliance",
                                                        "confidence": "Low",
                                                        "compliant_clause": "Consult ADGM legal advisor",
                                                        "section_index": getattr(
                                                            batch[idx], "index", idx
                                                        ),
                                                        "section_clause": getattr(
                                                            batch[idx],
                                                            "clause",
                                                            f"Section_{idx+1}",
                                                        ),
                                                        "section_text": (
                                                            getattr(
                                                                batch[idx], "text", ""
                                                            )[:200]
                                                            + "..."
                                                            if len(
                                                                getattr(
                                                                    batch[idx],
                                                                    "text",
                                                                    "",
                                                                )
                                                            )
                                                            > 200
                                                            else getattr(
                                                                batch[idx], "text", ""
                                                            )
                                                        ),
                                                        "section_type": getattr(
                                                            batch[idx],
                                                            "section_type",
                                                            "content",
                                                        ),
                                                        "analysis_method": "fallback_missing_result",
                                                    }
                                                    document_results.append(
                                                        fallback_result
                                                    )
                                    else:
                                        logger.warning(
                                            f"Invalid parsed results from batch: {type(parsed_results)}"
                                        )
                                        # Fallback to individual section processing
                                        for section in batch:
                                            try:
                                                section_result = {
                                                    "red_flag": "Batch parsing returned invalid results",
                                                    "law_citation": "ADGM Legal Framework - General Requirements",
                                                    "suggestion": "Manual legal review required due to parsing error",
                                                    "severity": "Medium",
                                                    "category": "compliance",
                                                    "confidence": "Low",
                                                    "compliant_clause": "Consult ADGM legal advisor",
                                                    "section_index": getattr(
                                                        section, "index", 0
                                                    ),
                                                    "section_clause": getattr(
                                                        section, "clause", "Unknown"
                                                    ),
                                                    "section_text": (
                                                        (
                                                            getattr(
                                                                section, "text", ""
                                                            )[:200]
                                                            + "..."
                                                        )
                                                        if len(
                                                            getattr(section, "text", "")
                                                        )
                                                        > 200
                                                        else getattr(
                                                            section, "text", ""
                                                        )
                                                    ),
                                                    "section_type": getattr(
                                                        section,
                                                        "section_type",
                                                        "content",
                                                    ),
                                                    "analysis_method": "batch_parse_fallback",
                                                }
                                                document_results.append(section_result)
                                            except Exception as section_error:
                                                logger.warning(
                                                    f"Error creating fallback section result: {section_error}"
                                                )
                                                continue
                                except Exception as parse_batch_error:
                                    logger.error(
                                        f"Critical error in parse_batch_response: {parse_batch_error}"
                                    )
                                    # Use comprehensive fallback for critical parsing errors
                                    for section_idx, section in enumerate(batch):
                                        try:
                                            if section_idx >= len(batch):
                                                logger.warning(
                                                    f"Section index {section_idx} out of range for batch length {len(batch)}"
                                                )
                                                break

                                            if not section or not hasattr(
                                                section, "text"
                                            ):
                                                logger.warning(
                                                    f"Invalid section object at index {section_idx}"
                                                )
                                                continue

                                            section_result = {
                                                "red_flag": f"Critical batch parsing error: {str(parse_batch_error)}",
                                                "law_citation": "ADGM Legal Framework - General Requirements",
                                                "suggestion": "Manual legal review required due to critical parsing error",
                                                "severity": "High",
                                                "category": "compliance",
                                                "confidence": "Low",
                                                "compliant_clause": "Consult ADGM legal advisor",
                                                "section_index": getattr(
                                                    section, "index", 0
                                                ),
                                                "section_clause": getattr(
                                                    section, "clause", "Unknown"
                                                ),
                                                "section_text": (
                                                    (
                                                        getattr(section, "text", "")[
                                                            :200
                                                        ]
                                                        + "..."
                                                    )
                                                    if len(getattr(section, "text", ""))
                                                    > 200
                                                    else getattr(section, "text", "")
                                                ),
                                                "section_type": getattr(
                                                    section, "section_type", "content"
                                                ),
                                                "analysis_method": "critical_batch_parse_error",
                                            }
                                            document_results.append(section_result)
                                        except Exception as section_error:
                                            logger.warning(
                                                f"Error creating critical fallback section result: {section_error}"
                                            )
                                            continue
                            else:
                                # Single section response, convert to batch format
                                for section_idx, section in enumerate(batch):
                                    try:
                                        if section_idx >= len(batch):
                                            logger.warning(
                                                f"Section index {section_idx} out of range for batch length {len(batch)}"
                                            )
                                            break

                                        section_result = {
                                            **batch_result,
                                            "section_index": getattr(
                                                section, "index", 0
                                            ),
                                            "section_clause": getattr(
                                                section, "clause", "Unknown"
                                            ),
                                            "section_text": (
                                                (
                                                    getattr(section, "text", "")[:200]
                                                    + "..."
                                                )
                                                if len(getattr(section, "text", ""))
                                                > 200
                                                else getattr(section, "text", "")
                                            ),
                                            "section_type": getattr(
                                                section, "section_type", "content"
                                            ),
                                            "analysis_method": "api",
                                        }
                                        document_results.append(section_result)
                                    except Exception as section_error:
                                        logger.warning(
                                            f"Error processing section in API result: {section_error}"
                                        )
                                        # Create minimal section result
                                        document_results.append(
                                            {
                                                **batch_result,
                                                "section_index": getattr(
                                                    section, "index", 0
                                                ),
                                                "section_clause": getattr(
                                                    section, "clause", "Unknown"
                                                ),
                                                "section_text": "Error processing section",
                                                "section_type": getattr(
                                                    section, "section_type", "content"
                                                ),
                                                "analysis_method": "api_error",
                                            }
                                        )

                        total_processed += len(batch)

                    except Exception as e:
                        logger.error(f"Error processing batch {batch_idx+1}: {e}")
                        total_errors += len(batch) if batch else 0

                        # Handle batch errors gracefully
                        try:
                            error_result = ErrorHandler.handle_api_error(
                                e, f"batch_{batch_idx+1}"
                            )
                            for section_idx, section in enumerate(batch):
                                try:
                                    if section_idx >= len(batch):
                                        logger.warning(
                                            f"Section index {section_idx} out of range for batch length {len(batch)}"
                                        )
                                        break

                                    section_result = {
                                        **error_result,
                                        "section_index": getattr(section, "index", 0),
                                        "section_clause": getattr(
                                            section, "clause", "Unknown"
                                        ),
                                        "section_text": (
                                            (getattr(section, "text", "")[:200] + "...")
                                            if len(getattr(section, "text", "")) > 200
                                            else getattr(section, "text", "")
                                        ),
                                        "section_type": getattr(
                                            section, "section_type", "content"
                                        ),
                                        "analysis_method": "error_fallback",
                                    }
                                    document_results.append(section_result)
                                except Exception as section_error:
                                    logger.warning(
                                        f"Error processing section in error fallback: {section_error}"
                                    )
                                    # Create minimal error result
                                    document_results.append(
                                        {
                                            **error_result,
                                            "section_index": getattr(
                                                section, "index", 0
                                            ),
                                            "section_clause": getattr(
                                                section, "clause", "Unknown"
                                            ),
                                            "section_text": "Error processing section",
                                            "section_type": getattr(
                                                section, "section_type", "content"
                                            ),
                                            "analysis_method": "error_fallback_error",
                                        }
                                    )
                        except Exception as error_handler_error:
                            logger.error(
                                f"Error in error handler: {error_handler_error}"
                            )
                            # Create minimal error result
                            for section_idx, section in enumerate(batch):
                                try:
                                    if section_idx >= len(batch):
                                        logger.warning(
                                            f"Section index {section_idx} out of range for batch length {len(batch)}"
                                        )
                                        break

                                    if not section or not hasattr(section, "text"):
                                        logger.warning(
                                            f"Invalid section object at index {section_idx}"
                                        )
                                        continue

                                    document_results.append(
                                        {
                                            "red_flag": f"Critical error: {str(e)}",
                                            "law_citation": "ADGM Legal Framework - General Requirements",
                                            "suggestion": "Manual legal review required due to critical error",
                                            "severity": "High",
                                            "category": "compliance",
                                            "confidence": "Low",
                                            "compliant_clause": "Consult ADGM legal advisor",
                                            "section_index": getattr(
                                                section, "index", 0
                                            ),
                                            "section_clause": getattr(
                                                section, "clause", "Unknown"
                                            ),
                                            "section_text": "Critical error processing section",
                                            "section_type": getattr(
                                                section, "section_type", "content"
                                            ),
                                            "analysis_method": "critical_error",
                                        }
                                    )
                                except Exception as critical_error:
                                    logger.error(
                                        f"Error creating critical error result: {critical_error}"
                                    )
                                    continue

                # Add document results to overall results
                if document_results:
                    all_results.append(
                        {
                            "filename": document.get("filename"),
                            "doc_type": document.get("doc_type"),
                            "total_sections": len(valid_sections),
                            "processed_sections": len(document_results),
                            "results": document_results,
                            "processing_status": "completed",
                        }
                    )
                else:
                    all_results.append(
                        {
                            "filename": document.get("filename"),
                            "doc_type": document.get("doc_type"),
                            "total_sections": len(valid_sections),
                            "processed_sections": 0,
                            "results": [],
                            "processing_status": "failed",
                            "error": "No sections could be processed",
                        }
                    )
            except Exception as doc_error:
                logger.error(f"Error processing document {i+1}: {doc_error}")
                # Safely get sections length
                try:
                    sections_count = (
                        len(document.get("sections", []))
                        if document.get("sections")
                        else 0
                    )
                except Exception:
                    sections_count = 0
                total_errors += sections_count
                error_summary.append(
                    {
                        "error_type": "document_processing_failure",
                        "severity": "High",
                        "message": f"Failed to process document: {str(doc_error)}",
                    }
                )
                # Add failed document to results
                all_results.append(
                    {
                        "filename": document.get("filename", "Unknown"),
                        "doc_type": document.get("doc_type", "unknown"),
                        "total_sections": sections_count,
                        "processed_sections": 0,
                        "results": [],
                        "processing_status": "failed",
                        "error": f"Document processing error: {str(doc_error)}",
                    }
                )

        # Create final summary
        final_summary = {
            "total_documents": len(documents),
            "successful_documents": len(
                [r for r in all_results if r["processing_status"] == "completed"]
            ),
            "failed_documents": len(
                [r for r in all_results if r["processing_status"] == "failed"]
            ),
            "total_sections_processed": total_processed,
            "total_errors": total_errors,
            "success_rate": (
                (total_processed / (total_processed + total_errors) * 100)
                if (total_processed + total_errors) > 0
                else 0
            ),
            "error_summary": GracefulDegradation.create_error_summary(error_summary),
            "processing_timestamp": datetime.datetime.now().isoformat(),
        }

        logger.info(
            f"Batch analysis completed. Success rate: {final_summary['success_rate']:.1f}%"
        )

        return {"summary": final_summary, "results": all_results}

    except Exception as e:
        logger.error(f"Critical error in batch analysis: {e}")
        # Safely calculate total sections
        try:
            total_sections = sum(
                len(doc.get("sections", [])) for doc in documents if doc.get("sections")
            )
        except Exception:
            total_sections = 0

        return {
            "summary": {
                "total_documents": len(documents),
                "successful_documents": 0,
                "failed_documents": len(documents),
                "total_sections_processed": 0,
                "total_errors": total_sections,
                "success_rate": 0.0,
                "error_summary": GracefulDegradation.create_error_summary(
                    [{"error_type": "critical_failure", "severity": "High"}]
                ),
                "processing_timestamp": datetime.datetime.now().isoformat(),
                "critical_error": str(e),
            },
            "results": [],
            "error": f"Critical analysis failure: {str(e)}",
        }


def validate_adgm_compliance_comprehensive(
    document_content: str, doc_type: str
) -> Dict[str, Any]:
    """Comprehensive ADGM compliance validation using all available sources"""

    # Get relevant RAG snippets for the document type
    relevant_refs = retrieve_relevant_snippets(
        f"ADGM {doc_type} requirements compliance jurisdiction", top_k=5
    )

    # Enhanced compliance checks
    compliance_issues = []

    # 1. Jurisdiction Check
    jurisdiction_issues = check_jurisdiction_compliance(document_content)
    if jurisdiction_issues:
        compliance_issues.extend(jurisdiction_issues)

    # 2. Required Clauses Check
    missing_clauses = check_required_clauses(document_content, doc_type)
    if missing_clauses:
        compliance_issues.extend(missing_clauses)

    # 3. Formatting Check
    formatting_issues = check_document_formatting(document_content, doc_type)
    if formatting_issues:
        compliance_issues.extend(formatting_issues)

    # 4. LLM-based analysis for complex issues
    llm_analysis = gemini_legal_analysis(
        f"Comprehensive ADGM compliance analysis for {doc_type}: {document_content[:2000]}",
        relevant_refs,
    )

    if llm_analysis.get("red_flag") and llm_analysis["red_flag"] != "null":
        compliance_issues.append(llm_analysis)

    return {
        "doc_type": doc_type,
        "compliance_score": calculate_compliance_score(compliance_issues),
        "issues": compliance_issues,
        "total_issues": len(compliance_issues),
        "high_priority_issues": len(
            [i for i in compliance_issues if i.get("severity") == "High"]
        ),
        "recommendations": generate_compliance_recommendations(
            compliance_issues, doc_type
        ),
    }


def check_jurisdiction_compliance(content: str) -> List[Dict[str, Any]]:
    """Check for jurisdiction compliance issues"""
    issues = []
    content_lower = content.lower()

    # Check for incorrect jurisdiction references
    incorrect_jurisdictions = [
        "uae federal court",
        "dubai court",
        "abu dhabi court",
        "sharjah court",
        "federal court of uae",
        "emirates court",
    ]

    for jurisdiction in incorrect_jurisdictions:
        if jurisdiction in content_lower:
            issues.append(
                {
                    "red_flag": f"Document references '{jurisdiction}' instead of ADGM Courts",
                    "law_citation": "ADGM Companies Regulations 2020, Article 6 - Jurisdiction",
                    "suggestion": "Replace with 'ADGM Courts' or 'Abu Dhabi Global Market Courts'",
                    "severity": "High",
                    "category": "jurisdiction",
                    "confidence": "High",
                }
            )

    # Check for missing ADGM jurisdiction reference
    if "adgm" not in content_lower or "abu dhabi global market" not in content_lower:
        if any(
            word in content_lower
            for word in ["court", "jurisdiction", "dispute", "law"]
        ):
            issues.append(
                {
                    "red_flag": "Document may be missing explicit ADGM jurisdiction clause",
                    "law_citation": "ADGM Companies Regulations 2020, Article 6 - Jurisdiction",
                    "suggestion": "Add clause: 'This document shall be governed by ADGM law and subject to ADGM Courts jurisdiction'",
                    "severity": "Medium",
                    "category": "jurisdiction",
                    "confidence": "Medium",
                }
            )

    return issues


def check_required_clauses(content: str, doc_type: str) -> List[Dict[str, Any]]:
    """Check for missing required clauses based on document type"""
    issues = []
    content_lower = content.lower()

    required_clauses = {
        "Articles of Association (AoA)": [
            ("registered office", "Registered office address in ADGM"),
            ("share capital", "Share capital and class of shares"),
            ("director", "Director appointment and powers"),
            ("shareholder", "Shareholder rights and obligations"),
        ],
        "Memorandum of Association (MoA/MoU)": [
            ("company name", "Company name and registration details"),
            ("object", "Company objects and purposes"),
            ("liability", "Limitation of liability clause"),
            ("registered office", "Registered office in ADGM"),
        ],
        "Employment Contract": [
            ("termination", "Termination notice and procedures"),
            ("salary", "Salary and compensation details"),
            ("probation", "Probation period terms"),
            ("governing law", "ADGM governing law clause"),
        ],
    }

    if doc_type in required_clauses:
        for clause_keyword, clause_description in required_clauses[doc_type]:
            if clause_keyword not in content_lower:
                issues.append(
                    {
                        "red_flag": f"Missing required clause: {clause_description}",
                        "law_citation": get_adgm_citation_for_issue(
                            "missing_clauses", doc_type
                        ),
                        "suggestion": f"Add {clause_description} section to document",
                        "severity": "High",
                        "category": "missing_clauses",
                        "confidence": "High",
                    }
                )

    return issues


def check_document_formatting(content: str, doc_type: str) -> List[Dict[str, Any]]:
    """Check document formatting and structure"""
    issues = []

    # Check for signature requirements
    if "signature" not in content.lower() and "signed" not in content.lower():
        if doc_type in [
            "Articles of Association (AoA)",
            "Memorandum of Association (MoA/MoU)",
            "Board Resolution",
        ]:
            issues.append(
                {
                    "red_flag": "Document appears to be missing signature section",
                    "law_citation": "ADGM Companies Regulations 2020, Article 25 - Execution of Documents",
                    "suggestion": "Add proper signature block with authorized signatory details",
                    "severity": "Medium",
                    "category": "formatting",
                    "confidence": "Medium",
                }
            )

    # Check for date requirements
    if (
        doc_type in ["Board Resolution", "Shareholder Resolution"]
        and "date" not in content.lower()
    ):
        issues.append(
            {
                "red_flag": "Resolution document missing date",
                "law_citation": "ADGM Companies Regulations 2020, Article 30 - Board Resolutions",
                "suggestion": "Add proper date of resolution at the beginning of document",
                "severity": "Medium",
                "category": "formatting",
                "confidence": "High",
            }
        )

    return issues


def calculate_compliance_score(issues: List[Dict[str, Any]]) -> float:
    """Calculate overall compliance score based on issues found"""
    if not issues:
        return 100.0

    # Weight issues by severity
    total_weight = 0
    for issue in issues:
        severity = issue.get("severity", "Medium")
        weight = {"High": 10, "Medium": 5, "Low": 2}.get(severity, 5)
        total_weight += weight

    # Calculate score (100 is perfect, decreases with issues)
    max_possible_weight = len(issues) * 10  # If all were High severity
    score = max(0, 100 - (total_weight / max_possible_weight * 100))

    return round(score, 1)


def generate_compliance_recommendations(
    issues: List[Dict[str, Any]], doc_type: str
) -> List[str]:
    """Generate specific recommendations based on compliance issues"""
    recommendations = []

    # Group issues by category
    categories = {}
    for issue in issues:
        category = issue.get("category", "general")
        if category not in categories:
            categories[category] = []
        categories[category].append(issue)

    # Generate category-specific recommendations
    if "jurisdiction" in categories:
        recommendations.append(
            "Update all jurisdiction references to specify 'ADGM Courts' and remove any references to UAE Federal or local courts"
        )

    if "missing_clauses" in categories:
        recommendations.append(
            f"Add missing mandatory clauses as required by ADGM regulations for {doc_type}"
        )

    if "formatting" in categories:
        recommendations.append(
            "Ensure proper document formatting including signature blocks, dates, and structural requirements"
        )

    if "compliance" in categories:
        recommendations.append(
            "Review document against current ADGM templates and regulations to ensure full compliance"
        )

    # Add general recommendation
    recommendations.append(
        "Consider engaging qualified ADGM legal counsel for final compliance verification before submission"
    )

    return recommendations


# Export key functions for backward compatibility
__all__ = [
    "retrieve_relevant_snippets",
    "gemini_legal_analysis",
    "analyze_document_completeness",
    "batch_analysis",
    "validate_adgm_compliance_comprehensive",
    "REFERENCE_SNIPPETS",
    "RED_FLAG_CATEGORIES",
]
