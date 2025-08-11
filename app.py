import asyncio
import datetime
import json
import logging
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from adgm_checklists import (
    PROCESS_TO_DOCS,
    analyze_document_gaps,
    calculate_process_confidence,
    detect_process_from_documents,
    get_all_document_types,
    get_document_priority,
    get_missing_documents_for_process,
    get_process_description,
    get_required_docs_for_process,
    suggest_alternative_processes,
    validate_document_combinations,
    validate_process_requirements,
)
from adgm_rag import batch_analysis  # Updated to use new enhanced function
from adgm_rag import (
    RED_FLAG_CATEGORIES,
    REFERENCE_SNIPPETS,
    analyze_document_completeness,
    gemini_legal_analysis,
    retrieve_relevant_snippets,
    validate_adgm_compliance_comprehensive,
)

# Import enhanced modules
from config import (
    INITIAL_SIDEBAR_STATE,
    LAYOUT,
    PAGE_ICON,
    PAGE_TITLE,
    PROCESSED_DIR,
    UPLOAD_DIR,
    USE_CHROMADB,
    get_config_summary,
    get_rag_config,
    validate_config,
)
from doc_utils import analyze_document_completeness as analyze_doc_structure
from doc_utils import (
    build_red_flag_prompt,
    create_compliance_report,
    detect_document_type,
    extract_document_metadata,
    extract_text_sections,
    insert_comments,
    save_docx,
    validate_document_structure,
)
from error_handler import GracefulDegradation  # Import for graceful degradation


# Processing status update function
def update_processing_status(status_text, message):
    """Update processing status with enhanced styling"""
    try:
        # Determine status type based on message content
        if "error" in message.lower() or "failed" in message.lower():
            status_class = "error"
            status_icon = "❌"
        elif "completed" in message.lower() or "success" in message.lower():
            status_class = "success"
            status_icon = "✅"
        elif "processing" in message.lower() or "analyzing" in message.lower():
            status_class = "processing"
            status_icon = "🔄"
        else:
            status_class = "info"
            status_icon = "ℹ️"

        status_text.markdown(
            f"""
            <div class="processing-status {status_class}">
                <span class="status-icon">{status_icon}</span>
                <strong>Processing Status:</strong> {message}
            </div>
            """,
            unsafe_allow_html=True,
        )
    except Exception as e:
        logger.warning(f"Error updating processing status: {e}")
        # Fallback to simple text
        status_text.text(f"Status: {message}")


# Page configuration with enhanced settings
st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout=LAYOUT,
    initial_sidebar_state=INITIAL_SIDEBAR_STATE,
    menu_items={
        "About": "ADGM-Compliant Corporate Agent - Intelligent legal document analysis for ADGM jurisdiction"
    },
)

# Enhanced CSS styling
st.markdown(
    """
<style>
    /* Main styling */
    .main-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
    }
    
    /* Status boxes */
    .status-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid;
    }
    
    .status-box.success {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border-left-color: #28a745;
        color: #155724;
    }
    
    .status-box.warning {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        border-left-color: #ffc107;
        color: #856404;
    }
    
    .status-box.error {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        border-left-color: #dc3545;
        color: #721c24;
    }
    
    /* Progress styling */
    .progress-container {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 1px solid #e9ecec;
    }
    
    /* Processing status styling - Fixed visibility */
    .processing-status {
        background: #e3f2fd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #007bff;
        margin: 1rem 0;
        color: #0d47a1;
        font-weight: 500;
    }
    
    .processing-status .status-icon {
        font-size: 1.5rem;
        margin-right: 0.5rem;
    }
    
    /* Status type variations */
    .processing-status.success {
        background: #d4edda;
        border-left-color: #28a745;
        color: #155724;
    }
    
    .processing-status.error {
        background: #f8d7da;
        border-left-color: #dc3545;
        color: #721c24;
    }
    
    .processing-status.info {
        background: #d1ecf1;
        border-left-color: #17a2b8;
        color: #0c5460;
    }
    
    .processing-status.processing {
        background: #fff3cd;
        border-left-color: #ffc107;
        color: #856404;
    }
    
    /* File upload styling */
    .file-upload {
        border: 2px dashed #dee2e6;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background: #f8f9fa;
        transition: all 0.3s ease;
    }
    
    .file-upload:hover {
        border-color: #007bff;
        background: #e3f2fd;
    }
    
    /* Results styling */
    .results-container {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    /* Metric styling */
    .metric-container {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        text-align: center;
    }
    
    .metric-value {
        font-size: 1.5rem;
        font-weight: bold;
        color: #007bff;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #6c757d;
        margin-top: 0.5rem;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Validation configuration
config_errors = validate_config()
if config_errors:
    st.error("Configuration errors found:")
    for error in config_errors:
        st.error(f"• {error}")
    st.stop()

# Main header with enhanced design
st.markdown(
    f"""
<div class="main-header">
    <h1>{PAGE_ICON} ADGM-Compliant Corporate Agent</h1>
    <h3>Intelligent AI-powered legal assistant for ADGM jurisdiction</h3>
    <p>🚀 Enhanced document intelligence • 📊 Comprehensive compliance analysis • ⚖️ Real-time legal validation</p>
</div>
""",
    unsafe_allow_html=True,
)

# Enhanced sidebar with comprehensive information
st.sidebar.markdown("## 📋 ADGM Document Categories")
st.sidebar.markdown("*Comprehensive coverage of all ADGM processes*")

# Display process categories with statistics
for process, docs in PROCESS_TO_DOCS.items():
    with st.sidebar.expander(f"📄 {process} ({len(docs)} docs)"):
        st.markdown(f"*{get_process_description(process)}*")

        # Show document priorities
        critical_docs = [
            doc for doc in docs if get_document_priority(process, doc) == "Critical"
        ]
        important_docs = [
            doc for doc in docs if get_document_priority(process, doc) == "Important"
        ]

        if critical_docs:
            st.markdown("**🔴 Critical:**")
            for doc in critical_docs[:3]:
                st.markdown(f"• {doc}")

        if important_docs:
            st.markdown("**🟡 Important:**")
            for doc in important_docs[:3]:
                st.markdown(f"• {doc}")

        if len(docs) > 6:
            st.markdown(f"*... and {len(docs) - 6} more documents*")

st.sidebar.markdown("---")
st.sidebar.markdown("## 🔗 Official ADGM References")
st.sidebar.markdown("*All data sources from official ADGM platforms*")

# Display reference sources by category
reference_categories = {}
for ref in REFERENCE_SNIPPETS[:8]:
    category = ref.get("category", "General")
    if category not in reference_categories:
        reference_categories[category] = []
    reference_categories[category].append(ref)

for category, refs in reference_categories.items():
    with st.sidebar.expander(f"📚 {category} ({len(refs)})"):
        for ref in refs:
            st.markdown(f"[{ref['title'][:40]}...]({ref['url']})")

# Configuration summary in sidebar
with st.sidebar.expander("⚙️ System Configuration"):
    config_summary = get_config_summary()
    rag_config = get_rag_config()

    # Show RAG backend prominently
    st.markdown(f"**🗄️ RAG Backend:** {rag_config['backend']}")
    if rag_config["backend"] == "ChromaDB":
        st.success("✅ Using ChromaDB with cosine similarity")
    else:
        st.info("⚡ Using in-memory with dot product similarity")

    st.json(config_summary)

# Main content area
st.markdown("## 📤 Document Upload & Analysis")

# Enhanced file uploader with detailed instructions
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown(
        """
    ### Instructions:
    1. **Upload Documents**: Select one or more ADGM legal documents (.docx format)
    2. **Automatic Processing**: System detects document types and legal processes
    3. **AI Analysis**: Each document analyzed for ADGM compliance using official sources
    4. **Review Results**: Download marked-up documents with inline comments
    5. **Get Reports**: Comprehensive JSON analysis with recommendations
    """
    )

with col2:
    st.markdown(
        """
    ### Supported Types:
    • **Company Formation** (16 doc types)
    • **Licensing** (14 doc types)  
    • **Employment** (15 doc types)
    • **Commercial** (15 doc types)
    • **Compliance** (20 doc types)
    • **Financial Services** (12 doc types)
    """
    )

# File uploader with enhanced validation
uploaded_files = st.file_uploader(
    "Choose ADGM legal documents (.docx)",
    type=["docx"],
    accept_multiple_files=True,
    help="Upload up to 10 documents. Each file will be analyzed for ADGM compliance.",
    key="document_uploader",
)

if uploaded_files:
    # Validate file count and sizes
    if len(uploaded_files) > 10:
        st.error("❌ Maximum 10 files allowed per upload session")
        st.stop()

    total_size = sum(len(file.getbuffer()) for file in uploaded_files)
    if total_size > 100 * 1024 * 1024:  # 100MB total limit
        st.error("❌ Total file size exceeds 100MB limit")
        st.stop()

    # Display upload summary
    st.success(f"✅ {len(uploaded_files)} file(s) uploaded successfully!")

    # Create upload summary table
    file_info = []
    for file in uploaded_files:
        size_mb = len(file.getbuffer()) / (1024 * 1024)
        file_info.append(
            {
                "File Name": file.name,
                "Size (MB)": f"{size_mb:.2f}",
                "Type": "DOCX Document",
            }
        )

    st.dataframe(pd.DataFrame(file_info), use_container_width=True)

    # Processing section
    st.markdown("## 🔄 Processing Documents")

    # Initialize processing containers
    progress_container = st.container()
    results_container = st.container()

    with progress_container:
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Add processing status indicator
        processing_status = st.empty()
        processing_status.markdown(
            """
            <div class="processing-status info">
                <span class="status-icon">🚀</span>
                <strong>Processing Status:</strong> <span id="processing-status-text">Ready to start</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Processing metrics
        processing_metrics = st.columns(4)

        with processing_metrics[0]:
            processed_count = st.metric("Processed", 0)
        with processing_metrics[1]:
            issues_found = st.metric("Issues Found", 0)
        with processing_metrics[2]:
            compliance_avg = st.metric("Avg Compliance", "0%")
        with processing_metrics[3]:
            # Removed processing time metric
            pass

    # Start processing
    start_time = datetime.datetime.now()
    # Start document analysis
    update_processing_status(processing_status, "Starting document analysis...")

    # Initialize variables
    all_document_data = []
    doc_results = []
    found_types = []
    all_flagged_sections = []
    process_guess = None
    error_docs = []
    total_issues = 0
    compliance_scores = []

    # Process each uploaded file
    for idx, uploaded_file in enumerate(uploaded_files):
        try:
            # Update processing status
            current_status = (
                f"Analyzing {uploaded_file.name}... ({idx + 1}/{len(uploaded_files)})"
            )
            update_processing_status(processing_status, current_status)
            status_text.text(current_status)

            # Save uploaded file
            uploaded_save_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
            with open(uploaded_save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Extract text and metadata
            update_processing_status(
                processing_status, f"Extracting text from {uploaded_file.name}..."
            )
            sections = extract_text_sections(uploaded_file)
            metadata = extract_document_metadata(uploaded_file)

            # Continue with comprehensive analysis

            # Detect document type with enhanced algorithm
            update_processing_status(
                processing_status,
                f"Detecting document type for {uploaded_file.name}...",
            )
            doc_type = "Unknown"
            type_confidence = 0.0

            # Try multiple detection methods
            # Ensure sections is a list and has elements before slicing
            if not isinstance(sections, list):
                logger.warning(
                    f"Sections is not a list: {type(sections)}, using empty list"
                )
                sections = []

            detection_sections = sections[:5] if len(sections) >= 5 else sections

            for section in detection_sections:
                if section.get("text"):
                    detected_type = detect_document_type(
                        section["text"], get_all_document_types()
                    )
                    if detected_type != "Unknown":
                        doc_type = detected_type
                        type_confidence = 0.8
                        break

            # Fallback to filename analysis
            if doc_type == "Unknown":
                filename_type = detect_document_type(
                    uploaded_file.name, get_all_document_types()
                )
                if filename_type != "Unknown":
                    doc_type = filename_type
                    type_confidence = 0.6

            found_types.append(doc_type)

            # Document structure analysis
            update_processing_status(
                processing_status, f"Analyzing structure of {uploaded_file.name}..."
            )
            structure_analysis = analyze_doc_structure(sections, doc_type)
            structure_issues = validate_document_structure(doc_type, metadata)

            # Continue with comprehensive analysis

            # Comprehensive ADGM compliance analysis
            # Ensure sections is a list before processing
            if not isinstance(sections, list):
                logger.warning(
                    f"Sections is not a list: {type(sections)}, using empty list"
                )
                sections = []

            full_document_text = " ".join(
                [section.get("text", "") for section in sections]
            )
            compliance_analysis = validate_adgm_compliance_comprehensive(
                full_document_text, doc_type
            )

            # RAG-based detailed analysis using enhanced batch processing
            rag_refs = retrieve_relevant_snippets(
                f"ADGM {doc_type} compliance requirements"
            )

            # Prepare document data for batch analysis
            document_data = {
                "filename": uploaded_file.name,
                "doc_type": doc_type,
                "sections": sections,
                "category": "legal_document",
            }

            # Use enhanced batch analysis with rate limiting and smart batching
            try:
                # Run async batch analysis
                batch_result = asyncio.run(
                    batch_analysis([document_data], batch_size=5, max_concurrent=3)
                )

                if (
                    batch_result
                    and "results" in batch_result
                    and batch_result["results"]
                    and len(batch_result["results"]) > 0  # Ensure list has elements
                    and isinstance(batch_result["results"], list)  # Ensure it's a list
                ):
                    # Extract section issues from batch results
                    doc_batch_result = batch_result["results"][
                        0
                    ]  # First (and only) document

                    # Validate doc_batch_result structure
                    if not isinstance(doc_batch_result, dict):
                        logger.warning(
                            f"Document batch result is not a dictionary: {type(doc_batch_result)}"
                        )
                        doc_batch_result = {"results": []}

                    section_issues = []

                    # Validate results list before processing
                    results_list = doc_batch_result.get("results", [])
                    if not isinstance(results_list, list):
                        logger.warning(
                            f"Results is not a list: {type(results_list)}, using empty list"
                        )
                        results_list = []

                    for section_result in results_list:
                        if (
                            section_result.get("red_flag")
                            and section_result["red_flag"] != "null"
                            and section_result["red_flag"] != "Analysis completed"
                        ):

                            section_issues.append(
                                {
                                    **section_result,
                                    "section_index": section_result.get(
                                        "section_index", 0
                                    ),
                                    "section_clause": section_result.get(
                                        "section_clause", "Unknown"
                                    ),
                                    "section_text": section_result.get(
                                        "section_text", ""
                                    ),
                                    "section_type": section_result.get(
                                        "section_type", "content"
                                    ),
                                    "analysis_method": section_result.get(
                                        "analysis_method", "api"
                                    ),
                                }
                            )

                    # Log batch analysis results
                    logger.info(
                        f"Batch analysis completed for {uploaded_file.name}: {len(section_issues)} issues found"
                    )

                else:
                    logger.warning(
                        f"Batch analysis failed for {uploaded_file.name}, falling back to individual analysis"
                    )
                    section_issues = []

                    # Fallback to individual analysis for critical sections only
                    # Ensure sections is a list and has elements before slicing
                    if not isinstance(sections, list):
                        logger.warning(
                            f"Sections is not a list: {type(sections)}, using empty list"
                        )
                        sections = []

                    fallback_sections = sections[:3] if len(sections) >= 3 else sections

                    for section_idx, section in enumerate(fallback_sections):
                        if section.get("text") and len(section["text"]) > 20:
                            prompt = build_red_flag_prompt(
                                section["text"],
                                doc_type,
                                {
                                    "document_metadata": metadata,
                                    "structure_analysis": structure_analysis,
                                },
                            )

                            try:
                                analysis = gemini_legal_analysis(prompt, rag_refs)

                                if (
                                    isinstance(analysis, dict)
                                    and analysis.get("red_flag")
                                    and analysis["red_flag"] != "null"
                                ):

                                    section_issues.append(
                                        {
                                            **analysis,
                                            "section_index": section.get("index"),
                                            "section_clause": section.get("clause"),
                                            "section_text": (
                                                section["text"][:100] + "..."
                                                if len(section["text"]) > 100
                                                else section["text"]
                                            ),
                                            "section_type": section.get(
                                                "type", "content"
                                            ),
                                            "analysis_method": "fallback",
                                        }
                                    )
                            except Exception as e:
                                logger.error(
                                    f"Fallback analysis failed for section {section_idx}: {e}"
                                )
                                continue

            except Exception as e:
                logger.error(
                    f"Enhanced batch analysis failed for {uploaded_file.name}: {e}"
                )
                # Use graceful degradation
                section_issues = []

                # Basic compliance check using local rules
                # Ensure sections is a list before processing
                if not isinstance(sections, list):
                    logger.warning(
                        f"Sections is not a list: {type(sections)}, using empty list"
                    )
                    sections = []

                full_document_text = " ".join(
                    [section.get("text", "") for section in sections]
                )
                fallback_analysis = GracefulDegradation.basic_compliance_check(
                    full_document_text, doc_type
                )

                for issue in fallback_analysis.get("issues", []):
                    section_issues.append(
                        {
                            **issue,
                            "section_index": 0,
                            "section_clause": "Document",
                            "section_text": (
                                full_document_text[:200] + "..."
                                if len(full_document_text) > 200
                                else full_document_text
                            ),
                            "section_type": "content",
                            "analysis_method": "graceful_degradation",
                        }
                    )

            # Add structure issues to flagged sections
            for issue in structure_issues:
                section_issues.append(
                    {
                        "red_flag": issue,
                        "law_citation": "ADGM Document Structure Requirements",
                        "suggestion": "Review document structure and ensure all required sections are present",
                        "severity": "Medium",
                        "category": "formatting",
                        "section_index": 0,
                        "section_type": "structure",
                    }
                )

            # Combine all issues
            all_issues = section_issues + compliance_analysis.get("issues", [])
            all_flagged_sections.extend(all_issues)

            # Create marked-up document
            if all_issues:
                try:
                    docx_out = insert_comments(uploaded_file, all_issues)
                    processed_filename = f"reviewed_{uploaded_file.name}"
                    processed_path = os.path.join(PROCESSED_DIR, processed_filename)
                    save_success = save_docx(docx_out, processed_path)

                except Exception as e:
                    logger.error(
                        f"Error creating marked-up document for {uploaded_file.name}: {e}"
                    )
                    processed_path = None
                    save_success = False
            else:
                processed_path = None
                save_success = False

            # Calculate compliance score
            doc_compliance_score = compliance_analysis.get("compliance_score", 100.0)
            compliance_scores.append(doc_compliance_score)

            # Store document results
            doc_results.append(
                {
                    "filename": uploaded_file.name,
                    "doc_type": doc_type,
                    "type_confidence": type_confidence,
                    "flagged_sections": all_issues,
                    "marked_docx": processed_path if save_success else None,
                    "metadata": metadata,
                    "structure_issues": structure_issues,
                    "structure_analysis": structure_analysis,
                    "compliance_analysis": compliance_analysis,
                    "compliance_score": doc_compliance_score,
                    "total_issues": len(all_issues),
                    "high_priority_issues": len(
                        [i for i in all_issues if i.get("severity") == "High"]
                    ),
                    "processing_status": "completed",
                }
            )

            # Update metrics
            total_issues += len(all_issues)
            avg_compliance = sum(compliance_scores) / len(compliance_scores)

            # Update progress
            progress_bar.progress((idx + 1) / len(uploaded_files))

            # Update real-time metrics
            processed_count.metric("Processed", idx + 1)
            issues_found.metric("Issues Found", total_issues)
            compliance_avg.metric("Avg Compliance", f"{avg_compliance:.1f}%")

        except Exception as e:
            logger.error(f"Error processing {uploaded_file.name}: {e}")
            error_docs.append((uploaded_file.name, str(e)))

            # Add error document to results
            doc_results.append(
                {
                    "filename": uploaded_file.name,
                    "doc_type": "Unknown",
                    "type_confidence": 0.0,
                    "flagged_sections": [],
                    "marked_docx": None,
                    "processing_status": "error",
                    "error": str(e),
                }
            )

    # Final processing metrics
    end_time = datetime.datetime.now()
    total_processing_time = (end_time - start_time).total_seconds()
    # Processing completed
    update_processing_status(processing_status, "✅ Analysis completed successfully!")

    status_text.text("✅ Processing completed!")

    # Process detection and comprehensive analysis
    if found_types:
        # Process detection and analysis

        process_guess = detect_process_from_documents(found_types)
        process_confidence = calculate_process_confidence(found_types, process_guess)

        # Alternative process suggestions
        alternative_processes = suggest_alternative_processes(
            found_types, process_guess
        )

        # Document combination validation
        combination_issues = validate_document_combinations(found_types)

        # Enhanced completeness analysis
        completeness_analysis = get_missing_documents_for_process(
            process_guess, found_types
        )

        # Gap analysis
        gap_analysis = analyze_document_gaps(found_types, process_guess)

        # Process requirements validation
        process_validation = validate_process_requirements(found_types, process_guess)

        # Display comprehensive results
        with results_container:
            st.markdown("## 📊 Comprehensive Analysis Results")

            # Executive Summary
            st.markdown("### 🎯 Executive Summary")

            summary_cols = st.columns(4)
            with summary_cols[0]:
                st.metric("Documents Analyzed", len(uploaded_files))
            with summary_cols[1]:
                st.metric("Issues Identified", total_issues)
            with summary_cols[2]:
                avg_compliance = (
                    sum(compliance_scores) / len(compliance_scores)
                    if compliance_scores
                    else 0
                )
                st.metric("Average Compliance", f"{avg_compliance:.1f}%")
            with summary_cols[3]:
                st.metric("Process Confidence", f"{process_confidence * 100:.1f}%")

            # Processing Time Summary
            st.markdown("### ⏱️ Processing Summary")
            processing_summary_cols = st.columns(3)
            with processing_summary_cols[0]:
                st.metric("Total Processing Time", f"{total_processing_time:.1f}s")
            with processing_summary_cols[1]:
                st.metric(
                    "Documents per Second",
                    f"{len(uploaded_files)/total_processing_time:.2f}",
                )
            with processing_summary_cols[2]:
                st.metric(
                    "Issues per Document", f"{total_issues/len(uploaded_files):.1f}"
                )

            # Process Analysis
            st.markdown("### 🎯 Legal Process Analysis")

            col1, col2 = st.columns([2, 1])

            with col1:
                # Process selection with alternatives
                process_options = [process_guess] + [
                    alt["process"] for alt in alternative_processes
                ]
                selected_process = st.selectbox(
                    "Detected Legal Process:",
                    options=process_options,
                    index=0,
                    help="Primary detected process based on uploaded documents. You can select alternative processes if needed.",
                )

                if selected_process != process_guess:
                    # Recalculate for selected process
                    completeness_analysis = get_missing_documents_for_process(
                        selected_process, found_types
                    )
                    gap_analysis = analyze_document_gaps(found_types, selected_process)
                    process_guess = selected_process

                st.markdown(
                    f"**Description:** {get_process_description(process_guess)}"
                )
                st.markdown(f"**Confidence Level:** {process_confidence * 100:.1f}%")

            with col2:
                st.markdown("**Process Statistics:**")
                required_docs = get_required_docs_for_process(process_guess)
                st.markdown(f"• Required Documents: {len(required_docs)}")
                st.markdown(f"• Documents Uploaded: {len(uploaded_files)}")
                st.markdown(
                    f"• Completeness: {completeness_analysis['completeness_percentage']:.1f}%"
                )
                st.markdown(
                    f"• Weighted Score: {completeness_analysis['weighted_completeness']:.1f}%"
                )

            # Alternative processes
            if alternative_processes:
                st.markdown("**Alternative Process Suggestions:**")
                alt_df = pd.DataFrame(alternative_processes)
                alt_df["relevance"] = alt_df["relevance"].apply(
                    lambda x: f"{x*100:.1f}%"
                )
                alt_df["confidence"] = alt_df["confidence"].apply(
                    lambda x: f"{x*100:.1f}%"
                )
                st.dataframe(alt_df, use_container_width=True)

            # Completeness Analysis
            st.markdown("### 📋 Document Completeness Analysis")

            if completeness_analysis["missing"]:
                st.markdown(
                    f"""
                <div class="warning-box">
                    ⚠️ <strong>Incomplete Submission</strong><br>
                    You have uploaded {len(uploaded_files)} out of {len(get_required_docs_for_process(process_guess))} required documents.<br><br>
                    <strong>Completeness:</strong> {completeness_analysis["completeness_percentage"]:.1f}% (Standard) | {completeness_analysis["weighted_completeness"]:.1f}% (Weighted)<br>
                    <strong>Missing Documents:</strong> {len(completeness_analysis["missing"])}
                </div>
                """,
                    unsafe_allow_html=True,
                )

                # Missing documents breakdown
                missing_breakdown = completeness_analysis.get("missing_by_priority", {})

                col1, col2, col3 = st.columns(3)
                with col1:
                    if missing_breakdown.get("Critical"):
                        st.markdown("**🔴 Critical Missing:**")
                        for doc in missing_breakdown["Critical"][:5]:
                            st.markdown(f"• {doc}")

                with col2:
                    if missing_breakdown.get("Important"):
                        st.markdown("**🟡 Important Missing:**")
                        for doc in missing_breakdown["Important"][:5]:
                            st.markdown(f"• {doc}")

                with col3:
                    if missing_breakdown.get("Optional"):
                        st.markdown("**🟢 Optional Missing:**")
                        for doc in missing_breakdown["Optional"][:5]:
                            st.markdown(f"• {doc}")
            else:
                st.markdown(
                    f"""
                <div class="success-box">
                    ✅ <strong>Complete Submission</strong><br>
                    All required documents for {process_guess} have been uploaded!<br>
                    <strong>Completeness:</strong> 100%
                </div>
                """,
                    unsafe_allow_html=True,
                )

            # Gap Analysis and Recommendations
            st.markdown("### 🎯 Gap Analysis & Recommendations")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**📋 Recommendations:**")
                for rec in gap_analysis.get("recommendations", []):
                    st.markdown(f"• {rec}")

            with col2:
                st.markdown("**🎯 Next Steps:**")
                for step in gap_analysis.get("next_steps", []):
                    st.markdown(f"• {step}")

            # Document combination issues
            if combination_issues:
                st.markdown("**⚠️ Document Combination Issues:**")
                for issue in combination_issues:
                    st.warning(f"• {issue}")

            # Process validation results
            if not process_validation.get("valid", True):
                st.markdown("**❌ Process Validation Issues:**")
                for issue in process_validation.get("issues", []):
                    st.error(f"• {issue}")

            # Document Analysis Details
            st.markdown("### 📄 Document Analysis Details")

            # Create comprehensive document summary
            summary_data = []
            for doc in doc_results:
                if doc.get("processing_status") == "completed":
                    priority = get_document_priority(process_guess, doc["doc_type"])
                    issues_count = doc.get("total_issues", 0)
                    high_issues = doc.get("high_priority_issues", 0)
                    compliance_score = doc.get("compliance_score", 0)

                    status_emoji = (
                        "✅" if issues_count == 0 else "⚠️" if high_issues == 0 else "❌"
                    )

                    summary_data.append(
                        {
                            "Document": doc["filename"],
                            "Type": doc["doc_type"],
                            "Priority": priority,
                            "Compliance Score": f"{compliance_score:.1f}%",
                            "Total Issues": issues_count,
                            "High Priority": high_issues,
                            "Status": f"{status_emoji} {'Clean' if issues_count == 0 else 'Issues Found'}",
                        }
                    )
                else:
                    summary_data.append(
                        {
                            "Document": doc["filename"],
                            "Type": "Error",
                            "Priority": "N/A",
                            "Compliance Score": "0%",
                            "Total Issues": 0,
                            "High Priority": 0,
                            "Status": "❌ Processing Error",
                        }
                    )

            # Display document summary table
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True)

            # Detailed Issues Analysis
            st.markdown("### 🚨 Detailed Issues Analysis")

            if all_flagged_sections:
                # Group issues by severity
                high_issues = [
                    i for i in all_flagged_sections if i.get("severity") == "High"
                ]
                medium_issues = [
                    i for i in all_flagged_sections if i.get("severity") == "Medium"
                ]
                low_issues = [
                    i for i in all_flagged_sections if i.get("severity") == "Low"
                ]

                # Severity tabs
                severity_tabs = st.tabs(
                    [
                        "🔴 High Priority",
                        "🟡 Medium Priority",
                        "🟢 Low Priority",
                        "📊 Summary",
                    ]
                )

                with severity_tabs[0]:
                    if high_issues:
                        for i, issue in enumerate(
                            high_issues[:10]
                        ):  # Limit to first 10
                            with st.expander(
                                f"Issue {i+1}: {issue.get('red_flag', 'Unknown issue')[:50]}..."
                            ):
                                st.markdown(
                                    f"**Document:** {next((doc['filename'] for doc in doc_results if any(flag == issue for flag in doc.get('flagged_sections', []))), 'Unknown')}"
                                )
                                st.markdown(
                                    f"**Issue:** {issue.get('red_flag', 'Not specified')}"
                                )
                                st.markdown(
                                    f"**Citation:** {issue.get('law_citation', 'Not specified')}"
                                )
                                st.markdown(
                                    f"**Suggestion:** {issue.get('suggestion', 'Not specified')}"
                                )
                                st.markdown(
                                    f"**Category:** {issue.get('category', 'Not specified')}"
                                )
                                if issue.get("compliant_clause"):
                                    st.markdown(
                                        f"**Compliant Clause:** {issue['compliant_clause']}"
                                    )
                    else:
                        st.success("✅ No high priority issues found!")

                with severity_tabs[1]:
                    if medium_issues:
                        for i, issue in enumerate(
                            medium_issues[:15]
                        ):  # Limit to first 15
                            with st.expander(
                                f"Issue {i+1}: {issue.get('red_flag', 'Unknown issue')[:50]}..."
                            ):
                                st.markdown(
                                    f"**Document:** {next((doc['filename'] for doc in doc_results if any(flag == issue for flag in doc.get('flagged_sections', []))), 'Unknown')}"
                                )
                                st.markdown(
                                    f"**Issue:** {issue.get('red_flag', 'Not specified')}"
                                )
                                st.markdown(
                                    f"**Citation:** {issue.get('law_citation', 'Not specified')}"
                                )
                                st.markdown(
                                    f"**Suggestion:** {issue.get('suggestion', 'Not specified')}"
                                )
                    else:
                        st.success("✅ No medium priority issues found!")

                with severity_tabs[2]:
                    if low_issues:
                        for i, issue in enumerate(low_issues[:20]):  # Limit to first 20
                            with st.expander(
                                f"Issue {i+1}: {issue.get('red_flag', 'Unknown issue')[:50]}..."
                            ):
                                st.markdown(
                                    f"**Document:** {next((doc['filename'] for doc in doc_results if any(flag == issue for flag in doc.get('flagged_sections', []))), 'Unknown')}"
                                )
                                st.markdown(
                                    f"**Issue:** {issue.get('red_flag', 'Not specified')}"
                                )
                                st.markdown(
                                    f"**Citation:** {issue.get('law_citation', 'Not specified')}"
                                )
                                st.markdown(
                                    f"**Suggestion:** {issue.get('suggestion', 'Not specified')}"
                                )
                    else:
                        st.success("✅ No low priority issues found!")

                with severity_tabs[3]:
                    # Issues summary statistics
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("**Issues by Severity:**")
                        severity_data = {
                            "Severity": ["High", "Medium", "Low"],
                            "Count": [
                                len(high_issues),
                                len(medium_issues),
                                len(low_issues),
                            ],
                        }
                        st.bar_chart(pd.DataFrame(severity_data).set_index("Severity"))

                    with col2:
                        st.markdown("**Issues by Category:**")
                        category_counts = {}
                        for issue in all_flagged_sections:
                            category = issue.get("category", "Unknown")
                            category_counts[category] = (
                                category_counts.get(category, 0) + 1
                            )

                        if category_counts:
                            st.bar_chart(
                                pd.DataFrame(
                                    list(category_counts.items()),
                                    columns=["Category", "Count"],
                                ).set_index("Category")
                            )

            else:
                st.success(
                    "🎉 Excellent! No compliance issues found in any of the uploaded documents!"
                )

            # Generate Comprehensive Report
            st.markdown("### 📋 Comprehensive Compliance Report")

            # Create structured summary
            comprehensive_summary = {
                "analysis_metadata": {
                    "process": process_guess,
                    "process_confidence": process_confidence,
                    "documents_uploaded": len(uploaded_files),
                    "required_documents": len(
                        get_required_docs_for_process(process_guess)
                    ),
                    "processing_time_seconds": total_processing_time,
                    "analysis_timestamp": datetime.datetime.now().isoformat(),
                },
                "completeness_analysis": {
                    "missing_documents": completeness_analysis["missing"],
                    "present_documents": completeness_analysis.get("present", []),
                    "extra_documents": completeness_analysis.get("extra", []),
                    "completeness_percentage": completeness_analysis[
                        "completeness_percentage"
                    ],
                    "weighted_completeness": completeness_analysis[
                        "weighted_completeness"
                    ],
                    "missing_by_priority": completeness_analysis.get(
                        "missing_by_priority", {}
                    ),
                },
                "compliance_summary": {
                    "total_issues": total_issues,
                    "average_compliance_score": avg_compliance,
                    "issues_by_severity": {
                        "high": len(high_issues),
                        "medium": len(medium_issues),
                        "low": len(low_issues),
                    },
                    "issues_by_category": (
                        category_counts if "category_counts" in locals() else {}
                    ),
                },
                "document_analysis": summary_data,
                "detailed_issues": all_flagged_sections,
                "recommendations": gap_analysis.get("recommendations", []),
                "next_steps": gap_analysis.get("next_steps", []),
                "alternative_processes": alternative_processes,
                "combination_issues": combination_issues,
                "process_validation": process_validation,
            }

            # Display JSON summary
            st.json(comprehensive_summary)

            # Download Section
            st.markdown("### 📥 Download Results")

            download_cols = st.columns([1, 1, 1])

            with download_cols[0]:
                st.markdown("**📄 Reviewed Documents:**")
                for doc in doc_results:
                    if doc.get("marked_docx") and os.path.exists(doc["marked_docx"]):
                        with open(doc["marked_docx"], "rb") as f:
                            st.download_button(
                                label=f"📄 {doc['filename']}",
                                data=f.read(),
                                file_name=f"reviewed_{doc['filename']}",
                                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                                key=f"download_{doc['filename']}",
                            )

            with download_cols[1]:
                st.markdown("**📊 Analysis Reports:**")

                # Save comprehensive summary
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                summary_filename = f"adgm_analysis_{process_guess.replace(' ', '_').lower()}_{timestamp}.json"
                summary_path = os.path.join(PROCESSED_DIR, summary_filename)

                with open(summary_path, "w", encoding="utf-8") as f:
                    json.dump(comprehensive_summary, f, indent=2, ensure_ascii=False)

                with open(summary_path, "r", encoding="utf-8") as f:
                    st.download_button(
                        label="📊 Complete Analysis Report",
                        data=f.read(),
                        file_name=f"adgm_analysis_{timestamp}.json",
                        mime="application/json",
                        key="download_summary",
                    )

            with download_cols[2]:
                st.markdown("**📋 Quick Summaries:**")

                # Create quick summary CSV
                summary_csv = pd.DataFrame(summary_data).to_csv(index=False)
                st.download_button(
                    label="📋 Document Summary (CSV)",
                    data=summary_csv,
                    file_name=f"document_summary_{timestamp}.csv",
                    mime="text/csv",
                    key="download_csv",
                )

                # Create issues summary
                if all_flagged_sections:
                    issues_summary = []
                    for issue in all_flagged_sections:
                        issues_summary.append(
                            {
                                "Severity": issue.get("severity", "Unknown"),
                                "Category": issue.get("category", "Unknown"),
                                "Issue": issue.get("red_flag", "Unknown"),
                                "Citation": issue.get("law_citation", "Unknown"),
                            }
                        )

                    issues_csv = pd.DataFrame(issues_summary).to_csv(index=False)
                    st.download_button(
                        label="🚨 Issues Summary (CSV)",
                        data=issues_csv,
                        file_name=f"issues_summary_{timestamp}.csv",
                        mime="text/csv",
                        key="download_issues",
                    )

            # Error handling display
            if error_docs:
                st.markdown("### ❌ Processing Errors")
                st.error("Some documents could not be processed:")
                for filename, error in error_docs:
                    st.error(f"• **{filename}**: {error}")

else:
    # Instructions for when no files are uploaded
    st.markdown("## 📋 Getting Started")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown(
            """
        ### How It Works:
        
        1. **📤 Upload Documents**: Select your ADGM legal documents (.docx format)
        2. **🤖 AI Analysis**: Our system automatically:
           - Detects document types using advanced pattern recognition
           - Identifies the legal process (incorporation, licensing, etc.)
           - Analyzes each document for ADGM compliance issues
           - Uses official ADGM sources for accurate legal validation
        
        3. **📊 Get Results**: Receive:
           - Comprehensive compliance reports
           - Documents with inline legal comments
           - Missing document notifications
           - Structured JSON analysis
           - Actionable recommendations
        
        4. **📥 Download**: Get marked-up documents and detailed reports
        """
        )

    with col2:
        st.markdown(
            """
        ### ✨ Key Features:
        
        - **🎯 Process Detection**: Automatically identifies legal processes
        - **📋 Completeness Check**: Compares against ADGM checklists  
        - **⚖️ Compliance Analysis**: Flags jurisdiction and regulatory issues
        - **📝 Inline Comments**: Adds suggestions directly in documents
        - **🔗 Legal Citations**: References exact ADGM regulations
        - **📊 Detailed Reports**: Comprehensive JSON and CSV outputs
        - **🚀 Batch Processing**: Handle multiple documents simultaneously
        """
        )

    # Sample supported documents
    st.markdown("### 📄 Supported Document Types")

    tabs = st.tabs(
        [
            "Company Formation",
            "Licensing",
            "Employment",
            "Commercial",
            "Compliance",
            "Financial Services",
        ]
    )

    with tabs[0]:
        st.markdown("**Company Incorporation & Formation (16 documents):**")
        company_docs = PROCESS_TO_DOCS["Company Incorporation"]
        for i, doc in enumerate(company_docs[:10], 1):
            priority = get_document_priority("Company Incorporation", doc)
            priority_emoji = {
                "Critical": "🔴",
                "Important": "🟡",
                "Optional": "🟢",
            }.get(priority, "⚪")
            st.markdown(f"{i}. {priority_emoji} {doc}")
        if len(company_docs) > 10:
            st.markdown(f"*... and {len(company_docs) - 10} more documents*")

    with tabs[1]:
        st.markdown("**Licensing & Regulatory (14 documents):**")
        licensing_docs = PROCESS_TO_DOCS["Licensing"]
        for i, doc in enumerate(licensing_docs[:8], 1):
            st.markdown(f"{i}. • {doc}")
        if len(licensing_docs) > 8:
            st.markdown(f"*... and {len(licensing_docs) - 8} more documents*")

    with tabs[2]:
        st.markdown("**Employment & HR (15 documents):**")
        employment_docs = PROCESS_TO_DOCS["Employment"]
        for i, doc in enumerate(employment_docs[:8], 1):
            st.markdown(f"{i}. • {doc}")
        if len(employment_docs) > 8:
            st.markdown(f"*... and {len(employment_docs) - 8} more documents*")

    with tabs[3]:
        st.markdown("**Commercial Agreements (15 documents):**")
        commercial_docs = PROCESS_TO_DOCS["Commercial"]
        for i, doc in enumerate(commercial_docs[:8], 1):
            st.markdown(f"{i}. • {doc}")
        if len(commercial_docs) > 8:
            st.markdown(f"*... and {len(commercial_docs) - 8} more documents*")

    with tabs[4]:
        st.markdown("**Compliance & Risk (20 documents):**")
        compliance_docs = PROCESS_TO_DOCS["Compliance"]
        for i, doc in enumerate(compliance_docs[:8], 1):
            st.markdown(f"{i}. • {doc}")
        if len(compliance_docs) > 8:
            st.markdown(f"*... and {len(compliance_docs) - 8} more documents*")

    with tabs[5]:
        st.markdown("**Financial Services (12 documents):**")
        fs_docs = PROCESS_TO_DOCS["Financial Services"]
        for i, doc in enumerate(fs_docs[:8], 1):
            st.markdown(f"{i}. • {doc}")
        if len(fs_docs) > 8:
            st.markdown(f"*... and {len(fs_docs) - 8} more documents*")

# Footer with system information
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**🤖 AI Powered**")
    st.markdown("Google Gemini 2.0 Flash")
    st.markdown("RAG with official ADGM sources")

with col2:
    st.markdown("**⚖️ Legal Accuracy**")
    st.markdown("Official ADGM data sources")
    st.markdown("Current regulations & templates")

with col3:
    st.markdown("**🔒 Compliance**")
    st.markdown("ADGM jurisdiction focus")
    st.markdown("Structured legal analysis")

st.markdown(
    """
<div style='text-align: center; color: #666; margin-top: 2rem;'>
<small>
ADGM Corporate Agent | Enhanced Document Intelligence<br>
</small>
</div>
""",
    unsafe_allow_html=True,
)
