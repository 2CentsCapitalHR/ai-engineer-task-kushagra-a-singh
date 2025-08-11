"""
Demo script for ADGM Corporate Agent
This script demonstrates the system's capabilities with example documents
"""

import datetime
import json
import os
from pathlib import Path

# Import the main modules
from adgm_checklists import (
    detect_process_from_documents,
    get_all_document_types,
    get_missing_documents_for_process,
    get_required_docs_for_process,
    validate_document_combinations,
)
from adgm_rag import (
    analyze_document_completeness,
    gemini_legal_analysis,
    retrieve_relevant_snippets,
)
from doc_utils import (
    detect_document_type,
    extract_document_metadata,
    extract_text_sections,
    validate_document_structure,
)


def run_demo():
    """Run a comprehensive demo of the ADGM Corporate Agent"""

    print("=" * 60)
    print("🚀 ADGM CORPORATE AGENT - DEMO")
    print("=" * 60)

    # Check if example documents exist
    example_dir = Path("example_documents")
    if not example_dir.exists():
        print(
            "❌ Example documents not found. Please run 'python example_documents.py' first."
        )
        return

    # Get example documents
    doc_files = list(example_dir.glob("*.docx"))
    if not doc_files:
        print("❌ No .docx files found in example_documents directory.")
        return

    print(f"📄 Found {len(doc_files)} example documents:")
    for doc in doc_files:
        print(f"   • {doc.name}")

    print("\n" + "=" * 60)
    print("🔍 DOCUMENT ANALYSIS")
    print("=" * 60)

    # Analyze each document
    all_types = get_all_document_types()
    found_types = []
    doc_results = []

    for doc_file in doc_files:
        print(f"\n📄 Analyzing: {doc_file.name}")

        try:
            # Extract text sections
            sections = extract_text_sections(doc_file)
            print(f"   📝 Extracted {len(sections)} text sections")

            # Detect document type
            doc_type = "Unknown"
            for section in sections[:3]:  # Check first 3 sections
                detected = detect_document_type(section["text"], all_types)
                if detected != "Unknown":
                    doc_type = detected
                    break

            found_types.append(doc_type)
            print(f"   🏷️  Detected type: {doc_type}")

            # Extract metadata
            metadata = extract_document_metadata(doc_file)
            print(
                f"   📊 Metadata: {len(metadata['headers'])} headers, {len(metadata['signatures'])} signatures, {len(metadata['dates'])} dates"
            )

            # Validate structure
            structure_issues = validate_document_structure(doc_type, metadata)
            if structure_issues:
                print(f"   ⚠️  Structure issues: {', '.join(structure_issues)}")
            else:
                print(f"   ✅ Structure validation passed")

            # RAG analysis (simplified for demo)
            rag_refs = retrieve_relevant_snippets(f"ADGM {doc_type} requirements")
            print(f"   🔍 Retrieved {len(rag_refs)} relevant ADGM references")

            doc_results.append(
                {
                    "filename": doc_file.name,
                    "doc_type": doc_type,
                    "metadata": metadata,
                    "structure_issues": structure_issues,
                }
            )

        except Exception as e:
            print(f"   ❌ Error analyzing {doc_file.name}: {str(e)}")

    print("\n" + "=" * 60)
    print("🎯 PROCESS DETECTION")
    print("=" * 60)

    # Detect legal process
    process_guess = detect_process_from_documents(found_types)
    print(f"🔍 Detected legal process: {process_guess}")

    # Analyze completeness
    completeness_analysis = get_missing_documents_for_process(
        process_guess, found_types
    )
    required_docs = get_required_docs_for_process(process_guess)

    print(f"📊 Completeness analysis:")
    print(f"   • Documents uploaded: {len(doc_files)}")
    print(f"   • Required documents: {len(required_docs)}")
    print(f"   • Missing documents: {len(completeness_analysis.get('missing', []))}")

    if completeness_analysis.get("missing"):
        print(f"   ⚠️  Missing: {', '.join(completeness_analysis['missing'])}")

    # Calculate completeness percentage
    uploaded_set = set(found_types)
    required_set = set(required_docs)
    completeness_percentage = (
        (len(uploaded_set & required_set) / len(required_set)) * 100
        if required_set
        else 0
    )
    print(f"   • Completeness: {completeness_percentage:.1f}%")

    # Validate document combinations
    combination_issues = validate_document_combinations(found_types)
    if combination_issues:
        print(f"   ⚠️  Combination issues: {', '.join(combination_issues)}")

    print("\n" + "=" * 60)
    print("📋 DOCUMENT BREAKDOWN")
    print("=" * 60)

    for doc in doc_results:
        print(f"\n📄 {doc['filename']}")
        print(f"   Type: {doc['doc_type']}")
        print(f"   Headers: {len(doc['metadata']['headers'])}")
        print(f"   Signatures: {len(doc['metadata']['signatures'])}")
        print(f"   Dates: {len(doc['metadata']['dates'])}")
        if doc["structure_issues"]:
            print(f"   Issues: {', '.join(doc['structure_issues'])}")

    print("\n" + "=" * 60)
    print("📊 DEMO SUMMARY")
    print("=" * 60)

    summary = {
        "demo_timestamp": datetime.datetime.now().isoformat(),
        "documents_analyzed": len(doc_results),
        "detected_process": process_guess,
        "document_types_found": found_types,
        "completeness_percentage": completeness_percentage,
        "missing_documents": completeness_analysis.get("missing", []),
        "combination_issues": combination_issues,
        "structure_issues": [
            doc["structure_issues"] for doc in doc_results if doc["structure_issues"]
        ],
    }

    print(f"✅ Demo completed successfully!")
    print(f"📊 Analysis summary saved to 'demo_summary.json'")

    # Save demo summary
    with open("demo_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("🎉 DEMO COMPLETE")
    print("=" * 60)
    print("To run the full application with UI:")
    print("   streamlit run app.py")
    print("\nTo test with the example documents:")
    print("   1. Open the Streamlit app")
    print("   2. Upload files from 'example_documents/' directory")
    print("   3. View the comprehensive analysis results")


if __name__ == "__main__":
    run_demo()
