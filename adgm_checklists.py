# Enhanced ADGM Document Types and Checklists
# Comprehensive mappings for document type detection and required checklists for all ADGM processes

import logging
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# Core Company Formation Documents - Enhanced with priorities
COMPANY_INCORPORATION_DOCS = [
    "Articles of Association (AoA)",
    "Memorandum of Association (MoA/MoU)",
    "Board Resolution (for Incorporation)",
    "Shareholder Resolution (for Incorporation)",
    "Incorporation Application Form",
    "Ultimate Beneficial Owner (UBO) Declaration Form",
    "Register of Members and Directors",
    "Change of Registered Address Notice",
    "Identification documents for Authorised Signatories and Directors",
    "Source of Wealth Declaration",
    "Hub71 Approval Letter (if applicable)",
    "Lease Agreement or Office Space Registration",
    "Audited Annual Accounts",
    "Auditors Report",
    "Director's Report",
    "Board Resolution approving accounts",
]

# Branch Company Formation Documents
BRANCH_COMPANY_DOCS = [
    "Parent Company Incorporation Documents",
    "Power of Attorney for Local Representative",
    "Board Resolution from Parent Company",
    "Financial Statements of Parent Company",
    "Business Plan for ADGM Operations",
    "Local Representative Appointment Documents",
    "Registered Office Lease Agreement in ADGM",
    "Certificate of Good Standing from Home Jurisdiction",
    "Apostilled Documents (if required)",
]

# Licensing and Regulatory Documents - Enhanced
LICENSING_DOCS = [
    "License Application Form",
    "Regulatory Approval",
    "Supporting Documents for Licensing",
    "Financial Services Approval (if applicable)",
    "Supplementary Information for Sensitive Terms",
    "Business Plan",
    "Financial Projections",
    "Compliance Manual",
    "Risk Management Policy",
    "Anti-Money Laundering Policy",
    "Know Your Customer Policy",
    "Professional Indemnity Insurance",
    "Key Personnel CVs and Qualifications",
    "Regulatory Reference Letters",
]

# Employment and HR Documents - Complete set
EMPLOYMENT_DOCS = [
    "Standard Employment Contract (2024 update)",
    "Standard Employment Contract (2019 short version)",
    "Employee Handbook",
    "Offer Letter Template",
    "NOC for Employee Transfer (if applicable)",
    "Employment Policy",
    "Code of Conduct",
    "Disciplinary Policy",
    "Grievance Policy",
    "Leave Policy",
    "Health and Safety Policy",
    "Equal Opportunities Policy",
    "Data Protection Policy for Employees",
    "Confidentiality Agreement",
    "Non-Compete Agreement (if applicable)",
]

# Commercial Agreements - Comprehensive list
COMMERCIAL_AGREEMENTS_DOCS = [
    "Shareholder Agreement (SHA)",
    "Non-Disclosure Agreement (NDA)",
    "Consultancy Agreement",
    "Commercial Lease Agreement",
    "Service Agreement",
    "Joint Venture Agreement",
    "Distribution Agreement",
    "Franchise Agreement",
    "Supply Agreement",
    "Partnership Agreement",
    "Licensing Agreement",
    "Technology Transfer Agreement",
    "Management Services Agreement",
    "Outsourcing Agreement",
    "Subscription Agreement",
]

# Compliance and Risk Policies - Full compliance framework
COMPLIANCE_RISK_POLICIES_DOCS = [
    "Risk Policy Statement",
    "Annual Accounts",
    "Appropriate Policy Document (Data Protection)",
    "KYC Policy",
    "AML Policy",
    "IT/Cybersecurity Policy",
    "Whistleblowing Policy",
    "Corporate Governance Policy",
    "ESG Policy",
    "Compliance Manual",
    "Business Continuity Plan",
    "Incident Response Plan",
    "Regulatory Compliance Policy",
    "Record Keeping Policy",
    "Conflict of Interest Policy",
    "Anti-Bribery and Corruption Policy",
    "Sanctions Compliance Policy",
    "Market Abuse Policy",
    "Client Assets Policy",
    "Outsourcing Policy",
]

# Letters and Permits - All types
LETTERS_PERMITS_DOCS = [
    "Application for Official Letters",
    "Event/Training/Seminar Permit Application",
    "NOC Application",
    "Certificate Request",
    "Letter of Good Standing Request",
    "Certificate of Incorporation Request",
    "Certified True Copy Request",
    "Apostille Service Request",
    "Name Reservation Request",
    "Certificate of Amendment Request",
]

# Financial Services Documents - Specialized for FS entities
FINANCIAL_SERVICES_DOCS = [
    "FSRA License Application",
    "Programme Document",
    "Operating Manual",
    "Internal Audit Manual",
    "Client Assets Rules Compliance",
    "Prudential Returns",
    "Capital Adequacy Assessment",
    "Liquidity Management Policy",
    "Investment Committee Charter",
    "Trading Policy",
    "Valuation Policy",
    "Client Onboarding Procedures",
]

# Enhanced process detection with weighted keywords
PROCESS_KEYWORDS = {
    "Company Incorporation": {
        "primary": [
            "incorporation",
            "company formation",
            "articles of association",
            "memorandum of association",
        ],
        "secondary": [
            "board resolution",
            "shareholder resolution",
            "UBO declaration",
            "register of members",
        ],
        "weight": {"primary": 3, "secondary": 2},
    },
    "Branch Company Setup": {
        "primary": [
            "branch",
            "parent company",
            "power of attorney",
            "local representative",
        ],
        "secondary": ["home jurisdiction", "certificate of good standing"],
        "weight": {"primary": 3, "secondary": 2},
    },
    "Licensing": {
        "primary": [
            "license application",
            "regulatory approval",
            "FSRA",
            "financial services",
        ],
        "secondary": ["business license", "operating license", "regulatory compliance"],
        "weight": {"primary": 3, "secondary": 2},
    },
    "Employment": {
        "primary": ["employment contract", "employee handbook", "employment policy"],
        "secondary": ["offer letter", "code of conduct", "disciplinary policy"],
        "weight": {"primary": 3, "secondary": 2},
    },
    "Commercial": {
        "primary": ["shareholder agreement", "commercial lease", "joint venture"],
        "secondary": [
            "consultancy agreement",
            "service agreement",
            "distribution agreement",
        ],
        "weight": {"primary": 3, "secondary": 2},
    },
    "Compliance": {
        "primary": ["risk policy", "compliance manual", "AML policy", "KYC policy"],
        "secondary": ["data protection", "cybersecurity", "whistleblowing"],
        "weight": {"primary": 3, "secondary": 2},
    },
    "Letters/Permits": {
        "primary": [
            "official letter",
            "permit application",
            "NOC",
            "certificate request",
        ],
        "secondary": ["good standing", "apostille", "name reservation"],
        "weight": {"primary": 3, "secondary": 2},
    },
    "Financial Services": {
        "primary": ["FSRA license", "programme document", "operating manual"],
        "secondary": ["client assets", "prudential returns", "capital adequacy"],
        "weight": {"primary": 3, "secondary": 2},
    },
}

# Complete process to documents mapping
PROCESS_TO_DOCS = {
    "Company Incorporation": COMPANY_INCORPORATION_DOCS,
    "Branch Company Setup": BRANCH_COMPANY_DOCS,
    "Licensing": LICENSING_DOCS,
    "Employment": EMPLOYMENT_DOCS,
    "Commercial": COMMERCIAL_AGREEMENTS_DOCS,
    "Compliance": COMPLIANCE_RISK_POLICIES_DOCS,
    "Letters/Permits": LETTERS_PERMITS_DOCS,
    "Financial Services": FINANCIAL_SERVICES_DOCS,
}

# Document priority classifications with detailed mappings
DOCUMENT_PRIORITIES = {
    "Company Incorporation": {
        "Critical": [
            "Articles of Association (AoA)",
            "Memorandum of Association (MoA/MoU)",
            "Incorporation Application Form",
            "Ultimate Beneficial Owner (UBO) Declaration Form",
        ],
        "Important": [
            "Board Resolution (for Incorporation)",
            "Shareholder Resolution (for Incorporation)",
            "Register of Members and Directors",
            "Identification documents for Authorised Signatories and Directors",
            "Lease Agreement or Office Space Registration",
        ],
        "Optional": [
            "Hub71 Approval Letter (if applicable)",
            "Source of Wealth Declaration",
            "Change of Registered Address Notice",
        ],
    },
    "Branch Company Setup": {
        "Critical": [
            "Parent Company Incorporation Documents",
            "Power of Attorney for Local Representative",
            "Board Resolution from Parent Company",
        ],
        "Important": [
            "Financial Statements of Parent Company",
            "Business Plan for ADGM Operations",
            "Local Representative Appointment Documents",
        ],
        "Optional": [
            "Certificate of Good Standing from Home Jurisdiction",
            "Apostilled Documents (if required)",
        ],
    },
    "Licensing": {
        "Critical": [
            "License Application Form",
            "Regulatory Approval",
            "Business Plan",
        ],
        "Important": [
            "Supporting Documents for Licensing",
            "Compliance Manual",
            "Risk Management Policy",
        ],
        "Optional": [
            "Financial Services Approval (if applicable)",
            "Supplementary Information for Sensitive Terms",
        ],
    },
    "Employment": {
        "Critical": ["Standard Employment Contract (2024 update)", "Employee Handbook"],
        "Important": ["Offer Letter Template", "Employment Policy", "Code of Conduct"],
        "Optional": [
            "NOC for Employee Transfer (if applicable)",
            "Equal Opportunities Policy",
        ],
    },
    "Financial Services": {
        "Critical": [
            "FSRA License Application",
            "Programme Document",
            "Operating Manual",
        ],
        "Important": [
            "Internal Audit Manual",
            "Client Assets Rules Compliance",
            "Capital Adequacy Assessment",
        ],
        "Optional": ["Trading Policy", "Valuation Policy"],
    },
}

# Enhanced document type aliases for better detection
DOCUMENT_ALIASES = {
    "Articles of Association (AoA)": [
        "articles",
        "AOA",
        "constitutional document",
        "company constitution",
    ],
    "Memorandum of Association (MoA/MoU)": [
        "memorandum",
        "MOA",
        "MOU",
        "memorandum of understanding",
    ],
    "Board Resolution (for Incorporation)": [
        "board resolution",
        "directors resolution",
        "board meeting minutes",
    ],
    "Shareholder Resolution (for Incorporation)": [
        "shareholders resolution",
        "members resolution",
        "AGM resolution",
    ],
    "Ultimate Beneficial Owner (UBO) Declaration Form": [
        "UBO",
        "beneficial owner",
        "ultimate owner",
        "ownership declaration",
    ],
    "Standard Employment Contract (2024 update)": [
        "employment contract",
        "employment agreement",
        "service contract",
    ],
    "Shareholder Agreement (SHA)": [
        "SHA",
        "shareholders agreement",
        "investment agreement",
    ],
    "Non-Disclosure Agreement (NDA)": [
        "NDA",
        "confidentiality agreement",
        "secrecy agreement",
    ],
    "Risk Policy Statement": ["risk policy", "risk management", "risk framework"],
    "KYC Policy": ["know your customer", "customer due diligence", "CDD policy"],
    "AML Policy": [
        "anti-money laundering",
        "money laundering prevention",
        "AML procedures",
    ],
}


def get_required_docs_for_process(process_name: str) -> List[str]:
    """Get required documents for a specific process"""
    return PROCESS_TO_DOCS.get(process_name, [])


def get_all_document_types() -> List[str]:
    """Get all document types across all processes"""
    types = set()
    for docs in PROCESS_TO_DOCS.values():
        types.update(docs)
    return list(types)


def detect_process_from_documents(doc_types: List[str]) -> str:
    """Enhanced process detection with weighted scoring"""
    process_scores = defaultdict(float)

    for process, keywords in PROCESS_KEYWORDS.items():
        score = 0.0

        for doc_type in doc_types:
            doc_lower = doc_type.lower()

            # Check primary keywords
            for keyword in keywords["primary"]:
                if keyword.lower() in doc_lower:
                    score += keywords["weight"]["primary"]

            # Check secondary keywords
            for keyword in keywords["secondary"]:
                if keyword.lower() in doc_lower:
                    score += keywords["weight"]["secondary"]

            # Check document aliases
            for canonical_name, aliases in DOCUMENT_ALIASES.items():
                if canonical_name == doc_type:
                    # Direct match gets highest score
                    score += 5.0
                    break
                for alias in aliases:
                    if alias.lower() in doc_lower:
                        score += 2.0
                        break

        process_scores[process] = score

    # Return process with highest score, with minimum threshold
    if process_scores:
        best_process = max(process_scores.items(), key=lambda x: x[1])
        if best_process[1] >= 2.0:  # Minimum confidence threshold
            return best_process[0]

    return "Unknown"


def get_missing_documents_for_process(
    process_name: str, uploaded_docs: List[str]
) -> Dict[str, Any]:
    """Enhanced missing document analysis with detailed breakdown"""
    required_docs = get_required_docs_for_process(process_name)
    uploaded_set = set(uploaded_docs)
    required_set = set(required_docs)

    missing = required_set - uploaded_set
    extra = uploaded_set - required_set
    present = uploaded_set & required_set

    # Categorize missing documents by priority
    missing_by_priority = {"Critical": [], "Important": [], "Optional": []}

    for doc in missing:
        priority = get_document_priority(process_name, doc)
        missing_by_priority[priority].append(doc)

    # Calculate weighted completeness
    total_weight = 0
    achieved_weight = 0

    for doc in required_docs:
        priority = get_document_priority(process_name, doc)
        weight = {"Critical": 3, "Important": 2, "Optional": 1}.get(priority, 2)
        total_weight += weight
        if doc in uploaded_set:
            achieved_weight += weight

    weighted_completeness = (
        (achieved_weight / total_weight * 100) if total_weight > 0 else 0
    )

    return {
        "missing": list(missing),
        "extra": list(extra),
        "present": list(present),
        "completeness_percentage": (
            (len(present) / len(required_set)) * 100 if required_set else 0
        ),
        "weighted_completeness": weighted_completeness,
        "missing_by_priority": missing_by_priority,
        "process_confidence": calculate_process_confidence(uploaded_docs, process_name),
        "alternative_processes": suggest_alternative_processes(
            uploaded_docs, process_name
        ),
    }


def get_document_priority(process_name: str, doc_type: str) -> str:
    """Get priority level for documents with enhanced mapping"""
    priorities = DOCUMENT_PRIORITIES.get(process_name, {})

    for priority, docs in priorities.items():
        if doc_type in docs:
            return priority

    # Default priority based on document type
    if any(
        keyword in doc_type.lower()
        for keyword in ["articles", "memorandum", "application", "UBO"]
    ):
        return "Critical"
    elif any(
        keyword in doc_type.lower()
        for keyword in ["resolution", "register", "policy", "manual"]
    ):
        return "Important"
    else:
        return "Optional"


def validate_document_combinations(doc_types: List[str]) -> List[str]:
    """Enhanced validation with comprehensive business logic"""
    issues = []
    doc_set = set(doc_types)

    # Core incorporation document validation
    core_incorporation = [
        "Articles of Association (AoA)",
        "Memorandum of Association (MoA/MoU)",
        "Incorporation Application Form",
    ]

    if any(doc in doc_set for doc in core_incorporation):
        missing_core = [doc for doc in core_incorporation if doc not in doc_set]
        if missing_core:
            issues.append(
                f"Incomplete incorporation package - missing: {', '.join(missing_core)}"
            )

    # Resolution consistency checks
    board_res = "Board Resolution (for Incorporation)" in doc_set
    shareholder_res = "Shareholder Resolution (for Incorporation)" in doc_set

    if board_res and not shareholder_res:
        issues.append(
            "Board Resolution present but corresponding Shareholder Resolution missing"
        )

    if shareholder_res and not board_res:
        issues.append(
            "Shareholder Resolution present but corresponding Board Resolution missing"
        )

    # UBO and register consistency
    ubo_present = "Ultimate Beneficial Owner (UBO) Declaration Form" in doc_set
    register_present = "Register of Members and Directors" in doc_set

    if ubo_present and not register_present:
        issues.append("UBO Declaration requires Register of Members and Directors")

    # Employment document consistency
    employment_docs = [
        doc
        for doc in doc_set
        if "employment" in doc.lower() or "employee" in doc.lower()
    ]
    if employment_docs and len(employment_docs) == 1:
        issues.append(
            "Single employment document detected - consider adding Employee Handbook or Employment Policy"
        )

    # Financial services validation
    fs_docs = [
        doc
        for doc in doc_set
        if any(
            fs_keyword in doc.lower()
            for fs_keyword in ["fsra", "financial services", "prudential"]
        )
    ]
    if fs_docs:
        required_fs = [
            "FSRA License Application",
            "Operating Manual",
            "Programme Document",
        ]
        missing_fs = [doc for doc in required_fs if doc not in doc_set]
        if missing_fs:
            issues.append(f"Financial Services setup requires: {', '.join(missing_fs)}")

    # Branch company validation
    if "Parent Company Incorporation Documents" in doc_set:
        branch_required = [
            "Power of Attorney for Local Representative",
            "Board Resolution from Parent Company",
        ]
        missing_branch = [doc for doc in branch_required if doc not in doc_set]
        if missing_branch:
            issues.append(f"Branch setup requires: {', '.join(missing_branch)}")

    return issues


def calculate_process_confidence(
    uploaded_docs: List[str], detected_process: str
) -> float:
    """Calculate confidence level for process detection"""
    if not uploaded_docs or detected_process == "Unknown":
        return 0.0

    required_docs = get_required_docs_for_process(detected_process)
    if not required_docs:
        return 0.0

    # Calculate based on critical documents present
    priorities = DOCUMENT_PRIORITIES.get(detected_process, {})
    critical_docs = priorities.get("Critical", [])

    uploaded_set = set(uploaded_docs)
    critical_present = len([doc for doc in critical_docs if doc in uploaded_set])
    total_critical = len(critical_docs)

    if total_critical == 0:
        return 0.5  # Medium confidence if no critical docs defined

    return min(1.0, critical_present / total_critical)


def suggest_alternative_processes(
    uploaded_docs: List[str], current_process: str
) -> List[Dict[str, Any]]:
    """Suggest alternative processes based on uploaded documents"""
    alternatives = []

    for process in PROCESS_TO_DOCS.keys():
        if process != current_process:
            score = 0
            required_docs = get_required_docs_for_process(process)
            uploaded_set = set(uploaded_docs)

            # Calculate overlap score
            overlap = len(uploaded_set.intersection(set(required_docs)))
            if overlap > 0:
                relevance = overlap / len(required_docs) if required_docs else 0
                alternatives.append(
                    {
                        "process": process,
                        "relevance": relevance,
                        "overlapping_docs": overlap,
                        "confidence": calculate_process_confidence(
                            uploaded_docs, process
                        ),
                    }
                )

    # Sort by relevance and return top 3
    alternatives.sort(key=lambda x: x["relevance"], reverse=True)
    return alternatives[:3]


def get_process_description(process_name: str) -> str:
    """Enhanced process descriptions with more detail"""
    descriptions = {
        "Company Incorporation": "Complete process for incorporating a new company in ADGM, including constitutional documents, resolutions, regulatory filings, and compliance requirements.",
        "Branch Company Setup": "Process for establishing a branch office of a foreign company in ADGM, requiring parent company documentation and local representation.",
        "Licensing": "Process for obtaining business licenses and regulatory approvals in ADGM, including specialized financial services licenses.",
        "Employment": "Process for establishing employment relationships in ADGM, including contracts, policies, and compliance with employment regulations.",
        "Commercial": "Process for establishing commercial relationships and agreements in ADGM, including partnerships, service contracts, and commercial arrangements.",
        "Compliance": "Process for establishing comprehensive compliance frameworks, risk management policies, and regulatory compliance in ADGM.",
        "Letters/Permits": "Process for obtaining official letters, permits, certificates, and other documentary requirements from ADGM authorities.",
        "Financial Services": "Specialized process for financial services entities, including FSRA licensing, prudential requirements, and regulatory compliance.",
    }
    return descriptions.get(process_name, "Unknown process type")


def get_document_category_mapping() -> Dict[str, str]:
    """Get mapping of documents to their categories for better organization"""
    mapping = {}

    for process, docs in PROCESS_TO_DOCS.items():
        for doc in docs:
            if doc not in mapping:  # First occurrence wins
                mapping[doc] = process

    return mapping


def analyze_document_gaps(
    uploaded_docs: List[str], target_process: str
) -> Dict[str, Any]:
    """Comprehensive gap analysis for document completeness"""
    required_docs = get_required_docs_for_process(target_process)
    uploaded_set = set(uploaded_docs)
    required_set = set(required_docs)

    gaps = {
        "critical_gaps": [],
        "important_gaps": [],
        "optional_gaps": [],
        "recommendations": [],
        "next_steps": [],
    }

    # Analyze gaps by priority
    for doc in required_set - uploaded_set:
        priority = get_document_priority(target_process, doc)
        if priority == "Critical":
            gaps["critical_gaps"].append(doc)
        elif priority == "Important":
            gaps["important_gaps"].append(doc)
        else:
            gaps["optional_gaps"].append(doc)

    # Generate recommendations
    if gaps["critical_gaps"]:
        gaps["recommendations"].append(
            "Immediate action required: Prepare critical missing documents"
        )
        gaps["next_steps"].append(
            "Focus on preparing Articles of Association and Memorandum of Association first"
        )

    if gaps["important_gaps"]:
        gaps["recommendations"].append(
            "Important documents missing: Plan to prepare within 2-3 business days"
        )
        gaps["next_steps"].append(
            "Schedule board/shareholder meetings for required resolutions"
        )

    if not gaps["critical_gaps"] and not gaps["important_gaps"]:
        gaps["recommendations"].append(
            "Core documentation complete! Consider optional enhancements"
        )
        gaps["next_steps"].append(
            "Review optional documents and prepare for submission"
        )

    return gaps


def validate_process_requirements(
    doc_types: List[str], process_name: str
) -> Dict[str, Any]:
    """Validate that uploaded documents meet specific process requirements"""
    validation_rules = {
        "Company Incorporation": {
            "minimum_docs": 4,
            "required_types": ["constitutional", "resolution", "application"],
            "mandatory_combinations": [
                ("Articles of Association (AoA)", "Memorandum of Association (MoA/MoU)")
            ],
        },
        "Financial Services": {
            "minimum_docs": 3,
            "required_types": ["license", "manual", "policy"],
            "mandatory_combinations": [
                ("FSRA License Application", "Operating Manual")
            ],
        },
        "Employment": {
            "minimum_docs": 2,
            "required_types": ["contract", "policy"],
            "mandatory_combinations": [],
        },
    }

    rules = validation_rules.get(process_name, {})
    results = {"valid": True, "issues": [], "warnings": [], "recommendations": []}

    # Check minimum document count
    min_docs = rules.get("minimum_docs", 1)
    if len(doc_types) < min_docs:
        results["valid"] = False
        results["issues"].append(
            f"Insufficient documents: {len(doc_types)} provided, minimum {min_docs} required"
        )

    # Check mandatory combinations
    for combo in rules.get("mandatory_combinations", []):
        if not all(doc in doc_types for doc in combo):
            results["valid"] = False
            results["issues"].append(
                f"Missing mandatory combination: {' and '.join(combo)}"
            )

    return results


# Export all functions
__all__ = [
    "get_required_docs_for_process",
    "get_all_document_types",
    "detect_process_from_documents",
    "get_missing_documents_for_process",
    "get_document_priority",
    "validate_document_combinations",
    "get_process_description",
    "calculate_process_confidence",
    "suggest_alternative_processes",
    "analyze_document_gaps",
    "validate_process_requirements",
    "PROCESS_TO_DOCS",
    "DOCUMENT_PRIORITIES",
    "DOCUMENT_ALIASES",
]
