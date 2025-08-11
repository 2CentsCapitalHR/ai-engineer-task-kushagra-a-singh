"""
Configuration file for ADGM Corporate Agent
Centralizes all settings, parameters, and constants
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_MODEL = os.getenv("GOOGLE_MODEL", "gemini-2.0-flash")

# Model Configuration
LLM_TEMPERATURE = 0.1
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RAG_TOP_K = 3

# File Processing Configuration
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
MAX_FILES_PER_UPLOAD = 10
SUPPORTED_FORMATS = [".docx"]

# Document Analysis Configuration
MIN_SECTION_LENGTH = 10  # Minimum text length for analysis
MAX_SECTIONS_PER_DOC = 50  # Maximum sections to analyze per document

# UI Configuration
PAGE_TITLE = "ADGM Corporate Agent v2.0"
PAGE_ICON = "⚖️"
LAYOUT = "wide"
INITIAL_SIDEBAR_STATE = "expanded"

# File Paths
UPLOAD_DIR = Path(__file__).parent / "uploads"
PROCESSED_DIR = Path(__file__).parent / "processed"
CHROMA_DB_DIR = Path(__file__).parent / "chroma_db"

# Create directories if they don't exist
for directory in [UPLOAD_DIR, PROCESSED_DIR, CHROMA_DB_DIR]:
    directory.mkdir(exist_ok=True)

# Severity Levels
SEVERITY_LEVELS = ["High", "Medium", "Low"]
SEVERITY_COLORS = {
    "High": "#FF0000",  # Red
    "Medium": "#FFA500",  # Orange
    "Low": "#FFFF00",  # Yellow
}

# Document Priority Levels
PRIORITY_LEVELS = ["Critical", "Important", "Optional"]
PRIORITY_COLORS = {
    "Critical": "#FF0000",  # Red
    "Important": "#FFA500",  # Orange
    "Optional": "#008000",  # Green
}

# Issue Categories
ISSUE_CATEGORIES = [
    "jurisdiction",
    "missing_clauses",
    "formatting",
    "compliance",
    "ambiguity",
    "outdated",
]

# Process Types
PROCESS_TYPES = [
    "Company Incorporation",
    "Licensing",
    "Employment",
    "Commercial",
    "Compliance",
    "Letters/Permits",
]

# Error Messages
ERROR_MESSAGES = {
    "file_too_large": "File size exceeds maximum limit of 10MB",
    "unsupported_format": "Unsupported file format. Only .docx files are supported",
    "too_many_files": "Too many files uploaded. Maximum 10 files allowed",
    "processing_error": "Error processing document",
    "api_error": "API error occurred during analysis",
    "save_error": "Error saving processed document",
}

# Success Messages
SUCCESS_MESSAGES = {
    "upload_success": "Files uploaded successfully",
    "processing_complete": "Document processing completed",
    "analysis_complete": "Analysis completed successfully",
    "download_ready": "Download ready",
}

# Validation Rules
VALIDATION_RULES = {
    "min_document_length": 100,  # Minimum document length in characters
    "required_sections": {
        "Articles of Association (AoA)": [
            "JURISDICTION",
            "OBJECTS",
            "SHARE CAPITAL",
            "DIRECTORS",
        ],
        "Memorandum of Association (MoA/MoU)": [
            "COMPANY NAME",
            "REGISTERED OFFICE",
            "OBJECTS",
            "LIABILITY",
        ],
        "Board Resolution (for Incorporation)": ["RESOLUTION", "SIGNATURES"],
        "Employment Contract": [
            "PARTIES",
            "TERMS OF EMPLOYMENT",
            "GOVERNING LAW",
            "SIGNATURES",
        ],
    },
}

# RAG Configuration
RAG_CONFIG = {
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "similarity_threshold": 0.7,
    "max_context_length": 4000,
}

# Analysis Configuration
ANALYSIS_CONFIG = {
    "max_issues_per_document": 20,
    "min_confidence_score": 0.6,
    "enable_structure_validation": True,
    "enable_citation_generation": True,
    "enable_suggestion_generation": True,
}

# Export Configuration
EXPORT_CONFIG = {
    "json_indent": 2,
    "include_metadata": True,
    "include_timestamps": True,
    "compress_output": False,
}

# Logging Configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": "adgm_agent.log",
}

# Performance Configuration
PERFORMANCE_CONFIG = {
    "enable_caching": True,
    "cache_ttl": 3600,  # 1 hour
    "max_concurrent_requests": 5,
    "request_timeout": 30,
}

# Security Configuration
SECURITY_CONFIG = {
    "allowed_file_types": [".docx"],
    "max_file_size_mb": 10,
    "sanitize_filenames": True,
    "validate_file_content": True,
}

# RAG Configuration
USE_CHROMADB = os.getenv("USE_CHROMADB", "false").lower() == "true"
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "3"))
RAG_SIMILARITY_THRESHOLD = float(os.getenv("RAG_SIMILARITY_THRESHOLD", "0.1"))

# LLM Configuration
GEMINI_TEMPERATURE = float(os.getenv("GEMINI_TEMPERATURE", "0.1"))
GEMINI_MAX_TOKENS = int(os.getenv("GEMINI_MAX_TOKENS", "1024"))

# Processing configuration
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "5"))
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "3"))

# Analysis configuration
ENABLE_STRUCTURE_ANALYSIS = (
    os.getenv("ENABLE_STRUCTURE_ANALYSIS", "true").lower() == "true"
)
ENABLE_COMPLIANCE_ANALYSIS = (
    os.getenv("ENABLE_COMPLIANCE_ANALYSIS", "true").lower() == "true"
)
ENABLE_RED_FLAG_ANALYSIS = (
    os.getenv("ENABLE_RED_FLAG_ANALYSIS", "true").lower() == "true"
)

# Document type confidence thresholds
TYPE_DETECTION_CONFIDENCE_THRESHOLD = float(
    os.getenv("TYPE_DETECTION_CONFIDENCE_THRESHOLD", "0.6")
)

# UI Configuration
SHOW_DEBUG_INFO = os.getenv("SHOW_DEBUG_INFO", "false").lower() == "true"
ENABLE_DOWNLOAD_ALL = os.getenv("ENABLE_DOWNLOAD_ALL", "true").lower() == "true"


def validate_config():
    """Validate configuration settings"""
    errors = []

    if not GOOGLE_API_KEY:
        errors.append("GOOGLE_API_KEY is required")

    if not os.path.exists(UPLOAD_DIR):
        try:
            os.makedirs(UPLOAD_DIR, exist_ok=True)
        except Exception as e:
            errors.append(f"Cannot create upload directory: {e}")

    if not os.path.exists(PROCESSED_DIR):
        try:
            os.makedirs(PROCESSED_DIR, exist_ok=True)
        except Exception as e:
            errors.append(f"Cannot create processed directory: {e}")

    return errors


def get_config_summary() -> Dict[str, Any]:
    """Get a summary of current configuration"""
    return {
        "rag_backend": "ChromaDB" if USE_CHROMADB else "In-Memory",
        "rag_top_k": RAG_TOP_K,
        "max_files": MAX_FILES_PER_UPLOAD,
        "max_file_size_mb": MAX_FILE_SIZE // (1024 * 1024),
        "batch_size": BATCH_SIZE,
        "max_workers": MAX_WORKERS,
        "gemini_temperature": GEMINI_TEMPERATURE,
        "structure_analysis": ENABLE_STRUCTURE_ANALYSIS,
        "compliance_analysis": ENABLE_COMPLIANCE_ANALYSIS,
        "red_flag_analysis": ENABLE_RED_FLAG_ANALYSIS,
        "debug_mode": SHOW_DEBUG_INFO,
    }


def get_rag_config() -> Dict[str, Any]:
    """Get RAG-specific configuration"""
    return {
        "backend": "ChromaDB" if USE_CHROMADB else "In-Memory",
        "top_k": RAG_TOP_K,
        "similarity_threshold": RAG_SIMILARITY_THRESHOLD,
        "persist_directory": str(CHROMA_DB_DIR) if USE_CHROMADB else None,
    }


# Enhanced System Configuration Classes
@dataclass
class RateLimitSettings:
    """Configuration for rate limiting and API management"""

    max_requests_per_minute: int = 15  # Free plan limit
    max_requests_per_second: int = 1  # Conservative approach
    retry_attempts: int = 3
    base_delay: float = 1.0  # Base delay in seconds
    max_delay: float = 60.0  # Maximum delay in seconds
    jitter: bool = True  # Add random jitter to avoid thundering herd
    enable_rate_limiting: bool = True
    enable_retry_logic: bool = True


@dataclass
class BatchingSettings:
    """Configuration for smart batching of document sections"""

    max_sections_per_batch: int = 5  # Maximum sections to combine
    max_total_chars: int = 4000  # Maximum characters per batch
    min_section_length: int = 50  # Minimum section length to include
    priority_sections: List[str] = None  # High-priority section types
    enable_smart_batching: bool = True
    enable_priority_sorting: bool = True
    enable_content_deduplication: bool = True

    def __post_init__(self):
        if self.priority_sections is None:
            self.priority_sections = [
                "RESOLUTION",
                "APPOINTMENT",
                "ARTICLES",
                "SHARE CAPITAL",
                "DIRECTOR",
                "SECRETARY",
                "SIGNATORY",
            ]


@dataclass
class ErrorHandlingSettings:
    """Configuration for error handling and validation"""

    enable_input_validation: bool = True
    enable_graceful_degradation: bool = True
    enable_error_logging: bool = True
    enable_error_recovery: bool = True
    valid_section_types: List[str] = None
    max_validation_errors: int = 10
    enable_fallback_analysis: bool = True

    def __post_init__(self):
        if self.valid_section_types is None:
            self.valid_section_types = ["header", "content", "footer", "table", "list"]


@dataclass
class AnalysisSettings:
    """Configuration for document analysis and processing"""

    enable_structure_analysis: bool = True
    enable_compliance_analysis: bool = True
    enable_red_flag_analysis: bool = True
    enable_citation_generation: bool = True
    enable_suggestion_generation: bool = True
    max_issues_per_document: int = 20
    min_confidence_score: float = 0.6
    enable_batch_processing: bool = True
    enable_async_processing: bool = True


@dataclass
class EnhancedSystemConfig:
    """Main configuration class that aggregates all settings"""

    rate_limits: RateLimitSettings
    batching: BatchingSettings
    error_handling: ErrorHandlingSettings
    analysis: AnalysisSettings

    def __post_init__(self):
        """Validate configuration after initialization"""
        if self.rate_limits.max_requests_per_minute <= 0:
            raise ValueError("max_requests_per_minute must be positive")
        if self.batching.max_sections_per_batch <= 0:
            raise ValueError("max_sections_per_batch must be positive")
        if (
            self.analysis.min_confidence_score < 0
            or self.analysis.min_confidence_score > 1
        ):
            raise ValueError("min_confidence_score must be between 0 and 1")

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "rate_limits": {
                "max_requests_per_minute": self.rate_limits.max_requests_per_minute,
                "max_requests_per_second": self.rate_limits.max_requests_per_second,
                "retry_attempts": self.rate_limits.retry_attempts,
                "base_delay": self.rate_limits.base_delay,
                "max_delay": self.rate_limits.max_delay,
                "jitter": self.rate_limits.jitter,
                "enable_rate_limiting": self.rate_limits.enable_rate_limiting,
                "enable_retry_logic": self.rate_limits.enable_retry_logic,
            },
            "batching": {
                "max_sections_per_batch": self.batching.max_sections_per_batch,
                "max_total_chars": self.batching.max_total_chars,
                "min_section_length": self.batching.min_section_length,
                "priority_sections": self.batching.priority_sections,
                "enable_smart_batching": self.batching.enable_smart_batching,
                "enable_priority_sorting": self.batching.enable_priority_sorting,
                "enable_content_deduplication": self.batching.enable_content_deduplication,
            },
            "error_handling": {
                "enable_input_validation": self.error_handling.enable_input_validation,
                "enable_graceful_degradation": self.error_handling.enable_graceful_degradation,
                "enable_error_logging": self.error_handling.enable_error_logging,
                "enable_error_recovery": self.error_handling.enable_error_recovery,
                "valid_section_types": self.error_handling.valid_section_types,
                "max_validation_errors": self.error_handling.max_validation_errors,
                "enable_fallback_analysis": self.error_handling.enable_fallback_analysis,
            },
            "analysis": {
                "enable_structure_analysis": self.analysis.enable_structure_analysis,
                "enable_compliance_analysis": self.analysis.enable_compliance_analysis,
                "enable_red_flag_analysis": self.analysis.enable_red_flag_analysis,
                "enable_citation_generation": self.analysis.enable_citation_generation,
                "enable_suggestion_generation": self.analysis.enable_suggestion_generation,
                "max_issues_per_document": self.analysis.max_issues_per_document,
                "min_confidence_score": self.analysis.min_confidence_score,
                "enable_batch_processing": self.analysis.enable_batch_processing,
                "enable_async_processing": self.analysis.enable_async_processing,
            },
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "EnhancedSystemConfig":
        """Create configuration from dictionary"""
        return cls(
            rate_limits=RateLimitSettings(**config_dict.get("rate_limits", {})),
            batching=BatchingSettings(**config_dict.get("batching", {})),
            error_handling=ErrorHandlingSettings(
                **config_dict.get("error_handling", {})
            ),
            analysis=AnalysisSettings(**config_dict.get("analysis", {})),
        )

    def get_free_plan_optimized_config(self) -> "EnhancedSystemConfig":
        """Get configuration optimized for Gemini free plan"""
        return EnhancedSystemConfig(
            rate_limits=RateLimitSettings(
                max_requests_per_minute=15,  # Free plan limit
                max_requests_per_second=1,  # Conservative approach
                retry_attempts=3,
                base_delay=2.0,  # Longer delays for free plan
                max_delay=120.0,  # Longer max delay
                jitter=True,
                enable_rate_limiting=True,
                enable_retry_logic=True,
            ),
            batching=BatchingSettings(
                max_sections_per_batch=8,  # More aggressive batching
                max_total_chars=6000,  # Larger batches
                min_section_length=30,  # Include more sections
                priority_sections=self.batching.priority_sections,
                enable_smart_batching=True,
                enable_priority_sorting=True,
                enable_content_deduplication=True,
            ),
            error_handling=ErrorHandlingSettings(
                enable_input_validation=True,
                enable_graceful_degradation=True,
                enable_error_logging=True,
                enable_error_recovery=True,
                valid_section_types=self.error_handling.valid_section_types,
                max_validation_errors=15,
                enable_fallback_analysis=True,
            ),
            analysis=AnalysisSettings(
                enable_structure_analysis=True,
                enable_compliance_analysis=True,
                enable_red_flag_analysis=True,
                enable_citation_generation=True,
                enable_suggestion_generation=True,
                max_issues_per_document=25,
                min_confidence_score=0.5,  # Lower threshold for free plan
                enable_batch_processing=True,
                enable_async_processing=True,
            ),
        )


# Default configuration
DEFAULT_CONFIG = EnhancedSystemConfig(
    rate_limits=RateLimitSettings(),
    batching=BatchingSettings(),
    error_handling=ErrorHandlingSettings(),
    analysis=AnalysisSettings(),
)

# Free plan optimized configuration
FREE_PLAN_CONFIG = DEFAULT_CONFIG.get_free_plan_optimized_config()


def get_enhanced_config() -> EnhancedSystemConfig:
    """Get the current enhanced system configuration"""
    # Check if we should use free plan optimization
    if os.getenv("GEMINI_FREE_PLAN", "true").lower() == "true":
        return FREE_PLAN_CONFIG
    return DEFAULT_CONFIG


def update_config_from_env(config: EnhancedSystemConfig) -> EnhancedSystemConfig:
    """Update configuration from environment variables"""
    # Rate limiting
    if os.getenv("GEMINI_MAX_REQUESTS_PER_MINUTE"):
        config.rate_limits.max_requests_per_minute = int(
            os.getenv("GEMINI_MAX_REQUESTS_PER_MINUTE")
        )

    if os.getenv("GEMINI_RETRY_ATTEMPTS"):
        config.rate_limits.retry_attempts = int(os.getenv("GEMINI_RETRY_ATTEMPTS"))

    # Batching
    if os.getenv("GEMINI_MAX_SECTIONS_PER_BATCH"):
        config.batching.max_sections_per_batch = int(
            os.getenv("GEMINI_MAX_SECTIONS_PER_BATCH")
        )

    if os.getenv("GEMINI_MAX_TOTAL_CHARS"):
        config.batching.max_total_chars = int(os.getenv("GEMINI_MAX_TOTAL_CHARS"))

    # Analysis
    if os.getenv("GEMINI_MIN_CONFIDENCE_SCORE"):
        config.analysis.min_confidence_score = float(
            os.getenv("GEMINI_MIN_CONFIDENCE_SCORE")
        )

    return config
