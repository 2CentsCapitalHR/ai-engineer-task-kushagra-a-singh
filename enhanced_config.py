"""
Enhanced Configuration for ADGM Analysis System
Configures rate limiting, batching, and error handling parameters
"""

import os
from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class RateLimitSettings:
    """Rate limiting configuration for free plan"""

    max_requests_per_minute: int = 15  # Free plan limit
    max_requests_per_second: int = 1  # Conservative approach
    retry_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    jitter: bool = True
    enable_rate_limiting: bool = True


@dataclass
class BatchingSettings:
    """Smart batching configuration"""

    max_sections_per_batch: int = 5
    max_total_chars: int = 4000
    min_section_length: int = 50
    enable_smart_batching: bool = True
    priority_sections: list = None

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
                "JURISDICTION",
            ]


@dataclass
class ErrorHandlingSettings:
    """Error handling and recovery configuration"""

    enable_graceful_degradation: bool = True
    enable_fallback_analysis: bool = True
    max_fallback_sections: int = 3
    log_error_details: bool = True
    save_error_reports: bool = True


@dataclass
class AnalysisSettings:
    """Analysis quality and performance settings"""

    enable_rag_enhancement: bool = True
    max_rag_snippets: int = 5
    analysis_temperature: float = 0.1
    max_output_tokens: int = 1024
    batch_analysis_timeout: int = 300  # 5 minutes
    enable_progress_tracking: bool = True


@dataclass
class EnhancedSystemConfig:
    """Complete configuration for enhanced system"""

    # Core settings
    rate_limiting: RateLimitSettings = None
    batching: BatchingSettings = None
    error_handling: ErrorHandlingSettings = None
    analysis: AnalysisSettings = None

    # Feature flags
    enable_enhanced_analysis: bool = True
    enable_smart_caching: bool = False  # For future implementation
    enable_async_processing: bool = True
    enable_batch_optimization: bool = True

    def __post_init__(self):
        if self.rate_limiting is None:
            self.rate_limiting = RateLimitSettings()
        if self.batching is None:
            self.batching = BatchingSettings()
        if self.error_handling is None:
            self.error_handling = ErrorHandlingSettings()
        if self.analysis is None:
            self.analysis = AnalysisSettings()

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "rate_limiting": {
                "max_requests_per_minute": self.rate_limiting.max_requests_per_minute,
                "max_requests_per_second": self.rate_limiting.max_requests_per_second,
                "retry_attempts": self.rate_limiting.retry_attempts,
                "base_delay": self.rate_limiting.base_delay,
                "max_delay": self.rate_limiting.max_delay,
                "jitter": self.rate_limiting.jitter,
                "enable_rate_limiting": self.rate_limiting.enable_rate_limiting,
            },
            "batching": {
                "max_sections_per_batch": self.batching.max_sections_per_batch,
                "max_total_chars": self.batching.max_total_chars,
                "min_section_length": self.batching.min_section_length,
                "enable_smart_batching": self.batching.enable_smart_batching,
                "priority_sections": self.batching.priority_sections,
            },
            "error_handling": {
                "enable_graceful_degradation": self.error_handling.enable_graceful_degradation,
                "enable_fallback_analysis": self.error_handling.enable_fallback_analysis,
                "max_fallback_sections": self.error_handling.max_fallback_sections,
                "log_error_details": self.error_handling.log_error_details,
                "save_error_reports": self.error_handling.save_error_reports,
            },
            "analysis": {
                "enable_rag_enhancement": self.analysis.enable_rag_enhancement,
                "max_rag_snippets": self.analysis.max_rag_snippets,
                "analysis_temperature": self.analysis.analysis_temperature,
                "max_output_tokens": self.analysis.max_output_tokens,
                "batch_analysis_timeout": self.analysis.batch_analysis_timeout,
                "enable_progress_tracking": self.analysis.enable_progress_tracking,
            },
            "features": {
                "enable_enhanced_analysis": self.enable_enhanced_analysis,
                "enable_smart_caching": self.enable_smart_caching,
                "enable_async_processing": self.enable_async_processing,
                "enable_batch_optimization": self.enable_batch_optimization,
            },
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "EnhancedSystemConfig":
        """Create configuration from dictionary"""
        config = cls()

        if "rate_limiting" in config_dict:
            rate_config = config_dict["rate_limiting"]
            config.rate_limiting = RateLimitSettings(**rate_config)

        if "batching" in config_dict:
            batch_config = config_dict["batching"]
            config.batching = BatchingSettings(**batch_config)

        if "error_handling" in config_dict:
            error_config = config_dict["error_handling"]
            config.error_handling = ErrorHandlingSettings(**error_config)

        if "analysis" in config_dict:
            analysis_config = config_dict["analysis"]
            config.analysis = AnalysisSettings(**analysis_config)

        if "features" in config_dict:
            features = config_dict["features"]
            config.enable_enhanced_analysis = features.get(
                "enable_enhanced_analysis", True
            )
            config.enable_smart_caching = features.get("enable_smart_caching", False)
            config.enable_async_processing = features.get(
                "enable_async_processing", True
            )
            config.enable_batch_optimization = features.get(
                "enable_batch_optimization", True
            )

        return config

    def get_free_plan_optimized_config(self) -> "EnhancedSystemConfig":
        """Get configuration optimized for free plan limitations"""
        config = self.__class__()

        # Conservative rate limiting for free plan
        config.rate_limiting = RateLimitSettings(
            max_requests_per_minute=15,
            max_requests_per_second=1,
            retry_attempts=3,
            base_delay=2.0,  # Longer delays
            max_delay=120.0,  # Longer max delay
            jitter=True,
            enable_rate_limiting=True,
        )

        # Aggressive batching to reduce API calls
        config.batching = BatchingSettings(
            max_sections_per_batch=8,  # More sections per batch
            max_total_chars=6000,  # Larger batches
            min_section_length=30,  # Include more sections
            enable_smart_batching=True,
            priority_sections=self.batching.priority_sections,
        )

        # Robust error handling
        config.error_handling = ErrorHandlingSettings(
            enable_graceful_degradation=True,
            enable_fallback_analysis=True,
            max_fallback_sections=2,  # Limit fallback to save API calls
            log_error_details=True,
            save_error_reports=True,
        )

        # Optimized analysis settings
        config.analysis = AnalysisSettings(
            enable_rag_enhancement=True,
            max_rag_snippets=3,  # Fewer RAG snippets
            analysis_temperature=0.1,
            max_output_tokens=1024,
            batch_analysis_timeout=600,  # Longer timeout for batches
            enable_progress_tracking=True,
        )

        return config


# Default configuration
DEFAULT_CONFIG = EnhancedSystemConfig()

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
        config.rate_limiting.max_requests_per_minute = int(
            os.getenv("GEMINI_MAX_REQUESTS_PER_MINUTE")
        )

    if os.getenv("GEMINI_RETRY_ATTEMPTS"):
        config.rate_limiting.retry_attempts = int(os.getenv("GEMINI_RETRY_ATTEMPTS"))

    # Batching
    if os.getenv("GEMINI_MAX_SECTIONS_PER_BATCH"):
        config.batching.max_sections_per_batch = int(
            os.getenv("GEMINI_MAX_SECTIONS_PER_BATCH")
        )

    if os.getenv("GEMINI_MAX_TOTAL_CHARS"):
        config.batching.max_total_chars = int(os.getenv("GEMINI_MAX_TOTAL_CHARS"))

    # Error handling
    if os.getenv("GEMINI_ENABLE_GRACEFUL_DEGRADATION"):
        config.error_handling.enable_graceful_degradation = (
            os.getenv("GEMINI_ENABLE_GRACEFUL_DEGRADATION").lower() == "true"
        )

    return config
