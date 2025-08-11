"""
Enhanced Error Handling and Input Validation
Provides robust error handling, input validation, and graceful degradation
"""

import json
import logging
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from config import get_enhanced_config

logger = logging.getLogger(__name__)


@dataclass
class ValidationError:
    """Represents a validation error with context"""

    field: str
    message: str
    severity: str = "error"  # error, warning, info
    suggested_fix: str = ""
    context: Dict[str, Any] = None

    def __post_init__(self):
        try:
            if self.context is None:
                self.context = {}
        except Exception as e:
            logger.warning(f"Error in ValidationError __post_init__: {e}")
            self.context = {}


class InputValidator:
    """Validates input data to prevent common errors"""

    @staticmethod
    def validate_section_data(section_data: Dict[str, Any]) -> List[ValidationError]:
        """Validate document section data structure"""
        errors = []

        try:
            # Check required fields
            if not isinstance(section_data, dict):
                errors.append(
                    ValidationError(
                        field="section_data",
                        message="Section data must be a dictionary",
                        severity="error",
                        suggested_fix="Ensure section_data is a valid dictionary",
                    )
                )
                return errors

            # Validate text field
            text = section_data.get("text")
            if not text or not isinstance(text, str):
                errors.append(
                    ValidationError(
                        field="text",
                        message="Text field is required and must be a string",
                        severity="error",
                        suggested_fix="Provide valid text content for the section",
                    )
                )

            # Validate index field
            index = section_data.get("index")
            if index is not None and not isinstance(index, (int, str)):
                errors.append(
                    ValidationError(
                        field="index",
                        message="Index must be an integer or string",
                        severity="warning",
                        suggested_fix="Convert index to integer or string",
                    )
                )

            # Validate clause field
            clause = section_data.get("clause")
            if clause is not None and not isinstance(clause, str):
                errors.append(
                    ValidationError(
                        field="clause",
                        message="Clause must be a string",
                        severity="warning",
                        suggested_fix="Convert clause to string or set to None",
                    )
                )

            # Validate section_type field
            # Get valid types from config with error handling
            try:
                enhanced_config = get_enhanced_config()
                valid_types = enhanced_config.error_handling.valid_section_types
            except Exception as config_error:
                logger.warning(
                    f"Error accessing config for section type validation: {config_error}"
                )
                # Use default valid types
                valid_types = [
                    "header",
                    "content",
                    "footer",
                    "clause",
                    "resolution",
                    "appointment",
                    "articles",
                    "share_capital",
                    "director",
                    "secretary",
                    "signatory",
                ]

            section_type = section_data.get("section_type")
            if section_type and section_type not in valid_types:
                errors.append(
                    ValidationError(
                        field="section_type",
                        message=f"Section type must be one of: {', '.join(valid_types)}",
                        severity="warning",
                        suggested_fix=f"Use one of the valid section types: {', '.join(valid_types)}",
                    )
                )

            return errors
        except Exception as e:
            logger.error(f"Error in validate_section_data: {e}")
            # Return critical error
            return [
                ValidationError(
                    field="validation_system",
                    message=f"Validation system error: {str(e)}",
                    severity="error",
                    suggested_fix="Contact system administrator",
                )
            ]

    @staticmethod
    def validate_document_data(document_data: Dict[str, Any]) -> List[ValidationError]:
        """Validate document data structure"""
        errors = []

        try:
            # Check if document_data is a dictionary
            if not isinstance(document_data, dict):
                errors.append(
                    ValidationError(
                        field="document_data",
                        message="Document data must be a dictionary",
                        severity="error",
                        suggested_fix="Ensure document_data is a valid dictionary",
                    )
                )
                return errors

            # Validate filename
            filename = document_data.get("filename")
            if not filename or not isinstance(filename, str):
                errors.append(
                    ValidationError(
                        field="filename",
                        message="Filename is required and must be a string",
                        severity="error",
                        suggested_fix="Provide a valid filename string",
                    )
                )

            # Validate doc_type
            doc_type = document_data.get("doc_type")
            if doc_type and not isinstance(doc_type, str):
                errors.append(
                    ValidationError(
                        field="doc_type",
                        message="Document type must be a string",
                        severity="warning",
                        suggested_fix="Convert doc_type to string or set to None",
                    )
                )

            # Validate sections
            sections = document_data.get("sections")
            if sections is None:
                errors.append(
                    ValidationError(
                        field="sections",
                        message="Sections field is required",
                        severity="error",
                        suggested_fix="Provide sections data for the document",
                    )
                )
            elif not isinstance(sections, list):
                errors.append(
                    ValidationError(
                        field="sections",
                        message="Sections must be a list",
                        severity="error",
                        suggested_fix="Convert sections to a list format",
                    )
                )
            else:
                # Validate each section
                for i, section in enumerate(sections):
                    try:
                        section_errors = InputValidator.validate_section_data(section)
                        for error in section_errors:
                            error.field = f"sections[{i}].{error.field}"
                            errors.append(error)
                    except Exception as section_error:
                        logger.warning(f"Error validating section {i}: {section_error}")
                        errors.append(
                            ValidationError(
                                field=f"sections[{i}]",
                                message=f"Section validation error: {str(section_error)}",
                                severity="error",
                                suggested_fix="Check section data structure",
                            )
                        )

            return errors
        except Exception as e:
            logger.error(f"Error in validate_document_data: {e}")
            # Return critical error
            return [
                ValidationError(
                    field="validation_system",
                    message=f"Document validation system error: {str(e)}",
                    severity="error",
                    suggested_fix="Contact system administrator",
                )
            ]

    @staticmethod
    def safe_get_nested(
        data: Dict[str, Any], keys: List[str], default: Any = None
    ) -> Any:
        """Safely get nested dictionary values without raising KeyError"""
        try:
            if not isinstance(data, dict) or not isinstance(keys, list):
                return default

            current = data
            for key in keys:
                if isinstance(current, dict) and key in current:
                    current = current[key]
                else:
                    return default
            return current
        except (TypeError, AttributeError, IndexError) as e:
            logger.debug(f"Error in safe_get_nested: {e}")
            return default
        except Exception as e:
            logger.warning(f"Unexpected error in safe_get_nested: {e}")
            return default


class ErrorHandler:
    """Handles errors gracefully and provides recovery options"""

    @staticmethod
    def handle_api_error(error: Exception, context: str = "") -> Dict[str, Any]:
        """Handle API-related errors gracefully"""
        try:
            error_msg = str(error)

            # Log the error with context
            logger.error(f"API error in {context}: {error_msg}")
            logger.debug(f"Full traceback: {traceback.format_exc()}")

            # Get error handling settings from config with error handling
            try:
                enhanced_config = get_enhanced_config()
                error_settings = enhanced_config.error_handling
            except Exception as config_error:
                logger.warning(
                    f"Error accessing config for error handling: {config_error}"
                )
                # Use default error settings
                error_settings = type(
                    "obj",
                    (object,),
                    {
                        "retry_on_errors": [
                            "rate_limit",
                            "quota",
                            "timeout",
                            "network",
                        ],
                        "log_error_details": True,
                        "include_stack_trace": False,
                    },
                )()

            # Determine error type and provide appropriate response
            if "429" in error_msg or "quota" in error_msg.lower():
                return {
                    "red_flag": f"Rate limit exceeded: {error_msg}",
                    "law_citation": "ADGM Legal Framework - General Requirements",
                    "suggestion": "Wait before retrying or upgrade API plan",
                    "severity": "Medium",
                    "category": "compliance",
                    "confidence": "High",
                    "compliant_clause": "Retry analysis after rate limit reset",
                    "error_type": "rate_limit",
                    "recoverable": True,
                }

            elif "list indices" in error_msg:
                return {
                    "red_flag": f"Programming error: {error_msg}",
                    "law_citation": "ADGM Legal Framework - General Requirements",
                    "suggestion": "Check input data structure and validation",
                    "severity": "High",
                    "category": "compliance",
                    "confidence": "High",
                    "compliant_clause": "Fix data structure issues before retry",
                    "error_type": "programming_error",
                    "recoverable": False,
                }

            elif "timeout" in error_msg.lower():
                return {
                    "red_flag": f"Request timeout: {error_msg}",
                    "law_citation": "ADGM Legal Framework - General Requirements",
                    "suggestion": "Retry with shorter content or check network",
                    "severity": "Medium",
                    "category": "compliance",
                    "confidence": "Medium",
                    "compliant_clause": "Retry analysis with reduced content",
                    "error_type": "timeout",
                    "recoverable": True,
                }

            else:
                return {
                    "red_flag": f"API error: {error_msg}",
                    "law_citation": "ADGM Legal Framework - General Requirements",
                    "suggestion": "Check API configuration and retry",
                    "severity": "Medium",
                    "category": "compliance",
                    "confidence": "Low",
                    "compliant_clause": "Retry analysis or contact support",
                    "error_type": "unknown",
                    "recoverable": True,
                }
        except Exception as e:
            logger.error(f"Error in handle_api_error: {e}")
            # Return minimal error response
            return {
                "red_flag": f"Critical error handler failure: {str(e)}",
                "law_citation": "System Error - Contact Administrator",
                "suggestion": "Contact system administrator immediately",
                "severity": "Critical",
                "category": "system_error",
                "confidence": "High",
                "compliant_clause": "System requires immediate attention",
                "error_type": "error_handler_failure",
                "recoverable": False,
            }

    @staticmethod
    def handle_parsing_error(
        error: Exception, content: str, context: str = ""
    ) -> Dict[str, Any]:
        """Handle parsing errors gracefully"""
        try:
            error_msg = str(error)

            # Log the parsing error
            logger.error(f"Parsing error in {context}: {error_msg}")
            logger.debug(f"Content that caused error: {content[:200]}...")

            # Determine error type and provide appropriate response
            if "json" in error_msg.lower():
                return {
                    "red_flag": f"JSON parsing error: {error_msg}",
                    "law_citation": "ADGM Legal Framework - General Requirements",
                    "suggestion": "Check response format and retry analysis",
                    "severity": "Medium",
                    "category": "parsing",
                    "confidence": "High",
                    "compliant_clause": "Retry analysis with corrected format",
                    "error_type": "json_parsing",
                    "recoverable": True,
                }
            elif "index" in error_msg.lower():
                return {
                    "red_flag": f"Index error: {error_msg}",
                    "law_citation": "ADGM Legal Framework - General Requirements",
                    "suggestion": "Check data structure and validation",
                    "severity": "High",
                    "category": "parsing",
                    "confidence": "High",
                    "compliant_clause": "Fix data structure issues before retry",
                    "error_type": "index_error",
                    "recoverable": False,
                }
            else:
                return {
                    "red_flag": f"Parsing error: {error_msg}",
                    "law_citation": "ADGM Legal Framework - General Requirements",
                    "suggestion": "Check input format and retry",
                    "severity": "Medium",
                    "category": "parsing",
                    "confidence": "Medium",
                    "compliant_clause": "Retry analysis with corrected input",
                    "error_type": "parsing_error",
                    "recoverable": True,
                }
        except Exception as e:
            logger.error(f"Error in handle_parsing_error: {e}")
            # Return minimal error response
            return {
                "red_flag": f"Critical parsing error handler failure: {str(e)}",
                "law_citation": "System Error - Contact Administrator",
                "suggestion": "Contact system administrator immediately",
                "severity": "Critical",
                "category": "system_error",
                "confidence": "High",
                "compliant_clause": "System requires immediate attention",
                "error_type": "parsing_error_handler_failure",
                "recoverable": False,
            }

    @staticmethod
    def handle_validation_errors(errors: List[ValidationError]) -> Dict[str, Any]:
        """Handle validation errors and provide recovery guidance"""
        try:
            if not errors:
                return {"valid": True}

            # Categorize errors by severity
            error_count = len([e for e in errors if e.severity == "error"])
            warning_count = len([e for e in errors if e.severity == "warning"])

            # Create summary
            summary = {
                "valid": error_count == 0,
                "error_count": error_count,
                "warning_count": warning_count,
                "total_issues": len(errors),
                "errors": [e for e in errors if e.severity == "error"],
                "warnings": [e for e in errors if e.severity == "warning"],
                "can_proceed": error_count == 0,
            }

            # Log issues
            if error_count > 0:
                logger.error(f"Validation failed with {error_count} errors:")
                for error in summary["errors"]:
                    try:
                        logger.error(f"  {error.field}: {error.message}")
                    except Exception as log_error:
                        logger.warning(f"Error logging validation error: {log_error}")
                        continue

            if warning_count > 0:
                logger.warning(f"Validation warnings ({warning_count}):")
                for warning in summary["warnings"]:
                    try:
                        logger.warning(f"  {warning.field}: {warning.message}")
                    except Exception as log_error:
                        logger.warning(f"Error logging validation warning: {log_error}")
                        continue

            return summary
        except Exception as e:
            logger.error(f"Error in handle_validation_errors: {e}")
            # Return minimal fallback summary
            return {
                "valid": False,
                "error_count": len(errors) if errors else 0,
                "warning_count": 0,
                "total_issues": len(errors) if errors else 0,
                "errors": [],
                "warnings": [],
                "can_proceed": False,
                "validation_error": f"Validation system error: {str(e)}",
            }


class GracefulDegradation:
    """Provides fallback analysis when primary methods fail"""

    @staticmethod
    def basic_compliance_check(content: str, doc_type: str) -> Dict[str, Any]:
        """Basic compliance check using local rules when API fails"""
        try:
            issues = []

            # Ensure content is a string
            if not isinstance(content, str):
                content = str(content) if content else ""

            # Basic jurisdiction check
            if (
                "ADGM" not in content.upper()
                and "ABU DHABI GLOBAL MARKET" not in content.upper()
            ):
                issues.append(
                    {
                        "red_flag": "Document does not explicitly mention ADGM jurisdiction",
                        "law_citation": "ADGM Companies Regulations 2020, Article 1",
                        "suggestion": "Ensure document explicitly references ADGM jurisdiction",
                        "severity": "High",
                        "category": "jurisdiction",
                        "confidence": "Medium",
                        "compliant_clause": "This document is governed by ADGM laws and regulations",
                    }
                )

            # Basic structure check
            if len(content.strip()) < 100:
                issues.append(
                    {
                        "red_flag": "Document content appears too short for proper legal analysis",
                        "law_citation": "ADGM Legal Framework - General Requirements",
                        "suggestion": "Ensure document contains sufficient content for analysis",
                        "severity": "Medium",
                        "category": "formatting",
                        "confidence": "Medium",
                        "compliant_clause": "Expand document content to meet minimum requirements",
                    }
                )

            # Check for common required elements
            required_terms = ["RESOLVED", "APPOINT", "DIRECTOR", "SHARE", "ARTICLES"]
            missing_terms = [
                term for term in required_terms if term not in content.upper()
            ]

            if len(missing_terms) > 3:
                issues.append(
                    {
                        "red_flag": f"Document missing several common legal elements: {', '.join(missing_terms[:3])}",
                        "law_citation": "ADGM Companies Regulations 2020 - General Requirements",
                        "suggestion": "Review document against ADGM templates for completeness",
                        "severity": "Medium",
                        "category": "missing_clauses",
                        "confidence": "Low",
                        "compliant_clause": "Include all required legal elements as per ADGM regulations",
                    }
                )

            return {
                "issues": issues,
                "total_issues": len(issues),
                "analysis_method": "local_fallback",
                "confidence": "Low",
                "suggestion": "Use API analysis when available for comprehensive review",
            }
        except Exception as e:
            logger.error(f"Error in basic_compliance_check: {e}")
            # Return minimal fallback result
            return {
                "issues": [
                    {
                        "red_flag": f"Fallback analysis error: {str(e)}",
                        "law_citation": "System Error - Contact Administrator",
                        "suggestion": "Contact system administrator for assistance",
                        "severity": "High",
                        "category": "system_error",
                        "confidence": "Low",
                        "compliant_clause": "System requires attention",
                    }
                ],
                "total_issues": 1,
                "analysis_method": "error_fallback",
                "confidence": "Low",
                "suggestion": "Contact system administrator immediately",
            }

    @staticmethod
    def create_error_summary(errors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create a summary of errors for reporting"""
        try:
            if not errors:
                return {
                    "total_errors": 0,
                    "error_summary": "No errors found",
                    "severity_breakdown": {},
                    "category_breakdown": {},
                }

            # Count errors by severity and category
            severity_counts = {}
            category_counts = {}

            for error in errors:
                try:
                    # Count by severity
                    severity = error.get("severity", "unknown")
                    severity_counts[severity] = severity_counts.get(severity, 0) + 1

                    # Count by category
                    category = error.get("category", "unknown")
                    category_counts[category] = category_counts.get(category, 0) + 1
                except Exception as error_processing_error:
                    logger.warning(
                        f"Error processing error entry: {error_processing_error}"
                    )
                    continue

            # Create summary
            summary = {
                "total_errors": len(errors),
                "error_summary": f"Found {len(errors)} errors during processing",
                "severity_breakdown": severity_counts,
                "category_breakdown": category_counts,
                "most_common_severity": (
                    max(severity_counts.items(), key=lambda x: x[1])[0]
                    if severity_counts
                    else "none"
                ),
                "most_common_category": (
                    max(category_counts.items(), key=lambda x: x[1])[0]
                    if category_counts
                    else "none"
                ),
            }

            return summary
        except Exception as e:
            logger.error(f"Error in create_error_summary: {e}")
            # Return minimal fallback summary
            return {
                "total_errors": len(errors) if errors else 0,
                "error_summary": f"Error creating summary: {str(e)}",
                "severity_breakdown": {},
                "category_breakdown": {},
                "most_common_severity": "unknown",
                "most_common_category": "unknown",
            }
