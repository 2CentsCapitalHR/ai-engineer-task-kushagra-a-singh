"""
Smart Batching for Document Analysis
Combines multiple sections into single API calls to reduce API usage while maintaining quality
"""

import hashlib
import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from config import get_enhanced_config

logger = logging.getLogger(__name__)


@dataclass
class BatchConfig:
    """Configuration for smart batching"""

    max_sections_per_batch: int = 5  # Maximum sections to combine
    max_total_chars: int = 4000  # Maximum characters per batch
    min_section_length: int = 50  # Minimum section length to include
    priority_sections: List[str] = None  # High-priority section types

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

    @classmethod
    def from_enhanced_config(cls, config):
        """Create BatchConfig from enhanced system config"""
        try:
            batch_settings = config.batching
            return cls(
                max_sections_per_batch=batch_settings.max_sections_per_batch,
                max_total_chars=batch_settings.max_total_chars,
                min_section_length=batch_settings.min_section_length,
                priority_sections=batch_settings.priority_sections,
            )
        except Exception as e:
            logger.warning(f"Error creating BatchConfig from enhanced config: {e}")
            # Return default configuration
            return cls()


class Section:
    """Represents a document section with metadata"""

    def __init__(
        self,
        text: str,
        clause: str = None,
        index: int = 0,
        section_type: str = "content",
    ):
        """Initialize a document section"""
        try:
            # Ensure text is a string
            if not isinstance(text, str):
                text = str(text) if text is not None else ""

            # Ensure clause is a string
            if not isinstance(clause, str):
                clause = str(clause) if clause is not None else "Unknown"

            # Ensure index is an integer
            if not isinstance(index, int):
                try:
                    index = int(index) if index is not None else 0
                except (ValueError, TypeError):
                    index = 0

            # Ensure section_type is a string
            if not isinstance(section_type, str):
                section_type = (
                    str(section_type) if section_type is not None else "content"
                )

            self.text = text
            self.clause = clause
            self.index = index
            self.section_type = section_type
            self.length = len(text)
            self.priority_score = self._calculate_priority()
            self.content_hash = self._calculate_hash()

        except Exception as e:
            logger.warning(f"Error initializing Section: {e}, using defaults")
            # Set safe defaults
            self.text = str(text) if text is not None else "Error processing section"
            self.clause = str(clause) if clause is not None else "Unknown"
            self.index = 0
            self.section_type = "content"
            self.length = len(self.text)
            self.priority_score = 0
            self.content_hash = "error_hash"

    def _calculate_priority(self) -> int:
        """Calculate priority score for this section"""
        try:
            priority = 0

            # Check if this is a priority section type
            priority_sections = None
            try:
                priority_sections = getattr(
                    getattr(self, "config", None), "priority_sections", None
                )
            except Exception:
                priority_sections = None
            if priority_sections is None:
                priority_sections = [
                    "RESOLUTION",
                    "APPOINTMENT",
                    "ARTICLES",
                    "SHARE CAPITAL",
                    "DIRECTOR",
                    "SECRETARY",
                    "SIGNATORY",
                ]

            if self.section_type and self.section_type.upper() in priority_sections:
                priority += 10

            # Boost priority for longer sections (more content to analyze)
            if self.length > 500:
                priority += 5
            elif self.length > 200:
                priority += 3
            elif self.length > 100:
                priority += 1

            # Boost priority for sections with specific keywords
            text_lower = self.text.lower()
            priority_keywords = [
                "resolution",
                "appointment",
                "director",
                "secretary",
                "shareholder",
                "capital",
                "articles",
                "incorporation",
            ]

            for keyword in priority_keywords:
                if keyword in text_lower:
                    priority += 2
                    break

            return priority

        except Exception as e:
            logger.warning(f"Error calculating priority for section {self.index}: {e}")
            return 0  # Return default priority

    def _calculate_hash(self) -> str:
        """Calculate content hash for deduplication"""
        try:
            import hashlib

            content = f"{self.text}_{self.clause}_{self.section_type}"
            return hashlib.md5(content.encode()).hexdigest()
        except Exception as e:
            logger.warning(f"Error calculating hash for section {self.index}: {e}")
            return f"error_hash_{self.index}"  # Return fallback hash

    def __repr__(self):
        return (
            f"Section({self.clause}, len={self.length}, priority={self.priority_score})"
        )


class SmartBatcher:
    """Intelligent batching of document sections for analysis"""

    def __init__(self, config: BatchConfig = None):
        """Initialize the smart batcher"""
        try:
            if config is None:
                try:
                    enhanced_config = get_enhanced_config()
                    self.config = BatchConfig.from_enhanced_config(enhanced_config)
                except Exception as config_error:
                    logger.warning(f"Error getting enhanced config: {config_error}")
                    # Use default configuration
                    self.config = BatchConfig()
            else:
                self.config = config

            self.processed_hashes = set()
            logger.info(f"SmartBatcher initialized with config: {self.config}")

        except Exception as e:
            logger.error(f"Error initializing SmartBatcher: {e}")
            # Use minimal default configuration
            self.config = BatchConfig()
            self.processed_hashes = set()
            logger.info("SmartBatcher initialized with default config due to error")

    def create_batches(self, sections: List[Dict[str, Any]]) -> List[List[Section]]:
        """Create optimal batches from document sections"""

        try:
            # Validate input
            if not sections or not isinstance(sections, list):
                logger.warning(f"Invalid sections input: {type(sections)}")
                return []

            # Convert to Section objects and filter
            section_objects = []
            for i, section_data in enumerate(sections):
                try:
                    # Add bounds checking to prevent index errors
                    if i >= len(sections):
                        logger.warning(
                            f"Section index {i} out of range for sections length {len(sections)}"
                        )
                        break

                    if not isinstance(section_data, dict):
                        logger.warning(
                            f"Invalid section data at index {i}: {type(section_data)}"
                        )
                        continue

                    text = section_data.get("text", "")
                    if not text or not isinstance(text, str):
                        logger.warning(f"Invalid text at section {i}: {type(text)}")
                        continue

                    if len(text) >= self.config.min_section_length:
                        section = Section(
                            text=text,
                            clause=section_data.get("clause"),
                            index=i,
                            section_type=section_data.get("section_type", "content"),
                        )

                        # Skip if we've already processed this content
                        if section.content_hash not in self.processed_hashes:
                            section_objects.append(section)
                            self.processed_hashes.add(section.content_hash)
                except IndexError as index_error:
                    logger.warning(f"Index error accessing section {i}: {index_error}")
                    break
                except Exception as section_error:
                    logger.warning(f"Error creating section {i}: {section_error}")
                    # Create minimal section as fallback
                    try:
                        fallback_section = Section(
                            text=(
                                section_data.get("text", "Error processing section")
                                if isinstance(section_data, dict)
                                else "Error processing section"
                            ),
                            clause=(
                                section_data.get("clause", f"Section_{i}")
                                if isinstance(section_data, dict)
                                else f"Section_{i}"
                            ),
                            index=i,
                            section_type=(
                                section_data.get("section_type", "content")
                                if isinstance(section_data, dict)
                                else "content"
                            ),
                        )
                        section_objects.append(fallback_section)
                    except Exception as fallback_error:
                        logger.error(
                            f"Failed to create fallback section {i}: {fallback_error}"
                        )
                        continue

            if not section_objects:
                logger.warning("No valid sections could be created")
                return []

            # Sort by priority (highest first)
            try:
                section_objects.sort(key=lambda x: x.priority_score, reverse=True)
            except Exception as sort_error:
                logger.warning(f"Error sorting sections by priority: {sort_error}")
                # Continue with unsorted sections

            # Create batches
            batches = []
            current_batch = []
            current_chars = 0

            for section in section_objects:
                try:
                    # Check if adding this section would exceed batch limits
                    if (
                        len(current_batch) >= self.config.max_sections_per_batch
                        or current_chars + section.length > self.config.max_total_chars
                    ):

                        if current_batch:
                            batches.append(current_batch)
                            current_batch = []
                            current_chars = 0

                    # Add section to current batch
                    current_batch.append(section)
                    current_chars += section.length
                except Exception as batch_creation_error:
                    logger.warning(
                        f"Error adding section to batch: {batch_creation_error}"
                    )
                    # Add current batch if it has content
                    if current_batch:
                        batches.append(current_batch)
                        current_batch = []
                        current_chars = 0
                    continue

            # Add final batch if it has content
            if current_batch:
                batches.append(current_batch)

            logger.info(
                f"Created {len(batches)} batches from {len(section_objects)} sections"
            )
            return batches

        except Exception as e:
            logger.error(f"Error in create_batches: {e}")
            # Return minimal fallback batches
            try:
                fallback_batches = []
                for i, section_data in enumerate(sections):
                    try:
                        # Add bounds checking to prevent index errors
                        if i >= len(sections):
                            logger.warning(
                                f"Section index {i} out of range for sections length {len(sections)}"
                            )
                            break

                        if not isinstance(section_data, dict):
                            continue
                        fallback_section = Section(
                            text=section_data.get("text", "Error processing section"),
                            clause=section_data.get("clause", f"Section_{i}"),
                            index=i,
                            section_type=section_data.get("section_type", "content"),
                        )
                        fallback_batches.append([fallback_section])
                    except IndexError as index_error:
                        logger.warning(
                            f"Index error accessing section {i}: {index_error}"
                        )
                        break
                    except Exception as fallback_error:
                        logger.error(
                            f"Failed to create fallback batch for section {i}: {fallback_error}"
                        )
                        continue

                logger.info(
                    f"Created {len(fallback_batches)} fallback batches due to error"
                )
                return fallback_batches
            except Exception as fallback_batch_error:
                logger.error(f"Error creating fallback batches: {fallback_batch_error}")
                return []

    def create_batch_prompt(self, batch: List[Section], doc_type: str) -> str:
        """Create a comprehensive prompt for batch analysis"""

        try:
            # Build section content
            sections_content = []
            for i, section in enumerate(batch):
                try:
                    # Add bounds checking to prevent index errors
                    if i >= len(batch):
                        logger.warning(
                            f"Section index {i} out of range for batch length {len(batch)}"
                        )
                        break

                    section_text = getattr(section, "text", "Error processing section")
                    section_clause = getattr(section, "clause", f"Section_{i+1}")
                    sections_content.append(
                        f"Section {i+1} ({section_clause}):\n{section_text}\n"
                    )
                except IndexError as index_error:
                    logger.warning(f"Index error accessing section {i}: {index_error}")
                    break
                except Exception as section_error:
                    logger.warning(
                        f"Error processing section {i} in prompt creation: {section_error}"
                    )
                    sections_content.append(
                        f"Section {i+1} (Error):\nError processing section\n"
                    )

            # Combine all sections
            combined_content = "\n".join(sections_content)

            # Create the prompt
            prompt = f"""Analyze the following {len(batch)} sections from a {doc_type} document for ADGM compliance issues.

Document Type: {doc_type}
Number of Sections: {len(batch)}

{combined_content}

Please provide a comprehensive analysis in the following JSON format:

{{
    "batched_response": true,
    "sections_analyzed": {len(batch)},
    "overall_compliance": "compliant|non_compliant|requires_review",
    "total_issues": <number>,
    "section_1": {{
        "red_flag": "<issue description or 'No issues found'>",
        "law_citation": "<ADGM regulation reference>",
        "suggestion": "<specific recommendation>",
        "severity": "<High|Medium|Low>",
        "category": "<compliance|formatting|content|other>"
    }},
    "section_2": {{
        "red_flag": "<issue description or 'No issues found'>",
        "law_citation": "<ADGM regulation reference>",
        "suggestion": "<specific recommendation>",
        "severity": "<High|Medium|Low>",
        "category": "<compliance|formatting|content|other>"
    }}
    // ... continue for each section
}}

Focus on:
1. ADGM Companies Regulations 2020 compliance
2. Required clauses and formatting
3. Jurisdiction-specific requirements
4. Corporate governance standards
5. Regulatory compliance issues

Provide specific citations from ADGM regulations where possible."""

            return prompt
        except Exception as e:
            logger.error(f"Error creating batch prompt: {e}")
            # Return minimal fallback prompt
            return f"""Analyze the following document sections for ADGM compliance.

Document Type: {doc_type}
Number of Sections: {len(batch)}

Error occurred while creating detailed prompt. Please analyze the document sections for compliance issues.

Provide analysis in JSON format with section_1, section_2, etc."""

    def parse_batch_response(
        self, response: str, batch: List[Section]
    ) -> List[Dict[str, Any]]:
        """Parse the batch response back into individual section results with enhanced error handling"""

        results = []

        try:
            # Validate input parameters
            if not isinstance(batch, list) or not batch:
                logger.warning(f"Invalid batch: {type(batch)}, returning empty results")
                return []

            # Enhanced response preprocessing and validation
            parsed_response = self._preprocess_and_validate_response(response)

            if parsed_response is None:
                logger.warning(
                    "Response preprocessing failed, using comprehensive fallback"
                )
                return self._create_fallback_results(
                    batch, "Response preprocessing failed"
                )

            # Parse the response into individual section results
            results = self._extract_section_results(parsed_response, batch)

            if not results:
                logger.warning("Section extraction failed, using fallback results")
                return self._create_fallback_results(batch, "Section extraction failed")

            return results

        except Exception as e:
            logger.error(f"Critical error in parse_batch_response: {e}")
            # Return fallback results for all sections
            return self._create_fallback_results(
                batch, f"Critical parsing error: {str(e)}"
            )

    def _preprocess_and_validate_response(
        self, response: Any
    ) -> Optional[Dict[str, Any]]:
        """Enhanced response preprocessing with multiple fallback strategies"""

        try:
            # Handle different response types
            if isinstance(response, dict):
                return response
            elif isinstance(response, list):
                # Convert list to dict format
                converted = {}
                for i, item in enumerate(response):
                    if isinstance(item, dict):
                        converted[f"section_{i+1}"] = item
                    else:
                        converted[f"section_{i+1}"] = {
                            "error": f"Invalid item type: {type(item)}"
                        }
                return converted

            # Convert to string if needed
            if not isinstance(response, str):
                response = str(response)

            # Skip processing if response is empty or whitespace
            if not response or not response.strip():
                logger.warning("Empty or whitespace-only response received")
                return None

            # Clean response text with multiple strategies
            cleaned_response = self._clean_response_text(response)

            # Try multiple JSON parsing strategies
            parsed = self._parse_json_with_fallbacks(cleaned_response)

            if parsed is None:
                logger.warning("All JSON parsing strategies failed")
                return None

            return parsed

        except Exception as e:
            logger.error(f"Error in response preprocessing: {e}")
            return None

    def _clean_response_text(self, response: str) -> str:
        """Clean response text with multiple strategies"""

        try:
            # Remove markdown code blocks
            cleaned = response.replace("```json", "").replace("```", "").strip()

            # Remove common prefixes/suffixes that might cause parsing issues
            prefixes_to_remove = [
                "Here's the analysis:",
                "Analysis results:",
                "JSON response:",
                "The analysis shows:",
                "Based on the document:",
            ]

            for prefix in prefixes_to_remove:
                if cleaned.startswith(prefix):
                    cleaned = cleaned[len(prefix) :].strip()

            # Remove trailing punctuation that might break JSON
            while cleaned and cleaned[-1] in [".", ",", ";", ":", "!", "?"]:
                cleaned = cleaned[:-1].strip()

            # Fix common JSON formatting issues
            cleaned = self._fix_common_json_issues(cleaned)

            return cleaned

        except Exception as e:
            logger.warning(f"Error cleaning response text: {e}")
            return response

    def _fix_common_json_issues(self, text: str) -> str:
        """Fix common JSON formatting issues that cause parsing failures"""

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

            return fixed_text

        except Exception as e:
            logger.warning(f"Error fixing JSON issues: {e}")
            return text

    def _parse_json_with_fallbacks(self, text: str) -> Optional[Dict[str, Any]]:
        """Parse JSON with multiple fallback strategies"""

        strategies = [
            self._try_direct_json_parse,
            self._try_fix_and_parse,
            self._try_extract_json_blocks,
            self._try_manual_json_construction,
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

    def _try_direct_json_parse(self, text: str) -> Optional[Dict[str, Any]]:
        """Try direct JSON parsing"""
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None

    def _try_fix_and_parse(self, text: str) -> Optional[Dict[str, Any]]:
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

    def _try_extract_json_blocks(self, text: str) -> Optional[Dict[str, Any]]:
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

    def _try_manual_json_construction(self, text: str) -> Optional[Dict[str, Any]]:
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

    def _extract_section_results(
        self, parsed_response: Dict[str, Any], batch: List[Section]
    ) -> List[Dict[str, Any]]:
        """Extract individual section results from parsed response"""

        results = []

        try:
            for i, section in enumerate(batch):
                try:
                    # Validate section object
                    if not hasattr(section, "text") or not section.text:
                        logger.warning(f"Section {i} has no text, skipping...")
                        continue

                    # Try to get section result from parsed response
                    section_key = f"section_{i+1}"
                    section_result = parsed_response.get(section_key, {})

                    # Ensure section_result is a dictionary
                    if not isinstance(section_result, dict):
                        logger.warning(
                            f"Section result for {section_key} is not a dictionary: {type(section_result)}"
                        )
                        section_result = {}

                    # Create result with section metadata
                    result = {
                        **section_result,
                        "section_index": getattr(section, "index", i),
                        "section_clause": getattr(section, "clause", f"Section_{i+1}"),
                        "section_text": (
                            (getattr(section, "text", "")[:150] + "...")
                            if len(getattr(section, "text", "")) > 150
                            else getattr(section, "text", "")
                        ),
                        "section_type": getattr(section, "section_type", "content"),
                    }

                    # Validate and set required fields with intelligent defaults
                    result = self._validate_and_set_required_fields(result, section, i)

                    results.append(result)

                except Exception as section_error:
                    logger.warning(
                        f"Error processing section {i} in batch response: {section_error}"
                    )
                    # Create fallback result for this section
                    fallback_result = self._create_single_fallback_result(
                        section, i, f"Section processing error: {str(section_error)}"
                    )
                    if fallback_result:
                        results.append(fallback_result)

            return results

        except Exception as e:
            logger.error(f"Error extracting section results: {e}")
            return self._create_fallback_results(
                batch, f"Section extraction error: {str(e)}"
            )

    def _validate_and_set_required_fields(
        self, result: Dict[str, Any], section: Section, index: int
    ) -> Dict[str, Any]:
        """Validate and set required fields with intelligent defaults"""

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

    def _create_fallback_results(
        self, batch: List[Section], error_message: str
    ) -> List[Dict[str, Any]]:
        """Create fallback results for all sections in a batch"""
        fallback_results = []

        for i, section in enumerate(batch):
            try:
                # Add bounds checking to prevent index errors
                if i >= len(batch):
                    logger.warning(
                        f"Section index {i} out of range for batch length {len(batch)}"
                    )
                    break

                fallback_result = self._create_single_fallback_result(
                    section, i, error_message
                )
                if fallback_result:
                    fallback_results.append(fallback_result)
            except IndexError as index_error:
                logger.warning(f"Index error accessing section {i}: {index_error}")
                break
            except Exception as section_error:
                logger.warning(
                    f"Error creating fallback result for section {i}: {section_error}"
                )
                # Create minimal fallback result - faster processing
                fallback_results.append(
                    {
                        "red_flag": f"Processing error: {error_message[:100]}...",
                        "law_citation": "ADGM Legal Framework - General Requirements",
                        "suggestion": "Manual review required due to processing error",
                        "severity": "Medium",
                        "category": "compliance",
                        "confidence": "Low",
                        "compliant_clause": "Consult ADGM legal advisor",
                        "section_index": i,
                        "section_clause": "Unknown",
                        "section_text": "Processing error occurred",
                        "section_type": "content",
                        "analysis_method": "fast_fallback",
                    }
                )

        return fallback_results

    def _create_single_fallback_result(
        self, section: Section, index: int, error_message: str
    ) -> Dict[str, Any]:
        """Create a single fallback result for a section"""
        try:
            # Validate section object
            if not section or not hasattr(section, "text"):
                logger.warning(f"Invalid section object for index {index}")
                return None

            return {
                "red_flag": f"Processing error: {error_message[:80]}...",
                "law_citation": "ADGM Legal Framework - General Requirements",
                "suggestion": "Manual legal review required due to processing error",
                "severity": "Medium",
                "category": "compliance",
                "confidence": "Low",
                "compliant_clause": "Consult ADGM legal advisor",
                "section_index": getattr(section, "index", index),
                "section_clause": getattr(section, "clause", f"Section_{index+1}"),
                "section_text": (
                    (getattr(section, "text", "")[:150] + "...")
                    if len(getattr(section, "text", "")) > 150
                    else getattr(section, "text", "")
                ),
                "section_type": getattr(section, "section_type", "content"),
                "analysis_method": "fast_fallback",
            }
        except Exception as e:
            logger.warning(f"Error creating single fallback result: {e}")
            return None


def estimate_api_savings(
    original_sections: int, batched_sections: int
) -> Dict[str, Any]:
    """Estimate API call savings from batching"""
    try:
        original_calls = original_sections
        batched_calls = batched_sections

        savings = original_calls - batched_calls
        savings_percentage = (
            (savings / original_calls * 100) if original_calls > 0 else 0
        )

        return {
            "original_api_calls": original_calls,
            "batched_api_calls": batched_calls,
            "api_calls_saved": savings,
            "savings_percentage": savings_percentage,
            "estimated_time_saved_minutes": savings
            * 4,  # Assuming 4 seconds per API call
        }
    except Exception as e:
        logger.warning(f"Error calculating API savings: {e}")
        # Return default values
        return {
            "original_api_calls": original_sections,
            "batched_api_calls": batched_sections,
            "api_calls_saved": 0,
            "savings_percentage": 0,
            "estimated_time_saved_minutes": 0,
        }
