from typing import List, Tuple, Optional, Dict, Any
import re
import logging
from fuzzywuzzy import fuzz, process

logger = logging.getLogger(__name__)


class FuzzyMatchingService:
    def __init__(
        self,
        similarity_threshold: int = 80,
        scorer=fuzz.token_sort_ratio
    ):
        """
        Initialize fuzzy matching service for student name matching.

        Args:
            similarity_threshold: Minimum similarity score (0-100) for a match
            scorer: Fuzzy matching algorithm to use
        """
        if not 0 <= similarity_threshold <= 100:
            raise ValueError("Similarity threshold must be between 0 and 100")

        self.similarity_threshold = similarity_threshold
        self.scorer = scorer
        self.student_roster = []
        self.normalized_roster = {}  # Maps normalized names to original names

        logger.info(f"Initialized fuzzy matching with threshold: {similarity_threshold}")

    def load_student_roster(self, student_names: List[str]) -> None:
        """
        Load and preprocess student roster for matching.

        Args:
            student_names: List of student names from roster
        """
        if not isinstance(student_names, list):
            raise ValueError("Student names must be a list")

        if not student_names:
            logger.warning("Empty student roster provided")
            return

        self.student_roster = student_names
        self.normalized_roster = {}

        # Create normalized name mapping
        for name in student_names:
            normalized = self._normalize_name(name)
            self.normalized_roster[normalized] = name

        logger.info(f"Loaded {len(student_names)} students in roster")
        logger.debug(f"Sample normalized mappings: {list(self.normalized_roster.items())[:3]}")

    def find_best_match(
        self,
        extracted_name: str,
        return_all_candidates: bool = False
    ) -> Tuple[Optional[str], int, Dict[str, Any]]:
        """
        Find the best matching student name from the roster.

        Args:
            extracted_name: Name extracted from OCR
            return_all_candidates: If True, include top candidates in metadata

        Returns:
            Tuple of (matched_name, confidence_score, metadata)
            - matched_name: Best matching name from roster (None if no good match)
            - confidence_score: Similarity score (0-100)
            - metadata: Additional matching information
        """
        if not self.student_roster:
            logger.error("Student roster not loaded")
            return None, 0, {"error": "No roster loaded"}

        if not extracted_name or not extracted_name.strip():
            logger.warning("Empty or None extracted name provided")
            return None, 0, {"error": "Empty input name"}

        # Clean and normalize the extracted name
        normalized_extracted = self._normalize_name(extracted_name)

        if not normalized_extracted:
            logger.warning(f"Name normalization resulted in empty string: '{extracted_name}'")
            return None, 0, {"error": "Invalid name after normalization", "original": extracted_name}

        try:
            # Get all normalized roster names for matching
            normalized_names = list(self.normalized_roster.keys())

            # Find best match using fuzzy matching
            result = process.extractOne(
                normalized_extracted,
                normalized_names,
                scorer=self.scorer
            )

            if result is None:
                logger.warning(f"No fuzzy match result for: '{extracted_name}'")
                return None, 0, {"error": "No match found", "original": extracted_name}

            best_match_normalized, confidence = result
            best_match_original = self.normalized_roster[best_match_normalized]

            # Prepare metadata
            metadata = {
                "original_extracted": extracted_name,
                "normalized_extracted": normalized_extracted,
                "matched_normalized": best_match_normalized,
                "confidence": confidence,
                "above_threshold": confidence >= self.similarity_threshold
            }

            # Add top candidates if requested
            if return_all_candidates:
                all_matches = process.extract(
                    normalized_extracted,
                    normalized_names,
                    scorer=self.scorer,
                    limit=5
                )
                metadata["top_candidates"] = [
                    {
                        "name": self.normalized_roster[match[0]],
                        "normalized": match[0],
                        "score": match[1]
                    }
                    for match in all_matches
                ]

            # Check if match meets threshold
            if confidence >= self.similarity_threshold:
                logger.debug(f"Successful match: '{extracted_name}' -> '{best_match_original}' (score: {confidence})")
                return best_match_original, confidence, metadata
            else:
                logger.info(f"Match below threshold: '{extracted_name}' -> '{best_match_original}' (score: {confidence} < {self.similarity_threshold})")
                return None, confidence, metadata

        except Exception as e:
            logger.error(f"Error during fuzzy matching for '{extracted_name}': {e}")
            return None, 0, {"error": str(e), "original": extracted_name}

    def batch_match_names(
        self,
        extracted_names: List[str]
    ) -> List[Tuple[str, Optional[str], int, Dict[str, Any]]]:
        """
        Match multiple names in batch.

        Args:
            extracted_names: List of names extracted from OCR

        Returns:
            List of tuples: (original_name, matched_name, confidence, metadata)
        """
        if not isinstance(extracted_names, list):
            raise ValueError("Extracted names must be a list")

        results = []
        successful_matches = 0

        for i, name in enumerate(extracted_names):
            matched_name, confidence, metadata = self.find_best_match(name, return_all_candidates=False)

            results.append((name, matched_name, confidence, metadata))

            if matched_name is not None:
                successful_matches += 1

            logger.debug(f"Batch match {i+1}/{len(extracted_names)}: '{name}' -> {matched_name}")

        success_rate = successful_matches / len(extracted_names) * 100 if extracted_names else 0
        logger.info(f"Batch matching: {successful_matches}/{len(extracted_names)} successful ({success_rate:.1f}%)")

        return results

    def get_match_statistics(self, results: List[Tuple[str, Optional[str], int, Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Generate statistics from batch matching results.

        Args:
            results: Results from batch_match_names

        Returns:
            Dictionary with matching statistics
        """
        if not results:
            return {"total": 0, "matched": 0, "success_rate": 0.0}

        total = len(results)
        matched = sum(1 for _, matched_name, _, _ in results if matched_name is not None)

        confidences = [confidence for _, _, confidence, _ in results if confidence > 0]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0

        # Count by confidence ranges
        high_confidence = sum(1 for conf in confidences if conf >= 90)
        medium_confidence = sum(1 for conf in confidences if 70 <= conf < 90)
        low_confidence = sum(1 for conf in confidences if conf < 70)

        return {
            "total": total,
            "matched": matched,
            "unmatched": total - matched,
            "success_rate": (matched / total * 100) if total > 0 else 0.0,
            "average_confidence": avg_confidence,
            "confidence_distribution": {
                "high (90-100)": high_confidence,
                "medium (70-89)": medium_confidence,
                "low (0-69)": low_confidence
            }
        }

    def _normalize_name(self, name: str) -> str:
        """
        Normalize a name for consistent matching.

        Handles common variations and OCR errors:
        - Convert to lowercase
        - Remove extra spaces and punctuation
        - Handle common OCR substitutions
        - Remove titles and suffixes
        """
        if not name:
            return ""

        # Convert to lowercase and strip
        normalized = name.lower().strip()

        # Remove common title prefixes
        prefixes = ["mr.", "mrs.", "ms.", "dr.", "prof.", "student:", "name:"]
        for prefix in prefixes:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):].strip()

        # Remove common suffixes
        suffixes = ["jr.", "sr.", "iii", "ii", "iv"]
        for suffix in suffixes:
            if normalized.endswith(suffix):
                normalized = normalized[:-len(suffix)].strip()

        # Handle common OCR errors and character substitutions
        ocr_corrections = {
            '0': 'o',
            '1': 'l',
            '5': 's',
            '8': 'b',
            'rn': 'm',
            'cl': 'd',
            'vv': 'w'
        }

        for wrong, correct in ocr_corrections.items():
            normalized = normalized.replace(wrong, correct)

        # Remove special characters but keep spaces, hyphens, and apostrophes
        normalized = re.sub(r"[^\w\s\-']", "", normalized)

        # Collapse multiple spaces
        normalized = re.sub(r'\s+', ' ', normalized)

        # Remove leading/trailing whitespace
        normalized = normalized.strip()

        return normalized

    def update_threshold(self, new_threshold: int) -> None:
        """Update the similarity threshold."""
        if not 0 <= new_threshold <= 100:
            raise ValueError("Threshold must be between 0 and 100")

        old_threshold = self.similarity_threshold
        self.similarity_threshold = new_threshold
        logger.info(f"Updated similarity threshold: {old_threshold} -> {new_threshold}")

    def get_service_info(self) -> Dict[str, Any]:
        """Get information about the service configuration."""
        return {
            "similarity_threshold": self.similarity_threshold,
            "scorer": self.scorer.__name__ if hasattr(self.scorer, '__name__') else str(self.scorer),
            "roster_size": len(self.student_roster),
            "normalized_entries": len(self.normalized_roster)
        }


def create_fuzzy_matching_service(
    similarity_threshold: int = 80,
    scorer=fuzz.token_sort_ratio
) -> FuzzyMatchingService:
    """Factory function to create a FuzzyMatchingService instance."""
    return FuzzyMatchingService(
        similarity_threshold=similarity_threshold,
        scorer=scorer
    )