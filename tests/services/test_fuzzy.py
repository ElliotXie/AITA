import pytest
from unittest.mock import patch, MagicMock
from fuzzywuzzy import fuzz

from aita.services.fuzzy import FuzzyMatchingService, create_fuzzy_matching_service


class TestFuzzyMatchingService:
    """Test suite for FuzzyMatchingService."""

    def setup_method(self):
        """Set up test fixtures."""
        self.service = FuzzyMatchingService(similarity_threshold=80)
        self.sample_roster = [
            "Smith, John",
            "Johnson, Mary",
            "Williams, David",
            "Brown, Lisa",
            "Davis, Michael",
            "Miller, Sarah",
            "Wilson, Robert",
            "Moore, Jennifer",
            "Taylor, Christopher",
            "Anderson, Amanda"
        ]

    def test_init_valid_threshold(self):
        """Test initialization with valid threshold."""
        service = FuzzyMatchingService(similarity_threshold=75)
        assert service.similarity_threshold == 75
        assert service.scorer == fuzz.token_sort_ratio

    def test_init_invalid_threshold(self):
        """Test initialization with invalid threshold."""
        with pytest.raises(ValueError, match="Similarity threshold must be between 0 and 100"):
            FuzzyMatchingService(similarity_threshold=150)

        with pytest.raises(ValueError, match="Similarity threshold must be between 0 and 100"):
            FuzzyMatchingService(similarity_threshold=-10)

    def test_load_student_roster_valid(self):
        """Test loading valid student roster."""
        self.service.load_student_roster(self.sample_roster)

        assert self.service.student_roster == self.sample_roster
        assert len(self.service.normalized_roster) == len(self.sample_roster)

        # Check that normalization happened
        assert "smith, john" in self.service.normalized_roster
        assert self.service.normalized_roster["smith, john"] == "Smith, John"

    def test_load_student_roster_empty(self):
        """Test loading empty roster."""
        self.service.load_student_roster([])
        assert self.service.student_roster == []
        assert self.service.normalized_roster == {}

    def test_load_student_roster_invalid_type(self):
        """Test loading roster with invalid type."""
        with pytest.raises(ValueError, match="Student names must be a list"):
            self.service.load_student_roster("not a list")

    def test_normalize_name_basic(self):
        """Test basic name normalization."""
        test_cases = [
            ("Smith, John", "smith, john"),
            ("JOHNSON, MARY", "johnson, mary"),
            ("  Williams,  David  ", "williams, david"),
            ("Brown   Lisa", "brown lisa"),
            ("Da-vis, Michael", "da-vis, michael"),
            ("O'Connor, Patrick", "o'connor, patrick")
        ]

        for input_name, expected in test_cases:
            result = self.service._normalize_name(input_name)
            assert result == expected, f"Failed for {input_name}: got {result}, expected {expected}"

    def test_normalize_name_ocr_corrections(self):
        """Test OCR error corrections in name normalization."""
        test_cases = [
            ("5mith, J0hn", "smith, john"),  # 5->s, 0->o
            ("J0hn50n, Mary", "johnson, mary"),  # 0->o, 5->s
            ("Wi11iams, David", "williams, david"),  # 1->l
            ("8rown, Lisa", "brown, lisa"),  # 8->b
            ("Dav1s, Michaei", "davis, michaei"),  # 1->l
        ]

        for input_name, expected in test_cases:
            result = self.service._normalize_name(input_name)
            assert result == expected, f"Failed OCR correction for {input_name}: got {result}, expected {expected}"

    def test_normalize_name_titles_and_suffixes(self):
        """Test removal of titles and suffixes."""
        test_cases = [
            ("Mr. Smith, John", "smith, john"),
            ("Dr. Johnson, Mary", "johnson, mary"),
            ("Smith, John Jr.", "smith, john"),
            ("Brown, Lisa III", "brown, lisa"),
            ("Name: Williams, David", "williams, david"),
            ("Student: Davis, Michael", "davis, michael")
        ]

        for input_name, expected in test_cases:
            result = self.service._normalize_name(input_name)
            assert result == expected, f"Failed title/suffix removal for {input_name}: got {result}, expected {expected}"

    def test_find_best_match_exact(self):
        """Test finding best match with exact name."""
        self.service.load_student_roster(self.sample_roster)

        matched_name, confidence, metadata = self.service.find_best_match("Smith, John")

        assert matched_name == "Smith, John"
        assert confidence == 100
        assert metadata["above_threshold"] is True
        assert metadata["original_extracted"] == "Smith, John"

    def test_find_best_match_fuzzy(self):
        """Test finding best match with fuzzy matching."""
        self.service.load_student_roster(self.sample_roster)

        # Test with common OCR errors
        matched_name, confidence, metadata = self.service.find_best_match("5mith, J0hn")

        assert matched_name == "Smith, John"
        assert confidence >= 80
        assert metadata["above_threshold"] is True

    def test_find_best_match_below_threshold(self):
        """Test finding match below threshold."""
        self.service.load_student_roster(self.sample_roster)

        # Test with very different name
        matched_name, confidence, metadata = self.service.find_best_match("Xyz, Abc")

        assert matched_name is None
        assert confidence < 80
        assert metadata["above_threshold"] is False

    def test_find_best_match_empty_name(self):
        """Test finding match with empty name."""
        self.service.load_student_roster(self.sample_roster)

        matched_name, confidence, metadata = self.service.find_best_match("")

        assert matched_name is None
        assert confidence == 0
        assert "Empty input name" in metadata["error"]

    def test_find_best_match_no_roster(self):
        """Test finding match without loaded roster."""
        matched_name, confidence, metadata = self.service.find_best_match("Smith, John")

        assert matched_name is None
        assert confidence == 0
        assert "No roster loaded" in metadata["error"]

    def test_find_best_match_with_candidates(self):
        """Test finding match with candidate information."""
        self.service.load_student_roster(self.sample_roster)

        matched_name, confidence, metadata = self.service.find_best_match(
            "Smith, John",
            return_all_candidates=True
        )

        assert matched_name == "Smith, John"
        assert "top_candidates" in metadata
        assert len(metadata["top_candidates"]) <= 5
        assert all("name" in candidate for candidate in metadata["top_candidates"])

    def test_batch_match_names(self):
        """Test batch name matching."""
        self.service.load_student_roster(self.sample_roster)

        test_names = [
            "Smith, John",
            "Johnson, Mary",
            "Unknown Person",
            "5mith, J0hn"  # OCR errors
        ]

        results = self.service.batch_match_names(test_names)

        assert len(results) == 4

        # Check first result (exact match)
        original, matched, confidence, metadata = results[0]
        assert original == "Smith, John"
        assert matched == "Smith, John"
        assert confidence == 100

        # Check third result (no match)
        original, matched, confidence, metadata = results[2]
        assert original == "Unknown Person"
        assert matched is None

    def test_batch_match_names_invalid_input(self):
        """Test batch matching with invalid input."""
        with pytest.raises(ValueError, match="Extracted names must be a list"):
            self.service.batch_match_names("not a list")

    def test_get_match_statistics_empty(self):
        """Test statistics with empty results."""
        stats = self.service.get_match_statistics([])

        assert stats["total"] == 0
        assert stats["matched"] == 0
        assert stats["success_rate"] == 0.0

    def test_get_match_statistics_with_results(self):
        """Test statistics with actual results."""
        # Mock results
        results = [
            ("Name1", "Matched1", 95, {}),
            ("Name2", "Matched2", 85, {}),
            ("Name3", None, 65, {}),
            ("Name4", "Matched4", 75, {})
        ]

        stats = self.service.get_match_statistics(results)

        assert stats["total"] == 4
        assert stats["matched"] == 3
        assert stats["unmatched"] == 1
        assert stats["success_rate"] == 75.0
        assert stats["confidence_distribution"]["high (90-100)"] == 1
        assert stats["confidence_distribution"]["medium (70-89)"] == 2
        assert stats["confidence_distribution"]["low (0-69)"] == 1

    def test_update_threshold(self):
        """Test updating similarity threshold."""
        initial_threshold = self.service.similarity_threshold
        self.service.update_threshold(90)
        assert self.service.similarity_threshold == 90

        with pytest.raises(ValueError, match="Threshold must be between 0 and 100"):
            self.service.update_threshold(150)

    def test_get_service_info(self):
        """Test getting service information."""
        self.service.load_student_roster(self.sample_roster)

        info = self.service.get_service_info()

        assert info["similarity_threshold"] == 80
        assert info["roster_size"] == len(self.sample_roster)
        assert info["normalized_entries"] == len(self.sample_roster)
        assert "scorer" in info

    def test_factory_function(self):
        """Test factory function."""
        service = create_fuzzy_matching_service(similarity_threshold=75)

        assert isinstance(service, FuzzyMatchingService)
        assert service.similarity_threshold == 75

    def test_edge_case_special_characters(self):
        """Test handling of special characters."""
        roster = ["O'Connor, Patrick", "Smith-Jones, Mary", "De La Cruz, Carlos"]
        self.service.load_student_roster(roster)

        # Test matching with various special character issues
        test_cases = [
            ("O Connor, Patrick", "O'Connor, Patrick"),
            ("Smith Jones, Mary", "Smith-Jones, Mary"),
            ("De La Cruz, Carlos", "De La Cruz, Carlos")
        ]

        for test_name, expected in test_cases:
            matched_name, confidence, metadata = self.service.find_best_match(test_name)
            assert matched_name == expected or confidence >= 70  # Should get good match

    def test_case_insensitive_matching(self):
        """Test case insensitive matching."""
        roster = ["Smith, John", "JOHNSON, MARY", "williams, david"]
        self.service.load_student_roster(roster)

        test_cases = [
            ("smith, john", "Smith, John"),
            ("SMITH, JOHN", "Smith, John"),
            ("Johnson, Mary", "JOHNSON, MARY"),
            ("WILLIAMS, DAVID", "williams, david")
        ]

        for test_name, expected in test_cases:
            matched_name, confidence, metadata = self.service.find_best_match(test_name)
            assert matched_name == expected
            assert confidence >= 90  # Should get very high confidence for case differences


if __name__ == "__main__":
    pytest.main([__file__])