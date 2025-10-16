"""
Unit tests for transcription pipeline.

Tests transcription logic, JSON parsing, error handling, and data structures.
"""

import pytest
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os

from aita.domain.models import Student, StudentAnswer
from aita.pipelines.transcribe import (
    TranscriptionPipeline,
    PageTranscription,
    StudentTranscriptionResult,
    TranscriptionResults,
    TranscriptionError,
    create_transcription_pipeline,
    transcribe_all_students,
    transcribe_single_student
)


class TestJSONExtraction:
    """Test JSON extraction from transcription responses."""

    def test_extract_json_from_valid_response(self):
        """Test extracting JSON from a well-formed response."""
        pipeline = self._create_mock_pipeline()

        response = json.dumps({
            "transcribed_text": "The answer is 42",
            "confidence": 0.95,
            "notes": "Clear handwriting"
        })

        result = pipeline._parse_transcription_response(response)

        assert result['transcribed_text'] == "The answer is 42"
        assert result['confidence'] == 0.95
        assert result['notes'] == "Clear handwriting"

    def test_extract_json_with_markdown_wrapping(self):
        """Test extracting JSON from markdown code block."""
        pipeline = self._create_mock_pipeline()

        response = """
        Here's the transcription:
        ```json
        {
            "transcribed_text": "Mathematical formula: x = y + z",
            "confidence": 0.88,
            "notes": "Some unclear notation"
        }
        ```
        """

        result = pipeline._parse_transcription_response(response)

        assert result['transcribed_text'] == "Mathematical formula: x = y + z"
        assert result['confidence'] == 0.88

    def test_extract_json_without_markdown(self):
        """Test extracting raw JSON."""
        pipeline = self._create_mock_pipeline()

        response = '{"transcribed_text": "Simple answer", "confidence": 0.75}'

        result = pipeline._parse_transcription_response(response)

        assert result['transcribed_text'] == "Simple answer"
        assert result['confidence'] == 0.75

    def test_parse_response_with_invalid_confidence(self):
        """Test that invalid confidence values are normalized."""
        pipeline = self._create_mock_pipeline()

        response = json.dumps({
            "transcribed_text": "Answer",
            "confidence": 1.5,  # Invalid - over 1.0
            "notes": "Test"
        })

        result = pipeline._parse_transcription_response(response)

        assert result['transcribed_text'] == "Answer"
        assert result['confidence'] == 0.5  # Should be normalized

    def test_parse_response_fallback_to_raw_text(self):
        """Test fallback when JSON parsing fails."""
        pipeline = self._create_mock_pipeline()

        response = "This is not JSON at all, just raw text response"

        result = pipeline._parse_transcription_response(response)

        assert result['transcribed_text'] == response.strip()
        assert result['confidence'] == 0.3  # Lower confidence for raw text
        assert "JSON parsing failed" in result['notes']

    def test_parse_response_missing_required_field(self):
        """Test handling when required fields are missing."""
        pipeline = self._create_mock_pipeline()

        response = json.dumps({
            "confidence": 0.9,
            "notes": "Missing transcribed_text field"
        })

        # Should raise ValueError for missing required field
        with pytest.raises(ValueError, match="Missing 'transcribed_text'"):
            pipeline._parse_transcription_response(response)

    def _create_mock_pipeline(self):
        """Create a mock pipeline for testing."""
        mock_llm = Mock()
        mock_storage = Mock()
        return TranscriptionPipeline(
            llm_client=mock_llm,
            storage_service=mock_storage,
            data_dir=Path("/tmp"),
            assignment_name="test"
        )


class TestPageTranscription:
    """Test PageTranscription data structure."""

    def test_page_transcription_creation(self):
        """Test creating PageTranscription object."""
        page = PageTranscription(
            page_number=1,
            image_path=Path("/test/page1.jpg"),
            transcribed_text="Student answer here",
            confidence=0.92,
            notes="Good quality image",
            processing_time=2.5,
            public_url="https://storage.googleapis.com/test/page1.jpg"
        )

        assert page.page_number == 1
        assert page.image_path == Path("/test/page1.jpg")
        assert page.transcribed_text == "Student answer here"
        assert page.confidence == 0.92
        assert page.notes == "Good quality image"
        assert page.processing_time == 2.5
        assert page.public_url == "https://storage.googleapis.com/test/page1.jpg"

    def test_page_transcription_with_defaults(self):
        """Test PageTranscription with default values."""
        page = PageTranscription(
            page_number=2,
            image_path=Path("/test/page2.jpg"),
            transcribed_text="Answer",
            confidence=0.8
        )

        assert page.notes is None
        assert page.processing_time is None
        assert page.public_url is None
        assert page.raw_llm_response is None


class TestStudentTranscriptionResult:
    """Test StudentTranscriptionResult data structure."""

    def test_student_result_success_rate(self):
        """Test success rate calculation."""
        student = Student(name="Test Student")

        result = StudentTranscriptionResult(
            student_name="Test Student",
            student=student,
            total_pages=5,
            successful_pages=4,
            failed_pages=1
        )

        assert result.success_rate == 80.0

    def test_student_result_success_rate_zero_pages(self):
        """Test success rate with zero pages."""
        student = Student(name="Test Student")

        result = StudentTranscriptionResult(
            student_name="Test Student",
            student=student,
            total_pages=0,
            successful_pages=0,
            failed_pages=0
        )

        assert result.success_rate == 0.0

    def test_student_result_average_confidence(self):
        """Test average confidence calculation."""
        pages = [
            PageTranscription(1, Path("/p1.jpg"), "text1", 0.9),
            PageTranscription(2, Path("/p2.jpg"), "text2", 0.8),
            PageTranscription(3, Path("/p3.jpg"), "text3", 0.0),  # Failed page
            PageTranscription(4, Path("/p4.jpg"), "text4", 0.95)
        ]

        student = Student(name="Test Student")
        result = StudentTranscriptionResult(
            student_name="Test Student",
            student=student,
            pages=pages
        )

        # Should average only non-zero confidences: (0.9 + 0.8 + 0.95) / 3 = 0.883...
        assert abs(result.average_confidence - 0.8833333333333333) < 0.001

    def test_student_result_average_confidence_no_successful_pages(self):
        """Test average confidence with no successful pages."""
        pages = [
            PageTranscription(1, Path("/p1.jpg"), "", 0.0),  # Failed
            PageTranscription(2, Path("/p2.jpg"), "", 0.0)   # Failed
        ]

        student = Student(name="Test Student")
        result = StudentTranscriptionResult(
            student_name="Test Student",
            student=student,
            pages=pages
        )

        assert result.average_confidence == 0.0


class TestTranscriptionResults:
    """Test TranscriptionResults data structure."""

    def test_transcription_results_success_rate(self):
        """Test overall success rate calculation."""
        results = TranscriptionResults(
            total_students=3,
            successful_students=2,
            total_pages=15,
            successful_pages=12
        )

        assert results.success_rate == 80.0

    def test_transcription_results_average_confidence(self):
        """Test overall average confidence calculation."""
        student1_pages = [
            PageTranscription(1, Path("/s1p1.jpg"), "text", 0.9),
            PageTranscription(2, Path("/s1p2.jpg"), "text", 0.8)
        ]

        student2_pages = [
            PageTranscription(1, Path("/s2p1.jpg"), "text", 0.95),
            PageTranscription(2, Path("/s2p2.jpg"), "", 0.0)  # Failed
        ]

        student1 = StudentTranscriptionResult(
            student_name="Student 1",
            student=Student("Student 1"),
            pages=student1_pages
        )

        student2 = StudentTranscriptionResult(
            student_name="Student 2",
            student=Student("Student 2"),
            pages=student2_pages
        )

        results = TranscriptionResults(
            total_students=2,
            successful_students=2,
            total_pages=4,
            successful_pages=3,
            student_results=[student1, student2]
        )

        # Should average: (0.9 + 0.8 + 0.95) / 3 = 0.883...
        assert abs(results.average_confidence - 0.8833333333333333) < 0.001


class TestTranscriptionPipelineInit:
    """Test TranscriptionPipeline initialization."""

    def test_pipeline_initialization(self):
        """Test pipeline initialization with required parameters."""
        mock_llm = Mock()
        mock_storage = Mock()
        data_dir = Path("/test/data")

        pipeline = TranscriptionPipeline(
            llm_client=mock_llm,
            storage_service=mock_storage,
            data_dir=data_dir,
            assignment_name="test_exam"
        )

        assert pipeline.llm_client == mock_llm
        assert pipeline.storage_service == mock_storage
        assert pipeline.data_dir == data_dir
        assert pipeline.assignment_name == "test_exam"
        assert pipeline.max_retries == 3
        assert pipeline.retry_delay == 1.0

        # Check directory paths
        assert pipeline.grouped_dir == data_dir / "grouped"
        assert pipeline.output_dir == data_dir / "intermediateproduct" / "transcription_results"

    def test_pipeline_initialization_with_custom_params(self):
        """Test pipeline initialization with custom parameters."""
        mock_llm = Mock()
        mock_storage = Mock()

        pipeline = TranscriptionPipeline(
            llm_client=mock_llm,
            storage_service=mock_storage,
            data_dir=Path("/test"),
            assignment_name="custom_exam",
            max_retries=5,
            retry_delay=2.0
        )

        assert pipeline.assignment_name == "custom_exam"
        assert pipeline.max_retries == 5
        assert pipeline.retry_delay == 2.0


class TestTranscriptionPipelineMethods:
    """Test TranscriptionPipeline core methods."""

    def test_find_student_images(self):
        """Test finding image files in student folder."""
        with tempfile.TemporaryDirectory() as temp_dir:
            student_dir = Path(temp_dir) / "student1"
            student_dir.mkdir()

            # Create test files
            (student_dir / "page_001.jpg").touch()
            (student_dir / "page_002.png").touch()
            (student_dir / "not_image.txt").touch()
            (student_dir / "page_003.jpeg").touch()

            pipeline = self._create_test_pipeline(temp_dir)
            images = pipeline._find_student_images(student_dir)

            assert len(images) == 3
            # Check that files are sorted by name
            assert images[0].name == "page_001.jpg"
            assert images[1].name == "page_002.png"
            assert images[2].name == "page_003.jpeg"

    def test_find_student_images_empty_folder(self):
        """Test finding images in empty folder."""
        with tempfile.TemporaryDirectory() as temp_dir:
            student_dir = Path(temp_dir) / "student1"
            student_dir.mkdir()

            pipeline = self._create_test_pipeline(temp_dir)
            images = pipeline._find_student_images(student_dir)

            assert len(images) == 0

    def test_discover_student_folders(self):
        """Test discovering student folders in grouped directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            grouped_dir = Path(temp_dir) / "grouped"
            grouped_dir.mkdir()

            # Create test student folders
            (grouped_dir / "Student_001").mkdir()
            (grouped_dir / "Student_002").mkdir()
            (grouped_dir / ".hidden_folder").mkdir()  # Should be ignored
            (grouped_dir / "test_file.txt").touch()  # Should be ignored

            pipeline = self._create_test_pipeline(temp_dir)
            folders = pipeline._discover_student_folders()

            assert len(folders) == 2
            folder_names = [f.name for f in folders]
            assert "Student_001" in folder_names
            assert "Student_002" in folder_names
            assert ".hidden_folder" not in folder_names

    def test_discover_student_folders_missing_grouped_dir(self):
        """Test error when grouped directory doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline = self._create_test_pipeline(temp_dir)

            with pytest.raises(TranscriptionError, match="Grouped directory not found"):
                pipeline._discover_student_folders()

    @patch('time.sleep')  # Mock sleep to speed up tests
    def test_transcribe_page_with_retries(self, mock_sleep):
        """Test page transcription with retry logic."""
        mock_llm = Mock()
        mock_storage = Mock()

        # First two calls fail, third succeeds
        mock_storage.upload_image.side_effect = [
            Exception("Upload failed"),
            Exception("Upload failed again"),
            "https://storage.googleapis.com/test/page1.jpg"
        ]

        mock_llm.complete_with_image.return_value = json.dumps({
            "transcribed_text": "Success on retry",
            "confidence": 0.85,
            "notes": "Finally worked"
        })

        pipeline = TranscriptionPipeline(
            llm_client=mock_llm,
            storage_service=mock_storage,
            data_dir=Path("/test"),
            assignment_name="test",
            max_retries=3
        )

        result = pipeline._transcribe_page(
            image_path=Path("/test/page1.jpg"),
            student_name="Test Student",
            page_number=1
        )

        assert result.transcribed_text == "Success on retry"
        assert result.confidence == 0.85
        assert mock_storage.upload_image.call_count == 3
        assert mock_sleep.call_count == 2  # Two retries with sleep

    @patch('time.sleep')
    def test_transcribe_page_all_retries_fail(self, mock_sleep):
        """Test page transcription when all retries fail."""
        mock_llm = Mock()
        mock_storage = Mock()

        # All calls fail
        mock_storage.upload_image.side_effect = Exception("Persistent failure")

        pipeline = TranscriptionPipeline(
            llm_client=mock_llm,
            storage_service=mock_storage,
            data_dir=Path("/test"),
            assignment_name="test",
            max_retries=2
        )

        result = pipeline._transcribe_page(
            image_path=Path("/test/page1.jpg"),
            student_name="Test Student",
            page_number=1
        )

        assert result.transcribed_text == ""
        assert result.confidence == 0.0
        assert "failed after 2 attempts" in result.notes
        assert mock_storage.upload_image.call_count == 2

    def _create_test_pipeline(self, temp_dir):
        """Helper to create pipeline for testing."""
        mock_llm = Mock()
        mock_storage = Mock()
        return TranscriptionPipeline(
            llm_client=mock_llm,
            storage_service=mock_storage,
            data_dir=Path(temp_dir),
            assignment_name="test"
        )


class TestFactoryFunctions:
    """Test factory and convenience functions."""

    @patch('aita.pipelines.transcribe.create_openrouter_client')
    @patch('aita.pipelines.transcribe.create_storage_service')
    @patch('aita.pipelines.transcribe.get_config')
    def test_create_transcription_pipeline(self, mock_config, mock_storage, mock_llm):
        """Test factory function for creating pipeline."""
        mock_config.return_value.data_dir = "/test/data"
        mock_llm_instance = Mock()
        mock_storage_instance = Mock()
        mock_llm.return_value = mock_llm_instance
        mock_storage.return_value = mock_storage_instance

        pipeline = create_transcription_pipeline("test_assignment")

        assert pipeline.llm_client == mock_llm_instance
        assert pipeline.storage_service == mock_storage_instance
        assert pipeline.assignment_name == "test_assignment"

        mock_llm.assert_called_once()
        mock_storage.assert_called_once()

    @patch('aita.pipelines.transcribe.create_transcription_pipeline')
    def test_transcribe_all_students_convenience_function(self, mock_create_pipeline):
        """Test convenience function for transcribing all students."""
        mock_pipeline = Mock()
        mock_results = Mock()
        mock_pipeline.transcribe_all_students.return_value = mock_results
        mock_create_pipeline.return_value = mock_pipeline

        results = transcribe_all_students("test_assignment")

        assert results == mock_results
        mock_create_pipeline.assert_called_once_with("test_assignment", None)
        mock_pipeline.transcribe_all_students.assert_called_once()

    @patch('aita.pipelines.transcribe.create_transcription_pipeline')
    def test_transcribe_single_student_convenience_function(self, mock_create_pipeline):
        """Test convenience function for transcribing single student."""
        mock_pipeline = Mock()
        mock_pipeline.data_dir = Path("/test/data")
        mock_results = Mock()
        mock_pipeline._transcribe_student.return_value = mock_results
        mock_create_pipeline.return_value = mock_pipeline

        with tempfile.TemporaryDirectory() as temp_dir:
            student_dir = Path(temp_dir) / "grouped" / "Test_Student"
            student_dir.mkdir(parents=True)

            # Mock the pipeline's data_dir to point to our temp directory
            mock_pipeline.data_dir = Path(temp_dir)

            result = transcribe_single_student("Test_Student", "test_assignment", Path(temp_dir))

            assert result == mock_results
            mock_create_pipeline.assert_called_once_with("test_assignment", Path(temp_dir))
            mock_pipeline._transcribe_student.assert_called_once()

    @patch('aita.pipelines.transcribe.create_transcription_pipeline')
    def test_transcribe_single_student_not_found(self, mock_create_pipeline):
        """Test single student transcription when folder doesn't exist."""
        mock_pipeline = Mock()
        mock_pipeline.data_dir = Path("/nonexistent")
        mock_create_pipeline.return_value = mock_pipeline

        with pytest.raises(TranscriptionError, match="Student folder not found"):
            transcribe_single_student("Nonexistent_Student")


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_parse_response_with_special_characters(self):
        """Test parsing responses with special mathematical characters."""
        pipeline = self._create_mock_pipeline()

        response = json.dumps({
            "transcribed_text": "∫(x²+1)dx = x³/3 + x + C\n∑i=1→∞ 1/i² = π²/6",
            "confidence": 0.88,
            "notes": "Mathematical notation detected"
        })

        result = pipeline._parse_transcription_response(response)

        assert "∫" in result['transcribed_text']
        assert "∑" in result['transcribed_text']
        assert "π²" in result['transcribed_text']

    def test_parse_response_with_crossed_out_text(self):
        """Test parsing responses with crossed-out text notation."""
        pipeline = self._create_mock_pipeline()

        response = json.dumps({
            "transcribed_text": "Initial answer: ~~x = 5~~ Corrected: x = 7",
            "confidence": 0.75,
            "notes": "Student made corrections"
        })

        result = pipeline._parse_transcription_response(response)

        assert "~~x = 5~~" in result['transcribed_text']
        assert "x = 7" in result['transcribed_text']

    def test_parse_response_with_diagram_description(self):
        """Test parsing responses with diagram descriptions."""
        pipeline = self._create_mock_pipeline()

        response = json.dumps({
            "transcribed_text": "The graph shows [DIAGRAM: parabola opening upward with vertex at (0,1)]",
            "confidence": 0.82,
            "notes": "Diagram interpretation included"
        })

        result = pipeline._parse_transcription_response(response)

        assert "[DIAGRAM:" in result['transcribed_text']
        assert "parabola opening upward" in result['transcribed_text']

    def test_transcription_results_with_empty_students(self):
        """Test TranscriptionResults with no students."""
        results = TranscriptionResults(
            total_students=0,
            successful_students=0,
            total_pages=0,
            successful_pages=0
        )

        assert results.success_rate == 0.0
        assert results.average_confidence == 0.0

    def test_student_result_with_mixed_success_failure(self):
        """Test student result with mix of successful and failed pages."""
        pages = [
            PageTranscription(1, Path("/p1.jpg"), "Good answer", 0.95),
            PageTranscription(2, Path("/p2.jpg"), "", 0.0, "Transcription failed"),
            PageTranscription(3, Path("/p3.jpg"), "Another answer", 0.87),
            PageTranscription(4, Path("/p4.jpg"), "", 0.0, "Upload failed")
        ]

        student = Student(name="Test Student")
        result = StudentTranscriptionResult(
            student_name="Test Student",
            student=student,
            pages=pages,
            total_pages=4,
            successful_pages=2,
            failed_pages=2
        )

        assert result.success_rate == 50.0
        assert abs(result.average_confidence - 0.91) < 0.01  # (0.95 + 0.87) / 2

    def _create_mock_pipeline(self):
        """Create a mock pipeline for testing."""
        mock_llm = Mock()
        mock_storage = Mock()
        return TranscriptionPipeline(
            llm_client=mock_llm,
            storage_service=mock_storage,
            data_dir=Path("/tmp"),
            assignment_name="test"
        )


class TestIntegrationScenarios:
    """Test integration scenarios and realistic use cases."""

    def test_full_transcription_workflow_simulation(self):
        """Test a complete transcription workflow with mocked services."""
        # Setup mocks
        mock_llm = Mock()
        mock_storage = Mock()

        # Mock successful transcription responses
        transcription_responses = [
            json.dumps({
                "transcribed_text": "Answer to question 1: The mean is 42.5",
                "confidence": 0.92,
                "notes": "Clear handwriting"
            }),
            json.dumps({
                "transcribed_text": "Question 2: [DIAGRAM: scatter plot showing positive correlation]",
                "confidence": 0.85,
                "notes": "Diagram included"
            }),
            json.dumps({
                "transcribed_text": "Final answer: p = 0.023, reject null hypothesis",
                "confidence": 0.88,
                "notes": "Statistical conclusion"
            })
        ]

        mock_llm.complete_with_image.side_effect = transcription_responses
        mock_storage.upload_image.side_effect = [
            "https://storage.googleapis.com/test/page1.jpg",
            "https://storage.googleapis.com/test/page2.jpg",
            "https://storage.googleapis.com/test/page3.jpg"
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test student folder with images
            student_dir = Path(temp_dir) / "grouped" / "Test_Student"
            student_dir.mkdir(parents=True)

            for i in range(1, 4):
                (student_dir / f"page_{i:03d}.jpg").touch()

            pipeline = TranscriptionPipeline(
                llm_client=mock_llm,
                storage_service=mock_storage,
                data_dir=Path(temp_dir),
                assignment_name="test_exam"
            )

            # Test single student transcription
            result = pipeline._transcribe_student(student_dir)

            assert result.student_name == "Test_Student"
            assert result.total_pages == 3
            assert result.successful_pages == 3
            assert result.failed_pages == 0
            assert len(result.pages) == 3

            # Check individual page results
            assert "mean is 42.5" in result.pages[0].transcribed_text
            assert "scatter plot" in result.pages[1].transcribed_text
            assert "reject null hypothesis" in result.pages[2].transcribed_text

            # Verify all pages have good confidence
            assert all(page.confidence > 0.8 for page in result.pages)

    def test_error_recovery_during_batch_processing(self):
        """Test that pipeline continues processing when individual students fail."""
        mock_llm = Mock()
        mock_storage = Mock()

        # First student succeeds, second fails, third succeeds
        mock_storage.upload_image.side_effect = [
            "https://storage.googleapis.com/test/s1p1.jpg",  # Student 1 succeeds
            Exception("Network error"),  # Student 2 fails
            "https://storage.googleapis.com/test/s3p1.jpg"   # Student 3 succeeds
        ]

        mock_llm.complete_with_image.side_effect = [
            json.dumps({"transcribed_text": "Student 1 answer", "confidence": 0.9}),
            json.dumps({"transcribed_text": "Student 3 answer", "confidence": 0.85})
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create grouped directory with multiple students
            grouped_dir = Path(temp_dir) / "grouped"
            grouped_dir.mkdir()

            for student_name in ["Student_001", "Student_002", "Student_003"]:
                student_dir = grouped_dir / student_name
                student_dir.mkdir()
                (student_dir / "page_001.jpg").touch()

            pipeline = TranscriptionPipeline(
                llm_client=mock_llm,
                storage_service=mock_storage,
                data_dir=Path(temp_dir),
                assignment_name="test_exam",
                max_retries=1  # Quick failure for testing
            )

            # Should not raise exception, but continue processing
            results = pipeline.transcribe_all_students()

            assert results.total_students == 3
            assert results.successful_students == 2  # Only 2 successful
            assert len(results.student_results) == 3  # All attempted

            # Check that successful students have good results
            successful_results = [r for r in results.student_results if r.successful_pages > 0]
            assert len(successful_results) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])