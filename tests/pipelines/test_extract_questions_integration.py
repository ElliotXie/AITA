"""
Integration tests for question extraction pipeline.

Tests the full end-to-end pipeline with real services (or mocked as needed).
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import json

from aita.pipelines.extract_questions import (
    QuestionExtractionPipeline,
    extract_questions_from_student,
    QuestionExtractionError
)
from aita.domain.models import ExamSpec, Question, QuestionType


@pytest.fixture
def mock_llm_client():
    """Create a mock LLM client."""
    mock = Mock()
    mock.complete_with_images = Mock(return_value=json.dumps({
        "exam_name": "BMI541 Midterm",
        "total_pages": 5,
        "questions": [
            {
                "question_id": "1a",
                "question_text": "Calculate the mean and standard deviation of the following dataset.",
                "points": 5.0,
                "page_number": 1,
                "question_type": "calculation"
            },
            {
                "question_id": "1b",
                "question_text": "Interpret the results in the context of the problem.",
                "points": 3.0,
                "page_number": 1,
                "question_type": "short_answer"
            },
            {
                "question_id": "2",
                "question_text": "Explain the difference between Type I and Type II errors.",
                "points": 8.0,
                "page_number": 2,
                "question_type": "long_answer"
            },
            {
                "question_id": "3a",
                "question_text": "What is the probability that X > 1.5?",
                "points": 4.0,
                "page_number": 3,
                "question_type": "calculation"
            },
            {
                "question_id": "3b",
                "question_text": "Draw the distribution and shade the relevant area.",
                "points": 3.0,
                "page_number": 3,
                "question_type": "diagram"
            }
        ]
    }))
    return mock


@pytest.fixture
def mock_storage_service():
    """Create a mock GCS storage service."""
    mock = Mock()
    mock.upload_images_batch = Mock(return_value=[
        {
            'local_path': 'page_1.jpg',
            'gcs_path': 'exam1/2024-10-15/student_001/page_1.jpg',
            'public_url': 'https://storage.googleapis.com/bucket/exam1/2024-10-15/student_001/page_1.jpg'
        },
        {
            'local_path': 'page_2.jpg',
            'gcs_path': 'exam1/2024-10-15/student_001/page_2.jpg',
            'public_url': 'https://storage.googleapis.com/bucket/exam1/2024-10-15/student_001/page_2.jpg'
        },
        {
            'local_path': 'page_3.jpg',
            'gcs_path': 'exam1/2024-10-15/student_001/page_3.jpg',
            'public_url': 'https://storage.googleapis.com/bucket/exam1/2024-10-15/student_001/page_3.jpg'
        },
        {
            'local_path': 'page_4.jpg',
            'gcs_path': 'exam1/2024-10-15/student_001/page_4.jpg',
            'public_url': 'https://storage.googleapis.com/bucket/exam1/2024-10-15/student_001/page_4.jpg'
        },
        {
            'local_path': 'page_5.jpg',
            'gcs_path': 'exam1/2024-10-15/student_001/page_5.jpg',
            'public_url': 'https://storage.googleapis.com/bucket/exam1/2024-10-15/student_001/page_5.jpg'
        }
    ])
    return mock


@pytest.fixture
def temp_student_folder(tmp_path):
    """Create a temporary student folder with mock image files."""
    student_dir = tmp_path / "Student_001"
    student_dir.mkdir()

    # Create 5 mock image files
    for i in range(1, 6):
        image_file = student_dir / f"page_{i}.jpg"
        image_file.write_bytes(b"fake image data")

    return student_dir


class TestQuestionExtractionPipeline:
    """Test the QuestionExtractionPipeline class."""

    def test_pipeline_initialization(self, mock_llm_client, mock_storage_service, tmp_path):
        """Test that pipeline initializes correctly."""
        pipeline = QuestionExtractionPipeline(
            llm_client=mock_llm_client,
            storage_service=mock_storage_service,
            data_dir=tmp_path
        )

        assert pipeline.llm_client is mock_llm_client
        assert pipeline.storage_service is mock_storage_service
        assert pipeline.data_dir == tmp_path

    def test_find_exam_images(
        self,
        mock_llm_client,
        mock_storage_service,
        temp_student_folder
    ):
        """Test finding exam images in student folder."""
        pipeline = QuestionExtractionPipeline(
            llm_client=mock_llm_client,
            storage_service=mock_storage_service,
            data_dir=temp_student_folder.parent
        )

        images = pipeline._find_exam_images(temp_student_folder)

        assert len(images) == 5
        assert all(img.suffix == '.jpg' for img in images)
        assert all(img.exists() for img in images)

    def test_find_exam_images_no_images_raises_error(
        self,
        mock_llm_client,
        mock_storage_service,
        tmp_path
    ):
        """Test that missing images raises error."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        pipeline = QuestionExtractionPipeline(
            llm_client=mock_llm_client,
            storage_service=mock_storage_service,
            data_dir=tmp_path
        )

        with pytest.raises(QuestionExtractionError, match="No image files found"):
            pipeline._find_exam_images(empty_dir)

    def test_upload_images_to_gcs(
        self,
        mock_llm_client,
        mock_storage_service,
        temp_student_folder
    ):
        """Test uploading images to GCS."""
        pipeline = QuestionExtractionPipeline(
            llm_client=mock_llm_client,
            storage_service=mock_storage_service,
            data_dir=temp_student_folder.parent
        )

        images = pipeline._find_exam_images(temp_student_folder)
        urls = pipeline._upload_images_to_gcs(
            images,
            "student_001",
            "exam1"
        )

        assert len(urls) == 5
        assert all('https://storage.googleapis.com' in url for url in urls)
        mock_storage_service.upload_images_batch.assert_called_once()

    def test_upload_images_all_fail_raises_error(
        self,
        mock_llm_client,
        mock_storage_service,
        temp_student_folder
    ):
        """Test that all upload failures raises error."""
        # Mock all uploads failing
        mock_storage_service.upload_images_batch = Mock(return_value=[
            {'local_path': 'page_1.jpg', 'error': 'Upload failed'},
            {'local_path': 'page_2.jpg', 'error': 'Upload failed'}
        ])

        pipeline = QuestionExtractionPipeline(
            llm_client=mock_llm_client,
            storage_service=mock_storage_service,
            data_dir=temp_student_folder.parent
        )

        images = pipeline._find_exam_images(temp_student_folder)

        with pytest.raises(QuestionExtractionError, match="All image uploads failed"):
            pipeline._upload_images_to_gcs(images, "student", "exam")

    def test_extract_with_llm(
        self,
        mock_llm_client,
        mock_storage_service,
        tmp_path
    ):
        """Test LLM extraction."""
        pipeline = QuestionExtractionPipeline(
            llm_client=mock_llm_client,
            storage_service=mock_storage_service,
            data_dir=tmp_path
        )

        urls = [f"https://example.com/page_{i}.jpg" for i in range(1, 6)]
        exam_spec = pipeline._extract_with_llm(urls, "BMI541 Midterm", 5)

        assert exam_spec.exam_name == "BMI541 Midterm"
        assert exam_spec.total_pages == 5
        assert len(exam_spec.questions) == 5

        # Verify LLM was called correctly
        mock_llm_client.complete_with_images.assert_called_once()
        call_args = mock_llm_client.complete_with_images.call_args
        assert call_args.kwargs['image_urls'] == urls
        assert call_args.kwargs['temperature'] == 0.1

    def test_extract_with_llm_retries_on_json_error(
        self,
        mock_llm_client,
        mock_storage_service,
        tmp_path
    ):
        """Test that LLM extraction retries on JSON parse errors."""
        # First call returns invalid JSON, second call returns valid JSON
        mock_llm_client.complete_with_images = Mock(side_effect=[
            "This is not valid JSON!",
            json.dumps({
                "exam_name": "Test",
                "total_pages": 2,
                "questions": [
                    {"question_id": "1", "question_text": "Q1", "points": 10}
                ]
            })
        ])

        pipeline = QuestionExtractionPipeline(
            llm_client=mock_llm_client,
            storage_service=mock_storage_service,
            data_dir=tmp_path
        )

        urls = ["https://example.com/page_1.jpg"]
        exam_spec = pipeline._extract_with_llm(urls, "Test", 1)

        # Should succeed after retry
        assert exam_spec.exam_name == "Test"
        assert len(exam_spec.questions) == 1

        # LLM should have been called twice
        assert mock_llm_client.complete_with_images.call_count == 2

    def test_extract_with_llm_fails_after_max_retries(
        self,
        mock_llm_client,
        mock_storage_service,
        tmp_path
    ):
        """Test that extraction fails after max retries."""
        # Always return invalid JSON
        mock_llm_client.complete_with_images = Mock(return_value="Invalid JSON!")

        pipeline = QuestionExtractionPipeline(
            llm_client=mock_llm_client,
            storage_service=mock_storage_service,
            data_dir=tmp_path,
            max_parse_retries=2
        )

        urls = ["https://example.com/page_1.jpg"]

        with pytest.raises(QuestionExtractionError, match="Failed to parse"):
            pipeline._extract_with_llm(urls, "Test", 1)

    def test_validate_exam_spec_success(
        self,
        mock_llm_client,
        mock_storage_service,
        tmp_path
    ):
        """Test validation of valid exam spec."""
        pipeline = QuestionExtractionPipeline(
            llm_client=mock_llm_client,
            storage_service=mock_storage_service,
            data_dir=tmp_path
        )

        exam_spec = ExamSpec(
            exam_name="Test",
            total_pages=2,
            questions=[
                Question("1", "Q1", 10.0, page_number=1),
                Question("2", "Q2", 15.0, page_number=2)
            ]
        )

        # Should not raise
        pipeline._validate_exam_spec(exam_spec)

    def test_validate_exam_spec_failure_raises_error(
        self,
        mock_llm_client,
        mock_storage_service,
        tmp_path
    ):
        """Test validation of invalid exam spec raises error."""
        pipeline = QuestionExtractionPipeline(
            llm_client=mock_llm_client,
            storage_service=mock_storage_service,
            data_dir=tmp_path
        )

        # Create invalid exam spec (no questions)
        exam_spec = ExamSpec(
            exam_name="Test",
            total_pages=1,
            questions=[]
        )

        with pytest.raises(QuestionExtractionError, match="validation failed"):
            pipeline._validate_exam_spec(exam_spec)

    def test_end_to_end_extraction(
        self,
        mock_llm_client,
        mock_storage_service,
        temp_student_folder
    ):
        """Test complete end-to-end extraction pipeline."""
        pipeline = QuestionExtractionPipeline(
            llm_client=mock_llm_client,
            storage_service=mock_storage_service,
            data_dir=temp_student_folder.parent
        )

        exam_spec = pipeline.extract_from_student(
            student_folder=temp_student_folder,
            assignment_name="exam1",
            exam_name="BMI541 Midterm"
        )

        # Verify results
        assert exam_spec.exam_name == "BMI541 Midterm"
        assert exam_spec.total_pages == 5
        assert len(exam_spec.questions) == 5
        assert exam_spec.total_points == 23.0  # Sum of all points

        # Verify services were called
        mock_storage_service.upload_images_batch.assert_called_once()
        mock_llm_client.complete_with_images.assert_called_once()

        # Verify exam spec was saved
        results_dir = temp_student_folder.parent / "results"
        exam_spec_file = results_dir / "exam_spec.json"
        assert exam_spec_file.exists()

        # Verify saved content
        with open(exam_spec_file, 'r') as f:
            saved_data = json.load(f)
        assert saved_data['exam_name'] == "BMI541 Midterm"
        assert len(saved_data['questions']) == 5


class TestConvenienceFunctions:
    """Test high-level convenience functions."""

    @patch('aita.pipelines.extract_questions.create_extraction_pipeline')
    def test_extract_questions_from_student(self, mock_create_pipeline, tmp_path):
        """Test the high-level extraction function."""
        # Create mock pipeline
        mock_pipeline = Mock()
        mock_exam_spec = ExamSpec(
            exam_name="Test",
            total_pages=1,
            questions=[Question("1", "Q1", 10.0)]
        )
        mock_pipeline.extract_from_student = Mock(return_value=mock_exam_spec)
        mock_create_pipeline.return_value = mock_pipeline

        # Call convenience function
        student_folder = str(tmp_path / "student")
        result = extract_questions_from_student(
            student_folder=student_folder,
            assignment_name="exam1",
            exam_name="Custom Exam"
        )

        # Verify
        assert result.exam_name == "Test"
        mock_create_pipeline.assert_called_once()
        mock_pipeline.extract_from_student.assert_called_once()


@pytest.mark.skipif(
    not Path("data/grouped/Student_001").exists(),
    reason="Real test data not available"
)
class TestRealDataIntegration:
    """Integration tests with real data (optional - requires data folder)."""

    def test_extract_from_real_student_folder_mocked_services(
        self,
        mock_llm_client,
        mock_storage_service
    ):
        """Test extraction with real student folder but mocked services."""
        data_dir = Path("C:/Users/ellio/OneDrive - UW-Madison/AITA/data")
        student_folder = data_dir / "grouped" / "Student_001"

        if not student_folder.exists():
            pytest.skip("Real student data not found")

        pipeline = QuestionExtractionPipeline(
            llm_client=mock_llm_client,
            storage_service=mock_storage_service,
            data_dir=data_dir
        )

        exam_spec = pipeline.extract_from_student(
            student_folder=student_folder,
            assignment_name="test_exam"
        )

        # Should successfully extract with real images
        assert exam_spec is not None
        assert len(exam_spec.questions) > 0
        assert exam_spec.total_pages == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
