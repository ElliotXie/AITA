"""Tests for LLM cost tracking functionality."""

import unittest
from pathlib import Path
import tempfile
import json
import shutil
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

import sys
sys.path.append(str(Path(__file__).parent.parent))

from aita.services.llm.cost_tracker import (
    CostTracker, CostEntry, SessionSummary,
    get_global_tracker, set_global_tracker, init_global_tracker
)
from aita.services.llm.model_pricing import (
    ModelPricing, get_model_pricing, calculate_cost, MODEL_PRICING
)
from aita.services.llm.cost_analysis import CostAnalyzer, CostAnalysis
from aita.services.llm.base import BaseLLMClient, LLMMessage, LLMResponse


class TestModelPricing(unittest.TestCase):
    """Test model pricing data and calculations."""

    def test_get_model_pricing(self):
        """Test retrieving model pricing."""
        # Test known model
        pricing = get_model_pricing("google/gemini-2.5-flash")
        self.assertIsNotNone(pricing)
        self.assertEqual(pricing.input_per_million, 0.30)
        self.assertEqual(pricing.output_per_million, 2.50)
        self.assertEqual(pricing.image_per_thousand, 1.238)

        # Test unknown model
        pricing = get_model_pricing("unknown/model")
        self.assertIsNone(pricing)

    def test_calculate_cost(self):
        """Test cost calculation."""
        # Test with Gemini model
        result = calculate_cost(
            "google/gemini-2.5-flash",
            input_tokens=1000,
            output_tokens=500,
            image_count=2
        )

        self.assertEqual(result["model"], "google/gemini-2.5-flash")
        self.assertEqual(result["input_tokens"], 1000)
        self.assertEqual(result["output_tokens"], 500)
        self.assertEqual(result["image_count"], 2)

        # Calculate expected costs
        expected_input_cost = (1000 / 1_000_000) * 0.30
        expected_output_cost = (500 / 1_000_000) * 2.50
        expected_image_cost = (2 / 1_000) * 1.238

        self.assertAlmostEqual(result["input_cost"], expected_input_cost, places=6)
        self.assertAlmostEqual(result["output_cost"], expected_output_cost, places=6)
        self.assertAlmostEqual(result["image_cost"], expected_image_cost, places=6)
        self.assertAlmostEqual(
            result["total_cost"],
            expected_input_cost + expected_output_cost + expected_image_cost,
            places=6
        )

    def test_calculate_cost_unknown_model(self):
        """Test cost calculation with unknown model."""
        result = calculate_cost(
            "unknown/model",
            input_tokens=1000,
            output_tokens=500
        )

        self.assertIn("error", result)
        self.assertEqual(result["total_cost"], 0.0)

    def test_all_pricing_data_valid(self):
        """Test that all pricing data is valid."""
        for model_name, pricing in MODEL_PRICING.items():
            self.assertIsInstance(pricing, ModelPricing)
            self.assertGreaterEqual(pricing.input_per_million, 0)
            self.assertGreaterEqual(pricing.output_per_million, 0)
            if pricing.image_per_thousand is not None:
                self.assertGreaterEqual(pricing.image_per_thousand, 0)
            self.assertEqual(pricing.currency, "USD")


class TestCostTracker(unittest.TestCase):
    """Test cost tracking functionality."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.tracker = CostTracker(data_dir=Path(self.temp_dir), auto_save=False)

    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_initialization(self):
        """Test tracker initialization."""
        self.assertIsNotNone(self.tracker.session_id)
        self.assertIsNotNone(self.tracker.start_time)
        self.assertEqual(len(self.tracker.entries), 0)

    def test_track_call(self):
        """Test tracking a single LLM call."""
        entry = self.tracker.track_call(
            model="google/gemini-2.5-flash",
            operation_type="transcription",
            input_tokens=1000,
            output_tokens=500,
            image_count=1,
            metadata={"page": 1}
        )

        self.assertEqual(len(self.tracker.entries), 1)
        self.assertEqual(entry.model, "google/gemini-2.5-flash")
        self.assertEqual(entry.operation_type, "transcription")
        self.assertEqual(entry.input_tokens, 1000)
        self.assertEqual(entry.output_tokens, 500)
        self.assertEqual(entry.image_count, 1)
        self.assertGreater(entry.total_cost, 0)

    def test_track_call_with_usage_data(self):
        """Test tracking with usage data from API response."""
        usage_data = {
            "prompt_tokens": 2000,
            "completion_tokens": 1000
        }

        entry = self.tracker.track_call(
            model="google/gemini-2.5-flash",
            operation_type="question_extraction",
            usage_data=usage_data
        )

        self.assertEqual(entry.input_tokens, 2000)
        self.assertEqual(entry.output_tokens, 1000)

    def test_get_summary(self):
        """Test getting session summary."""
        # Track multiple calls
        self.tracker.track_call(
            model="google/gemini-2.5-flash",
            operation_type="transcription",
            input_tokens=1000,
            output_tokens=500
        )
        self.tracker.track_call(
            model="google/gemini-2.5-flash",
            operation_type="name_extraction",
            input_tokens=500,
            output_tokens=100,
            image_count=1
        )

        summary = self.tracker.get_summary()

        self.assertEqual(summary.total_calls, 2)
        self.assertEqual(summary.total_input_tokens, 1500)
        self.assertEqual(summary.total_output_tokens, 600)
        self.assertEqual(summary.total_images, 1)
        self.assertGreater(summary.total_cost, 0)
        self.assertIn("transcription", summary.cost_by_operation)
        self.assertIn("name_extraction", summary.cost_by_operation)

    def test_save_and_load(self):
        """Test saving and loading session data."""
        # Track some calls
        self.tracker.track_call(
            model="google/gemini-2.5-flash",
            operation_type="transcription",
            input_tokens=1000,
            output_tokens=500
        )

        # Save
        self.tracker.save()
        self.assertTrue(self.tracker.session_file.exists())

        # Create new tracker and load
        new_tracker = CostTracker(data_dir=Path(self.temp_dir), auto_save=False)
        new_tracker.load(self.tracker.session_file)

        self.assertEqual(len(new_tracker.entries), 1)
        entry = new_tracker.entries[0]
        self.assertEqual(entry.model, "google/gemini-2.5-flash")
        self.assertEqual(entry.operation_type, "transcription")

    def test_infer_operation_type(self):
        """Test operation type inference."""
        # Test with various context clues
        result = CostTracker.infer_operation_type({
            "prompt": "Extract the student name from this image"
        })
        self.assertEqual(result, "name_extraction")

        result = CostTracker.infer_operation_type({
            "prompt": "Transcribe the handwriting in this exam"
        })
        self.assertEqual(result, "transcription")

        result = CostTracker.infer_operation_type({
            "prompt": "Grade this answer using the rubric"
        })
        self.assertEqual(result, "grading")

    def test_global_tracker(self):
        """Test global tracker functionality."""
        # Initially no global tracker
        self.assertIsNone(get_global_tracker())

        # Initialize global tracker
        tracker = init_global_tracker(data_dir=Path(self.temp_dir))
        self.assertIsNotNone(tracker)
        self.assertEqual(get_global_tracker(), tracker)

        # Set different tracker
        new_tracker = CostTracker(data_dir=Path(self.temp_dir))
        set_global_tracker(new_tracker)
        self.assertEqual(get_global_tracker(), new_tracker)


class TestCostIntegration(unittest.TestCase):
    """Test cost tracking integration with LLM clients."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        set_global_tracker(None)  # Clear global tracker

    @patch('aita.services.llm.base.BaseLLMClient._make_request')
    def test_base_client_cost_tracking(self, mock_make_request):
        """Test cost tracking in BaseLLMClient."""
        # Set up mock response
        mock_response = LLMResponse(
            content="Test response",
            usage={"prompt_tokens": 100, "completion_tokens": 50},
            model="test/model",
            finish_reason="stop"
        )
        mock_make_request.return_value = mock_response

        # Initialize global tracker
        tracker = init_global_tracker(data_dir=Path(self.temp_dir))

        # Create a mock client (need to create concrete class)
        class TestClient(BaseLLMClient):
            def _make_request(self, messages, **kwargs):
                return mock_make_request(messages, **kwargs)

        client = TestClient(model="test/model")

        # Make a request
        messages = [LLMMessage(role="user", content="Test message")]
        response = client.complete(messages)

        # Verify response
        self.assertEqual(response.content, "Test response")

        # Verify cost was tracked
        self.assertEqual(len(tracker.entries), 1)
        entry = tracker.entries[0]
        self.assertEqual(entry.model, "test/model")
        self.assertEqual(entry.input_tokens, 100)
        self.assertEqual(entry.output_tokens, 50)

    @patch('aita.services.llm.base.BaseLLMClient._make_request')
    def test_image_counting(self, mock_make_request):
        """Test image counting in messages."""
        mock_response = LLMResponse(
            content="Test response",
            usage={"prompt_tokens": 100, "completion_tokens": 50}
        )
        mock_make_request.return_value = mock_response

        class TestClient(BaseLLMClient):
            def _make_request(self, messages, **kwargs):
                return mock_make_request(messages, **kwargs)

        client = TestClient(model="test/model")

        # Test counting images
        messages = [
            LLMMessage(role="user", content=[
                {"type": "text", "text": "Test"},
                {"type": "image_url", "image_url": {"url": "http://example.com/image1.jpg"}},
                {"type": "image_url", "image_url": {"url": "http://example.com/image2.jpg"}}
            ])
        ]

        image_count = client._count_images_in_messages(messages)
        self.assertEqual(image_count, 2)


class TestCostAnalyzer(unittest.TestCase):
    """Test cost analysis functionality."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.analyzer = CostAnalyzer(data_dir=Path(self.temp_dir))

    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_analyze_empty_directory(self):
        """Test analyzing empty directory."""
        analysis = self.analyzer.analyze_all_sessions()

        self.assertEqual(analysis.total_sessions, 0)
        self.assertEqual(analysis.total_calls, 0)
        self.assertEqual(analysis.total_cost, 0.0)

    def test_analyze_multiple_sessions(self):
        """Test analyzing multiple sessions."""
        # Create test sessions
        for i in range(3):
            tracker = CostTracker(
                data_dir=Path(self.temp_dir),
                session_id=f"test_session_{i}",
                auto_save=False
            )

            tracker.track_call(
                model="google/gemini-2.5-flash",
                operation_type="transcription",
                input_tokens=1000 * (i + 1),
                output_tokens=500 * (i + 1)
            )

            tracker.save()

        # Analyze
        analysis = self.analyzer.analyze_all_sessions()

        self.assertEqual(analysis.total_sessions, 3)
        self.assertEqual(analysis.total_calls, 3)
        self.assertGreater(analysis.total_cost, 0)
        self.assertIn("transcription", analysis.cost_by_operation)

    def test_cost_projections(self):
        """Test cost projection calculations."""
        # Create sample data
        tracker = CostTracker(
            data_dir=Path(self.temp_dir),
            auto_save=False
        )

        # Simulate processing one student
        tracker.track_call(
            model="google/gemini-2.5-flash",
            operation_type="name_extraction",
            input_tokens=500,
            output_tokens=100,
            image_count=1
        )

        for page in range(5):
            tracker.track_call(
                model="google/gemini-2.5-flash",
                operation_type="transcription",
                input_tokens=2000,
                output_tokens=1000,
                image_count=1
            )

        tracker.save()

        # Get projections
        projections = self.analyzer.get_cost_projections(
            students_per_exam=30,
            pages_per_student=5
        )

        self.assertEqual(projections["students"], 30)
        self.assertEqual(projections["pages_per_student"], 5)
        self.assertEqual(projections["total_pages"], 150)
        self.assertIn("total_projected_cost", projections)
        self.assertIn("cost_per_student", projections)

    def test_export_to_csv(self):
        """Test CSV export functionality."""
        # Create test session
        tracker = CostTracker(
            data_dir=Path(self.temp_dir),
            auto_save=False
        )

        tracker.track_call(
            model="google/gemini-2.5-flash",
            operation_type="transcription",
            input_tokens=1000,
            output_tokens=500
        )

        tracker.save()

        # Export to CSV
        csv_file = Path(self.temp_dir) / "costs.csv"
        self.analyzer.export_to_csv(csv_file)

        self.assertTrue(csv_file.exists())

        # Verify CSV content
        import csv
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["model"], "google/gemini-2.5-flash")
        self.assertEqual(rows[0]["operation_type"], "transcription")


if __name__ == "__main__":
    unittest.main()