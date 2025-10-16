import pytest
import os
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from aita.services.storage import (
    GoogleCloudStorageService,
    create_storage_service,
    upload_exam_images,
    cleanup_old_images
)


class TestGoogleCloudStorageService:
    """Test suite for GoogleCloudStorageService"""

    @pytest.fixture
    def mock_credentials_json(self):
        """Mock service account credentials JSON"""
        return {
            "type": "service_account",
            "project_id": "test-project",
            "private_key_id": "test-key-id",
            "private_key": "-----BEGIN PRIVATE KEY-----\ntest-key\n-----END PRIVATE KEY-----\n",
            "client_email": "test@test-project.iam.gserviceaccount.com",
            "client_id": "test-client-id",
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token"
        }

    @pytest.fixture
    def mock_storage_client(self):
        """Mock Google Cloud Storage client"""
        with patch('aita.services.storage.storage.Client') as mock_client:
            # Mock client instance
            client_instance = Mock()
            mock_client.return_value = client_instance

            # Mock bucket
            mock_bucket = Mock()
            client_instance.bucket.return_value = mock_bucket
            mock_bucket.exists.return_value = True

            # Mock blob
            mock_blob = Mock()
            mock_bucket.blob.return_value = mock_blob
            mock_blob.make_public.return_value = None

            yield {
                'client_class': mock_client,
                'client_instance': client_instance,
                'bucket': mock_bucket,
                'blob': mock_blob
            }

    @pytest.fixture
    def mock_credentials(self):
        """Mock service account credentials"""
        with patch('aita.services.storage.service_account.Credentials') as mock_creds:
            mock_creds.from_service_account_file.return_value = Mock()
            mock_creds.from_service_account_info.return_value = Mock()
            yield mock_creds

    def test_init_with_credentials_path(self, mock_storage_client, mock_credentials):
        """Test initialization with credentials file path"""
        with patch('pathlib.Path.exists', return_value=True):
            service = GoogleCloudStorageService(
                project_id="test-project",
                bucket_name="test-bucket",
                credentials_path="/path/to/creds.json"
            )

            assert service.project_id == "test-project"
            assert service.bucket_name == "test-bucket"
            assert service.make_public is True

            # Verify credentials were loaded from file
            mock_credentials.from_service_account_file.assert_called_once_with(
                "/path/to/creds.json"
            )

    def test_init_with_credentials_json(self, mock_storage_client, mock_credentials, mock_credentials_json):
        """Test initialization with credentials JSON"""
        service = GoogleCloudStorageService(
            project_id="test-project",
            bucket_name="test-bucket",
            credentials_json=mock_credentials_json
        )

        assert service.project_id == "test-project"
        assert service.bucket_name == "test-bucket"

        # Verify credentials were loaded from JSON
        mock_credentials.from_service_account_info.assert_called_once_with(
            mock_credentials_json
        )

    def test_init_with_default_credentials(self, mock_storage_client):
        """Test initialization with default credentials"""
        service = GoogleCloudStorageService(
            project_id="test-project",
            bucket_name="test-bucket"
        )

        assert service.project_id == "test-project"
        assert service.bucket_name == "test-bucket"
        assert service.credentials is None

    def test_generate_destination_path(self, mock_storage_client, mock_credentials):
        """Test destination path generation"""
        service = GoogleCloudStorageService(
            project_id="test-project",
            bucket_name="test-bucket"
        )

        # Test with all parameters
        path = service._generate_destination_path(
            filename="exam.jpg",
            student_name="John Doe",
            assignment_name="Midterm Exam",
            page_number=1
        )

        # Should generate: midterm_exam/YYYY-MM-DD/john_doe/page_1.jpg
        parts = path.split('/')
        assert parts[0] == "midterm_exam"
        assert len(parts[1]) == 10  # Date format YYYY-MM-DD
        assert parts[2] == "john_doe"
        assert parts[3] == "page_1.jpg"

    def test_generate_destination_path_minimal(self, mock_storage_client, mock_credentials):
        """Test destination path generation with minimal parameters"""
        service = GoogleCloudStorageService(
            project_id="test-project",
            bucket_name="test-bucket"
        )

        path = service._generate_destination_path("test.jpg")

        parts = path.split('/')
        assert parts[0] == "exams"
        assert len(parts[1]) == 10  # Date
        assert parts[2] == "unknown_student"
        assert parts[3] == "test.jpg"

    def test_upload_image_success(self, mock_storage_client, mock_credentials):
        """Test successful image upload"""
        service = GoogleCloudStorageService(
            project_id="test-project",
            bucket_name="test-bucket"
        )

        # Mock file operations
        test_file = Path("test_image.jpg")
        with patch.object(Path, 'exists', return_value=True):
            with patch('builtins.open', mock_open_multiple_files({'test_image.jpg': b'fake_image_data'})):

                url = service.upload_image(
                    local_path="test_image.jpg",
                    student_name="test_student",
                    assignment_name="test_exam"
                )

                # Verify upload was called
                mock_storage_client['blob'].upload_from_file.assert_called_once()
                mock_storage_client['blob'].make_public.assert_called_once()

                # Verify URL format
                assert url.startswith("https://storage.googleapis.com/test-bucket/")
                assert "test_exam" in url

    def test_upload_image_file_not_found(self, mock_storage_client, mock_credentials):
        """Test upload with non-existent file"""
        service = GoogleCloudStorageService(
            project_id="test-project",
            bucket_name="test-bucket"
        )

        with patch.object(Path, 'exists', return_value=False):
            with pytest.raises(FileNotFoundError):
                service.upload_image("nonexistent.jpg")

    def test_upload_images_batch(self, mock_storage_client, mock_credentials):
        """Test batch image upload"""
        service = GoogleCloudStorageService(
            project_id="test-project",
            bucket_name="test-bucket"
        )

        image_paths = ["image1.jpg", "image2.jpg", "image3.jpg"]

        with patch.object(Path, 'exists', return_value=True):
            with patch('builtins.open', mock_open_multiple_files({
                'image1.jpg': b'data1',
                'image2.jpg': b'data2',
                'image3.jpg': b'data3'
            })):
                results = service.upload_images_batch(
                    image_paths=image_paths,
                    student_name="test_student",
                    assignment_name="test_exam"
                )

                assert len(results) == 3
                for result in results:
                    assert 'local_path' in result
                    assert 'gcs_path' in result
                    assert 'public_url' in result

    def test_delete_image_success(self, mock_storage_client, mock_credentials):
        """Test successful image deletion"""
        service = GoogleCloudStorageService(
            project_id="test-project",
            bucket_name="test-bucket"
        )

        result = service.delete_image("test/path/image.jpg")
        assert result is True
        mock_storage_client['blob'].delete.assert_called_once()

    def test_delete_image_not_found(self, mock_storage_client, mock_credentials):
        """Test deletion of non-existent image"""
        service = GoogleCloudStorageService(
            project_id="test-project",
            bucket_name="test-bucket"
        )

        # Mock NotFound exception
        from google.api_core import exceptions as gcs_exceptions
        mock_storage_client['blob'].delete.side_effect = gcs_exceptions.NotFound("File not found")

        result = service.delete_image("nonexistent.jpg")
        assert result is False

    def test_list_images(self, mock_storage_client, mock_credentials):
        """Test listing images in bucket"""
        service = GoogleCloudStorageService(
            project_id="test-project",
            bucket_name="test-bucket"
        )

        # Mock blob objects
        mock_blobs = []
        for i in range(3):
            blob = Mock()
            blob.name = f"test/image{i}.jpg"
            blob.size = 1024 * (i + 1)
            blob.time_created = datetime.now()
            mock_blobs.append(blob)

        mock_storage_client['client_instance'].list_blobs.return_value = mock_blobs

        results = service.list_images(prefix="test/")

        assert len(results) == 3
        for i, result in enumerate(results):
            assert result['gcs_path'] == f"test/image{i}.jpg"
            assert 'public_url' in result
            assert 'size' in result
            assert 'created' in result

    def test_check_image_exists(self, mock_storage_client, mock_credentials):
        """Test checking if image exists"""
        service = GoogleCloudStorageService(
            project_id="test-project",
            bucket_name="test-bucket"
        )

        # Test existing file
        mock_storage_client['blob'].exists.return_value = True
        assert service.check_image_exists("existing.jpg") is True

        # Test non-existing file
        mock_storage_client['blob'].exists.return_value = False
        assert service.check_image_exists("nonexistent.jpg") is False

    def test_get_public_url(self, mock_storage_client, mock_credentials):
        """Test public URL generation"""
        service = GoogleCloudStorageService(
            project_id="test-project",
            bucket_name="test-bucket"
        )

        url = service.get_public_url("test/image.jpg")
        expected = "https://storage.googleapis.com/test-bucket/test/image.jpg"
        assert url == expected

    def test_get_public_url_with_special_chars(self, mock_storage_client, mock_credentials):
        """Test public URL generation with special characters"""
        service = GoogleCloudStorageService(
            project_id="test-project",
            bucket_name="test-bucket"
        )

        url = service.get_public_url("test folder/image with spaces.jpg")
        assert "test%20folder/image%20with%20spaces.jpg" in url

    def test_sanitize_filename(self, mock_storage_client, mock_credentials):
        """Test filename sanitization"""
        service = GoogleCloudStorageService(
            project_id="test-project",
            bucket_name="test-bucket"
        )

        # Test various problematic characters
        assert service._sanitize_filename("John Doe") == "john_doe"
        assert service._sanitize_filename("Test@File#123!") == "testfile123"
        assert service._sanitize_filename("file-name_test.jpg") == "file-name_test.jpg"

    def test_get_content_type(self, mock_storage_client, mock_credentials):
        """Test content type detection"""
        service = GoogleCloudStorageService(
            project_id="test-project",
            bucket_name="test-bucket"
        )

        assert service._get_content_type(".jpg") == "image/jpeg"
        assert service._get_content_type(".png") == "image/png"
        assert service._get_content_type(".gif") == "image/gif"
        assert service._get_content_type(".unknown") == "application/octet-stream"


class TestStorageFactoryFunctions:
    """Test factory functions and utilities"""

    @patch('aita.services.storage.get_config')
    def test_create_storage_service(self, mock_get_config):
        """Test storage service factory function"""
        # Mock config
        mock_config = Mock()
        mock_config.google_cloud = Mock()
        mock_config.google_cloud.project_id = "test-project"
        mock_config.google_cloud.bucket_name = "test-bucket"
        mock_config.google_cloud.credentials_path = "/path/to/creds.json"
        mock_get_config.return_value = mock_config

        with patch('aita.services.storage.GoogleCloudStorageService') as mock_service:
            create_storage_service()

            mock_service.assert_called_once_with(
                project_id="test-project",
                bucket_name="test-bucket",
                credentials_path="/path/to/creds.json"
            )

    def test_upload_exam_images(self):
        """Test convenience function for uploading exam images"""
        mock_service = Mock()
        mock_service.upload_images_batch.return_value = [
            {'local_path': 'img1.jpg', 'public_url': 'http://example.com/img1.jpg'},
            {'local_path': 'img2.jpg', 'public_url': 'http://example.com/img2.jpg'},
            {'local_path': 'img3.jpg', 'error': 'Upload failed'}
        ]

        urls = upload_exam_images(
            image_paths=['img1.jpg', 'img2.jpg', 'img3.jpg'],
            student_name="John Doe",
            assignment_name="Midterm",
            storage_service=mock_service
        )

        assert len(urls) == 2  # Only successful uploads
        assert urls == ['http://example.com/img1.jpg', 'http://example.com/img2.jpg']

    def test_cleanup_old_images(self):
        """Test cleanup function for old images"""
        mock_service = Mock()

        # Mock images with different ages
        old_date = (datetime.now() - timedelta(days=40)).isoformat()
        recent_date = (datetime.now() - timedelta(days=10)).isoformat()

        mock_service.list_images.return_value = [
            {'gcs_path': 'old1.jpg', 'created': old_date},
            {'gcs_path': 'old2.jpg', 'created': old_date},
            {'gcs_path': 'recent.jpg', 'created': recent_date}
        ]

        mock_service.delete_image.return_value = True

        deleted_count = cleanup_old_images(days_old=30, storage_service=mock_service)

        assert deleted_count == 2
        # Verify only old images were deleted
        assert mock_service.delete_image.call_count == 2


class TestIntegrationWithAITAWorkflow:
    """Test integration with AITA workflow components"""

    def test_integration_with_config(self):
        """Test that storage service integrates with AITA config"""
        from aita.config import GoogleCloudConfig

        # Test config creation
        config = GoogleCloudConfig(
            project_id="vital-orb-340815",
            bucket_name="exam1_uwmadison",
            credentials_path="C:\\Users\\ellio\\OneDrive - UW-Madison\\AITA\\vital-orb-340815-7a97e5eea09e.json"
        )

        assert config.project_id == "vital-orb-340815"
        assert config.bucket_name == "exam1_uwmadison"

    @patch('aita.services.storage.GoogleCloudStorageService')
    def test_storage_with_llm_workflow(self, mock_storage_service):
        """Test storage service in LLM workflow context"""
        # Mock storage service
        mock_service_instance = Mock()
        mock_storage_service.return_value = mock_service_instance
        mock_service_instance.upload_image.return_value = "https://storage.googleapis.com/bucket/image.jpg"

        # Simulate LLM workflow
        from aita.services.storage import GoogleCloudStorageService

        storage = GoogleCloudStorageService(
            project_id="test-project",
            bucket_name="test-bucket"
        )

        # This would be used in the actual LLM workflow
        # url = storage.upload_image("local_exam.jpg", student_name="John", assignment_name="Midterm")
        # then pass url to LLM client for processing


def mock_open_multiple_files(files_dict):
    """Helper to mock multiple file opens"""
    def mock_open_func(*args, **kwargs):
        filename = args[0]
        if isinstance(filename, Path):
            filename = str(filename)

        # Extract just the filename from path
        filename = Path(filename).name

        if filename in files_dict:
            mock_file = Mock()
            mock_file.read.return_value = files_dict[filename]
            mock_file.__enter__.return_value = mock_file
            mock_file.__exit__.return_value = None
            return mock_file
        else:
            raise FileNotFoundError(f"No such file: {filename}")

    return mock_open_func


if __name__ == "__main__":
    pytest.main([__file__, "-v"])