import os
import json
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import logging
from urllib.parse import quote

from google.cloud import storage
from google.oauth2 import service_account
from google.api_core import exceptions as gcs_exceptions

logger = logging.getLogger(__name__)


class GoogleCloudStorageService:
    """
    Google Cloud Storage service for AITA.

    Handles uploading exam images to a public GCS bucket and generating
    public URLs that can be used with the LLM vision API.
    """

    def __init__(
        self,
        project_id: str,
        bucket_name: str,
        credentials_path: Optional[str] = None,
        credentials_json: Optional[Dict[str, Any]] = None,
        make_public: bool = True
    ):
        """
        Initialize the GCS service.

        Args:
            project_id: Google Cloud project ID
            bucket_name: Name of the GCS bucket
            credentials_path: Path to service account JSON file
            credentials_json: Service account credentials as dict
            make_public: Whether to make uploaded files publicly readable
                        (Note: With uniform bucket-level access, this is controlled
                         at the bucket level, not per-object)
        """
        self.project_id = project_id
        self.bucket_name = bucket_name
        self.make_public = make_public

        # Initialize credentials
        if credentials_path and Path(credentials_path).exists():
            self.credentials = service_account.Credentials.from_service_account_file(
                credentials_path
            )
            logger.info(f"Loaded GCS credentials from {credentials_path}")
        elif credentials_json:
            self.credentials = service_account.Credentials.from_service_account_info(
                credentials_json
            )
            logger.info("Loaded GCS credentials from provided JSON")
        else:
            # Try default credentials (environment variable GOOGLE_APPLICATION_CREDENTIALS)
            self.credentials = None
            logger.info("Using default GCS credentials from environment")

        # Initialize the client
        try:
            if self.credentials:
                self.client = storage.Client(
                    project=self.project_id,
                    credentials=self.credentials
                )
            else:
                self.client = storage.Client(project=self.project_id)

            # Get bucket reference
            self.bucket = self.client.bucket(self.bucket_name)

            # Test connection
            self._test_connection()
            logger.info(f"Successfully connected to GCS bucket: {self.bucket_name}")

        except Exception as e:
            logger.error(f"Failed to initialize GCS client: {e}")
            raise

    def _test_connection(self) -> None:
        """Test the GCS connection and bucket access."""
        try:
            # Try to get bucket metadata to test connection
            _ = self.bucket.exists()
            logger.debug("GCS connection test successful")
        except Exception as e:
            logger.error(f"GCS connection test failed: {e}")
            raise

    def upload_image(
        self,
        local_path: str,
        destination_path: Optional[str] = None,
        student_name: Optional[str] = None,
        assignment_name: Optional[str] = None,
        page_number: Optional[int] = None
    ) -> str:
        """
        Upload an image to GCS and return the public URL.

        Args:
            local_path: Local path to the image file
            destination_path: Custom destination path in bucket (optional)
            student_name: Student name for organized storage
            assignment_name: Assignment/exam name
            page_number: Page number of the exam

        Returns:
            Public URL to the uploaded image

        Raises:
            FileNotFoundError: If local file doesn't exist
            Exception: If upload fails
        """
        local_file = Path(local_path)
        if not local_file.exists():
            raise FileNotFoundError(f"Local file not found: {local_path}")

        # Generate destination path if not provided
        if destination_path is None:
            destination_path = self._generate_destination_path(
                local_file.name,
                student_name=student_name,
                assignment_name=assignment_name,
                page_number=page_number
            )

        try:
            # Create blob and upload
            blob = self.bucket.blob(destination_path)

            # Set content type based on file extension
            content_type = self._get_content_type(local_file.suffix.lower())
            blob.content_type = content_type

            # Upload the file
            with open(local_file, 'rb') as file_obj:
                blob.upload_from_file(file_obj, content_type=content_type)

            # Note: With uniform bucket-level access, public access is controlled
            # at the bucket level, not per-object. The bucket should be configured
            # to allow public read access for all objects.
            # Individual blob.make_public() calls are not needed and will fail.

            # Generate and return public URL
            public_url = self._generate_public_url(destination_path)

            logger.info(f"Successfully uploaded {local_path} to {destination_path}")
            logger.debug(f"Public URL: {public_url}")

            return public_url

        except Exception as e:
            logger.error(f"Failed to upload {local_path} to GCS: {e}")
            raise

    def upload_images_batch(
        self,
        image_paths: List[str],
        student_name: Optional[str] = None,
        assignment_name: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """
        Upload multiple images in batch.

        Args:
            image_paths: List of local image paths
            student_name: Student name for organized storage
            assignment_name: Assignment/exam name

        Returns:
            List of dicts with 'local_path', 'gcs_path', and 'public_url'
        """
        results = []

        for i, image_path in enumerate(image_paths):
            try:
                public_url = self.upload_image(
                    local_path=image_path,
                    student_name=student_name,
                    assignment_name=assignment_name,
                    page_number=i + 1
                )

                # Generate the GCS path for this upload
                local_file = Path(image_path)
                gcs_path = self._generate_destination_path(
                    local_file.name,
                    student_name=student_name,
                    assignment_name=assignment_name,
                    page_number=i + 1
                )

                results.append({
                    'local_path': image_path,
                    'gcs_path': gcs_path,
                    'public_url': public_url
                })

            except Exception as e:
                logger.error(f"Failed to upload {image_path}: {e}")
                # Continue with other uploads
                results.append({
                    'local_path': image_path,
                    'gcs_path': None,
                    'public_url': None,
                    'error': str(e)
                })

        return results

    def delete_image(self, gcs_path: str) -> bool:
        """
        Delete an image from GCS.

        Args:
            gcs_path: Path to the file in GCS bucket

        Returns:
            True if deletion was successful, False otherwise
        """
        try:
            blob = self.bucket.blob(gcs_path)
            blob.delete()
            logger.info(f"Successfully deleted {gcs_path} from GCS")
            return True

        except gcs_exceptions.NotFound:
            logger.warning(f"File not found in GCS: {gcs_path}")
            return False
        except Exception as e:
            logger.error(f"Failed to delete {gcs_path} from GCS: {e}")
            return False

    def list_images(
        self,
        prefix: Optional[str] = None,
        student_name: Optional[str] = None,
        assignment_name: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """
        List images in the bucket with optional filtering.

        Args:
            prefix: Prefix to filter by
            student_name: Filter by student name
            assignment_name: Filter by assignment name

        Returns:
            List of dicts with 'gcs_path' and 'public_url'
        """
        try:
            # Build prefix for filtering
            if prefix is None and (student_name or assignment_name):
                prefix = self._generate_prefix(
                    student_name=student_name,
                    assignment_name=assignment_name
                )

            # List blobs with prefix
            blobs = self.client.list_blobs(self.bucket_name, prefix=prefix)

            results = []
            for blob in blobs:
                # Skip directories (blobs ending with /)
                if blob.name.endswith('/'):
                    continue

                results.append({
                    'gcs_path': blob.name,
                    'public_url': self._generate_public_url(blob.name),
                    'size': blob.size,
                    'created': blob.time_created.isoformat() if blob.time_created else None
                })

            logger.info(f"Listed {len(results)} images with prefix: {prefix}")
            return results

        except Exception as e:
            logger.error(f"Failed to list images: {e}")
            raise

    def check_image_exists(self, gcs_path: str) -> bool:
        """
        Check if an image exists in GCS.

        Args:
            gcs_path: Path to the file in GCS bucket

        Returns:
            True if the file exists, False otherwise
        """
        try:
            blob = self.bucket.blob(gcs_path)
            return blob.exists()
        except Exception as e:
            logger.error(f"Error checking if {gcs_path} exists: {e}")
            return False

    def get_public_url(self, gcs_path: str) -> str:
        """
        Get the public URL for a file in GCS.

        Args:
            gcs_path: Path to the file in GCS bucket

        Returns:
            Public URL to the file
        """
        return self._generate_public_url(gcs_path)

    def _generate_destination_path(
        self,
        filename: str,
        student_name: Optional[str] = None,
        assignment_name: Optional[str] = None,
        page_number: Optional[int] = None
    ) -> str:
        """Generate organized destination path in GCS."""

        # Start with base structure
        parts = []

        # Add assignment/course folder
        if assignment_name:
            parts.append(self._sanitize_filename(assignment_name))
        else:
            parts.append("exams")

        # Add date folder (YYYY-MM-DD)
        today = datetime.now().strftime("%Y-%m-%d")
        parts.append(today)

        # Add student folder
        if student_name:
            parts.append(self._sanitize_filename(student_name))
        else:
            parts.append("unknown_student")

        # Generate filename
        if page_number is not None:
            # Replace original filename with page number
            file_ext = Path(filename).suffix
            final_filename = f"page_{page_number}{file_ext}"
        else:
            final_filename = self._sanitize_filename(filename)

        parts.append(final_filename)

        return "/".join(parts)

    def _generate_prefix(
        self,
        student_name: Optional[str] = None,
        assignment_name: Optional[str] = None
    ) -> str:
        """Generate prefix for filtering."""
        parts = []

        if assignment_name:
            parts.append(self._sanitize_filename(assignment_name))

        if student_name:
            # Add today's date and student name
            today = datetime.now().strftime("%Y-%m-%d")
            parts.extend([today, self._sanitize_filename(student_name)])

        return "/".join(parts) + "/" if parts else ""

    def _generate_public_url(self, gcs_path: str) -> str:
        """Generate public URL for a GCS object."""
        # URL encode the path to handle special characters
        encoded_path = quote(gcs_path, safe='/')
        return f"https://storage.googleapis.com/{self.bucket_name}/{encoded_path}"

    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for GCS storage."""
        # Remove or replace problematic characters
        sanitized = filename.replace(' ', '_')
        sanitized = ''.join(c for c in sanitized if c.isalnum() or c in '-_.')
        return sanitized.lower()

    def _get_content_type(self, file_extension: str) -> str:
        """Get appropriate content type for file extension."""
        content_types = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.bmp': 'image/bmp',
            '.tiff': 'image/tiff',
            '.webp': 'image/webp'
        }
        return content_types.get(file_extension, 'application/octet-stream')


def create_storage_service(
    project_id: Optional[str] = None,
    bucket_name: Optional[str] = None,
    credentials_path: Optional[str] = None
) -> GoogleCloudStorageService:
    """
    Factory function to create a GCS service using environment variables.

    Args:
        project_id: Override project ID from config
        bucket_name: Override bucket name from config
        credentials_path: Override credentials path from config

    Returns:
        Configured GoogleCloudStorageService instance
    """
    from aita.config import get_config

    config = get_config()
    gcs_config = config.google_cloud

    return GoogleCloudStorageService(
        project_id=project_id or gcs_config.project_id,
        bucket_name=bucket_name or gcs_config.bucket_name,
        credentials_path=credentials_path or gcs_config.credentials_path
    )


# Convenience functions for common operations
def upload_exam_images(
    image_paths: List[str],
    student_name: str,
    assignment_name: str,
    storage_service: Optional[GoogleCloudStorageService] = None
) -> List[str]:
    """
    Upload exam images for a student and return public URLs.

    Args:
        image_paths: List of local image paths
        student_name: Name of the student
        assignment_name: Name of the assignment/exam
        storage_service: Optional storage service instance

    Returns:
        List of public URLs for the uploaded images
    """
    if storage_service is None:
        storage_service = create_storage_service()

    results = storage_service.upload_images_batch(
        image_paths=image_paths,
        student_name=student_name,
        assignment_name=assignment_name
    )

    # Extract just the URLs
    urls = []
    for result in results:
        if result.get('public_url'):
            urls.append(result['public_url'])
        else:
            logger.error(f"Failed to upload {result['local_path']}: {result.get('error')}")

    return urls


def cleanup_old_images(
    days_old: int = 30,
    storage_service: Optional[GoogleCloudStorageService] = None
) -> int:
    """
    Clean up images older than specified days.

    Args:
        days_old: Delete images older than this many days
        storage_service: Optional storage service instance

    Returns:
        Number of images deleted
    """
    if storage_service is None:
        storage_service = create_storage_service()

    cutoff_date = datetime.now() - timedelta(days=days_old)
    deleted_count = 0

    try:
        images = storage_service.list_images()

        for image in images:
            if image.get('created'):
                created_date = datetime.fromisoformat(image['created'].replace('Z', '+00:00'))
                if created_date < cutoff_date:
                    if storage_service.delete_image(image['gcs_path']):
                        deleted_count += 1

        logger.info(f"Cleaned up {deleted_count} images older than {days_old} days")
        return deleted_count

    except Exception as e:
        logger.error(f"Failed to cleanup old images: {e}")
        return 0