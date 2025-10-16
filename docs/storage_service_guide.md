# Google Cloud Storage Service Guide

This guide covers the Google Cloud Storage (GCS) service implementation for AITA, which handles uploading exam images to a public bucket and generating URLs for LLM processing.

## Overview

The GCS service is essential for AITA because:
1. **LLM Vision APIs require public URLs** - Images must be accessible via HTTP/HTTPS
2. **Organized Storage** - Images are automatically organized by assignment, date, and student
3. **Scalable** - Can handle large numbers of exam images efficiently
4. **Cost-effective** - Only pay for storage and bandwidth used

## Configuration

### 1. Environment Variables

Update your `.env` file with your GCS configuration:

```bash
# Google Cloud Storage Configuration
GOOGLE_CLOUD_PROJECT_ID=vital-orb-340815
GOOGLE_CLOUD_STORAGE_BUCKET=exam1_uwmadison
GOOGLE_APPLICATION_CREDENTIALS=C:\Users\ellio\OneDrive - UW-Madison\AITA\vital-orb-340815-7a97e5eea09e.json

# Intermediate results directory
AITA_INTERMEDIATE_DIR=C:\Users\ellio\OneDrive - UW-Madison\AITA\intermediateproduct
```

### 2. Service Account Setup

Your service account JSON file should have the following permissions:
- `Storage Object Admin` or `Storage Object Creator` for the bucket
- The bucket should be configured for public read access

### 3. Bucket Configuration

The bucket `exam1_uwmadison` should be configured as:
- **Public access**: Allowed for individual objects
- **Location**: Choose based on your region for performance
- **Storage class**: Standard (for frequent access during grading)

## Basic Usage

### Creating a Storage Service

```python
from aita.services.storage import GoogleCloudStorageService, create_storage_service

# Method 1: Factory function (uses config)
storage = create_storage_service()

# Method 2: Direct instantiation
storage = GoogleCloudStorageService(
    project_id="vital-orb-340815",
    bucket_name="exam1_uwmadison",
    credentials_path="path/to/credentials.json"
)
```

### Single Image Upload

```python
# Upload single exam page
public_url = storage.upload_image(
    local_path="data/raw_images/exam_page1.jpg",
    student_name="john_doe",
    assignment_name="midterm_exam",
    page_number=1
)

print(f"Image uploaded: {public_url}")
# Output: https://storage.googleapis.com/exam1_uwmadison/midterm_exam/2025-01-15/john_doe/page_1.jpg
```

### Batch Upload for Student Exams

```python
from aita.services.storage import upload_exam_images

# Upload all pages for a student
image_paths = [
    "data/grouped/john_doe/page1.jpg",
    "data/grouped/john_doe/page2.jpg",
    "data/grouped/john_doe/page3.jpg",
    "data/grouped/john_doe/page4.jpg",
    "data/grouped/john_doe/page5.jpg"
]

urls = upload_exam_images(
    image_paths=image_paths,
    student_name="john_doe",
    assignment_name="midterm_exam"
)

print(f"Uploaded {len(urls)} images for John Doe")
# These URLs can now be sent to the LLM API
```

## Storage Organization

The service automatically organizes files using this hierarchy:

```
bucket/
├── assignment_name/           # e.g., "midterm_exam"
│   └── YYYY-MM-DD/           # Upload date
│       └── student_name/      # e.g., "john_doe"
│           ├── page_1.jpg
│           ├── page_2.jpg
│           └── page_N.jpg
└── exams/                    # Default if no assignment name
    └── YYYY-MM-DD/
        └── unknown_student/   # Default if no student name
```

### Example Organization

```
exam1_uwmadison/
├── midterm_exam/
│   ├── 2025-01-15/
│   │   ├── alice_smith/
│   │   │   ├── page_1.jpg
│   │   │   ├── page_2.jpg
│   │   │   └── page_3.jpg
│   │   ├── bob_jones/
│   │   │   ├── page_1.jpg
│   │   │   └── page_2.jpg
│   │   └── charlie_brown/
│   │       ├── page_1.jpg
│   │       ├── page_2.jpg
│   │       └── page_3.jpg
│   └── 2025-01-20/
│       └── makeup_exams/
└── final_exam/
    └── 2025-03-15/
        └── ...
```

## Advanced Usage

### Custom Upload Paths

```python
# Upload with custom destination path
url = storage.upload_image(
    local_path="special_case.jpg",
    destination_path="special/custom/path/image.jpg"
)
```

### Listing and Management

```python
# List all images for an assignment
images = storage.list_images(
    assignment_name="midterm_exam",
    student_name="john_doe"
)

for image in images:
    print(f"Path: {image['gcs_path']}")
    print(f"URL: {image['public_url']}")
    print(f"Size: {image['size']} bytes")
    print(f"Created: {image['created']}")

# Check if specific image exists
exists = storage.check_image_exists("midterm_exam/2025-01-15/john_doe/page_1.jpg")

# Delete an image
deleted = storage.delete_image("path/to/image.jpg")
```

### Cleanup Operations

```python
from aita.services.storage import cleanup_old_images

# Clean up images older than 30 days
deleted_count = cleanup_old_images(days_old=30)
print(f"Cleaned up {deleted_count} old images")
```

## Integration with AITA Workflow

### Step 1: Upload After Grouping

```python
# After images are grouped by student (from ingest pipeline)
from pathlib import Path

def upload_student_exams(student_folder: Path, assignment_name: str):
    """Upload all exam images for a student"""

    image_files = list(student_folder.glob("*.jpg"))
    image_files.sort()  # Ensure consistent ordering

    urls = upload_exam_images(
        image_paths=[str(img) for img in image_files],
        student_name=student_folder.name,
        assignment_name=assignment_name
    )

    # Save URLs to intermediate directory for later use
    urls_file = Path("intermediateproduct/gcs_urls") / f"{student_folder.name}_urls.json"
    with open(urls_file, 'w') as f:
        json.dump({
            'student_name': student_folder.name,
            'assignment_name': assignment_name,
            'image_urls': urls,
            'upload_date': datetime.now().isoformat()
        }, f, indent=2)

    return urls
```

### Step 2: LLM Processing with URLs

```python
from aita.services.llm.openrouter import create_openrouter_client
from aita.utils.prompts import get_name_extraction_prompt, get_transcription_prompt

def process_exam_with_llm(image_urls: List[str], student_name: str):
    """Process uploaded exam images with LLM"""

    llm = create_openrouter_client()
    results = []

    # Extract name from first page
    name_prompt = get_name_extraction_prompt()
    extracted_name = llm.complete_with_image(name_prompt, image_urls[0])

    # Process each page
    for i, url in enumerate(image_urls):
        page_result = {
            'page_number': i + 1,
            'image_url': url,
            'extracted_name': extracted_name if i == 0 else None
        }

        # Transcribe the page
        transcription_prompt = get_transcription_prompt("General exam content")
        transcription = llm.complete_with_image(transcription_prompt, url)
        page_result['transcription'] = transcription

        results.append(page_result)

    return results
```

### Step 3: Saving Results

```python
def save_processing_results(results: List[Dict], student_name: str, assignment_name: str):
    """Save LLM processing results to intermediate directory"""

    # Save to intermediate directory
    output_file = Path("intermediateproduct/transcriptions") / f"{student_name}_{assignment_name}.json"

    with open(output_file, 'w') as f:
        json.dump({
            'student_name': student_name,
            'assignment_name': assignment_name,
            'processing_date': datetime.now().isoformat(),
            'results': results
        }, f, indent=2)
```

## Error Handling and Best Practices

### Robust Upload with Retry

```python
import time
from typing import Optional

def robust_upload(storage: GoogleCloudStorageService,
                 local_path: str,
                 max_retries: int = 3) -> Optional[str]:
    """Upload with retry logic for network issues"""

    for attempt in range(max_retries):
        try:
            return storage.upload_image(local_path)

        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"Upload failed (attempt {attempt + 1}), retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"Upload failed after {max_retries} attempts: {e}")
                return None
```

### Batch Processing with Progress

```python
from tqdm import tqdm

def upload_all_student_exams(data_dir: Path, assignment_name: str):
    """Upload exams for all students with progress tracking"""

    storage = create_storage_service()
    student_folders = [d for d in data_dir.iterdir() if d.is_dir()]

    results = {}

    for student_folder in tqdm(student_folders, desc="Uploading student exams"):
        try:
            urls = upload_student_exams(student_folder, assignment_name)
            results[student_folder.name] = {
                'status': 'success',
                'urls': urls,
                'count': len(urls)
            }
        except Exception as e:
            results[student_folder.name] = {
                'status': 'error',
                'error': str(e)
            }

    return results
```

### Cost Monitoring

```python
def estimate_storage_costs(total_images: int, avg_size_mb: float):
    """Estimate GCS storage costs"""

    # GCS pricing (approximate, check current rates)
    storage_cost_per_gb_month = 0.020  # Standard storage
    bandwidth_cost_per_gb = 0.12       # Egress to worldwide

    total_size_gb = (total_images * avg_size_mb) / 1024

    # Estimate monthly storage cost
    monthly_storage = total_size_gb * storage_cost_per_gb_month

    # Estimate bandwidth cost (assume each image accessed 3 times)
    bandwidth_usage_gb = total_size_gb * 3
    bandwidth_cost = bandwidth_usage_gb * bandwidth_cost_per_gb

    print(f"Estimated costs for {total_images} images ({total_size_gb:.2f} GB):")
    print(f"  Monthly storage: ${monthly_storage:.2f}")
    print(f"  Bandwidth (3x access): ${bandwidth_cost:.2f}")
    print(f"  Total first month: ${monthly_storage + bandwidth_cost:.2f}")
```

## Troubleshooting

### Common Issues

1. **Authentication Errors**
   ```python
   # Verify credentials file exists and is valid
   import json
   with open("path/to/credentials.json") as f:
       creds = json.load(f)
       print(f"Project ID: {creds.get('project_id')}")
       print(f"Client email: {creds.get('client_email')}")
   ```

2. **Permission Errors**
   ```bash
   # Test bucket access with gsutil
   gsutil ls gs://exam1_uwmadison/
   ```

3. **Network Issues**
   ```python
   # Test basic connectivity
   import requests
   response = requests.get("https://storage.googleapis.com")
   print(f"Status: {response.status_code}")
   ```

### Debugging Upload Issues

```python
def debug_upload_issue(storage: GoogleCloudStorageService, local_path: str):
    """Debug common upload issues"""

    # Check file exists
    file_path = Path(local_path)
    if not file_path.exists():
        print(f"❌ File not found: {local_path}")
        return

    print(f"✅ File exists: {file_path.name} ({file_path.stat().st_size} bytes)")

    # Check file format
    if file_path.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
        print(f"⚠️  Unusual file format: {file_path.suffix}")

    # Test bucket connection
    try:
        storage._test_connection()
        print("✅ Bucket connection successful")
    except Exception as e:
        print(f"❌ Bucket connection failed: {e}")
        return

    # Try upload
    try:
        url = storage.upload_image(local_path)
        print(f"✅ Upload successful: {url}")
    except Exception as e:
        print(f"❌ Upload failed: {e}")
```

## Next Steps

After setting up the storage service:

1. **Test with Sample Images**: Use the demo script to verify everything works
2. **Integrate with Ingest Pipeline**: Upload images after student grouping
3. **Connect to LLM Processing**: Use uploaded URLs in vision API calls
4. **Set up Monitoring**: Track storage usage and costs

For more examples, see:
- `examples/test_storage_demo.py` - Complete demo script
- `tests/services/test_storage.py` - Test examples
- Integration examples in the main AITA workflow documentation