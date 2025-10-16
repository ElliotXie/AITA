import os
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class OpenRouterConfig:
    api_key: str
    model: str = "google/gemini-2.5-flash"
    max_retries: int = 3
    retry_delay: float = 1.0

    @classmethod
    def from_env(cls) -> "OpenRouterConfig":
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is required")

        return cls(
            api_key=api_key,
            model=os.getenv("OPENROUTER_MODEL", cls.model),
            max_retries=int(os.getenv("AITA_MAX_RETRIES", cls.max_retries)),
            retry_delay=float(os.getenv("AITA_RETRY_DELAY", cls.retry_delay))
        )


@dataclass
class GoogleCloudConfig:
    project_id: str
    bucket_name: str
    credentials_path: Optional[str] = None

    @classmethod
    def from_env(cls) -> "GoogleCloudConfig":
        project_id = os.getenv("GOOGLE_CLOUD_PROJECT_ID")
        bucket_name = os.getenv("GOOGLE_CLOUD_STORAGE_BUCKET")

        if not project_id:
            raise ValueError("GOOGLE_CLOUD_PROJECT_ID environment variable is required")
        if not bucket_name:
            raise ValueError("GOOGLE_CLOUD_STORAGE_BUCKET environment variable is required")

        return cls(
            project_id=project_id,
            bucket_name=bucket_name,
            credentials_path=os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        )


@dataclass
class OCRConfig:
    crop_top_percent: int = 20
    confidence_threshold: float = 0.5
    languages: list = None

    def __post_init__(self):
        if self.languages is None:
            self.languages = ['en']

    @classmethod
    def from_env(cls) -> "OCRConfig":
        return cls(
            crop_top_percent=int(os.getenv("OCR_CROP_TOP_PERCENT", cls.crop_top_percent)),
            confidence_threshold=float(os.getenv("OCR_CONFIDENCE_THRESHOLD", cls.confidence_threshold))
        )


@dataclass
class FuzzyMatchConfig:
    threshold: int = 80

    @classmethod
    def from_env(cls) -> "FuzzyMatchConfig":
        return cls(
            threshold=int(os.getenv("FUZZY_MATCH_THRESHOLD", cls.threshold))
        )


@dataclass
class ParallelExecutionConfig:
    """Configuration for parallel LLM execution."""
    max_workers: int = 5
    rate_limit_rps: float = 10.0  # requests per second
    enable_checkpointing: bool = True
    checkpoint_interval: int = 10
    retry_failed_tasks: bool = True
    show_progress: bool = True

    @classmethod
    def from_env(cls) -> "ParallelExecutionConfig":
        return cls(
            max_workers=int(os.getenv("AITA_MAX_WORKERS", cls.max_workers)),
            rate_limit_rps=float(os.getenv("AITA_RATE_LIMIT_RPS", cls.rate_limit_rps)),
            enable_checkpointing=os.getenv("AITA_ENABLE_CHECKPOINTING", str(cls.enable_checkpointing)).lower() == "true",
            checkpoint_interval=int(os.getenv("AITA_CHECKPOINT_INTERVAL", cls.checkpoint_interval)),
            retry_failed_tasks=os.getenv("AITA_RETRY_FAILED_TASKS", str(cls.retry_failed_tasks)).lower() == "true",
            show_progress=os.getenv("AITA_SHOW_PROGRESS", str(cls.show_progress)).lower() == "true"
        )


@dataclass
class AITAConfig:
    data_dir: Path
    intermediate_dir: Path
    openrouter: OpenRouterConfig
    google_cloud: GoogleCloudConfig
    ocr: OCRConfig
    fuzzy_match: FuzzyMatchConfig
    parallel_execution: ParallelExecutionConfig
    log_level: str = "INFO"
    default_pages_per_student: int = 5

    @classmethod
    def from_env(cls) -> "AITAConfig":
        data_dir = Path(os.getenv("AITA_DATA_DIR", "./data"))
        intermediate_dir = Path(os.getenv("AITA_INTERMEDIATE_DIR", "./intermediateproduct"))
        log_level = os.getenv("AITA_LOG_LEVEL", "INFO")
        default_pages = int(os.getenv("DEFAULT_PAGES_PER_STUDENT", 5))

        return cls(
            data_dir=data_dir,
            intermediate_dir=intermediate_dir,
            openrouter=OpenRouterConfig.from_env(),
            google_cloud=GoogleCloudConfig.from_env(),
            ocr=OCRConfig.from_env(),
            fuzzy_match=FuzzyMatchConfig.from_env(),
            parallel_execution=ParallelExecutionConfig.from_env(),
            log_level=log_level,
            default_pages_per_student=default_pages
        )

    def ensure_data_directories(self) -> None:
        directories = [
            self.data_dir,
            self.data_dir / "raw_images",
            self.data_dir / "grouped",
            self.data_dir / "results",
            self.data_dir / "results" / "transcriptions",
            self.data_dir / "reports",
            self.intermediate_dir,
            self.intermediate_dir / "gcs_urls",
            self.intermediate_dir / "student_groups",
            self.intermediate_dir / "question_extractions",
            self.intermediate_dir / "rubrics",
            self.intermediate_dir / "transcriptions",
            self.intermediate_dir / "grades"
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)


# Global configuration instance
_config: Optional[AITAConfig] = None


def get_config() -> AITAConfig:
    global _config
    if _config is None:
        _config = AITAConfig.from_env()
    return _config


def set_config(config: AITAConfig) -> None:
    global _config
    _config = config