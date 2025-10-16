"""
AITA Report Generation

Modules for generating various types of reports (HTML, PDF, etc.)
for exam transcription and grading results.
"""

from .transcription_report import TranscriptionReportGenerator, generate_transcription_reports

__all__ = [
    'TranscriptionReportGenerator',
    'generate_transcription_reports'
]
