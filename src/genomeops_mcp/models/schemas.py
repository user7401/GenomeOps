"""Pydantic models for nf-core pipeline metadata and tool responses."""

from typing import Any
from pydantic import BaseModel, Field


class PipelineSummary(BaseModel):
    name: str
    description: str
    topics: list[str] = Field(default_factory=list)
    latest_version: str
    url: str


class PipelineInfo(BaseModel):
    name: str
    description: str
    topics: list[str] = Field(default_factory=list)
    latest_version: str
    url: str
    docs_url: str
    schema_url: str
    input_schema_url: str
    supported_profiles: list[str] = Field(default_factory=list)
    required_params: list[str] = Field(default_factory=list)
    output_description: str = ""
    extra: dict[str, Any] = Field(default_factory=dict)


class SchemaField(BaseModel):
    name: str
    type: str
    required: bool
    description: str = ""
    allowed_values: list[str] = Field(default_factory=list)
    default: Any = None
    format: str | None = None


class SamplesheetSchema(BaseModel):
    pipeline: str
    version: str
    fields: list[SchemaField]


class ValidationError(BaseModel):
    row: int
    column: str
    message: str
    fix_suggestion: str = ""


class ValidationResult(BaseModel):
    valid: bool
    errors: list[ValidationError] = Field(default_factory=list)
