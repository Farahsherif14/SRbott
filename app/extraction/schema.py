"""Shared contract: the structured SR the LLM must produce."""
from enum import Enum

from pydantic import BaseModel, Field


class Category(str, Enum):
    plumbing = "plumbing"
    electricity = "electricity"
    ac = "ac"
    carpentry = "carpentry"
    pest_control = "pest_control"
    general = "general"


class Urgency(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"
    emergency = "emergency"


REQUIRED_FIELDS = ["compound", "building", "unit_number", "name", "phone", "issue_description"]


class ExtractedSR(BaseModel):
    """All fields optional so partial extraction is valid; use missing_fields()."""

    compound: str | None = Field(default=None, description="Compound / community name")
    building: str | None = None
    unit_number: str | None = None
    name: str | None = None
    phone: str | None = None
    issue_description: str | None = None
    language: str | None = Field(default=None, description="'ar' or 'en'")

    def missing_fields(self) -> list[str]:
        return [f for f in REQUIRED_FIELDS if not getattr(self, f)]

    def is_complete(self) -> bool:
        return not self.missing_fields()
