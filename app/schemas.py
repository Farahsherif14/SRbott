from typing import Literal
from pydantic import BaseModel, Field

Category = Literal["plumbing", "electricity", "AC", "carpentry", "pest control", "general"]
Urgency = Literal["low", "medium", "high", "emergency"]

class UnitInfo(BaseModel):
    compound: str | None = None
    building: str | None = Field(
        default=None,
        description="Building number: digits optionally followed by one Latin letter, e.g. 40 or 40B.",
    )
    unit_number: str | None = Field(
        default=None,
        description="Unit/apartment number: digits only, with no letters, e.g. 7 or 204.",
    )

class ExtractedRequest(BaseModel):
    name: str | None = None
    phone: str | None = None
    unit: UnitInfo = Field(default_factory=UnitInfo)
    issue_description: str | None = None
    issue_needs_detail: bool = False
    missing_fields: list[str] = Field(default_factory=list)
    follow_up_question: str | None = None

class IntakeMessage(BaseModel):
    chat_id: str
    user_id: str
    text: str

class ConfirmRequest(BaseModel):
    chat_id: str
    user_id: str
    confirmed: bool
    fields: ExtractedRequest

class StatusRequest(BaseModel):
    ticket_number: str
