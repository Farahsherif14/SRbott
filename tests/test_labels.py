"""Tests for app/labels.py: the labels exist in one place, with the team's values. No database."""

from app import labels
from app.db import models
from app.extraction.schema import Category as ExtractionCategory


def test_database_uses_the_shared_lists_not_a_copy() -> None:
    # "is" = the very same object, so the CHECK rules can't drift away from app/labels.py
    assert models.CATEGORIES is labels.CATEGORIES
    assert models.URGENCIES is labels.URGENCIES
    assert models.LANGUAGES is labels.LANGUAGES
    assert models.MESSAGE_ROLES is labels.MESSAGE_ROLES


def test_label_values() -> None:
    # Migration 1 writes these lists out in full: a change here needs a new migration
    assert labels.CATEGORIES == (
        "plumbing",
        "electricity",
        "ac",
        "carpentry",
        "pest_control",
        "general",
    )
    assert labels.URGENCIES == ("low", "medium", "high")
    assert labels.LANGUAGES == ("ar", "en")
    assert labels.MESSAGE_ROLES == ("user", "assistant")


def test_categories_match_the_extraction_schema() -> None:
    # The team contract (docs/contracts.md): the extraction schema uses the same 6 categories
    assert set(labels.CATEGORIES) == {category.value for category in ExtractionCategory}
