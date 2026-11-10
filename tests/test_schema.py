from app.extraction.schema import ExtractedSR


def test_missing_fields():
    sr = ExtractedSR(compound="Uptown", building="B", unit_number="204")
    assert set(sr.missing_fields()) == {"name", "phone", "issue_description"}
    assert not sr.is_complete()


def test_complete():
    sr = ExtractedSR(
        compound="Uptown",
        building="B",
        unit_number="204",
        name="Ali",
        phone="+201000000000",
        issue_description="Leaking sink",
    )
    assert sr.is_complete()
