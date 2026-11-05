# Shared Contracts (frozen at Week 2 scope gate)

1. **Labels**: see data/labeling_guide.md
2. **SR JSON schema**: app/extraction/schema.py (ExtractedSR)
3. **DB schema**: app/db/schema.sql
4. **Memory rules**: Redis session keyed by chat ID (30 min TTL); profile in Postgres;
   LLM only sees the current user's data; unit info saved only after user confirmation;
   no PII in logs.

Any change to these needs a PR approved by all four interns.
