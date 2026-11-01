from fastapi import FastAPI

app = FastAPI(title="SR Bot Orchestrator")


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}
