"""Keep `python -m app.bot` pointed at the integrated service-request bot."""
from pathlib import Path
import runpy


implementation = Path(__file__).resolve().parents[1] / "bot.py"
runpy.run_path(str(implementation), run_name="__main__")
