"""The Oracle: one engine, many domains, many learners.

No subject matter lives in this package. Domains are data.
"""

from __future__ import annotations

__version__ = "0.1.0"


def main() -> None:
    """Console entry point. Delegates to the Typer app."""
    from the_oracle.cli import app

    app()


__all__ = ["__version__", "main"]
