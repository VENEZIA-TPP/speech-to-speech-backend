"""Re-exports every ORM model, so Alembic autogenerate can see all of them.

`alembic/env.py` does `from app.models import *`, and autogenerate only emits a
table it has imported: a model missing from here is silently left out of the
migration. `__all__` is what makes that star-import explicit rather than
"whatever public name happens to be in scope", and it is also why these imports
are not unused despite nothing in this package referring to them.
"""

from app.models.translation_session import TranslationSession, SessionStatus
from app.models.transcription import Transcription
from app.models.translation import Translation

__all__ = [
    "TranslationSession",
    "SessionStatus",
    "Transcription",
    "Translation",
]
