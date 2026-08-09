"""add ws_token to translation_sessions

Revision ID: 0726eb3362ee
Revises: 7b179e326c4f
Create Date: 2026-08-09 10:50:26.719533

"""

import secrets
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "0726eb3362ee"
down_revision: Union[str, Sequence[str], None] = "7b179e326c4f"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        "translation_sessions",
        sa.Column("ws_token", sa.String(length=64), nullable=True),
    )
    # Backfill existing rows with the same CSPRNG + format the model uses for
    # new inserts (secrets.token_urlsafe(32)) - not SQL random(), which is
    # documented as non-cryptographic and would collapse entropy for a
    # security token.
    conn = op.get_bind()
    ids = (
        conn.execute(
            sa.text("SELECT id FROM translation_sessions WHERE ws_token IS NULL")
        )
        .scalars()
        .all()
    )
    for row_id in ids:
        conn.execute(
            sa.text("UPDATE translation_sessions SET ws_token = :t WHERE id = :i"),
            {"t": secrets.token_urlsafe(32), "i": row_id},
        )
    op.alter_column("translation_sessions", "ws_token", nullable=False)
    op.create_index(
        op.f("ix_translation_sessions_ws_token"),
        "translation_sessions",
        ["ws_token"],
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(
        op.f("ix_translation_sessions_ws_token"), table_name="translation_sessions"
    )
    op.drop_column("translation_sessions", "ws_token")
