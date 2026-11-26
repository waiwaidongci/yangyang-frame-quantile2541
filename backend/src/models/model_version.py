from sqlalchemy import JSON, Boolean, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from src.config.database import Base


class ModelVersionRecord(Base):
    """Persisted model artifact and its release status.

    Only one row may have ``is_active=True``.  Release transactions lock the
    singleton :class:`ModelReleaseState` row before comparing and swapping.
    """

    __tablename__ = 'model_versions'

    id: Mapped[int] = mapped_column(primary_key=True)
    version: Mapped[str] = mapped_column(String(80), unique=True, index=True)
    status: Mapped[str] = mapped_column(String(16), index=True, default='candidate')
    is_active: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    artifact: Mapped[dict] = mapped_column(JSON)
    p50_pinball: Mapped[float | None] = mapped_column(Float, nullable=True)
    p90_pinball: Mapped[float | None] = mapped_column(Float, nullable=True)
    dataset_id: Mapped[int | None] = mapped_column(ForeignKey('datasets.id'), nullable=True, index=True)
    settings: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[str] = mapped_column(String(32))


class ModelReleaseState(Base):
    """Singleton pointer row used as a cross-process release lock."""

    __tablename__ = 'model_release_state'

    id: Mapped[int] = mapped_column(Integer, primary_key=True, default=1)
    active_version_id: Mapped[int | None] = mapped_column(ForeignKey('model_versions.id'), nullable=True)
    updated_at: Mapped[str | None] = mapped_column(String(32), nullable=True)
