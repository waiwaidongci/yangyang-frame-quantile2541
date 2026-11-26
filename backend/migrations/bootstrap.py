from sqlalchemy import inspect, text

from src.config.database import Base, Session, engine
from src.models import account, audit, dataset, model_version, review, run  # noqa: F401
from src.models.model_version import ModelReleaseState
from src.utils.time import utc_stamp


def bootstrap():
    Base.metadata.create_all(engine)
    # create_all does not add columns to a table created by an older snapshot.
    inspector = inspect(engine)
    if "analysis_runs" in inspector.get_table_names():
        columns = {column["name"] for column in inspector.get_columns("analysis_runs")}
        if "model_version" not in columns:
            with engine.begin() as connection:
                connection.execute(text("ALTER TABLE analysis_runs ADD COLUMN model_version VARCHAR(80)"))

    with Session.begin() as session:
        if session.get(ModelReleaseState, 1) is None:
            session.add(ModelReleaseState(id=1, active_version_id=None, updated_at=utc_stamp()))
