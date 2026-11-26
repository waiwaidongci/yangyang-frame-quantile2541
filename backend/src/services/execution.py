import json
import logging

from sqlalchemy import select, update

from src.config.database import Session
from src.constants.workflow import RunStatus
from src.domain.project import analyze
from src.domain.versions import models_from_data
from src.models.dataset import Dataset
from src.models.run import Run
from src.services.audit import record
from src.services.model_releases import get_active_record


def recover_jobs():
    with Session.begin() as session:
        session.execute(update(Run).where(Run.status == RunStatus.RUNNING).values(status=RunStatus.QUEUED))


def claim():
    with Session.begin() as session:
        run = session.scalar(select(Run).where(Run.status == RunStatus.QUEUED)
                             .order_by(Run.id).limit(1).with_for_update(skip_locked=True))
        if run is None:
            return None
        run.status = RunStatus.RUNNING
        dataset = session.get(Dataset, run.dataset_id)
        record(session, 'run.started', run.id)
        return run.id, dataset.rows, run.settings


def load_active_model():
    """Load the single released version outside the long computation transaction."""
    with Session() as session:
        active = get_active_record(session)
        if active is None:
            return None
        return active.version, models_from_data(active.artifact)


def execute(job):
    run_id, rows, settings = job
    result = None
    error_message = None
    try:
        active_model = load_active_model()
        if active_model is None:
            result = analyze(rows, settings)
        else:
            active_version, models = active_model
            result = analyze(rows, settings, model_version=active_version, models=models)
        json.dumps(result, ensure_ascii=False, allow_nan=False)
    except ValueError as error:
        result = None
        error_message = str(error)
    except Exception:
        logging.getLogger(__name__).exception('计算运行 %s 失败', run_id)
        result = None
        error_message = '计算遇到内部错误，请检查台账后重试。'
    with Session.begin() as session:
        run = session.get(Run, run_id)
        run.status = RunStatus.FAILED if error_message is not None else RunStatus.COMPLETED
        run.result = result
        run.model_version = None if result is None else result.get('model_version')
        run.error = error_message
        record(session, 'run.' + run.status, run.id)
