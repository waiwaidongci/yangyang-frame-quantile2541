from fastapi import APIRouter

from src.controllers.dependencies import Actor, Db, Writer
from src.services.catalog import get_dataset
from src.services.model_releases import get_active_record, publish_model
from src.services.runs import validate_settings
from src.domain.release_workflow import train_version

router = APIRouter(tags=['model-releases'])


def _record_info(record):
    if record is None:
        return None
    return {
        'id': record.id,
        'version': record.version,
        'model_version': record.version,
        'status': record.status,
        'is_active': record.is_active,
        'p50_pinball': record.p50_pinball,
        'p90_pinball': record.p90_pinball,
        'dataset_id': record.dataset_id,
        'message': record.message,
        'created_at': record.created_at,
    }


@router.get('/models/active')
@router.get('/model-versions/active')
@router.get('/model_versions/active')
@router.get('/model-releases/active')
@router.get('/model_releases/active')
@router.get('/releases/active')
def active_model(session: Db, actor: Actor):
    record = get_active_record(session)
    return {'active': _record_info(record), 'model_version': None if record is None else record.version}


@router.post('/models')
@router.post('/models/publish')
@router.post('/models/releases')
@router.post('/model-versions')
@router.post('/model_versions')
@router.post('/model-releases')
@router.post('/model_releases')
@router.post('/releases/publish')
@router.post('/models/compare-and-switch')
def publish_release(payload: dict, session: Db, actor: Writer):
    dataset_id = payload.get('dataset_id', payload.get('dataset', payload.get('datasetId')))
    if dataset_id is None:
        raise ValueError('发布模型需要 dataset_id')
    dataset = get_dataset(session, int(dataset_id))
    settings = validate_settings(payload.get('settings') or {})
    requested_version = payload.get('version') or payload.get('model_version') or payload.get('name')
    candidate = train_version(dataset.rows, settings, requested_version)
    holdout = [row for row in dataset.rows if row.get('split') == 'holdout']
    record, decision = publish_model(
        session,
        version=candidate.version,
        models=candidate.models,
        holdout_rows=holdout,
        dataset_id=dataset.id,
        settings=settings,
        commit=True,
    )
    active = get_active_record(session)
    return {
        **decision,
        'model_version': record.version,
        'candidate_model_version': candidate.version,
        'active_model_version': decision['active_version'],
        'active': _record_info(active),
        'record': _record_info(record),
    }
