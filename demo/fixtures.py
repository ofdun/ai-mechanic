from ml.contracts import SessionRecord, Vehicle

DOCUMENT_ID = 'training-stand-manual'
MANUAL = 'Учебная инструкция стенда Д-01'
PDF_NAME = 'training-stand.pdf'
ENGINE = 'Учебный Д-01'
RISK = 'Ошибка настройки учебного стенда может исказить результат следующего замера.'
OPERATION = 'Сверить маркировку учебного узла с карточкой стенда.'
GUIDE_ENTRIES = (
    ('Почему это опасно?', 'risk', RISK),
    ('Что проверить?', 'operation', OPERATION),
)
PAGES = (
    'Демонстрационные данные. Инструкция относится только к учебному стенду Д-01.',
    'Учебная проверка топливной системы.\n' + '\n'.join(
        f'{question}\n{answer}' for question, _, answer in GUIDE_ENTRIES
    ),
)


def training_fleet():
    def session(vehicle_id, session_id, status, day):
        anomaly = status == 'anomaly'
        failed = status == 'check_failed'
        return SessionRecord(
            vehicle_id=vehicle_id, session_id=session_id, status=status,
            anomaly_type='F08' if anomaly else None,
            anomaly_title='Учебная топливная система' if anomaly else None,
            anomaly_score=None if failed else (0.84 if anomaly else 0.08),
            confidence=None if failed else 0.91, signal_quality='bad' if failed else 'good',
            description='Синтетический замер учебного стенда.', model_version='training-1',
            checked_at=f'2026-01-{day:02d}T12:00:00Z',
        )

    return [
        Vehicle(vehicle_id='ДЕМО-01', engine_model=ENGINE, fuel='дизель', sessions=[
            session('ДЕМО-01', 'F08-DEMO', 'anomaly', 12),
        ]),
        Vehicle(vehicle_id='ДЕМО-02', engine_model='Учебный Д-02', fuel='дизель', sessions=[
            session('ДЕМО-02', 'NORMAL-DEMO', 'normal', 13),
            session('ДЕМО-02', 'REPEAT-DEMO', 'check_failed', 14),
        ]),
    ]
