import React, { useContext, useMemo, useState } from 'react'
import { ChevronRight } from 'lucide-react'
import { formatDate } from './api'
import { score } from './ScoreChart'
import { UiLanguage, useUiText } from './uiLanguage'

// Labels for the statuses the worker computes in ml/tools/fleet_status.py.
const states = [
  { id: 'quarantine', label: 'Карантин', rule: 'Надёжное отклонение с высоким приоритетом: дефектовка до выпуска в рейс.' },
  { id: 'attention', label: 'Требует внимания', rule: 'При надёжном отклонении следуйте рекомендации в строке машины. При сбое записи, помехах или низкой уверенности ИИ повторите замер.' },
  { id: 'normal', label: 'Норма', rule: 'Последняя проверка без отклонений.' },
]

const steps = {
  high: 'дефектовка до выпуска в рейс',
  medium: 'дефектовка на ближайшем ТО, эксплуатация под наблюдением',
  low: 'проверить узел при следующей проверке',
}

// The worker's Russian reason, rebuilt from its fields in another language. The fault name comes from the data.
function reason(status, t, language) {
  if (language === 'ru') return status.reason
  if (status.kind === 'fault') return `${status.anomaly_type} «${status.anomaly_title}», ` +
    t('оценка аномалии {value}', { value: score(status.anomaly_score) }) + ': ' + t(steps[status.urgency]) + '.'
  return t({ remeasure: 'Запись нельзя использовать для вывода: нужен повторный замер. Это не норма и не доказанная неисправность.',
    normal: 'Последняя проверка без отклонений.', no_checks: 'Проверок нет: нужен первый замер.' }[status.kind])
}

function vehicleName({ make, model }) {
  return model?.startsWith(make || '') ? model : [make, model].filter(Boolean).join(' ')
}

/** B3: the whole fleet on one screen, most urgent first; a row opens the vehicle's check history. */
export function FleetOverview({ catalog, onOpen }) {
  const t = useUiText()
  const language = useContext(UiLanguage)
  const [filter, setFilter] = useState('')
  const rated = useMemo(() => catalog.filter(vehicle => vehicle.fleet_status)
    .sort((a, b) => a.fleet_status.rank - b.fleet_status.rank), [catalog])
  if (!rated.length) return <p role="status">{t('Статусы парка недоступны: обновите сервис агента.')}</p>
  const counts = Object.fromEntries(states.map(({ id }) => [id, rated.filter(v => v.fleet_status.state === id).length]))
  const shown = filter ? rated.filter(vehicle => vehicle.fleet_status.state === filter) : rated
  return <section className="fleet" aria-labelledby="fleet-title">
    <h2 id="fleet-title">{t('Парк · {count} машин', { count: rated.length })}</h2>
    <div className="fleet-counters" role="group" aria-label={t('Фильтр по статусу')}>
      {states.map(({ id, label }) => <button key={id} type="button" className={'counter ' + id} aria-pressed={filter === id}
        onClick={() => setFilter(filter === id ? '' : id)}><strong>{counts[id]}</strong>{t(label)}</button>)}
    </div>
    <a className="fleet-rules-link" href="#fleet-rules">{t('Как считаются статусы и порядок')}</a>
    <ol className="fleet-list">{shown.map(vehicle => {
      const { fleet_status: status } = vehicle
      return <li key={vehicle.vehicle_id} className={'fleet-row ' + status.state}>
        <button type="button" onClick={() => onOpen(vehicle)} aria-label={t('{id}: история проверок', { id: vehicle.vehicle_id })}>
          <span className="fleet-rank" aria-hidden="true">{status.rank}</span>
          <span className="fleet-main">
            <strong>{vehicle.vehicle_id}</strong> <span className="meta">{vehicleName(vehicle)} · {vehicle.engine_model}</span>
            <span className="fleet-reason"><span className={'fleet-state ' + status.state}>{t(states.find(s => s.id === status.state).label)}</span> {reason(status, t, language)}</span>
            {status.checked_at && <span className="meta">{t('Проверка')} {status.session_id} · {formatDate(status.checked_at, language)}</span>}
          </span>
          <ChevronRight aria-hidden="true" />
        </button>
      </li>
    })}</ol>
    <section className="fleet-rules" id="fleet-rules"><h3>{t('Как считаются статусы и порядок')}</h3>
      <dl>{states.map(({ id, label, rule }) => <React.Fragment key={id}><dt>{t(label)}</dt><dd>{t(rule)}</dd></React.Fragment>)}</dl>
      <p>{t('Очередь строится по последней проверке: карантин, отклонения по приоритету, повторный замер, норма. При равном приоритете выше машина с большей оценкой аномалии. Приоритет заявки уточняется при дефектовке. Стоимость ремонта и простоя не оценивается.')}</p>
    </section>
  </section>
}
