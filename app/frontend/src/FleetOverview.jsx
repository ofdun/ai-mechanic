import React, { useMemo, useState } from 'react'
import { ChevronRight } from 'lucide-react'
import { formatDate } from './api'

// Labels for the statuses the worker computes in ml/tools/fleet_status.py.
const states = [
  { id: 'quarantine', label: 'Карантин', rule: 'Надёжное отклонение с высоким приоритетом: дефектовка до выпуска в рейс.' },
  { id: 'attention', label: 'Требует внимания', rule: 'Отклонение с приоритетом ниже высокого или запись, которой нельзя доверять: сбой, помехи, низкая уверенность ИИ. Такую машину нужно перемерить — это не норма и не доказанная неисправность.' },
  { id: 'normal', label: 'Норма', rule: 'Последняя проверка без отклонений.' },
]

function vehicleName({ make, model }) {
  return model?.startsWith(make || '') ? model : [make, model].filter(Boolean).join(' ')
}

/** B3: the whole fleet on one screen, most urgent first; a row opens the vehicle's check history. */
export function FleetOverview({ catalog, onOpen }) {
  const [filter, setFilter] = useState('')
  const rated = useMemo(() => catalog.filter(vehicle => vehicle.fleet_status)
    .sort((a, b) => a.fleet_status.rank - b.fleet_status.rank), [catalog])
  if (!rated.length) return <p role="status">Статусы парка недоступны: обновите сервис агента.</p>
  const counts = Object.fromEntries(states.map(({ id }) => [id, rated.filter(v => v.fleet_status.state === id).length]))
  const shown = filter ? rated.filter(vehicle => vehicle.fleet_status.state === filter) : rated
  return <section className="fleet" aria-labelledby="fleet-title">
    <h2 id="fleet-title">Парк · {rated.length} машин</h2>
    <div className="fleet-counters" role="group" aria-label="Фильтр по статусу">
      {states.map(({ id, label }) => <button key={id} type="button" className={'counter ' + id} aria-pressed={filter === id}
        onClick={() => setFilter(filter === id ? '' : id)}><strong>{counts[id]}</strong>{label}</button>)}
    </div>
    <details className="fleet-rules"><summary>Как считаются статусы и порядок</summary>
      <dl>{states.map(({ id, label, rule }) => <React.Fragment key={id}><dt>{label}</dt><dd>{rule}</dd></React.Fragment>)}</dl>
      <p>Статус берётся по последней проверке машины. Порядок: карантин, затем отклонения по приоритету, затем перемер, затем норма; внутри группы — по оценке аномалии. Приоритет тот же, что в заявке на дефектовку. Стоимость ремонта и простоя не оценивается.</p>
    </details>
    <ol className="fleet-list">{shown.map(vehicle => {
      const { fleet_status: status } = vehicle
      return <li key={vehicle.vehicle_id} className={'fleet-row ' + status.state}>
        <button type="button" onClick={() => onOpen(vehicle)} aria-label={`${vehicle.vehicle_id}: история проверок`}>
          <span className="fleet-rank" aria-hidden="true">{status.rank}</span>
          <span className="fleet-main">
            <strong>{vehicle.vehicle_id}</strong> <span className="meta">{vehicleName(vehicle)} · {vehicle.engine_model}</span>
            <span className="fleet-reason"><span className={'fleet-state ' + status.state}>{states.find(s => s.id === status.state).label}</span> {status.reason}</span>
            {status.checked_at && <span className="meta">Проверка {status.session_id} · {formatDate(status.checked_at)}</span>}
          </span>
          <ChevronRight aria-hidden="true" />
        </button>
      </li>
    })}</ol>
  </section>
}
