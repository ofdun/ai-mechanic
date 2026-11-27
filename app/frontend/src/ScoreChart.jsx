import React from 'react'
import { useUiText } from './uiLanguage'

// Same rule and wording as trend() in ml/generation/finalizer.py, so the chat and the chart agree.
const TREND_STEP = 0.1
export const score = value => value.toFixed(2).replace('.', ',')

// ``t`` translates the Russian template; without it the text stays Russian, as in the chat answer.
export function scoreTrend(sessions, t = (text, values) => values ? text.replace(/\{(\w+)\}/g, (_, key) => values[key]) : text) {
  const scores = sessions.map(session => session.anomaly_score).filter(value => value != null)
  if (scores.length < 2) return t('Для оценки динамики нужны хотя бы две успешные проверки.')
  const path = scores.map(score).join(' → ')
  const steps = scores.slice(1).map((later, index) => later - scores[index])
  const first = scores[0], last = scores.at(-1)
  // Rounded to avoid 0.30 - 0.20 = 0.0999… missing the step.
  const atLeast = (difference) => Math.round(difference * 1000) >= TREND_STEP * 1000
  if (steps.every(step => step >= 0) && atLeast(last - first))
    return t('Оценка аномалии растёт: {path}. Состояние ухудшается — стоит запланировать дефектовку.', { path })
  if (steps.every(step => step <= 0) && atLeast(first - last)) return t('Оценка аномалии снижается: {path}.', { path })
  if (atLeast(steps.at(-1))) return t('Оценка аномалии выросла по сравнению с прошлой проверкой: {path}.', { path })
  if (atLeast(-steps.at(-1))) return t('Оценка аномалии снизилась по сравнению с прошлой проверкой: {path}.', { path })
  if (atLeast(Math.max(...scores) - Math.min(...scores))) return t('Оценка аномалии колеблется: {path}.', { path })
  return t('Оценка аномалии стабильна: {path}.', { path })
}

const shortDate = value => new Intl.DateTimeFormat('ru-RU', { day: '2-digit', month: '2-digit' }).format(new Date(value))

const W = 300, H = 170, LEFT = 28, RIGHT = 8, TOP = 22, BOTTOM = 34, INSET = 22

/** anomaly_score of the last checks in time order. A failed check has no score: the line breaks there. */
export function ScoreChart({ sessions, id }) {
  const t = useUiText()
  const points = [...sessions].sort((a, b) => Date.parse(a.checked_at) - Date.parse(b.checked_at))
  if (points.length < 2) return null
  // Points stay clear of the axis labels on the left and of the edge on the right.
  const step = (W - LEFT - RIGHT - 2 * INSET) / (points.length - 1)
  const x = index => LEFT + INSET + index * step
  const y = value => TOP + (1 - value) * (H - TOP - BOTTOM)
  const segments = []
  let current = []
  points.forEach((point, index) => {
    if (point.anomaly_score == null) {
      if (current.length) segments.push(current)
      current = []
    } else current.push([x(index), y(point.anomaly_score)])
  })
  if (current.length) segments.push(current)
  const caption = scoreTrend(points, t)
  return <figure className="score-chart" aria-labelledby={id}>
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-labelledby={id}>
      {[0, 0.5, 1].map(tick => <g key={tick} className="grid">
        <line x1={LEFT} x2={W - RIGHT} y1={y(tick)} y2={y(tick)} />
        <text x={LEFT - 6} y={y(tick)} dy="0.32em" textAnchor="end">{tick === 0.5 ? '0,5' : tick}</text>
      </g>)}
      {segments.filter(segment => segment.length > 1).map(segment =>
        <polyline key={segment[0].join()} className="score-line" points={segment.map(point => point.join(',')).join(' ')} />)}
      {points.map((point, index) => <g key={point.session_id} className={'point ' + point.status}>
        {point.anomaly_score == null ? <>
          <line className="gap" x1={x(index)} x2={x(index)} y1={TOP} y2={H - BOTTOM} />
          <text x={x(index)} y={TOP - 6} textAnchor="middle">{t('сбой')}</text>
        </> : <>
          <circle cx={x(index)} cy={y(point.anomaly_score)} r="4.5" />
          {/* Near the top the value goes under the point so it does not sit on the «1» line. */}
          <text x={x(index)} y={y(point.anomaly_score) + (point.anomaly_score > 0.85 ? 17 : -9)} textAnchor="middle">{score(point.anomaly_score)}</text>
        </>}
        <text className="date" x={x(index)} y={H - BOTTOM + 18} textAnchor="middle">{shortDate(point.checked_at)}</text>
      </g>)}
    </svg>
    <figcaption id={id}>{caption}</figcaption>
  </figure>
}
