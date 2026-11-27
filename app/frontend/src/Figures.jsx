import React from 'react'
import { Source } from './ManualReader'
import { useUiText } from './uiLanguage'

/** «В каком порядке затягивать…»: the figure from the manual; numbered steps only when people checked them. */
export function TighteningFigures({ result, className = 'claims' }) {
  const t = useUiText()
  if (!result.figures?.length) return null
  return <section className={className + ' tightening-figures'}><h4>{t('Порядок затяжки')}</h4>
    <div>{result.figures.map(figure => <div key={figure.document_id + figure.figure}>
      <p><strong>{t('Рисунок {number}', { number: figure.figure })}</strong>{figure.caption && <span lang="ru"> · {figure.caption}</span>}</p>
      {figure.sequence.length ? <ol>{figure.sequence.map((step, index) => <li key={index} lang="ru">{step}</li>)}</ol>
        : <p className="meta">{t('Последовательность показана только на рисунке. Откройте страницу: агент не перечисляет её без проверенных данных.')}</p>}
      <Source source={figure} />
    </div>)}</div>
  </section>
}
