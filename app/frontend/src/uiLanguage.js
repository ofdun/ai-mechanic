import { createContext, useContext } from 'react'
import uz from './locales/uz.json'
import tg from './locales/tg.json'
import ky from './locales/ky.json'
import kk from './locales/kk.json'
import hi from './locales/hi.json'

const dictionaries = { uz, tg, ky, kk, hi }
export const UiLanguage = createContext('ru')

export function translator(language) {
  return (source, values) => {
    const dictionary = dictionaries[language]
    const text = dictionary && Object.hasOwn(dictionary, source) ? dictionary[source] : source
    return values ? text.replace(/\{(\w+)\}/g, (_, key) => values[key]) : text
  }
}

export function useUiText() {
  return translator(useContext(UiLanguage))
}
