"""Fixed interface labels and a technical glossary for answers in CIS languages.

Labels never pass through a model, so section titles, urgency and safety notes read the same in every answer.
The glossary pins the translation of workshop terms; many mechanics use the Russian loanword, and the glossary
keeps it where it is the common workshop word. Labels and glossary need a native speaker's review.
"""

from ml.contracts import Language

LANGUAGE_NAMES: dict[Language, str] = {
    'ru': 'русский',
    'uz': 'узбекский (латиница)',
    'tg': 'таджикский (кириллица)',
    'ky': 'киргизский (кириллица)',
    'kk': 'казахский (кириллица)',
}

LABEL_KEYS = (
    'section.observation', 'section.cause', 'section.risk', 'section.check', 'section.reference_value',
    'section.operation', 'section.spare_part', 'conflict.title', 'conflict.note', 'assessment', 'original',
    'quote', 'decision_note', 'ml_note', 'urgency.high', 'urgency.medium', 'urgency.low', 'untranslated', 'page',
)

LABELS: dict[Language, dict[str, str]] = {
    'ru': {
        'section.observation': 'Признаки',
        'section.cause': 'Вероятная причина — надо проверить',
        'section.risk': 'Риск',
        'section.check': 'Что проверить',
        'section.reference_value': 'Норма',
        'section.operation': 'Что сделать',
        'section.spare_part': 'Запчасти',
        'conflict.title': 'Руководства пишут по-разному',
        'conflict.note': 'Уточните у мастера или специалиста, какая инструкция подходит вашему двигателю.',
        'assessment': 'Оценка агента, не из руководства',
        'original': 'В руководстве (по-русски)',
        'quote': 'Цитата из руководства',
        'decision_note': 'Решение о ремонте и выпуске машины принимает механик или ответственный специалист.',
        'ml_note': 'Тип неисправности — гипотеза акустической модели, её подтверждает проверка по руководству.',
        'urgency.high': 'высокий — дефектовка до выпуска в рейс',
        'urgency.medium': 'средний — дефектовка на ближайшем ТО, эксплуатация под наблюдением',
        'urgency.low': 'низкий — проверить узел при следующей проверке',
        'untranslated': 'Перевод не прошёл проверку, показан русский текст.',
        'page': 'стр.',
    },
    'uz': {
        'section.observation': 'Belgilar',
        'section.cause': 'Ehtimoliy sabab — tekshirish kerak',
        'section.risk': 'Xavf',
        'section.check': 'Nimani tekshirish kerak',
        'section.reference_value': "Me'yor",
        'section.operation': 'Nima qilish kerak',
        'section.spare_part': 'Ehtiyot qismlar',
        'conflict.title': "Qo'llanmalarda turlicha yozilgan",
        'conflict.note': "Qaysi ko'rsatma dvigatelingizga mos kelishini usta yoki mutaxassisdan aniqlang.",
        'assessment': "Agent bahosi, qo'llanmadan emas",
        'original': "Qo'llanmada (rus tilida)",
        'quote': "Qo'llanmadan iqtibos",
        'decision_note': "Ta'mirlash va mashinani reysga chiqarish haqida qarorni mexanik yoki mas'ul mutaxassis "
                         'qabul qiladi.',
        'ml_note': "Nosozlik turi — akustik modelning taxmini, uni qo'llanma bo'yicha tekshiruv tasdiqlaydi.",
        'urgency.high': 'yuqori — reysga chiqishdan oldin defektovka',
        'urgency.medium': "o'rta — eng yaqin texnik xizmatda defektovka, mashina kuzatuv ostida ishlaydi",
        'urgency.low': "past — keyingi tekshiruvda uzelni ko'rib chiqing",
        'untranslated': "Tarjima tekshiruvdan o'tmadi, rus tilidagi matn ko'rsatildi.",
        'page': 'bet',
    },
    'tg': {
        'section.observation': 'Аломатҳо',
        'section.cause': 'Сабаби эҳтимолӣ — бояд санҷид',
        'section.risk': 'Хатар',
        'section.check': 'Чиро бояд санҷид',
        'section.reference_value': 'Меъёр',
        'section.operation': 'Чӣ бояд кард',
        'section.spare_part': 'Қисмҳои эҳтиётӣ',
        'conflict.title': 'Дар дастурҳо гуногун навишта шудааст',
        'conflict.note': 'Аз усто ё мутахассис пурсед, ки кадом дастур ба муҳаррики шумо мувофиқ аст.',
        'assessment': 'Баҳои агент, на аз дастур',
        'original': 'Дар дастур (бо забони русӣ)',
        'quote': 'Иқтибос аз дастур',
        'decision_note': 'Қарорро дар бораи таъмир ва ба рейс баровардани мошин механик ё мутахассиси масъул '
                         'қабул мекунад.',
        'ml_note': 'Навъи носозӣ — фарзияи модели акустикӣ аст, онро санҷиш аз рӯи дастур тасдиқ мекунад.',
        'urgency.high': 'баланд — дефектовка пеш аз баромадан ба рейс',
        'urgency.medium': 'миёна — дефектовка дар хизматрасонии техникии наздиктарин, мошин зери назорат кор мекунад',
        'urgency.low': 'паст — дар санҷиши навбатӣ агрегатро тафтиш кунед',
        'untranslated': 'Тарҷума аз санҷиш нагузашт, матни русӣ нишон дода шуд.',
        'page': 'саҳ.',
    },
    'ky': {
        'section.observation': 'Белгилер',
        'section.cause': 'Мүмкүн болгон себеп — текшерүү керек',
        'section.risk': 'Коркунуч',
        'section.check': 'Эмнени текшерүү керек',
        'section.reference_value': 'Норма',
        'section.operation': 'Эмне кылуу керек',
        'section.spare_part': 'Запастык бөлүктөр',
        'conflict.title': 'Колдонмолордо ар башкача жазылган',
        'conflict.note': 'Кайсы нускама кыймылдаткычыңызга туура келерин устадан же адистен тактаңыз.',
        'assessment': 'Агенттин баасы, колдонмодон эмес',
        'original': 'Колдонмодо (орус тилинде)',
        'quote': 'Колдонмодон цитата',
        'decision_note': 'Оңдоо жана машинаны рейске чыгаруу боюнча чечимди механик же жооптуу адис кабыл алат.',
        'ml_note': 'Бузулуунун түрү — акустикалык моделдин божомолу, аны колдонмо боюнча текшерүү ырастайт.',
        'urgency.high': 'жогору — рейске чыгардан мурун дефектовка',
        'urgency.medium': 'орто — жакынкы техникалык тейлөөдө дефектовка, машина көзөмөл астында иштейт',
        'urgency.low': 'төмөн — кийинки текшерүүдө түйүндү карап чыгыңыз',
        'untranslated': 'Котормо текшерүүдөн өткөн жок, орусча текст көрсөтүлдү.',
        'page': 'бет',
    },
    'kk': {
        'section.observation': 'Белгілер',
        'section.cause': 'Ықтимал себеп — тексеру керек',
        'section.risk': 'Қауіп',
        'section.check': 'Нені тексеру керек',
        'section.reference_value': 'Норма',
        'section.operation': 'Не істеу керек',
        'section.spare_part': 'Қосалқы бөлшектер',
        'conflict.title': 'Нұсқаулықтарда әртүрлі жазылған',
        'conflict.note': 'Қай нұсқау қозғалтқышыңызға сәйкес келетінін шеберден немесе маманнан нақтылаңыз.',
        'assessment': 'Агенттің бағасы, нұсқаулықтан емес',
        'original': 'Нұсқаулықта (орыс тілінде)',
        'quote': 'Нұсқаулықтан дәйексөз',
        'decision_note': 'Жөндеу және машинаны рейске шығару туралы шешімді механик немесе жауапты маман қабылдайды.',
        'ml_note': 'Ақаулық түрі — акустикалық модельдің болжамы, оны нұсқаулық бойынша тексеру растайды.',
        'urgency.high': 'жоғары — рейске шықпас бұрын дефектовка',
        'urgency.medium': 'орташа — жақын техникалық қызмет көрсетуде дефектовка, машина бақылаумен жұмыс істейді',
        'urgency.low': 'төмен — келесі тексеруде торапты қарап шығыңыз',
        'untranslated': 'Аударма тексеруден өтпеді, орысша мәтін көрсетілді.',
        'page': 'бет',
    },
}

# Russian workshop term -> target term. Keys are lower-case word stems matched against the source text.
GLOSSARY: dict[Language, dict[str, str]] = {
    'uz': {
        'двигател': 'dvigatel', 'форсунк': 'forsunka', 'тнвд': "TNVD (yuqori bosimli yonilg'i nasosi)",
        'топливн насос': "yonilg'i nasosi", 'вкладыш': 'vkladish', 'коленчат': 'tirsakli val',
        'распределительн': 'taqsimlash vali', 'клапан': 'klapan', 'зазор': 'tirqish (zazor)',
        'поршнев кольц': 'porshen halqasi', 'поршен': 'porshen', 'гильз': 'gilza', 'головк цилиндр': 'silindrlar kallagi',
        'прокладк': 'qistirma (prokladka)', 'турбокомпрессор': 'turbokompressor', 'масл': 'moy',
        'охлаждающ жидкост': 'sovutish suyuqligi', 'радиатор': 'radiator', 'термостат': 'termostat',
        'водян насос': 'suv nasosi', 'ремен': 'tasma', 'генератор': 'generator', 'стартер': 'starter',
        'аккумулятор': 'akkumulyator', 'фильтр': 'filtr', 'момент затяжк': 'tortish momenti', 'болт': 'bolt',
        'гайк': 'gayka', 'шатун': 'shatun', 'коренн подшипник': 'asosiy podshipnik', 'свеч зажигани': "o't oldirish shami",
        'карбюратор': 'karbyurator', 'дефектовк': 'defektovka', 'холост ход': 'salt yurish',
        'частот вращени': 'aylanish chastotasi', 'перегрев': 'qizib ketish', 'стук': 'taqillash', 'дым': 'tutun',
        'давлени': 'bosim', 'утечк': 'oqish', 'регулировк': 'sozlash',
    },
    'tg': {
        'двигател': 'муҳаррик', 'форсунк': 'форсунка', 'тнвд': 'ТНВД (насоси фишори баланди сӯзишворӣ)',
        'топливн насос': 'насоси сӯзишворӣ', 'вкладыш': 'вкладиш', 'коленчат': 'вали тирсакдор',
        'распределительн': 'вали тақсимкунанда', 'клапан': 'клапан', 'зазор': 'фосила (зазор)',
        'поршнев кольц': 'ҳалқаи поршен', 'поршен': 'поршен', 'гильз': 'гилза', 'головк цилиндр': 'сари силиндрҳо',
        'прокладк': 'прокладка', 'турбокомпрессор': 'турбокомпрессор', 'масл': 'равған',
        'охлаждающ жидкост': 'моеъи хунуккунанда', 'радиатор': 'радиатор', 'термостат': 'термостат',
        'водян насос': 'насоси об', 'ремен': 'тасма', 'генератор': 'генератор', 'стартер': 'стартер',
        'аккумулятор': 'аккумулятор', 'фильтр': 'филтр', 'момент затяжк': 'моменти мустаҳкамкунӣ', 'болт': 'болт',
        'гайк': 'гайка', 'шатун': 'шатун', 'коренн подшипник': 'подшипники асосӣ', 'свеч зажигани': 'шамъи оташгиронӣ',
        'карбюратор': 'карбюратор', 'дефектовк': 'дефектовка', 'холост ход': 'гардиши холӣ',
        'частот вращени': 'басомади гардиш', 'перегрев': 'аз ҳад зиёд гарм шудан', 'стук': 'тақ-тақ', 'дым': 'дуд',
        'давлени': 'фишор', 'утечк': 'чакидан', 'регулировк': 'танзим',
    },
    'ky': {
        'двигател': 'кыймылдаткыч', 'форсунк': 'форсунка', 'тнвд': 'ТНВД (жогорку басымдагы күйүүчү май насосу)',
        'вкладыш': 'вкладыш', 'коленчат': 'тирсектүү вал', 'клапан': 'клапан', 'зазор': 'боштук (зазор)',
        'поршнев кольц': 'поршень шакеги', 'поршен': 'поршень', 'масл': 'май', 'охлаждающ жидкост': 'муздатуучу суюктук',
        'водян насос': 'суу насосу', 'ремен': 'кайыш', 'фильтр': 'чыпка (фильтр)', 'момент затяжк': 'бекитүү моменти',
        'холост ход': 'бош жүрүш', 'частот вращени': 'айлануу жыштыгы', 'перегрев': 'ысып кетүү', 'стук': 'тыкылдоо',
        'дым': 'түтүн', 'давлени': 'басым', 'регулировк': 'жөнгө салуу',
    },
    'kk': {
        'двигател': 'қозғалтқыш', 'форсунк': 'форсунка', 'тнвд': 'ТНВД (жоғары қысымды отын сорғысы)',
        'вкладыш': 'вкладыш', 'коленчат': 'иінді білік', 'клапан': 'клапан', 'зазор': 'саңылау (зазор)',
        'поршнев кольц': 'поршень сақинасы', 'поршен': 'поршень', 'масл': 'май', 'охлаждающ жидкост': 'салқындатқыш сұйықтық',
        'водян насос': 'су сорғысы', 'ремен': 'белдік', 'фильтр': 'сүзгі', 'момент затяжк': 'тарту моменті',
        'холост ход': 'бос жүріс', 'частот вращени': 'айналу жиілігі', 'перегрев': 'қызып кету', 'стук': 'тықылдау',
        'дым': 'түтін', 'давлени': 'қысым', 'регулировк': 'реттеу',
    },
}


def labels(language: Language) -> dict[str, str]:
    """All fixed labels for ``language``; a missing one falls back to Russian."""
    return {key: LABELS.get(language, {}).get(key) or LABELS['ru'][key] for key in LABEL_KEYS}


def glossary_for(language: Language, texts: list[str]) -> dict[str, str]:
    """Glossary entries whose Russian stems occur in ``texts`` (each word of a key must match a word prefix)."""
    words = ' '.join(texts).lower().replace('ё', 'е').split()
    found = {}
    for key, value in GLOSSARY.get(language, {}).items():
        stems = key.split()
        if all(any(word.strip('.,;:()«»"').startswith(stem) for word in words) for stem in stems):
            found[key] = value
    return found
