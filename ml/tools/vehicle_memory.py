"""B1: what the agent knows about the selected vehicle from its own checks and the tickets it created.

Only data of this vehicle is used. A date is the date of the check, never a forecast."""
from collections import Counter

from ml.contracts import DiagnosticResult, PastTicket, SessionRecord, Vehicle

STATUS_NAMES = {'normal': 'норма', 'anomaly': 'отклонение', 'check_failed': 'сбой'}


def day(session: SessionRecord) -> str:
    return f'{session.checked_at:%d.%m.%Y}'


def dates(sessions: list[SessionRecord]) -> str:
    return ', '.join(f'{day(session)} ({session.session_id})' for session in sessions)


def tickets_line(tickets: list[PastTicket] | None) -> str:
    if tickets is None:
        return 'Журнал заявок сейчас недоступен: прошлые заявки не учтены.'
    if not tickets:
        return 'Заявок по этой машине агент ещё не создавал.'
    listed = '; '.join(f'{ticket.ticket_id} — проверка {ticket.session_id}, {ticket.anomaly_type}' for ticket in tickets)
    return f'Заявки по этой машине: {listed}.'


def diagnosis_memory(vehicle: Vehicle, current: SessionRecord, result: DiagnosticResult,
                     tickets: list[PastTicket] | None, reliable_sessions: list[SessionRecord]) -> list[str]:
    """Compares a reliable new diagnosis with the earlier checks of the same vehicle."""
    earlier = sorted((session for session in reliable_sessions if session.checked_at < current.checked_at),
                     key=lambda session: session.checked_at)
    if not earlier:
        lines = ['Для сравнения нет пригодных предыдущих проверок этой машины.'
                 if any(session.checked_at < current.checked_at for session in vehicle.sessions)
                 else 'Это первая проверка этой машины в системе: сравнить не с чем.']
    elif result.status == 'anomaly':
        same = [session for session in earlier
                if session.status == 'anomaly' and session.anomaly_type == result.anomaly_type]
        fault = f'«{result.anomaly_title}» ({result.anomaly_type})'
        lines = [f'Повтор: {fault} уже находили на этой машине {dates(same)}.' if same
                 else f'Раньше {fault} на этой машине не находили (пригодных проверок до этой: {len(earlier)}).']
    else:
        found = [session for session in earlier if session.status == 'anomaly']
        lines = ['Раньше на этой машине находили: '
                 + '; '.join(f'{session.anomaly_type} «{session.anomaly_title}» — {day(session)}' for session in found) + '.'
                 if found else f'Раньше отклонений на этой машине не находили (пригодных проверок до этой: {len(earlier)}).']
    return [*lines, tickets_line(tickets)]


def history_memory(vehicle: Vehicle, tickets: list[PastTicket] | None,
                   reliable_sessions: list[SessionRecord]) -> list[str]:
    """Counts over all checks of the vehicle, repeated faults and the tickets created by the agent."""
    sessions = sorted(vehicle.sessions, key=lambda session: session.checked_at)
    if not sessions:
        return ['Проверок этой машины в системе нет.', tickets_line(tickets)]
    counts = Counter(session.status for session in sessions)
    parts = ', '.join(f'{STATUS_NAMES[status]} {counts[status]}' for status in STATUS_NAMES if counts[status])
    lines = [f'Всего проверок: {len(sessions)} — {parts}.']
    faults: dict[str, list[SessionRecord]] = {}
    for session in sorted(reliable_sessions, key=lambda session: session.checked_at):
        if session.status == 'anomaly':
            faults.setdefault(session.anomaly_type, []).append(session)
    repeated = [(code, found) for code, found in faults.items() if len(found) > 1]
    lines.extend(f'Повторялось: «{found[0].anomaly_title}» ({code}) — {len(found)} раза: {dates(found)}.'
                 for code, found in repeated)
    if faults and not repeated:
        lines.append('На пригодных проверках повторов одной и той же неисправности не было.')
    if counts['anomaly'] and not faults:
        lines.append('Недостаточно надёжных результатов, чтобы оценить повторы отклонений.')
    lines.append(tickets_line(tickets))
    return lines
