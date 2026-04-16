import json
import os
from datetime import date, timedelta
from urllib.parse import urlencode
from urllib.request import urlopen


OPENHOLIDAYS_API_BASE = os.getenv("OPENHOLIDAYS_API_BASE", "https://openholidaysapi.org").rstrip("/")
OPENHOLIDAYS_TIMEOUT_SECONDS = float(os.getenv("OPENHOLIDAYS_TIMEOUT_SECONDS", "8"))


COUNTRY_ALIASES = {
    "GERMANY": "DE",
    "DEUTSCHLAND": "DE",
    "IRELAND": "IE",
    "POLAND": "PL",
    "FRANCE": "FR",
    "SPAIN": "ES",
    "ITALY": "IT",
    "UNITED KINGDOM": "GB",
    "UK": "GB",
    "GREAT BRITAIN": "GB",
}


def normalize_country_iso(raw_country: str | None) -> str:
    value = (raw_country or "").strip()
    if not value:
        return "DE"

    upper = value.upper()
    if len(upper) == 2 and upper.isalpha():
        return upper

    if upper in COUNTRY_ALIASES:
        return COUNTRY_ALIASES[upper]

    return "DE"


def normalize_subdivision_code(raw_subdivision: str | None) -> str | None:
    value = (raw_subdivision or "").strip()
    if not value:
        return None
    return value.upper()


def _fetch_openholidays_json(path: str, params: dict) -> list[dict]:
    query = urlencode(params)
    url = f"{OPENHOLIDAYS_API_BASE}{path}?{query}"
    try:
        with urlopen(url, timeout=OPENHOLIDAYS_TIMEOUT_SECONDS) as response:
            payload = response.read().decode("utf-8")
        parsed = json.loads(payload)
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]
    except Exception:
        return []
    return []


def _parse_date(value) -> date | None:
    if value is None:
        return None

    if isinstance(value, str):
        try:
            return date.fromisoformat(value[:10])
        except ValueError:
            return None

    if isinstance(value, dict):
        year = value.get("year")
        month = value.get("month")
        day = value.get("day")
        if year is not None and month is not None and day is not None:
            try:
                return date(int(year), int(month), int(day))
            except ValueError:
                return None

        nested_value = value.get("date")
        if nested_value is not None:
            return _parse_date(nested_value)

    return None


def _extract_name(item: dict) -> str | None:
    names = item.get("name")
    if not isinstance(names, list):
        return None

    first_text = None
    for entry in names:
        if not isinstance(entry, dict):
            continue
        text = entry.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        language = str(entry.get("language", "")).upper()
        if language == "EN":
            return text.strip()
        if first_text is None:
            first_text = text.strip()
    return first_text


def _date_range(item: dict) -> tuple[date, date] | None:
    start = (
        _parse_date(item.get("startDate"))
        or _parse_date(item.get("start"))
        or _parse_date(item.get("date"))
        or _parse_date(item.get("validFrom"))
    )
    end = (
        _parse_date(item.get("endDate"))
        or _parse_date(item.get("end"))
        or _parse_date(item.get("date"))
        or _parse_date(item.get("validTo"))
    )
    if start is None:
        return None
    if end is None:
        end = start
    if end < start:
        end = start
    return start, end


def _state_holiday_code(public_holiday_name: str) -> str:
    label = public_holiday_name.lower()
    if "christmas" in label:
        return "c"
    if "easter" in label or "good friday" in label:
        return "b"
    return "a"


def _is_nationwide(item: dict) -> bool:
    if bool(item.get("nationwide")):
        return True
    scope = str(item.get("regionalScope", "")).lower()
    return scope == "national"


def build_holiday_context_by_date_range(
    start_date: date,
    end_date: date,
    country_iso_code: str,
    subdivision_code: str | None = None,
) -> dict[str, dict]:
    if end_date < start_date:
        return {}

    public_params = {
        "countryIsoCode": country_iso_code,
        "validFrom": start_date.isoformat(),
        "validTo": end_date.isoformat(),
        "languageIsoCode": "EN",
    }
    school_params = dict(public_params)
    if subdivision_code:
        school_params["subdivisionCode"] = subdivision_code

    public_holidays = _fetch_openholidays_json("/PublicHolidays", public_params)
    school_holidays = _fetch_openholidays_json("/SchoolHolidays", school_params)

    by_date: dict[str, dict] = {}
    day = start_date
    while day <= end_date:
        by_date[day.isoformat()] = {
            "state_holiday": "0",
            "school_holiday": 0,
            "public_holiday_name": None,
            "school_holiday_name": None,
        }
        day += timedelta(days=1)

    for item in public_holidays:
        if not subdivision_code and not _is_nationwide(item):
            continue
        span = _date_range(item)
        if span is None:
            continue
        span_start, span_end = span
        holiday_name = _extract_name(item) or "Public holiday"
        day = max(span_start, start_date)
        stop = min(span_end, end_date)
        while day <= stop:
            key = day.isoformat()
            if key in by_date:
                by_date[key]["state_holiday"] = _state_holiday_code(holiday_name)
                by_date[key]["public_holiday_name"] = holiday_name
            day += timedelta(days=1)

    for item in school_holidays:
        if not subdivision_code and not _is_nationwide(item):
            continue
        span = _date_range(item)
        if span is None:
            continue
        span_start, span_end = span
        holiday_name = _extract_name(item) or "School holiday"
        day = max(span_start, start_date)
        stop = min(span_end, end_date)
        while day <= stop:
            key = day.isoformat()
            if key in by_date:
                by_date[key]["school_holiday"] = 1
                if by_date[key]["school_holiday_name"] is None:
                    by_date[key]["school_holiday_name"] = holiday_name
            day += timedelta(days=1)

    return by_date


def build_holiday_context_by_horizon(
    issue_date: date,
    max_horizon: int,
    country_iso_code: str,
    subdivision_code: str | None = None,
) -> dict[int, dict]:
    start_date = issue_date + timedelta(days=1)
    end_date = issue_date + timedelta(days=max_horizon)
    by_date = build_holiday_context_by_date_range(
        start_date=start_date,
        end_date=end_date,
        country_iso_code=country_iso_code,
        subdivision_code=subdivision_code,
    )

    out: dict[int, dict] = {}
    for offset in range(1, max_horizon + 1):
        d = issue_date + timedelta(days=offset)
        out[offset] = by_date[d.isoformat()]
    return out
