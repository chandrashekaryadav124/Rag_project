from decimal import Decimal
from typing import Any


def to_number(value: Any, as_int: bool = False) -> int | float:
    """Convert MySQL numeric/Decimal/None values into JSON-safe Python numbers."""
    if value is None:
        return 0 if as_int else 0.0
    if isinstance(value, Decimal):
        return int(value) if as_int else round(float(value), 2)
    return int(value) if as_int else round(float(value), 2)


def attach_yoy_deltas(
    yearly_rows: list[dict],
    metric_keys: tuple[str, ...],
    year_key: str = "year",
) -> list[dict]:
    """
    Given yearly aggregate rows sorted descending by year, enrich each row
    with previous-year comparison deltas and percentage changes.
    """
    normalized: list[dict] = []
    for row in yearly_rows:
        item = dict(row)
        item[year_key] = str(item.get(year_key) or "").strip()
        normalized.append(item)

    normalized.sort(key=lambda r: r[year_key], reverse=True)
    by_year = {r[year_key]: r for r in normalized if r[year_key].isdigit()}

    enriched: list[dict] = []
    for row in normalized:
        yr = row[year_key]
        prev_yr = str(int(yr) - 1) if yr.isdigit() else None
        prev_row = by_year.get(prev_yr) if prev_yr else None
        row["previous_year"] = prev_yr if prev_row else None

        for key in metric_keys:
            curr_val = to_number(row.get(key))
            if prev_row is not None:
                prev_val = to_number(prev_row.get(key))
                delta = round(curr_val - prev_val, 2)
                pct = (
                    round((delta / prev_val) * 100, 2)
                    if prev_val != 0
                    else None
                )
            else:
                delta = None
                pct = None

            row[f"{key}_vs_prev_year"] = delta
            row[f"{key}_yoy_pct"] = pct

        enriched.append(row)

    return enriched
