"""
FEWS NET IPC food-insecurity phase classifications for Afghanistan --
NOT the training target (see compute_spi.py's docstring for why: IPC phase
mixes conflict, economic access, and other non-meteorological drivers with
drought, so it would be a worse label than SPI-3 for a *drought* early-warning
system specifically). Used here only as an independent real-world check: do
the SPI-3 drought months this project identifies actually line up with
periods FEWS NET itself called food-security crises?

No API key needed -- https://fdw.fews.net/api/, the same public endpoint
zaminai/fews_net.py already uses for a different purpose (market prices +
IPC lookups on demand, not a historical pull like this).
"""
import csv
import os
import time

import requests

_BASE = "https://fdw.fews.net/api"


def fetch_national_ipc(country_code="AF", page_size=500, max_pages=2, timeout=60):
    # max_pages=2 (1000 rows): the API returns HTTP 403, not an empty page,
    # past offset=1000 for this query regardless of pacing -- a real,
    # reproducible server-side cap, not a rate limit retries can work around.
    # 1000 rows is still a substantial sample for this script's actual job
    # (an independent sanity check against the SPI-3 labels, not the
    # training target itself).
    """National-level 'Current Situation' IPC classifications, paginated."""
    rows = []
    url = f"{_BASE}/ipcphase.json"
    params = {
        "country_code": country_code, "fields": "simple", "format": "json",
        "page_size": page_size, "geographic_unit_name": "Afghanistan",
        "scenario": "CS",  # Current Situation, not a projection
    }
    for page in range(max_pages):
        for attempt in range(3):
            resp = requests.get(url, params=params, timeout=timeout)
            if resp.status_code == 200:
                break
            print(f"  page {page+1} attempt {attempt+1}: HTTP {resp.status_code}, retrying in 5s...", flush=True)
            time.sleep(5)
        else:
            print(f"  page {page+1}: giving up after 3 attempts, stopping with {len(rows)} rows collected", flush=True)
            break
        data = resp.json()
        for r in data["results"]:
            rows.append({
                "reporting_date": r.get("reporting_date"),
                "projection_start": r.get("projection_start"),
                "projection_end": r.get("projection_end"),
                "classification_scale": r.get("classification_scale"),
                "value": r.get("value"),
                "description": r.get("description"),
            })
        next_url = data.get("next")
        if not next_url:
            break
        url, params = next_url, None  # `next` already carries the full query string
        print(f"  page {page+1}: {len(rows)} rows so far", flush=True)
        time.sleep(1.5)
    return rows


def main():
    rows = fetch_national_ipc()
    os.makedirs("data/raw", exist_ok=True)
    out_path = "data/raw/fews_net_ipc_national.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["reporting_date", "projection_start", "projection_end",
                                            "classification_scale", "value", "description"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {out_path}")


if __name__ == "__main__":
    main()
