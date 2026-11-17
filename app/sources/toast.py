"""Covers history from the Toast API.

Aggregates Toast orders into the same daily covers series the CSV exports
provide, so the forecasting model is unchanged by the switch.

Endpoints used (both read-only, which is all Standard API Access grants):
  POST /authentication/v1/authentication/login  -- exchange client credentials
                                                   for a bearer token
  GET  /orders/v2/ordersBulk                    -- orders for one business date

Design notes
------------
* We page by `businessDate` (yyyymmdd) rather than a startDate/endDate range.
  Toast's business date already accounts for the restaurant's own day-rollover
  cutoff, so one request maps exactly to one row of covers with no timezone
  arithmetic on our side.
* Covers come from each order's `numberOfGuests`, falling back to
  `calculatedGuestCount` when the POS did not record a party size. Voided and
  deleted orders are excluded.
* A date with zero non-voided orders is reported as 0 covers, which the model
  reads as "closed". A date we could not fetch is omitted entirely, which the
  model reads as "unknown" -- the distinction matters, so failures are never
  silently turned into zeros.
"""
from __future__ import annotations

import datetime as dt
import logging
import time
from typing import Dict, List, Optional

import pandas as pd
import requests

from app.config import Settings
from app.sources.base import SourceError, SourceNotConfigured, validate_frame

logger = logging.getLogger(__name__)

LOGIN_PATH = "/authentication/v1/authentication/login"
ORDERS_BULK_PATH = "/orders/v2/ordersBulk"

# Toast caps ordersBulk at 100 objects per page.
MAX_PAGE_SIZE = 100
# Refresh slightly before expiry so a long ingest never uses a dead token.
TOKEN_EXPIRY_MARGIN_SECONDS = 60
# Standard API Access is rate limited, so retry politely and give up quickly.
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 2
DEFAULT_TOKEN_LIFETIME_SECONDS = 3600.0


class ToastAuthError(SourceError):
    """Authentication with Toast failed."""


def _normalize_hostname(hostname: str) -> str:
    host = hostname.strip()
    for prefix in ("https://", "http://"):
        if host.startswith(prefix):
            host = host[len(prefix):]
    return host.rstrip("/")


class ToastClient:
    """Thin authenticated client for the Toast REST API."""

    def __init__(
        self,
        hostname: str,
        client_id: str,
        client_secret: str,
        restaurant_guid: str,
        user_access_type: str = "TOAST_MACHINE_CLIENT",
        timeout: int = 30,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.base_url = f"https://{_normalize_hostname(hostname)}"
        self.client_id = client_id
        self.client_secret = client_secret
        self.restaurant_guid = restaurant_guid
        self.user_access_type = user_access_type
        self.timeout = timeout
        self.session = session or requests.Session()
        self._token: Optional[str] = None
        self._token_expires_at: float = 0.0

    # --- auth --------------------------------------------------------------

    def _token_is_valid(self) -> bool:
        return bool(self._token) and time.time() < self._token_expires_at

    def authenticate(self) -> str:
        """Fetch a bearer token, caching it until shortly before it expires."""
        if self._token_is_valid():
            return self._token

        try:
            resp = self.session.post(
                f"{self.base_url}{LOGIN_PATH}",
                json={
                    "clientId": self.client_id,
                    "clientSecret": self.client_secret,
                    "userAccessType": self.user_access_type,
                },
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise ToastAuthError(f"Could not reach Toast at {self.base_url}: {exc}") from exc

        if resp.status_code in (401, 403):
            raise ToastAuthError(
                f"Toast rejected the credentials (HTTP {resp.status_code}). Check "
                "COVERS_TOAST_CLIENT_ID / COVERS_TOAST_CLIENT_SECRET and that the "
                "credential set is enabled for this restaurant."
            )
        if not resp.ok:
            raise ToastAuthError(f"Toast login failed: HTTP {resp.status_code} {resp.text[:200]}")

        try:
            body = resp.json()
        except ValueError as exc:
            raise ToastAuthError("Toast login returned a non-JSON response.") from exc

        # Toast nests the token under `token` in some responses and returns it
        # flat in others; accept either rather than breaking on the shape.
        payload = body.get("token") or body
        token = payload.get("accessToken")
        if not token:
            raise ToastAuthError("Toast login response contained no accessToken.")

        try:
            lifetime = float(payload.get("expiresIn"))
        except (TypeError, ValueError):
            lifetime = DEFAULT_TOKEN_LIFETIME_SECONDS

        self._token = token
        self._token_expires_at = time.time() + max(0.0, lifetime - TOKEN_EXPIRY_MARGIN_SECONDS)
        return token

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.authenticate()}",
            "Toast-Restaurant-External-ID": self.restaurant_guid,
            "Accept": "application/json",
        }

    # --- requests ----------------------------------------------------------

    def _get(self, path: str, params: Dict) -> requests.Response:
        """GET with one re-auth on 401 and bounded backoff on 429 / 5xx."""
        url = f"{self.base_url}{path}"
        resp = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = self.session.get(
                    url, params=params, headers=self._headers(), timeout=self.timeout
                )
            except requests.RequestException as exc:
                if attempt == MAX_RETRIES:
                    raise SourceError(f"Toast request to {path} failed: {exc}") from exc
                time.sleep(RETRY_BACKOFF_SECONDS * attempt)
                continue

            if resp.status_code == 401 and attempt < MAX_RETRIES:
                # The token may have been revoked early; force a fresh login.
                self._token = None
                continue
            if resp.status_code in (429, 500, 502, 503, 504) and attempt < MAX_RETRIES:
                retry_after = resp.headers.get("Retry-After")
                try:
                    delay = float(retry_after) if retry_after else RETRY_BACKOFF_SECONDS * attempt
                except ValueError:
                    delay = RETRY_BACKOFF_SECONDS * attempt
                logger.warning(
                    "Toast %s returned %s; retrying in %.0fs", path, resp.status_code, delay
                )
                time.sleep(delay)
                continue
            return resp

        if resp is None:  # pragma: no cover - defensive
            raise SourceError(f"Toast request to {path} produced no response.")
        return resp

    def fetch_orders_for_business_date(
        self, day: dt.date, page_size: int = MAX_PAGE_SIZE
    ) -> List[Dict]:
        """All orders for one business date, following pagination to the end."""
        page_size = min(page_size, MAX_PAGE_SIZE)
        business_date = day.strftime("%Y%m%d")
        orders: List[Dict] = []
        page = 1

        while True:
            resp = self._get(
                ORDERS_BULK_PATH,
                {"businessDate": business_date, "page": page, "pageSize": page_size},
            )
            if resp.status_code == 404:
                # No orders recorded for that business date.
                return orders
            if not resp.ok:
                raise SourceError(
                    f"Toast ordersBulk failed for {business_date}: "
                    f"HTTP {resp.status_code} {resp.text[:200]}"
                )
            try:
                batch = resp.json()
            except ValueError as exc:
                raise SourceError(
                    f"Toast ordersBulk returned non-JSON for {business_date}."
                ) from exc

            if not isinstance(batch, list):
                raise SourceError(
                    f"Expected a JSON array of orders for {business_date}, "
                    f"got {type(batch).__name__}."
                )

            orders.extend(batch)
            # A short page is the last page.
            if len(batch) < page_size:
                return orders
            page += 1

    def fetch_restaurant(self) -> Dict:
        """Restaurant metadata -- used by /health to prove the credentials work."""
        resp = self._get(f"/restaurants/v1/restaurants/{self.restaurant_guid}", {})
        if not resp.ok:
            raise SourceError(
                f"Toast restaurant lookup failed: HTTP {resp.status_code} {resp.text[:200]}"
            )
        return resp.json()


def guests_in_order(order: Dict) -> float:
    """Guest count for a single order, or 0 if it should not be counted."""
    if order.get("voided") or order.get("deleted"):
        return 0.0

    guests = order.get("numberOfGuests")
    if isinstance(guests, (int, float)) and not isinstance(guests, bool) and guests > 0:
        return float(guests)

    # The POS did not record a party size -- fall back to Toast's own estimate
    # from the items ordered rather than dropping the order from covers.
    calculated = order.get("calculatedGuestCount")
    if isinstance(calculated, (int, float)) and not isinstance(calculated, bool) and calculated > 0:
        return float(calculated)
    return 0.0


class ToastCoversSource:
    """Daily covers aggregated from Toast orders."""

    name = "toast"

    def __init__(self, settings: Settings, client: Optional[ToastClient] = None) -> None:
        self.settings = settings
        if client is not None:
            self.client = client
            return

        missing = [
            env
            for env, value in (
                ("COVERS_TOAST_HOSTNAME", settings.toast_hostname),
                ("COVERS_TOAST_CLIENT_ID", settings.toast_client_id),
                ("COVERS_TOAST_CLIENT_SECRET", settings.toast_client_secret),
                ("COVERS_TOAST_RESTAURANT_GUID", settings.toast_restaurant_guid),
            )
            if not value
        ]
        if missing:
            raise SourceNotConfigured(
                "Toast source selected but these settings are missing: "
                + ", ".join(missing)
                + ". See .env.example."
            )

        self.client = ToastClient(
            hostname=settings.toast_hostname,
            client_id=settings.toast_client_id,
            client_secret=settings.toast_client_secret,
            restaurant_guid=settings.toast_restaurant_guid,
            user_access_type=settings.toast_user_access_type,
            timeout=settings.toast_timeout_seconds,
        )

    def fetch_daily_covers(
        self,
        start: Optional[dt.date] = None,
        end: Optional[dt.date] = None,
    ) -> pd.DataFrame:
        if start is None or end is None:
            raise SourceError(
                "The Toast source needs an explicit start and end date -- it fetches "
                "one business date per request, so an unbounded range is not safe."
            )
        if end < start:
            raise SourceError(f"end ({end}) is before start ({start}).")

        rows: List[Dict] = []
        failures: List[str] = []
        day = start
        while day <= end:
            try:
                orders = self.client.fetch_orders_for_business_date(
                    day, page_size=self.settings.toast_page_size
                )
            except SourceError as exc:
                # Record the gap and keep going; one bad day should not void a
                # multi-month backfill. Omitted dates read as "unknown", not 0.
                logger.warning("Skipping %s: %s", day, exc)
                failures.append(day.isoformat())
                day += dt.timedelta(days=1)
                continue

            live = [o for o in orders if not (o.get("voided") or o.get("deleted"))]
            rows.append(
                {
                    "date": pd.Timestamp(day),
                    "covers": float(sum(guests_in_order(o) for o in live)),
                    "order_count": len(live),
                }
            )
            day += dt.timedelta(days=1)

        if failures:
            logger.warning(
                "%d business date(s) could not be fetched from Toast: %s",
                len(failures),
                ", ".join(failures[:10]) + ("..." if len(failures) > 10 else ""),
            )
        if not rows:
            raise SourceError(
                f"Toast returned no usable business dates between {start} and {end}."
            )

        return validate_frame(pd.DataFrame(rows), self.name)
