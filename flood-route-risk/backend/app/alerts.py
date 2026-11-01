"""Flood alert monitor: sends a push notification when the zone rain trigger turns on.

Every CHECK_INTERVAL_S seconds it reads the zone trigger (24 h rain against the threshold).
When the trigger goes from off to on, it stores an AlertEvent and pushes a message to every
registered device through Expo's push service. It pushes once per rain episode, not on every check.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

import httpx

from . import accounts

log = logging.getLogger("flood.alerts")

EXPO_PUSH_URL = "https://exp.host/--/api/v2/push/send"
CHECK_INTERVAL_S = 300
BATCH = 100

TITLE = "FloodSafe: alerte inondation / flood alert"
BODY = ("Forte pluie dans la zone. Vérifiez la carte pour les routes inondables avant de partir. / "
        "Heavy rain in the zone. Check the map for flood-prone roads before you leave.")


def send_push(tokens: list[str]) -> int:
    """Send the alert to each Expo push token. Returns how many were accepted by Expo."""
    accepted = 0
    for i in range(0, len(tokens), BATCH):
        messages = [{"to": t, "title": TITLE, "body": BODY, "sound": "default",
                     "priority": "high", "channelId": "alerts", "data": {"screen": "map"}}
                    for t in tokens[i:i + BATCH]]
        try:
            r = httpx.post(EXPO_PUSH_URL, json=messages, timeout=15,
                           headers={"Accept": "application/json", "Content-Type": "application/json"})
            tickets = r.json().get("data", []) if r.status_code == 200 else []
            accepted += sum(1 for t in tickets if t.get("status") == "ok")
            if r.status_code != 200:
                log.warning("expo push returned HTTP %s", r.status_code)
        except (httpx.HTTPError, ValueError) as exc:
            log.warning("expo push failed: %s", exc)
    return accepted


class AlertMonitor:
    def __init__(self, get_zone, get_weather, interval_s: int = CHECK_INTERVAL_S):
        self.get_zone = get_zone          # callable -> ZoneModel | None
        self.get_weather = get_weather    # callable -> NationalWeatherService | None
        self.interval_s = interval_s
        self.was_on = False
        self.last: dict | None = None
        self.last_checked: str | None = None

    def check(self) -> dict | None:
        zone = self.get_zone()
        weather = self.get_weather()
        if zone is None or weather is None:
            return None
        now = datetime.now(timezone.utc)
        trig = zone.trigger(weather.store, now)
        on = trig["status"] == "on"
        self.last = trig
        self.last_checked = now.isoformat()
        if on and not self.was_on:
            self._on_trigger(trig)
        self.was_on = on
        return trig

    def _on_trigger(self, trig: dict) -> None:
        tokens = accounts.push_tokens()
        sent = send_push(tokens) if tokens else 0
        with accounts._sf()() as db:
            db.add(accounts.AlertEvent(max_rain_24h_mm=float(trig["max_rain_24h_mm"] or 0),
                                       threshold_mm=float(trig["threshold_mm"]), pushes_sent=sent))
            db.commit()
        log.info("zone trigger ON: %s mm, pushed to %d of %d devices",
                 trig["max_rain_24h_mm"], sent, len(tokens))

    async def run(self) -> None:
        while True:
            try:
                await asyncio.to_thread(self.check)
            except Exception:  # keep the monitor alive whatever goes wrong
                log.exception("alert check failed")
            await asyncio.sleep(self.interval_s)
