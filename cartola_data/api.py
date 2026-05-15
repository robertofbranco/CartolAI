import json
import logging
import time
from pathlib import Path

import requests

from .config import BASE_URL, DATA_DIR, GATOMESTRE_BASE
from .transforms import safe_filename

log = logging.getLogger(__name__)


class JsonAPIClient:
    def __init__(
        self,
        base_url: str,
        token: str | None = None,
        cache_dir: Path = DATA_DIR / "cache",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
        if token:
            self.session.headers["Authorization"] = token

    def get_json(self, path: str, use_cache: bool = True) -> dict:
        cache_file = self.cache_dir / safe_filename(f"{path}.json")
        if use_cache and cache_file.exists():
            return json.loads(cache_file.read_text(encoding="utf-8"))

        url = f"{self.base_url}{path}"
        for attempt in range(3):
            try:
                response = self.session.get(url, timeout=15)
                response.raise_for_status()
                data = response.json()
                if use_cache:
                    cache_file.write_text(
                        json.dumps(data, ensure_ascii=False),
                        encoding="utf-8",
                    )
                return data
            except requests.RequestException as exc:
                log.warning("Attempt %s failed for %s: %s", attempt + 1, path, exc)
                time.sleep(2**attempt)

        raise RuntimeError(f"Failed to fetch {path} after 3 attempts")


class CartolaAPI(JsonAPIClient):
    """Small wrapper around the Cartola FC API."""

    def __init__(self, token: str | None = None, cache_dir: Path = DATA_DIR / "cache"):
        super().__init__(BASE_URL, token=token, cache_dir=cache_dir)

    def _get(self, path: str, use_cache: bool = True) -> dict:
        return self.get_json(path, use_cache=use_cache)

    def market_status(self) -> dict:
        return self._get("/mercado/status", use_cache=False)

    def mercado(self) -> dict:
        return self._get("/atletas/mercado", use_cache=False)

    def atletas_mercado(self) -> dict:
        return self.mercado()

    def atletas_pontuados(self, rodada: int | None = None) -> dict:
        path = f"/atletas/pontuados/{rodada}" if rodada else "/atletas/pontuados"
        return self._get(path)

    def clubes(self) -> dict:
        return self._get("/clubes")

    def rodadas(self) -> dict:
        return self._get("/rodadas")

    def partidas(self, rodada: int | None = None) -> dict:
        path = f"/partidas/{rodada}" if rodada else "/partidas"
        return self._get(path, use_cache=rodada is not None)

    def pos_rodada(self, rodada: int) -> dict:
        return self._get(f"/pos-rodada/destaques/{rodada}", use_cache=bool(rodada))

    def league(self, liga: str) -> dict:
        return self._get(f"/auth/liga/{liga}?orderBy=rodada", use_cache=True)


class GatoMestreAPI:
    """Wrapper for Gato Mestre favorites/odds endpoint."""

    def __init__(
        self,
        token: str,
        temporada: int,
        cache_dir: Path = DATA_DIR / "cache",
    ) -> None:
        self.temporada = temporada
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Content-Type": "application/json",
                "Authorization": token,
            }
        )

    def favoritos(self, rodada: int) -> dict:
        index = rodada - 1
        cache_file = self.cache_dir / f"gato_favoritos_{self.temporada}_r{rodada}.json"
        if cache_file.exists():
            return json.loads(cache_file.read_text(encoding="utf-8"))

        url = (
            f"{GATOMESTRE_BASE}/api/v2/equipes/{self.temporada}"
            f"/campeonato-brasileiro/favoritos/rodadas/{index}?a=true"
        )
        for attempt in range(3):
            try:
                response = self.session.get(url, timeout=15)
                response.raise_for_status()
                data = response.json()
                cache_file.write_text(
                    json.dumps(data, ensure_ascii=False),
                    encoding="utf-8",
                )
                return data
            except requests.RequestException as exc:
                log.warning(
                    "Attempt %s failed for Gato Mestre round %s: %s",
                    attempt + 1,
                    rodada,
                    exc,
                )
                time.sleep(2**attempt)

        raise RuntimeError(f"Failed to fetch Gato Mestre favorites for round {rodada}")
