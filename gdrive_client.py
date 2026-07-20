"""
gdrive_client.py — Client Python pour parcourir/télécharger un dossier
Google Drive PUBLIC ("Tous les utilisateurs disposant du lien") via l'API
Drive v3, avec une simple clé API (pas d'OAuth).

Limite importante : ceci ne fonctionne QUE pour un dossier partagé en accès
"Tous les utilisateurs disposant du lien" (lecteur ou plus). Un dossier
privé, même partagé avec un compte précis, est invisible à la clé API seule
— il faudrait alors une authentification OAuth complète.

Voir Orchestrateur_GUI.py pour l'intégration (section "Source Google Drive").
"""

from __future__ import annotations

import os
import re
from urllib.parse import urlparse, parse_qs

import requests

FOLDER_MIME_TYPE = "application/vnd.google-apps.folder"
DRIVE_API_BASE = "https://www.googleapis.com/drive/v3"
FILE_FIELDS = "id,name,mimeType,size,modifiedTime,videoMediaMetadata(durationMillis)"


class GDriveError(Exception):
    """Erreur générique du client Google Drive."""


class GDriveClient:
    def __init__(self, api_key: str, timeout: int = 30):
        if not api_key or not api_key.strip():
            raise GDriveError("Clé API Google Drive manquante (google_drive_api_key).")
        self.api_key = api_key.strip()
        self.timeout = timeout

    # ------------------------------------------------------------------
    # 1. RÉSOLUTION D'UN LIEN DE DOSSIER EN ID
    # ------------------------------------------------------------------

    @staticmethod
    def resolve_folder_id(link_or_id: str) -> str:
        """Extrait l'ID de dossier depuis un lien Drive de n'importe quel
        format courant, ou renvoie tel quel si c'est déjà un ID brut.

        Formats supportés :
        - https://drive.google.com/drive/folders/<id>
        - https://drive.google.com/drive/u/0/folders/<id>?usp=sharing
        - https://drive.google.com/open?id=<id>
        - https://drive.google.com/drive/folders/<id>?usp=drive_link
        - un ID brut (pas de '/' ni 'http')
        """
        value = (link_or_id or "").strip()
        if not value:
            raise GDriveError("Lien ou ID de dossier Google Drive vide.")

        if "drive.google.com" not in value and "http" not in value:
            return value  # déjà un ID brut

        match = re.search(r"/folders/([a-zA-Z0-9_-]+)", value)
        if match:
            return match.group(1)

        parsed = urlparse(value)
        qs = parse_qs(parsed.query)
        if "id" in qs and qs["id"]:
            return qs["id"][0]

        raise GDriveError(
            f"Impossible d'extraire un ID de dossier depuis ce lien : {value}"
        )

    # ------------------------------------------------------------------
    # 2. EXPLORATION (même forme de retour que NasClient.browse_nas)
    # ------------------------------------------------------------------

    def browse(self, folder_id: str, max_attempts: int = 5) -> dict:
        """Liste le contenu direct d'un dossier Drive. Retourne
        {"items": [{"name","path"(=id fichier/dossier),"isdir","is_video",
        "size","mtime","duration_seconds"}, ...]}.

        Réessaie avec backoff sur les erreurs 5xx/429 (quota) et pannes
        réseau ; échoue immédiatement sur 4xx (dossier non public, ID
        invalide…) avec un message explicite pour le cas 403/404 fréquent
        d'un dossier pas assez partagé."""
        query = f"'{folder_id}' in parents and trashed = false"
        params = {
            "key": self.api_key,
            "q": query,
            "fields": f"files({FILE_FIELDS})",
            "pageSize": 1000,
            "supportsAllDrives": "true",
            "includeItemsFromAllDrives": "true",
        }

        items: list[dict] = []
        page_token = None
        while True:
            if page_token:
                params["pageToken"] = page_token
            data = self._get_with_retry(f"{DRIVE_API_BASE}/files", params, max_attempts, folder_id)
            for f in data.get("files", []):
                is_dir = f.get("mimeType") == FOLDER_MIME_TYPE
                mime = f.get("mimeType") or ""
                duration_ms = (f.get("videoMediaMetadata") or {}).get("durationMillis")
                items.append({
                    "name": f.get("name"),
                    "path": f.get("id"),
                    "isdir": is_dir,
                    "is_video": (not is_dir) and mime.startswith("video/"),
                    "size": int(f["size"]) if f.get("size") else None,
                    "mtime": f.get("modifiedTime"),
                    "duration_seconds": (int(duration_ms) / 1000.0) if duration_ms else None,
                })
            page_token = data.get("nextPageToken")
            if not page_token:
                break

        return {"items": items}

    def _get_with_retry(self, url: str, params: dict, max_attempts: int, folder_id: str) -> dict:
        last_error: GDriveError | None = None
        for attempt in range(1, max_attempts + 1):
            try:
                response = requests.get(url, params=params, timeout=self.timeout)
            except requests.exceptions.RequestException as exc:
                last_error = GDriveError(f"Exploration Drive échouée (réseau) : {exc}")
            else:
                if response.ok:
                    return response.json()
                if response.status_code in (403, 404):
                    raise GDriveError(
                        f"Dossier Drive inaccessible (HTTP {response.status_code}) — vérifie qu'il est "
                        f"bien partagé en \"Tous les utilisateurs disposant du lien\" (id={folder_id}). "
                        f"Détail : {response.text[:300]}"
                    )
                last_error = GDriveError(f"Exploration Drive échouée ({response.status_code}) : {response.text[:300]}")
                if response.status_code < 500 and response.status_code != 429:
                    raise last_error
            if attempt < max_attempts:
                import time
                time.sleep(min(2 ** attempt, 30))
        raise last_error

    # ------------------------------------------------------------------
    # 3. TÉLÉCHARGEMENT
    # ------------------------------------------------------------------

    def download_file(self, file_id: str, filename: str, dest_folder: str = ".") -> str:
        """Télécharge un fichier Drive par son ID. Retourne le chemin local.

        Mêmes garde-fous que nas_client : rejette une réponse JSON/HTML
        d'erreur écrite comme si c'était le fichier, et vérifie la taille
        reçue contre Content-Length."""
        os.makedirs(dest_folder, exist_ok=True)
        url = f"{DRIVE_API_BASE}/files/{file_id}"
        params = {"key": self.api_key, "alt": "media", "supportsAllDrives": "true"}
        dest_path = os.path.join(dest_folder, filename)

        with requests.get(url, params=params, stream=True, timeout=(30, 600)) as response:
            if not response.ok:
                raise GDriveError(
                    f"Téléchargement Drive échoué ({response.status_code}) : {response.text[:300]}"
                )

            content_type = (response.headers.get("Content-Type") or "").lower()
            chunks = response.iter_content(chunk_size=8 * 1024 * 1024)
            first = next(chunks, b"")
            if (
                "application/json" in content_type
                or "text/html" in content_type
                or first[:1] in (b"{", b"[", b"<")
            ):
                raise GDriveError(
                    "Le serveur a renvoyé une erreur JSON/HTML au lieu du fichier : "
                    f"{first[:200].decode('utf-8', 'replace')}"
                )

            with open(dest_path, "wb") as f:
                if first:
                    f.write(first)
                for chunk in chunks:
                    if chunk:
                        f.write(chunk)

            expected = response.headers.get("Content-Length")
            if expected is not None:
                actual = os.path.getsize(dest_path)
                if actual != int(expected):
                    raise GDriveError(
                        f"Téléchargement Drive incomplet : {actual} octets reçus "
                        f"sur {expected} attendus pour {filename}."
                    )

        return dest_path
