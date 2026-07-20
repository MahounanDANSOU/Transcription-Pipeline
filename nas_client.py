"""
nas_client.py — Client Python pour l'API de stockage NAS (v3)
=============================================================

Module d'intégration pour le pipeline de transcription.
Permet de : s'authentifier, lister les vidéos (privées incluses via
/documents/library), obtenir un flux de lecture, télécharger une vidéo
(par objet ou par chemin NAS), ou extraire directement la piste audio
pour la transcription (sans télécharger la vidéo complète).

Voir docs/integrating-a-new-source.md pour le guide complet.
Prérequis : pip install requests  |  ffmpeg installé sur la machine.

Nouveautés v3 :
- list_videos() utilise /documents/library (contenu privé inclus,
  selon les droits du compte connecté)
- download_from_nas_path() : téléchargement direct par chemin NAS
- extract_audio_from_nas_path() : extraction audio par chemin NAS
"""

import base64
import json
import os
import subprocess
import time
from urllib.parse import quote

import requests


class NasError(Exception):
    """Erreur générique du client NAS."""


class NasClient:
    def __init__(self, base_url: str, timeout: int = 30, use_minio: bool = False):
        """
        base_url  : URL de l'API (ex: https://your-nas-api.example.com)
        timeout   : timeout (s) des requêtes de contrôle
        use_minio : True SEULEMENT si cette machine est dans le réseau
                    Tailscale du projet (les URLs MinIO utilisent une IP
                    privée 100.x.x.x). Par défaut False : tout passe par
                    l'API publique, fonctionne partout.
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.use_minio = use_minio
        self.token: str | None = None
        self.username: str | None = None
        # Horodatage (epoch) d'émission/expiration du token JWT courant,
        # extraits de son payload (claims "iat"/"exp"). Permettent d'anticiper
        # l'expiration de la session NAS côté serveur (cf. session_pct_elapsed).
        self.token_issued_at: float | None = None
        self.token_expires_at: float | None = None

    # ------------------------------------------------------------------
    # 1. AUTHENTIFICATION
    # ------------------------------------------------------------------

    def login(self, username: str, password: str, otp_code: str | None = None,
              max_attempts: int = 3) -> None:
        """Se connecte à l'API (et au NAS côté serveur). À appeler en premier.

        IMPORTANT : les identifiants sont ceux du NAS Synology
        (le login de l'API valide aussi la session NAS).

        Réessaie sur les erreurs 5xx et pannes réseau (le backend renvoie
        parfois un 500/502 transitoire pendant qu'il redémarre ou que sa
        propre session NAS se rétablit). Les 4xx (mauvais identifiants)
        échouent immédiatement."""
        url = f"{self.base_url}/auth/login"
        if otp_code:
            url += f"?otp_code={otp_code}"

        last_error: NasError | None = None
        response = None
        for attempt in range(1, max_attempts + 1):
            try:
                response = requests.post(
                    url,
                    data={"username": username, "password": password},
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                    timeout=60,  # la connexion NAS peut prendre quelques secondes
                )
            except requests.exceptions.RequestException as exc:
                last_error = NasError(f"Login échoué (réseau) : {exc}")
                response = None
            else:
                if response.ok:
                    break
                last_error = NasError(f"Login échoué ({response.status_code}) : {response.text[:300]}")
                if response.status_code < 500:
                    raise last_error
            if attempt < max_attempts:
                time.sleep(2 * attempt)
        else:
            raise last_error  # toutes les tentatives épuisées

        data = response.json()
        self.token = data["access_token"]
        self.username = data["username"]

        claims = self._decode_jwt_payload(self.token)
        self.token_expires_at = claims.get("exp") if claims else None
        # "iat" n'est pas toujours présent dans le payload ; à défaut on
        # prend l'instant de connexion, ce qui reste une base fiable pour
        # calculer la fraction de durée de vie écoulée.
        self.token_issued_at = (claims or {}).get("iat") or time.time()

    @staticmethod
    def _decode_jwt_payload(token: str) -> dict | None:
        """Décode (sans vérifier la signature) le payload d'un JWT pour en
        lire les claims "iat"/"exp". Utilisé uniquement pour estimer la durée
        de vie de la session, jamais pour une décision de sécurité."""
        try:
            payload_b64 = token.split(".")[1]
            padding = "=" * (-len(payload_b64) % 4)
            payload_bytes = base64.urlsafe_b64decode(payload_b64 + padding)
            return json.loads(payload_bytes)
        except Exception:
            return None

    def session_pct_elapsed(self) -> float:
        """Fraction (0.0 à 1.0+) de la durée de vie du token actuel déjà
        écoulée. Retourne 0.0 si on ne peut pas la déterminer (pas encore
        connecté, ou token sans claim "exp" décodable) : dans ce cas, tout
        code appelant doit se comporter comme si aucun renouvellement
        anticipé n'était nécessaire."""
        if not self.token_issued_at or not self.token_expires_at:
            return 0.0
        lifetime = self.token_expires_at - self.token_issued_at
        if lifetime <= 0:
            return 1.0
        elapsed = time.time() - self.token_issued_at
        return max(0.0, elapsed / lifetime)

    def _headers(self) -> dict:
        if not self.token:
            raise NasError("Non authentifié : appeler login() d'abord.")
        return {"Authorization": f"Bearer {self.token}"}

    # ------------------------------------------------------------------
    # 2. LISTING DES VIDÉOS
    # ------------------------------------------------------------------

    def list_videos(self) -> list[dict]:
        """Retourne la bibliothèque complète des vidéos indexées
        (GET /documents/library), CONTENU PRIVÉ INCLUS selon les droits
        du compte connecté (un compte admin voit tout).

        Chaque élément contient : id, title, nas_path, privacy,
        is_cached, duration, mime_type, filesize, etc.
        """
        response = requests.get(
            f"{self.base_url}/documents/library",
            headers=self._headers(),
            timeout=self.timeout,
        )
        if not response.ok:
            raise NasError(f"Listing échoué ({response.status_code}) : {response.text}")
        return response.json().get("results", [])

    def list_public_videos(self) -> list[dict]:
        """Variante : uniquement le fil public (GET /documents/)."""
        response = requests.get(
            f"{self.base_url}/documents/",
            headers=self._headers(),
            timeout=self.timeout,
        )
        if not response.ok:
            raise NasError(f"Listing échoué ({response.status_code}) : {response.text}")
        return response.json().get("results", [])

    # ------------------------------------------------------------------
    # 3. OBTENTION D'UN FLUX DE LECTURE
    # ------------------------------------------------------------------

    def get_stream_url(self, video: dict) -> str:
        """Retourne la meilleure URL de lecture pour une vidéo (dict issu
        de list_videos()).

        - use_minio=True et vidéo cachée : URL signée MinIO (rapide,
          mais nécessite le réseau Tailscale).
        - Sinon : URL du proxy de streaming de l'API publique
          (fonctionne partout).

        L'URL retournée est lisible par ffmpeg, requests, VLC, etc.
        """
        if self.use_minio and video.get("is_cached"):
            response = requests.get(
                f"{self.base_url}/media/{video['id']}/signed-url",
                headers=self._headers(),
                timeout=self.timeout,
            )
            if response.ok:
                return response.json()["url"]
            # En cas d'échec, on retombe sur le proxy API ci-dessous.

        return (
            f"{self.base_url}/documents/{video['id']}/stream"
            f"?token={self.token}"
        )

    # ------------------------------------------------------------------
    # 4. TÉLÉCHARGEMENT COMPLET
    # ------------------------------------------------------------------

    def download_video(self, video: dict, dest_folder: str = ".") -> str:
        """Télécharge la vidéo complète en local (à partir du dict vidéo).
        Retourne le chemin du fichier.

        ⚠️ Pour de la transcription pure, préférer extract_audio() :
        beaucoup plus rapide (on ne récupère que la piste audio).
        """
        os.makedirs(dest_folder, exist_ok=True)
        url = self.get_stream_url(video)

        filename = video.get("filename") or os.path.basename(video["nas_path"])
        dest_path = os.path.join(dest_folder, filename)

        with requests.get(url, stream=True, timeout=(30, 600)) as response:
            if not response.ok:
                raise NasError(f"Téléchargement échoué : HTTP {response.status_code}")

            # Même garde-fou que download_from_nas_path : ne jamais écrire une
            # réponse d'erreur JSON/HTML (session expirée, page 502 de proxy
            # relayée en 200) comme si c'était la vidéo. Un vrai fichier vidéo
            # ne commence jamais par '{', '[' ou '<'.
            content_type = (response.headers.get("Content-Type") or "").lower()
            chunks = response.iter_content(chunk_size=8 * 1024 * 1024)
            first = next(chunks, b"")
            if (
                "application/json" in content_type
                or "text/html" in content_type
                or first[:1] in (b"{", b"[", b"<")
            ):
                raise NasError(
                    "SESSION_NAS: le serveur a renvoyé une erreur JSON/HTML au lieu "
                    "du fichier (session NAS probablement expirée ou backend en panne) : "
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
                    raise NasError(
                        f"Téléchargement incomplet : {actual} octets reçus "
                        f"sur {expected} attendus pour {filename}."
                    )

        return dest_path

    def download_from_nas_path(self, nas_path: str, dest_folder: str = ".") -> str:
        """Télécharge un fichier du NAS par son chemin (sans passer par
        l'id du média). Utilise la route GET /nas/download de l'API.

        Retourne le chemin local du fichier téléchargé.
        """
        os.makedirs(dest_folder, exist_ok=True)

        if not self.token:
            raise NasError("Non authentifié : appeler login() d'abord.")

        url = (
            f"{self.base_url}/nas/download"
            f"?path={quote(nas_path)}&token={self.token}"
        )

        filename = os.path.basename(nas_path)
        dest_path = os.path.join(dest_folder, filename)

        with requests.get(url, stream=True, timeout=(30, 600)) as response:
            if not response.ok:
                raise NasError(
                    f"Téléchargement NAS échoué ({response.status_code}) : "
                    f"{response.text[:200]}"
                )

            # Le proxy /nas/download peut relayer une erreur du Synology avec
            # un statut 200 (ex: {"success":false,"error":{"code":119}} quand
            # la session NAS côté serveur a expiré), ou une page HTML d'erreur
            # de proxy. On inspecte donc le début du flux avant d'écrire quoi
            # que ce soit sur disque — un vrai fichier vidéo ne commence
            # jamais par '{', '[' ou '<'.
            content_type = (response.headers.get("Content-Type") or "").lower()
            chunks = response.iter_content(chunk_size=8 * 1024 * 1024)
            first = next(chunks, b"")
            if (
                "application/json" in content_type
                or "text/html" in content_type
                or first[:1] in (b"{", b"[", b"<")
            ):
                raise NasError(
                    "SESSION_NAS: le serveur a renvoyé une erreur JSON/HTML au lieu "
                    "du fichier (session NAS probablement expirée ou backend en panne) : "
                    f"{first[:200].decode('utf-8', 'replace')}"
                )

            with open(dest_path, "wb") as f:
                if first:
                    f.write(first)
                for chunk in chunks:
                    if chunk:
                        f.write(chunk)

            # Téléchargement tronqué (coupure réseau, session fermée en cours
            # de route…) : mieux vaut échouer ici que d'envoyer un fichier
            # incomplet à la transcription.
            expected = response.headers.get("Content-Length")
            if expected is not None:
                actual = os.path.getsize(dest_path)
                if actual != int(expected):
                    raise NasError(
                        f"Téléchargement NAS incomplet : {actual} octets reçus "
                        f"sur {expected} attendus pour {filename}."
                    )

        return dest_path

    # ------------------------------------------------------------------
    # 4bis. EXPLORATION BRUTE DU NAS (arborescence façon explorateur de fichiers)
    # ------------------------------------------------------------------

    def browse_nas(self, path: str = "", max_attempts: int = 5) -> dict:
        """Explore l'arborescence brute du NAS (dossiers/fichiers), indépendamment
        du catalogue indexé (utile pour les fichiers pas encore synchronisés).
        Retourne {"success", "path", "parent", "items": [{"name","path","isdir","size","mtime","is_video"}, ...]}.

        Réessaie avec backoff progressif sur les erreurs 5xx et les pannes
        réseau : le backend de l'API renvoie par vagues des 502 Bad Gateway
        (nginx debout, backend en train de redémarrer). Sans ces retries,
        chaque dossier touché pendant la vague est silencieusement exclu du
        lot — on a vu ~1150 vidéos disparaître d'une exploration ainsi.
        Les erreurs 4xx (chemin inexistant, droits) ne sont PAS retentées."""
        last_error: NasError | None = None
        for attempt in range(1, max_attempts + 1):
            try:
                response = requests.get(
                    f"{self.base_url}/nas/browse",
                    params={"path": path},
                    headers=self._headers(),
                    timeout=self.timeout,
                )
            except requests.exceptions.RequestException as exc:
                last_error = NasError(f"Exploration NAS échouée ({path}) : {exc}")
            else:
                if response.ok:
                    return response.json()
                last_error = NasError(
                    f"Exploration NAS échouée ({response.status_code}) : {response.text[:300]}"
                )
                if response.status_code < 500:
                    raise last_error
            if attempt < max_attempts:
                time.sleep(min(2 ** attempt, 30))
        raise last_error

    # ------------------------------------------------------------------
    # 4ter. SYNCHRONISATION DU CATALOGUE INDEXÉ (public.medias)
    # ------------------------------------------------------------------

    def sync_all_nas(self, max_depth: int = 20, privacy: str = "private") -> dict:
        """Déclenche une synchronisation complète du catalogue indexé
        (/documents/library, utilisé par list_videos()) avec le contenu réel
        du NAS. Asynchrone côté serveur : cet appel ne fait que LANCER la
        tâche — suivre sync_status() pour connaître sa progression.

        Sans ce catalogue à jour, list_videos() peut largement sous-compter
        le contenu réel du NAS (constaté : 3929 vs 22554 après une synchro,
        aucun mécanisme de resynchronisation automatique n'existant côté
        serveur)."""
        response = requests.post(
            f"{self.base_url}/nas/sync-all",
            headers=self._headers(),
            params={"max_depth": max_depth, "privacy": privacy},
            timeout=60,
        )
        if not response.ok:
            raise NasError(f"Déclenchement de la synchronisation NAS échoué ({response.status_code}) : {response.text[:300]}")
        return response.json()

    def sync_status(self) -> dict:
        """État de la synchronisation en cours (ou de la dernière terminée) :
        {"status", "progress", "message", "videos_found", "created", "updated", "errors"}."""
        response = requests.get(
            f"{self.base_url}/nas/sync-status",
            headers=self._headers(),
            timeout=30,
        )
        if not response.ok:
            raise NasError(f"Lecture du statut de synchronisation échouée ({response.status_code}) : {response.text[:300]}")
        return response.json()

    # ------------------------------------------------------------------
    # 5. EXTRACTION AUDIO DIRECTE (recommandé pour la transcription)
    # ------------------------------------------------------------------

    def extract_audio(
        self,
        video: dict,
        dest_folder: str = ".",
        sample_rate: int = 16000,
    ) -> str:
        """Extrait UNIQUEMENT la piste audio de la vidéo (dict issu de
        list_videos()), sans télécharger la vidéo complète. FFmpeg lit
        le flux HTTP et n'écrit que l'audio.

        Sortie : un fichier .wav mono 16 kHz (format standard attendu
        par Whisper et la plupart des moteurs de transcription).

        Retourne le chemin du fichier audio.
        """
        os.makedirs(dest_folder, exist_ok=True)
        url = self.get_stream_url(video)

        base = os.path.splitext(
            video.get("filename") or os.path.basename(video["nas_path"])
        )[0]
        audio_path = os.path.join(dest_folder, f"{base}.wav")

        return self._run_ffmpeg_audio(url, audio_path, sample_rate, base)

    def extract_audio_from_nas_path(
        self,
        nas_path: str,
        dest_folder: str = ".",
        sample_rate: int = 16000,
    ) -> str:
        """Extrait l'audio d'un fichier NAS par son chemin, sans
        télécharger la vidéo complète. Utilise la route GET /nas/download
        de l'API (streaming) comme source FFmpeg.

        Retourne le chemin du fichier .wav local.
        """
        os.makedirs(dest_folder, exist_ok=True)

        if not self.token:
            raise NasError("Non authentifié : appeler login() d'abord.")

        url = (
            f"{self.base_url}/nas/download"
            f"?path={quote(nas_path)}&token={self.token}"
        )

        base = os.path.splitext(os.path.basename(nas_path))[0]
        audio_path = os.path.join(dest_folder, f"{base}.wav")

        return self._run_ffmpeg_audio(url, audio_path, sample_rate, base)

    @staticmethod
    def _run_ffmpeg_audio(source_url: str, audio_path: str, sample_rate: int, label: str) -> str:
        """Lance FFmpeg pour extraire l'audio d'une source HTTP en .wav mono."""
        cmd = [
            "ffmpeg",
            "-y",
            "-i", source_url,     # ffmpeg lit directement le flux HTTP
            "-vn",                # pas de vidéo
            "-ac", "1",           # mono
            "-ar", str(sample_rate),
            "-f", "wav",
            audio_path,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise NasError(
                f"Extraction audio échouée pour {label} : "
                f"{result.stderr[-500:]}"
            )

        return audio_path

    # ------------------------------------------------------------------
    # 6. IMPORT MINIO (optionnel : pré-chauffer une vidéo en zone rapide)
    # ------------------------------------------------------------------

    def import_to_minio(self, nas_path: str, wait: bool = False) -> dict:
        """Lance l'import NAS -> MinIO en tâche de fond côté API.

        Si wait=True, bloque jusqu'à la fin de l'import (polling).
        Utile si le fichier doit ensuite être lu plusieurs fois.
        """
        response = requests.post(
            f"{self.base_url}/media/import-from-nas-start",
            headers={**self._headers(), "Content-Type": "application/json"},
            json={"nas_path": nas_path},
            timeout=self.timeout,
        )
        if not response.ok:
            raise NasError(f"Import échoué : {response.text}")

        data = response.json()

        if wait:
            import_id = data["import_id"]
            while True:
                status = requests.get(
                    f"{self.base_url}/media/import-status/{import_id}",
                    timeout=self.timeout,
                ).json()

                if status["status"] == "done":
                    break
                if status["status"] == "error":
                    raise NasError(f"Import MinIO échoué : {status['message']}")

                time.sleep(3)

        return data


# ----------------------------------------------------------------------
# EXEMPLE D'INTÉGRATION DANS UN PIPELINE DE TRANSCRIPTION
# ----------------------------------------------------------------------

if __name__ == "__main__":
    # Configuration (à passer en variables d'environnement en production)
    API_URL = os.getenv("NAS_API_URL", "https://your-nas-api.example.com")
    USERNAME = os.getenv("NAS_USERNAME", "")
    PASSWORD = os.getenv("NAS_PASSWORD", "")

    client = NasClient(API_URL)  # use_minio=False par défaut : fonctionne partout
    client.login(USERNAME, PASSWORD)
    print(f"✅ Connecté : {client.username}")

    videos = client.list_videos()   # bibliothèque complète (privé inclus)
    print(f"📼 {len(videos)} vidéo(s) disponibles")

    for video in videos:
        print(f"\n--- {video['title']} (id={video['id']}) ---")

        # OPTION A (recommandée) : extraction audio directe
        audio_path = client.extract_audio(video, dest_folder="audio_tmp")
        print(f"🎧 Audio extrait : {audio_path}")

        # OPTION B : téléchargement complet par chemin NAS
        # video_path = client.download_from_nas_path(video["nas_path"], "videos_tmp")

        # >>> ICI : brancher le moteur de transcription <<<
        # transcript = transcrire(audio_path)          # Whisper, AssemblyAI...
        # sauvegarder_captions(video["id"], transcript)

        # Nettoyage de l'audio temporaire
        # os.remove(audio_path)
