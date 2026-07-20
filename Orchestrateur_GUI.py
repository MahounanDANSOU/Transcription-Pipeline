from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path
from queue import Empty, Queue
from typing import TYPE_CHECKING
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

import psutil
import psycopg2
import mimetypes
from psycopg2 import sql
from psycopg2.extras import Json
import requests

if TYPE_CHECKING:
    # Références de type découplées du fallback runtime ci-dessous, pour que
    # Pylance ne déduise pas "NasClient | None" à chaque usage (nas_client.py
    # est toujours présent en pratique ; seule une install cassée déclenche le
    # fallback, auquel cas les fonctionnalités NAS sont désactivées à l'exécution).
    from nas_client import NasClient as NasClientCls
    from nas_client import NasError as NasErrorCls
    from gdrive_client import GDriveClient as GDriveClientCls

try:
    from nas_client import NasClient, NasError
except ImportError:
    NasClient = None
    NasError = Exception

try:
    from gdrive_client import GDriveClient, GDriveError
except ImportError:
    GDriveClient = None
    GDriveError = Exception

# Erreurs de téléchargement NAS récupérables par reconnexion + nouvelle
# tentative (session expirée, coupure réseau, timeout…).
DOWNLOAD_RETRYABLE_ERRORS = (NasError, requests.exceptions.RequestException, OSError)
# Idem pour Google Drive.
DRIVE_RETRYABLE_ERRORS = (GDriveError, requests.exceptions.RequestException, OSError)

# ─────────────────────────── CONSTANTES ────────────────────────────────────

APP_TITLE = "Transcription Pipeline — Batch GUI"
APP_VERSION = "2.1.1"

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_FW_PYTHON = str(PROJECT_ROOT / "Faster-Whisper" / "fw_env" / "Scripts" / "python.exe")
DEFAULT_WX_PYTHON = str(PROJECT_ROOT / "WhisperX" / "wx_env" / "Scripts" / "python.exe")

LOCAL_CONFIG_PATH = PROJECT_ROOT / "config.json"

def load_local_config() -> dict:
    """Charge les secrets/chemins locaux depuis config.json (JSON, jamais commité —
    voir config.example.json pour le modèle à copier/remplir)."""
    defaults = {
        # Base de données PostgreSQL (ex: un projet Supabase, ou tout Postgres accessible).
        "db_host": "",
        "db_port": "5432",
        "db_name": "postgres",
        "db_user": "",
        "db_password": "",
        "db_sslmode": "require",
        "hf_token": "",
        "fw_python": DEFAULT_FW_PYTHON,
        "wx_python": DEFAULT_WX_PYTHON,
        "local_model_dir": "",
        # URL de l'API de ton propre backend NAS (voir nas_client.py — nom
        # générique conservé pour ce qu'il fait : parler à un stockage réseau
        # via une API REST à authentification par login/mot de passe).
        "nas_api_url": "https://your-nas-api.example.com",
        "nas_username": "",
        "nas_password": "",
        "google_drive_api_key": "",
    }
    if not LOCAL_CONFIG_PATH.exists():
        return defaults
    try:
        raw = json.loads(LOCAL_CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[Config] Impossible de lire {LOCAL_CONFIG_PATH}: {exc}")
        return defaults
    for key in defaults:
        value = raw.get(key)
        if value:
            defaults[key] = str(value)
    return defaults

COMMON_MODELS = [
    "tiny", "base", "small", "medium",
    "large-v2", "large-v3", "turbo", "distil-large-v3",
]

PRIVACY_VALUES = ["PRIVATE", "PUBLIC"]

# Paramètres techniques figés dans le code : plus aucune case à cocher pour ça,
# le comportement est toujours le même (auto-détection matérielle, multi-langue
# fr/en automatique dès qu'une des deux langues apparaît, segmentation fine).
AUTO_DEVICE = "auto"
AUTO_COMPUTE_TYPE = "auto"
BEAM_SIZE = 5
MIN_SILENCE_DURATION_MS = 200
MAX_WORDS_PER_SEGMENT = 10
MAX_CHARS_PER_SEGMENT = 42
CONDITION_ON_PREVIOUS_TEXT = False
ALIGN_CACHE_ONLY = False
VERBOSE_LOGS = False
MULTILANG_CANDIDATE_LANGUAGES = ["fr", "en"]

# Délai (secondes) après le DÉBUT de la transcription locale d'un item avant
# de déclencher le pré-téléchargement en arrière-plan de l'item suivant.
# La session DSM Synology (côté serveur, gérée par l'API) expire par
# inactivité (codes DSM 106 "Session timeout" / 119 "session invalide") —
# constaté en pratique sur des écarts de quelques dizaines de minutes sans
# aucun appel /nas/*, PAS sur une durée de vie fixe du token JWT (24h, lui,
# n'expire jamais dans cette fenêtre). Un contact NAS toutes les 5 minutes
# pendant la transcription locale (qui ne touche jamais le NAS) sert de
# keepalive, largement sous n'importe quel seuil d'inactivité plausible.
NAS_PREFETCH_DELAY_SECONDS = 300

MEDIA_EXTENSIONS = {
    ".mp3", ".wav", ".m4a", ".flac", ".ogg", ".wma", ".aac",
    ".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v", ".mpg", ".mpeg",
}

# ─────────────────────────── UTILITAIRES GÉNÉRAUX ──────────────────────────

def now_dt() -> datetime:
    return datetime.now()

def now_iso() -> str:
    return now_dt().isoformat(timespec="seconds")

def runtime_dir() -> Path:
    base = Path(__file__).resolve().parent / "runtime_db_gui"
    base.mkdir(parents=True, exist_ok=True)
    return base

def is_media_file(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in MEDIA_EXTENSIONS

def collect_media_inputs(
    single_file: str, multi_files: list[str], folders: list[str], recursive: bool,
) -> list[Path]:
    """Rassemble tous les médias à traiter. `folders` peut contenir plusieurs dossiers
    (ex: les parties -1-006, -2-006... d'un export Google Drive scindé) ; chacun est
    scanné récursivement (sous-dossiers de sous-dossiers inclus) si `recursive` est vrai."""
    result: list[Path] = []
    if single_file.strip():
        p = Path(single_file.strip())
        if is_media_file(p):
            result.append(p.resolve())
    for item in multi_files:
        p = Path(item)
        if is_media_file(p):
            result.append(p.resolve())
    for folder in folders:
        if not folder.strip():
            continue
        base = Path(folder.strip())
        if base.is_dir():
            it = base.rglob("*") if recursive else base.glob("*")
            for p in it:
                if is_media_file(p):
                    result.append(p.resolve())
    seen: set[str] = set()
    deduped: list[Path] = []
    for p in result:
        k = str(p).lower()
        if k not in seen:
            seen.add(k)
            deduped.append(p)
    return deduped

# ─────────────────────────── DÉTECTION MATÉRIEL ────────────────────────────

_HW_PROBE = r"""
import json, platform, sys
d = {"ok": True, "python": sys.executable, "version": sys.version.split()[0], "platform": platform.platform()}
try:
    import torch
    d["torch"] = True
    d["torch_version"] = getattr(torch, "__version__", "")
    d["cuda"] = bool(torch.cuda.is_available())
    d["cuda_device_count"] = torch.cuda.device_count() if d["cuda"] else 0
    d["cuda_name"] = torch.cuda.get_device_name(0) if d["cuda"] else ""
    d["torch_cuda_version"] = getattr(getattr(torch, "version", None), "cuda", None)
except Exception as e:
    d["torch"] = False
    d["cuda"] = False
    d["torch_error"] = str(e)
print(json.dumps(d, ensure_ascii=False))
"""

def inspect_runtime(python_exe: Path) -> dict:
    if not python_exe.exists():
        return {"ok": False, "error": f"Introuvable: {python_exe}"}
    try:
        proc = subprocess.run(
            [str(python_exe), "-c", _HW_PROBE],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=True,
            timeout=30,
        )
        return json.loads(proc.stdout.strip() or "{}")
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "Timeout dépassé lors de la détection."}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}

def cuda_available(info: dict) -> bool:
    return bool(info.get("ok")) and bool(info.get("cuda"))

def choose_auto_device(fw_info: dict, wx_info: dict, use_alignment: bool) -> str:
    if use_alignment:
        return "cuda" if cuda_available(fw_info) and cuda_available(wx_info) else "cpu"
    return "cuda" if cuda_available(fw_info) else "cpu"

def choose_auto_compute_type(device: str) -> str:
    return "float16" if device == "cuda" else "int8"

def choose_effective_devices(
    requested_device: str,
    fw_info: dict,
    wx_info: dict,
    use_alignment: bool,
) -> tuple[str, str]:
    if requested_device == "cpu":
        return "cpu", "cpu"

    if requested_device == "cuda":
        if not cuda_available(fw_info):
            raise RuntimeError("Le runtime Faster-Whisper ne supporte pas CUDA.")
        if use_alignment and not cuda_available(wx_info):
            raise RuntimeError("Le runtime WhisperX ne supporte pas CUDA.")
        return "cuda", "cuda" if use_alignment else "cpu"

    if requested_device == "auto":
        if use_alignment:
            both_cuda = cuda_available(fw_info) and cuda_available(wx_info)
            return ("cuda", "cuda") if both_cuda else ("cpu", "cpu")
        return ("cuda", "cpu") if cuda_available(fw_info) else ("cpu", "cpu")

    return "cpu", "cpu"

# ─────────────────────────── FFPROBE ───────────────────────────────────────

def ffprobe_metadata(media_path: Path) -> dict:
    _empty = {
        "duration": None, "bitrate": None, "width": None, "height": None,
        "video_codec": None, "audio_codec": None, "format_name": None,
        "frame_rate": None, "audio_sample_rate": None, "audio_channels": None,
        "stream_count": None, "tags": None,
    }
    try:
        proc = subprocess.run(
            ["ffprobe", "-v", "error", "-print_format", "json",
             "-show_format", "-show_streams", str(media_path)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=True,
            timeout=60,
        )
        data = json.loads(proc.stdout or "{}")
    except Exception:
        return _empty

    def _f(v):
        try:
            return float(v) if v is not None else None
        except Exception:
            return None

    def _i(v):
        try:
            return int(v) if v is not None else None
        except Exception:
            return None

    def _fps(rate_str):
        try:
            num, den = str(rate_str).split("/")
            den = float(den)
            return round(float(num) / den, 3) if den else None
        except Exception:
            return None

    fmt = data.get("format", {})
    vst = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), {})
    ast = next((s for s in data.get("streams", []) if s.get("codec_type") == "audio"), {})
    return {
        "duration": _f(fmt.get("duration")),
        "bitrate": _i(fmt.get("bit_rate")),
        "width": _i(vst.get("width")),
        "height": _i(vst.get("height")),
        "video_codec": vst.get("codec_name"),
        "audio_codec": ast.get("codec_name"),
        "format_name": fmt.get("format_name"),
        "frame_rate": _fps(vst.get("avg_frame_rate")),
        "audio_sample_rate": _i(ast.get("sample_rate")),
        "audio_channels": _i(ast.get("channels")),
        "stream_count": len(data.get("streams", [])) or None,
        "tags": fmt.get("tags") or None,
    }

# ─────────────────────────── BASE DE DONNÉES ───────────────────────────────

# Paramètres de connexion DB : vides par défaut, remplis une seule fois au
# démarrage par configure_db(cfg) à partir de config.json — cf. App.__init__.
# Gardés en constantes de module (plutôt que passés en argument partout)
# pour ne pas changer la signature des dizaines d'appels à connect_db()/
# ensure_connection() existants dans le reste du pipeline.
DB_HOST = ""
DB_PORT = 5432
DB_NAME = "postgres"
DB_USER = ""
DB_SSLMODE = "require"

def configure_db(cfg: dict) -> None:
    """À appeler une fois, avant toute connexion DB (voir App.__init__)."""
    global DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_SSLMODE
    DB_HOST = cfg["db_host"]
    DB_PORT = int(cfg["db_port"] or 5432)
    DB_NAME = cfg["db_name"] or "postgres"
    DB_USER = cfg["db_user"]
    DB_SSLMODE = cfg["db_sslmode"] or "require"

def connect_db(password: str):
    if not DB_HOST or not DB_USER:
        raise RuntimeError(
            "Connexion DB non configurée : renseigne 'db_host'/'db_user' dans config.json "
            "(voir config.example.json)."
        )
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=password,
        sslmode=DB_SSLMODE,
        connect_timeout=20,
        keepalives=1,
        keepalives_idle=30,
        keepalives_interval=10,
        keepalives_count=5,
        application_name="transcription-pipeline-gui",
    )

def is_connection_usable(conn) -> bool:
    try:
        if conn is None or conn.closed:
            return False
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
            cur.fetchone()
        return True
    except Exception:
        return False

def ensure_connection(conn, password: str):
    if is_connection_usable(conn):
        return conn
    try:
        if conn is not None and not conn.closed:
            conn.close()
    except Exception:
        pass
    return connect_db(password)

def rollback_safely(conn) -> None:
    try:
        if conn is not None and not conn.closed:
            conn.rollback()
    except Exception:
        pass

def get_columns(conn, schema: str, table: str) -> set[str]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema=%s AND table_name=%s",
            (schema, table),
        )
        return {row[0] for row in cur.fetchall()}

def select_scalar(conn, query: str, params: tuple = ()):
    with conn.cursor() as cur:
        cur.execute(query, params)
        row = cur.fetchone()
        return row[0] if row else None

def cleaned_payload(payload: dict, allowed: set[str], keep_none: bool = False) -> dict:
    return {
        k: v for k, v in payload.items()
        if k in allowed and (v is not None or keep_none)
    }

def insert_row(conn, schema: str, table: str, payload: dict) -> int:
    cols = list(payload.keys())
    values = [payload[c] for c in cols]
    q = sql.SQL("INSERT INTO {}.{} ({}) VALUES ({}) RETURNING id").format(
        sql.Identifier(schema), sql.Identifier(table),
        sql.SQL(", ").join(sql.Identifier(c) for c in cols),
        sql.SQL(", ").join(sql.Placeholder() for _ in cols),
    )
    with conn.cursor() as cur:
        cur.execute(q, values)
        return int(cur.fetchone()[0])

def update_row_by_id(conn, schema: str, table: str, row_id: int, payload: dict) -> None:
    if not payload:
        return
    cols = list(payload.keys())
    values = [payload[c] for c in cols]
    assigns = [sql.SQL("{} = {}").format(sql.Identifier(c), sql.Placeholder()) for c in cols]
    q = sql.SQL("UPDATE {}.{} SET {} WHERE id = %s").format(
        sql.Identifier(schema), sql.Identifier(table),
        sql.SQL(", ").join(assigns),
    )
    with conn.cursor() as cur:
        cur.execute(q, values + [row_id])

def infer_media_payload(
    video_path: Path, title: str, owner_id: int, privacy: str, ffmeta: dict,
    nas_path_override: str | None = None,
) -> dict:
    stat = video_path.stat()
    mime_type = mimetypes.guess_type(video_path.name)[0] or "video/mp4"
    return {
        "title": title,
        # Nom de fichier ORIGINAL, distinct de l'emplacement de stockage
        # (nas_path) : stable même si le fichier est déplacé (NAS, Drive,
        # local — la même vidéo peut transiter par plusieurs emplacements
        # au fil du temps, son nom, lui, ne change pas). C'est la clé de
        # réconciliation utilisée par upsert_media ci-dessous.
        "filename": video_path.name,
        "description": "Ajout local via GUI batch de transcription.",
        "nas_path": nas_path_override or str(video_path.resolve()),
        "thumbnail_path": None,
        "privacy": privacy,
        "views_count": 0,
        "owner_id": owner_id,
        "filesize": stat.st_size,
        "duration": ffmeta.get("duration"),
        "mime_type": mime_type,
        "created_at": now_dt(),
        "updated_at": now_dt(),
        "last_scanned_at": now_dt(),
        "file_extension": video_path.suffix.lower() or None,
        "nas_mtime": int(stat.st_mtime),
        "width": ffmeta.get("width"),
        "height": ffmeta.get("height"),
        "video_codec": ffmeta.get("video_codec"),
        "audio_codec": ffmeta.get("audio_codec"),
        "bitrate": ffmeta.get("bitrate"),
        "hls_path": None,
        "is_missing": False,
    }

def upsert_media(
    conn, video_path: Path, title: str, owner_id: int, privacy: str, ffmeta: dict,
    nas_path_override: str | None = None,
) -> tuple[int, str | None]:
    """Crée ou met à jour la ligne `medias` pour ce fichier.

    Deux stratégies de correspondance, dans cet ordre :
    1. `nas_path` exact — l'emplacement de stockage n'a pas changé.
    2. `filename` + `filesize` identiques sur un média dont l'emplacement
       DIFFÈRE — signature fiable d'un même fichier déplacé (ex: une vidéo
       d'abord transcrite depuis Google Drive, puis retrouvée sur le NAS une
       fois synchronisée dans la même arborescence). Dans ce cas on ne
       recrée PAS de doublon à retranscrire : l'ancien emplacement part
       dans `previous_paths` (historique), le nouveau devient `nas_path` —
       la transcription déjà faite reste associée à la bonne vidéo.

    Retourne (media_id, ancien_chemin_si_reconciliation_sinon_None) — le
    second élément permet à l'appelant de loguer clairement ce qui s'est
    passé plutôt que de le faire silencieusement."""
    cols = get_columns(conn, "public", "medias")
    payload = cleaned_payload(
        infer_media_payload(video_path, title, owner_id, privacy, ffmeta, nas_path_override), cols
    )
    nas_path = nas_path_override or str(video_path.resolve())
    filename = video_path.name
    filesize = video_path.stat().st_size

    existing_id = select_scalar(
        conn,
        "SELECT id FROM public.medias WHERE nas_path=%s ORDER BY id DESC LIMIT 1",
        (nas_path,),
    )

    relocated_from: str | None = None
    if existing_id is None and "filename" in cols and "previous_paths" in cols:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, nas_path, previous_paths FROM public.medias "
                "WHERE filename=%s AND filesize=%s AND nas_path IS DISTINCT FROM %s "
                "ORDER BY id DESC LIMIT 1",
                (filename, filesize, nas_path),
            )
            row = cur.fetchone()
        if row is not None:
            existing_id, old_path, old_previous = row
            relocated_from = old_path
            history = f"{old_previous}\n{old_path}" if old_previous else (old_path or "")
            payload["previous_paths"] = history

    if existing_id is not None:
        upd = {k: v for k, v in payload.items() if k != "created_at"}
        upd["updated_at"] = now_dt()
        if "last_scanned_at" in cols:
            upd["last_scanned_at"] = now_dt()
        update_row_by_id(conn, "public", "medias", int(existing_id), upd)
        conn.commit()
        return int(existing_id), relocated_from

    media_id = insert_row(conn, "public", "medias", payload)
    conn.commit()
    return media_id, None

def patch_media_duration(conn, media_id: int, duration_seconds: float | None) -> None:
    if duration_seconds is None:
        return
    cols = get_columns(conn, "public", "medias")
    upd = cleaned_payload(
        {"duration": duration_seconds, "updated_at": now_dt(), "last_scanned_at": now_dt()},
        cols,
    )
    update_row_by_id(conn, "public", "medias", media_id, upd)
    conn.commit()

# ── Domaine "transcription" (schéma dédié : runs → transcripts → files) ────

def create_run(conn, media_id: int, payload: dict) -> int:
    """Ouvre un run (un passage du pipeline sur une vidéo) et retourne son id."""
    row = {"media_id": media_id, "status": "running", **payload}
    run_id = insert_row(conn, "transcription", "runs", row)
    conn.commit()
    return run_id

def update_run(conn, run_id: int, payload: dict) -> None:
    update_row_by_id(conn, "transcription", "runs", run_id, payload)
    conn.commit()

def finish_run(conn, run_id: int, status: str, error: str | None = None) -> None:
    update_run(conn, run_id, {"status": status, "error": error, "finished_at": now_dt()})

def create_transcript(conn, run_id: int, media_id: int, language: str,
                      is_primary: bool, language_probability: float | None) -> int:
    transcript_id = insert_row(conn, "transcription", "transcripts", {
        "run_id": run_id,
        "media_id": media_id,
        "language": language,
        "is_primary_language": is_primary,
        "language_probability": language_probability,
        "status": "processing",
        "transcription_started_at": now_dt(),
    })
    conn.commit()
    return transcript_id

def finish_transcript_done(conn, transcript_id: int, bundle: dict, engine: str,
                           aligned: bool, diarized: bool, quality: dict | None = None) -> None:
    row = {
        "engine": engine,
        "aligned": aligned,
        "diarized": diarized,
        "status": "done",
        "error": None,
        "segment_count": bundle.get("segment_count"),
        "transcript_txt": bundle["txt"],
        "transcript_srt": bundle["srt"],
        "transcript_vtt": bundle["vtt"],
        "transcript_json": Json(bundle["json"]),
        "transcription_finished_at": now_dt(),
    }
    if quality:
        row.update({
            "word_count": quality.get("word_count"),
            "avg_confidence": quality.get("avg_confidence"),
            "avg_no_speech_prob": quality.get("avg_no_speech_prob"),
            "speech_seconds": quality.get("speech_seconds"),
        })
    update_row_by_id(conn, "transcription", "transcripts", transcript_id, row)
    conn.commit()

def finish_transcript_failed(conn, transcript_id: int, error_message: str) -> None:
    update_row_by_id(conn, "transcription", "transcripts", transcript_id, {
        "status": "failed",
        "error": error_message[:5000],
        "transcription_finished_at": now_dt(),
    })
    conn.commit()

def register_transcript_files(conn, transcript_id: int, output_specs: list[tuple[Path, str, str]]) -> None:
    with conn.cursor() as cur:
        for path, fmt, mime_type in output_specs:
            if not path.exists():
                continue
            cur.execute(
                "INSERT INTO transcription.transcript_files "
                "(transcript_id, file_format, file_name, file_path, mime_type, file_size) "
                "VALUES (%s, %s, %s, %s, %s, %s) "
                "ON CONFLICT (transcript_id, file_format) DO UPDATE SET "
                "file_name = EXCLUDED.file_name, file_path = EXCLUDED.file_path, "
                "mime_type = EXCLUDED.mime_type, file_size = EXCLUDED.file_size",
                (transcript_id, fmt, path.name, str(path.resolve()), mime_type, path.stat().st_size),
            )
    conn.commit()

def fetch_already_transcribed(conn) -> tuple[set[int], set[str]]:
    """Médias INTÉGRALEMENT transcrits : pour CHAQUE langue déjà tentée, le
    DERNIER essai (le plus récent) est 'done' (ids, nas_paths). Permet de
    sauter automatiquement les vidéos déjà transcrites quand on coche tout
    le NAS (les éléments issus de l'explorateur n'ont qu'un nas_path).

    On regarde le dernier essai PAR LANGUE (pas "aucun échec dans
    l'historique") : sinon un média qui a échoué une fois puis réussi au
    retry resterait bloqué indéfiniment à cause de la vieille ligne
    'failed' laissée par la première tentative (cf. fetch_media_language_status,
    utilisée pour ne retenter QUE les langues en échec plutôt que tout le média).

    Tri sur id DESC (pas sur l'horodatage) : le batch tourne potentiellement
    plusieurs jours et peut changer de machine/horloge entre deux passages —
    seul l'id, strictement croissant à l'insertion, garantit de retrouver le
    DERNIER essai sans dépendre d'une horloge synchronisée."""
    with conn.cursor() as cur:
        cur.execute(
            "WITH latest AS ("
            "  SELECT DISTINCT ON (media_id, language) media_id, language, status "
            "  FROM transcription.transcripts "
            "  ORDER BY media_id, language, id DESC"
            ") "
            "SELECT DISTINCT m.id, m.nas_path "
            "FROM public.medias m "
            "JOIN latest l ON l.media_id = m.id "
            "WHERE NOT EXISTS ("
            "  SELECT 1 FROM latest l2 WHERE l2.media_id = m.id AND l2.status != 'done'"
            ")"
        )
        rows = cur.fetchall()
    return {int(r[0]) for r in rows}, {r[1] for r in rows if r[1]}

def fetch_media_language_status(conn, media_id: int) -> dict[str, str]:
    """Statut du DERNIER essai de transcription, par langue, pour un média
    donné (ex: {"en": "done", "fr": "failed"}). Utilisée pour ne retenter
    que les langues effectivement en échec/absentes lors d'un nouveau
    passage sur un média déjà partiellement traité — évite de regaspiller
    du calcul GPU (FW + alignement WhisperX) sur une langue déjà réussie."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT DISTINCT ON (language) language, status "
            "FROM transcription.transcripts WHERE media_id = %s "
            "ORDER BY language, id DESC",
            (media_id,),
        )
        rows = cur.fetchall()
    return {r[0]: r[1] for r in rows}

# ─────────────────────────── SOUS-PROCESSUS ────────────────────────────────

def run_subprocess(
    python_exe: Path, script_path: Path, config: dict,
    work_dir: Path, label: str, log_func, verbose: bool,
    progress_func=None, on_start=None,
) -> dict[str, Path]:
    """Lance un sous-processus enfant (FW ou WX) et récupère un chemin JSON de résultat
    par langue transcrite (le sous-processus peut émettre plusieurs RESULT_JSON|<lang>|<path>).
    Les lignes PROGRESS|<langue>|<pct> pilotent la barre de progression via progress_func
    plutôt que d'être écrites dans le journal (pour ne pas le saturer).
    on_start(proc), si fourni, est appelé juste après le lancement — utilisé pour
    pouvoir suspendre/reprendre ce sous-processus depuis le bouton Pause du GUI."""
    cfg_path = work_dir / f"{label.lower()}_config.json"
    cfg_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    if config.get("hf_token"):
        env["HF_TOKEN"] = config["hf_token"]
        env["HUGGINGFACE_HUB_TOKEN"] = config["hf_token"]

    proc = subprocess.Popen(
        [str(python_exe), "-X", "utf8", str(script_path), str(cfg_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    if on_start is not None:
        on_start(proc)

    results: dict[str, Path] = {}
    tail: list[str] = []
    assert proc.stdout is not None

    for raw in proc.stdout:
        line = raw.rstrip("\r\n")
        if not line:
            continue

        tail.append(line)
        if len(tail) > 60:
            tail.pop(0)

        if line.startswith("RESULT_JSON|"):
            _, lang, path_str = line.split("|", 2)
            results[lang.strip()] = Path(path_str.strip())
            continue

        if line.startswith("PROGRESS|"):
            parts = line.split("|", 2)
            if len(parts) == 3 and progress_func is not None:
                _, sub_label, pct_str = parts
                try:
                    progress_func(f"{label} {sub_label.strip()}", int(pct_str.strip()))
                except ValueError:
                    pass
            if verbose:
                log_func(f"[{label}] {line}")
            continue

        if verbose or line.startswith(("INFO|", "ERROR|", "TRACE|")):
            log_func(f"[{label}] {line}")

    code = proc.wait()
    if code != 0:
        detail = "\n".join(tail[-25:])
        raise RuntimeError(
            f"Sous-processus {label} terminé avec code {code}.\n"
            f"Dernières lignes:\n{detail}"
        )

    if not results:
        detail = "\n".join(tail[-25:])
        raise RuntimeError(
            f"Aucun RESULT_JSON reçu pour {label}.\n"
            f"Dernières lignes:\n{detail}"
        )

    return results

def load_output_bundle(work_dir: Path, json_path: Path, prefix: str) -> dict:
    txt = work_dir / f"{prefix}.txt"
    srt = work_dir / f"{prefix}.srt"
    vtt = work_dir / f"{prefix}.vtt"
    data = json.loads(json_path.read_text(encoding="utf-8"))
    return {
        "txt": txt.read_text(encoding="utf-8") if txt.exists() else "",
        "srt": srt.read_text(encoding="utf-8") if srt.exists() else "",
        "vtt": vtt.read_text(encoding="utf-8") if vtt.exists() else "",
        "json": data,
        "detected_language": data.get("language") or "fr",
        "duration_seconds": data.get("duration_seconds"),
        "segment_count": data.get("segment_count"),
        "specs": [
            (txt, "txt", "text/plain"),
            (srt, "srt", "application/x-subrip"),
            (vtt, "vtt", "text/vtt"),
            (json_path, "json", "application/json"),
        ],
    }

class _NasPrefetch:
    """État partagé du pré-téléchargement en arrière-plan de la vidéo NAS
    suivante, déclenché NAS_PREFETCH_DELAY_SECONDS après le début de la
    transcription locale de l'item courant (le sous-processus FW/WX
    consomme CPU/GPU, pas le réseau — la session DSM Synology, elle, expire
    par inactivité réseau pendant ce laps de temps). Un nouvel objet est
    créé après chaque consommation/annulation — jamais réutilisé d'un item
    à l'autre."""

    def __init__(self):
        self.lock = threading.Lock()
        self.item: dict | None = None
        self.thread: threading.Thread | None = None
        self.cancel_event = threading.Event()
        self.started_download = False
        self.work_dir: Path | None = None
        self.video_path: Path | None = None
        self.error: Exception | None = None


# ─────────────────────────── GUI ───────────────────────────────────────────

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(f"{APP_TITLE} v{APP_VERSION}")
        self.geometry("1300x920")
        self.minsize(1100, 800)

        self.queue: Queue = Queue()
        self.worker_thread: threading.Thread | None = None
        self.pause_requested = threading.Event()
        self.stop_requested = threading.Event()
        self._current_proc: subprocess.Popen | None = None

        self.local_config = load_local_config()
        configure_db(self.local_config)
        self.config_status_var = tk.StringVar(value=self._describe_config_status())

        self.single_file_var = tk.StringVar()
        self.folders_var = tk.StringVar(value="Aucun dossier sélectionné.")
        self.title_var = tk.StringVar()
        self.privacy_var = tk.StringVar(value="PRIVATE")
        self.model_var = tk.StringVar(value="large-v3")
        self.fw_python_var = tk.StringVar(value=self.local_config["fw_python"])
        self.wx_python_var = tk.StringVar(value=self.local_config["wx_python"])
        self.db_password_var = tk.StringVar(value=self.local_config["db_password"])
        self.hf_token_var = tk.StringVar(value=self.local_config["hf_token"])
        self.local_model_dir_var = tk.StringVar(value=self.local_config["local_model_dir"])
        self.use_alignment_var = tk.BooleanVar(value=True)
        self.diarization_var = tk.BooleanVar(value=False)
        self.recursive_var = tk.BooleanVar(value=True)
        self.keep_work_dirs_var = tk.BooleanVar(value=False)
        self.use_nas_var = tk.BooleanVar(value=False)
        self.use_drive_var = tk.BooleanVar(value=False)
        self.drive_link_var = tk.StringVar()
        self.status_var = tk.StringVar(value="Prêt.")
        self.stage_var = tk.StringVar(value="")
        self.hw_var = tk.StringVar(value="Matériel non détecté.")
        self.batch_var = tk.StringVar(value="Aucun média sélectionné.")
        self.nas_status_var = tk.StringVar(value="Arborescence NAS non chargée.")
        self.drive_status_var = tk.StringVar(value="Aucun dossier Drive exploré.")
        self.db_status_var = tk.StringVar(value="⏳ Base de données : vérification…")

        self.multi_files: list[str] = []
        self.multi_folders: list[str] = []
        self.nas_client: "NasClientCls | None" = None
        self.gdrive_client: "GDriveClientCls | None" = None
        self.nas_total_videos: int | None = None
        self.nas_duration_by_path: dict[str, float] = {}
        self._apply_style()
        self._build_ui()
        self.after(100, self._consume_queue)
        self._detect_hardware_clicked()  # détection automatique au démarrage, sans clic requis
        self._check_db_clicked()  # test de connexion DB automatique au démarrage
        self._sync_nas_catalog_clicked()  # synchronisation catalogue NAS automatique au démarrage (aucun mécanisme équivalent côté serveur)

    # ── Style ─────────────────────────────────────────────────────────────

    def _apply_style(self):
        # Palette "ice" : fond bleu glacé très pâle, blancs francs, accent cyan froid,
        # typographie fine (Segoe UI 9-10, gras réservé aux titres).
        self.BG = "#edf4fb"
        self.SURFACE = "#ffffff"
        self.ACCENT = "#0284c7"
        self.ACCENT_DARK = "#075985"
        self.ACCENT_ICE = "#e0f2fe"
        self.TEXT = "#0f172a"
        self.MUTED = "#64748b"
        self.BORDER = "#d3e2f0"

        self.configure(background=self.BG)

        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        base_font = ("Segoe UI", 10)
        style.configure(".", background=self.BG, foreground=self.TEXT, font=base_font)
        style.configure("TFrame", background=self.BG)
        style.configure("TLabel", background=self.BG, foreground=self.TEXT)
        style.configure("Muted.TLabel", background=self.BG, foreground=self.MUTED, font=("Segoe UI", 9))
        style.configure("TCheckbutton", background=self.BG, foreground=self.TEXT, font=("Segoe UI", 9))

        style.configure(
            "TLabelframe", background=self.BG, bordercolor=self.BORDER,
            relief="solid", borderwidth=1,
        )
        style.configure("TLabelframe.Label", background=self.BG, foreground=self.ACCENT_DARK, font=("Segoe UI", 10, "bold"))

        style.configure("TEntry", fieldbackground=self.SURFACE, bordercolor=self.BORDER, lightcolor=self.BORDER, padding=4)
        style.configure("TCombobox", fieldbackground=self.SURFACE, padding=4)

        style.configure("TButton", padding=(10, 5), font=("Segoe UI", 9), background=self.SURFACE, bordercolor=self.BORDER)
        style.map("TButton", background=[("active", self.ACCENT_ICE)])
        style.configure(
            "Accent.TButton", background=self.ACCENT, foreground="white",
            padding=(14, 7), font=("Segoe UI", 10, "bold"), borderwidth=0,
        )
        style.map(
            "Accent.TButton",
            background=[("active", self.ACCENT_DARK), ("disabled", "#a5d8f0")],
            foreground=[("disabled", "#f0f9ff")],
        )

        style.configure(
            "Link.TButton", padding=2, relief="flat",
            background=self.BG, foreground=self.ACCENT, font=("Segoe UI", 9),
        )
        style.map("Link.TButton", foreground=[("active", self.ACCENT_DARK)])

        # En-tête clair "ice" avec liseré accent (posé dans _build_ui)
        style.configure("Header.TFrame", background=self.ACCENT_ICE)
        style.configure("Header.TLabel", background=self.ACCENT_ICE, foreground=self.ACCENT_DARK, font=("Segoe UI Semibold", 15))
        style.configure("HeaderSub.TLabel", background=self.ACCENT_ICE, foreground=self.MUTED, font=("Segoe UI", 9))
        style.configure("HeaderOk.TLabel", background=self.ACCENT_ICE, foreground="#047857", font=("Segoe UI", 9, "bold"))
        style.configure("HeaderBad.TLabel", background=self.ACCENT_ICE, foreground="#b91c1c", font=("Segoe UI", 9, "bold"))
        style.configure("HeaderWait.TLabel", background=self.ACCENT_ICE, foreground=self.MUTED, font=("Segoe UI", 9))

        style.configure("TNotebook", background=self.BG, borderwidth=0, tabmargins=(0, 4, 0, 0))
        style.configure(
            "TNotebook.Tab", background=self.BG, foreground=self.MUTED,
            padding=(18, 7), font=("Segoe UI", 10), borderwidth=0,
        )
        style.map(
            "TNotebook.Tab",
            background=[("selected", self.SURFACE)],
            foreground=[("selected", self.ACCENT_DARK)],
            expand=[("selected", (0, 0, 0, 1))],
        )

        style.configure("TProgressbar", background=self.ACCENT, troughcolor=self.ACCENT_ICE, bordercolor=self.ACCENT_ICE)

        style.configure(
            "Nas.Treeview", background=self.SURFACE, fieldbackground=self.SURFACE,
            foreground=self.TEXT, borderwidth=1, relief="solid", rowheight=26, font=("Segoe UI", 9),
        )
        style.map("Nas.Treeview", background=[("selected", "#cfe8fa")], foreground=[("selected", self.TEXT)])
        style.configure(
            "Nas.Treeview.Heading", background=self.ACCENT_ICE, foreground=self.ACCENT_DARK,
            font=("Segoe UI", 9, "bold"), relief="flat", padding=(6, 4),
        )
        style.map("Nas.Treeview.Heading", background=[("active", "#cfe8fa")])

    # ── Construction UI ──────────────────────────────────────────────────

    def _build_ui(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)

        header = ttk.Frame(self, style="Header.TFrame", padding=(16, 10))
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(1, weight=1)
        ttk.Label(header, text=f"🎙️  {APP_TITLE}", style="Header.TLabel").grid(row=0, column=0, sticky="w")
        self.db_status_label = ttk.Label(header, textvariable=self.db_status_var, style="HeaderWait.TLabel")
        self.db_status_label.grid(row=0, column=1, sticky="e", padx=(0, 16))
        ttk.Label(header, text=f"v{APP_VERSION}", style="HeaderSub.TLabel").grid(row=0, column=2, sticky="e")
        tk.Frame(self, height=2, background=self.ACCENT, borderwidth=0).grid(row=1, column=0, sticky="ew")

        # ── Onglets : Réglages / Journal (pleine hauteur) ────────────────
        self.notebook = ttk.Notebook(self)
        self.notebook.grid(row=2, column=0, sticky="nsew", padx=12, pady=(10, 4))

        settings_tab = ttk.Frame(self.notebook, padding=(2, 10, 2, 2))
        self.journal_tab = ttk.Frame(self.notebook, padding=(2, 10, 2, 2))
        self.notebook.add(settings_tab, text="  ⚙  Réglages  ")
        self.notebook.add(self.journal_tab, text="  🖥  Journal  ")
        settings_tab.columnconfigure(0, weight=1)
        settings_tab.rowconfigure(1, weight=1)  # la zone NAS absorbe la hauteur disponible
        settings_tab.rowconfigure(2, weight=1)  # idem pour la zone Drive
        self.journal_tab.columnconfigure(0, weight=1)
        self.journal_tab.rowconfigure(0, weight=1)

        src = ttk.LabelFrame(settings_tab, text="📂  Sources", padding=10)
        src.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        src.columnconfigure(1, weight=1)

        ttk.Label(src, text="Fichier unique").grid(row=0, column=0, sticky="w", pady=3)
        ttk.Entry(src, textvariable=self.single_file_var).grid(row=0, column=1, sticky="ew", padx=6, pady=3)
        ttk.Button(src, text="Parcourir…", command=self._browse_single_file).grid(row=0, column=2, pady=3)

        ttk.Label(src, text="Dossiers").grid(row=1, column=0, sticky="w", pady=3)
        ttk.Entry(src, textvariable=self.folders_var, state="readonly").grid(row=1, column=1, sticky="ew", padx=6, pady=3)
        ttk.Button(src, text="Ajouter un dossier…", command=self._browse_folder).grid(row=1, column=2, pady=3)

        row2 = ttk.Frame(src)
        row2.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(4, 0))
        ttk.Button(row2, text="Ajouter plusieurs fichiers…", command=self._browse_multi_files).pack(side="left")
        ttk.Button(row2, text="Effacer dossiers", command=self._clear_folders).pack(side="left", padx=(8, 0))
        ttk.Checkbutton(row2, text="Scan récursif (sous-dossiers de sous-dossiers inclus)", variable=self.recursive_var).pack(side="left", padx=12)
        ttk.Button(row2, text="Effacer sélection multi", command=self._clear_multi_files).pack(side="left")
        ttk.Label(row2, textvariable=self.batch_var, style="Muted.TLabel").pack(side="right")

        self._create_checkbox_images()

        nas = ttk.LabelFrame(settings_tab, text="🗄️  Source NAS (API)", padding=10)
        nas.grid(row=1, column=0, sticky="nsew", pady=(0, 8))
        nas.columnconfigure(0, weight=1)

        ttk.Checkbutton(
            nas, text="Traiter les éléments du NAS cochés ci-dessous (au lieu des sources locales ci-dessus)",
            variable=self.use_nas_var, command=self._toggle_nas_browser_visibility,
        ).grid(row=0, column=0, sticky="w")

        self.nas_browser_frame = ttk.Frame(nas)
        self.nas_browser_frame.grid(row=1, column=0, sticky="nsew", pady=(6, 0))
        self.nas_browser_frame.columnconfigure(0, weight=1)
        nas.rowconfigure(1, weight=1)

        nas_btn_row = ttk.Frame(self.nas_browser_frame)
        nas_btn_row.grid(row=0, column=0, sticky="ew", pady=(0, 4))
        ttk.Button(nas_btn_row, text="🔄 Parcourir le NAS", command=self._list_nas_videos_clicked).pack(side="left")
        ttk.Button(nas_btn_row, text="☑ Tout cocher", command=self._select_all_nas).pack(side="left", padx=(8, 0))
        ttk.Button(nas_btn_row, text="☐ Tout décocher", command=self._select_none_nas).pack(side="left", padx=(8, 0))
        ttk.Label(nas_btn_row, textvariable=self.nas_status_var, style="Muted.TLabel").pack(side="right")

        nas_hint = ttk.Label(
            self.nas_browser_frame, style="Muted.TLabel",
            text="▸ déplie un dossier  •  clique sur la case pour cocher (un dossier coché = toutes ses vidéos, "
                 "sous-dossiers inclus)  •  cliquer sur le nom sélectionne sans cocher.",
        )
        nas_hint.grid(row=1, column=0, sticky="w", pady=(0, 4))

        nas_tree_container = ttk.Frame(self.nas_browser_frame)
        nas_tree_container.grid(row=2, column=0, sticky="nsew")
        self.nas_browser_frame.rowconfigure(2, weight=1)
        nas_tree_container.columnconfigure(0, weight=1)
        nas_tree_container.rowconfigure(0, weight=1)

        self.nas_tree = ttk.Treeview(
            nas_tree_container, columns=("size", "duration", "mtime"), show="tree headings",
            height=10, selectmode="browse", style="Nas.Treeview",
        )
        self.nas_tree.heading("#0", text="Nom", anchor="w")
        self.nas_tree.heading("size", text="Taille", anchor="e")
        self.nas_tree.heading("duration", text="Durée", anchor="center")
        self.nas_tree.heading("mtime", text="Modifié le", anchor="center")
        self.nas_tree.column("#0", stretch=True, minwidth=280)
        self.nas_tree.column("size", width=90, stretch=False, anchor="e")
        self.nas_tree.column("duration", width=80, stretch=False, anchor="center")
        self.nas_tree.column("mtime", width=130, stretch=False, anchor="center")
        self.nas_tree.tag_configure("even", background=self.SURFACE)
        self.nas_tree.tag_configure("odd", background="#f4f9fd")
        self.nas_tree.tag_configure("checked", background=self.ACCENT_ICE)
        self.nas_tree.grid(row=0, column=0, sticky="nsew")
        nas_scroll = ttk.Scrollbar(nas_tree_container, orient="vertical", command=self.nas_tree.yview)
        nas_scroll.grid(row=0, column=1, sticky="ns")
        self.nas_tree.configure(yscrollcommand=nas_scroll.set)
        self.nas_tree.bind("<Button-1>", self._on_nas_tree_click)
        self.nas_tree.bind("<<TreeviewOpen>>", self._on_nas_tree_open)
        self.nas_tree_data: dict[str, dict] = {}

        if not self.use_nas_var.get():
            self.nas_browser_frame.grid_remove()

        # ── Source Google Drive (lien public "Tous les utilisateurs disposant du lien") ──
        drive = ttk.LabelFrame(settings_tab, text="☁️  Source Google Drive (lien)", padding=10)
        drive.grid(row=2, column=0, sticky="nsew", pady=(0, 8))
        drive.columnconfigure(0, weight=1)

        ttk.Checkbutton(
            drive, text="Traiter les éléments Drive cochés ci-dessous (au lieu des sources locales ci-dessus)",
            variable=self.use_drive_var, command=self._toggle_drive_browser_visibility,
        ).grid(row=0, column=0, sticky="w")

        self.drive_browser_frame = ttk.Frame(drive)
        self.drive_browser_frame.grid(row=1, column=0, sticky="nsew", pady=(6, 0))
        self.drive_browser_frame.columnconfigure(0, weight=1)
        drive.rowconfigure(1, weight=1)

        drive_link_row = ttk.Frame(self.drive_browser_frame)
        drive_link_row.grid(row=0, column=0, sticky="ew", pady=(0, 4))
        drive_link_row.columnconfigure(0, weight=1)
        ttk.Entry(drive_link_row, textvariable=self.drive_link_var).grid(row=0, column=0, sticky="ew", padx=(0, 6))
        ttk.Button(drive_link_row, text="🔄 Parcourir Drive", command=self._list_drive_videos_clicked).grid(row=0, column=1)

        drive_btn_row = ttk.Frame(self.drive_browser_frame)
        drive_btn_row.grid(row=1, column=0, sticky="ew", pady=(0, 4))
        ttk.Button(drive_btn_row, text="☑ Tout cocher", command=self._select_all_drive).pack(side="left")
        ttk.Button(drive_btn_row, text="☐ Tout décocher", command=self._select_none_drive).pack(side="left", padx=(8, 0))
        ttk.Label(drive_btn_row, textvariable=self.drive_status_var, style="Muted.TLabel").pack(side="right")

        drive_hint = ttk.Label(
            self.drive_browser_frame, style="Muted.TLabel",
            text="Colle le lien d'un dossier Google Drive partagé en \"Tous les utilisateurs disposant du lien\", "
                 "clique Parcourir, puis coche les vidéos ou dossiers à traiter.",
        )
        drive_hint.grid(row=2, column=0, sticky="w", pady=(0, 4))

        drive_tree_container = ttk.Frame(self.drive_browser_frame)
        drive_tree_container.grid(row=3, column=0, sticky="nsew")
        self.drive_browser_frame.rowconfigure(3, weight=1)
        drive_tree_container.columnconfigure(0, weight=1)
        drive_tree_container.rowconfigure(0, weight=1)

        self.drive_tree = ttk.Treeview(
            drive_tree_container, columns=("size", "duration", "mtime"), show="tree headings",
            height=10, selectmode="browse", style="Nas.Treeview",
        )
        self.drive_tree.heading("#0", text="Nom", anchor="w")
        self.drive_tree.heading("size", text="Taille", anchor="e")
        self.drive_tree.heading("duration", text="Durée", anchor="center")
        self.drive_tree.heading("mtime", text="Modifié le", anchor="center")
        self.drive_tree.column("#0", stretch=True, minwidth=280)
        self.drive_tree.column("size", width=90, stretch=False, anchor="e")
        self.drive_tree.column("duration", width=80, stretch=False, anchor="center")
        self.drive_tree.column("mtime", width=130, stretch=False, anchor="center")
        self.drive_tree.tag_configure("even", background=self.SURFACE)
        self.drive_tree.tag_configure("odd", background="#f4f9fd")
        self.drive_tree.tag_configure("checked", background=self.ACCENT_ICE)
        self.drive_tree.grid(row=0, column=0, sticky="nsew")
        drive_scroll = ttk.Scrollbar(drive_tree_container, orient="vertical", command=self.drive_tree.yview)
        drive_scroll.grid(row=0, column=1, sticky="ns")
        self.drive_tree.configure(yscrollcommand=drive_scroll.set)
        self.drive_tree.bind("<Button-1>", self._on_drive_tree_click)
        self.drive_tree.bind("<<TreeviewOpen>>", self._on_drive_tree_open)
        self.drive_tree_data: dict[str, dict] = {}

        if not self.use_drive_var.get():
            self.drive_browser_frame.grid_remove()

        form = ttk.LabelFrame(settings_tab, text="🎛️  Paramètres de transcription", padding=10)
        form.grid(row=3, column=0, sticky="ew", pady=(0, 8))
        for i in range(3):
            form.columnconfigure(i, weight=1)

        ttk.Label(form, text="Titre par défaut").grid(row=0, column=0, sticky="w")
        ttk.Entry(form, textvariable=self.title_var).grid(row=1, column=0, sticky="ew", padx=(0, 8), pady=2)

        ttk.Label(form, text="Privacy").grid(row=0, column=1, sticky="w")
        ttk.Combobox(form, textvariable=self.privacy_var, values=PRIVACY_VALUES,
                     state="readonly", width=10).grid(row=1, column=1, sticky="w", padx=(0, 8), pady=2)

        ttk.Label(form, text="Modèle").grid(row=0, column=2, sticky="w")
        ttk.Combobox(form, textvariable=self.model_var, values=COMMON_MODELS,
                     state="normal").grid(row=1, column=2, sticky="ew", pady=2)

        checks = ttk.Frame(form)
        checks.grid(row=2, column=0, columnspan=3, sticky="w", pady=(14, 0))
        ttk.Checkbutton(checks, text="Alignement WhisperX", variable=self.use_alignment_var).pack(side="left", padx=(0, 14))
        ttk.Checkbutton(checks, text="Diarisation (HF requis)", variable=self.diarization_var).pack(side="left", padx=(0, 14))
        ttk.Checkbutton(checks, text="Conserver les fichiers de travail temporaires", variable=self.keep_work_dirs_var).pack(side="left")

        config_frame = ttk.LabelFrame(settings_tab, text="🔐  Configuration locale", padding=(10, 6))
        config_frame.grid(row=4, column=0, sticky="ew")
        ttk.Label(config_frame, textvariable=self.config_status_var, style="Muted.TLabel").pack(side="left", fill="x")

        # ── Barre d'actions : toujours visible, sous les onglets ─────────
        actions = ttk.Frame(self, padding=(12, 4, 12, 8))
        actions.grid(row=3, column=0, sticky="ew")
        actions.columnconfigure(7, weight=1)

        ttk.Button(actions, text="🔍 Re-détecter matériel", command=self._detect_hardware_clicked).grid(row=0, column=0, padx=(0, 6))
        self.start_button = ttk.Button(actions, text="▶  Lancer le batch", command=self._start_pipeline, style="Accent.TButton")
        self.start_button.grid(row=0, column=1, padx=(0, 6))
        self.pause_button = ttk.Button(actions, text="⏸ Pause", command=self._toggle_pause, state="disabled")
        self.pause_button.grid(row=0, column=2, padx=(0, 6))
        self.stop_button = ttk.Button(actions, text="⏹ Stop", command=self._request_stop, state="disabled")
        self.stop_button.grid(row=0, column=3, padx=(0, 6))
        ttk.Button(actions, text="🗑 Effacer journal", command=self._clear_log).grid(row=0, column=4, padx=(0, 12))
        self.progress = ttk.Progressbar(actions, mode="indeterminate", length=200)
        self.progress.grid(row=0, column=5, padx=(0, 8))
        ttk.Label(actions, textvariable=self.stage_var, style="Muted.TLabel").grid(row=0, column=6, sticky="w", padx=(0, 12))
        ttk.Label(actions, textvariable=self.hw_var, style="Muted.TLabel").grid(row=0, column=7, sticky="w")
        ttk.Label(actions, textvariable=self.status_var, font=("Segoe UI", 9, "bold")).grid(row=1, column=0, columnspan=8, sticky="w", pady=(6, 0))

        # Journal pleine hauteur dans son propre onglet
        self.log_text = scrolledtext.ScrolledText(
            self.journal_tab, wrap="word", font=("Consolas", 10),
            background="#0b1220", foreground="#cbd5e1", insertbackground="white",
            borderwidth=0, highlightthickness=0, padx=10, pady=8,
        )
        self.log_text.grid(row=0, column=0, sticky="nsew")
        self.log_text.tag_configure("ok", foreground="#34d399")
        self.log_text.tag_configure("error", foreground="#f87171")
        self.log_text.tag_configure("warn", foreground="#fbbf24")
        self.log_text.tag_configure("info", foreground="#7dd3fc")
        self._log_raw(f"GUI prête. {APP_TITLE} v{APP_VERSION}", "ok")

    # ── Navigation ───────────────────────────────────────────────────────

    def _browse_single_file(self):
        p = filedialog.askopenfilename(
            title="Choisir un média",
            filetypes=[("Audio/Vidéo", "*.mp3 *.wav *.m4a *.flac *.ogg *.wma *.aac *.mp4 *.mkv *.avi *.mov *.webm *.m4v *.mpg *.mpeg"),
                       ("Tous les fichiers", "*.*")],
        )
        if p:
            self.single_file_var.set(p)
            if not self.title_var.get().strip():
                self.title_var.set(Path(p).stem)
            self._refresh_batch_count()

    def _browse_multi_files(self):
        paths = filedialog.askopenfilenames(
            title="Choisir plusieurs médias",
            filetypes=[("Audio/Vidéo", "*.mp3 *.wav *.m4a *.flac *.ogg *.wma *.aac *.mp4 *.mkv *.avi *.mov *.webm *.m4v *.mpg *.mpeg"),
                       ("Tous les fichiers", "*.*")],
        )
        if paths:
            self.multi_files = list(paths)
            self._refresh_batch_count()

    def _clear_multi_files(self):
        self.multi_files = []
        self._refresh_batch_count()

    def _browse_folder(self):
        p = filedialog.askdirectory(title="Ajouter un dossier (clique plusieurs fois pour en ajouter d'autres)")
        if p and p not in self.multi_folders:
            self.multi_folders.append(p)
            self._update_folders_label()
            self._refresh_batch_count()

    def _clear_folders(self):
        self.multi_folders = []
        self._update_folders_label()
        self._refresh_batch_count()

    def _update_folders_label(self):
        n = len(self.multi_folders)
        if n == 0:
            self.folders_var.set("Aucun dossier sélectionné.")
        elif n == 1:
            self.folders_var.set(self.multi_folders[0])
        else:
            self.folders_var.set(f"{n} dossiers sélectionnés : " + " | ".join(Path(p).name for p in self.multi_folders))

    # ── NAS (API) ────────────────────────────────────────────────────────

    def _toggle_nas_browser_visibility(self):
        if self.use_nas_var.get():
            self.use_drive_var.set(False)
            self.drive_browser_frame.grid_remove()
            self.nas_browser_frame.grid()
        else:
            self.nas_browser_frame.grid_remove()
        self._refresh_batch_count()

    def _toggle_drive_browser_visibility(self):
        if self.use_drive_var.get():
            self.use_nas_var.set(False)
            self.nas_browser_frame.grid_remove()
            self.drive_browser_frame.grid()
        else:
            self.drive_browser_frame.grid_remove()
        self._refresh_batch_count()

    def _list_nas_videos_clicked(self):
        if NasClient is None:
            messagebox.showerror(
                "NAS indisponible",
                "Le module nas_client.py (ou la dépendance 'requests') est introuvable.",
            )
            return
        threading.Thread(target=self._browse_nas_root_worker, daemon=True).start()

    def _require_nas_client(self) -> "NasClientCls":
        """Retourne self.nas_client, ou lève une erreur claire s'il n'est pas
        encore connecté — évite de propager des None jusqu'à un appel réseau."""
        if self.nas_client is None:
            raise RuntimeError("Non connecté au NAS : clique d'abord sur 'Parcourir le NAS'.")
        return self.nas_client

    def _browse_nas_root_worker(self):
        cfg = self.local_config
        if not cfg["nas_username"].strip() or not cfg["nas_password"].strip():
            self._show_error(
                "Identifiants NAS manquants",
                f"Renseigne 'nas_username' et 'nas_password' dans {LOCAL_CONFIG_PATH}.",
            )
            return
        self._set_status("Connexion au NAS (API)…")
        assert NasClient is not None  # garanti par _list_nas_videos_clicked avant de lancer ce thread
        try:
            client = NasClient(cfg["nas_api_url"])
            client.login(cfg["nas_username"], cfg["nas_password"])
            result = client.browse_nas("")
        except Exception as exc:
            self._log(f"[ERREUR] NAS: {exc}")
            self._show_error("Connexion NAS échouée", str(exc))
            self._set_status("Prêt.")
            return

        self.nas_client = client

        # La durée n'est PAS renvoyée par /nas/browse (exploration brute du
        # système de fichiers) — seul le catalogue indexé (/documents/library)
        # la connaît. On construit donc une table nas_path -> durée à partir
        # de ce catalogue pour enrichir l'affichage de l'arbre, du mieux
        # possible (les fichiers pas encore catalogués resteront sans durée).
        try:
            indexed_videos = client.list_videos()
            total_videos = len(indexed_videos)
            self.nas_duration_by_path = {
                str(v["nas_path"]): float(v["duration"])
                for v in indexed_videos
                if v.get("nas_path") and v.get("duration")
            }
        except Exception as exc:
            total_videos = None
            self.nas_duration_by_path = {}
            self._log(f"[INFO] Impossible de charger le catalogue indexé (durées indisponibles) : {exc}")

        self._log(
            f"[OK] NAS: connecté en tant que {client.username}, arborescence chargée"
            + (f" — {total_videos} vidéos indexées au total." if total_videos is not None else ".")
        )
        self.queue.put(("nas_tree_root", (result.get("items", []), total_videos)))
        self._set_status("Arborescence NAS chargée.")

    def _load_nas_children_worker(self, item_id: str, path: str):
        try:
            result = self._require_nas_client().browse_nas(path)
        except Exception as exc:
            self._log(f"[ERREUR] Exploration NAS ({path}) : {exc}")
            return
        self.queue.put(("nas_tree_children", (item_id, result.get("items", []))))

    def _sync_nas_catalog_clicked(self):
        if NasClient is None:
            return
        threading.Thread(target=self._sync_nas_catalog_worker, daemon=True).start()

    def _sync_nas_catalog_worker(self):
        """Synchronise le catalogue indexé (public.medias, lu par list_videos())
        avec le contenu réel du NAS. Aucun mécanisme équivalent n'existe côté
        serveur (vérifié : aucune route de planification dans l'API) — sans ce
        déclenchement à chaque démarrage, le catalogue dérive silencieusement
        de la réalité (constaté : 3929 vs 22554 après une synchro manuelle).
        Tourne entièrement en arrière-plan : ne bloque ni l'UI, ni la détection
        matériel/DB, ni la possibilité de parcourir/cocher/lancer un batch —
        ce dernier explore le NAS en direct (/nas/browse), pas le catalogue."""
        cfg = self.local_config
        if not cfg["nas_username"].strip() or not cfg["nas_password"].strip():
            return  # pas d'identifiants NAS configurés : rien à synchroniser
        assert NasClient is not None  # garanti par _sync_nas_catalog_clicked
        try:
            client = NasClient(cfg["nas_api_url"])
            client.login(cfg["nas_username"], cfg["nas_password"])
            client.sync_all_nas(max_depth=20, privacy="private")
            self._log("[INFO] Synchronisation du catalogue NAS démarrée en arrière-plan…")
        except Exception as exc:
            self._log(f"[AVERTISSEMENT] Déclenchement de la synchronisation NAS échoué : {exc}")
            return

        last_reported = -20
        while True:
            time.sleep(20)
            try:
                status = client.sync_status()
            except Exception:
                continue  # panne transitoire de sondage : retente au tour suivant
            state = status.get("status")
            progress = status.get("progress") or 0
            if state not in ("running", "in_progress", "syncing"):
                errors = status.get("errors") or []
                self._log(
                    f"[OK] Synchronisation du catalogue NAS terminée : "
                    f"{status.get('videos_found')} vidéo(s) trouvée(s), "
                    f"{status.get('created')} créée(s), {status.get('updated')} mise(s) à jour"
                    + (f", {len(errors)} erreur(s)." if errors else ".")
                )
                return
            if progress >= last_reported + 20:
                last_reported = progress
                self._log(
                    f"[INFO] Synchronisation NAS en cours… {progress}% "
                    f"({status.get('videos_found')} vidéo(s) trouvée(s))"
                )

    # ── Google Drive (lien public) ──────────────────────────────────────

    def _list_drive_videos_clicked(self):
        if GDriveClient is None:
            messagebox.showerror(
                "Drive indisponible",
                "Le module gdrive_client.py (ou la dépendance 'requests') est introuvable.",
            )
            return
        link = self.drive_link_var.get().strip()
        if not link:
            messagebox.showerror("Lien manquant", "Colle d'abord un lien de dossier Google Drive.")
            return
        threading.Thread(target=self._browse_drive_root_worker, args=(link,), daemon=True).start()

    def _require_gdrive_client(self) -> "GDriveClientCls":
        """Retourne self.gdrive_client, ou lève une erreur claire s'il n'est pas
        encore connecté — évite de propager des None jusqu'à un appel réseau."""
        if self.gdrive_client is None:
            raise RuntimeError("Non connecté à Drive : clique d'abord sur 'Parcourir Drive'.")
        return self.gdrive_client

    def _browse_drive_root_worker(self, link: str):
        api_key = self.local_config["google_drive_api_key"].strip()
        if not api_key:
            self._show_error(
                "Clé API Drive manquante",
                f"Renseigne 'google_drive_api_key' dans {LOCAL_CONFIG_PATH}.",
            )
            return
        self._set_status("Exploration du dossier Google Drive…")
        assert GDriveClient is not None  # garanti par _list_drive_videos_clicked avant de lancer ce thread
        try:
            folder_id = GDriveClient.resolve_folder_id(link)
            client = GDriveClient(api_key)
            result = client.browse(folder_id)
        except Exception as exc:
            self._log(f"[ERREUR] Drive: {exc}")
            self._show_error("Exploration Drive échouée", str(exc))
            self._set_status("Prêt.")
            return

        self.gdrive_client = client
        self._log(f"[OK] Drive: dossier ouvert ({len(result.get('items', []))} élément(s) à la racine).")
        self.queue.put(("drive_tree_root", result.get("items", [])))
        self._set_status("Arborescence Drive chargée.")

    def _load_drive_children_worker(self, item_id: str, folder_id: str):
        try:
            result = self._require_gdrive_client().browse(folder_id)
        except Exception as exc:
            self._log(f"[ERREUR] Exploration Drive ({folder_id}) : {exc}")
            return
        self.queue.put(("drive_tree_children", (item_id, result.get("items", []))))

    def _create_checkbox_images(self):
        """Dessine les cases à cocher (pixel par pixel, sans dépendance) utilisées
        dans l'arbre NAS : seule la case est cliquable pour cocher, comme dans
        l'Explorateur Windows. Références gardées sur self (sinon Tk les libère)."""
        s = 18
        self.img_unchecked = tk.PhotoImage(width=s, height=s)
        self.img_unchecked.put(self.SURFACE, to=(2, 2, 16, 16))
        for x0, y0, x1, y1 in [(2, 2, 16, 3), (2, 15, 16, 16), (2, 2, 3, 16), (15, 2, 16, 16)]:
            self.img_unchecked.put("#94a3b8", to=(x0, y0, x1, y1))

        self.img_checked = tk.PhotoImage(width=s, height=s)
        self.img_checked.put(self.ACCENT, to=(2, 2, 16, 16))
        for x, y in [(4, 8), (5, 9), (6, 10), (7, 11), (8, 10), (9, 9), (10, 8), (11, 7), (12, 6)]:
            self.img_checked.put("white", to=(x, y, x + 2, y + 2))

        self.img_blank = tk.PhotoImage(width=s, height=s)  # fichiers non cochables

    def _fmt_size(self, n) -> str:
        if not n:
            return ""
        n = float(n)
        for unit in ("o", "Ko", "Mo", "Go", "To"):
            if n < 1024:
                return f"{n:.0f} {unit}" if unit == "o" else f"{n:.1f} {unit}"
            n /= 1024
        return f"{n:.1f} Po"

    def _fmt_mtime(self, ts) -> str:
        """Formate un mtime epoch (NAS)."""
        if not ts:
            return ""
        try:
            return datetime.fromtimestamp(int(ts)).strftime("%d/%m/%Y %H:%M")
        except Exception:
            return ""

    def _fmt_drive_mtime(self, iso_ts) -> str:
        """Formate un modifiedTime Drive (ISO-8601, ex: 2026-07-19T12:34:56.000Z)."""
        if not iso_ts:
            return ""
        try:
            return datetime.fromisoformat(str(iso_ts).replace("Z", "+00:00")).strftime("%d/%m/%Y %H:%M")
        except Exception:
            return ""

    def _fmt_duration(self, seconds) -> str:
        if not seconds:
            return ""
        try:
            total = int(float(seconds))
        except Exception:
            return ""
        h, rem = divmod(total, 3600)
        m, s = divmod(rem, 60)
        return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"

    def _nas_tree_columns(self, it: dict) -> tuple:
        duration = self.nas_duration_by_path.get(it.get("path") or "")
        return (self._fmt_size(it.get("size")), self._fmt_duration(duration), self._fmt_mtime(it.get("mtime")))

    def _drive_tree_columns(self, it: dict) -> tuple:
        return (
            self._fmt_size(it.get("size")),
            self._fmt_duration(it.get("duration_seconds")),
            self._fmt_drive_mtime(it.get("mtime")),
        )

    # ── Arbre de sélection générique (partagé NAS / Drive) ──────────────

    def _populate_source_tree(
        self, tree: ttk.Treeview, data_store: dict, parent_item_id: str,
        items: list[dict], extra_columns,
    ):
        """Remplit `tree` avec `items` (même forme que browse_nas/GDriveClient.browse).
        `extra_columns(item) -> tuple` fournit les colonnes size/durée/date,
        propres à chaque source."""
        if parent_item_id:
            for child in tree.get_children(parent_item_id):
                tree.delete(child)
                data_store.pop(child, None)
            data_store[parent_item_id]["loaded"] = True

        ordered = sorted(items, key=lambda d: (not d.get("isdir"), (d.get("name") or "").lower()))
        for row_index, it in enumerate(ordered):
            node_id = tree.insert(parent_item_id, "end", text="", values=extra_columns(it))
            data_store[node_id] = {
                "path": it.get("path"), "name": it.get("name") or "?",
                "isdir": bool(it.get("isdir")), "is_video": bool(it.get("is_video")),
                "checked": False, "loaded": False,
                "parity": "even" if row_index % 2 == 0 else "odd",
            }
            self._refresh_tree_item_label(tree, data_store, node_id)
            if it.get("isdir"):
                tree.insert(node_id, "end", text="   chargement…", values=("", "", ""))

    def _refresh_tree_item_label(self, tree: ttk.Treeview, data_store: dict, item_id: str):
        data = data_store.get(item_id)
        if not data:
            return
        selectable = data["isdir"] or data["is_video"]
        icon = "📁" if data["isdir"] else ("🎬" if data["is_video"] else "📄")
        image = (self.img_checked if data["checked"] else self.img_unchecked) if selectable else self.img_blank
        tags = ("checked",) if data["checked"] else (data.get("parity", "even"),)
        tree.item(item_id, text=f" {icon}  {data['name']}", image=image, tags=tags)

    def _on_source_tree_click(self, tree: ttk.Treeview, data_store: dict, event, on_toggled) -> str | None:
        # Seul un clic sur la CASE coche/décoche. La flèche ▸ déplie, le nom
        # sélectionne la ligne : plus de coche accidentelle en visant autre chose.
        if tree.identify("region", event.x, event.y) != "tree":
            return None
        element = tree.identify("element", event.x, event.y)
        if "image" not in element:
            return None
        item_id = tree.identify_row(event.y)
        data = data_store.get(item_id)
        if not data or not (data["isdir"] or data["is_video"]):
            return None  # fichier non vidéo : pas de case
        data["checked"] = not data["checked"]
        self._refresh_tree_item_label(tree, data_store, item_id)
        self._refresh_batch_count()
        on_toggled()
        return "break"  # ne pas déclencher en plus la sélection/dépliage

    def _select_all_source(self, tree: ttk.Treeview, data_store: dict, on_changed):
        for item_id, data in data_store.items():
            if data["isdir"] or data["is_video"]:
                data["checked"] = True
                self._refresh_tree_item_label(tree, data_store, item_id)
        self._refresh_batch_count()
        on_changed()

    def _select_none_source(self, tree: ttk.Treeview, data_store: dict, on_changed):
        for item_id, data in data_store.items():
            data["checked"] = False
            self._refresh_tree_item_label(tree, data_store, item_id)
        self._refresh_batch_count()
        on_changed()

    # ── NAS : wrappers concrets sur les helpers génériques ──────────────

    def _on_nas_tree_open(self, _event):
        item_id = self.nas_tree.focus()
        data = self.nas_tree_data.get(item_id)
        if not data or not data["isdir"] or data["loaded"]:
            return
        threading.Thread(target=self._load_nas_children_worker, args=(item_id, data["path"]), daemon=True).start()

    def _on_nas_tree_click(self, event):
        return self._on_source_tree_click(self.nas_tree, self.nas_tree_data, event, self._update_nas_checked_count)

    def _update_nas_checked_count(self):
        n = sum(1 for d in self.nas_tree_data.values() if d["checked"])
        checked_part = f"{n} élément(s) coché(s)" if n else "aucun élément coché"
        if self.nas_total_videos is not None:
            self.nas_status_var.set(f"📼 {self.nas_total_videos} vidéos au total sur le NAS  •  {checked_part}.")
        else:
            self.nas_status_var.set(f"{checked_part.capitalize()}.")

    def _select_all_nas(self):
        self._select_all_source(self.nas_tree, self.nas_tree_data, self._update_nas_checked_count)

    def _select_none_nas(self):
        self._select_none_source(self.nas_tree, self.nas_tree_data, self._update_nas_checked_count)

    def _get_checked_nas_nodes(self) -> list[dict]:
        """Sélection brute (pas encore résolue) : fichiers et dossiers cochés dans l'arbre.
        Les dossiers cochés sont développés en fichiers vidéo individuels au lancement du batch
        (voir _expand_nas_selection), pour ne pas bloquer l'UI avec des appels réseau ici."""
        return [
            {"nas_path": data["path"], "title": data["name"], "isdir": data["isdir"], "id": None, "source": "nas"}
            for data in self.nas_tree_data.values()
            if data["checked"]
        ]

    # ── Drive : wrappers concrets sur les helpers génériques ────────────

    def _on_drive_tree_open(self, _event):
        item_id = self.drive_tree.focus()
        data = self.drive_tree_data.get(item_id)
        if not data or not data["isdir"] or data["loaded"]:
            return
        threading.Thread(target=self._load_drive_children_worker, args=(item_id, data["path"]), daemon=True).start()

    def _on_drive_tree_click(self, event):
        return self._on_source_tree_click(self.drive_tree, self.drive_tree_data, event, self._update_drive_checked_count)

    def _update_drive_checked_count(self):
        n = sum(1 for d in self.drive_tree_data.values() if d["checked"])
        checked_part = f"{n} élément(s) coché(s)" if n else "aucun élément coché"
        self.drive_status_var.set(f"{checked_part.capitalize()}.")

    def _select_all_drive(self):
        self._select_all_source(self.drive_tree, self.drive_tree_data, self._update_drive_checked_count)

    def _select_none_drive(self):
        self._select_none_source(self.drive_tree, self.drive_tree_data, self._update_drive_checked_count)

    def _get_checked_drive_nodes(self) -> list[dict]:
        """Sélection brute (pas encore résolue) : fichiers et dossiers Drive cochés
        (voir _expand_drive_selection pour la résolution récursive au lancement)."""
        return [
            {"file_id": data["path"], "title": data["name"], "isdir": data["isdir"], "id": None, "source": "drive"}
            for data in self.drive_tree_data.values()
            if data["checked"]
        ]

    def _expand_nas_selection(self, selection: list[dict]) -> list[dict]:
        """Développe récursivement les dossiers cochés en fichiers vidéo individuels
        (appels réseau bloquants : à exécuter uniquement depuis le thread worker du pipeline).
        Déduplique par nas_path : cocher un dossier ET un fichier qu'il contient (ou deux
        dossiers imbriqués) ne produit qu'une seule transcription par vidéo.

        Un dossier dont l'exploration échoue (vagues de 502 quand le backend
        de l'API redémarre) n'est JAMAIS abandonné en silence : au-delà des
        retries du client, une seconde passe est faite après une pause, et
        s'il reste des échecs ils sont listés en avertissement — leurs vidéos
        seront reprises à la prochaine exécution grâce au saut automatique
        des médias déjà transcrits."""
        resolved: list[dict] = []
        seen_paths: set[str] = set()
        seen_dirs: set[str] = set()
        failed_dirs: list[str] = []

        def add(nas_path: str, title: str | None):
            if nas_path not in seen_paths:
                seen_paths.add(nas_path)
                resolved.append({"nas_path": nas_path, "title": title, "id": None, "source": "nas"})

        def walk(stack: list[str]):
            while stack:
                path = stack.pop()
                if path in seen_dirs:
                    continue
                seen_dirs.add(path)
                try:
                    result = self._require_nas_client().browse_nas(path)
                except Exception as exc:
                    self._log(f"[ERREUR] Exploration NAS ({path}) échouée : {exc}")
                    failed_dirs.append(path)
                    continue
                for it in result.get("items", []):
                    if it.get("isdir"):
                        stack.append(it["path"])
                    elif it.get("is_video"):
                        add(it["path"], it.get("name"))

        for node in selection:
            if not node["isdir"]:
                add(node["nas_path"], node["title"])
                continue
            self._log(f"[INFO] Exploration récursive NAS : {node['nas_path']}")
            walk([node["nas_path"]])

        if failed_dirs:
            self._log(
                f"[AVERTISSEMENT] {len(failed_dirs)} dossier(s) NAS inaccessibles (serveur API instable). "
                "Seconde passe dans 60s pour laisser le backend redémarrer…"
            )
            if not self._interruptible_sleep(60):
                retry_dirs, failed_dirs = failed_dirs, []
                for path in retry_dirs:
                    seen_dirs.discard(path)
                walk(retry_dirs)

        if failed_dirs:
            self._log(
                f"[AVERTISSEMENT] {len(failed_dirs)} dossier(s) NAS restés inaccessibles après la seconde passe — "
                "leurs vidéos ne sont PAS dans ce lot et seront reprises à la prochaine exécution :"
            )
            for path in failed_dirs:
                self._log(f"[AVERTISSEMENT]   • {path}")

        return resolved

    def _expand_drive_selection(self, selection: list[dict]) -> list[dict]:
        """Équivalent Drive de _expand_nas_selection : développe récursivement
        les dossiers cochés en fichiers vidéo individuels. Déduplique par
        file_id. Même résilience (seconde passe après pause sur les dossiers
        en échec) que côté NAS."""
        resolved: list[dict] = []
        seen_paths: set[str] = set()
        seen_dirs: set[str] = set()
        failed_dirs: list[str] = []

        def add(file_id: str, title: str | None):
            if file_id not in seen_paths:
                seen_paths.add(file_id)
                resolved.append({"file_id": file_id, "title": title, "id": None, "source": "drive"})

        def walk(stack: list[str]):
            while stack:
                folder_id = stack.pop()
                if folder_id in seen_dirs:
                    continue
                seen_dirs.add(folder_id)
                try:
                    result = self._require_gdrive_client().browse(folder_id)
                except Exception as exc:
                    self._log(f"[ERREUR] Exploration Drive ({folder_id}) échouée : {exc}")
                    failed_dirs.append(folder_id)
                    continue
                for it in result.get("items", []):
                    if it.get("isdir"):
                        stack.append(it["path"])
                    elif it.get("is_video"):
                        add(it["path"], it.get("name"))

        for node in selection:
            if not node["isdir"]:
                add(node["file_id"], node["title"])
                continue
            self._log(f"[INFO] Exploration récursive Drive : {node['title']} ({node['file_id']})")
            walk([node["file_id"]])

        if failed_dirs:
            self._log(
                f"[AVERTISSEMENT] {len(failed_dirs)} dossier(s) Drive inaccessibles. "
                "Seconde passe dans 60s…"
            )
            if not self._interruptible_sleep(60):
                retry_dirs, failed_dirs = failed_dirs, []
                for fid in retry_dirs:
                    seen_dirs.discard(fid)
                walk(retry_dirs)

        if failed_dirs:
            self._log(
                f"[AVERTISSEMENT] {len(failed_dirs)} dossier(s) Drive restés inaccessibles après la seconde passe — "
                "leurs vidéos ne sont PAS dans ce lot et seront reprises à la prochaine exécution :"
            )
            for fid in failed_dirs:
                self._log(f"[AVERTISSEMENT]   • {fid}")

        return resolved

    def _describe_config_status(self) -> str:
        cfg = self.local_config
        checks = [
            ("DB", bool(cfg["db_password"].strip())),
            ("HF token", bool(cfg["hf_token"].strip())),
            ("Env FW", Path(cfg["fw_python"]).exists()),
            ("Env WX", Path(cfg["wx_python"]).exists()),
            ("NAS API", bool(cfg["nas_username"].strip()) and bool(cfg["nas_password"].strip())),
            ("Drive API", bool(cfg["google_drive_api_key"].strip())),
        ]
        badges = "    ".join(f"{'✓' if ok else '✗'} {label}" for label, ok in checks)
        return f"{badges}    •    modifiable dans {LOCAL_CONFIG_PATH.name}"

    def _refresh_batch_count(self):
        if self.use_nas_var.get():
            n = len(self._get_checked_nas_nodes())
            self.batch_var.set(
                f"{n} élément(s) NAS coché(s) (résolus en vidéos au lancement)" if n else "Aucun élément NAS coché."
            )
            return
        if self.use_drive_var.get():
            n = len(self._get_checked_drive_nodes())
            self.batch_var.set(
                f"{n} élément(s) Drive coché(s) (résolus en vidéos au lancement)" if n else "Aucun élément Drive coché."
            )
            return
        files = collect_media_inputs(
            self.single_file_var.get(), self.multi_files,
            self.multi_folders, self.recursive_var.get(),
        )
        n = len(files)
        self.batch_var.set(f"{n} média(s) détecté(s)" if n else "Aucun média sélectionné.")

    # ── Journal ─────────────────────────────────────────────────────────

    MAX_LOG_LINES = 5000

    def _log_raw(self, message: str, tag: str = ""):
        self.log_text.configure(state="normal")
        self.log_text.insert("end", message + "\n", tag)
        line_count = int(self.log_text.index("end-1c").split(".")[0])
        if line_count > self.MAX_LOG_LINES:
            self.log_text.delete("1.0", f"{line_count - self.MAX_LOG_LINES + 1}.0")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _log(self, message: str):
        tag = (
            "error" if "[ERREUR" in message
            else "warn" if "[AVERTISSEMENT" in message
            else "ok" if "[OK]" in message
            else "info" if "[INFO]" in message
            else ""
        )
        self.queue.put(("log", (message, tag)))

    def _set_status(self, message: str):
        self.queue.put(("status", message))

    def _show_info(self, title: str, message: str):
        self.queue.put(("info", (title, message)))

    def _show_error(self, title: str, message: str):
        self.queue.put(("error", (title, message)))

    def _clear_log(self):
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")

    def _consume_queue(self):
        try:
            while True:
                kind, payload = self.queue.get_nowait()
                if kind == "log":
                    msg, tag = payload
                    self._log_raw(msg, tag)
                elif kind == "status":
                    self.status_var.set(payload)
                elif kind == "info":
                    messagebox.showinfo(*payload)
                elif kind == "error":
                    messagebox.showerror(*payload)
                elif kind == "done":
                    self.progress.stop()
                    self.progress.configure(mode="indeterminate", value=0)
                    self.stage_var.set("")
                    self.start_button.configure(state="normal")
                    self.pause_button.configure(state="disabled", text="⏸ Pause")
                    self.stop_button.configure(state="disabled")
                    self.pause_requested.clear()
                    self.stop_requested.clear()
                    self._current_proc = None
                elif kind == "running":
                    self.start_button.configure(state="disabled")
                    self.pause_button.configure(state="normal", text="⏸ Pause")
                    self.stop_button.configure(state="normal")
                    self.pause_requested.clear()
                    self.stop_requested.clear()
                    self.progress.configure(mode="indeterminate")
                    self.progress.start(10)
                    self.notebook.select(self.journal_tab)
                elif kind == "db_status":
                    ok, reason = payload
                    if ok:
                        self.db_status_var.set("🟢 Base de données connectée")
                        self.db_status_label.configure(style="HeaderOk.TLabel")
                    else:
                        self.db_status_var.set(f"🔴 Base de données : {reason}")
                        self.db_status_label.configure(style="HeaderBad.TLabel")
                elif kind == "stage_reset":
                    self.progress.stop()
                    self.progress.configure(mode="determinate", maximum=100, value=0)
                    self.stage_var.set(payload or "")
                elif kind == "stage_progress":
                    label, pct = payload
                    self.progress.configure(mode="determinate", maximum=100)
                    self.progress["value"] = pct
                    self.stage_var.set(f"{label} — {pct}%")
                elif kind == "nas_tree_root":
                    items, total_videos = payload
                    self.nas_tree.delete(*self.nas_tree.get_children())
                    self.nas_tree_data.clear()
                    self.nas_total_videos = total_videos
                    self._populate_source_tree(self.nas_tree, self.nas_tree_data, "", items, self._nas_tree_columns)
                    self._update_nas_checked_count()
                elif kind == "nas_tree_children":
                    item_id, items = payload
                    self._populate_source_tree(self.nas_tree, self.nas_tree_data, item_id, items, self._nas_tree_columns)
                elif kind == "drive_tree_root":
                    items = payload
                    self.drive_tree.delete(*self.drive_tree.get_children())
                    self.drive_tree_data.clear()
                    self._populate_source_tree(self.drive_tree, self.drive_tree_data, "", items, self._drive_tree_columns)
                    self._update_drive_checked_count()
                elif kind == "drive_tree_children":
                    item_id, items = payload
                    self._populate_source_tree(self.drive_tree, self.drive_tree_data, item_id, items, self._drive_tree_columns)
        except Empty:
            pass
        self.after(100, self._consume_queue)

    # ── Détection matériel ───────────────────────────────────────────────

    # ── Vérification DB au démarrage ─────────────────────────────────────

    def _check_db_clicked(self):
        threading.Thread(target=self._check_db_worker, daemon=True).start()

    def _check_db_worker(self):
        password = self.db_password_var.get().strip()
        if not password:
            self.queue.put(("db_status", (False, f"mot de passe manquant dans {LOCAL_CONFIG_PATH.name}")))
            self._log(f"[ERREUR] DB : 'db_password' est vide dans {LOCAL_CONFIG_PATH}.")
            return
        try:
            conn = connect_db(password)
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
            conn.close()
        except Exception as exc:
            reason = str(exc).strip().splitlines()[0][:120] or type(exc).__name__
            self.queue.put(("db_status", (False, reason)))
            self._log(f"[ERREUR] Connexion DB impossible : {exc}")
            return
        self.queue.put(("db_status", (True, "")))
        self._log("[OK] Connexion à la base de données vérifiée.")

    def _wait_if_paused(self):
        """Bloque le thread worker (pas le GUI) tant que la pause est active.
        À appeler entre deux fichiers ou avant de démarrer un nouveau sous-processus.
        Un arrêt demandé (Stop) lève immédiatement la pause."""
        if not self.pause_requested.is_set():
            return
        self._set_status("⏸ En pause…")
        while self.pause_requested.is_set() and not self.stop_requested.is_set():
            time.sleep(0.3)

    def _toggle_pause(self):
        if not self.pause_requested.is_set():
            self.pause_requested.set()
            self.pause_button.configure(text="▶ Reprendre")
            self._log("[INFO] ⏸ Pause demandée…")
            self._suspend_or_resume_current_proc(suspend=True)
        else:
            self.pause_requested.clear()
            self.pause_button.configure(text="⏸ Pause")
            self._log("[INFO] ▶ Reprise…")
            self._suspend_or_resume_current_proc(suspend=False)

    def _suspend_or_resume_current_proc(self, suspend: bool):
        proc = self._current_proc
        if proc is None or proc.poll() is not None:
            return  # rien en cours (entre deux fichiers/étapes) : le point de contrôle suffit
        try:
            ps_proc = psutil.Process(proc.pid)
            if suspend:
                ps_proc.suspend()
                self._log(f"[OK] Sous-processus (pid={proc.pid}) suspendu.")
            else:
                ps_proc.resume()
                self._log(f"[OK] Sous-processus (pid={proc.pid}) repris.")
        except Exception as exc:
            self._log(f"[ERREUR] Impossible de {'suspendre' if suspend else 'reprendre'} le sous-processus : {exc}")

    # ── Arrêt propre du batch ────────────────────────────────────────────

    def _request_stop(self):
        """Arrêt propre du batch : interrompt l'étape en cours (le sous-processus
        est tué), l'item courant est marqué en échec — il sera automatiquement
        repris à la prochaine exécution grâce au saut des médias déjà transcrits."""
        if not (self.worker_thread and self.worker_thread.is_alive()):
            return
        if not messagebox.askyesno(
            "Arrêter le batch",
            "Arrêter le traitement en cours ?\n\n"
            "L'étape en cours est interrompue immédiatement ; l'item courant sera "
            "marqué en échec et repris automatiquement à la prochaine exécution.",
        ):
            return
        self.stop_requested.set()
        self.stop_button.configure(state="disabled")
        self._log("[INFO] ⏹ Arrêt demandé — interruption de l'étape en cours…")
        # Une pause active bloquerait le worker (et laisserait le sous-processus
        # suspendu, donc jamais terminé) : on la lève d'abord.
        if self.pause_requested.is_set():
            self.pause_requested.clear()
            self.pause_button.configure(text="⏸ Pause")
            self._suspend_or_resume_current_proc(suspend=False)
        self._terminate_current_proc()

    def _terminate_current_proc(self):
        proc = self._current_proc
        if proc is None or proc.poll() is not None:
            return
        try:
            ps_proc = psutil.Process(proc.pid)
            # Le sous-processus peut avoir ses propres enfants (ffmpeg lancé par
            # whisperx.load_audio, etc.) : on tue tout l'arbre.
            for child in ps_proc.children(recursive=True):
                try:
                    child.kill()
                except Exception:
                    pass
            ps_proc.kill()
            self._log(f"[INFO] Sous-processus (pid={proc.pid}) arrêté.")
        except Exception as exc:
            self._log(f"[ERREUR] Impossible d'arrêter le sous-processus : {exc}")

    def _interruptible_sleep(self, seconds: float) -> bool:
        """Attend `seconds`, en se réveillant immédiatement si l'arrêt du batch
        est demandé. Retourne True si l'arrêt a été demandé."""
        return self.stop_requested.wait(timeout=seconds)

    def _detect_hardware_clicked(self):
        threading.Thread(target=self._detect_hardware_worker, daemon=True).start()

    def _detect_hardware_worker(self):
        self._set_status("Détection du matériel en cours…")
        fw_info = inspect_runtime(Path(self.fw_python_var.get().strip()))
        wx_info = inspect_runtime(Path(self.wx_python_var.get().strip()))

        def _fmt(info: dict, label: str) -> str:
            if not info.get("ok"):
                return f"{label}: ERREUR — {info.get('error')}"
            return (
                f"{label}: Python {info.get('version')} | "
                f"Torch {info.get('torch_version', 'N/A')} | "
                f"CUDA={info.get('cuda')} | GPU={info.get('cuda_name', '')}"
            )

        self._log("[INFO] " + _fmt(fw_info, "FW"))
        self._log("[INFO] " + _fmt(wx_info, "WX"))

        auto_device = choose_auto_device(fw_info, wx_info, self.use_alignment_var.get())
        auto_compute = choose_auto_compute_type(auto_device)
        self.hw_var.set(
            f"FW cuda={cuda_available(fw_info)} | WX cuda={cuda_available(wx_info)} | "
            f"device_auto={auto_device} compute={auto_compute}"
        )
        self._set_status("Détection terminée.")

    # ── Validation ───────────────────────────────────────────────────────

    def _validate_inputs(self) -> list:
        if self.use_nas_var.get():
            cfg = self.local_config
            if not cfg["nas_username"].strip() or not cfg["nas_password"].strip():
                raise RuntimeError(f"Identifiants NAS manquants. Renseigne 'nas_username'/'nas_password' dans {LOCAL_CONFIG_PATH}.")
            if self.nas_client is None:
                raise RuntimeError("Clique d'abord sur 'Parcourir le NAS'.")
            files = self._get_checked_nas_nodes()
            if not files:
                raise RuntimeError("Aucun fichier ni dossier NAS coché dans l'arborescence.")
        elif self.use_drive_var.get():
            cfg = self.local_config
            if not cfg["google_drive_api_key"].strip():
                raise RuntimeError(f"Clé API Google Drive manquante. Renseigne 'google_drive_api_key' dans {LOCAL_CONFIG_PATH}.")
            if self.gdrive_client is None:
                raise RuntimeError("Clique d'abord sur 'Parcourir Drive'.")
            files = self._get_checked_drive_nodes()
            if not files:
                raise RuntimeError("Aucun fichier ni dossier Drive coché dans l'arborescence.")
        else:
            files = collect_media_inputs(
                self.single_file_var.get(), self.multi_files,
                self.multi_folders, self.recursive_var.get(),
            )
            if not files:
                raise RuntimeError("Sélectionne au moins un fichier ou un dossier contenant des médias.")
        if not self.db_password_var.get().strip():
            raise RuntimeError(f"Le mot de passe DB est manquant. Renseigne 'db_password' dans {LOCAL_CONFIG_PATH}.")
        if not Path(self.fw_python_var.get().strip()).exists():
            raise RuntimeError(f"Chemin Faster-Whisper python.exe introuvable. Vérifie 'fw_python' dans {LOCAL_CONFIG_PATH}.")
        if self.use_alignment_var.get() and not Path(self.wx_python_var.get().strip()).exists():
            raise RuntimeError(f"Chemin WhisperX python.exe introuvable. Vérifie 'wx_python' dans {LOCAL_CONFIG_PATH}.")
        if self.diarization_var.get() and not self.hf_token_var.get().strip():
            raise RuntimeError(f"La diarisation requiert un HF Token. Renseigne 'hf_token' dans {LOCAL_CONFIG_PATH}.")
        return files

    # ── Pipeline ─────────────────────────────────────────────────────────

    def _start_pipeline(self):
        if self.worker_thread and self.worker_thread.is_alive():
            messagebox.showinfo("En cours", "Un traitement est déjà actif.")
            return
        try:
            media_files = self._validate_inputs()
        except Exception as exc:
            messagebox.showerror("Erreur de configuration", str(exc))
            return
        self._refresh_batch_count()
        self.queue.put(("running", None))
        self.worker_thread = threading.Thread(
            target=self._pipeline_worker, args=(media_files,), daemon=True
        )
        self.worker_thread.start()

    def _pipeline_worker(self, media_files: list):
        try:
            self._run_pipeline_batch(media_files)
        except Exception as exc:
            self._log(f"[ERREUR] {exc}")
            self._show_error("Erreur pipeline", str(exc))
        finally:
            self.queue.put(("done", None))

    # ── Téléchargement NAS + pré-téléchargement en arrière-plan ────────────

    def _download_nas_video(self, item: dict, work_dir: Path, nas_username: str, nas_password: str) -> Path:
        """Télécharge une vidéo NAS avec retries + reconnexion automatique
        (session NAS/API expirée, réponse JSON d'erreur relayée en 200,
        téléchargement tronqué, 502 du backend en redémarrage, fichier
        suspect…). Utilisée à la fois pour le téléchargement synchrone de
        l'item courant et pour le pré-téléchargement en arrière-plan de
        l'item suivant.

        Les attentes s'allongent fortement (5s → 60s) : les vagues de 502
        Bad Gateway observées correspondent à un backend d'API en cours de
        redémarrage, qui met couramment 30-60s à revenir — des retries à
        2-4s d'écart brûlaient toutes les tentatives pendant la panne."""
        max_attempts = 5
        backoff = [5, 15, 30, 60]

        def wait_and_relogin(attempt: int):
            if self._interruptible_sleep(backoff[min(attempt - 1, len(backoff) - 1)]):
                raise RuntimeError("Arrêt du batch demandé pendant les tentatives de téléchargement.")
            # La reconnexion rafraîchit la session NAS (utile si l'échec
            # venait d'une session expirée) ; si l'API est encore en panne,
            # elle échoue proprement et la tentative suivante tranchera.
            try:
                self._require_nas_client().login(nas_username, nas_password)
            except DOWNLOAD_RETRYABLE_ERRORS as login_exc:
                self._log(f"[AVERTISSEMENT] Reconnexion échouée : {login_exc}")

        for attempt in range(1, max_attempts + 1):
            try:
                video_path_str = self._require_nas_client().download_from_nas_path(item["nas_path"], dest_folder=str(work_dir))
            except DOWNLOAD_RETRYABLE_ERRORS as exc:
                if attempt == max_attempts:
                    raise
                self._log(f"[INFO] Téléchargement échoué (tentative {attempt}/{max_attempts}) : {exc}")
                self._log(f"[INFO] Nouvelle tentative dans {backoff[min(attempt - 1, len(backoff) - 1)]}s (reconnexion incluse)…")
                wait_and_relogin(attempt)
                continue
            candidate = Path(video_path_str)
            size = candidate.stat().st_size if candidate.exists() else 0
            if size >= 1024:
                return candidate
            # Fichier minuscule : réponse d'erreur écrite sur disque ou
            # raccourci — on retente après reconnexion.
            candidate.unlink(missing_ok=True)
            if attempt == max_attempts:
                raise ValueError(
                    f"Fichier téléchargé invalide ou vide (taille: {size} octets) "
                    f"après {max_attempts} tentatives avec reconnexion. "
                    "Le fichier sur le NAS est probablement corrompu ou un raccourci."
                )
            self._log(
                f"[INFO] Fichier téléchargé suspect ({size} octets, tentative {attempt}/{max_attempts}) — "
                f"reconnexion puis nouvelle tentative dans {backoff[min(attempt - 1, len(backoff) - 1)]}s…"
            )
            wait_and_relogin(attempt)
        raise RuntimeError("Téléchargement NAS : boucle de tentatives épuisée sans résultat.")  # inatteignable

    def _download_drive_video(self, item: dict, work_dir: Path) -> Path:
        """Télécharge une vidéo Google Drive avec retries. Pas de notion de
        session/reconnexion ici (auth par simple clé API, pas de session NAS
        à rafraîchir) : juste un backoff progressif sur les erreurs réseau,
        quota (429) ou fichier suspect."""
        max_attempts = 5
        backoff = [5, 15, 30, 60]
        client = self._require_gdrive_client()
        filename = item.get("title") or f"{item['file_id']}.mp4"
        if not Path(filename).suffix:
            filename += ".mp4"

        for attempt in range(1, max_attempts + 1):
            try:
                video_path_str = client.download_file(item["file_id"], filename, dest_folder=str(work_dir))
            except DRIVE_RETRYABLE_ERRORS as exc:
                if attempt == max_attempts:
                    raise
                wait = backoff[min(attempt - 1, len(backoff) - 1)]
                self._log(f"[INFO] Téléchargement Drive échoué (tentative {attempt}/{max_attempts}) : {exc}")
                self._log(f"[INFO] Nouvelle tentative dans {wait}s…")
                if self._interruptible_sleep(wait):
                    raise RuntimeError("Arrêt du batch demandé pendant les tentatives de téléchargement Drive.")
                continue
            candidate = Path(video_path_str)
            size = candidate.stat().st_size if candidate.exists() else 0
            if size >= 1024:
                return candidate
            candidate.unlink(missing_ok=True)
            if attempt == max_attempts:
                raise ValueError(
                    f"Fichier Drive téléchargé invalide ou vide (taille: {size} octets) "
                    f"après {max_attempts} tentatives."
                )
            wait = backoff[min(attempt - 1, len(backoff) - 1)]
            self._log(f"[INFO] Fichier Drive suspect ({size} octets, tentative {attempt}/{max_attempts}) — nouvelle tentative dans {wait}s…")
            if self._interruptible_sleep(wait):
                raise RuntimeError("Arrêt du batch demandé pendant les tentatives de téléchargement Drive.")
        raise RuntimeError("Téléchargement Drive : boucle de tentatives épuisée sans résultat.")  # inatteignable

    def _maybe_start_nas_prefetch(
        self, media_files: list, index: int, total: int,
        base: Path, nas_username: str, nas_password: str,
    ) -> None:
        """Programme, si pertinent, le téléchargement en arrière-plan de la
        prochaine vidéo NAS PENDANT que la transcription locale de l'item
        courant tourne (le sous-processus FW/WX sature le CPU/GPU, pas le
        réseau — le moment idéal pour superposer le téléchargement suivant).

        Le déclenchement réel est différé de NAS_PREFETCH_DELAY_SECONDS
        (5 min) après le début de cette transcription : la session DSM
        Synology expire par INACTIVITÉ (codes 106/119 observés en
        production, dans la même journée, dès qu'aucun appel /nas/* n'a
        lieu pendant la durée de la transcription) — ce n'est pas une durée
        de vie fixe qu'on peut calculer à l'avance (le token JWT, lui, dure
        24h et n'est pas en cause). Recontacter le NAS toutes les 5 min sert
        de keepalive, largement sous n'importe quel seuil d'inactivité
        plausible, et prépare en prime le fichier suivant à l'avance."""
        prefetch = self._nas_prefetch
        if prefetch.thread is not None:
            return  # un pré-téléchargement est déjà programmé/en cours pour le prochain item
        if index >= total:
            return  # dernier item du lot : rien à préchercher
        next_item = media_files[index]  # media_files est 0-indexé ; index (1-based) pointe déjà sur le suivant
        if not isinstance(next_item, dict) or next_item.get("source") == "drive":
            # Prochain item local ou Drive : rien à anticiper ici (Drive n'a pas
            # le problème de session qui motive ce keepalive NAS — cf. docstring).
            return

        prefetch.item = next_item

        def worker():
            if prefetch.cancel_event.wait(NAS_PREFETCH_DELAY_SECONDS):
                return  # annulé avant l'échéance : transcription déjà terminée, rien à faire
            with prefetch.lock:
                if prefetch.cancel_event.is_set():
                    return
                prefetch.started_download = True
            next_work_dir = Path(tempfile.mkdtemp(prefix="pipeline_gui_", dir=str(base)))
            try:
                self._log(
                    f"[INFO] Pré-téléchargement en arrière-plan (keepalive NAS, "
                    f"{NAS_PREFETCH_DELAY_SECONDS}s après le début de la transcription en cours) : "
                    f"{next_item.get('title') or next_item.get('nas_path')}…"
                )
                video_path = self._download_nas_video(next_item, next_work_dir, nas_username, nas_password)
                with prefetch.lock:
                    if prefetch.cancel_event.is_set():
                        # Lot arrêté pendant le téléchargement : personne ne
                        # consommera ce fichier, on le supprime nous-mêmes.
                        shutil.rmtree(next_work_dir, ignore_errors=True)
                        return
                    prefetch.work_dir = next_work_dir
                    prefetch.video_path = video_path
                self._log(
                    f"[OK] Pré-téléchargement terminé en arrière-plan : "
                    f"{video_path.name} ({video_path.stat().st_size} octets)"
                )
            except Exception as exc:
                prefetch.error = exc
                self._log(f"[AVERTISSEMENT] Pré-téléchargement en arrière-plan échoué (nouvelle tentative normale en son tour) : {exc}")
                shutil.rmtree(next_work_dir, ignore_errors=True)

        prefetch.thread = threading.Thread(target=worker, daemon=True)
        prefetch.thread.start()

    def _consume_nas_prefetch(self, item: dict) -> tuple[Path, Path] | None:
        """Récupère le résultat d'un pré-téléchargement lancé pendant l'item
        précédent, s'il correspond à `item`. Ne bloque JAMAIS sur un
        pré-téléchargement qui n'a pas encore commencé à télécharger
        (encore en attente du délai de NAS_PREFETCH_DELAY_SECONDS) : il est
        simplement annulé et l'item sera téléchargé normalement en son tour."""
        prefetch = self._nas_prefetch
        if prefetch.thread is None or prefetch.item is not item:
            return None
        with prefetch.lock:
            started = prefetch.started_download
            if not started:
                prefetch.cancel_event.set()
        if not started:
            self._nas_prefetch = _NasPrefetch()
            return None
        prefetch.thread.join()
        result = None
        if prefetch.error is None and prefetch.video_path is not None and prefetch.work_dir is not None:
            self._log(f"[INFO] Réutilisation du pré-téléchargement en arrière-plan : {prefetch.video_path.name}")
            result = (prefetch.work_dir, prefetch.video_path)
        self._nas_prefetch = _NasPrefetch()
        return result

    # ── Sous-processus FW/WX avec reprise sur crash natif ───────────────────

    def _run_subprocess_resilient(
        self, python_exe: Path, script_path: Path, config: dict, work_dir: Path,
        label: str, verbose: bool, progress_func, on_start, max_attempts: int = 2,
    ) -> dict[str, Path]:
        """Enveloppe run_subprocess() avec une reprise automatique : les
        sous-processus FW/WhisperX tournent sur CUDA et peuvent occasionnellement
        se terminer par un crash natif (ex: code 3221226505 = 0xC0000409
        STATUS_STACK_BUFFER_OVERRUN, souvent un plantage CUDA/pilote GPU
        transitoire) au lieu d'une exception Python propre. Un tel crash ne
        remonte aucune trace utile et n'est presque jamais reproductible :
        une nouvelle tentative repart avec un contexte CUDA neuf et réussit
        généralement. Sans cette reprise, la langue concernée était perdue
        définitivement (cf. fetch_already_transcribed, qui exclut désormais
        les médias avec une transcription en échec — mais mieux vaut ne pas
        en arriver là)."""
        for attempt in range(1, max_attempts + 1):
            try:
                return run_subprocess(python_exe, script_path, config, work_dir, label, self._log, verbose, progress_func, on_start)
            except Exception as exc:
                # Un Stop tue volontairement le sous-processus : ne pas le
                # relancer, laisser l'échec remonter pour clore l'item.
                if attempt == max_attempts or self.stop_requested.is_set():
                    raise
                self._log(f"[AVERTISSEMENT] Sous-processus {label} en échec (tentative {attempt}/{max_attempts}) : {exc}")
                self._log(f"[INFO] Nouvelle tentative du sous-processus {label} (crash probablement transitoire)…")
                if self._interruptible_sleep(5):
                    raise
        raise RuntimeError(f"Sous-processus {label} : boucle de tentatives épuisée sans résultat.")  # inatteignable

    def _run_pipeline_batch(self, media_files: list):
        """media_files : liste de Path (sources locales) ou de dict (vidéos NAS, via l'API NAS)."""
        fw_python = Path(self.fw_python_var.get().strip())
        wx_python = Path(self.wx_python_var.get().strip())
        use_alignment = self.use_alignment_var.get()
        align_cache_only = ALIGN_CACHE_ONLY
        diarization = self.diarization_var.get()
        condition_on_previous_text = CONDITION_ON_PREVIOUS_TEXT
        verbose_logs = VERBOSE_LOGS
        hf_token = self.hf_token_var.get().strip()
        local_model_dir = self.local_model_dir_var.get().strip()
        db_password = self.db_password_var.get().strip()
        multilang_enabled = True
        candidate_languages = MULTILANG_CANDIDATE_LANGUAGES
        nas_username = self.local_config["nas_username"]
        nas_password = self.local_config["nas_password"]

        fw_info = inspect_runtime(fw_python)
        wx_info = inspect_runtime(wx_python) if use_alignment else {"ok": True, "cuda": False}

        requested_device = AUTO_DEVICE
        fw_device, wx_device = choose_effective_devices(
            requested_device=requested_device,
            fw_info=fw_info,
            wx_info=wx_info,
            use_alignment=use_alignment,
        )

        compute_type = AUTO_COMPUTE_TYPE
        if compute_type == "auto":
            compute_type = choose_auto_compute_type(fw_device)

        self._log(f"[INFO] Device demandé: {requested_device}")
        self._log(f"[INFO] FW CUDA dispo: {cuda_available(fw_info)}")
        self._log(f"[INFO] WX CUDA dispo: {cuda_available(wx_info)}")
        self._log(f"[INFO] Device FW retenu: {fw_device}")
        self._log(f"[INFO] Device WX retenu: {wx_device if use_alignment else 'n/a'}")
        self._log(f"[INFO] Compute type retenu: {compute_type}")

        # Métadonnées d'environnement, capturées une fois par batch pour les runs
        run_hostname = socket.gethostname()
        gpu_name = fw_info.get("cuda_name") or wx_info.get("cuda_name") or None
        engine_info = {
            "fw": {k: fw_info.get(k) for k in ("python", "version", "torch_version", "cuda", "torch_cuda_version")},
            "wx": {k: wx_info.get(k) for k in ("python", "version", "torch_version", "cuda", "torch_cuda_version")},
        }

        base = runtime_dir()
        # Les scripts enfants sont maintenant des fichiers externes
        script_dir = Path(__file__).resolve().parent
        fw_script = script_dir / "fw_child_runner.py"
        wx_script = script_dir / "wx_child_runner.py"
        model_cache_dir = base / "hf_cache_fw"
        model_cache_dir.mkdir(parents=True, exist_ok=True)
        align_model_cache_dir = base / "hf_cache_wx_align"
        align_model_cache_dir.mkdir(parents=True, exist_ok=True)
        # Pour le mode offline, on peut mettre align_model_cache_only à True
        # après un pré-téléchargement manuel.

        conn = None
        ok_count = 0
        fail_count = 0
        # Disjoncteur : après plusieurs échecs de téléchargement consécutifs,
        # le backend de l'API est très probablement en panne prolongée (vagues
        # de 502) — on marque une longue pause au lieu de brûler ~2 min de
        # retries par item sur des centaines d'items voués au même échec.
        consecutive_download_failures = 0

        def progress_func(label: str, pct: int) -> None:
            self.queue.put(("stage_progress", (label, pct)))

        def on_proc_start(proc: subprocess.Popen) -> None:
            self._current_proc = proc
            if self.pause_requested.is_set():
                self._suspend_or_resume_current_proc(suspend=True)

        try:
            self._set_status("Connexion à la base de données…")
            conn = connect_db(db_password)
            self._log("[OK] Connexion DB établie.")

            if media_files and isinstance(media_files[0], dict) and "isdir" in media_files[0]:
                source = media_files[0].get("source", "nas")
                self._set_status(f"Résolution de la sélection {source.upper()} (dossiers → fichiers)…")
                media_files = (
                    self._expand_drive_selection(media_files) if source == "drive"
                    else self._expand_nas_selection(media_files)
                )
                self._log(f"[INFO] {len(media_files)} fichier(s) vidéo résolu(s) depuis la sélection {source.upper()}.")

                # Saut automatique de ce qui a déjà une transcription terminée :
                # on peut cocher tout le NAS/Drive, seul le restant sera traité.
                # Les items Drive sont enregistrés en base sous un pseudo-chemin
                # "gdrive://<file_id>" (cf. upsert_media plus bas) : c'est cette
                # même clé qu'il faut comparer ici pour le dédoublonnage.
                def _dedup_key(v: dict) -> str | None:
                    if v.get("source") == "drive":
                        return f"gdrive://{v['file_id']}" if v.get("file_id") else None
                    return v.get("nas_path")

                done_ids, done_paths = fetch_already_transcribed(conn)
                before = len(media_files)
                media_files = [
                    v for v in media_files
                    if v.get("id") not in done_ids and _dedup_key(v) not in done_paths
                ]
                skipped = before - len(media_files)
                if skipped:
                    self._log(f"[INFO] {skipped} vidéo(s) déjà transcrite(s) — sautée(s). Restant : {len(media_files)}.")

                if not media_files:
                    raise RuntimeError(
                        "Rien à traiter : aucune vidéo dans la sélection, ou toutes déjà transcrites."
                    )

            total = len(media_files)
            self._nas_prefetch = _NasPrefetch()

            for index, item in enumerate(media_files, start=1):
                if self.stop_requested.is_set():
                    self._log(f"[INFO] ⏹ Batch interrompu par l'utilisateur — {index - 1}/{total} item(s) parcourus.")
                    break
                self._wait_if_paused()
                is_nas = isinstance(item, dict)
                is_drive = is_nas and item.get("source") == "drive"
                is_nas = is_nas and not is_drive  # dicts NAS et Drive partagent le même isinstance()
                is_remote = is_nas or is_drive
                display_name = (
                    (item.get("title") or f"nas#{item.get('id')}") if is_nas
                    else (item.get("title") or f"drive#{item.get('file_id')}") if is_drive
                    else item.name
                )

                self._set_status(f"[{index}/{total}] {display_name}")
                self._log(f"[INFO] ═══ [{index}/{total}] {display_name} ═══")
                self.queue.put(("stage_reset", f"[{index}/{total}] Démarrage…"))

                # Titre : celui saisi dans l'UI, sinon le nom du fichier SANS
                # extension (les items distants portent le nom complet "X.mp4").
                raw_remote_title = item.get("title") if is_remote else None
                remote_title = Path(str(raw_remote_title)).stem if raw_remote_title else None
                title = self.title_var.get().strip() or (remote_title if is_remote else item.stem) or display_name
                language = None  # Toujours auto : la détection multi-langue fr/en est automatique.
                owner_id = 1  # Hardcodé pour simplifier l'UI
                privacy = self.privacy_var.get().strip() or "PRIVATE"
                model_name = self.model_var.get().strip() or "small"
                beam_size = BEAM_SIZE
                min_silence = MIN_SILENCE_DURATION_MS
                max_words = MAX_WORDS_PER_SEGMENT
                max_chars = MAX_CHARS_PER_SEGMENT
                prefetched = self._consume_nas_prefetch(item) if is_nas else None
                prefetched_work_dir, prefetched_video_path = prefetched if prefetched else (None, None)
                work_dir = prefetched_work_dir or Path(tempfile.mkdtemp(prefix="pipeline_gui_", dir=str(base)))
                media_id = None
                run_id = None
                download_started_at = None
                download_finished_at = None
                languages_processed: list[str] = []
                lang_ok_count = 0
                lang_fail_count = 0

                try:
                    conn = ensure_connection(conn, db_password)

                    if is_nas:
                        download_started_at = now_dt()
                        if prefetched_video_path is not None:
                            # Déjà téléchargée en arrière-plan pendant la
                            # transcription locale de l'item précédent.
                            video_path = prefetched_video_path
                            self._log(
                                f"[OK] Vidéo NAS déjà disponible (pré-téléchargée en arrière-plan) : "
                                f"{video_path.name} ({video_path.stat().st_size} octets)"
                            )
                        else:
                            self._log(f"[INFO] Téléchargement vidéo NAS (id={item['id']}, nas_path={item['nas_path']}) via l'API…")
                            # La session NAS côté serveur (et le jeton API, valable
                            # ~24 h) expirent forcément pendant un lot de plusieurs
                            # milliers de vidéos : quel que soit l'échec (401, JSON
                            # d'erreur relayé en 200, téléchargement tronqué), on se
                            # reconnecte et on réessaie avant de déclarer l'item en
                            # erreur — sinon tout le reste du lot échoue en cascade.
                            video_path = self._download_nas_video(item, work_dir, nas_username, nas_password)
                        download_finished_at = now_dt()
                        consecutive_download_failures = 0

                        ffmeta = ffprobe_metadata(video_path)
                        dl_seconds = (download_finished_at - download_started_at).total_seconds()
                        self._log(
                            f"[INFO] Vidéo NAS téléchargée en {dl_seconds:.0f}s: "
                            f"duration={ffmeta.get('duration') or item.get('duration')} | "
                            f"size={video_path.stat().st_size} | "
                            f"codecs={ffmeta.get('video_codec')}/{ffmeta.get('audio_codec')}"
                        )
                        if item.get("id") is not None:
                            # Vidéo déjà cataloguée côté NAS (id connu via l'API) : ne pas
                            # écraser les métadonnées réelles (owner_id/privacy...) déjà en base.
                            media_id = item["id"]
                            self._log(f"[OK] media_id={media_id} (déjà catalogué côté NAS)")
                        else:
                            # Fichier trouvé par exploration brute (/nas/browse), pas encore
                            # catalogué : on l'enregistre avec son vrai chemin NAS (pas le
                            # chemin temporaire local du téléchargement).
                            media_id, relocated_from = upsert_media(
                                conn, video_path, title, owner_id, privacy, ffmeta,
                                nas_path_override=item["nas_path"],
                            )
                            if relocated_from:
                                self._log(f"[OK] media_id={media_id} (retrouvé — déplacé depuis {relocated_from})")
                            else:
                                self._log(f"[OK] media_id={media_id} (nouveau, catalogué depuis l'explorateur NAS)")
                    elif is_drive:
                        download_started_at = now_dt()
                        self._log(f"[INFO] Téléchargement vidéo Drive ({item.get('title')}, id={item['file_id']}) via l'API…")
                        video_path = self._download_drive_video(item, work_dir)
                        download_finished_at = now_dt()
                        consecutive_download_failures = 0

                        ffmeta = ffprobe_metadata(video_path)
                        dl_seconds = (download_finished_at - download_started_at).total_seconds()
                        self._log(
                            f"[INFO] Vidéo Drive téléchargée en {dl_seconds:.0f}s: "
                            f"duration={ffmeta.get('duration')} | "
                            f"size={video_path.stat().st_size} | "
                            f"codecs={ffmeta.get('video_codec')}/{ffmeta.get('audio_codec')}"
                        )
                        # Identifié par un pseudo-chemin "gdrive://<file_id>" stable
                        # (dédoublonnage aux prochains passages). Si un média du même
                        # nom+taille existe déjà à un AUTRE emplacement (ex: déjà transcrit
                        # depuis le NAS puis retrouvé sur Drive, ou l'inverse plus tard),
                        # upsert_media le reconnaît au lieu de dupliquer.
                        media_id, relocated_from = upsert_media(
                            conn, video_path, title, owner_id, privacy, ffmeta,
                            nas_path_override=f"gdrive://{item['file_id']}",
                        )
                        if relocated_from:
                            self._log(f"[OK] media_id={media_id} (retrouvé — déplacé depuis {relocated_from})")
                        else:
                            self._log(f"[OK] media_id={media_id} (nouveau, catalogué depuis Google Drive)")
                    else:
                        video_path = item
                        ffmeta = ffprobe_metadata(video_path)
                        self._log(
                            f"[INFO] Métadonnées vidéo: duration={ffmeta.get('duration')} | "
                            f"size={video_path.stat().st_size} | "
                            f"codecs={ffmeta.get('video_codec')}/{ffmeta.get('audio_codec')}"
                        )
                        media_id, relocated_from = upsert_media(conn, video_path, title, owner_id, privacy, ffmeta)
                        if relocated_from:
                            self._log(f"[OK] media_id={media_id} (retrouvé — déplacé depuis {relocated_from})")
                        else:
                            self._log(f"[OK] media_id={media_id}")

                    # Ouverture du run : trace complète du passage (source, copie NAS/Drive,
                    # sondage média, machine) dans le schéma transcription.
                    if is_nas:
                        source_kind, source_path = "nas", item["nas_path"]
                    elif is_drive:
                        source_kind, source_path = "drive", f"gdrive://{item['file_id']}"
                    else:
                        source_kind, source_path = "local", str(video_path)
                    run_id = create_run(conn, media_id, {
                        "source_kind": source_kind,
                        "source_nas_path": source_path,
                        "source_filesize": video_path.stat().st_size,
                        "download_started_at": download_started_at,
                        "download_finished_at": download_finished_at,
                        "download_bytes": video_path.stat().st_size if is_remote else None,
                        "media_duration_seconds": ffmeta.get("duration") or (item.get("duration") if is_remote else None),
                        "video_codec": ffmeta.get("video_codec"),
                        "audio_codec": ffmeta.get("audio_codec"),
                        "width": ffmeta.get("width"),
                        "height": ffmeta.get("height"),
                        "bitrate": ffmeta.get("bitrate"),
                        "format_name": ffmeta.get("format_name"),
                        "frame_rate": ffmeta.get("frame_rate"),
                        "audio_sample_rate": ffmeta.get("audio_sample_rate"),
                        "audio_channels": ffmeta.get("audio_channels"),
                        "stream_count": ffmeta.get("stream_count"),
                        "container_tags": Json(ffmeta.get("tags")) if ffmeta.get("tags") else None,
                        "hostname": run_hostname,
                        "device": fw_device,
                        "compute_type": compute_type,
                        "gpu_name": gpu_name,
                        "model_name": model_name,
                        "pipeline_version": APP_VERSION,
                        "engine_info": Json(engine_info),
                    })
                    self._log(f"[OK] run_id={run_id}")

                    fw_cfg = {
                        "source_file": str(video_path),
                        "output_dir": str(work_dir),
                        "model_name": model_name,
                        "language": language,
                        "beam_size": beam_size,
                        "min_silence_duration_ms": min_silence,
                        "max_words_per_segment": max_words,
                        "max_chars_per_segment": max_chars,
                        "device": fw_device,
                        "compute_type": compute_type,
                        "condition_on_previous_text": condition_on_previous_text,
                        "started_at": now_iso(),
                        "hf_token": hf_token,
                        "model_cache_dir": str(model_cache_dir),
                        "local_model_dir": local_model_dir,
                        "multilang_enabled": multilang_enabled,
                        "candidate_languages": candidate_languages,
                    }
                    self._wait_if_paused()
                    # La transcription locale qui suit sature le CPU/GPU pendant
                    # potentiellement plusieurs minutes sans toucher au réseau :
                    # c'est la fenêtre idéale pour pré-télécharger la prochaine
                    # vidéo NAS en arrière-plan (cf. _maybe_start_nas_prefetch).
                    self._maybe_start_nas_prefetch(media_files, index, total, base, nas_username, nas_password)
                    fw_results = self._run_subprocess_resilient(fw_python, fw_script, fw_cfg, work_dir, "FW", verbose_logs, progress_func, on_proc_start)
                    languages_processed = sorted(fw_results.keys())
                    self._log(f"[INFO] Langues transcrites intégralement: {languages_processed}")

                    conn = ensure_connection(conn, db_password)
                    patch_media_duration(conn, media_id, ffmeta.get("duration") or (item.get("duration") if is_nas else None))

                    # Si ce média a déjà été (partiellement) traité — par ex. un
                    # crash WhisperX avait fait échouer une langue lors d'un
                    # précédent passage — on ne refait PAS l'alignement (l'étape
                    # GPU lourde et la plus sujette aux crashs natifs) pour les
                    # langues déjà réussies : seules les langues manquantes/en
                    # échec sont retraitées.
                    existing_lang_status = fetch_media_language_status(conn, media_id)

                    # Chaque langue devient une transcription indépendante et complète
                    # (sa propre ligne DB, son propre alignement WhisperX si activé).
                    for lang in languages_processed:
                        if existing_lang_status.get(lang) == "done":
                            self._log(f"[INFO] {video_path.name} [{lang}] — déjà transcrit avec succès, pas de nouvel alignement.")
                            lang_ok_count += 1
                            continue
                        transcript_id = None
                        try:
                            fw_json = fw_results[lang]
                            fw_bundle = load_output_bundle(work_dir, fw_json, f"fw_transcript_{lang}")

                            fw_json_data = fw_bundle["json"]
                            primary_language = fw_json_data.get("primary_language") or fw_json_data.get("language")
                            is_primary = (lang == primary_language)

                            conn = ensure_connection(conn, db_password)
                            transcript_id = create_transcript(
                                conn, run_id, media_id, lang,
                                is_primary=is_primary,
                                language_probability=fw_json_data.get("language_probability") if is_primary else None,
                            )
                            self._log(f"[OK] transcript_id={transcript_id} status=processing (lang={lang})")

                            if use_alignment:
                                wx_cfg = {
                                    "source_file": str(video_path),
                                    "output_dir": str(work_dir),
                                    "fw_json": str(fw_json),
                                    "language": lang,
                                    "device": wx_device,
                                    "diarization": diarization,
                                    "hf_token": hf_token,
                                    "align_model_cache_dir": str(align_model_cache_dir),
                                    "align_model_cache_only": align_cache_only,
                                    "started_at": now_iso(),
                                }
                                self._wait_if_paused()
                                wx_results = self._run_subprocess_resilient(wx_python, wx_script, wx_cfg, work_dir, "WX", verbose_logs, progress_func, on_proc_start)
                                wx_json = wx_results.get(lang) or next(iter(wx_results.values()))
                                bundle = load_output_bundle(work_dir, wx_json, f"wx_aligned_{lang}")
                                engine_name = "whisperx"
                            else:
                                bundle = fw_bundle
                                engine_name = "faster-whisper"

                            # Le JSON des sous-processus ne porte que des chemins locaux
                            # éphémères (work_dir temporaire, supprimé après coup) : on y
                            # ajoute l'identité durable de la vidéo — comme les métadonnées
                            # de titre qu'on retrouve en streaming — pour qu'un export de
                            # transcription reste identifiable une fois isolé de la base.
                            bundle["json"]["original_filename"] = video_path.name
                            bundle["json"]["media_title"] = title
                            bundle["json"]["media_source"] = source_path

                            conn = ensure_connection(conn, db_password)
                            quality = {k: fw_json_data.get(k) for k in
                                       ("word_count", "avg_confidence", "avg_no_speech_prob", "speech_seconds")}
                            finish_transcript_done(
                                conn, transcript_id, bundle, engine_name,
                                aligned=use_alignment, diarized=diarization, quality=quality,
                            )
                            # L'enregistrement des chemins de fichiers locaux est désactivé car il prête à confusion.
                            # Le contenu des transcriptions est déjà dans la table `transcripts`.
                            # register_transcript_files(conn, transcript_id, bundle["specs"])

                            self._log(
                                f"[OK] {video_path.name} [{lang}] — segments={bundle.get('segment_count')} "
                                f"— transcript_id={transcript_id}"
                            )
                            lang_ok_count += 1

                        except Exception as lang_exc:
                            lang_fail_count += 1
                            rollback_safely(conn)
                            self._log(f"[ERREUR] {video_path.name} [{lang}] → {lang_exc}")
                            if transcript_id is not None:
                                try:
                                    conn = ensure_connection(conn, db_password)
                                    finish_transcript_failed(conn, transcript_id, str(lang_exc))
                                    self._log(f"[OK] Statut failed enregistré (lang={lang}).")
                                except Exception as inner_exc:
                                    self._log(f"[ERREUR] Échec finish_transcript_failed (lang={lang}): {inner_exc}")

                    manifest = {
                        "media_id": media_id,
                        "source_file": str(video_path),
                        "work_dir": str(work_dir),
                        "languages_processed": languages_processed,
                        "fw_device": fw_device,
                        "wx_device": wx_device if use_alignment else None,
                        "compute_type": compute_type,
                        "model": model_name,
                        "alignment": use_alignment,
                        "diarization": diarization,
                        "finished_at": now_iso(),
                    }
                    (work_dir / "db_manifest.json").write_text(
                        json.dumps(manifest, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )

                    conn = ensure_connection(conn, db_password)
                    if lang_fail_count == 0 and lang_ok_count > 0:
                        ok_count += 1
                        finish_run(conn, run_id, "done")
                        self._log(f"[OK] {video_path.name} — {lang_ok_count} langue(s) traitée(s) avec succès.")
                    else:
                        fail_count += 1
                        finish_run(conn, run_id, "partial" if lang_ok_count > 0 else "failed",
                                   f"{lang_fail_count} langue(s) en échec")
                        self._log(
                            f"[ERREUR] {video_path.name} — {lang_ok_count} succès / {lang_fail_count} échec(s) "
                            "parmi les langues traitées."
                        )

                except Exception as exc:
                    fail_count += 1
                    rollback_safely(conn)
                    self._log(f"[ERREUR] {display_name} → {exc}")
                    if run_id is not None:
                        try:
                            conn = ensure_connection(conn, db_password)
                            finish_run(conn, run_id, "failed", str(exc)[:5000])
                            self._log("[OK] Run marqué failed.")
                        except Exception as inner_exc:
                            self._log(f"[ERREUR] Échec finish_run: {inner_exc}")

                    # download_finished_at reste None quand l'échec a eu lieu
                    # pendant le téléchargement lui-même : c'est le signal du
                    # disjoncteur anti-panne-serveur.
                    if is_nas and download_finished_at is None:
                        consecutive_download_failures += 1
                        if consecutive_download_failures >= 3:
                            self._log(
                                f"[AVERTISSEMENT] {consecutive_download_failures} échecs de téléchargement consécutifs — "
                                "le serveur API semble en panne prolongée. Pause de 5 minutes avant de poursuivre le lot…"
                            )
                            if not self._interruptible_sleep(300):
                                try:
                                    self._require_nas_client().login(nas_username, nas_password)
                                    self._log("[OK] Reconnexion à l'API réussie après la pause — reprise du lot.")
                                except DOWNLOAD_RETRYABLE_ERRORS as login_exc:
                                    self._log(f"[AVERTISSEMENT] API toujours indisponible après la pause : {login_exc}")
                            consecutive_download_failures = 0

                finally:
                    # Le dossier de travail est supprimé après chaque média, sauf si l'option
                    # de conservation est cochée. Cela s'applique en cas de succès comme d'échec.
                    if work_dir.exists() and not self.keep_work_dirs_var.get():
                        try:
                            shutil.rmtree(work_dir)
                            self._log(f"[INFO] Dossier de travail temporaire supprimé : {work_dir.name}")
                        except Exception as e:
                            self._log(f"[ERREUR] Impossible de supprimer le dossier de travail {work_dir.name}: {e}")
                    elif work_dir.exists():
                        self._log(f"[INFO] Dossier de travail conservé pour inspection : {work_dir}")

            # Nettoyage d'un pré-téléchargement encore en attente ou déjà
            # terminé mais jamais consommé (arrêt anticipé du lot) : sans ça,
            # son dossier temporaire resterait orphelin dans runtime_db_gui.
            prefetch = self._nas_prefetch
            prefetch.cancel_event.set()
            with prefetch.lock:
                if prefetch.work_dir is not None:
                    shutil.rmtree(prefetch.work_dir, ignore_errors=True)
                    prefetch.work_dir = None
                    prefetch.video_path = None

            if self.stop_requested.is_set():
                self._set_status("Batch interrompu.")
                self._log(f"[INFO] Résultat (interrompu) — fichiers: succès={ok_count} échecs={fail_count}")
                self._show_info(
                    "Traitement interrompu",
                    f"Batch interrompu.\n\n✅ Succès : {ok_count}\n❌ Échecs : {fail_count}\n\n"
                    "Les vidéos restantes seront reprises automatiquement à la prochaine exécution.",
                )
            else:
                self._set_status("Batch terminé.")
                self._log(f"[INFO] Résultat final — fichiers: succès={ok_count} échecs={fail_count}")
                self._show_info(
                    "Traitement terminé",
                    f"Batch terminé.\n\n✅ Succès : {ok_count}\n❌ Échecs : {fail_count}",
                )

        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass

# ──────────────────────────── POINT D'ENTRÉE ──────────────────────────────

if __name__ == "__main__":
    app = App()
    app.mainloop()