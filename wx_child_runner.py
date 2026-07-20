from __future__ import annotations

import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

# HF_HUB_OFFLINE forcé, et positionné AVANT le premier import de
# huggingface_hub (transitif via whisperx) : sa valeur est lue une seule fois
# à l'import et figée dans des constantes de module — la repositionner plus
# tard dans le même processus n'a aucun effet (vérifié empiriquement).
#
# Sur cette machine, un logiciel de sécurité local (Avast/aswMonFltProxy)
# coupe la plupart des connexions HTTPS sortantes des interpréteurs Python
# de venv vers huggingface.co (SSL_CERT_VERIFY_FAILED / connexion
# réinitialisée), de façon intermittente et imprévisible — une sonde réseau
# préalable donne des faux positifs/négatifs et n'est pas fiable. Les
# modèles nécessaires (alignement fr/en + diarisation) sont déjà tous
# pré-téléchargés dans le cache local (cf. prefetch_diarization_models),
# donc le mode hors-ligne fonctionne toujours ici et évite en prime de
# perdre ~90s en tentatives réseau vouées à l'échec à chaque sous-processus.
os.environ["HF_HUB_OFFLINE"] = "1"

import types

import pandas as pd  # type: ignore
import torch  # type: ignore
import whisperx  # type: ignore
from whisperx.diarize import DiarizationPipeline  # type: ignore
from whisperx.audio import SAMPLE_RATE  # type: ignore


def normalize_segments(segments: list[dict]) -> list[dict]:
    out = []
    for seg in segments:
        item = dict(seg)
        item["start"] = float(item.get("start", 0.0))
        item["end"] = float(item.get("end", 0.0))
        item["text"] = str(item.get("text", "")).strip()
        item["words"] = list(item.get("words", []) or [])
        out.append(item)
    return out

def _ts_ms(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"

def srt_ts(s): return _ts_ms(s).replace(".", ",")
def vtt_ts(s): return _ts_ms(s)
def now_iso(): return datetime.now().isoformat(timespec="seconds")

def load_align_model_safe(lang: str, device: str, cache_dir: Path, cache_only: bool):
    try:
        return whisperx.load_align_model(language_code=lang, device=device, model_dir=str(cache_dir), model_cache_only=cache_only)
    except Exception as exc:
        if cache_only:
            raise RuntimeError(f"Modèle d'alignement absent en cache pour langue={lang}. Cache={cache_dir}. Pré-télécharge les modèles puis relance.") from exc
        raise RuntimeError(f"Échec téléchargement modèle d'alignement langue={lang}. Cache={cache_dir}. Pré-télécharge les modèles puis relance en cache_only=True. Détail={exc}") from exc

def load_diarization_pipeline_safe(hf_token: str | None, device: str, cache_dir: Path):
    """Charge le pipeline de diarisation pyannote, entièrement depuis le cache
    local (HF_HUB_OFFLINE=1 est forcé en tête de fichier, avant l'import de
    whisperx — voir le commentaire là-bas)."""
    try:
        return DiarizationPipeline(token=hf_token, device=device, cache_dir=cache_dir)
    except Exception as exc:
        raise RuntimeError(
            f"Échec chargement pipeline de diarisation. Cache={cache_dir}. "
            f"Si hors-ligne : pré-télécharge les modèles avec prefetch_diarization_models "
            f"puis relance. Détail={exc}"
        ) from exc

def run_diarization_safe(diarize_model: DiarizationPipeline, audio) -> pd.DataFrame:
    """Reproduit whisperx.diarize.DiarizationPipeline.__call__ en corrigeant
    une incompatibilité avec pyannote.audio>=4 (constatée ici en 4.0.6) :
    pyannote.audio.core.pipeline.Pipeline.__call__ contient un `yield` dans
    sa branche "liste de fichiers", ce qui — règle Python — en fait une
    fonction génératrice pour TOUS ses chemins d'exécution, y compris
    l'appel simple fichier (qui `return`). Résultat : self.model(audio_data)
    renvoie toujours un générateur, dont la vraie valeur (pour un fichier
    seul) n'est récupérable que via StopIteration.value puisqu'aucun `yield`
    n'est atteint. whisperx 3.8.6 ne le gère pas et déréférence directement
    le générateur, d'où `AttributeError: 'generator' object has no
    attribute 'speaker_diarization'`. On réplique donc l'appel ici avec le
    dépliage correct plutôt que d'utiliser diarize_model(audio)."""
    audio_data = {
        "waveform": torch.from_numpy(audio[None, :]),
        "sample_rate": SAMPLE_RATE,
    }
    result = diarize_model.model(audio_data, num_speakers=None, min_speakers=None, max_speakers=None)
    if isinstance(result, types.GeneratorType):
        try:
            next(result)
        except StopIteration as stop:
            output = stop.value
        else:
            raise RuntimeError("Sortie inattendue (multi-fichiers) du pipeline de diarisation pyannote.")
    else:
        output = result

    diarization = output.speaker_diarization
    diarize_df = pd.DataFrame(diarization.itertracks(yield_label=True), columns=["segment", "label", "speaker"])
    diarize_df["start"] = diarize_df["segment"].apply(lambda x: x.start)
    diarize_df["end"] = diarize_df["segment"].apply(lambda x: x.end)
    return diarize_df

def segment_text(seg: dict) -> str:
    text = str(seg.get("text", "")).strip()
    speaker = seg.get("speaker")
    return f"[{speaker}] {text}" if speaker and text else text

def main(config_path: str):
    try:
        cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))

        if cfg.get("hf_token"):
            os.environ["HF_TOKEN"] = cfg["hf_token"]
            os.environ["HUGGINGFACE_HUB_TOKEN"] = cfg["hf_token"]

        source = Path(cfg["source_file"])
        out_dir = Path(cfg["output_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)

        fw_data = json.loads(Path(str(cfg["fw_json"])).read_text(encoding="utf-8"))
        raw_segments = normalize_segments(fw_data.get("segments", []))
        lang = str(cfg.get("language") or fw_data.get("language") or "fr")

        requested_device = cfg.get("device", "cpu")
        torch_cuda = bool(torch.cuda.is_available())
        effective_device = "cuda" if requested_device == "cuda" and torch_cuda else "cpu"

        align_model_cache_dir = Path(cfg.get("align_model_cache_dir") or (out_dir / "align_model_cache"))
        align_model_cache_dir.mkdir(parents=True, exist_ok=True)
        align_model_cache_only = bool(cfg.get("align_model_cache_only", False))

        os.environ.setdefault("HF_HOME", str(align_model_cache_dir))
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(align_model_cache_dir))
        os.environ.setdefault("TRANSFORMERS_CACHE", str(align_model_cache_dir))
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

        t0 = time.time()
        print(f"INFO|Alignement WhisperX langue={lang} requested_device={requested_device} effective_device={effective_device} torch_cuda={torch_cuda}", flush=True)
        print(f"INFO|Align cache={align_model_cache_dir}", flush=True)
        print(f"INFO|Align cache_only={align_model_cache_only}", flush=True)

        print(f"PROGRESS|{lang}|5", flush=True)
        audio = whisperx.load_audio(str(source))

        print(f"INFO|Chargement align model langue={lang}", flush=True)
        align_model, metadata = load_align_model_safe(
            lang=lang, device=effective_device, cache_dir=align_model_cache_dir, cache_only=align_model_cache_only
        )
        print(f"PROGRESS|{lang}|25", flush=True)
        aligned = whisperx.align(raw_segments, align_model, metadata, audio, effective_device)
        aligned_segments = normalize_segments(aligned.get("segments", []))
        for seg in aligned_segments:
            seg["language"] = lang
        aligned_segments.sort(key=lambda s: (float(s.get("start", 0.0)), float(s.get("end", 0.0))))
        print(f"PROGRESS|{lang}|70", flush=True)

        diarization_rows, diarization_used = [], False
        if cfg.get("diarization"):
            print(f"INFO|Diarisation WhisperX device={effective_device}", flush=True)
            diarize_model = load_diarization_pipeline_safe(
                hf_token=cfg.get("hf_token"), device=effective_device, cache_dir=align_model_cache_dir,
            )
            diarization = run_diarization_safe(diarize_model, audio)
            aligned = whisperx.assign_word_speakers(diarization, {"segments": aligned_segments})
            aligned_segments = normalize_segments(aligned.get("segments", []))
            diarization_used = True
            try:
                # La colonne "segment" porte un objet pyannote.core.Segment,
                # non sérialisable en JSON — redondant de toute façon avec
                # les colonnes "start"/"end" (float) déjà présentes.
                diarization_rows = diarization.drop(columns=["segment"], errors="ignore").to_dict("records")
            except Exception:
                diarization_rows = []
            print(f"PROGRESS|{lang}|95", flush=True)

        elapsed = time.time() - t0

        prefix = f"wx_aligned_{lang}"
        (out_dir / f"{prefix}.txt").write_text(" ".join(segment_text(seg) for seg in aligned_segments if segment_text(seg)), encoding="utf-8")
        with (out_dir / f"{prefix}.srt").open("w", encoding="utf-8") as f:
            for i, seg in enumerate(aligned_segments, 1):
                if text := segment_text(seg): f.write(f"{i}\n{srt_ts(float(seg['start']))} --> {srt_ts(float(seg['end']))}\n{text}\n\n")
        with (out_dir / f"{prefix}.vtt").open("w", encoding="utf-8") as f:
            f.write("WEBVTT\n\n")
            for i, seg in enumerate(aligned_segments, 1):
                if text := segment_text(seg): f.write(f"{i}\n{vtt_ts(float(seg['start']))} --> {vtt_ts(float(seg['end']))}\n{text}\n\n")

        jout = out_dir / f"{prefix}.json"
        payload = {"source_file": str(source), "language": lang, "elapsed_seconds": elapsed, "segment_count": len(aligned_segments), "segments": aligned_segments, "diarization_used": diarization_used, "diarization_rows": diarization_rows, "device_requested": requested_device, "device_effective": effective_device, "torch_cuda": torch_cuda, "started_at": cfg.get("started_at"), "finished_at": now_iso()}
        jout.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"PROGRESS|{lang}|100", flush=True)
        print(f"RESULT_JSON|{lang}|{jout}", flush=True)

    except Exception as exc:
        print(f"ERROR|{type(exc).__name__}: {exc}", flush=True)
        print("TRACE|" + traceback.format_exc().replace("\n", " | "), flush=True)
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
