from __future__ import annotations

import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

from faster_whisper import WhisperModel

from language_utils import smooth_language_labels


def _ts_ms(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def srt_ts(s): return _ts_ms(s).replace(".", ",")
def vtt_ts(s): return _ts_ms(s)
def now_iso(): return datetime.now().isoformat(timespec="seconds")


def transcribe_once(model, source: Path, language, beam_size: int, min_silence_duration_ms: int,
                     condition_on_previous_text: bool, progress_tag: str | None = None):
    """Lance une passe de transcription complète du fichier, éventuellement avec une langue forcée.
    Émet des lignes PROGRESS|<tag>|<pct> au fil des segments, basées sur la durée réelle de l'audio,
    pour permettre une vraie barre de progression côté GUI."""
    segments_iter, info = model.transcribe(
        str(source), task="transcribe", language=language,
        beam_size=beam_size, word_timestamps=True, vad_filter=True,
        vad_parameters={"min_silence_duration_ms": min_silence_duration_ms},
        condition_on_previous_text=condition_on_previous_text,
    )
    duration = float(getattr(info, "duration", 0.0) or 0.0)
    tag = progress_tag or getattr(info, "language", None) or language or "?"

    rows = []
    last_reported = -1
    for i, segment in enumerate(segments_iter, start=1):
        row = {
            "id": i,
            "start": float(segment.start),
            "end": float(segment.end),
            "text": segment.text,
            "avg_logprob": float(getattr(segment, "avg_logprob", 0.0) or 0.0),
            "no_speech_prob": float(getattr(segment, "no_speech_prob", 0.0) or 0.0),
            "compression_ratio": float(getattr(segment, "compression_ratio", 0.0) or 0.0),
            "words": [],
        }
        for word in (getattr(segment, "words", None) or []):
            row["words"].append({"start": float(word.start), "end": float(word.end), "word": word.word, "probability": float(word.probability)})
        rows.append(row)

        if duration > 0:
            pct = max(0, min(99, int((row["end"] / duration) * 100)))
            if pct != last_reported:
                print(f"PROGRESS|{tag}|{pct}", flush=True)
                last_reported = pct

    print(f"PROGRESS|{tag}|100", flush=True)
    return rows, info


_SENTENCE_END = (".", "!", "?", "…")


def resegment_by_words(rows: list[dict], max_words: int, max_chars: int) -> list[dict]:
    """Redécoupe des segments Whisper en segments plus courts à partir des
    timestamps mot-à-mot, pour produire davantage de segments/sous-titres.
    Coupe à la ponctuation de fin de phrase ou dès qu'une limite de mots/caractères est atteinte."""
    out: list[dict] = []

    def flush(buf: list[dict], avg_logprob: float, no_speech_prob: float):
        if not buf:
            return
        text = "".join(w["word"] for w in buf).strip()
        if not text:
            return
        out.append({
            "id": len(out) + 1,
            "start": buf[0]["start"],
            "end": buf[-1]["end"],
            "text": text,
            "avg_logprob": avg_logprob,
            "no_speech_prob": no_speech_prob,
            "words": list(buf),
        })

    for row in rows:
        words = row.get("words") or []
        if not words:
            out.append({**row, "id": len(out) + 1})
            continue

        buf: list[dict] = []
        for word in words:
            buf.append(word)
            joined = "".join(w["word"] for w in buf).strip()
            word_text = str(word.get("word", "")).strip()
            ends_sentence = word_text.endswith(_SENTENCE_END)
            if len(buf) >= max_words or len(joined) >= max_chars or ends_sentence:
                flush(buf, row.get("avg_logprob", 0.0), row.get("no_speech_prob", 0.0))
                buf = []
        flush(buf, row.get("avg_logprob", 0.0), row.get("no_speech_prob", 0.0))

    return out


def write_language_bundle(out_dir: Path, prefix: str, rows: list[dict]) -> Path:
    (out_dir / f"{prefix}.txt").write_text(
        " ".join(r["text"].strip() for r in rows if r["text"].strip()), encoding="utf-8"
    )
    with (out_dir / f"{prefix}.srt").open("w", encoding="utf-8") as f:
        for i, seg in enumerate(rows, 1):
            f.write(f"{i}\n{srt_ts(seg['start'])} --> {srt_ts(seg['end'])}\n{seg['text'].strip()}\n\n")
    with (out_dir / f"{prefix}.vtt").open("w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for i, seg in enumerate(rows, 1):
            f.write(f"{i}\n{vtt_ts(seg['start'])} --> {vtt_ts(seg['end'])}\n{seg['text'].strip()}\n\n")
    return out_dir / f"{prefix}.json"


def main(config_path: str):
    try:
        cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))

        source = Path(cfg["source_file"])
        out_dir = Path(cfg["output_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)

        cache_dir = Path(cfg.get("model_cache_dir") or str(out_dir / "model_cache"))
        cache_dir.mkdir(parents=True, exist_ok=True)

        if cfg.get("hf_token"):
            os.environ["HF_TOKEN"] = cfg["hf_token"]
            os.environ["HUGGINGFACE_HUB_TOKEN"] = cfg["hf_token"]

        os.environ.setdefault("HF_HOME", str(cache_dir))
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(cache_dir))
        os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_dir))
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

        model_path = cfg["model_name"]
        local_dir = cfg.get("local_model_dir")
        if local_dir:
            local_root = Path(local_dir)
            per_model_dir = local_root / cfg["model_name"]
            if (per_model_dir / "model.bin").exists():
                model_path = str(per_model_dir)
                print(f"INFO|Modèle local utilisé: {model_path}", flush=True)
            elif (local_root / "model.bin").exists():
                model_path = str(local_root)
                print(f"INFO|Modèle local utilisé: {model_path}", flush=True)
            else:
                print(
                    f"INFO|Aucun modèle local trouvé pour '{cfg['model_name']}' dans {local_root} "
                    "— téléchargement depuis Hugging Face.",
                    flush=True,
                )

        t0 = time.time()
        print(f"INFO|Chargement modèle={cfg['model_name']} device={cfg['device']} compute_type={cfg['compute_type']}", flush=True)
        print(f"INFO|Cache={cache_dir}", flush=True)

        try:
            model = WhisperModel(model_path, device=cfg["device"], compute_type=cfg["compute_type"], download_root=str(cache_dir))
        except Exception as exc:
            raise RuntimeError(f"Impossible de charger le modèle Faster-Whisper: {exc}") from exc

        requested_language = cfg.get("language") or None
        beam_size = int(cfg.get("beam_size", 5))
        min_silence_duration_ms = int(cfg.get("min_silence_duration_ms", 500))
        condition_on_previous_text = bool(cfg.get("condition_on_previous_text", True))
        max_words = int(cfg.get("max_words_per_segment", 10))
        max_chars = int(cfg.get("max_chars_per_segment", 42))

        multilang_enabled = bool(cfg.get("multilang_enabled", True)) and requested_language is None
        candidate_languages = [
            str(lang).strip().lower()
            for lang in (cfg.get("candidate_languages") or ["fr", "en"])
            if str(lang).strip()
        ]

        print(f"INFO|Passe 1 (langue={requested_language or 'auto'})", flush=True)
        rows1, info = transcribe_once(
            model, source, requested_language, beam_size, min_silence_duration_ms, condition_on_previous_text,
        )

        primary_language = getattr(info, "language", None) or (requested_language or "fr")
        primary_language_probability = getattr(info, "language_probability", None)

        # ── Détermination des langues à transcrire intégralement ───────────
        languages_to_process = [primary_language]
        if multilang_enabled and len(candidate_languages) > 1:
            probe = smooth_language_labels([dict(r) for r in rows1], primary_language)
            present_languages = sorted({
                str(seg.get("language") or primary_language)
                for seg in probe
                if seg.get("language") in candidate_languages
            })
            if primary_language not in present_languages:
                present_languages.append(primary_language)
            secondary_languages = [
                lang for lang in present_languages
                if lang != primary_language and lang in candidate_languages
            ]
            if secondary_languages:
                print(
                    f"INFO|Langues candidates détectées: {present_languages} "
                    f"— transcriptions complètes supplémentaires: {secondary_languages}",
                    flush=True,
                )
                languages_to_process += secondary_languages

        # ── Une transcription complète et indépendante par langue ──────────
        passes: dict[str, list[dict]] = {primary_language: rows1}
        for lang in languages_to_process:
            if lang == primary_language:
                continue
            print(f"INFO|Transcription complète forcée langue={lang}", flush=True)
            rows_lang, _ = transcribe_once(
                model, source, lang, beam_size, min_silence_duration_ms, condition_on_previous_text,
                progress_tag=lang,
            )
            passes[lang] = rows_lang

        for lang in languages_to_process:
            rows = resegment_by_words(passes[lang], max_words=max_words, max_chars=max_chars)
            for i, row in enumerate(rows, start=1):
                row["id"] = i
                row["language"] = lang
            passes[lang] = rows
            print(f"INFO|Langue={lang}: {len(rows)} segments après re-segmentation", flush=True)

        elapsed = time.time() - t0

        for lang in languages_to_process:
            rows = passes[lang]
            last_end = max((r["end"] for r in rows), default=0.0)
            word_count = sum(len(r.get("words") or []) for r in rows)
            avg_logprobs = [r["avg_logprob"] for r in rows if r.get("avg_logprob") is not None]
            no_speech_probs = [r["no_speech_prob"] for r in rows if r.get("no_speech_prob") is not None]
            avg_confidence = sum(avg_logprobs) / len(avg_logprobs) if avg_logprobs else None
            avg_no_speech_prob = sum(no_speech_probs) / len(no_speech_probs) if no_speech_probs else None
            speech_seconds = sum(max(0.0, r["end"] - r["start"]) for r in rows)

            prefix = f"fw_transcript_{lang}"
            jout = write_language_bundle(out_dir, prefix, rows)
            payload = {
                "source_file": str(source), "model": cfg["model_name"], "language": lang,
                "primary_language": primary_language, "language_probability": primary_language_probability,
                "elapsed_seconds": elapsed, "duration_seconds": last_end if last_end > 0 else None,
                "segment_count": len(rows), "word_count": word_count,
                "avg_confidence": avg_confidence, "avg_no_speech_prob": avg_no_speech_prob,
                "speech_seconds": speech_seconds, "segments": rows,
                "languages_processed": languages_to_process,
                "started_at": cfg.get("started_at"), "finished_at": now_iso(),
            }
            jout.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"RESULT_JSON|{lang}|{jout}", flush=True)

    except Exception as exc:
        print(f"ERROR|{type(exc).__name__}: {exc}", flush=True)
        print("TRACE|" + traceback.format_exc().replace("\n", " | "), flush=True)
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
