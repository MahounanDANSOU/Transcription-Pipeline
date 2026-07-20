#!/usr/bin/env python
"""
Pré-télécharge les modèles d'alignement WhisperX pour FR et EN dans le cache local.

INSTRUCTIONS:
1. Ouvre PowerShell en tant qu'administrateur
2. Exécute: & 'D:\Mike\WhisperX\wx_env\Scripts\Activate.ps1'
3. Puis: python D:\Mike\predownload_align_models.py
4. Une fois terminé, tu peux lancer le batch sans connexion réseau

Le cache sera stocké dans: D:\Mike\runtime_db_gui\hf_cache_wx_align
"""
import os
import sys
from pathlib import Path

# Assure que WhisperX est disponible
try:
    import whisperx
    import torch
except ImportError:
    print("ERROR|WhisperX n'est pas installé dans cet environnement.", file=sys.stderr)
    print("       Utilise l'environnement WhisperX:", file=sys.stderr)
    print("       & 'D:\\Mike\\WhisperX\\wx_env\\Scripts\\Activate.ps1'", file=sys.stderr)
    sys.exit(1)

CACHE_DIR = Path(r"D:\Mike\runtime_db_gui\hf_cache_wx_align")
LANGUAGES = ["fr", "en"]
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN") or None

def main():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"INFO|Cache directory: {CACHE_DIR}", flush=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"INFO|Utilisation du device: {device}", flush=True)

    if HF_TOKEN:
        print(f"INFO|Hugging Face token trouvé", flush=True)
    else:
        print(f"WARNING|Pas de token HF détecté. Certains modèles peuvent nécessiter l'authentification.", flush=True)

    success_count = 0
    for lang in LANGUAGES:
        try:
            print(f"\nINFO|Téléchargement du modèle d'alignement: {lang}", flush=True)
            model, metadata = whisperx.load_align_model(
                language_code=lang,
                device=device,
                model_dir=str(CACHE_DIR),
                model_cache_only=False,
            )
            print(f"OK|Modèle {lang.upper()} téléchargé et cachéavec succès", flush=True)
            success_count += 1
        except Exception as exc:
            print(f"ERROR|Échec du téléchargement pour {lang}: {exc}", file=sys.stderr, flush=True)

    print("\n" + "="*70, flush=True)
    if success_count == len(LANGUAGES):
        print(f"✓ Pré-téléchargement réussi ({success_count}/{len(LANGUAGES)} modèles)", flush=True)
        print("Tu peux maintenant lancer le batch en mode cache_only=True", flush=True)
        return 0
    else:
        print(f"✗ Pré-téléchargement partiel ({success_count}/{len(LANGUAGES)} modèles)", file=sys.stderr, flush=True)
        print("Vérifie les erreurs ci-dessus", file=sys.stderr, flush=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())