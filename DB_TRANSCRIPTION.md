# Schéma `transcription` — base de données du pipeline

Domaine dédié à la transcription, isolé dans le schéma PostgreSQL `transcription`.
Les tables partagées de `public` (`medias`, `users`, `playlists`…) ne sont **pas modifiées** :
le schéma les référence par clé étrangère.

**Un seul système de transcription doit exister.** Une migration antérieure
sur ce projet a mis au jour un piège classique : un ancien schéma de suivi
parallèle, abandonné en cours de route mais jamais supprimé, plus consulté
par aucun consommateur réel (vérifié en interrogeant les routes exposées
par l'API) — deux sources de vérité qui divergent silencieusement.
Leçon retenue : avant de supprimer un schéma qu'on croit mort, vérifier
par la preuve (aucune route ne le référence, aucune donnée n'y est unique)
plutôt que par supposition, et sauvegarder avant de trancher.

## Modèle

```
public.medias (catalogue partagé, inchangé)
      │ 1
      │ n
transcription.runs          ── un passage du pipeline sur une vidéo
      │ 1                      (téléchargement NAS, ffprobe, machine, statut)
      │ n
transcription.transcripts   ── une transcription complète par (run, langue)
      │ 1                      (txt/srt/vtt/json, timings, statut)
      │ n
transcription.transcript_files ── artefacts fichiers sur disque
```

## `public.medias` — identité vs emplacement

Deux informations distinctes, deux colonnes distinctes — les confondre
rend les vidéos issues de sources cloud illisibles (ex: un `nas_path`
du type `gdrive://1bnnZOhCe...` sans aucun nom associé) :

- **`filename`** — nom de fichier ORIGINAL, stable quel que soit l'endroit où
  la vidéo se trouve (NAS, Drive, disque local). C'est la clé humaine.
- **`nas_path`** — emplacement de stockage ACTUEL : chemin NAS réel,
  pseudo-chemin `gdrive://<file_id>` pour Drive, ou chemin local absolu.
  Change quand le fichier est déplacé.

`upsert_media()` réconcilie automatiquement : si un média du même
`filename` + `filesize` existe déjà à un AUTRE `nas_path`, c'est la même
vidéo qui a changé d'emplacement (ex: transcrite depuis Drive, puis
retrouvée sur le NAS une fois synchronisée dans la même arborescence) — pas
de doublon recréé, l'ancien chemin part dans `previous_paths` (historique),
le nouveau devient `nas_path`. La correspondance exige une taille de
fichier IDENTIQUE (protection contre deux vidéos homonymes différentes).

## `transcription.runs`

| Groupe | Colonnes |
|---|---|
| Source | `source_kind` (`nas`/`local`/`drive`/`legacy`), `source_nas_path`, `source_filesize` |
| Copie NAS | `download_started_at`, `download_finished_at`, `download_bytes` |
| Sondage média | `media_duration_seconds`, `video_codec`, `audio_codec`, `width`, `height`, `bitrate`, `format_name`, `frame_rate`, `audio_sample_rate`, `audio_channels`, `stream_count`, `container_tags` (JSONB) |
| Machine | `hostname`, `device`, `compute_type`, `gpu_name`, `model_name`, `pipeline_version`, `engine_info` (JSONB : versions torch/CUDA) |
| Cycle de vie | `status` (`running`/`done`/`partial`/`failed`), `error`, `started_at`, `finished_at` |

## `transcription.transcripts`

`UNIQUE (run_id, language)`. Colonnes : `language`, `is_primary_language`,
`language_probability`, `engine` (`faster-whisper`/`whisperx`), `aligned`, `diarized`,
`status` (`processing`/`done`/`failed`), `error`, `segment_count`,
`transcript_txt/srt/vtt/json`, `transcription_started_at/finished_at`.

`transcript_json` embarque aussi `original_filename`, `media_title` et
`media_source` — l'identité durable de la vidéo, contrairement à
`source_file` (chemin du dossier de travail temporaire, supprimé après
coup). Un export de transcription reste ainsi identifiable même isolé de
la base, comme les métadonnées de titre qu'on retrouve en streaming.

## `transcription.transcript_files`

`UNIQUE (transcript_id, file_format)`. Colonnes : `file_format` (`txt`/`srt`/`vtt`/`json`),
`file_name`, `file_path`, `mime_type`, `file_size`.

## Vues

- **`transcription.v_activity`** — vue de suivi EN DIRECT : titre, `filename`,
  emplacement actuel, langue, statut (`processing`/`done`/`failed`), triée par
  activité la plus récente, SANS les gros blobs texte/JSON (rapide à l'œil).
  À garder ouverte pendant un batch pour voir ce qui se passe, plutôt que la
  table `transcripts` brute (illisible avec son contenu texte complet).
- **`transcription.v_latest_transcripts`** — la dernière transcription `done` par
  (média, langue) : la version à servir (sous-titres, recherche…).
- **`transcription.v_pending_medias`** — vidéos du catalogue sans aucune transcription
  `done` : la file d'attente restante.

## Requêtes types

```sql
-- Suivi en direct pendant un batch
SELECT * FROM transcription.v_activity LIMIT 50;

-- Sous-titres français d'une vidéo
SELECT t.transcript_vtt
FROM transcription.v_latest_transcripts v
JOIN transcription.transcripts t ON t.id = v.transcript_id
WHERE v.media_id = 123 AND v.language = 'fr';

-- Avancement global
SELECT count(*) FILTER (WHERE EXISTS (
         SELECT 1 FROM transcription.transcripts t
         WHERE t.media_id = m.id AND t.status = 'done')) AS transcrits,
       count(*) AS total
FROM public.medias m WHERE m.nas_path IS NOT NULL;

-- Historique complet d'une vidéo (runs + langues)
SELECT r.id, r.started_at, r.status, r.model_name, r.gpu_name,
       t.language, t.status, t.segment_count
FROM transcription.runs r
LEFT JOIN transcription.transcripts t ON t.run_id = r.id
WHERE r.media_id = 123 ORDER BY r.started_at DESC;
```

## Comportement du pipeline

- Chaque lancement sur une vidéo ouvre un **run** (métadonnées complètes), puis une
  **transcript par langue détectée** (fr/en automatique), clôturées `done`/`failed` ;
  le run finit `done`, `partial` (une langue en échec) ou `failed`.
- **Incrémental** : au lancement d'une sélection NAS, les vidéos ayant déjà une
  transcription `done` sont sautées automatiquement — cocher tout le NAS ne retraite
  que le restant.
