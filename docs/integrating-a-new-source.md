# Intégrer une nouvelle source vidéo (NAS, cloud, ou autre)

Le pipeline traite déjà deux sources hétérogènes (une API NAS propriétaire
via `nas_client.py`, Google Drive via `gdrive_client.py`) derrière une
interface strictement identique côté GUI et pipeline. Ajouter une troisième
source (Dropbox, S3, un autre NAS, une API interne quelconque…) suit le
même patron.

## L'interface attendue

Chaque client de source doit exposer deux méthodes avec cette forme exacte
de retour — c'est tout ce dont le reste du pipeline a besoin pour savoir
faire fonctionner l'explorateur à cases à cocher, la résolution récursive
des dossiers, et le téléchargement :

### `browse(path) -> dict`

```python
{
    "items": [
        {
            "name": "Nom affiché",
            "path": "identifiant unique (chemin, ID de fichier…)",
            "isdir": bool,
            "is_video": bool,
            "size": int | None,              # octets
            "mtime": <epoch int | ISO 8601>,  # date de modification
            "duration_seconds": float | None, # si connue sans téléchargement
        },
        ...
    ]
}
```

`path` est opaque pour le reste du système — un chemin réel pour un NAS,
un ID de fichier pour une API cloud, peu importe, du moment qu'il permet
de rappeler `browse(path)` pour descendre dans un dossier et
`download(path, ...)` pour récupérer le fichier.

### `download_file(file_id, filename, dest_folder) -> str`

Télécharge le fichier vers `dest_folder`, retourne le chemin local créé.
**Deux garde-fous non négociables**, appris à la dure sur les deux
intégrations existantes :

1. **Ne jamais faire confiance à un code HTTP 200.** Un proxy ou un backend
   en redémarrage peut répondre 200 avec une page d'erreur JSON/HTML à la
   place du fichier. Inspecter le `Content-Type` et le premier octet du
   corps (`{`, `[`, `<` = presque certainement une erreur, pas une vidéo)
   avant d'écrire quoi que ce soit sur disque.
2. **Vérifier la taille reçue contre `Content-Length`** une fois le
   téléchargement terminé — un flux coupé en cours de route produit un
   fichier tronqué silencieusement acceptable par la commande qui suit
   (ffprobe, whisper…) tant qu'on ne compare pas explicitement les tailles.

Voir `nas_client.py::download_from_nas_path` et
`gdrive_client.py::download_file` pour deux implémentations complètes de
ce patron, avec retry/backoff sur les pannes réseau et 5xx.

## Câblage côté pipeline (`Orchestrateur_GUI.py`)

Une fois le client écrit, l'intégration dans l'UI et le pipeline batch
suit un schéma répétable (voir la section "Google Drive" du fichier comme
gabarit à dupliquer) :

1. Un onglet/section dans l'UI avec son propre `ttk.Treeview`, réutilisant
   les helpers génériques `_populate_source_tree` /
   `_refresh_tree_item_label` / `_on_source_tree_click` /
   `_select_all_source` / `_select_none_source` (déjà factorisés, pas de
   duplication de la logique d'arbre/cases à cocher).
2. Un worker de résolution récursive (`_expand_<source>_selection`) qui
   développe les dossiers cochés en fichiers vidéo individuels, avec la
   même résilience que les deux existants : une seconde passe après une
   pause sur les dossiers en échec, jamais d'abandon silencieux.
3. Un identifiant de stockage préfixé (ex: `gdrive://<id>`) enregistré
   dans `medias.nas_path`, pendant que `medias.filename` porte le nom de
   fichier ORIGINAL — c'est cette séparation qui permet à `upsert_media()`
   de reconnaître une vidéo qui change de source/emplacement au fil du
   temps sans la retranscrire (voir `DB_TRANSCRIPTION.md`).

Le reste (téléchargement avec retry, transcription, alignement,
enregistrement en base) est déjà générique et fonctionne pour n'importe
quelle source respectant ce contrat.
