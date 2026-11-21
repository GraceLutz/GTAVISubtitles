"""French UI texts (keys: see en.py)."""

MESSAGES: dict[str, str] = {
    # Settings window
    "settings.title": "Réglages du panneau de sous-titres",
    "settings.ui_language": "Langue du programme",
    "settings.ui_language_auto": "Automatique (langue de Windows)",
    "settings.target_language": "Langue cible",
    "settings.custom_code_hint": "(ou un code de langue + Entrée, p. ex. lt)",
    "settings.translator": "Traducteur",
    "settings.api_key": "Clé API",
    "settings.save": "Enregistrer",
    "settings.delete": "Supprimer",
    "settings.ai_model": "Modèle d'IA",
    "settings.font_size": "Taille de police",
    "settings.background_opacity": "Opacité du fond en %",
    "settings.hide_original": "Masquer le sous-titre anglais",
    "settings.reset_position": "Replacer le panneau sur le sous-titre anglais",
    "settings.panel_hint": ("Dans la fenêtre du jeu, déplace le panneau à la souris, redimensionne-le avec la molette, "
                            "clic droit pour le remettre en place. {preview_hint}"),
    "settings.preview_hint_checkbox": "La case ci-dessous affiche un texte d'exemple.",
    "settings.preview_hint_key": "Appuie sur r pour afficher un texte d'exemple.",
    "settings.preview": "Afficher un texte d'exemple et la zone surveillée (pour le réglage)",
    "settings.only_spoken": "Dialogues uniquement (lignes « Nom : texte »)",
    "settings.thin_font": "Police fine (comme dans GTA)",
    "settings.visible_in_recordings": "Afficher dans les enregistrements et les streams (Medal, OBS, Twitch, Kick, YouTube)",
    "settings.key_not_needed": "Pas besoin de clé",
    "settings.key_auto": "Choisit parmi les clés que tu as déjà",
    "settings.key_saved": "✓ Clé enregistrée (Gestionnaire d'identification Windows)",
    "settings.key_session": "✓ Clé disponible (pour cette session)",
    "settings.key_missing": "Pas de clé : saisis-la ci-dessus ou définis la variable d'environnement {variable}",
    # Translator names
    "translator.auto": "Automatique",
    "translator.google": "Google (gratuit)",
    # Subtitle panel
    "layer.preview_text": ("Panneau de sous-titres – déplace-le à la souris, redimensionne avec la molette, "
                           "clic droit : position d'origine"),
    # Command line and console
    "cli.description": "Panneau de sous-titres traduits pour carte d'acquisition ou PS Remote Play",
    "cli.help_config": "fichier de réglages (config.json)",
    "cli.help_device": "numéro de la carte d'acquisition (voir --list-devices)",
    "cli.help_remote_play": ("affiche les sous-titres par-dessus la fenêtre PS Remote Play "
                             "(pas besoin de carte d'acquisition)"),
    "cli.help_window": "titre d'une autre fenêtre à utiliser à la place de Remote Play, p. ex. Xbox",
    "cli.help_webcam": "numéro de la webcam : ta caméra s'affiche dans le coin",
    "cli.help_translator": "service de traduction",
    "cli.help_lang": "code de la langue cible, p. ex. hu, de, es, pl",
    "cli.help_no_panel": "démarrer sans la fenêtre de réglages",
    "cli.help_windowed": "démarrer en fenêtre au lieu du plein écran",
    "cli.help_list_devices": "lister les périphériques vidéo disponibles",
    "cli.help_image": "test : utiliser une capture d'écran au lieu d'une carte d'acquisition",
    "cli.config_create_failed": ("Attention : impossible de créer {path}, démarrage avec les réglages "
                                 "par défaut : {err}"),
    "cli.no_devices": "Aucun périphérique vidéo trouvé. La carte d'acquisition est-elle branchée ?",
    "cli.image_unreadable": "Impossible de lire l'image : {path}",
    "cli.error": "Erreur : {err}",
    "cli.remote_play_mode": "Mode Remote Play : les sous-titres s'affichent par-dessus la fenêtre « {title} ».",
    "cli.remote_play_hint": ("Panneau de sous-titres : déplace-le avec le clic gauche, redimensionne avec la "
                             "molette, clic droit : retour à la position d'origine. Ensuite, reclique dans la "
                             "fenêtre Remote Play. Pour quitter, ferme la fenêtre de réglages."),
    "cli.keys": ("Touches : q/Esc quitter · p panneau de réglages · f plein écran · h masquer l'original · "
                 "r position du panneau de sous-titres · o sous-titres on/off · c webcam on/off"),
    "cli.panel_hint": ("Panneau de sous-titres : déplace-le avec le clic gauche, redimensionne avec la "
                       "molette, clic droit : retour à la position d'origine"),
    "cli.press_enter": "Appuie sur Entrée pour fermer...",
    # Running app
    "app.invalid_setting": "Réglage invalide : {err}",
    "app.save_failed": "Impossible d'enregistrer le réglage : {err}",
    "app.key_vault_unavailable": "L'enregistrement des clés n'est pas disponible : {err}",
    "app.webcam_disabled": "Webcam désactivée : {err}",
    # Settings file (config.json)
    "config.region_range": "les valeurs de subtitle_region doivent être comprises entre 0.0 et 1.0",
    "config.region_order": "subtitle_region : il faut left < right et top < bottom",
    "config.translator_choice": "translator doit être l'une des valeurs suivantes : {choices}",
    "config.device_negative": "device_index ne peut pas être négatif",
    "config.stable_reads_min": "stable_reads doit valoir au moins 1",
    "config.similarity_range": "similarity_threshold doit être compris entre 0 et 1",
    "config.timing_range": "il faut ocr_interval_seconds > 0 et hold_seconds >= 0",
    "config.min_score_range": "min_text_score doit être compris entre 0 et 1",
    "config.ui_language_choice": "ui_language : \"auto\" ou l'une des valeurs suivantes : {choices}",
    "config.target_lang_empty": "target_lang ne peut pas être vide",
    "config.font_size_range": "font_size_ratio doit être compris entre {minimum} et {maximum}",
    "config.opacity_range": "panel_opacity doit être compris entre 0 et 255",
    "config.speaker_color_format": "speaker_color doit être une couleur \"#RRGGBB\", p. ex. \"#7ECFD6\"",
    "config.subtitle_xy_pair": "subtitle_x et subtitle_y doivent être tous deux null ou tous deux des nombres",
    "config.subtitle_xy_range": "subtitle_x et subtitle_y doivent être compris entre 0.0 et 1.0",
    "config.webcam_xy_range": "webcam_x et webcam_y doivent être compris entre 0.0 et 1.0",
    "config.webcam_width_range": "webcam_width_ratio doit être compris entre {minimum} et {maximum}",
    "config.webcam_device_negative": "webcam_device_index ne peut pas être négatif",
    "config.webcam_same_device": ("La webcam et la carte d'acquisition ne peuvent pas utiliser le même "
                                  "numéro de périphérique"),
    "config.unknown_settings": "Réglage(s) inconnu(s) : {names}",
    "config.region_not_object": "subtitle_region doit être un objet (left, top, right, bottom)",
    "config.region_invalid": "subtitle_region est invalide : {err}",
    "config.invalid_json": "{path} n'est pas un JSON valide : {err}",
    "config.root_not_object": "La racine de {path} doit être un objet",
    "config.unknown_setting_save": "Impossible d'enregistrer le(s) réglage(s) inconnu(s) : {names}",
    "config.invalid_json_not_saved": "{path} n'est pas un JSON valide, il n'est pas écrasé : {err}",
    # Capture card and window capture
    "capture.device_open_failed": "Impossible d'ouvrir le périphérique vidéo {index}",
    "capture.no_signal": "La carte d'acquisition n'envoie pas d'image (câble / appareil ?)",
    "capture.read_failed": "La lecture d'une image a échoué {count} fois de suite",
    "capture.window_not_found": ("Fenêtre « {title} » introuvable. Est-elle lancée et connectée "
                                 "à la console ?"),
    "capture.window_fallback": ("Impossible de capturer le contenu de la fenêtre ({reason}) : lecture depuis "
                                "l'écran à la place, donc avec l'overlay visible dans les enregistrements, "
                                "le sous-titre anglais ne peut pas être masqué"),
    "capture.printwindow_failed": "PrintWindow ne fonctionne pas sur cette fenêtre",
    "capture.window_black": "le contenu de la fenêtre est capturé en noir",
    # Remote Play overlay
    "overlay.capture_affinity_failed": ("Impossible de définir si l'overlay apparaît dans les enregistrements "
                                        "(nécessite Windows 10 2004+)"),
    "remote.ctrl_c_quit": "Ctrl+C : fermeture",
    "remote.layer_error": "Erreur dans l'overlay Remote Play",
    "remote.window_closed": "La fenêtre Remote Play a été fermée, fermeture du programme",
    "remote.band_covered": "Une autre fenêtre couvre la bande de sous-titres, lecture suspendue",
    "remote.band_visible": "La bande de sous-titres est de nouveau visible",
    "remote.capture_continuous": "Capture de la bande de sous-titres : continue ({source})",
    "remote.capture_per_read": "Capture de la bande de sous-titres : une par lecture",
    "remote.reading_full": "Lecture : toute la bande de sous-titres à chaque fois",
    "remote.reading_rows": "Lecture : ligne par ligne (seules les nouvelles lignes sont lues)",
    "remote.rows_failed": "La lecture ligne par ligne a échoué ({worker}) : la bande entière est lue désormais",
    # Subtitle processing
    "pipeline.ocr_error": "Erreur OCR",
    "pipeline.translation_timeout": "Pas de traduction en {seconds:.0f} s, abandon",
    "pipeline.failed_prefix": "(échec de la traduction)",
    "pipeline.timeout_prefix": "(pas de traduction en {seconds:.0f} s)",
    # Translation
    "translator.active": "Traducteur : {translator} → {language}",
    "translator.missing_key": ("{translator} : clé API manquante ({variable}), saisis-la dans le panneau "
                               "de réglages ou définis la variable d'environnement"),
    "translator.unknown": "Traducteur inconnu : {name}",
    "translator.failed": "Échec de la traduction : {err}",
    "translator.empty_response": "Le traducteur a renvoyé une réponse vide",
    "service.key_used_this_run": "{err} ; on l'utilise quand même pour cette session",
    "service.prefetch_failed": "La traduction anticipée a échoué, nouvelle tentative",
    "service.no_translator": "Aucun traducteur configuré",
    "service.last_latency": "dernière traduction : {ms} ms",
    "store.load_failed": "Impossible de lire les traductions enregistrées, démarrage à vide : {err}",
    "store.save_failed": "Impossible d'enregistrer la traduction sur le disque : {err}",
    # Translation services
    "backend.network_error": "Erreur réseau : {error_type}",
    "backend.bad_key": "Clé API invalide ou expirée",
    "backend.http_error": "Le service a renvoyé une erreur (HTTP {status})",
    "backend.overloaded": "Le service est surchargé, réessaie plus tard",
    "backend.deepl_unsupported": "DeepL ne traduit pas vers cette langue : {language}",
    "backend.unexpected_response": "{service} a renvoyé une réponse inattendue",
    # Saved API keys (Windows Credential Manager)
    "vault.credential_comment": "Clé API du traducteur de sous-titres GTA",
    "vault.load_failed": "Impossible de lire la clé {name} (erreur Windows {code})",
    "vault.save_failed": "Impossible d'enregistrer la clé {name} (erreur Windows {code})",
    "vault.delete_failed": "Impossible de supprimer la clé {name} (erreur Windows {code})",
}
