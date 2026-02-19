def reward(action, ocr_result, state, waited_frames, min_blur_ok=60.0, min_area_ok=0.01):
    # state = (conf_bin, area_bin, blur_bin, ar_bin) mais on peut aussi passer les valeurs brutes si tu veux
    if action == 0:
        if waited_frames >= 15:
            return -0.2
        return -0.05

    # action == 1 => OCR lancé
    if not ocr_result:
        return -1.0

    text = (ocr_result.get("norm") or "").strip()
    serie = ocr_result.get("serie", "Inconnue")
    score = float(ocr_result.get("score", 0.0))

    if text == "":
        return -1.0

    # OCR "valide" si la série est reconnue ou score élevé
    ok = (serie != "Inconnue") or (score >= 0.75)

    # pénalité si état visuellement mauvais (proxy "trop tôt")
    # (ici tu peux aussi utiliser area_bin/blur_bin au lieu de valeurs brutes)
    premature = False
    # simple: si score final faible => probablement trop tôt
    if score < 0.55:
        premature = True

    if ok:
        return 1.0
    if premature:
        return -0.5
    return -1.0