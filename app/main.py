import cv2
import numpy as np
import json
from pathlib import Path

BASE_DIR  = Path(__file__).resolve().parent.parent
IMG_DIR   = BASE_DIR / "img"
PLANTILLA = IMG_DIR / "base.png"

ANCHO_OBJ      = 1584     # ancho al que se normalizan todas las imagenes
N_FILAS        = 30
OPTIONS        = "12345"
DIFF_THR       = 0.20     # fraccion minima de pixeles "marca" en la burbuja
DIFF_DELTA     = 0.10     # diferencia minima entre top1 y top2
HEADER_FILL_THR = 0.43    # para descartar filas no-respuesta del grid


# ---------- utilidades comunes ----------

def _redim(img):
    h, w = img.shape[:2]
    if w == ANCHO_OBJ:
        return img
    escala = ANCHO_OBJ / w
    return cv2.resize(img, (ANCHO_OBJ, round(h * escala)),
                      interpolation=cv2.INTER_AREA)


def _agrupar_1d(valores, tolerancia):
    clusters = []
    for v in sorted(valores):
        ok = False
        for c in clusters:
            if abs(v - sum(c) / len(c)) < tolerancia:
                c.append(v)
                ok = True
                break
        if not ok:
            clusters.append([v])
    return sorted(round(sum(c) / len(c)) for c in clusters)


def _bloque_principal(centros, max_gap=35):
    if not centros:
        return centros
    s = sorted(centros)
    mejor, actual = [s[0]], [s[0]]
    for prev, curr in zip(s, s[1:]):
        if curr - prev <= max_gap:
            actual.append(curr)
        else:
            if len(actual) > len(mejor):
                mejor = actual
            actual = [curr]
    return mejor if len(mejor) >= len(actual) else actual


def _detectar_circulos(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    blur = cv2.GaussianBlur(gray, (5, 5), 1)
    circles = cv2.HoughCircles(
        blur, cv2.HOUGH_GRADIENT, dp=1.2,
        minDist=20, param1=60, param2=18,
        minRadius=4, maxRadius=14
    )
    if circles is None:
        return []
    return [(int(x), int(y), int(r)) for x, y, r in np.round(circles[0]).astype(int)]


# ---------- alineacion por grid ----------

def _puntos_referencia(secs, ys_grid):
    return np.float32([
        [secs[0][0],  ys_grid[0]],
        [secs[3][4],  ys_grid[0]],
        [secs[0][0],  ys_grid[-1]],
    ])


def _alinear_plantilla_a_respuesta(plantilla_gray, pts_plantilla,
                                   pts_respuesta, shape_destino):
    M = cv2.getAffineTransform(pts_plantilla, pts_respuesta)
    h, w = shape_destino
    return cv2.warpAffine(
        plantilla_gray, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )


# ---------- calibracion de grid ----------

def _calibrar_grid(img_color, thresh):
    todos = _detectar_circulos(img_color)
    if not todos:
        return None

    x_todos  = _agrupar_1d([b[0] for b in todos], 14)
    conteo_x = {cx: sum(1 for b in todos if abs(b[0] - cx) <= 14) for cx in x_todos}

    cands = []
    for i in range(len(x_todos) - 4):
        g  = x_todos[i:i + 5]
        gp = [g[j + 1] - g[j] for j in range(4)]
        if all(14 <= x <= 40 for x in gp):
            cands.append((-min(conteo_x.get(c, 0) for c in g),
                          float(np.var(gp)), i, g))
    cands.sort()

    secs, usados = [], set()
    for _, _, i, g in cands:
        rg = set(range(i, i + 5))
        if not (rg & usados):
            secs.append(g)
            usados |= rg
        if len(secs) == 4:
            break
    if len(secs) < 4:
        return None
    secs.sort(key=lambda g: g[0])

    xmin, xmax = secs[0][0] - 20, secs[-1][-1] + 20
    cg = [(x, y, r) for x, y, r in todos if xmin <= x <= xmax]
    if not cg:
        return None
    avg_r = max(6, round(np.mean([r for _, _, r in cg])))

    zo = sorted(cg, key=lambda b: b[1])
    fr = []
    for b in zo:
        ok = False
        for f in fr:
            if abs(b[1] - sum(c[1] for c in f) / len(f)) < 16:
                f.append(b); ok = True; break
        if not ok:
            fr.append([b])

    fv = sorted([f for f in fr if len(f) >= 3],
                key=lambda f: sum(b[1] for b in f) / len(f))

    def _fila_score(f):
        s = 0.0
        for x, y, r in f:
            mask = np.zeros(thresh.shape, np.uint8)
            cv2.circle(mask, (x, y), max(r, 4), 255, -1)
            s += cv2.countNonZero(cv2.bitwise_and(thresh, thresh, mask=mask)) \
                 / max(1, cv2.countNonZero(mask))
        return s / len(f)

    fre = [f for f in fv if _fila_score(f) < HEADER_FILL_THR]
    if not fre:
        return None

    ys = [round(sum(b[1] for b in f) / len(f)) for f in fre]
    yb = _bloque_principal(ys, 60)
    ymn, ymx = yb[0], yb[-1]
    fre = [f for f, yc in zip(fre, ys) if ymn <= yc <= ymx]
    if len(fre) < 2:
        return None

    ys_d = sorted(round(sum(b[1] for b in f) / len(f)) for f in fre)
    gp2  = [ys_d[i + 1] - ys_d[i] for i in range(len(ys_d) - 1)]
    mg   = min(gp2)
    ug   = [g for g in gp2 if g <= 2 * mg]
    sp   = float(np.median(ug)) if ug else float(np.median(gp2))
    y0   = ys_d[0]
    ys_grid = [round(y0 + i * sp) for i in range(N_FILAS)]
    return secs, ys_grid, avg_r


# ---------- deteccion grid CODIGO ----------

# Coordenadas del grid CODIGO en la plantilla (base.png redimensionada a 1584).
CODIGO_XS_PLANTILLA = [289, 313, 338, 362, 386, 411, 435, 459]
CODIGO_YS_PLANTILLA = [341, 367, 393, 418, 443, 469, 494, 519, 544, 571]


def _detectar_codigo_grid(img_color, hint_xs=None, hint_ys=None):
    h, w = img_color.shape[:2]
    if hint_ys is not None:
        y0 = max(0, int(min(hint_ys)) - 30)
        y1 = min(h, int(max(hint_ys)) + 30)
    else:
        y0, y1 = 0, int(h * 0.65)
    if hint_xs is not None:
        x0 = max(0, int(min(hint_xs)) - 25)
        x1 = min(w, int(max(hint_xs)) + 25)
    else:
        x0, x1 = 0, int(w * 0.48)

    roi = img_color[y0:y1, x0:x1]
    todos_roi = _detectar_circulos(roi)
    todos = [(x + x0, y + y0, r) for x, y, r in todos_roi]
    if len(todos) < 30:
        return None

    def _agr(vals, tol=10):
        clusters = []
        for v in sorted(vals):
            ok = False
            for c in clusters:
                if abs(v - sum(c) / len(c)) < tol:
                    c.append(v); ok = True; break
            if not ok:
                clusters.append([v])
        return sorted(round(sum(c) / len(c)) for c in clusters)

    # 1) Tomar solo columnas X que tengan >=7 circulos (descarta texto/ruido).
    xs_all = _agr([b[0] for b in todos], 8)
    xs_dens = []  # [(cx, n_filas, [ys])]
    for cx in xs_all:
        ys_c = _agr([b[1] for b in todos if abs(b[0] - cx) <= 10], 10)
        if len(ys_c) >= 7:
            xs_dens.append((cx, len(ys_c), ys_c))

    if len(xs_dens) < 8:
        return None

    # 2) Eliminar columnas duplicadas cercanas (<20 px), conservar la mas densa.
    xs_dens.sort(key=lambda t: t[0])
    filt = []
    for t in xs_dens:
        if filt and (t[0] - filt[-1][0]) < 20:
            if t[1] > filt[-1][1]:
                filt[-1] = t
        else:
            filt.append(t)

    if len(filt) < 8:
        return None

    # 3) Buscar 8 columnas consecutivas con espaciado uniforme.
    mejor = None
    for i in range(len(filt) - 7):
        g = filt[i:i + 8]
        xs_g = [t[0] for t in g]
        gaps = [xs_g[j + 1] - xs_g[j] for j in range(7)]
        if not (all(14 <= d <= 35 for d in gaps) and max(gaps) - min(gaps) < 10):
            continue
        var = float(np.var(gaps))
        score = (-sum(t[1] for t in g), var)
        if mejor is None or score < mejor[0]:
            mejor = (score, g)

    if mejor is None:
        return None

    g = mejor[1]
    cod_xs = [t[0] for t in g]

    # Estimar paso vertical (sp) desde gaps internos de las 8 columnas.
    gaps_pool = []
    for cx in cod_xs:
        ys_c = sorted(b[1] for b in todos if abs(b[0] - cx) <= 10)
        for k in range(len(ys_c) - 1):
            d = ys_c[k + 1] - ys_c[k]
            if 18 <= d <= 32:
                gaps_pool.append(d)
    if len(gaps_pool) < 5:
        return None
    sp = float(np.median(gaps_pool))

    # Reunir Y por columna (set).
    ys_set_por_col = []
    for cx in cod_xs:
        ys_set_por_col.append(sorted(b[1] for b in todos if abs(b[0] - cx) <= 10))

    if not any(ys_set_por_col):
        return None
    y_min = min(min(s) for s in ys_set_por_col if s)
    y_max = max(max(s) for s in ys_set_por_col if s)

    def _hay_circulo(ci, y, tol=7):
        return any(abs(yy - y) <= tol for yy in ys_set_por_col[ci])

    # Probar y0 en pasos de 1 px y contar matches en grid 8x10.
    mejor_y0 = None
    for y0_int in range(int(y_min) - 5, int(y_max - 9 * sp) + 6):
        hits = 0
        for k in range(10):
            yk = y0_int + k * sp
            for ci in range(8):
                if _hay_circulo(ci, yk):
                    hits += 1
        if mejor_y0 is None or hits > mejor_y0[0]:
            mejor_y0 = (hits, y0_int)

    if mejor_y0 is None or mejor_y0[0] < 40:
        return None

    y0 = mejor_y0[1]
    ys_grid = [round(y0 + k * sp) for k in range(10)]

    avg_r = max(4, round(float(np.mean([b[2] for b in todos]))))
    return cod_xs, ys_grid, avg_r


# ---------- proceso principal ----------

def process(img, plantilla_info):
    """plantilla_info = (plantilla_gray, secs_p, ys_p)."""
    plantilla_gray, secs_p, ys_p = plantilla_info

    img = _redim(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Calibrar grid de la respuesta sobre su propio thresh.
    thresh_resp = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 35, 10
    )
    cal = _calibrar_grid(img, thresh_resp)
    if cal is None:
        return {}, None
    secs_r, ys_r, avg_r = cal

    # Alinear plantilla -> espacio de la respuesta usando 3 puntos del grid.
    pts_p = _puntos_referencia(secs_p, ys_p)
    pts_r = _puntos_referencia(secs_r, ys_r)
    M = cv2.getAffineTransform(pts_p, pts_r)

    def _aplicar_M(x, y):
        return (M[0, 0] * x + M[0, 1] * y + M[0, 2],
                M[1, 0] * x + M[1, 1] * y + M[1, 2])
    h_img, w_img = gray.shape
    plantilla_alineada = cv2.warpAffine(
        plantilla_gray, M, (w_img, h_img),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )

    # Diff: pixeles mas oscuros en la respuesta que en la plantilla = marcas.
    diff = cv2.subtract(plantilla_alineada, gray)
    _, marcas = cv2.threshold(diff, 40, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    marcas = cv2.morphologyEx(marcas, cv2.MORPH_OPEN, kernel)

    answers = {}
    for ri, ry in enumerate(ys_r):
        for si, sxs in enumerate(secs_r):
            q = si * N_FILAS + ri + 1
            scores = []
            for cx in sxs:
                mask = np.zeros(marcas.shape, np.uint8)
                cv2.circle(mask, (cx, ry), max(avg_r, 4), 255, -1)
                filled = cv2.countNonZero(cv2.bitwise_and(marcas, marcas, mask=mask))
                total  = cv2.countNonZero(mask)
                scores.append(filled / total if total else 0.0)

            mejor_oi = int(np.argmax(scores))
            top      = scores[mejor_oi]
            segundo  = sorted(scores, reverse=True)[1]

            if top >= DIFF_THR and (top - segundo) >= DIFF_DELTA:
                answers[q] = OPTIONS[mejor_oi]
            else:
                answers[q] = None

    # Leer CODIGO: predecir zona usando M y refinar con deteccion directa.
    hint_ys = [_aplicar_M(CODIGO_XS_PLANTILLA[0], y)[1] for y in CODIGO_YS_PLANTILLA]
    cod_cal = _detectar_codigo_grid(img, hint_ys=hint_ys)
    if cod_cal is not None:
        cod_xs, cod_ys, cod_r = cod_cal
        digitos = []
        for cx in cod_xs:
            scores_d = []
            for cy in cod_ys:
                mask = np.zeros(marcas.shape, np.uint8)
                cv2.circle(mask, (cx, cy), max(cod_r, 4), 255, -1)
                filled = cv2.countNonZero(cv2.bitwise_and(marcas, marcas, mask=mask))
                total  = cv2.countNonZero(mask)
                scores_d.append(filled / total if total else 0.0)
            mejor_ri = int(np.argmax(scores_d))
            top_d    = scores_d[mejor_ri]
            seg_d    = sorted(scores_d, reverse=True)[1]
            digitos.append(str(mejor_ri) if top_d >= DIFF_THR and (top_d - seg_d) >= DIFF_DELTA else None)
        codigo = "".join(d if d is not None else "?" for d in digitos)
    else:
        codigo = None

    return {k: answers[k] for k in sorted(answers)}, codigo


def cargar_plantilla():
    p = cv2.imread(str(PLANTILLA))
    if p is None:
        raise FileNotFoundError(f"No se encontro la plantilla {PLANTILLA}")
    p = _redim(p)
    gray = cv2.cvtColor(p, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 35, 10
    )
    cal = _calibrar_grid(p, thresh)
    if cal is None:
        raise RuntimeError("No se pudo calibrar el grid de la plantilla")
    secs, ys_grid, _ = cal
    return gray, secs, ys_grid


def main():
    import sys
    args = sys.argv[1:]
    if args:
        img_paths = [Path(args[0])]
    else:
        img_paths = [
            p for p in IMG_DIR.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
            and p.name != PLANTILLA.name
        ]

    plantilla_gray = cargar_plantilla()

    results = []
    for img_path in sorted(img_paths):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        respuestas, codigo = process(img, plantilla_gray)
        marcadas = sum(1 for v in respuestas.values() if v is not None)
        nulas    = sum(1 for v in respuestas.values() if v is None)
        results.append({
            "archivo": img_path.name,
            "codigo": codigo,
            "total_preguntas": len(respuestas),
            "respuestas": respuestas,
            "resumen": {"marcadas": marcadas, "en_blanco": nulas}
        })

    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
