"""
generar_figuras.py - genera todas las ilustraciones de la presentacion.

Uso (desde la carpeta del proyecto):
    python scripts/generar_figuras.py            # todo
    python scripts/generar_figuras.py portada    # solo una parte

Requiere: numpy, scipy, matplotlib, pillow.
Las salidas van a figuras/.
"""
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter, zoom

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
from xray import (AZUL, BLANCO, FONDO, MAGENTA, NARANJA, Lienzo, a_rgba,  # noqa: E402
                  aplanar, duotono, grano, guardar_rgb, guardar_rgba, revelar,
                  sobre, ventana)

RAIZ = os.path.dirname(AQUI)
FIG = os.path.join(RAIZ, "figuras")
FUENTES = os.path.join(RAIZ, "fonts")
os.makedirs(FIG, exist_ok=True)


def ruta(nombre):
    return os.path.join(FIG, nombre)


# =========================================================== curvas base
def fay(t):
    """Curva mariposa de Temple H. Fay (1989)."""
    r = np.exp(np.cos(t)) - 2 * np.cos(4 * t) - np.sin(t / 12) ** 5
    return np.sin(t) * r, np.cos(t) * r


def dibujar_mariposa(lz, f, n=90000, pesos=1.0, manchas=True):
    t = np.linspace(0, 24 * np.pi, n)
    x, y = fay(t)
    r = np.hypot(x, y)
    px, py = f(x, y)
    lz.relleno(px, py, 0.5)
    peso = np.clip((r / r.max()) ** 1.2 * 1.6, 0.06, 1.0) * pesos
    lz.linea(px, py, peso * 0.8)
    for s, w in [(0.94, 0.22), (0.86, 0.16), (0.76, 0.12)]:
        qx, qy = f(s * x, s * y)
        lz.linea(qx, qy, peso * w)
    if manchas:
        from scipy.signal import argrelmax
        idx = argrelmax(r, order=200)[0]
        esc = f.escala
        for i in idx:
            if r[i] > 0.55 * r.max():
                for s, rr in ((0.90, 0.010), (0.80, 0.007)):
                    qx, qy = f(s * x[i], s * y[i])
                    lz.mancha(float(qx), float(qy), rr * esc * 2.2, 0.55)


# =========================================================== texto gigante
def capa_texto(texto, W, H, ancho=0.94, base=0.905, color=(233, 230, 224),
               fuente="Anton-Regular.ttf", espaciado=0.0):
    """Texto enorme como RGBA + cajas (x0, x1) de cada letra."""
    ft_path = os.path.join(FUENTES, fuente)
    size = 400
    ft = ImageFont.truetype(ft_path, size)

    def medir(sz):
        ft = ImageFont.truetype(ft_path, sz)
        xs = []
        x = 0.0
        for ch in texto:
            bb = ft.getbbox(ch)
            xs.append((x + bb[0], x + bb[2]))
            x += ft.getlength(ch) + espaciado * sz
        total = xs[-1][1] - xs[0][0]
        return ft, xs, total

    ft, xs, total = medir(size)
    size = int(size * ancho * W / total)
    ft, xs, total = medir(size)
    x_ini = (W - total) / 2 - xs[0][0]
    img = Image.new("L", (W * 2, H * 2), 0)
    d = ImageDraw.Draw(img)
    ft2 = ImageFont.truetype(ft_path, size * 2)
    cajas = [(x_ini + a, x_ini + b) for (a, b) in xs]
    # dibujar letra por letra (mismo avance que en medir)
    x = x_ini
    for ch in texto:
        d.text((2 * x, 2 * base * H), ch, font=ft2, fill=255, anchor="ls")
        x += ft.getlength(ch) + espaciado * size
    a = np.asarray(img, np.float32) / 255
    a = a.reshape(H, 2, W, 2).mean(axis=(1, 3))
    # textura de papel muy sutil
    rng = np.random.default_rng(3)
    tex = gaussian_filter(rng.standard_normal((H, W)), 9)
    tex = 1 - 0.0 * (tex / tex.std())
    rgb = (np.array(color, np.float32) / 255)[None, None, :] * tex[..., None]
    out = np.concatenate([rgb, a[..., None]], -1)
    alto = ft.getbbox(texto)[3] - ft.getbbox(texto)[1]
    return out, cajas, alto


def entrelazar(obj, txt, cajas, delante, W):
    """El objeto pasa por delante de las letras en 'delante' y por detras del resto."""
    M = np.zeros((1, W, 1), np.float32)
    for i in delante:
        x0, x1 = cajas[i]
        # extender hasta la mitad del hueco con las vecinas
        if i > 0:
            x0 = (cajas[i - 1][1] + x0) / 2
        if i < len(cajas) - 1:
            x1 = (x1 + cajas[i + 1][0]) / 2
        M[0, int(x0):int(np.ceil(x1)), 0] = 1
    frente = sobre(obj, txt)
    detras = sobre(txt, obj)
    return M * frente + (1 - M) * detras


# =========================================================== 0. grano
# Intensidad del grano. Los puntos claros son la "estatica" que se ve sobre el
# fondo negro; los oscuros solo se notan encima de lo claro (letras, figuras).
GRANO_CLARO = 0.035
GRANO_OSCURO = 0.085


def fig_grano():
    W, H = 1600, 900
    n = grano(W, H, sigma=0.6, semilla=11)
    # variacion de baja frecuencia (manchas de revelado)
    rng = np.random.default_rng(5)
    lf = gaussian_filter(rng.standard_normal((H, W)), 60)
    lf = 1 + 0.20 * lf / lf.std()
    k = np.where(n > 0, GRANO_CLARO, GRANO_OSCURO)
    a = np.clip(k * np.abs(n) * lf, 0, 0.40)
    rgb = np.where(n[..., None] > 0, 1.0, 0.0) * np.ones(3)
    guardar_rgba(np.concatenate([rgb, a[..., None]], -1), ruta("grano.png"))
    print("grano.png")


# =========================================================== 1. portada
# Ventana de la portada en fracciones de la pagina (x0, y0 desde arriba, w, h).
VENTANA_PORTADA = (0.245, 0.14, 0.165, 0.22)


def fig_portada():
    W, H = 1920, 1080
    lz = Lienzo(W, H)
    t = np.linspace(0, 24 * np.pi, 1000)
    x, y = fay(t)
    # la mariposa ocupa ~85% del alto, centrada un poco arriba
    alto = 0.86 * H
    s = alto / (y.max() - y.min())
    cx, cy = 0.5 * W, 0.445 * H

    def f(xx, yy):
        return cx + s * np.asarray(xx), cy - s * (np.asarray(yy) - (y.max() + y.min()) / 2)

    f.escala = s
    dibujar_mariposa(lz, f, n=120000)
    I, A = revelar(lz, brillo=(0.38, 0.24, 0.14), ganancia=2.0)
    obj = a_rgba(I, A)
    txt, cajas, _ = capa_texto("FOURIER", W, H, ancho=0.935, base=0.915)
    hero = entrelazar(obj, txt, cajas, delante=[2, 3], W=W)
    guardar_rgba(hero, ruta("portada.png"))
    x0, y0, w, h = VENTANA_PORTADA
    # la ventana muestra el cuerpo de la mariposa (donde se cruzan las lineas)
    v = ventana(obj, 0.5, 0.50, w, h, zoom_f=2.1, color=MAGENTA)
    guardar_rgb(v, ruta("portada_ventana.jpg"))
    # vista previa aplanada
    guardar_rgb(aplanar(hero), os.path.join(RAIZ, "tmp", "prev_portada.png"))
    print("portada.png", [tuple(map(int, c)) for c in cajas])


# =========================================================== 2. cierre
VENTANA_CIERRE = (0.555, 0.14, 0.15, 0.22)


def flor(lz, cx, cy, R, rot=0.0, aplast=0.78, capas=((6, 1.0, 0.0), (5, 0.68, 0.5)),
         tallo=True, H=None):
    rng = np.random.default_rng(4)
    if tallo:
        # tallo: curva suave hacia abajo y a la izquierda
        tt = np.linspace(0, 1, 400)
        P0 = np.array([cx, cy + 0.1 * R])
        P1 = np.array([cx + 0.15 * R, cy + 1.2 * R])
        P2 = np.array([cx - 0.35 * R, cy + 1.8 * R])
        P3 = np.array([cx - 0.25 * R, cy + 3.2 * R])
        B = ((1 - tt) ** 3)[:, None] * P0 + (3 * (1 - tt) ** 2 * tt)[:, None] * P1 \
            + (3 * (1 - tt) * tt ** 2)[:, None] * P2 + (tt ** 3)[:, None] * P3
        for off in (-0.022 * R, 0.022 * R):
            lz.linea(B[:, 0] + off, B[:, 1], 0.55)
        # hojas largas
        for (u, lado, largo) in ((0.55, -1, 1.25), (0.78, 1, 1.0)):
            i = int(u * 399)
            base = B[i]
            ang = np.deg2rad(-110 if lado < 0 else -50)
            d = np.array([np.cos(ang), np.sin(ang)])
            nrm = np.array([-d[1], d[0]])
            ss = np.linspace(0, 1, 200)
            ancho = 0.16 * R * np.sin(np.pi * ss) ** 0.9
            curv = 0.25 * R * ss ** 2 * lado
            eje = base[None, :] + (largo * R * ss)[:, None] * d[None, :] \
                + curv[:, None] * nrm[None, :]
            izq = eje + ancho[:, None] * nrm[None, :]
            der = eje - ancho[:, None] * nrm[None, :]
            poly = np.vstack([izq, der[::-1]])
            lz.relleno(poly[:, 0], poly[:, 1], 0.45)
            lz.linea(poly[:, 0], poly[:, 1], 0.45, cerrada=True)
            lz.linea(eje[:, 0], eje[:, 1], 0.35)
            for k in range(1, 9):
                j = int(k / 9 * 199)
                p = eje[j]
                q = p + 0.6 * ancho[j] * (nrm + 0.8 * d)
                lz.linea([p[0], q[0]], [p[1], q[1]], 0.18)
                q = p - 0.6 * ancho[j] * (nrm - 0.8 * d)
                lz.linea([p[0], q[0]], [p[1], q[1]], 0.18)
    # petalos
    for (n, esc, giro) in capas:
        for i in range(n):
            th = rot + giro + 2 * np.pi * i / n + rng.normal(0, 0.08)
            alfa = np.pi / n * 1.25
            ph = np.linspace(-alfa, alfa, 300)
            borde = (np.cos(np.pi * ph / (2 * alfa)) ** 0.55) * \
                (1 + 0.045 * np.sin(11 * ph + i) + 0.03 * np.sin(23 * ph + 2 * i))
            r = R * esc * borde * (0.92 + 0.12 * rng.random())
            ang = th + ph
            xs = cx + r * np.cos(ang)
            ys = cy - aplast * r * np.sin(ang)
            xs = np.concatenate([[cx], xs, [cx]])
            ys = np.concatenate([[cy], ys, [cy]])
            lz.relleno(xs, ys, 0.30)
            lz.linea(xs, ys, 0.55)
            # venas
            for v in np.linspace(-0.8, 0.8, 9):
                pv = np.linspace(0, 1, 120)
                a2 = th + v * alfa * (0.3 + 0.7 * pv)
                rr = R * esc * 0.95 * pv * np.cos(np.pi * v / 2) ** 0.4
                lz.linea(cx + rr * np.cos(a2), cy - aplast * rr * np.sin(a2), 0.12)
    # centro: rosas polares (rodoneas) como estambres
    tt = np.linspace(0, 2 * np.pi, 4000)
    for k, rr, w in ((5, 0.23, 0.35), (7, 0.17, 0.3), (12, 0.30, 0.18)):
        r = R * rr * np.cos(k * tt)
        lz.linea(cx + r * np.cos(tt), cy - aplast * r * np.sin(tt), w)
    for i in range(70):
        a = 2 * np.pi * i / 70 + rng.normal(0, 0.03)
        rr = R * (0.28 + 0.08 * rng.random())
        x1, y1 = cx + rr * np.cos(a), cy - aplast * rr * np.sin(a)
        lz.linea([cx, x1], [cy, y1], 0.10)
        lz.mancha(x1, y1, R * 0.008, 0.6)


def fig_cierre():
    W, H = 1920, 1080
    lz = Lienzo(W, H)
    flor(lz, 0.64 * W, 0.40 * H, 0.36 * H, rot=0.2)
    flor(lz, 0.31 * W, 0.31 * H, 0.18 * H, rot=1.0, tallo=True,
         capas=((5, 1.0, 0.0), (5, 0.62, 0.6)))
    I, A = revelar(lz, brillo=(0.38, 0.24, 0.14), ganancia=2.0)
    obj = a_rgba(I, A)
    txt, cajas, _ = capa_texto("GRACIAS", W, H, ancho=0.955, base=0.915)
    hero = entrelazar(obj, txt, cajas, delante=[1, 4], W=W)
    guardar_rgba(hero, ruta("cierre.png"))
    x0, y0, w, h = VENTANA_CIERRE
    v = ventana(obj, 0.70, 0.30, w, h, zoom_f=1.8, color=AZUL)
    guardar_rgb(v, ruta("cierre_ventana.jpg"))
    guardar_rgb(aplanar(hero), os.path.join(RAIZ, "tmp", "prev_cierre.png"))
    print("cierre.png", [tuple(map(int, c)) for c in cajas])


# =========================================================== 3. epiciclos (seccion 01)
def coef_mariposa(M=8192):
    t = np.linspace(0, 24 * np.pi, M, endpoint=False)
    x, y = fay(t)
    z = x + 1j * y
    c = np.fft.fft(z) / M
    k = np.fft.fftfreq(M, 1 / M)
    return c, k


def reconstruir(c, k, K, t):
    """Suma de los K coeficientes de mayor magnitud evaluada en t (fase 0..2pi)."""
    orden = np.argsort(-np.abs(c))[:K]
    z = np.zeros_like(t, dtype=complex)
    for i in orden:
        z += c[i] * np.exp(1j * k[i] * t)
    return z, orden


def fig_epiciclos():
    c, k = coef_mariposa()
    W, H = 1500, 1300
    lz = Lienzo(W, H)
    t = np.linspace(0, 2 * np.pi, 30000)
    zf, _ = reconstruir(c, k, 400, t)
    f = lz.mapa(zf.real.min(), zf.real.max(), zf.imag.min(), zf.imag.max(), margen=0.08)
    # objetivo tenue
    px, py = f(zf.real, zf.imag)
    lz.linea(px, py, 0.10)
    # trazo con 60 circulos hasta 82% del periodo
    K = 60
    t_fin = 0.82 * 2 * np.pi
    tt = np.linspace(0, t_fin, 26000)
    z60, orden = reconstruir(c, k, K, tt)
    px, py = f(z60.real, z60.imag)
    w = np.linspace(0.25, 1.0, len(tt)) ** 1.5
    lz.linea(px, py, w * 0.9)
    # cadena de epiciclos en t_fin
    p = 0j
    for idx in orden:
        q = p + c[idx] * np.exp(1j * k[idx] * t_fin)
        r = abs(c[idx]) * f.escala
        X, Y = f(p.real, p.imag)
        if r > 1.5:
            lz.circulo(float(X), float(Y), r, peso=0.32)
        X2, Y2 = f(q.real, q.imag)
        lz.linea([X, X2], [Y, Y2], 0.75)
        lz.mancha(float(X), float(Y), 2.2, 0.5)
        p = q
    X, Y = f(p.real, p.imag)
    lz.mancha(float(X), float(Y), 6, 1.2)
    I, A = revelar(lz, brillo=(0.40, 0.26, 0.16), ganancia=2.1)
    obj = a_rgba(I, A)
    guardar_rgba(obj, ruta("sec1_epiciclos.png"))
    v = ventana(obj, 0.42, 0.40, 0.36, zoom_f=1.6, color=NARANJA, aspecto=0.83)
    guardar_rgb(v, ruta("sec1_ventana.jpg"))
    guardar_rgb(aplanar(obj), os.path.join(RAIZ, "tmp", "prev_sec1.png"))

    # paneles de reconstruccion para la diapositiva de epiciclos
    for K in (2, 4, 8, 17):
        W2, H2 = 640, 560
        lz2 = Lienzo(W2, H2)
        f2 = lz2.mapa(zf.real.min(), zf.real.max(), zf.imag.min(), zf.imag.max(), margen=0.07)
        px, py = f2(zf.real, zf.imag)
        lz2.linea(px, py, 0.07)
        tt = np.linspace(0, 2 * np.pi, 20000)
        zk, orden = reconstruir(c, k, K, tt)
        px, py = f2(zk.real, zk.imag)
        lz2.linea(px, py, 0.85)
        # cadena en un instante
        t0 = 0.37 * 2 * np.pi
        p = 0j
        for idx in orden:
            q = p + c[idx] * np.exp(1j * k[idx] * t0)
            r = abs(c[idx]) * f2.escala
            X, Y = f2(p.real, p.imag)
            if r > 1.2:
                lz2.circulo(float(X), float(Y), r, peso=0.35)
            X2, Y2 = f2(q.real, q.imag)
            lz2.linea([X, X2], [Y, Y2], 0.7)
            p = q
        X, Y = f2(p.real, p.imag)
        lz2.mancha(float(X), float(Y), 4, 1.2)
        I, A = revelar(lz2, brillo=(0.35, 0.22, 0.12), ganancia=2.0)
        guardar_rgba(a_rgba(I, A), ruta(f"epiciclos_K{K}.png"))
    print("epiciclos")


# =========================================================== 4. red mariposa (seccion 02)
def fig_red():
    W, H = 1500, 1300
    lz = Lienzo(W, H)
    N = 16
    etapas = 4
    mx, my = 0.08 * W, 0.06 * H
    xs = np.linspace(mx, W - mx, etapas + 1)
    ys = np.linspace(my, H - my, N)
    for s in range(1, etapas + 1):
        span = 2 ** (s - 1)
        x0, x1 = xs[s - 1], xs[s]
        for b in range(0, N, 2 * span):
            for j in range(span):
                i0, i1 = b + j, b + j + span
                for (a, bb) in ((i0, i0), (i0, i1), (i1, i0), (i1, i1)):
                    ya, yb = ys[a], ys[bb]
                    # curva suave tipo "S"
                    tt = np.linspace(0, 1, 200)
                    sm = tt * tt * (3 - 2 * tt)
                    X = x0 + (x1 - x0) * tt
                    Y = ya + (yb - ya) * sm
                    lz.linea(X, Y, 0.55 if a != bb else 0.42)
    for s in range(etapas + 1):
        for i in range(N):
            lz.circulo(xs[s], ys[i], 9, peso=0.7)
            lz.mancha(xs[s], ys[i], 3.2, 0.9)
    I, A = revelar(lz, brillo=(0.42, 0.28, 0.16), ganancia=2.0)
    obj = a_rgba(I, A)
    guardar_rgba(obj, ruta("sec2_red.png"))
    v = ventana(obj, 0.52, 0.47, 0.34, zoom_f=1.7, color=AZUL, aspecto=0.83)
    guardar_rgb(v, ruta("sec2_ventana.jpg"))
    guardar_rgb(aplanar(obj), os.path.join(RAIZ, "tmp", "prev_sec2.png"))
    # mariposa de verdad (Fay) para la diapositiva "La mariposa"
    W3, H3 = 1300, 1150
    lz3 = Lienzo(W3, H3)
    t = np.linspace(0, 24 * np.pi, 1000)
    x, y = fay(t)
    f3 = lz3.mapa(x.min(), x.max(), y.min(), y.max(), margen=0.05)
    dibujar_mariposa(lz3, f3)
    I, A = revelar(lz3, brillo=(0.38, 0.24, 0.14), ganancia=2.0)
    obj3 = a_rgba(I, A)
    guardar_rgba(obj3, ruta("mariposa.png"))
    v = ventana(obj3, 0.5, 0.56, 0.40, zoom_f=1.7, color=NARANJA, aspecto=1.0)
    guardar_rgb(v, ruta("mariposa_ventana.jpg"))
    print("red + mariposa")


# =========================================================== 5. difraccion (seccion 03)
def pupila_hex(Ng=2048, D=260):
    """Espejo de 18 hexagonos (como el JWST) con 3 soportes. D = diametro en px."""
    yy, xx = np.mgrid[0:Ng, 0:Ng] - Ng / 2
    s = D / 5.0                      # distancia entre centros (aprox)
    a = s / np.sqrt(3) * 0.97        # radio del hexagono (vertice), con huecos

    def hexa(cx, cy):
        # hexagono con vertices arriba/abajo (pointy-top)
        x = np.abs(xx - cx)
        y = np.abs(yy - cy)
        return (x <= a * np.sqrt(3) / 2) & (y + x / np.sqrt(3) <= a)

    P = np.zeros((Ng, Ng), bool)
    centros = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            if abs(q + r) > 2 or (q == 0 and r == 0):
                continue
            cx = s * (q + r / 2)
            cy = s * (r * np.sqrt(3) / 2)
            centros.append((cx, cy))
            P |= hexa(cx, cy)
    # soportes del secundario: uno vertical y dos a +-60 deg hacia abajo
    w = 0.6
    P &= ~((np.abs(xx) < w) & (yy < 0))
    for ang in (np.deg2rad(-30), np.deg2rad(210)):
        d = np.array([np.cos(ang), -np.sin(ang)])
        proy = xx * d[0] + yy * d[1]
        perp = np.abs(-xx * d[1] + yy * d[0])
        P &= ~((perp < w) & (proy > 0))
    return P.astype(np.float32), len(centros)


def fig_difraccion():
    from scipy.ndimage import affine_transform
    Ng, D = 4096, 400
    P, n = pupila_hex(Ng, D)
    F = np.fft.fftshift(np.fft.fft2(P.astype(np.complex64)))
    I = (np.abs(F) ** 2).astype(np.float64)
    del F
    I /= I.max()
    c = Ng // 2
    r = 300                                  # ~ +-29 lambda/D
    Ic = I[c - r:c + r, c - r:c + r]
    # luz de banda ancha: promedio de la PSF a varias longitudes de onda
    # (la PSF escala con lambda). Esto borra la reticula y deja las espigas.
    acc = np.zeros_like(Ic)
    escalas = np.linspace(0.70, 1.30, 13)
    for sc in escalas:
        m = 1 / sc
        off = r - m * r
        acc += affine_transform(Ic, [m, m], offset=[off, off], order=1) / sc ** 2
    acc /= acc.max()
    L = np.log10(acc + 1e-14)
    v = np.clip((L + 4.6) / 4.6, 0, 1) ** 1.45
    v = zoom(v, 2.4, order=3)
    v = np.clip(v, 0, 1)
    glow = v + 0.30 * gaussian_filter(v, 4) + 0.22 * gaussian_filter(v, 16)
    Iv = (1 - np.exp(-2.4 * glow)).astype(np.float32)
    obj = a_rgba(Iv, np.zeros_like(Iv))
    guardar_rgba(obj, ruta("sec3_difraccion.png"))
    vv = ventana(obj, 0.5, 0.5, 0.42, zoom_f=1.15, color=MAGENTA, aspecto=0.83)
    guardar_rgb(vv, ruta("sec3_ventana.jpg"))
    guardar_rgb(duotono(Iv, AZUL), ruta("difraccion_azul.jpg"))
    # pupila (misma geometria, dibujada a mas resolucion)
    del I
    Pg, _ = pupila_hex(1100, 900)
    cp = Pg.astype(np.float32)
    lz = Lienzo(cp.shape[1], cp.shape[0])
    lz.mem = cp * 0.9
    Ip, Ap = revelar(lz, brillo=(0.3, 0.2, 0.1), borde_mem=2.4, mem_alpha=0.9)
    guardar_rgba(a_rgba(Ip, Ap, color_mem=(120, 120, 124)), ruta("pupila.png"))
    guardar_rgb(aplanar(obj), os.path.join(RAIZ, "tmp", "prev_sec3.png"))
    print("difraccion", n, "segmentos")


# =========================================================== 6. wavetable (seccion 04)
def frames_wavetable(F=34, N=600, K=40):
    """Morph lineal entre 5 formas clave (como el modo por defecto de un
    oscilador de tablas): seno -> triangulo -> sierra -> cuadrada -> pulso."""
    ph = np.linspace(0, 2 * np.pi, N, endpoint=False)
    k = np.arange(1, K + 1)[:, None]
    seno = np.sin(ph)
    tri = (8 / np.pi ** 2) * np.sum(((k % 2) * (-1.0) ** ((k - 1) // 2) / k ** 2) * np.sin(k * ph), 0)
    sie = (2 / np.pi) * np.sum(((-1.0) ** (k + 1) / k) * np.sin(k * ph), 0)
    cua = (4 / np.pi) * np.sum(((k % 2) / k) * np.sin(k * ph), 0)
    d = 0.18
    pul = np.sum((2 / (np.pi * k)) * np.sin(np.pi * k * d) * np.cos(k * (ph - np.pi)), 0)
    clave = [w / np.abs(w).max() for w in (seno, tri, sie, cua, pul)]
    out = []
    for j in range(F):
        u = j / (F - 1) * (len(clave) - 1)
        i = min(int(u), len(clave) - 2)
        a = u - i
        w = (1 - a) * clave[i] + a * clave[i + 1]
        out.append(w / (np.abs(w).max() + 1e-9))
    return ph, np.array(out)


def fig_wavetable():
    ph, fr = frames_wavetable()
    F, N = fr.shape
    W, H = 1600, 1250
    lz = Lienzo(W, H)
    x0, y0 = 0.06 * W, 0.70 * H
    ancho = 0.56 * W
    dx, dy = 0.30 * W / F, -0.52 * H / F
    amp = 0.12 * H
    for j in range(F - 1, -1, -1):
        X = x0 + ancho * ph / (2 * np.pi) + j * dx
        Y = y0 - amp * fr[j] + j * dy
        base = y0 + j * dy
        poly_x = np.concatenate([X, [X[-1], X[0]]])
        poly_y = np.concatenate([Y, [base, base]])
        lz.relleno(poly_x, poly_y, 0.05)
        clave = (j % ((F - 1) // 4) == 0)
        brillo = (0.30 + 0.45 * (1 - j / F)) * (1.9 if clave else 1.0)
        lz.linea(X, Y, brillo)
    # ejes tenues
    yb = y0 + 0.16 * H
    lz.linea([x0, x0 + ancho], [yb, yb], 0.3)
    lz.linea([x0 + ancho, x0 + ancho + F * dx], [yb, yb + F * dy], 0.3)
    for j in range(0, F, (F - 1) // 4):
        xx = x0 + ancho + j * dx
        yy = yb + j * dy
        lz.linea([xx, xx + 0.012 * W], [yy, yy], 0.3)
    I, A = revelar(lz, brillo=(0.40, 0.26, 0.15), ganancia=2.0, borde_mem=0.6, mem_alpha=0.7)
    obj = a_rgba(I, A)
    guardar_rgba(obj, ruta("sec4_wavetable.png"))
    v = ventana(obj, 0.45, 0.52, 0.30, zoom_f=1.6, color=NARANJA, aspecto=0.83)
    guardar_rgb(v, ruta("sec4_ventana.jpg"))
    v2 = ventana(obj, 0.62, 0.40, 0.30, zoom_f=1.6, color=MAGENTA, aspecto=1.0)
    guardar_rgb(v2, ruta("wavetable_ventana.jpg"))
    guardar_rgb(aplanar(obj), os.path.join(RAIZ, "tmp", "prev_sec4.png"))
    print("wavetable")


# =========================================================== 7. matriz DFT
def fig_matriz():
    N = 32
    n = np.arange(N)
    F = np.exp(-2j * np.pi * np.outer(n, n) / N)
    v = (F.real + 1) / 2
    v = np.kron(v, np.ones((20, 20)))
    # separacion de celdas
    g = np.ones_like(v)
    g[::20, :] = 0.55
    g[:, ::20] = 0.55
    v = v * g
    guardar_rgb(duotono(v, AZUL, blanco=0.6), ruta("matriz_dft.jpg"))
    print("matriz_dft")


# =========================================================== 8. espectrogramas
def stft_mag(x, N, H, fs):
    w = np.hanning(N)
    nfr = 1 + (len(x) - N) // H
    S = np.array([np.abs(np.fft.rfft(w * x[i * H:i * H + N])) for i in range(nfr)]).T
    return S


def senal_prueba(fs=16000, T=2.0):
    t = np.arange(int(fs * T)) / fs
    # "chirp" + tono con vibrato + clics
    x = 0.5 * np.sin(2 * np.pi * (300 * t + 0.5 * 2600 * t ** 2 / T))
    f_v = 1100 + 60 * np.sin(2 * np.pi * 5.5 * t)
    x += 0.35 * np.sin(2 * np.pi * np.cumsum(f_v) / fs)
    for tc in (0.35, 0.75, 1.15, 1.55):
        i = int(tc * fs)
        x[i:i + 40] += np.hanning(40) * 1.5
    return x, fs


def img_espectro(S, color, W=1100, H=560, rango=70, fmax_frac=1.0):
    S = S[: max(2, int(S.shape[0] * fmax_frac))]
    L = 20 * np.log10(S / S.max() + 1e-9)
    v = np.clip((L + rango) / rango, 0, 1) ** 1.6
    v = v[::-1]  # frecuencias altas arriba
    img = Image.fromarray((v * 255).astype(np.uint8)).resize((W, H), Image.BICUBIC)
    v = np.asarray(img, np.float32) / 255
    v = np.clip(0.9 * v + 0.15 * gaussian_filter(v, 3), 0, 1)
    return duotono(v, color)


def fig_espectrogramas():
    x, fs = senal_prueba()
    S1 = stft_mag(x, 64, 16, fs)
    S2 = stft_mag(x, 2048, 64, fs)
    fr = 3600 / (fs / 2)
    guardar_rgb(img_espectro(S1, NARANJA, fmax_frac=fr), ruta("espec_corta.jpg"))
    guardar_rgb(img_espectro(S2, AZUL, fmax_frac=fr), ruta("espec_larga.jpg"))
    print("espectrogramas")


# =========================================================== 9. vocoder de fase
def vocoder(x, factor, N=2048, Ha=256):
    """Estiramiento temporal por vocoder de fase (sin cambiar la altura)."""
    Hs = int(round(Ha * factor))
    w = np.hanning(N)
    nfr = 1 + (len(x) - N) // Ha
    X = np.array([np.fft.rfft(w * x[i * Ha:i * Ha + N]) for i in range(nfr)])
    omega = 2 * np.pi * np.arange(N // 2 + 1) * Ha / N
    fase = np.angle(X[0])
    y = np.zeros(nfr * Hs + N)
    norm = np.zeros_like(y)
    for m in range(nfr):
        if m > 0:
            dphi = np.angle(X[m]) - np.angle(X[m - 1]) - omega
            dphi = (dphi + np.pi) % (2 * np.pi) - np.pi
            fase = fase + (omega + dphi) * Hs / Ha
        Y = np.abs(X[m]) * np.exp(1j * fase)
        seg = np.fft.irfft(Y, N) * w
        y[m * Hs:m * Hs + N] += seg
        norm[m * Hs:m * Hs + N] += w ** 2
    y /= np.maximum(norm, 1e-3)
    return y


def fig_vocoder():
    fs = 16000
    t = np.arange(int(fs * 1.2)) / fs
    # frase "melodica": notas de sierra limitada en banda con envolvente
    notas = [196.0, 293.7, 392.0, 246.9]
    x = np.zeros_like(t)
    dur = len(t) // len(notas)
    for i, f0 in enumerate(notas):
        seg = slice(i * dur, (i + 1) * dur)
        tt = t[seg] - t[seg][0]
        env = np.minimum(1, tt / 0.01) * np.exp(-tt * 3)
        s = sum(np.sin(2 * np.pi * k * f0 * tt) / k for k in range(1, int(7000 / f0)))
        x[seg] += env * s
    y = vocoder(x, 2.0)
    S1 = stft_mag(x, 1024, 128, fs)
    S2 = stft_mag(y, 1024, 128, fs)
    # misma escala temporal: la original ocupa la mitad
    fr = 2600 / (fs / 2)
    i1 = img_espectro(S1, MAGENTA, W=500, H=420, rango=60, fmax_frac=fr)
    i2 = img_espectro(S2, MAGENTA, W=1000, H=420, rango=60, fmax_frac=fr)
    guardar_rgb(i1, ruta("vocoder_original.jpg"))
    guardar_rgb(i2, ruta("vocoder_estirado.jpg"))
    print("vocoder")


# =========================================================== 10. graficas (matplotlib)
def estilo_mpl():
    import matplotlib as mpl
    from matplotlib import font_manager
    for f in ("JetBrainsMono-Regular.ttf", "JetBrainsMono-Bold.ttf"):
        font_manager.fontManager.addfont(os.path.join(FUENTES, f))
    mpl.rcParams.update({
        "font.family": "JetBrains Mono",
        "font.size": 9,
        "axes.edgecolor": "#5c5c5c",
        "axes.labelcolor": "#bdbab4",
        "xtick.color": "#8c8c8c",
        "ytick.color": "#8c8c8c",
        "text.color": "#e6e3dd",
        "axes.facecolor": "none",
        "figure.facecolor": "none",
        "savefig.facecolor": "none",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "lines.linewidth": 1.4,
        "axes.unicode_minus": True,
        "pdf.fonttype": 42,
        "legend.frameon": False,
    })


def hx(c):
    return "#%02x%02x%02x" % c


BL = "#e9e6e0"
GR = "#7a7a7a"


def graf_aliasing():
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.6, 2.1))
    t = np.linspace(0, 1, 2000)
    f, fs = 9, 10
    ax.plot(t, np.sin(2 * np.pi * f * t), color=BL, lw=1.0, alpha=0.85, label="señal real: 9 Hz")
    ts = np.arange(0, 1.0001, 1 / fs)
    ax.plot(t, np.sin(2 * np.pi * (f - fs) * t), color=hx(NARANJA), lw=1.8,
            label="lo que «ve» el muestreo: −1 Hz")
    ax.plot(ts, np.sin(2 * np.pi * f * ts), "o", color=hx(NARANJA), ms=5,
            mec="#111", mew=0.6)
    ax.set_xlabel("tiempo (s)   ·   f_s = 10 Hz")
    ax.set_yticks([-1, 0, 1])
    ax.legend(loc="upper right", fontsize=7, ncol=2, bbox_to_anchor=(1.0, 1.22))
    fig.tight_layout()
    fig.savefig(ruta("aliasing.pdf"))
    plt.close(fig)


def graf_complejidad():
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.2, 2.7))
    N = 2.0 ** np.arange(3, 25)
    ax.loglog(N, N ** 2, color=hx(NARANJA), lw=1.8, label="DFT directa:  N²")
    ax.loglog(N, N / 2 * np.log2(N), color=hx(AZUL), lw=1.8, label="FFT:  (N/2)·log₂N")
    for n in (2 ** 10, 2 ** 20):
        ax.plot([n, n], [n / 2 * np.log2(n), n ** 2], color=GR, lw=0.7, ls=(0, (2, 2)))
        ax.text(n * 1.25, np.sqrt(n ** 2 * n / 2 * np.log2(n)),
                f"×{(2 * n / np.log2(n)):,.0f}".replace(",", " "), color=BL, fontsize=8,
                va="center")
    ax.set_xlabel("N (muestras)")
    ax.set_ylabel("multiplicaciones complejas")
    ax.legend(loc="upper left", fontsize=7.5)
    ax.set_xticks([1e1, 1e3, 1e5, 1e7])
    fig.tight_layout()
    fig.savefig(ruta("complejidad.pdf"))
    plt.close(fig)


def graf_fuga():
    import matplotlib.pyplot as plt
    N = 64
    n = np.arange(N)
    x = np.sin(2 * np.pi * 10.5 * n / N)
    fig, axs = plt.subplots(2, 1, figsize=(4.0, 3.15), gridspec_kw={"height_ratios": [1, 1.5]})
    ax = axs[0]
    w = np.hanning(N)
    ax.plot(n, x, color=GR, lw=0.9, label="rectangular")
    ax.plot(n, x * w, color=hx(AZUL), lw=1.4, label="× Hann")
    ax.plot(n, w, color=hx(AZUL), lw=0.8, ls=(0, (3, 2)))
    ax.set_xlabel("n")
    ax.set_yticks([-1, 0, 1])
    ax.legend(fontsize=7, loc="lower left", ncol=2, bbox_to_anchor=(0, 0.98))
    ax = axs[1]
    M = 4096
    for xx, col, lab in ((x, hx(NARANJA), "rectangular"), (x * w, hx(AZUL), "Hann")):
        X = np.abs(np.fft.rfft(xx, M))
        X = 20 * np.log10(X / X.max() + 1e-12)
        k = np.arange(len(X)) * N / M
        ax.plot(k, X, color=col, lw=1.3, label=lab)
    Xd = np.abs(np.fft.rfft(x))
    Xd = 20 * np.log10(Xd / Xd.max() + 1e-12)
    ax.plot(np.arange(len(Xd)), Xd, "o", ms=2.6, color=hx(NARANJA), mec="none")
    ax.set_xlim(0, 32)
    ax.set_ylim(-90, 5)
    ax.set_xlabel("bin k   (la señal está en k = 10.5)")
    ax.set_ylabel("|X| (dB)")
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(ruta("fuga.pdf"))
    plt.close(fig)


def graf_calor():
    import matplotlib.pyplot as plt
    N, L = 512, 2 * np.pi
    x = np.linspace(0, L, N, endpoint=False)
    u0 = ((x > 1.2) & (x < 2.6)).astype(float) + 0.6 * ((x > 3.8) & (x < 4.3))
    u0 += 0.08 * np.sin(23 * x)
    k = 2 * np.pi * np.fft.fftfreq(N, d=L / N)
    U0 = np.fft.fft(u0)
    fig, ax = plt.subplots(figsize=(4.4, 2.4))
    cols = [BL, "#d7d2c8", hx(AZUL), "#6aa7e6", "#3b6da3", "#29466a"]
    for tt, col in zip((0, 0.002, 0.01, 0.04, 0.15, 0.6), cols):
        u = np.fft.ifft(U0 * np.exp(-k ** 2 * tt)).real
        ax.plot(x, u, color=col, lw=1.3 if tt else 0.9, label=f"t = {tt:g}")
    ax.set_xlabel("x")
    ax.set_ylabel("u(x, t)")
    ax.legend(fontsize=6.5, ncol=3, loc="upper right", bbox_to_anchor=(1.0, 1.18))
    ax.set_yticks([0, 0.5, 1])
    fig.tight_layout()
    fig.savefig(ruta("calor.pdf"))
    plt.close(fig)


def graf_aditiva():
    import matplotlib.pyplot as plt
    t = np.linspace(0, 2, 3000)
    fig, ax = plt.subplots(figsize=(4.6, 2.5))
    cols = [GR, "#b8b3aa", hx(NARANJA), BL]
    for K, col in zip((1, 3, 10, 50), cols):
        s = sum((-1) ** (k + 1) * np.sin(2 * np.pi * k * t) / k for k in range(1, K + 1)) * 2 / np.pi
        ax.plot(t, s, color=col, lw=1.0 if K < 50 else 1.3, label=f"K = {K}")
    ax.set_xlabel("t / T")
    ax.set_yticks([-1, 0, 1])
    ax.legend(fontsize=7, ncol=4, loc="upper left", bbox_to_anchor=(0, 1.16))
    fig.tight_layout()
    fig.savefig(ruta("aditiva.pdf"))
    plt.close(fig)


def graf_alias_wt():
    """Sierra 'ingenua' leida de una tabla vs tabla limitada en banda (mipmap)."""
    import matplotlib.pyplot as plt
    fs = 48000
    f0 = 2093.0          # Do7
    Nt = 2048
    k = np.arange(1, Nt // 2)
    tabla_ingenua = np.fft.irfft(np.r_[0, -1j / k, 0], Nt)
    kmax = int((fs / 2) // f0)
    tabla_lim = np.fft.irfft(np.r_[0, -1j / k * (k <= kmax), 0], Nt)
    n = np.arange(int(0.25 * fs))
    fase = (n * f0 / fs) % 1.0

    def leer(tab):
        pos = fase * Nt
        i0 = np.floor(pos).astype(int) % Nt
        fr = pos - np.floor(pos)
        return (1 - fr) * tab[i0] + fr * tab[(i0 + 1) % Nt]

    fig, axs = plt.subplots(2, 1, figsize=(4.8, 2.9), sharex=True)
    for ax, tab, col, lab in ((axs[0], tabla_ingenua, hx(MAGENTA), "tabla completa (1023 armónicos)"),
                              (axs[1], tabla_lim, hx(AZUL), f"tabla limitada: k ≤ {kmax}")):
        y = leer(tab)
        Y = np.abs(np.fft.rfft(y * np.hanning(len(y))))
        Y = 20 * np.log10(Y / Y.max() + 1e-12)
        fr = np.fft.rfftfreq(len(y), 1 / fs)
        ax.plot(fr / 1000, Y, color=col, lw=0.7)
        for h in range(1, kmax + 1):
            ax.axvline(h * f0 / 1000, color=GR, lw=0.4, ls=(0, (1, 2)))
        ax.set_ylim(-100, 5)
        ax.set_yticks([-80, -40, 0])
        ax.text(0.99, 0.92, lab, transform=ax.transAxes, ha="right", va="top",
                fontsize=7.5, color=col)
    axs[1].set_xlabel("frecuencia (kHz)   ·   nota: Do7 = 2093 Hz,  f_s = 48 kHz")
    axs[0].set_ylabel("dB")
    axs[1].set_ylabel("dB")
    axs[1].set_xlim(0, 24)
    fig.tight_layout(h_pad=0.4)
    fig.savefig(ruta("alias_wavetable.pdf"))
    plt.close(fig)


def graf_buffers():
    import matplotlib.pyplot as plt
    fs = 48000
    n = np.arange(2048)
    rng = np.random.default_rng(1)
    x = (np.sin(2 * np.pi * 220 * n / fs) * 0.6 + 0.25 * np.sin(2 * np.pi * 660 * n / fs + 1)
         + 0.08 * rng.standard_normal(len(n)) * np.exp(-n / 900))
    x *= np.minimum(1, n / 200)
    fig, ax = plt.subplots(figsize=(7.2, 1.3))
    ax.plot(n, x, color=BL, lw=0.8)
    for b in range(0, 2049, 256):
        ax.axvline(b, color=hx(NARANJA), lw=0.8, ls=(0, (2, 2)))
    for i, b in enumerate(range(0, 2048, 256)):
        ax.text(b + 128, 1.12, f"bloque {i + 1}", ha="center", fontsize=6.5, color=GR)
    ax.set_xlim(0, 2048)
    ax.set_ylim(-1.1, 1.3)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("muestras  (256 muestras a 48 kHz = 5.3 ms)")
    fig.tight_layout()
    fig.savefig(ruta("buffers.pdf"))
    plt.close(fig)


def fig_graficas():
    estilo_mpl()
    graf_aliasing()
    graf_complejidad()
    graf_fuga()
    graf_calor()
    graf_aditiva()
    graf_alias_wt()
    graf_buffers()
    print("graficas")


TAREAS = {
    "grano": fig_grano,
    "portada": fig_portada,
    "cierre": fig_cierre,
    "epiciclos": fig_epiciclos,
    "red": fig_red,
    "difraccion": fig_difraccion,
    "wavetable": fig_wavetable,
    "matriz": fig_matriz,
    "espectrogramas": fig_espectrogramas,
    "vocoder": fig_vocoder,
    "graficas": fig_graficas,
}

if __name__ == "__main__":
    os.makedirs(os.path.join(RAIZ, "tmp"), exist_ok=True)
    pedidas = sys.argv[1:] or list(TAREAS)
    for p in pedidas:
        TAREAS[p]()
