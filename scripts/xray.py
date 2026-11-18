"""
xray.py - utilidades para dibujar ilustraciones con estetica de "rayos X"
(lineas blancas brillantes, membranas translucidas, brillo difuso).

Todo se dibuja por acumulacion: las lineas se "salpican" (splat) como
muchos puntos sobre una malla de flotantes, asi las intersecciones
brillan mas, igual que en una radiografia.
"""
import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_gradient_magnitude, zoom
from PIL import Image, ImageDraw

# ------------------------------------------------------------------ colores
FONDO = (8, 8, 8)               # mismo negro del tema (rxfondo = #080808)
BLANCO = (238, 236, 232)
NARANJA = (242, 107, 42)
AZUL = (76, 155, 232)
MAGENTA = (232, 51, 109)


# ------------------------------------------------------------------ lienzo
class Lienzo:
    """Lienzo de acumulacion. Coordenadas en pixeles (x a la derecha, y abajo)."""

    def __init__(self, W, H):
        self.W, self.H = int(W), int(H)
        self.lin = np.zeros((self.H, self.W), np.float32)   # lineas
        self.mem = np.zeros((self.H, self.W), np.float32)   # membranas (relleno)

    # -- mapeo de coordenadas "del mundo" a pixeles
    def mapa(self, xmin, xmax, ymin, ymax, margen=0.06, centro=None):
        """Devuelve f(x,y)->(px,py) conservando proporcion."""
        W, H = self.W, self.H
        sx = (1 - 2 * margen) * W / (xmax - xmin)
        sy = (1 - 2 * margen) * H / (ymax - ymin)
        s = min(sx, sy)
        cx = (xmin + xmax) / 2 if centro is None else centro[0]
        cy = (ymin + ymax) / 2 if centro is None else centro[1]

        def f(x, y):
            return W / 2 + s * (np.asarray(x) - cx), H / 2 - s * (np.asarray(y) - cy)

        f.escala = s
        return f

    # -- salpicar puntos con interpolacion bilineal
    def _splat(self, M, xs, ys, w):
        xs = np.asarray(xs, np.float64).ravel()
        ys = np.asarray(ys, np.float64).ravel()
        w = np.broadcast_to(np.asarray(w, np.float64), xs.shape).ravel()
        x0 = np.floor(xs).astype(np.int64)
        y0 = np.floor(ys).astype(np.int64)
        fx = xs - x0
        fy = ys - y0
        for dx, dy, ww in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)),
                           (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
            xi = x0 + dx
            yi = y0 + dy
            m = (xi >= 0) & (xi < self.W) & (yi >= 0) & (yi < self.H)
            np.add.at(M, (yi[m], xi[m]), (w * ww)[m])

    def linea(self, xs, ys, peso=1.0, paso=0.35, cerrada=False):
        """Polilinea densificada: intensidad integrada = peso por pixel de largo."""
        xs = np.asarray(xs, np.float64)
        ys = np.asarray(ys, np.float64)
        if cerrada:
            xs = np.append(xs, xs[0])
            ys = np.append(ys, ys[0])
        seg = np.hypot(np.diff(xs), np.diff(ys))
        L = np.concatenate([[0], np.cumsum(seg)])
        if L[-1] < 1e-9:
            return
        n = max(int(L[-1] / paso), 2)
        s = np.linspace(0, L[-1], n)
        px = np.interp(s, L, xs)
        py = np.interp(s, L, ys)
        pw = peso * (L[-1] / n)
        if np.ndim(peso) == 0:
            self._splat(self.lin, px, py, pw)
        else:  # peso variable a lo largo de la linea
            pw = np.interp(s, L, np.asarray(peso)) * (L[-1] / n)
            self._splat(self.lin, px, py, pw)

    def punto(self, x, y, peso=1.0):
        self._splat(self.lin, [x], [y], [peso])

    def circulo(self, cx, cy, r, peso=1.0, n=None):
        n = n or max(int(2 * np.pi * r / 0.35), 24)
        t = np.linspace(0, 2 * np.pi, n, endpoint=False)
        self.linea(cx + r * np.cos(t), cy + r * np.sin(t), peso, cerrada=True)

    def mancha(self, cx, cy, r, peso=1.0):
        """Disco brillante (manchas de las alas, nodos)."""
        if not hasattr(self, "pts"):
            self.pts = np.zeros((self.H, self.W), np.float32)
        img = Image.new("L", (self.W * 2, self.H * 2), 0)
        d = ImageDraw.Draw(img)
        d.ellipse([2 * (cx - r), 2 * (cy - r), 2 * (cx + r), 2 * (cy + r)], fill=255)
        a = np.asarray(img, np.float32) / 255.0
        self.pts += peso * a.reshape(self.H, 2, self.W, 2).mean(axis=(1, 3))

    def relleno(self, xs, ys, valor=0.2, regla="par-impar"):
        """Relleno translucido (membrana). Se acumula."""
        img = Image.new("L", (self.W * 2, self.H * 2), 0)
        d = ImageDraw.Draw(img)
        pts = list(zip((np.asarray(xs) * 2).tolist(), (np.asarray(ys) * 2).tolist()))
        d.polygon(pts, fill=255)
        a = np.asarray(img, np.float32) / 255.0
        a = a.reshape(self.H, 2, self.W, 2).mean(axis=(1, 3))
        self.mem += valor * a


# ------------------------------------------------------------------ revelado
def revelar(lz, nucleo=0.75, brillo=(0.55, 0.40, 0.30), radios=(2.5, 9, 28),
            ganancia=2.6, borde_mem=1.6, mem_alpha=0.85):
    """Convierte las acumulaciones en (I, A):
    I = intensidad de lineas con brillo difuso, en [0,1]
    A = opacidad de membrana, en [0,1]
    """
    L = lz.lin
    core = gaussian_filter(L, nucleo)
    v = core.copy()
    for b, r in zip(brillo, radios):
        v += b * gaussian_filter(L, r) * (r / 2.5) ** 0.5
    # bordes de las membranas: brillan como en una placa de rayos X
    if lz.mem.any():
        M = gaussian_filter(lz.mem, 0.8)
        v += borde_mem * gaussian_gradient_magnitude(M, 1.2)
        v += 0.25 * gaussian_filter(M, 6)
    if hasattr(lz, "pts"):
        P = gaussian_filter(lz.pts, 0.9)
        v += 1.2 * P + 0.5 * gaussian_filter(lz.pts, 5)
    I = 1 - np.exp(-ganancia * v)
    A = np.clip(mem_alpha * (1 - np.exp(-2.2 * gaussian_filter(lz.mem, 0.7))), 0, 1)
    return I.astype(np.float32), A.astype(np.float32)


def a_rgba(I, A, color_linea=BLANCO, color_mem=(70, 70, 72), alfa_extra=1.0):
    """Objeto RGBA: membrana gris translucida + lineas brillantes encima."""
    cl = np.array(color_linea, np.float32) / 255
    cm = np.array(color_mem, np.float32) / 255
    # 'over' premultiplicado: lineas sobre membrana
    a_l = np.clip(I, 0, 1)[..., None]
    a_m = np.clip(A, 0, 1)[..., None]
    C = cl * a_l + cm * a_m * (1 - a_l)
    a = a_l + a_m * (1 - a_l)
    a = np.clip(a * alfa_extra, 0, 1)
    rgb = np.where(a > 1e-6, C / np.maximum(a, 1e-6), 0)
    out = np.concatenate([rgb, a], axis=-1)
    return out  # flotante [0,1], no premultiplicado


def sobre(A, B):
    """A sobre B (RGBA flotantes no premultiplicados)."""
    aa = A[..., 3:4]
    ab = B[..., 3:4]
    a = aa + ab * (1 - aa)
    C = A[..., :3] * aa + B[..., :3] * ab * (1 - aa)
    rgb = np.where(a > 1e-6, C / np.maximum(a, 1e-6), 0)
    return np.concatenate([rgb, a], axis=-1)


def aplanar(RGBA, fondo=FONDO):
    bg = np.array(fondo, np.float32) / 255
    a = RGBA[..., 3:4]
    return RGBA[..., :3] * a + bg * (1 - a)


def guardar_rgba(RGBA, ruta):
    img = Image.fromarray((np.clip(RGBA, 0, 1) * 255 + 0.5).astype(np.uint8), "RGBA")
    img.save(ruta, optimize=True)


def guardar_rgb(RGB, ruta, calidad=92):
    img = Image.fromarray((np.clip(RGB, 0, 1) * 255 + 0.5).astype(np.uint8), "RGB")
    if ruta.endswith(".jpg"):
        img.save(ruta, quality=calidad, subsampling=0, optimize=True)
    else:
        img.save(ruta, optimize=True)


# ------------------------------------------------------------------ ventanas de color
def duotono(v, color, gamma=1.0, blanco=0.75, corte=0.68):
    """Mapa negro -> color -> color aclarado. v en [0,1]."""
    c = np.array(color, np.float32) / 255
    w = c + (1 - c) * blanco
    v = np.clip(v, 0, 1) ** gamma
    t1 = np.clip(v / corte, 0, 1)[..., None]
    t2 = np.clip((v - corte) / (1 - corte), 0, 1)[..., None]
    rgb = c * t1 * (1 - t2) + w * t2
    return rgb


def ventana(RGBA, cx, cy, w, h=None, zoom_f=1.7, color=NARANJA, contraste=1.0, fondo=0.05,
            aspecto=None):
    """Recorta (en fracciones) una region, la amplia y la tiñe. Devuelve RGB.
    Si se da 'aspecto' (ancho/alto fisico), h se calcula a partir de w."""
    H, W = RGBA.shape[:2]
    if aspecto is not None:
        h = w * W / (aspecto * H)
    ww = int(w * W / zoom_f)
    hh = int(h * H / zoom_f)
    x0 = int(cx * W - ww / 2)
    y0 = int(cy * H - hh / 2)
    x0 = min(max(x0, 0), W - ww)
    y0 = min(max(y0, 0), H - hh)
    crop = RGBA[y0:y0 + hh, x0:x0 + ww]
    lum = (crop[..., :3].mean(-1)) * crop[..., 3]
    lum = zoom(lum, zoom_f, order=3)
    lum = np.clip(lum, 0, None)
    lum = lum / (np.percentile(lum, 99.3) + 1e-6)
    lum = np.clip(lum, 0, 1) ** 1.35
    lum = np.clip(fondo + lum * contraste * 0.92, 0, 1)
    # un poco de brillo extra para que "queme"
    lum = np.clip(lum + 0.22 * gaussian_filter(lum, 6), 0, 1)
    return duotono(lum, color)


def grano(W, H, sigma=0.55, semilla=7):
    rng = np.random.default_rng(semilla)
    n = rng.standard_normal((H, W)).astype(np.float32)
    if sigma > 0:
        n = gaussian_filter(n, sigma)
    n /= n.std()
    return n
