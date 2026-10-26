from package_paths import PACKAGE_ROOT, resource, resource_glob, open_resource, output_dir
from pathlib import Path
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, Polygon, Rectangle
from matplotlib import colors
from matplotlib.text import Text
import numpy as np
from PIL import Image, ImageOps
import geopandas as gpd
import pyogrio
ROOT = PACKAGE_ROOT
OUT = output_dir('Fig_1')
ASSETS = OUT / 'assets'
ORIGINAL = ROOT / 'All figures/最终图/正文图+扩展数据图/概念图.png'
W, H = (300, 205)
C = {'ink': '#1D314A', 'muted': '#60718A', 'line': '#CCD6E3', 'blue': '#3667B2', 'dark': '#174987', 'light': '#DCE8F7', 'pale': '#F3F6FA', 'teal': '#508F9A', 'orange': '#F27C2B', 'land': '#DBE2EC', 'white': '#FFFFFF', 'green': '#82AA9E'}
plt.rcParams.update({'font.family': 'Arial', 'font.size': 10, 'text.color': C['ink'], 'svg.fonttype': 'none', 'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.unicode_minus': True, 'image.composite_image': False})
fig, ax = plt.subplots(figsize=(W / 25.4, H / 25.4))
fig.subplots_adjust(0, 0, 1, 1)
ax.set(xlim=(0, W), ylim=(0, H), aspect='equal')
ax.axis('off')

def text(x, y, label, size=10, color=None, weight='normal', ha='left', va='center', **kw):
    return ax.text(x, y, label, fontsize=size, color=color or C['ink'], fontweight=weight, ha=ha, va=va, zorder=30, **kw)

def line(points, color=None, lw=0.8, zorder=5, **kw):
    p = np.array(points)
    return ax.plot(p[:, 0], p[:, 1], color=color or C['line'], lw=lw, zorder=zorder, solid_capstyle='round', **kw)[0]

def rect(x, y, w, h, fc='white', ec=None, lw=0.65, zorder=3, **kw):
    p = Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec or 'none', linewidth=lw, zorder=zorder, **kw)
    ax.add_patch(p)
    return p

def poly(points, fc, ec='white', lw=0.45, zorder=4, **kw):
    p = Polygon(points, facecolor=fc, edgecolor=ec, lw=lw, zorder=zorder, **kw)
    ax.add_patch(p)
    return p

def circle(x, y, r, fc, ec='none', lw=0.7, zorder=10):
    p = Circle((x, y), r, facecolor=fc, edgecolor=ec, lw=lw, zorder=zorder)
    ax.add_patch(p)
    return p

def arrow(points, color=None, lw=0.95, ms=8, dashed=False):
    color = color or C['muted']
    if len(points) > 2:
        line(points[:-1], color, lw, linestyle='--' if dashed else '-')
    p = FancyArrowPatch(points[-2], points[-1], arrowstyle='-|>', mutation_scale=ms, color=color, linewidth=lw, shrinkA=0, shrinkB=0, zorder=15, linestyle='--' if dashed else '-')
    ax.add_patch(p)

def tint(c, amount):
    return np.array(colors.to_rgb(c)) * (1 - amount) + amount

def photo(name, x, y, w, h):
    image = Image.open(resource(ASSETS / (name + '.png'))).convert('RGB')
    image = ImageOps.fit(image, (int(w * 22), int(h * 22)), method=Image.Resampling.LANCZOS)
    ax.imshow(image, extent=(x, x + w, y, y + h), zorder=8)
    rect(x, y, w, h, 'none', C['line'], lw=0.65, zorder=9)

def stage(x, end, number, title, subtitle):
    text(x, 196.7, number, 11, C['blue'], 'bold')
    text(x + 9, 196.7, title, 14, weight='bold')
    text(x + 9, 189.8, subtitle, 8.8, C['muted'])
    line([(x, 184.8), (end, 184.8)], C['line'], 0.8)
    line([(x, 184.8), (x + 7, 184.8)], C['blue'], 2)

def iso(cx, cy, i, j, s=3.0, z=0):
    return np.array([cx + (i - j) * s, cy + (i + j) * s * 0.46 + z])

def tile(cx, cy, i, j, s, fc, z=0, edge='white', lw=0.45, order=5):
    vertices = [iso(cx, cy, i, j, s, z), iso(cx, cy, i + 1, j, s, z), iso(cx, cy, i + 1, j + 1, s, z), iso(cx, cy, i, j + 1, s, z)]
    return poly(vertices, fc, edge, lw, order)

def building(cx, cy, i, j, s, height, fc, order=12):
    pad = 0.13
    a = iso(cx, cy, i + pad, j + pad, s)
    b = iso(cx, cy, i + 1 - pad, j + pad, s)
    c = iso(cx, cy, i + 1 - pad, j + 1 - pad, s)
    d = iso(cx, cy, i + pad, j + 1 - pad, s)
    up = np.array([0, height])
    poly([a, b, b + up, a + up], tint(fc, 0.06), 'white', 0.35, order)
    poly([a, d, d + up, a + up], tint(fc, 0.28), 'white', 0.35, order + 0.1)
    poly([a + up, b + up, c + up, d + up], tint(fc, 0.49), 'white', 0.35, order + 0.2)
    for frac in [0.33, 0.66]:
        p = (a + b) / 2 + up * frac
        line([p - [0.22, 0.05], p + [0.22, 0.05]], 'white', 0.5, order + 0.3)

def icon(kind, x, y, s=1, color=None):
    c = color or C['blue']

    def L(p, lw=0.9):
        return line([(x + a * s, y + b * s) for a, b in p], c, lw)

    def R(a, b, w, h, fc='none'):
        return rect(x + a * s, y + b * s, w * s, h * s, fc, c, 0.8, 10)

    def O(a, b, r, fc='white'):
        return circle(x + a * s, y + b * s, r * s, fc, c, 0.8)
    if kind == 'green':
        L([(0, -2.6), (0, 1.8)])
        O(0, 1.8, 1.8, tint(c, 0.78))
        O(-1.2, 0.5, 1.35, tint(c, 0.78))
        O(1.2, 0.5, 1.35, tint(c, 0.78))
        L([(-3, -2.6), (3, -2.6)])
    elif kind == 'built':
        R(-2.5, -2.5, 2.2, 5)
        R(0.4, -2.5, 2.2, 3.6)
        for a in [-1.8, 1.1]:
            L([(a, -0.9), (a + 0.6, -0.9)], 0.7)
            L([(a, 0.4), (a + 0.6, 0.4)], 0.7)
        L([(-3, -2.5), (3.2, -2.5)])
    elif kind == 'poi':
        O(0, 1, 1.7, tint(c, 0.85))
        O(0, 1, 0.55, 'white')
        L([(-1.4, 0), (0, -2.6), (1.4, 0)])
    elif kind == 'risk':
        poly([(x - 2.5 * s, y + 2.4 * s), (x + 2.5 * s, y + 2.4 * s), (x + 2 * s, y - 1.2 * s), (x, y - 3 * s), (x - 2 * s, y - 1.2 * s)], tint(c, 0.88), c, 0.85, 10)
        L([(0, 1.3), (0, -0.5)], 1)
        O(0, -1.4, 0.13, c)
    elif kind == 'road':
        L([(-2.4, -2.7), (-1.2, 2.6)])
        L([(2.4, -2.7), (1.2, 2.6)])
        L([(0, -2), (0, -0.8)])
        L([(0, 0.7), (0, 2)])
    elif kind == 'terrain':
        L([(-3, -2), (0, 2.6), (3, -2), (-3, -2)])
        L([(-1.1, 0.9), (0, 0.1), (1, 0.9)])
    elif kind == 'location':
        O(0, 0, 2.7, 'none')
        ax.add_patch(Ellipse((x, y), 2.5 * s, 5.4 * s, fc='none', ec=c, lw=0.8, zorder=10))
        L([(-2.7, 0), (2.7, 0)])
    elif kind == 'people':
        O(-1.1, 1.6, 0.9, tint(c, 0.65))
        O(1.5, 0.9, 0.7, tint(c, 0.65))
        L([(-2.3, -2.2), (-2.3, -0.2), (-1.1, 0.4), (0.1, -0.2), (0.1, -2.2)])
        L([(0.7, -2.2), (0.7, -0.7), (1.5, -0.2), (2.4, -0.7), (2.4, -2.2)])
    elif kind == 'mobility':
        O(-1.9, -1.2, 1.15, 'none')
        O(2, -1.2, 1.15, 'none')
        L([(-1.9, -1.2), (-0.5, 1), (0.7, -1.2), (-1.9, -1.2)])
        L([(0.7, -1.2), (2, -1.2), (0.7, 2), (-0.1, 2)])
    elif kind == 'satellite':
        poly([(x - s, y), (x, y + s), (x + s, y), (x, y - s)], tint(c, 0.6), c, 0.6, 10)
        poly([(x - 4 * s, y + 1 * s), (x - 2 * s, y + 2 * s), (x - s, y + 0.5 * s), (x - 3 * s, y - 0.5 * s)], tint(c, 0.6), c, 0.6, 10)
        poly([(x + s, y - 0.5 * s), (x + 3 * s, y + 0.5 * s), (x + 4 * s, y - s), (x + 2 * s, y - 2 * s)], tint(c, 0.6), c, 0.6, 10)
        L([(0, -1.1), (0, -2.4)])
    elif kind == 'camera':
        R(-3, -1.8, 6, 3.8)
        O(0, 0, 1.2, 'white')
        R(-1.6, 2, 2.2, 0.8, tint(c, 0.8))

def draw_environment():
    text(8, 82, 'Environmental predictors', 11.2, weight='bold')
    text(8, 76, 'Remote sensing + contextual data', 9, C['muted'])
    for layer, z in enumerate([0, 5.5, 11]):
        for i in range(6):
            for j in range(5):
                v = (i * 7 + j * 3 + layer * 5) % 13 / 12
                fc = tint([C['teal'], C['blue'], C['land']][layer], 0.35 + 0.5 * v)
                tile(31, 30, i, j, 3.6, fc, z, lw=0.35, order=6 + layer)
        p = iso(31, 30, 6, 2, 3.6, z)
        line([p, p + [6, 0]], C['line'], 0.65, 20)
        text(p[0] + 7, p[1], ['Land cover', 'Built form', 'Environment'][layer], 8.2, C['muted'])
    icon('satellite', 21, 61, 1.35)
    line([(23, 58), (28, 52)], C['line'], 0.65, linestyle=(0, (2, 2)))
    text(8, 21, 'Spatial covariates for prediction', 9, C['muted'])

def draw_model():
    rect(103, 150, 80, 28, C['pale'], C['line'], 0.7)
    text(143, 171.5, 'Image-based perception model', 11.1, weight='bold', ha='center')
    photo('place_pulse_street', 108, 155, 14, 9)
    arrow([(124, 159.5), (131, 159.5)], C['blue'], 0.85, 6)
    nodes = []
    for k, num in enumerate([3, 4, 3]):
        xs = 135 + k * 8
        ys = np.linspace(154.3, 164, num)
        nodes.append([(xs, y) for y in ys])
    for aa, bb in zip(nodes[:-1], nodes[1:]):
        for a in aa:
            for b in bb:
                line([a, b], '#CBD9EB', 0.55, 6)
    for col in nodes:
        for x, y in col:
            circle(x, y, 0.85, 'white', C['blue'], 0.7)
    arrow([(154, 159.5), (161, 159.5)], C['blue'], 0.85, 6)
    for k in range(6):
        rect(164 + k * 2.4, 155, 1.5, 4 + k * 3 % 5, tint(C['blue'], 0.12 + k * 0.08))
    text(171, 152.3, '6 scores', 7.8, C['muted'], ha='center')

def draw_georeferenced():
    text(143, 138.5, 'Georeferenced perception scores', 10.9, weight='bold', ha='center')
    rect(113, 106.3, 60, 25.5, '#F7F9FC')
    roads = [[(113, 111), (125, 115), (137, 111), (149, 116), (160, 114), (173, 120)], [(114, 124), (125, 122), (137, 125), (150, 122), (161, 126), (173, 124)], [(121, 107), (125, 115), (125, 122), (127, 131)], [(136, 107), (137, 111), (137, 125), (140, 132)], [(153, 107), (149, 116), (150, 122), (151, 131)], [(167, 107), (160, 114), (161, 126), (164, 132)]]
    for road in roads:
        line(road, '#D5DEEB', 1.4)
    for road in roads:
        for k, (p, q) in enumerate(zip(road[:-1], road[1:])):
            p, q = (np.array(p), np.array(q))
            for t in [0.24, 0.56, 0.82]:
                v = p + (q - p) * t
                circle(*v, 0.6, tint(C['blue'], k % 3 * 0.2), 'white', 0.25)
    for x, y, a, b in [(111, 105, 4, 4), (175, 105, -4, 4), (111, 133, 4, -4), (175, 133, -4, -4)]:
        line([(x, y + b), (x, y), (x + a, y)], C['blue'], 0.75)

def draw_stacking():
    rect(103, 74, 80, 21, C['pale'], C['line'], 0.7)
    text(143, 89.5, 'Spatial prediction', 11.1, weight='bold', ha='center')
    text(143, 79.7, 'Stacking model + environmental predictors', 8.7, C['muted'], ha='center')

def draw_field():
    text(143, 64.8, '50 m perception fields', 11.2, weight='bold', ha='center')
    cx, cy, s = (143, 31.2, 3.45)
    for i in range(10):
        for j in range(10):
            v = (np.sin(i * 0.7) + np.cos(j * 0.6) + 2) / 4
            fc = plt.get_cmap('Blues')(0.16 + 0.6 * v)
            tile(cx, cy, i, j, s, fc, lw=0.45, order=5)
    for i in range(10):
        p = iso(cx, cy, i, 0, s)
        q = iso(cx, cy, i + 1, 0, s)
        poly([p, q, q - [0, 1.4], p - [0, 1.4]], '#B3C9E4', 'white', 0.3, 6)
    for j in range(10):
        p = iso(cx, cy, 0, j, s)
        q = iso(cx, cy, 0, j + 1, s)
        poly([p, q, q - [0, 1.4], p - [0, 1.4]], '#D5E1F0', 'white', 0.3, 6)
    for i, j, h in [(7, 7, 7), (6, 7, 5), (7, 6, 4), (6, 6, 5.5), (3, 7, 2.7), (8, 4, 3.2)]:
        building(cx, cy, i, j, s, h, C['blue'], 30 - i - j)
    p = iso(cx, cy, 6, 0, s) + [1.5, -4]
    q = iso(cx, cy, 7, 0, s) + [1.5, -4]
    line([p, q], C['muted'], 0.75, 20)
    line([p + [-0.35, 0.5], p + [0.35, -0.5]], C['muted'], 0.75, 20)
    line([q + [-0.35, 0.5], q + [0.35, -0.5]], C['muted'], 0.75, 20)
    text(q[0] + 2, q[1] - 1.5, '50 m', 8.2, C['muted'])
    text(143, 25, 'Beauty +   Wealth +   Safety +', 9.1, ha='center')
    text(143, 19.8, 'Liveliness +   Boredom −   Depression −', 9.1, ha='center')
    line([(117, 15.3), (169, 15.3)], C['line'], 0.7)
    text(143, 10.5, 'Emotional livability index (ELI)', 10.4, C['blue'], 'bold', ha='center')

def qhead(y, n, label):
    text(203, y, n, 10.2, C['blue'], 'bold')
    text(214, y, label, 11.6, weight='bold')

def draw_q1():
    qhead(176.5, 'Q1', 'Global patterns & disparities')
    shp = next(resource_glob(Path(pyogrio.__file__).resolve().parent, 'naturalearth_lowres.shp'))
    world = gpd.read_file(resource(shp)).to_crs(4326)
    world = world[world.name != 'Antarctica']
    mx, my, mw, mh = (204, 143, 52, 25)

    def ring(coords):
        p = np.asarray(coords)
        return np.column_stack((mx + (p[:, 0] + 180) / 360 * mw, my + (p[:, 1] + 60) / 150 * mh))
    for geom in world.geometry:
        polys = list(geom.geoms) if geom.geom_type == 'MultiPolygon' else [geom]
        for p in polys:
            verts = ring(p.exterior.coords)
            poly(verts, C['land'], 'white', 0.15, 5)
    for y, label, c in [(159.5, 'Global North', C['blue']), (146.8, 'Global South', C['orange'])]:
        circle(263, y, 1.3, c)
        text(267, y, label, 9.4)
    line([(257, 153), (260, 153), (260, 159.5), (261.7, 159.5)], C['line'], 0.75)
    line([(260, 153), (260, 146.8), (261.7, 146.8)], C['line'], 0.75)
    text(203, 134, 'City means · inequality · group comparisons', 9.1, C['muted'])

def draw_q2():
    qhead(120, 'Q2', 'Within-city contrasts')
    cx, cy, s = (249, 85, 3.6)
    for i in range(8):
        for j in range(8):
            core = 2 <= i <= 5 and 2 <= j <= 5
            fc = tint(C['blue'], 0.45) if core else '#EDF1F6'
            tile(cx, cy, i, j, s, fc, edge='white', lw=0.5)
    for i, j, h in [(4, 5, 6.5), (5, 4, 4), (4, 4, 7), (3, 4, 5), (5, 3, 3), (3, 3, 4.5), (1, 6, 2), (6, 1, 2.2), (1, 2, 2.6), (6, 6, 3)]:
        core = 2 <= i <= 5 and 2 <= j <= 5
        building(cx, cy, i, j, s, h, C['blue'] if core else '#B4C2D5', 30 - i - j)
    corepoint = iso(cx, cy, 3.3, 3.3, s, 3)
    line([corepoint, [233, 96], [222, 96]], C['blue'], 0.8, 20)
    text(219, 96, 'Core', 9.4, C['blue'], ha='right')
    exppoint = iso(cx, cy, 6.8, 1.2, s)
    line([exppoint, [274, 88], [277, 88]], C['muted'], 0.8, 20)
    text(278, 88, 'Expansion', 9.4, C['muted'], ha='left')
    text(203, 78, 'Core versus areas urbanized in 2000–2020', 9.1, C['muted'])

def draw_tree(x, y, s=1):
    circle(x, y + 4 * s, 0.7 * s, C['blue'])
    for a in [-3, 3]:
        line([(x, y + 3.2 * s), (x, y + 1.7 * s), (x + a * s, y + 1.7 * s), (x + a * s, y)], C['blue'], 0.7)
        circle(x + a * s, y - 0.4 * s, 0.7 * s, C['blue'])
        for b in [-1.6, 1.6]:
            line([(x + a * s, y - 1.2 * s), (x + a * s, y - 2 * s), (x + (a + b) * s, y - 2 * s), (x + (a + b) * s, y - 3 * s)], C['blue'], 0.6)
            circle(x + (a + b) * s, y - 3.5 * s, 0.5 * s, C['blue'])

def draw_q3():
    qhead(64, 'Q3', 'Environmental associations')
    text(203, 56.5, 'Explainable machine learning', 9.2, C['muted'])
    draw_tree(212, 46.8, 0.75)
    arrow([(220, 47), (226, 47)], C['muted'], 0.7, 5)
    for k, length in enumerate([10, 7, 4]):
        rect(230, 49.8 - k * 2.6, length, 1.1, tint(C['blue'], k * 0.22))
    text(244, 47, 'Variable importance', 8.6, C['muted'])
    items = [('green', 'Green / blue'), ('built', 'Built environment'), ('risk', 'Environmental risk'), ('people', 'Economy / population'), ('poi', 'POI supply'), ('mobility', 'Non-motorized'), ('road', 'Road network'), ('terrain', 'Terrain'), ('location', 'Location')]
    for k, (kind, label) in enumerate(items):
        row, col = divmod(k, 3)
        xx = 216 + col * 30.3
        yy = 34 - row * 11.5
        icon(kind, xx, yy, 0.86)
        text(xx, yy - 4.5, label, 7.8, C['muted'], ha='center')

def build():
    stage(8, 86, '01', 'Observe & learn', 'Images, human judgments and spatial covariates')
    stage(103, 183, '02', 'Map perception', 'From image scores to continuous urban fields')
    stage(203, 292, '03', 'Compare across scales', 'Three linked research questions')
    text(8, 177, 'Global street-view imagery', 11.2, weight='bold')
    photo('street_view', 8, 150, 78, 22)
    text(8, 144.3, '~66 million images', 9.3, C['muted'])
    icon('camera', 79.5, 177, 0.8)
    text(8, 131.5, 'Human perception labels', 11.2, weight='bold')
    text(8, 125.5, 'Place Pulse', 9.3, C['muted'])
    photo('place_pulse_green', 8, 98.5, 37.5, 22.2)
    photo('place_pulse_street', 48.5, 98.5, 37.5, 22.2)
    line([(23, 95), (23, 92.5), (71, 92.5), (71, 95)], C['line'], 0.85)
    text(47, 88.5, 'Pairwise human judgments', 9.1, C['muted'], ha='center')
    draw_environment()
    draw_model()
    arrow([(87, 164), (102, 164)], C['blue'], 1, 7)
    text(94.5, 168, 'Score', 8.3, C['blue'], ha='center')
    arrow([(87, 110), (94, 110), (94, 154), (102, 154)], C['muted'], 0.9, 7)
    text(94, 116, 'Train', 8.3, C['muted'], ha='center', rotation=90, bbox={'fc': 'white', 'ec': 'none', 'pad': 1.2})
    arrow([(143, 149), (143, 143.5)], C['blue'], 0.9, 7)
    draw_georeferenced()
    arrow([(143, 104), (143, 96.5)], C['blue'], 0.9, 7)
    draw_stacking()
    arrow([(84, 52), (96, 52), (96, 84), (102, 84)], C['muted'], 0.9, 7)
    arrow([(143, 73), (143, 69)], C['blue'], 0.9, 7)
    draw_field()
    line([(177.5, 47.1), (193, 47.1), (193, 167)], C['blue'], 1)
    for y in [167, 110, 52]:
        circle(193, y, 0.8, C['blue'])
        arrow([(193, y), (200, y)], C['blue'], 0.9, 7)
    line([(203, 129), (292, 129)], C['line'], 0.75)
    line([(203, 72), (292, 72)], C['line'], 0.75)
    draw_q1()
    draw_q2()
    draw_q3()
    text(8, 8.5, 'Illustrations are schematic; no measured results are shown.', 8.1, C['muted'])
    text(8, 3.6, 'Street-view photographs reused from the supplied figure (Google / Place Pulse).', 7.7, C['muted'])
    return fig

def save():
    OUT.mkdir(parents=True, exist_ok=True)
    build()
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    outside = []
    for t in fig.findobj(match=lambda a: isinstance(a, Text)):
        if t.get_visible() and t.get_text():
            b = t.get_window_extent(renderer)
            if b.x0 < 0 or b.y0 < 0 or b.x1 > fig.bbox.width or (b.y1 > fig.bbox.height):
                outside.append(t.get_text())
    assert not outside, f'Text outside figure: {outside}'
    stem = OUT / '概念图_Python重绘'
    for ext in ['pdf', 'svg']:
        fig.savefig(stem.with_suffix('.' + ext), facecolor='white')
    fig.savefig(stem.with_suffix('.png'), dpi=600, facecolor='white')
    fig.savefig(OUT / 'preview.png', dpi=185, facecolor='white')
    im = Image.open(resource(OUT / 'preview.png')).convert('L')
    im.save(OUT / 'preview_grayscale.png')
    audit = {'original_figure': str(ORIGINAL), 'original_sha256': hashlib.sha256(resource(ORIGINAL).read_bytes()).hexdigest(), 'canvas_mm': [W, H], 'font': 'Arial', 'svg_text_editable': True, 'source_photographs': 'Cropped from the source concept figure.', 'world_geometry': 'Bundled Natural Earth low resolution, EPSG:4326; neutral land, no study coverage encoded.', 'schematic_elements': ['street locations and scores', '50 m tile values', 'city buildings', 'model network', 'variable importance glyph'], 'concepts_preserved': ['~66 million street-view images', 'Place Pulse labels', 'image perception model', 'georeferenced scores', 'stacking and remote sensing', '50 m fields', 'six perception dimensions', 'ELI', 'Q1 global disparities', 'Q2 core-expansion', 'Q3 environmental associations', 'nine predictor families'], 'no_estimated_results_or_statistics': True, 'text_outside_canvas': outside, 'minimum_text_pt': min((t.get_fontsize() for t in fig.findobj(match=lambda a: isinstance(a, Text)) if t.get_text())), 'exports': [str(stem.with_suffix('.' + e)) for e in ['png', 'pdf', 'svg']]}
    (OUT / 'design_audit.json').write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({'output': str(OUT), 'canvas_mm': [W, H], 'minimum_text_pt': audit['minimum_text_pt'], 'text_outside_canvas': outside}, ensure_ascii=False))

def _script_entry():
    save()
