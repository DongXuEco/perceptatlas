from package_paths import PACKAGE_ROOT, resource, resource_glob, open_resource, output_dir
from pathlib import Path
import numpy as np
from matplotlib.patches import Circle, Polygon
import contourpy
GROUND_ANCHORS = []
_WORLD = None

def draw_miniature(r, kind, x, y, s=1):
    C, ax = (r.C, r.ax)
    poly, line, oval, curve = (r.poly, r.line, r.oval, r.curve)
    unit = 2.1 * s
    cx = x - unit * 0.5
    P = lambda i, j, z=0: r.iso(cx, y, i, j, unit, z * s)
    r.base(cx, y, s=unit, fc='#EDF1F4')
    ground = Polygon([P(0, 0), P(4, 0), P(4, 3), P(0, 3)], closed=True, transform=ax.transData)
    jobs = []

    def patch(vertices, fc, z=7, ec='white', lw=0.2):
        return poly([P(*v) for v in vertices], fc, ec, lw, z)

    def paving(i0, j0, i1, j1, fc, z=7):
        p = patch([(i0, j0), (i1, j0), (i1, j1), (i0, j1)], fc, z)
        p.set_clip_path(ground)

    def register(label, i, j, draw, footprint=None):
        assert 0.2 <= i <= 3.8 and 0.2 <= j <= 2.8, (kind, label, i, j)
        if footprint:
            assert all((0.12 <= a <= 3.88 and 0.12 <= b <= 2.88 for a, b in footprint)), (kind, label, footprint)
        GROUND_ANCHORS.append({'scene': kind, 'object': label, 'root_ij': [i, j], 'footprint_ij': footprint, 'on_ground_slab': True})
        jobs.append((i + j, draw))

    def planted(i, j, size=0.47):

        def draw(z):
            first = len(ax.patches)
            r.tree(*P(i, j), size * s, z=z, fc='#7EA696')
            ax.patches[first].set_clip_path(ground)
            p = P(i, j)
            line([p + [-0.18 * s, 0], p + [0.18 * s, 0]], '#9AAA9D', 0.35, z + 0.1)
        register('tree', i, j, draw, [(i - 0.3, j - 0.3), (i + 0.3, j + 0.3)])

    def pedestrian(i, j, size=0.52, bag=False):

        def draw(z):
            first = len(ax.patches)
            r.person(*P(i, j), size * s, z=z, fc='#5379A0')
            ax.patches[first].set_clip_path(ground)
            if bag:
                q = P(i, j)
                r.rect(q[0] + 0.42 * size * s, q[1] + 0.55 * size * s, 0.6 * size * s, 0.65 * size * s, '#A48E78', zorder=z + 3.3)
        register('pedestrian', i, j, draw, [(i - 0.22, j - 0.22), (i + 0.22, j + 0.22)])

    def cube(i, j, di, dj, h, fc, role='building'):

        def draw(z):
            a, b, c, d = (P(i, j), P(i + di, j), P(i + di, j + dj), P(i, j + dj))
            up = np.array([0, h * s])
            shadow = poly([a + [0.5 * s, -0.15 * s], b + [0.5 * s, -0.15 * s], c + [0.5 * s, -0.15 * s], d + [0.5 * s, -0.15 * s]], '#56718C', 'none', 0, z - 0.1, alpha=0.1)
            shadow.set_clip_path(ground)
            poly([a, b, b + up, a + up], r.tint(fc, 0.04), 'white', 0.22, z)
            poly([a, d, d + up, a + up], r.tint(fc, 0.29), 'white', 0.22, z + 0.1)
            poly([a + up, b + up, c + up, d + up], r.tint(fc, 0.66), 'white', 0.22, z + 0.2)
            floors = max(2, int(h / 0.65))
            columns = 3 if di > 1 else 2
            for upper, side in [(b, 0), (d, 1)]:
                for row in range(floors):
                    hh = 0.32 + (h - 0.65) * (row + 0.25) / floors
                    for col in range(columns):
                        t = (col + 0.18) / columns
                        tt = (col + 0.73) / columns
                        q = a + (upper - a) * t + [0, hh * s]
                        q2 = a + (upper - a) * tt + [0, hh * s]
                        poly([q, q2, q2 + [0, 0.28 * s], q + [0, 0.28 * s]], '#D9EAF5' if side == 0 else '#ECF4F7', 'none', 0, z + 0.3)
            q = P(i + di * 0.38, j, 0.05)
            q2 = P(i + di * 0.63, j, 0.05)
            poly([q, q2, q2 + [0, 0.58 * s], q + [0, 0.58 * s]], '#587D9E', 'white', 0.12, z + 0.4)
            line([a + up, b + up], tuple(r.tint(fc, 0.82)), 0.45, z + 0.4)
            roof = (a + b + c + d) / 4 + up
            poly([roof + [-0.36 * s, 0], roof + [0, 0.18 * s], roof + [0.47 * s, 0], roof + [0.1 * s, -0.18 * s]], r.tint(fc, 0.4), 'white', 0.12, z + 0.5)
            if role == 'hospital':
                q = P(i + di * 0.5, j, h * 0.72)
                along = (b - a) / np.linalg.norm(b - a) * 0.48 * s
                line([q - along, q + along], '#47789A', 0.8, z + 0.7)
                line([q + [0, -0.48 * s], q + [0, 0.48 * s]], '#47789A', 0.8, z + 0.7)
            if role == 'shop':
                frontj = j - 0.25
                for k in range(5):
                    left = i + di * k / 5
                    right = i + di * (k + 1) / 5
                    patch([(left, j, 1.38), (right, j, 1.38), (right, frontj, 1.13), (left, frontj, 1.13)], '#D29B79' if k % 2 == 0 else '#F5EADC', z + 0.7, lw=0.12)
                q = P(i + 0.12, j, 0.3)
                q2 = P(i + di - 0.12, j, 0.3)
                poly([q, q2, q2 + [0, 0.65 * s], q + [0, 0.65 * s]], '#C7DBE4', 'white', 0.12, z + 0.6)
        register(role, i + di * 0.5, j + dj * 0.5, draw, [(i, j), (i + di, j + dj)])

    def sedan(i, j, size=0.38):

        def draw(z):
            first = len(ax.patches)
            r.q2_car(*P(i, j), s=size * s, z=z)
            ax.patches[first].set_clip_path(ground)
        register('car', i, j, draw, [(i - 0.48, j - 0.2), (i + 0.48, j + 0.2)])
    if kind == 'green':
        paving(0.18, 0.18, 3.82, 2.82, '#DCE8D9')
        pond = [(2.1, 0.65), (2.9, 0.35), (3.75, 0.8), (3.45, 1.35), (3.3, 1.9), (2.5, 1.9), (2.1, 1.35), (1.9, 1.1), (1.75, 0.8), (2.1, 0.65)]
        vertices = [P(*v) for v in pond]
        path = r.PathPatch(r.MPath(vertices, [r.MPath.MOVETO] + [r.MPath.CURVE4] * 9), facecolor='#9CC9D8', edgecolor='#D5E9DC', linewidth=1.2 * s, zorder=8)
        path.set_clip_path(ground)
        ax.add_patch(path)
        curve([P(0.3, 0.55), P(1.5, 1.35), P(1.3, 1.8), P(3.5, 2.55)], '#F6F3E7', 1.1 * s, 9).set_clip_path(ground)
        planted(0.8, 2.15, 0.49)
        planted(1.6, 2.25, 0.44)
        planted(0.75, 0.72, 0.45)

        def bench(z):
            a = P(2.6, 2.1, 0.3)
            q = P(3.35, 2.1, 0.3)
            line([a, q], '#9E8B71', 1.1 * s, z + 1)
            line([a + [0, 0.55 * s], q + [0, 0.55 * s]], '#AD9A7D', 0.65 * s, z + 1.1)
            for i in [2.65, 3.25]:
                line([P(i, 2.1), P(i, 2.1, 0.45)], '#6D817C', 0.45 * s, z)
        register('park_bench', 2.95, 2.1, bench, [(2.6, 1.95), (3.4, 2.25)])
    elif kind == 'built':
        paving(0.2, 0.2, 3.8, 1.2, '#DDE5EB')
        for i in [0.8, 1.6, 2.4, 3.2]:
            line([P(i, 0.25), P(i, 1.15)], '#EEF2F5', 0.35, 8)
        cube(0.55, 1.55, 1, 1.05, 2.45, '#7594B8')
        cube(2.05, 1.65, 1.05, 1.05, 3.05, '#527FB3')
        cube(2.65, 0.45, 0.95, 0.9, 1.55, '#96AFC5')
        paving(0.35, 0.3, 1.15, 1.1, '#D8E5DC', 8)
        planted(0.75, 0.7, 0.43)
        sedan(1.85, 0.65, 0.31)
    elif kind == 'risk':
        paving(0.2, 0.4, 3.8, 1.25, '#CAD5DF')
        cube(0.5, 1.55, 1.05, 1, 2.25, '#8DA6BF')
        cube(1.95, 1.6, 0.95, 1.05, 2.8, '#A3B5C5')
        sedan(1.25, 0.83, 0.34)

        def station(z):
            q = P(3.35, 1.35)
            oval(*q, 1.3 * s, 0.5 * s, '#C7D2DC', z=z)
            line([q + [0, 0.3 * s], q + [0, 2.7 * s]], '#D4DDE3', 1.5 * s, z + 1)
            line([q + [0, 0.4 * s], q + [0, 2.45 * s]], '#EDA371', 0.7 * s, z + 1.1)
            r.circle(q[0], q[1] + 0.45 * s, 0.38 * s, '#E89B68', zorder=z + 1.2)
            for h in [1.1, 1.65, 2.2]:
                line([q + [0.45 * s, h * s], q + [0.75 * s, h * s]], '#A7B7C7', 0.35, z + 1.3)
        register('temperature_sensor', 3.35, 1.35, station, [(3.1, 1.1), (3.6, 1.6)])
        for xx, yy, rr in [(-1, 5.7, 0.18), (-0.35, 6.1, 0.23), (0.4, 5.8, 0.17), (0.9, 6.3, 0.16)]:
            r.circle(x + xx * s, y + yy * s, rr * s, '#BAC6CE', zorder=9)
    elif kind == 'people':
        paving(0.25, 0.25, 3.75, 1.45, '#E4E4DF')
        for i in [0.8, 1.6, 2.4, 3.2]:
            line([P(i, 0.3), P(i, 1.4)], '#F7F7F3', 0.4, 8)
        cube(1.75, 1.65, 1.25, 0.95, 2.7, '#8FA8BD')
        pedestrian(0.7, 0.7, 0.52, True)
        pedestrian(1.7, 0.75, 0.49)
        pedestrian(2.8, 0.7, 0.5, True)
        pedestrian(0.8, 1.75, 0.47)
    elif kind == 'poi':
        paving(0.22, 0.25, 3.78, 1.32, '#E7E7E0')
        cube(0.5, 1.55, 1.12, 1.0, 2.45, '#C1D5DB', 'hospital')
        cube(2.35, 1.65, 1.13, 0.9, 1.85, '#BAAA93', 'shop')
        pedestrian(1.75, 0.8, 0.45)
        paving(0.45, 0.3, 1.15, 1.0, '#D7E5DD', 8)
        planted(0.8, 0.65, 0.35)
    elif kind == 'mobility':
        paving(0.2, 0.25, 3.8, 2.78, '#DCE7DD')
        paving(0.25, 0.6, 3.75, 1.55, '#C3D4D2', 8)
        for i in [0.4, 1.2, 2.0, 2.8]:
            line([P(i, 1.1), P(i + 0.35, 1.1)], '#F8FAF4', 0.4, 9)
        planted(0.8, 2.2, 0.44)
        planted(3.25, 2.4, 0.4)
        pedestrian(0.7, 0.85, 0.56)

        def cyclist(z):
            rear, front = (P(1.65, 1.05, 0.62), P(3.15, 1.05, 0.62))
            for q in [rear, front]:
                oval(*q, 1.1 * s, 1.22 * s, 'none', '#456789', 0.52, z=z)
                oval(*q, 0.88 * s, 1.0 * s, 'none', '#B8CBD9', 0.2, z=z + 0.1)
                r.circle(*q, 0.075 * s, '#6D8EAA', zorder=z + 0.15)
            seat, pedal, headtube = (P(2.08, 1.05, 1.65), P(2.25, 1.05, 0.52), P(2.85, 1.05, 1.78))
            line([rear, seat, pedal, rear], '#477A9E', 0.5, z + 0.2)
            line([seat, headtube, pedal], '#477A9E', 0.5, z + 0.2)
            line([headtube, front], '#426C8B', 0.5, z + 0.2)
            stem = P(2.83, 1.05, 2.13)
            grip = P(3.02, 1.05, 2.18)
            line([headtube, stem, grip, P(3.12, 1.05, 2.0)], '#59748B', 0.42, z + 0.3)
            line([seat + [-0.29 * s, 0.08 * s], seat + [0.3 * s, 0.08 * s]], '#405D75', 0.7, z + 0.3)
            r.circle(*pedal, 0.12 * s, '#718DA5', zorder=z + 0.25)
            line([pedal, P(2.4, 1.05, 0.35)], '#718DA5', 0.35, z + 0.3)
            hip = P(2.1, 1.05, 1.84)
            shoulder = P(2.35, 1.05, 2.54)
            line([hip, P(1.98, 1.05, 1.05), P(2.4, 1.05, 0.35)], '#9CADBA', 0.65, z + 0.3)
            line([hip, shoulder], '#789AB3', 1.35 * s, z + 0.4)
            head = P(2.44, 1.05, 3.03)
            r.circle(*head, 0.235 * s, '#D0B49A', zorder=z + 0.6)
            oval(head[0] - 0.025 * s, head[1] + 0.15 * s, 0.52 * s, 0.26 * s, '#AAC2CE', z=z + 0.7)
            line([shoulder, P(2.66, 1.05, 2.25), grip], '#C8AA8F', 0.6, z + 0.5)
            line([hip, P(2.62, 1.05, 1.2), P(2.2, 1.05, 0.7)], '#4C687F', 0.75, z + 0.5)
            line([P(2.13, 1.05, 0.66), P(2.38, 1.05, 0.66)], '#405C72', 0.6, z + 0.55)
        register('cyclist', 2.35, 1.05, cyclist, [(1.3, 0.72), (3.5, 1.38)])
    elif kind == 'road':
        paving(0.22, 0.22, 3.78, 2.78, '#DDE4E8')
        paving(0.25, 1.0, 3.75, 1.85, '#91A4B7', 8)
        paving(1.5, 0.25, 2.3, 2.75, '#91A4B7', 9)
        for i in [0.35, 0.9, 2.5, 3.1]:
            line([P(i, 1.43), P(i + 0.28, 1.43)], '#F9FAF8', 0.45, 10)
        for j in [0.35, 0.8, 2.1, 2.5]:
            line([P(1.9, j), P(1.9, j + 0.18)], '#F9FAF8', 0.45, 10)
        for j in [1.95, 2.1, 2.25, 2.4]:
            line([P(1.55, j), P(2.25, j)], 'white', 0.6, 11)
        sedan(3.1, 1.43, 0.38)

        def signal(z):
            q = P(0.85, 2.2)
            line([q, q + [0, 1.55 * s]], '#798FA2', 0.5, z)
            r.rect(q[0] - 0.25 * s, q[1] + 1.1 * s, 0.5 * s, 1.0 * s, '#7893A8', zorder=z + 1)
            for h, fc in [(1.85, '#D8AB9C'), (1.55, '#E5D6A5'), (1.25, '#9CBDA9')]:
                r.circle(q[0], q[1] + h * s, 0.09 * s, fc, zorder=z + 1.2)
        register('traffic_signal', 0.85, 2.2, signal)
    elif kind == 'terrain':
        ii = np.linspace(0.2, 3.8, 23)
        jj = np.linspace(0.2, 2.8, 17)
        I, J = np.meshgrid(ii, jj)
        heights = 3.7 * np.exp(-((I - 1.15) / 0.68) ** 2 - ((J - 1.75) / 0.78) ** 2)
        heights += 2.8 * np.exp(-((I - 2.8) / 0.65) ** 2 - ((J - 1.25) / 0.65) ** 2)
        cells = []
        for a in range(len(ii) - 1):
            for bb in range(len(jj) - 1):
                cells.append((ii[a] + jj[bb], a, bb))
        for n, (_, a, bb) in enumerate(sorted(cells, reverse=True)):
            inds = [(bb, a), (bb, a + 1), (bb + 1, a + 1), (bb + 1, a)]
            dh_i = (heights[bb, a + 1] - heights[bb, a]) / (ii[1] - ii[0])
            dh_j = (heights[bb + 1, a] - heights[bb, a]) / (jj[1] - jj[0])
            amount = np.clip(0.33 + 0.09 * dh_i + 0.12 * dh_j, 0.08, 0.68)
            points = [P(ii[k], jj[l], heights[l, k]) for l, k in inds]
            surface = poly(points, r.tint('#7E9D89', amount), 'none', 0, 9 + n * 0.0005)
            surface.set_antialiased(False)
        generator = contourpy.contour_generator(x=ii, y=jj, z=heights)
        for h in [0.55, 1.25, 2.1, 3.0]:
            for segment in generator.lines(h):
                line([P(i, j, h + 0.025) for i, j in segment], '#D3DFCE', 0.25, 11)
        GROUND_ANCHORS.append({'scene': kind, 'object': 'continuous_relief', 'on_ground_slab': True, 'footprint_ij': [[0.2, 0.2], [3.8, 2.8]]})
    elif kind == 'location':
        for i in [0.8, 1.6, 2.4, 3.2]:
            line([P(i, 0.3), P(i, 2.7)], '#D7E2EA', 0.3, 8)
        for j in [0.6, 1.2, 1.8, 2.4]:
            line([P(0.3, j), P(3.7, j)], '#D7E2EA', 0.3, 8)

        def globe(z):
            global _WORLD
            foot = P(2, 1.5)
            g = P(2, 1.5, 3.0)
            radius = 2.2 * s
            oval(*foot, 2.8 * s, 0.7 * s, '#C8D6E0', z=z)
            line([foot, g - [0, radius - 0.1 * s]], '#91A8BE', 0.9 * s, z + 0.1)
            disk = Circle(g, radius, facecolor='#EEF5F8', edgecolor='#9CB5CF', lw=0.4, zorder=z + 0.2)
            ax.add_patch(disk)
            if _WORLD is None:
                shp = next(resource_glob(Path(r.b.pyogrio.__file__).resolve().parent, 'naturalearth_lowres.shp'))
                _WORLD = r.b.gpd.read_file(resource(shp))
            for geometry in _WORLD.geometry:
                parts = list(geometry.geoms) if geometry.geom_type == 'MultiPolygon' else [geometry]
                for p in parts:
                    vertices = np.asarray(p.exterior.coords)
                    lon = np.deg2rad(vertices[:, 0] - 20)
                    lat = np.deg2rad(vertices[:, 1])
                    lat0 = np.deg2rad(20)
                    visible = np.sin(lat0) * np.sin(lat) + np.cos(lat0) * np.cos(lat) * np.cos(lon) >= 0
                    if visible.sum() < 3:
                        continue
                    xx = np.cos(lat) * np.sin(lon)
                    yy = np.cos(lat0) * np.sin(lat) - np.sin(lat0) * np.cos(lat) * np.cos(lon)
                    land = poly(g + radius * np.column_stack([xx[visible], yy[visible]]), '#A9C4B4', 'none', 0, z + 0.4)
                    land.set_clip_path(disk)
            for w, h in [(radius * 2, radius * 0.7), (radius * 0.85, radius * 2)]:
                p = oval(*g, w, h, 'none', '#B8CCDD', 0.25, z=z + 0.6)
                p.set_clip_path(disk)
            q = g + [0.67 * s, 0.1 * s]
            r.circle(*q, 0.32 * s, C['blue'], zorder=z + 0.8)
            poly([q + [-0.25 * s, -0.1 * s], q + [0, -0.65 * s], q + [0.25 * s, -0.1 * s]], C['blue'], 'none', 0, z + 0.8)
            r.circle(q[0], q[1] + 0.04 * s, 0.11 * s, 'white', zorder=z + 1)
        register('globe_on_stand', 2, 1.5, globe, [(1.4, 1.1), (2.6, 1.9)])
    for k, (_, draw) in enumerate(sorted(jobs, key=lambda v: v[0], reverse=True)):
        draw(12 + k * 5)
