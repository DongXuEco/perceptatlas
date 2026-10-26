"""Plot Supplementary_Fig_S4 using the packaged numerical tables or supplied local inputs."""
from __future__ import annotations
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_support"))
from figure_cli import prepare_figure as _prepare_figure, save_figure as _save_figure
from figure_cli import read_table as _read_table, model_tables as _model_tables
from figure_cli import configure_land_functions as _configure_land_functions

_figure_options = _prepare_figure('Supplementary_Fig_S4', 'external_inputs_required') if __name__ == "__main__" else None

from package_paths import PACKAGE_ROOT, resource, resource_glob, open_resource, output_dir
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from sklearn.cluster import MiniBatchKMeans
INPUT_PATH = str(PACKAGE_ROOT / 'data/Supplementary_Fig_S4/Supplementary_Fig_S4_endmember_points.csv')
OUTPUT_PATH = str(output_dir('Supplementary_Fig_S4') / 'Supplementary_Fig_S4.png')
LON_FIELD = 'lon'
LAT_FIELD = 'lat'
ID_FIELD = 'ID'
MANUAL_INSET_IDS = {1: ['197', '6831', '7866'], 2: ['10189', '10560', '17156'], 3: ['23069', '28427', '23672']}
CLASS_FIELD = 'em_class'
EXPORT_PDF = True
GLOBAL_EXTENT = [-180, 180, -60, 85]
N_INSETS_PER_CLASS = 3
INSET_HALF_SIZE = {1: 0.1, 2: 0.1, 3: 0.2}
MAIN_ZOOM = 6
INSET_ZOOM = 11
FIGSIZE = (18, 12)
DPI = 600
SHOW_MAIN_REPRESENTATIVE_POINTS = True
SHOW_MAIN_REPRESENTATIVE_LABELS = True
SHOW_INSET_COORD_LABELS = True
CALLOUT_POINT_SIZE = 125
CALLOUT_LINEWIDTH = 1.5
MAIN_LABEL_DX = 3.5
MAIN_LABEL_DY = 2
INSET_BORDER_COLOR = '#B0B0B0'
INSET_BORDER_LW = 0.55
MAIN_BASEMAP_WHITE_OVERLAY_ALPHA = 0.08
CLASS_STYLE = {1: {'label': 'Vegetation', 'color': '#2FBF71', 'marker': 'o'}, 2: {'label': 'Bright substrate', 'color': '#B83779', 'marker': '^'}, 3: {'label': 'Water', 'color': '#1F78B4', 'marker': 's'}}
MAIN_POINT_SIZE = 6.5
MAIN_POINT_ALPHA = 0.8
MAIN_POINT_EDGE_LW = 0.12
INSET_POINT_SIZE = 120
DATA_CRS = ccrs.PlateCarree()

def read_points(input_path, class_field='em_class', lon_field='lon', lat_field='lat'):
    input_path = Path(input_path)
    suffix = input_path.suffix.lower()
    if suffix == '.csv':
        df = pd.read_csv(resource(input_path))
        if lon_field not in df.columns or lat_field not in df.columns:
            raise ValueError(f'CSV 中未找到经纬度字段：{lon_field}, {lat_field}')
        if class_field not in df.columns:
            raise ValueError(f'CSV 中未找到类别字段：{class_field}')
        gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[lon_field], df[lat_field]), crs='EPSG:4326')
    else:
        gdf = gpd.read_file(resource(input_path))
        if class_field not in gdf.columns:
            raise ValueError(f'矢量文件中未找到类别字段：{class_field}')
        if gdf.crs is None:
            print('警告：输入数据没有 CRS，默认设置为 EPSG:4326')
            gdf = gdf.set_crs(epsg=4326)
        else:
            gdf = gdf.to_crs(epsg=4326)
    gdf = gdf[gdf.geometry.notnull()].copy()
    gdf = gdf[gdf.geometry.geom_type.isin(['Point', 'MultiPoint'])].copy()
    gdf = gdf.explode(index_parts=False).reset_index(drop=True)
    gdf = gdf[gdf[class_field].isin([1, 2, 3])].copy()
    gdf.reset_index(drop=True, inplace=True)
    if len(gdf) == 0:
        raise ValueError('读取后没有有效点数据，请检查输入文件和字段。')
    return gdf

def add_satellite_basemap(ax, zoom=2):
    extent = ax.get_extent(ccrs.PlateCarree())
    name = 'satellite_' + '_'.join((f'{v:.4f}' for v in extent)) + '.png'
    path = resource(PACKAGE_ROOT / 'external_inputs' / name)
    if not resource(path).is_file():
        raise FileNotFoundError(f'Local satellite image required: {name}')
    ax.imshow(plt.imread(path), extent=extent, transform=ccrs.PlateCarree(), origin='upper')

def add_white_overlay(ax, extent, alpha=0.08):
    if alpha <= 0:
        return
    arr = np.ones((2, 2, 4), dtype=float)
    arr[:, :, 0] = 1.0
    arr[:, :, 1] = 1.0
    arr[:, :, 2] = 1.0
    arr[:, :, 3] = alpha
    ax.imshow(arr, extent=extent, transform=DATA_CRS, origin='upper', zorder=1)

def style_geoaxes(ax, border_color='#B0B0B0', border_lw=0.55):
    ax.set_facecolor('white')
    try:
        ax.spines['geo'].set_edgecolor(border_color)
        ax.spines['geo'].set_linewidth(border_lw)
    except Exception:
        pass

def select_representative_points(gdf, class_value, n=3, class_field='em_class', random_state=42):
    sub = gdf[gdf[class_field] == class_value].copy()
    if len(sub) == 0:
        raise ValueError(f'类别 {class_value} 没有样本点。')
    if len(sub) <= n:
        selected = sub.copy()
        selected['lon_tmp'] = selected.geometry.x
        selected = selected.sort_values('lon_tmp').drop(columns='lon_tmp')
        return selected.reset_index(drop=True)
    sub_3857 = sub.to_crs(epsg=3857)
    coords = np.column_stack([sub_3857.geometry.x.values, sub_3857.geometry.y.values])
    km = MiniBatchKMeans(n_clusters=n, random_state=random_state, batch_size=min(1024, len(sub)), n_init=10)
    labels = km.fit_predict(coords)
    centers = km.cluster_centers_
    selected_indices = []
    for i in range(n):
        cluster_ids = np.where(labels == i)[0]
        if len(cluster_ids) == 0:
            continue
        cluster_coords = coords[cluster_ids]
        d2 = np.sum((cluster_coords - centers[i]) ** 2, axis=1)
        best_local = cluster_ids[np.argmin(d2)]
        selected_indices.append(sub.index[best_local])
    selected = sub.loc[selected_indices].copy()
    if len(selected) < n:
        remain = sub.drop(index=selected.index)
        extra_n = min(n - len(selected), len(remain))
        extra = remain.sample(extra_n, random_state=random_state)
        selected = pd.concat([selected, extra], axis=0)
    selected['lon_tmp'] = selected.geometry.x
    selected = selected.sort_values('lon_tmp').drop(columns='lon_tmp')
    selected = selected.reset_index(drop=True)
    return selected

def prepare_inset_points(gdf):
    selected_dict = {}
    prefix_map = {1: 'V', 2: 'B', 3: 'W'}
    for cls in [1, 2, 3]:
        if MANUAL_INSET_IDS is not None and cls in MANUAL_INSET_IDS:
            ids = MANUAL_INSET_IDS[cls]
            if ID_FIELD not in gdf.columns:
                raise ValueError(f'数据中没有找到 ID 字段：{ID_FIELD}')
            sel = gdf[(gdf[CLASS_FIELD] == cls) & gdf[ID_FIELD].astype(str).isin([str(x) for x in ids])].copy()
            if len(sel) == 0:
                raise ValueError(f'类别 {cls} 手动指定的点没有匹配到，请检查 {ID_FIELD} 和 ID 值。')
            id_order = {str(v): i for i, v in enumerate(ids)}
            sel['_order'] = sel[ID_FIELD].astype(str).map(id_order)
            sel = sel.sort_values('_order').drop(columns='_order').reset_index(drop=True)
        else:
            sel = select_representative_points(gdf, class_value=cls, n=N_INSETS_PER_CLASS, class_field=CLASS_FIELD, random_state=100 + cls).copy()
        sel['panel_label'] = [f'{prefix_map[cls]}{i + 1}' for i in range(len(sel))]
        selected_dict[cls] = sel
    return selected_dict

def plot_main_points(ax, gdf):
    for cls, style in CLASS_STYLE.items():
        sub = gdf[gdf[CLASS_FIELD] == cls]
        ax.scatter(sub.geometry.x, sub.geometry.y, s=MAIN_POINT_SIZE, c=style['color'], marker=style['marker'], alpha=MAIN_POINT_ALPHA, edgecolors='white', linewidths=MAIN_POINT_EDGE_LW, transform=DATA_CRS, zorder=4, rasterized=True)

def add_main_legend(ax):
    handles = []
    for cls, style in CLASS_STYLE.items():
        handles.append(Line2D([0], [0], marker=style['marker'], color='none', markerfacecolor=style['color'], markeredgecolor='white', markeredgewidth=0.8, markersize=8.8, label=style['label']))
    legend = ax.legend(handles=handles, loc='lower left', frameon=True, framealpha=1, facecolor='white', edgecolor='#D0D0D0', fontsize=13, title='Endmember class', title_fontsize=13, borderpad=0.65, labelspacing=0.45, handletextpad=0.65)
    legend.get_frame().set_linewidth(0.6)
    return legend

def highlight_representative_points(ax, selected_dict):
    if not SHOW_MAIN_REPRESENTATIVE_POINTS:
        return
    for cls, sel in selected_dict.items():
        style = CLASS_STYLE[cls]
        for _, row in sel.iterrows():
            lon = row.geometry.x
            lat = row.geometry.y
            label = row['panel_label']
            ax.scatter([lon], [lat], s=CALLOUT_POINT_SIZE, facecolors='none', edgecolors='white', linewidths=CALLOUT_LINEWIDTH, transform=DATA_CRS, zorder=7)
            ax.scatter([lon], [lat], s=CALLOUT_POINT_SIZE * 0.34, c=style['color'], marker=style['marker'], edgecolors='white', linewidths=0.65, transform=DATA_CRS, zorder=8)
            if SHOW_MAIN_REPRESENTATIVE_LABELS:
                txt = ax.text(lon + MAIN_LABEL_DX, lat + MAIN_LABEL_DY, label, transform=DATA_CRS, fontsize=11, fontweight='normal', color='#111111', zorder=9, bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='none', alpha=1))
                txt.set_path_effects([pe.withStroke(linewidth=1.8, foreground='white')])

def plot_inset(ax, row, cls_value):
    style = CLASS_STYLE[cls_value]
    lon = row.geometry.x
    lat = row.geometry.y
    panel_label = row['panel_label']
    half_size = INSET_HALF_SIZE[cls_value]
    ax.set_extent([lon - half_size, lon + half_size, lat - half_size, lat + half_size], crs=DATA_CRS)
    add_satellite_basemap(ax, zoom=INSET_ZOOM)
    ax.scatter([lon], [lat], s=INSET_POINT_SIZE, c=style['color'], marker=style['marker'], edgecolors='white', linewidths=1.3, transform=DATA_CRS, zorder=8)
    ax.text(0.055, 0.945, panel_label, transform=ax.transAxes, ha='left', va='top', fontsize=10, fontweight='normal', color='#111111', bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='none', alpha=1), zorder=9)
    if SHOW_INSET_COORD_LABELS:
        ax.text(0.945, 0.055, f'{lon:.3f}°, {lat:.3f}°', transform=ax.transAxes, ha='right', va='bottom', fontsize=10, color='#333333', bbox=dict(boxstyle='round,pad=0.12', facecolor='white', edgecolor='none', alpha=1), zorder=9)
    style_geoaxes(ax, border_color=INSET_BORDER_COLOR, border_lw=INSET_BORDER_LW)
    ax.set_xticks([])
    ax.set_yticks([])

def make_figure(input_path, output_path):
    gdf = read_points(input_path, class_field=CLASS_FIELD, lon_field=LON_FIELD, lat_field=LAT_FIELD)
    print('总点数:', len(gdf))
    print(gdf[CLASS_FIELD].value_counts().sort_index())
    selected_dict = prepare_inset_points(gdf)
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['axes.linewidth'] = 0.8
    fig = plt.figure(figsize=FIGSIZE, dpi=DPI, facecolor='white')
    main_left = 0.045
    main_width = 0.91
    main_right = main_left + main_width
    extent_width = GLOBAL_EXTENT[1] - GLOBAL_EXTENT[0]
    extent_height = GLOBAL_EXTENT[3] - GLOBAL_EXTENT[2]
    extent_ratio = extent_width / extent_height
    fig_w, fig_h = FIGSIZE
    main_height = main_width * fig_w / fig_h / extent_ratio
    main_pos = [main_left, 0.305, main_width, main_height]
    inset_y = 0.105
    inner_gap = 0.007
    group_gap = 0.01
    inset_w = (main_width - 6 * inner_gap - 2 * group_gap) / 9
    inset_h = inset_w * fig_w / fig_h
    group_width = 3 * inset_w + 2 * inner_gap
    vegetation_x0 = main_left
    bare_x0 = vegetation_x0 + group_width + group_gap
    water_x0 = bare_x0 + group_width + group_gap
    print('Main left/right:', main_left, main_right)
    print('Inset block left/right:', vegetation_x0, water_x0 + group_width)
    vegetation_positions = [[vegetation_x0 + i * (inset_w + inner_gap), inset_y, inset_w, inset_h] for i in range(3)]
    bare_positions = [[bare_x0 + i * (inset_w + inner_gap), inset_y, inset_w, inset_h] for i in range(3)]
    water_positions = [[water_x0 + i * (inset_w + inner_gap), inset_y, inset_w, inset_h] for i in range(3)]
    main_ax = fig.add_axes(main_pos, projection=DATA_CRS)
    main_ax.set_extent(GLOBAL_EXTENT, crs=DATA_CRS)
    add_satellite_basemap(main_ax, zoom=MAIN_ZOOM)
    add_white_overlay(main_ax, GLOBAL_EXTENT, alpha=MAIN_BASEMAP_WHITE_OVERLAY_ALPHA)
    main_ax.gridlines(crs=DATA_CRS, draw_labels=False, linewidth=0.28, color='white', alpha=0.3, linestyle='--')
    plot_main_points(main_ax, gdf)
    highlight_representative_points(main_ax, selected_dict)
    add_main_legend(main_ax)
    style_geoaxes(main_ax, border_color='#6F6F6F', border_lw=0.65)
    main_ax.text(0.015, 0.965, 'A', transform=main_ax.transAxes, ha='left', va='top', fontsize=17, fontweight='normal', color='#111111')
    title_y = inset_y + inset_h + 0.025
    veg_center = vegetation_x0 + group_width / 2
    bare_center = bare_x0 + group_width / 2
    water_center = water_x0 + group_width / 2
    title_x_pad = 0.003
    fig.text(vegetation_x0 + title_x_pad, title_y, 'B. Vegetation', ha='left', va='center', fontsize=15, fontweight='normal', color='#111111')
    fig.text(bare_x0 + title_x_pad, title_y, 'C. Bright substrate', ha='left', va='center', fontsize=15, fontweight='normal', color='#111111')
    fig.text(water_x0 + title_x_pad, title_y, 'D. Water', ha='left', va='center', fontsize=15, fontweight='normal', color='#111111')
    for i, (_, row) in enumerate(selected_dict[1].iterrows()):
        ax = fig.add_axes(vegetation_positions[i], projection=DATA_CRS)
        plot_inset(ax, row, cls_value=1)
    for i, (_, row) in enumerate(selected_dict[2].iterrows()):
        ax = fig.add_axes(bare_positions[i], projection=DATA_CRS)
        plot_inset(ax, row, cls_value=2)
    for i, (_, row) in enumerate(selected_dict[3].iterrows()):
        ax = fig.add_axes(water_positions[i], projection=DATA_CRS)
        plot_inset(ax, row, cls_value=3)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    if EXPORT_PDF:
        pdf_path = output_path.with_suffix('.pdf')
        plt.savefig(pdf_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'图片已保存到: {output_path}')
    if EXPORT_PDF:
        print(f'PDF 已保存到: {pdf_path}')

def _script_entry():
    make_figure(INPUT_PATH, OUTPUT_PATH)

def _build_manuscript_figure():
    _script_entry()
    return None, None


if __name__ == "__main__":
    _configure_land_functions(sys.modules[__name__])
    _figure, _bbox = _build_manuscript_figure()
    _save_figure(_figure, _bbox, 'Supplementary_Fig_S4', _figure_options)
