import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import traceback
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from datetime import datetime

from src.config import STATION_COORDS, CENTER_LAT, CENTER_LON, haversine, get_data_path, get_models_path
from src.database import init_db, load_listings, load_timeseries as load_ts_db, has_data as db_has_data

st.set_page_config(
    page_title="Аренда Минска — Дашборд",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

PLOTLY_CONFIG = {
    'displayModeBar': True,
    'modeBarButtonsToRemove': ['sendDataToCloud', 'lasso2d'],
    'displaylogo': False,
    'toImageButtonOptions': {
        'format': 'png', 'filename': 'chart', 'height': 600, 'width': 900,
    },
}

CHART_LAYOUT = dict(
    font=dict(family='-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Inter, sans-serif'),
    hovermode='x unified',
    hoverlabel=dict(
        bordercolor='rgba(0,0,0,0)',
        font=dict(size=12),
    ),
    margin=dict(l=50, r=20, t=40, b=50),
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
)

GRID_STYLE = dict(
    gridcolor='rgba(128,128,152,0.12)',
    zerolinecolor='rgba(128,128,152,0.12)',
)

st.title("🏠 Рынок аренды квартир в Минске")
st.caption("Данные собраны с Kufar.by и Realt.by")


@st.cache_data(ttl=3600)
def load_data():
    try:
        init_db()
        if db_has_data():
            df = load_listings()
            if not df.empty:
                return df
    except Exception:
        pass
    csv_path = get_data_path('processed', 'rentals_clean.csv')
    if not os.path.exists(csv_path):
        st.error(f"Файл данных не найден: {csv_path}")
        st.stop()
    return pd.read_csv(csv_path)


@st.cache_data(ttl=3600)
def load_timeseries():
    try:
        init_db()
        df = load_ts_db()
        if not df.empty:
            return df
    except Exception:
        pass
    csv_path = get_data_path('processed', 'timeseries_data.csv')
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path, parse_dates=['snapshot_date'])
    return df


@st.cache_resource
def load_model():
    model_path = get_models_path('rental_price_model.pkl')
    info_path = get_models_path('model_info.json')
    if not os.path.exists(model_path):
        alt_paths = [
            'models/rental_price_model.pkl',
            os.path.join(os.getcwd(), 'models', 'rental_price_model.pkl'),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models', 'rental_price_model.pkl'),
        ]
        for p in alt_paths:
            p = os.path.normpath(p)
            if os.path.exists(p):
                model_path = p
                info_path = os.path.join(os.path.dirname(p), 'model_info.json')
                break
        else:
            return None, None
    try:
        pipeline = joblib.load(model_path)
        with open(info_path, 'r') as f:
            model_info = json.load(f)
        return pipeline, model_info
    except Exception as e:
        st.error(f"Ошибка загрузки модели: {e}")
        return None, None


try:
    df = load_data()
    if 'price_usd' in df.columns:
        df = df[df['price_usd'] < 50000].copy()
except Exception as e:
    st.error(f"Ошибка загрузки данных: {e}")
    st.stop()

df_ts = load_timeseries()
pipeline, model_info = load_model()

# ── Вспомогательные функции ──

ROOM_LABELS = {1: '1 комната', 2: '2 комнаты', 3: '3 комнаты', 4: '4 комнаты', 5: '5 комнат'}
SOURCE_LABELS = {'kufar': 'Kufar.by', 'realt': 'Realt.by'}


def make_pdf(df_data, df_filtered_data, filters_text):
    from src.pdf_report import generate_pdf
    return generate_pdf(df_data, df_filtered_data, filters_text)


# ── САЙДБАР — ТЕМА / ФИЛЬТРЫ ──
with st.sidebar:
    dark_theme = st.toggle("🌙 Тёмная тема", value=False)
    pio.templates.default = 'plotly_dark' if dark_theme else 'plotly_white'
    if dark_theme:
        st.markdown("""
        <style>
        .stApp { background: #0e1117; }
        section[data-testid="stSidebar"] { background: #262730; }
        .stApp h1, .stApp h2, .stApp h3, .stApp h4 { color: #fafafa !important; }
        footer { color: #808495 !important; }
        </style>
        """, unsafe_allow_html=True)
    st.markdown("##### 🔍 Фильтры")

sources = st.sidebar.multiselect(
    "Источник",
    options=['kufar', 'realt'],
    format_func=lambda x: SOURCE_LABELS.get(x, x),
    default=['kufar', 'realt'],
)

max_rooms = int(df['rooms'].max()) if 'rooms' in df.columns else 5
rooms_filter = st.sidebar.multiselect(
    "Комнат",
    options=list(range(1, max_rooms + 1)),
    format_func=lambda x: ROOM_LABELS.get(x, f'{x} комн.'),
    default=[1, 2, 3],
)

max_price_in_data = int(df['price_usd'].max()) if 'price_usd' in df.columns else 20000
price_min, price_max = st.sidebar.slider(
    "Цена (USD/мес)",
    min_value=0, max_value=max_price_in_data,
    value=(100, min(2000, max_price_in_data)),
    step=50,
)

df_filtered = df[
    (df['source'].isin(sources)) &
    (df['rooms'].isin(rooms_filter)) &
    (df['price_usd'].between(price_min, price_max))
].copy()

if 'rooms' in df_filtered.columns:
    df_filtered['rooms'] = df_filtered['rooms'].astype(int)

# Latest date subset for KPI
if 'snapshot_date' in df_filtered.columns and not df_filtered.empty:
    latest_date = df_filtered['snapshot_date'].max()
    df_latest = df_filtered[df_filtered['snapshot_date'] == latest_date].copy()
else:
    df_latest = df_filtered

# ── KPI ──
col1, col2, col3, col4 = st.columns(4)

if df_filtered.empty:
    st.warning("⚠️ Нет объявлений, соответствующих выбранным фильтрам.")
    st.stop()

with col1:
    st.metric("📋 Объявлений", f"{len(df_latest):,}")

with col2:
    st.metric("💰 Средняя цена", f"${df_latest['price_usd'].mean():.0f}")

with col3:
    st.metric("📊 Медианная цена", f"${df_latest['price_usd'].median():.0f}")

with col4:
    st.metric("📐 Средняя площадь",
              f"{df_latest['area_total'].mean():.0f} м²")

# ── ВКЛАДКИ ──
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 Распределение цен",
    "🏙️ По комнатам",
    "🏢 Kufar и Realt",
    "📈 Динамика цен",
    "🌡️ Дополнительно",
    "🗺️ Тепловая карта",
    "🤖 Калькулятор цены",
])

# ── TAB 1: РАСПРЕДЕЛЕНИЕ ЦЕН ──
with tab1:
    st.subheader("Распределение цен аренды")

    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(
            df_filtered, x='price_usd', nbins=50,
            title="Распределение цен (USD/мес)",
            color_discrete_sequence=['#e94560'],
            labels={'price_usd': 'Цена (USD/мес)'},
        )
        fig.update_yaxes(title_text='Количество объявлений', **GRID_STYLE)
        fig.update_layout(**CHART_LAYOUT, showlegend=False, height=400)
        st.plotly_chart(fig, width='stretch', config=PLOTLY_CONFIG)

    with col2:
        fig = px.box(
            df_filtered, x='rooms', y='price_usd',
            title="Цена по количеству комнат",
            color_discrete_sequence=['#e94560'],
            labels={'rooms': 'Комнат', 'price_usd': 'Цена (USD/мес)'},
            points='outliers',
        )
        fig.update_layout(**CHART_LAYOUT, showlegend=False, height=400)
        st.plotly_chart(fig, width='stretch', config=PLOTLY_CONFIG)

    st.subheader("Статистика по площади")
    fig = px.scatter(
        df_filtered.sample(min(1000, len(df_filtered))),
        x='area_total', y='price_usd',
        color='rooms',
        title="Цена и площадь",
        color_discrete_sequence=px.colors.qualitative.Bold,
        labels={'area_total': 'Площадь (м²)', 'price_usd': 'Цена (USD/мес)', 'rooms': 'Комнат'},
    )
    fig.update_layout(**CHART_LAYOUT, height=400)
    st.plotly_chart(fig, width='stretch', config=PLOTLY_CONFIG)

# ── TAB 2: ПО КОМНАТАМ ──
with tab2:
    st.subheader("Статистика по количеству комнат")

    col1, col2 = st.columns(2)

    with col1:
        room_stats = df_filtered.groupby('rooms').agg(
            count=('price_usd', 'count'),
            mean_price=('price_usd', 'mean'),
            median_price=('price_usd', 'median'),
        ).reset_index()
        room_stats['rooms_label'] = room_stats['rooms'].map(ROOM_LABELS)

        fig = px.bar(
            room_stats,
            x='rooms_label', y='mean_price',
            title="Средняя цена по комнатам",
            color='mean_price',
            color_continuous_scale='Viridis',
            text_auto='.0f',
            labels={'rooms_label': '', 'mean_price': 'Средняя цена (USD)'},
        )
        fig.update_layout(**CHART_LAYOUT, showlegend=False, height=400)
        fig.update_xaxes(**GRID_STYLE)
        fig.update_yaxes(**GRID_STYLE)
        st.plotly_chart(fig, width='stretch', config=PLOTLY_CONFIG)

    with col2:
        fig = px.histogram(
            df_filtered, x='price_usd', color='rooms',
            title="Распределение цен по комнатам",
            barmode='overlay', nbins=40,
            color_discrete_sequence=px.colors.qualitative.Bold,
            labels={'price_usd': 'Цена (USD/мес)', 'rooms': 'Комнат'},
        )
        fig.update_yaxes(title_text='Количество', **GRID_STYLE)
        fig.update_layout(**CHART_LAYOUT, height=400)
        st.plotly_chart(fig, width='stretch', config=PLOTLY_CONFIG)

    with st.expander("📋 Таблица статистики по комнатам"):
        stats = df_filtered.groupby('rooms').agg(
            Количество=('price_usd', 'count'),
            Средняя_цена=('price_usd', 'mean'),
            Медианная_цена=('price_usd', 'median'),
            Мин_цена=('price_usd', 'min'),
            Макс_цена=('price_usd', 'max'),
            Средняя_площадь=('area_total', 'mean'),
        ).round(0).rename_axis('Комнат')
        st.dataframe(stats, width='stretch')

# ── TAB 3: KUFAR VS REALT ──
with tab3:
    st.subheader("Сравнение Kufar.by и Realt.by")

    col1, col2 = st.columns(2)

    with col1:
        source_counts = df_filtered['source'].value_counts()
        fig = px.pie(
            values=source_counts.values,
            names=[SOURCE_LABELS.get(s, s) for s in source_counts.index],
            title="Доля объявлений по источникам",
            color_discrete_sequence=['#e94560', '#3498db'],
        )
        st.plotly_chart(fig, width='stretch', config=PLOTLY_CONFIG)

    with col2:
        fig = px.box(
            df_filtered, x='source', y='price_usd',
            title="Цены: Kufar и Realt",
            color='source',
            color_discrete_sequence=['#e94560', '#3498db'],
            labels={'source': 'Источник', 'price_usd': 'Цена (USD/мес)'},
            points='outliers',
        )
        fig.update_layout(**CHART_LAYOUT, showlegend=False, height=400)
        st.plotly_chart(fig, width='stretch', config=PLOTLY_CONFIG)

    st.subheader("Детальное сравнение")

    col1, col2 = st.columns(2)

    with col1:
        source_stats = df_filtered.groupby('source').agg(
            count=('price_usd', 'count'),
            mean_price=('price_usd', 'mean'),
            mean_area=('area_total', 'mean'),
            agencies=('company_ad', 'sum'),
        ).reset_index()
        source_stats['source'] = source_stats['source'].map(SOURCE_LABELS)
        source_stats['agency_pct'] = (source_stats['agencies'] / source_stats['count'] * 100).round(1)

        fig = px.bar(
            source_stats,
            x='source', y='mean_price',
            title="Средняя цена по источнику",
            color='source',
            color_discrete_sequence=['#e94560', '#3498db'],
            text_auto='.0f',
            labels={'source': '', 'mean_price': 'Средняя цена (USD)'},
        )
        fig.update_layout(**CHART_LAYOUT, showlegend=False, height=400)
        fig.update_xaxes(**GRID_STYLE)
        fig.update_yaxes(**GRID_STYLE)
        st.plotly_chart(fig, width='stretch', config=PLOTLY_CONFIG)

    with col2:
        fig = px.bar(
            source_stats,
            x='source', y='agency_pct',
            title="Доля агентств по источнику (%)",
            color='source',
            color_discrete_sequence=['#e94560', '#3498db'],
            text_auto='.1f',
            labels={'source': '', 'agency_pct': '% агентств'},
        )
        fig.update_layout(**CHART_LAYOUT, showlegend=False, height=400)
        fig.update_xaxes(**GRID_STYLE)
        fig.update_yaxes(**GRID_STYLE)
        st.plotly_chart(fig, width='stretch', config=PLOTLY_CONFIG)

    with st.expander("📋 Таблица сравнения источников"):
        src_table = df_filtered.groupby('source').agg(
            Объявлений=('price_usd', 'count'),
            Средняя_цена=('price_usd', 'mean'),
            Медианная_цена=('price_usd', 'median'),
            Средняя_площадь=('area_total', 'mean'),
            Агентств=('company_ad', 'sum'),
        ).round(0).rename_axis('Источник')
        src_table['% агентств'] = (
            src_table['Агентств'] / src_table['Объявлений'] * 100).round(1)
        src_table.index = src_table.index.map(lambda x: SOURCE_LABELS.get(x, x))
        st.dataframe(src_table, width='stretch')

# ── TAB 4: ДИНАМИКА ЦЕН ──
with tab4:
    st.subheader("📈 Динамика цен во времени")

    if df_ts is None or df_ts.empty:
        st.warning("Нет данных временных рядов. Данные появятся после первого сбора.")
    else:
        ts_sources = st.multiselect(
            "Источник (временные ряды)",
            options=df_ts['source'].unique(),
            format_func=lambda x: SOURCE_LABELS.get(x, x),
            default=df_ts['source'].unique(),
            key='ts_source',
        )
        ts_rooms = st.multiselect(
            "Комнат (временные ряды)",
            options=sorted(df_ts['rooms'].dropna().unique().astype(int)),
            format_func=lambda x: ROOM_LABELS.get(x, f'{x} комн.'),
            default=[1, 2, 3],
            key='ts_rooms',
        )

        df_ts_filtered = df_ts[
            (df_ts['source'].isin(ts_sources)) &
            (df_ts['rooms'].isin(ts_rooms))
        ].copy()

        if df_ts_filtered.empty:
            st.info("Нет данных для выбранных фильтров.")
        else:
            df_ts_filtered['rooms'] = df_ts_filtered['rooms'].astype(int)

            col1, col2 = st.columns(2)

            with col1:
                price_col = 'median_price' if 'median_price' in df_ts_filtered.columns else 'price_usd'
                avg_by_rooms = df_ts_filtered.groupby(
                    ['snapshot_date', 'rooms'], as_index=False
                )[price_col].mean()

                fig = px.line(
                    avg_by_rooms,
                    x='snapshot_date', y=price_col, color='rooms',
                    title="Медианная цена по комнатам по дням",
                    markers=True,
                    color_discrete_sequence=px.colors.qualitative.Bold,
                    labels={'snapshot_date': 'Дата', price_col: 'Медианная цена (USD)', 'rooms': 'Комнат'},
                )
                fig.update_layout(**CHART_LAYOUT, height=400)
                fig.update_xaxes(**GRID_STYLE)
                fig.update_yaxes(**GRID_STYLE)
                st.plotly_chart(fig, width='stretch', config=PLOTLY_CONFIG)

            with col2:
                count_col = 'count' if 'count' in df_ts_filtered.columns else None
                if count_col:
                    count_by_date = df_ts_filtered.groupby(
                        'snapshot_date', as_index=False
                    )[count_col].sum()
                else:
                    count_by_date = df_ts_filtered.groupby(
                        'snapshot_date', as_index=False
                    ).size().rename(columns={'size': 'count'})

                fig = px.bar(
                    count_by_date,
                    x='snapshot_date', y='count',
                    title="Количество объявлений по дням",
                    color_discrete_sequence=['#3498db'],
                    text_auto=True,
                    labels={'snapshot_date': 'Дата', 'count': 'Количество объявлений'},
                )
                fig.update_layout(**CHART_LAYOUT, height=400)
                fig.update_xaxes(**GRID_STYLE)
                fig.update_yaxes(**GRID_STYLE)
                st.plotly_chart(fig, width='stretch', config=PLOTLY_CONFIG)

            col1, col2 = st.columns(2)

            with col1:
                df_ts_filtered['price_per_m2'] = (
                    df_ts_filtered['price_usd'] / df_ts_filtered['area_total'].replace(0, np.nan)
                )
                avg_per_m2 = df_ts_filtered.groupby(
                    ['snapshot_date', 'source'], as_index=False
                )['price_per_m2'].mean()

                fig = px.line(
                    avg_per_m2,
                    x='snapshot_date', y='price_per_m2', color='source',
                    title="Средняя цена за м² по дням",
                    markers=True,
                    color_discrete_sequence=['#e94560', '#3498db'],
                    labels={'snapshot_date': 'Дата', 'price_per_m2': 'Цена за м² (USD)', 'source': 'Источник'},
                )
                fig.update_layout(**CHART_LAYOUT, height=400)
                fig.update_xaxes(**GRID_STYLE)
                fig.update_yaxes(**GRID_STYLE)
                st.plotly_chart(fig, width='stretch', config=PLOTLY_CONFIG)

            with col2:
                avg_by_source = df_ts_filtered.groupby(
                    ['snapshot_date', 'source'], as_index=False
                )['price_usd'].mean()

                fig = px.line(
                    avg_by_source,
                    x='snapshot_date', y='price_usd', color='source',
                    title="Средняя цена: Kufar и Realt",
                    markers=True,
                    color_discrete_sequence=['#e94560', '#3498db'],
                    labels={'snapshot_date': 'Дата', 'price_usd': 'Средняя цена (USD)', 'source': 'Источник'},
                )
                fig.update_layout(**CHART_LAYOUT, height=400)
                fig.update_xaxes(**GRID_STYLE)
                fig.update_yaxes(**GRID_STYLE)
                st.plotly_chart(fig, width='stretch', config=PLOTLY_CONFIG)

# ── TAB 5: ДОПОЛНИТЕЛЬНО ──
with tab5:
    st.subheader("🌡️ Дополнительная аналитика")

    metro_col = 'metro_station' if 'metro_station' in df_filtered.columns else None
    has_metro = metro_col and df_filtered[metro_col].notna().any()

    if not has_metro:
        st.info("Нет данных по станциям метро. Данные появятся после нескольких дней сбора.")
    else:
        df_metro = df_filtered[df_filtered[metro_col].notna() & (df_filtered[metro_col] != '')].copy()
        metro_stats = df_metro.groupby(metro_col, as_index=False)['price_usd'].mean()
        metro_stats = metro_stats.sort_values('price_usd', ascending=False)

        fig = px.bar(
            metro_stats.head(15),
            x='price_usd', y=metro_col,
            title="Станции метро по средней цене аренды",
            orientation='h',
            color='price_usd',
            color_continuous_scale='Viridis',
            text_auto='.0f',
            labels={'price_usd': 'Средняя цена (USD)', metro_col: 'Станция метро'},
        )
        fig.update_layout(
            **CHART_LAYOUT,
            height=600,
            xaxis_title="Средняя цена (USD)",
            yaxis=dict(title=""),
        )
        fig.update_xaxes(**GRID_STYLE)
        st.plotly_chart(fig, width='stretch', config=PLOTLY_CONFIG)

# ── TAB 6: НА КАРТЕ ──
with tab6:
    st.subheader("🗺️ Тепловая карта цен")

    lat_col = 'location_lat' if 'location_lat' in df_filtered.columns else 'lat'
    lon_col = 'location_lon' if 'location_lon' in df_filtered.columns else 'lon'
    has_coords = df_filtered[lat_col].notna() & df_filtered[lon_col].notna()
    df_map = df_filtered[has_coords].copy()

    if df_map.empty:
        st.info("Нет объявлений с координатами для отображения на карте.")
    elif len(df_map) < 2:
        st.info("Слишком мало точек для отображения на карте.")
    else:
        fig = go.Figure()

        p01 = df_map['price_usd'].quantile(0.01)
        p99 = df_map['price_usd'].quantile(0.99)

        fig.add_trace(go.Densitymapbox(
            lat=df_map[lat_col],
            lon=df_map[lon_col],
            z=df_map['price_usd'],
            radius=12,
            colorscale='Viridis',
            zmin=p01,
            zmax=p99,
            opacity=0.85,
            colorbar=dict(title="Цена (USD)", thickness=15),
            hovertemplate='Цена: $%{z:.0f}<br>Лат: %{lat:.4f}<br>Лон: %{lon:.4f}<extra></extra>',
        ))

        metro_lats = [v[0] for v in STATION_COORDS.values()]
        metro_lons = [v[1] for v in STATION_COORDS.values()]
        metro_names = list(STATION_COORDS.keys())

        fig.add_trace(go.Scattermapbox(
            lat=metro_lats,
            lon=metro_lons,
            mode='markers+text',
            marker=dict(size=8, color='#e94560', symbol='circle'),
            text=metro_names,
            textposition='top center',
            textfont=dict(size=9, color='white'),
            name='Станции метро',
            hovertemplate='%{text}<extra></extra>',
        ))

        center_lat = df_map[lat_col].mean()
        center_lon = df_map[lon_col].mean()

        fig.update_layout(
            mapbox=dict(
                style='open-street-map',
                center=dict(lat=center_lat, lon=center_lon),
                zoom=11,
            ),
            height=650,
            margin=dict(l=0, r=0, t=30, b=0),
            legend=dict(yanchor='top', y=0.99, xanchor='left', x=0.01),
            font=dict(family='-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Inter, sans-serif'),
        )

        st.plotly_chart(fig, width='stretch', config=PLOTLY_CONFIG)

# ── TAB 7: КАЛЬКУЛЯТОР ЦЕНЫ ──
with tab7:
    st.subheader("🤖 Калькулятор справедливой цены")
    model_display = model_info.get('best_model', 'XGBoost') if model_info else 'XGBoost'
    st.caption(f"Модель машинного обучения ({model_display}) предсказывает цену на основе параметров квартиры.")

    if pipeline is None or model_info is None:
        st.error("Модель не найдена. Сначала обучите модель: `python src/train_final.py`")
        st.stop()

    required_features = model_info['features']
    use_log = model_info.get('use_log', True)

    col1, col2, col3 = st.columns(3)

    with col1:
        rooms_input = st.selectbox("Комнат", [1, 2, 3, 4, 5], index=1)
        area_input = st.number_input(
            "Площадь (м²)", min_value=15, max_value=300, value=50)
        floor_input = st.number_input(
            "Этаж", min_value=1, max_value=40, value=5)
        building_year_input = st.number_input(
            "Год постройки", min_value=1950, max_value=datetime.now().year, value=2000)

    with col2:
        floors_total_input = st.number_input(
            "Этажность дома", min_value=1, max_value=40, value=9)
        metro_dist_input = st.number_input(
            "Расстояние до метро (м)", min_value=0, max_value=5000, value=500)
        metro_station_input = st.selectbox(
            "Станция метро (для расчёта удалённости от центра)",
            options=[""] + sorted(STATION_COORDS.keys()),
            index=0,
        )
        has_furniture = st.checkbox("Мебель", value=True)
        has_appliances = st.checkbox("Бытовая техника", value=True)

    with col3:
        has_balcony = st.checkbox("Балкон/лоджия", value=True)
        has_elevator = st.checkbox("Лифт", value=True)
        company_ad = st.checkbox("Агентство")
        renovation_type = st.radio(
            "Ремонт",
            options=["Без ремонта", "Косметический", "Евроремонт"],
            index=1,
        )

    if st.button("💰 Рассчитать цену", type="primary"):
        with st.spinner("Предсказание..."):
            try:
                input_data = {}
                for feat in required_features:
                    if feat == 'rooms':
                        input_data[feat] = rooms_input
                    elif feat == 'area_total':
                        input_data[feat] = area_input
                    elif feat == 'area_living':
                        input_data[feat] = area_input * 0.6
                    elif feat == 'area_kitchen':
                        input_data[feat] = area_input * 0.15
                    elif feat == 'floor':
                        input_data[feat] = floor_input
                    elif feat == 'floors_total':
                        input_data[feat] = floors_total_input
                    elif feat == 'floor_ratio':
                        input_data[feat] = floor_input / max(floors_total_input, 1)
                    elif feat == 'building_year':
                        input_data[feat] = building_year_input
                    elif feat == 'building_age':
                        input_data[feat] = datetime.now().year - building_year_input
                    elif feat == 'company_ad':
                        input_data[feat] = int(company_ad)
                    elif feat == 'is_first_floor':
                        input_data[feat] = int(floor_input == 1)
                    elif feat == 'is_last_floor':
                        input_data[feat] = int(floor_input == floors_total_input)
                    elif feat == 'is_single_floor':
                        input_data[feat] = int(floors_total_input == 1)
                    elif feat == 'has_furniture':
                        input_data[feat] = int(has_furniture)
                    elif feat == 'has_appliances':
                        input_data[feat] = int(has_appliances)
                    elif feat == 'has_balcony_text':
                        input_data[feat] = int(has_balcony)
                    elif feat == 'has_parking':
                        input_data[feat] = 0
                    elif feat == 'has_concierge':
                        input_data[feat] = 0
                    elif feat == 'has_elevator':
                        input_data[feat] = int(has_elevator)
                    elif feat == 'no_animals':
                        input_data[feat] = 0
                    elif feat == 'renovation_euro':
                        input_data[feat] = int(renovation_type == "Евроремонт")
                    elif feat == 'renovation_cosmetic':
                        input_data[feat] = int(renovation_type == "Косметический")
                    elif feat == 'renovation_none':
                        input_data[feat] = int(renovation_type == "Без ремонта")
                    elif feat == 'owner_rents':
                        input_data[feat] = int(not company_ad)
                    elif feat == 'metro_nearby':
                        input_data[feat] = int(metro_dist_input <= 500)
                    elif feat == 'metro_distance':
                        input_data[feat] = metro_dist_input
                    elif feat == 'distance_to_center':
                        if metro_station_input and metro_station_input in STATION_COORDS:
                            lat, lon = STATION_COORDS[metro_station_input]
                            input_data[feat] = haversine(lat, lon, CENTER_LAT, CENTER_LON)
                        else:
                            input_data[feat] = metro_dist_input * 0.3 + 3000
                    elif feat == 'log_metro_distance':
                        input_data[feat] = np.log1p(metro_dist_input)
                    elif feat == 'floor_position':
                        if floor_input == 1:
                            input_data[feat] = 0
                        elif floor_input == floors_total_input:
                            input_data[feat] = 2
                        else:
                            input_data[feat] = 1
                    elif feat == 'is_studio':
                        input_data[feat] = int(rooms_input == 1)
                    elif feat == 'is_large':
                        input_data[feat] = int(rooms_input >= 4)
                    elif feat == 'total_x_metro':
                        input_data[feat] = area_input * int(metro_dist_input <= 500)
                    elif feat == 'building_age_sq':
                        age = datetime.now().year - building_year_input
                        input_data[feat] = age ** 2 / 100
                    elif feat in ('month', 'day_of_week', 'is_weekend', 'days_since_listed'):
                        input_data[feat] = 0
                    elif feat in ('has_area_living', 'has_area_kitchen', 'has_building_year', 'has_floor_info'):
                        input_data[feat] = 1
                    else:
                        input_data[feat] = 0

                input_df = pd.DataFrame([input_data])[required_features]
                pred_raw = pipeline.predict(input_df)[0]
                pred_price = np.expm1(pred_raw) if use_log else pred_raw

                if np.isinf(pred_price) or np.isnan(pred_price) or pred_price > 50000:
                    st.warning("Предсказание выходит за разумные пределы. Попробуйте другие параметры.")
                else:
                    st.success(f"🎯 Справедливая цена: **${pred_price:.0f}** / месяц")

                    avg_price = df['price_usd'].mean()
                    diff = pred_price - avg_price
                    if diff > 0:
                        st.info(
                            f"📈 Выше средней по рынку на ${diff:.0f} (средняя: ${avg_price:.0f})")
                    else:
                        st.info(
                            f"📉 Ниже средней по рынку на ${abs(diff):.0f} (средняя: ${avg_price:.0f})")

                    model_mae = model_info.get('mae', 335)
                    model_r2 = model_info.get('r2', 0.39)
                    model_name = model_info.get('best_model', 'XGBoost')
                    st.caption(f"Модель: {model_name}, "
                               f"R²={model_r2:.3f}, "
                               f"средняя ошибка ±${model_mae:.0f}")

            except Exception as e:
                st.error(f"Ошибка при расчёте: {e}")
                with st.expander("Детали ошибки"):
                    st.code(traceback.format_exc())

# ── ЭКСПОРТ ──
st.sidebar.divider()
st.sidebar.header("💾 Экспорт")

csv = df_filtered.to_csv(index=False).encode('utf-8-sig')
st.sidebar.download_button(
    label="📥 Скачать CSV",
    data=csv,
    file_name=f'rentals_minsk_{datetime.now().strftime("%Y-%m-%d")}.csv',
    mime='text/csv',
)

filters_text = (
    f"Sources: {', '.join(SOURCE_LABELS.get(s, s) for s in sources)} | "
    f"Rooms: {', '.join(str(r) for r in rooms_filter)} | "
    f"Price: ${price_min}-${price_max}"
)

if st.sidebar.button("📄 Сгенерировать PDF", type="primary"):
    with st.spinner("Генерация PDF..."):
        try:
            st.session_state.pdf_data = make_pdf(df, df_filtered, filters_text)
        except Exception as e:
            st.sidebar.error(f"Ошибка PDF: {e}")
            st.session_state.pdf_data = None

if st.session_state.get('pdf_data') is not None:
    pdf_bytes = st.session_state.pdf_data
    if isinstance(pdf_bytes, bytearray):
        pdf_bytes = bytes(pdf_bytes)
    st.sidebar.download_button(
        label="📄 Скачать PDF",
        data=pdf_bytes,
        file_name=f'report_minsk_{datetime.now().strftime("%Y-%m-%d")}.pdf',
        mime='application/pdf',
    )

cache_placeholder = st.sidebar.empty()
if st.sidebar.button("🔄 Очистить кэш"):
    st.cache_data.clear()
    st.cache_resource.clear()
    cache_placeholder.success("Кэш очищен! Перезагрузите страницу.")

st.divider()
st.caption(
    f"📊 Проект: Анализ рынка аренды Минска • Данные: Kufar.by, Realt.by • Модель: {model_info.get('model_type', model_info.get('best_model', 'StackingEnsemble')) if model_info else 'StackingEnsemble'}")
st.caption("🔗 [GitHub](https://github.com/Grinskirm) • Сделано с ❤️")
