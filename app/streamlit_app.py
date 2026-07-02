import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import traceback
import plotly.express as px
from datetime import datetime

from src.config import STATION_COORDS, CENTER_LAT, CENTER_LON, haversine, get_data_path, get_models_path

st.set_page_config(
    page_title="Аренда Минска — Дашборд",
    page_icon="🏠",
    layout="wide",
)

st.title("🏠 Рынок аренды квартир в Минске")
st.caption("Данные собраны с Kufar.by и Realt.by")


@st.cache_data(ttl=3600)
def load_data():
    csv_path = get_data_path('processed', 'rentals_clean.csv')
    if not os.path.exists(csv_path):
        st.error(f"Файл данных не найден: {csv_path}")
        st.stop()
    return pd.read_csv(csv_path)


@st.cache_data(ttl=3600)
def load_timeseries():
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
        return None, None
    pipeline = joblib.load(model_path)
    with open(info_path, 'r') as f:
        model_info = json.load(f)
    return pipeline, model_info


try:
    df = load_data()
except Exception as e:
    st.error(f"Ошибка загрузки данных: {e}")
    st.stop()

df_ts = load_timeseries()
pipeline, model_info = load_model()

# ── САЙДБАР — ФИЛЬТРЫ ──
st.sidebar.header("🔍 Фильтры")

sources = st.sidebar.multiselect(
    "Источник",
    options=['kufar', 'realt'],
    default=['kufar', 'realt'],
)

max_rooms = int(df['rooms'].max()) if 'rooms' in df.columns else 5
rooms_filter = st.sidebar.multiselect(
    "Комнат",
    options=list(range(1, max_rooms + 1)),
    default=[1, 2, 3],
)

price_min, price_max = st.sidebar.slider(
    "Цена (USD)",
    min_value=0, max_value=20000,
    value=(100, 2000),
    step=50,
)

df_filtered = df[
    (df['source'].isin(sources)) &
    (df['rooms'].isin(rooms_filter)) &
    (df['price_usd'].between(price_min, price_max))
]

# ── KPI ──
col1, col2, col3, col4 = st.columns(4)

if df_filtered.empty:
    st.warning("⚠️ Нет объявлений, соответствующих выбранным фильтрам.")
    st.stop()

with col1:
    st.metric("📋 Объявлений", f"{len(df_filtered):,}")

with col2:
    st.metric("💰 Средняя цена", f"${df_filtered['price_usd'].mean():.0f}")

with col3:
    st.metric("📊 Медианная цена", f"${df_filtered['price_usd'].median():.0f}")

with col4:
    st.metric("📐 Средняя площадь",
              f"{df_filtered['area_total'].mean():.0f} м²")

# ── ВКЛАДКИ ──
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 Распределение цен",
    "🏙️ По комнатам",
    "🏢 Kufar vs Realt",
    "📈 Динамика цен",
    "🌡️ Дополнительно",
    "🗺️ На карте",
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
        )
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.box(
            df_filtered, x='rooms', y='price_usd',
            title="Цена по количеству комнат",
            color_discrete_sequence=['#3498db'],
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Статистика по площади")
    fig = px.scatter(
        df_filtered.sample(min(1000, len(df_filtered))),
        x='area_total', y='price_usd',
        color='rooms',
        title="Цена vs Площадь",
        color_discrete_sequence=px.colors.qualitative.Bold,
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

# ── TAB 2: ПО КОМНАТАМ ──
with tab2:
    st.subheader("Статистика по количеству комнат")

    stats = df_filtered.groupby('rooms').agg(
        Количество=('price_usd', 'count'),
        Средняя_цена=('price_usd', 'mean'),
        Медианная_цена=('price_usd', 'median'),
        Мин_цена=('price_usd', 'min'),
        Макс_цена=('price_usd', 'max'),
        Средняя_площадь=('area_total', 'mean'),
    ).round(0)

    st.dataframe(stats, use_container_width=True)

    fig = px.histogram(
        df_filtered, x='price_usd', color='rooms',
        title="Распределение цен по комнатам",
        barmode='overlay', nbins=40,
        color_discrete_sequence=px.colors.qualitative.Bold,
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

# ── TAB 3: KUFAR VS REALT ──
with tab3:
    st.subheader("Сравнение Kufar.by и Realt.by")

    col1, col2 = st.columns(2)

    with col1:
        source_counts = df_filtered['source'].value_counts()
        fig = px.pie(
            values=source_counts.values,
            names=source_counts.index,
            title="Доля объявлений по источникам",
            color_discrete_sequence=['#e94560', '#3498db'],
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.box(
            df_filtered, x='source', y='price_usd',
            title="Цены: Kufar vs Realt",
            color='source',
            color_discrete_sequence=['#e94560', '#3498db'],
        )
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Детальное сравнение")
    source_stats = df_filtered.groupby('source').agg(
        Объявлений=('price_usd', 'count'),
        Средняя_цена=('price_usd', 'mean'),
        Медианная_цена=('price_usd', 'median'),
        Средняя_площадь=('area_total', 'mean'),
        Агентств=('company_ad', 'sum'),
    ).round(0)
    source_stats['% агентств'] = (
        source_stats['Агентств'] / source_stats['Объявлений'] * 100).round(1)

    st.dataframe(source_stats, use_container_width=True)

# ── TAB 4: ДИНАМИКА ЦЕН ──
with tab4:
    st.subheader("📈 Динамика цен во времени")

    if df_ts is None or df_ts.empty:
        st.warning("Нет данных временных рядов. Запустите `python src/build_timeseries.py` для сборки.")
    else:
        # Фильтр по источнику для временных рядов
        ts_sources = st.multiselect(
            "Источник (временные ряды)",
            options=df_ts['source'].unique(),
            default=df_ts['source'].unique(),
            key='ts_source',
        )
        ts_rooms = st.multiselect(
            "Комнат (временные ряды)",
            options=sorted(df_ts['rooms'].dropna().unique().astype(int)),
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
            col1, col2 = st.columns(2)

            with col1:
                # Средняя цена по комнатам по дням
                avg_by_rooms = df_ts_filtered.groupby(
                    ['snapshot_date', 'rooms'], as_index=False
                )['price_usd'].mean()

                fig = px.line(
                    avg_by_rooms,
                    x='snapshot_date', y='price_usd', color='rooms',
                    title="Средняя цена по комнатам по дням",
                    markers=True,
                    color_discrete_sequence=px.colors.qualitative.Bold,
                )
                fig.update_layout(height=400, xaxis_title="Дата", yaxis_title="Средняя цена (USD)")
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Количество объявлений по дням
                count_by_date = df_ts_filtered.groupby(
                    'snapshot_date', as_index=False
                ).size().rename(columns={'size': 'count'})

                fig = px.bar(
                    count_by_date,
                    x='snapshot_date', y='count',
                    title="Количество объявлений по дням",
                    color_discrete_sequence=['#3498db'],
                    text_auto=True,
                )
                fig.update_layout(height=400, xaxis_title="Дата", yaxis_title="Количество")
                st.plotly_chart(fig, use_container_width=True)

            col1, col2 = st.columns(2)

            with col1:
                # Средняя цена за м² по дням
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
                )
                fig.update_layout(height=400, xaxis_title="Дата", yaxis_title="Цена за м² (USD)")
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Kufar vs Realt по дням
                avg_by_source = df_ts_filtered.groupby(
                    ['snapshot_date', 'source'], as_index=False
                )['price_usd'].mean()

                fig = px.line(
                    avg_by_source,
                    x='snapshot_date', y='price_usd', color='source',
                    title="Средняя цена: Kufar vs Realt",
                    markers=True,
                    color_discrete_sequence=['#e94560', '#3498db'],
                )
                fig.update_layout(height=400, xaxis_title="Дата", yaxis_title="Средняя цена (USD)")
                st.plotly_chart(fig, use_container_width=True)

# ── TAB 5: ДОПОЛНИТЕЛЬНО ──
with tab5:
    st.subheader("🌡️ Дополнительная аналитика")

    if df_ts is None or df_ts.empty:
        st.warning("Нет данных временных рядов для доп. аналитики.")
    else:
        ts5_sources = st.multiselect(
            "Источник",
            options=df_ts['source'].unique(),
            default=df_ts['source'].unique(),
            key='ts5_source',
        )
        ts5_rooms = st.multiselect(
            "Комнат",
            options=sorted(df_ts['rooms'].dropna().unique().astype(int)),
            default=[1, 2, 3],
            key='ts5_rooms',
        )

        df_ts5 = df_ts[
            (df_ts['source'].isin(ts5_sources)) &
            (df_ts['rooms'].isin(ts5_rooms))
        ].copy()

        if df_ts5.empty:
            st.info("Нет данных для выбранных фильтров.")
            st.stop()

        col1, col2 = st.columns(2)

        with col1:
            # Средняя цена по станциям метро (из realt)
            if 'metro_station' in df_ts5.columns:
                metro_stats = df_ts5[
                    df_ts5['metro_station'].notna() & (df_ts5['metro_station'] != '')
                ].groupby('metro_station', as_index=False)['price_usd'].mean()
                metro_stats = metro_stats.sort_values('price_usd', ascending=False).head(10)

                if not metro_stats.empty:
                    fig = px.bar(
                        metro_stats,
                        x='price_usd', y='metro_station',
                        title="Топ-10 станций метро по средней цене",
                        orientation='h',
                        color='price_usd',
                        color_continuous_scale='Viridis',
                        text_auto='.0f',
                    )
                    fig.update_layout(height=500, xaxis_title="Средняя цена (USD)", yaxis_title="")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Нет данных по станциям метро.")

        with col2:
            # Доля собственников vs агентств по дням
            owner_col = 'company_ad'
            if owner_col in df_ts5.columns:
                owner_by_date = df_ts5.groupby(
                    'snapshot_date', as_index=False
                ).agg(
                    Всего=('price_usd', 'count'),
                    Агентства=('company_ad', 'sum'),
                )
                owner_by_date['Собственники'] = owner_by_date['Всего'] - owner_by_date['Агентства']

                fig = px.area(
                    owner_by_date,
                    x='snapshot_date',
                    y=['Собственники', 'Агентства'],
                    title="Доля собственников vs агентств по дням",
                    color_discrete_map={'Собственники': '#2ecc71', 'Агентства': '#e74c3c'},
                )
                fig.update_layout(height=500, xaxis_title="Дата", yaxis_title="Количество")
                st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)

        with col1:
            # Количество объявлений по дням с разбивкой по источнику
            count_by_source = df_ts5.groupby(
                ['snapshot_date', 'source'], as_index=False
            ).size().rename(columns={'size': 'count'})

            fig = px.bar(
                count_by_source,
                x='snapshot_date', y='count', color='source',
                title="Количество объявлений по дням (по источникам)",
                barmode='group',
                color_discrete_sequence=['#e94560', '#3498db'],
                text_auto=True,
            )
            fig.update_layout(height=400, xaxis_title="Дата", yaxis_title="Количество")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Распределение по состоянию (condition)
            if 'condition' in df_ts5.columns:
                cond_counts = df_ts5['condition'].value_counts().reset_index()
                cond_counts.columns = ['condition', 'count']
                cond_counts = cond_counts.dropna()

                if not cond_counts.empty:
                    fig = px.pie(
                        cond_counts,
                        values='count', names='condition',
                        title="Распределение по состоянию квартир",
                        color_discrete_sequence=px.colors.qualitative.Set2,
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)

# ── TAB 6: НА КАРТЕ ──
with tab6:
    st.subheader("🗺️ Объявления на карте Минска")

    lat_col = 'location_lat' if 'location_lat' in df_filtered.columns else 'lat'
    lon_col = 'location_lon' if 'location_lon' in df_filtered.columns else 'lon'
    has_coords = df_filtered[lat_col].notna() & df_filtered[lon_col].notna()
    df_map = df_filtered[has_coords].copy()

    if df_map.empty:
        st.info("Нет объявлений с координатами для отображения на карте.")
    elif len(df_map) < 2:
        st.info("Слишком мало точек для отображения на карте.")
    else:
        sample_size = min(500, len(df_map))
        df_map_sample = df_map.sample(sample_size, random_state=42)

        fig = px.scatter_mapbox(
            df_map_sample,
            lat=lat_col,
            lon=lon_col,
            color='price_usd',
            size='area_total',
            hover_name='source',
            hover_data={
                'price_usd': ':$',
                'rooms': True,
                'area_total': ':.,1f м²',
                lat_col: False,
                lon_col: False,
            },
            color_continuous_scale='Viridis',
            zoom=10,
            height=600,
            title=f"Показано {sample_size} объявлений",
        )
        fig.update_layout(mapbox_style='open-street-map')
        fig.update_layout(margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig, use_container_width=True)

# ── TAB 7: КАЛЬКУЛЯТОР ЦЕНЫ ──
with tab7:
    st.subheader("🤖 Калькулятор справедливой цены")
    st.caption("Модель машинного обучения (XGBoost + Random Forest) предсказывает цену на основе параметров квартиры.")

    if pipeline is None or model_info is None:
        st.error("Модель не найдена. Сначала обучите модель.")
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
                st.error(f"Некорректное значение. Попробуйте другие параметры.")
                st.stop()

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

cache_placeholder = st.sidebar.empty()
if st.sidebar.button("🔄 Очистить кэш"):
    st.cache_data.clear()
    st.cache_resource.clear()
    cache_placeholder.success("Кэш очищен! Перезагрузите страницу.")

st.divider()
st.caption(
    "📊 Проект: Анализ рынка аренды Минска • Данные: Kufar.by, Realt.by • Модель: Stacking Ensemble (XGBoost + Random Forest)")
st.caption("🔗 [GitHub](https://github.com/Grinskirm) • Сделано с ❤️")
