import sys, os, io, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fpdf import FPDF
import pandas as pd
import numpy as np
from datetime import datetime

SOURCE_LABELS = {'kufar': 'Kufar.by', 'realt': 'Realt.by'}
ROOM_LABELS = {1: '1 комната', 2: '2 комнаты', 3: '3 комнаты', 4: '4 комнаты', 5: '5 комнат'}


def sanitize(text):
    t = str(text)
    t = t.replace('\u2014', '-').replace('\u2013', '-')
    t = t.replace('\u2022', '*').replace('\u2026', '...')
    t = t.replace('\u00AB', '<<').replace('\u00BB', '>>')
    t = re.sub(r'[^\x00-\xFF]', '?', t)
    return t


class Report(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 10)
        self.set_text_color(233, 69, 96)
        self.cell(0, 8, sanitize('Rynok arendy Minska - Otchet'), align='L')
        self.ln(12)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, sanitize(f'Stranitsa {self.page_no()}/{{nb}}'), align='C')

    def section_title(self, title):
        self.set_font('Helvetica', 'B', 13)
        self.set_text_color(26, 26, 46)
        self.cell(0, 10, sanitize(title))
        self.ln(8)

    def kpi_block(self, kpis):
        self.set_font('Helvetica', '', 10)
        self.set_text_color(60, 60, 80)
        line = '  |  '.join(f'{sanitize(k)}: {sanitize(v)}' for k, v in kpis)
        self.multi_cell(0, 7, line)
        self.ln(4)

    def data_table(self, headers, rows, col_widths=None):
        self.set_font('Helvetica', 'B', 9)
        self.set_fill_color(233, 69, 96)
        self.set_text_color(255, 255, 255)

        if col_widths is None:
            col_widths = [self.w / (len(headers) + 1)] * len(headers)

        for i, h in enumerate(headers):
            self.cell(col_widths[i], 7, sanitize(h), border=1, fill=True, align='C')
        self.ln()

        self.set_font('Helvetica', '', 9)
        self.set_text_color(26, 26, 46)

        for row in rows:
            is_odd = rows.index(row) % 2 == 0
            if is_odd:
                self.set_fill_color(245, 245, 250)
            else:
                self.set_fill_color(255, 255, 255)

            for i, cell in enumerate(row):
                align = 'C' if isinstance(cell, (int, float)) else 'L'
                val = f'{cell:,.0f}' if isinstance(cell, float) and abs(cell) < 1e6 else str(cell)
                self.cell(col_widths[i], 6, sanitize(val), border=1, fill=True, align=align)
            self.ln()


def generate_pdf(df_full, df_filtered, filters_text):
    pdf = Report()
    pdf.alias_nb_pages()
    pdf.add_page()

    # Title
    pdf.set_font('Helvetica', 'B', 18)
    pdf.set_text_color(26, 26, 46)
    pdf.cell(0, 12, sanitize('Otchet rynka arendy Minska'))
    pdf.ln()

    pdf.set_font('Helvetica', '', 9)
    pdf.set_text_color(150, 150, 150)
    pdf.cell(0, 6, sanitize(f'Generirovan: {datetime.now().strftime("%d.%m.%Y %H:%M")}'))
    pdf.ln()
    pdf.set_font('Helvetica', 'I', 9)
    pdf.multi_cell(0, 6, sanitize(f'Filtry: {filters_text}'))
    pdf.ln(6)

    # KPI
    pdf.section_title('Osnovnye pokazateli')
    kpis = [
        ('Obyavleniy', f'{len(df_filtered):,}'),
        ('Srednyaya cena', f'${df_filtered["price_usd"].mean():.0f}'),
        ('Mediana', f'${df_filtered["price_usd"].median():.0f}'),
        ('Srednyaya ploshad', f'{df_filtered["area_total"].mean():.0f} m2'),
    ]
    pdf.kpi_block(kpis)

    # Table: by rooms
    pdf.section_title('Statistika po kolichestvu komnat')
    room_stats = df_filtered.groupby('rooms').agg(
        count=('price_usd', 'count'),
        mean_price=('price_usd', 'mean'),
        median_price=('price_usd', 'median'),
        min_price=('price_usd', 'min'),
        max_price=('price_usd', 'max'),
        mean_area=('area_total', 'mean'),
    ).round(0)

    headers = ['Komnat', 'Kolichestvo', 'Srednyaya', 'Mediana', 'Min', 'Max', 'Ploshad']
    col_widths = [25, 28, 32, 28, 26, 26, 25]

    rows = []
    for rooms, row in room_stats.iterrows():
        rows.append([f'{int(rooms)}', int(row['count']), f'${row["mean_price"]:.0f}',
                     f'${row["median_price"]:.0f}', f'${row["min_price"]:.0f}',
                     f'${row["max_price"]:.0f}', f'{row["mean_area"]:.0f}'])

    pdf.data_table(headers, rows, col_widths)
    pdf.ln(8)

    # Table: by source
    pdf.section_title('Sravnenie istochnikov')
    src_stats = df_filtered.groupby('source').agg(
        count=('price_usd', 'count'),
        mean_price=('price_usd', 'mean'),
        median_price=('price_usd', 'median'),
        mean_area=('area_total', 'mean'),
        agencies=('company_ad', 'sum'),
    ).round(0)

    headers2 = ['Istochnik', 'Obyavleniy', 'Srednyaya', 'Mediana', 'Ploshad', 'Agentstv', '% agentstv']
    col_widths2 = [25, 28, 30, 28, 24, 24, 30]

    rows2 = []
    for src, row in src_stats.iterrows():
        pct = (row['agencies'] / row['count'] * 100) if row['count'] > 0 else 0
        rows2.append([
            SOURCE_LABELS.get(src, src),
            int(row['count']),
            f'${row["mean_price"]:.0f}',
            f'${row["median_price"]:.0f}',
            f'{row["mean_area"]:.0f}',
            int(row['agencies']),
            f'{pct:.1f}%',
        ])

    pdf.data_table(headers2, rows2, col_widths2)

    # Charts (if possible)
    try:
        import plotly.express as px
        import plotly.io as pio

        chart_added = False

        fig1 = px.bar(
            room_stats.reset_index(),
            x='rooms', y='mean_price',
            title=sanitize('Srednyaya cena po komnatam'),
        )
        fig1.update_layout(showlegend=False, margin=dict(l=20, r=20, t=40, b=20))
        img_bytes = pio.to_image(fig1, format='png', width=600, height=350)
        chart_added = True

        pdf.add_page()
        pdf.section_title('Grafiki')
        with io.BytesIO(img_bytes) as buf:
            pdf.image(buf, x=10, w=180)

        if len(src_stats) >= 2:
            src_df = src_stats.reset_index()
            src_df['source'] = src_df['source'].map(SOURCE_LABELS)
            fig2 = px.bar(
                src_df,
                x='source', y='mean_price',
                title=sanitize('Srednyaya cena: Kufar i Realt'),
                color='source',
                color_discrete_sequence=['#e94560', '#3498db'],
            )
            fig2.update_layout(showlegend=False, margin=dict(l=20, r=20, t=40, b=20))
            img_bytes2 = pio.to_image(fig2, format='png', width=600, height=350)
            pdf.ln(10)
            with io.BytesIO(img_bytes2) as buf:
                pdf.image(buf, x=10, w=180)

    except Exception:
        pass  # charts are optional

    return pdf.output()
