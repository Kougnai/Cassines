import gspread
from google.oauth2.service_account import Credentials
import pandas as pd
import streamlit as st
import plotly.express as px
import requests
from prophet import Prophet
import datetime

# ==========================================
#  1. CONFIGURATION DE LA PAGE & STYLE UI
# ==========================================
st.set_page_config(page_title="Dashboard Cassines", layout="wide")

st.markdown("""
    <style>
    .stButton>button, .stMetric, .stTabs [data-baseweb="tab"], div[data-testid="stExpander"] {
        border-radius: 12px !important;
        border: none !important;
    }
    [data-testid="stMetric"] {
        background-color: #151921 !important;
        padding: 20px !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3) !important;
        border-left: 4px solid #E63946 !important;
        transition: transform 0.2s ease-in-out !important;
    }
    [data-testid="stMetric"]:hover {
        transform: translateY(-3px) !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: transparent;
        justify-content: center;
        width: 100%;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #151921;
        color: #808495;
        padding: 10px 20px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #E63946 !important;
        color: white !important;
    }
    header {visibility: hidden;}
    footer {visibility: hidden;}
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 95%;
    }
    </style>
""", unsafe_allow_html=True)

st.title('Les Cassines', text_alignment='center')
st.header('Tableau de bord', text_alignment='center')

# ==========================================
#  2. FONCTIONS DE GESTION DES DONNÉES
# ==========================================
@st.cache_data(ttl=900)
def get_data():
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds_info = dict(st.secrets["gcp_service_account"])
    
    if "private_key" in creds_info:
        creds_info["private_key"] = creds_info["private_key"].replace("\\n", "\n")
 
    creds = Credentials.from_service_account_info(creds_info, scopes=scope)
    client = gspread.authorize(creds)
    spreadsheet = client.open("Cassines_bdd")
    
    onglets = ['Ventes', 'Caisse', 'Events', 'Rh', 'Cash', 'Bon_livraison', 'Facture', 'Stock', 'Enveloppe', 'Bp26']
    return {nom: pd.DataFrame(spreadsheet.worksheet(nom).get_all_records(value_render_option='FORMATTED_VALUE')) for nom in onglets}

@st.cache_data
def add_weather_data(df):
    lat, lon = 45.84, 6.21 # Talloires-Montmin
    start = df['Date'].min().strftime('%Y-%m-%d')
    end = df['Date'].max().strftime('%Y-%m-%d')
    url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={start}&end_date={end}&daily=temperature_2m_max,precipitation_sum&timezone=Europe%2FParis"
    response = requests.get(url).json()
    df_meteo = pd.DataFrame(response['daily'])
  
    df_meteo['time'] = pd.to_datetime(df_meteo['time'])
    df_meteo.columns = ['Date', 'Temp_Max', 'Pluie_mm']
    return pd.merge(df, df_meteo, on='Date', how='left')

@st.cache_data(ttl=3600)
def get_weather_forecast():
    """Récupère les prévisions météo à 7 jours pour Talloires-Montmin (74290)"""
    lat, lon = 45.84, 6.21 
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&daily=temperature_2m_max,precipitation_sum&timezone=Europe%2FParis"
    try:
        response = requests.get(url).json()
        df_forecast = pd.DataFrame(response['daily'])
        df_forecast['time'] = pd.to_datetime(df_forecast['time'])
        df_forecast.columns = ['ds', 'Temp_Max', 'Pluie_mm']
        return df_forecast
    except:
        futur_dates = [datetime.date.today() + datetime.timedelta(days=i) for i in range(1, 8)]
        return pd.DataFrame({'ds': pd.to_datetime(futur_dates), 'Temp_Max': [22.0]*7, 'Pluie_mm': [0.0]*7})

# ==========================================
#  3. CHARGEMENT & NETTOYAGE DES TABLES
# ==========================================
dfs = get_data()
df_ventes, df_caisse, df_events, df_rh, df_cash, df_bl, df_facture, df_stock, df_enveloppe, df_bp = (
    dfs['Ventes'], dfs['Caisse'], dfs['Events'], dfs['Rh'], dfs['Cash'], 
    dfs['Bon_livraison'], dfs['Facture'], dfs['Stock'], dfs['Enveloppe'], dfs['Bp26']
)

onglets_list = [df_cash, df_caisse, df_events, df_rh, df_ventes, df_bl, df_facture, df_stock, df_enveloppe, df_bp]
col_num = ['Ca_ttc', 'Taxes_20', 'Taxes_10', 'Taxes_5.5', 'Ca_ht', 'Cb', 'Espece', 
           'Cheque', 'Autres_ht', 'Privatisation_ht', 'Food_ht', 'Bev_ht',
           'Nb_de_cvts', 'Autres', 'Tips', 'Autre_ht', 'Montant', 'Montant_ht', 
           'Quantité', 'Prix d\'achat', 'Total', 'Restaurant', 'Guinguette', 'LPB', 'Taxes', 'Montant_ttc']

for df in onglets_list:
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
        df['Date'] = df['Date'].dt.normalize()
    
    cols_presentes = [c for c in col_num if c in df.columns]
    for col in cols_presentes:
        df[col] = df[col].astype(str).replace(r'\s+', '', regex=True).str.replace(',', '.')
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

df_ventes = add_weather_data(df_ventes)

DATA_BP26 = {
    5: {"Restaurant": 120000, "Guinguette": 120000, "Le petit baigneur": 0, "Total": 240000},
    6: {"Restaurant": 150000, "Guinguette": 160000, "Le petit baigneur": 10000, "Total": 320000},
    7: {"Restaurant": 200000, "Guinguette": 250000, "Le petit baigneur": 45000, "Total": 495000},
    8: {"Restaurant": 200000, "Guinguette": 280000, "Le petit baigneur": 45000, "Total": 525000},
    9: {"Restaurant": 140000, "Guinguette": 100000, "Le petit baigneur": 0, "Total": 240000}
}

fmt_euro = lambda x: f"{x:,.0f} €".replace(",", " ")
fmt_euro_2d = lambda x: f"{x:,.2f} €".replace(",", " ")
fmt_qty = lambda x: f"{x:,.0f}".replace(",", " ")

def clean_chart_layout(fig):
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="#808495")
    return fig

tab1, tab2, tab_donnees, tab3, tab4, tab5, tab9 = st.tabs([
    " Vue globale", " Vue par site", " Onglet Données", " Compte Fournisseur", " Cash", " Masse salariale", " Archives"
])

# ==========================================
#  TAB 1 : VUE GLOBALE
# ==========================================
with tab1:
    df_ventes = df_ventes.assign(
        année=df_ventes['Date'].dt.year,
        mois=df_ventes['Date'].dt.month,
        iso_semaine=df_ventes['Date'].dt.isocalendar().week,
        ticket_moyen=df_ventes['Ca_ht'] / df_ventes['Nb_de_cvts'],
        jour_année=df_ventes['Date'].dt.day_of_year,
        jour_semaine=df_ventes['Date'].dt.day_of_week,
    )

    année_n = df_ventes['année'].max()
    année_n_1 = année_n - 1
    dernier_semaine_n = df_ventes.query('année == @année_n')['iso_semaine'].max()
    current_mois = int(df_ventes.query('année == @année_n')['mois'].max())

    df_année_n = df_ventes.query('année == @année_n').copy()
    df_année_n_1_ytd = df_ventes.query('année == @année_n_1 & iso_semaine <= @dernier_semaine_n')

    ca_année_n = df_année_n['Ca_ht'].sum()
    ca_année_n_1_ytd = df_année_n_1_ytd['Ca_ht'].sum()
    delta_ca = ca_année_n - ca_année_n_1_ytd

    nb_cvts_année_n = df_année_n['Nb_de_cvts'].sum()
    nb_cvts_n_1_ytd = df_année_n_1_ytd['Nb_de_cvts'].sum()
    ticket_moyen_n = ca_année_n / nb_cvts_année_n if nb_cvts_année_n else 0
    ticket_moyen_n_1_ytd = ca_année_n_1_ytd / nb_cvts_n_1_ytd if nb_cvts_n_1_ytd else 0
    delta_ticket_moyen = ticket_moyen_n - ticket_moyen_n_1_ytd

    df_rh['année'] = df_rh['Date'].dt.year 
    ms_c_année_n = df_rh.query("année == @année_n")['Montant'].sum() / ca_année_n if ca_année_n else 0
    delta_msc_c = ms_c_année_n - 0.35

    ca_du_mois_en_cours = df_année_n.query('mois == @current_mois')['Ca_ht'].sum()
    bp_global_mensuel = DATA_BP26.get(current_mois, {"Total": 0})["Total"]
    if bp_global_mensuel == 0:
        bp_global_mensuel = sum([v["Total"] for v in DATA_BP26.values()]) / len(DATA_BP26)
        
    ecart_bp = ca_du_mois_en_cours - bp_global_mensuel
    bp_hebdo = bp_global_mensuel / 4
    semaines_ecoulees_mois = max(1, df_année_n.query('mois == @current_mois')['iso_semaine'].nunique())
    ecart_bp_hebdo = (ca_du_mois_en_cours / semaines_ecoulees_mois) - bp_hebdo

    st.subheader(f'KPI : {année_n} — Semaine : N° {dernier_semaine_n}', divider='blue')

    cola, colb, colc, cold = st.columns(4)
    cola.metric("Chiffre d'affaire HT (YTD)", value=fmt_euro(ca_année_n), delta=fmt_euro(delta_ca), delta_description="VS N-1 ISO")
    colb.metric('MS/C Globale', value=f'{ms_c_année_n:.0%}', delta=f'{delta_msc_c:.0%}', delta_color='inverse')
    colc.metric("Écart BP M-Courant", value=fmt_euro(ca_du_mois_en_cours), delta=fmt_euro(ecart_bp), delta_description=f"VS ({fmt_euro(bp_global_mensuel)})")
    cold.metric("Rythme Hebdo M-Courant", value=fmt_euro(ca_du_mois_en_cours / semaines_ecoulees_mois), delta=fmt_euro(ecart_bp_hebdo), delta_description=f"VS ({fmt_euro(bp_hebdo)})")

    cola.metric('Nombre de couverts (YTD)', value=fmt_qty(nb_cvts_année_n), delta=fmt_qty(nb_cvts_année_n - nb_cvts_n_1_ytd), delta_description="VS N-1 ISO")
    colb.metric('Ticket moyen (YTD)', value=fmt_euro_2d(ticket_moyen_n), delta=fmt_euro_2d(delta_ticket_moyen), delta_description="VS N-1 ISO")

    st.write("")
    with st.container(border=True):
        st.markdown(f"#####  Suivi de l'objectif sur le mois en cours (Mois {current_mois})")
        col_mois, col_sem = st.columns(2)
        with col_mois:
            if ecart_bp < 0:
                st.warning(f"Il reste à réaliser **{fmt_euro(abs(ecart_bp))}** pour atteindre l'objectif mensuel global.")
            else:
                st.success(f"Objectif mensuel global dépassé de **{fmt_euro(ecart_bp)}** !")
        with col_sem:
            if ecart_bp_hebdo < 0:
                st.warning(f"Écart de **{fmt_euro(abs(ecart_bp_hebdo))}** par rapport au rythme hebdomadaire requis ce mois-ci.")
            else:
                st.success(f"Rythme hebdomadaire du mois en avance de **{fmt_euro(ecart_bp_hebdo)}** !")

    st.subheader('Évolution du Chiffre d\'affaire YoY', divider='blue')
    cols = st.columns(3)
    with cols[0]: mode = st.segmented_control('**Mode de vue**', options=['mois', 'iso_semaine'], default='mois')
    with cols[1]: year = st.pills('**Choisir l\'année**', options=df_ventes['année'].unique(), default=[2025, 2026], selection_mode='multi')
    with cols[2]: site = st.pills('**Point de vente**', options=df_ventes['Site'].unique(), default='Guinguette', selection_mode='multi')

    ca_month = df_ventes.query('année == @year and Site == @site').groupby(['année', mode])['Ca_ht'].sum().reset_index().sort_values(['année', mode], ascending=True)

    if not ca_month.empty:
        idx_max = ca_month['Ca_ht'].idxmax()
        ca_max = ca_month['Ca_ht'].max()
        max_x = ca_month.loc[idx_max, mode]
        ca_month_copy = ca_month.copy()
        ca_month_copy['année'] = ca_month_copy['année'].astype(str)
        
        fig_total = px.line(ca_month_copy, x=mode, y='Ca_ht', color='année', template='simple_white', labels={mode: f'<b>{mode.capitalize()}</b>', 'Ca_ht': '<b>CA HT</b>'})
        fig_total.add_scatter(x=[max_x], y=[ca_max], mode='markers+text', name='Record Historique', text=[f" Record : {ca_max:,.0f} €".replace(',', ' ')], textposition="top center", marker=dict(color='gold', size=12, symbol='star'), showlegend=False)
        fig_total.update_layout(hovermode='x unified')
        st.plotly_chart(clean_chart_layout(fig_total), use_container_width=True)

    st.subheader(f'Events : {année_n}', divider='blue')
    df_events['année'] = df_events['Date'].dt.year
    var = df_events.query('année == @année_n').groupby('Site')['Ca_ht'].sum().round().reset_index().sort_values('Ca_ht', ascending=False)
    ca_events = var['Ca_ht'].sum()
    ca_events_privat = df_events.query("année == @année_n and Ca_ht > 6000")['Ca_ht'].sum()
    ca_event_exploitation = df_events.query("année == @année_n and Ca_ht < 6000")['Ca_ht'].sum()
    pct_privatision = 1 - (ca_events_privat / ca_events) if ca_events else 0
    pct_exploitation = 1 - (ca_event_exploitation / ca_events) if ca_events else 0
    cvt_event = df_events.query('année == @année_n')['Nb_de_cvts'].sum()
    
    div_p = df_events.query("année == @année_n and Ca_ht > 6000")['Nb_de_cvts'].sum()
    event_tckmean_privat = ca_events_privat / div_p if div_p else 0
    div_e = df_events.query("année == @année_n and Ca_ht < 6000")['Nb_de_cvts'].sum()
    event_tckmean_exploit = ca_event_exploitation / div_e if div_e else 0

    col_1, col_2, col_3 = st.columns(3)
    col_1.metric("**CA HT Général Events**", value=f'{ca_events:,.0f} €'.replace(","," "), delta='100 %')
    col_1.metric('**Nombre total de clients**', value=f'{cvt_event:,.0f}'.replace(",", " "), delta=f'{cvt_event/nb_cvts_année_n:.0%}' if nb_cvts_année_n else '0%', delta_arrow='off', delta_color='blue')
    col_2.metric("**CA — Privatisation**", value=f'{ca_events_privat:,.0f} €'.replace(",", " "), delta=f'{pct_exploitation:.0%}', delta_arrow='off')
    col_2.metric('**Ticket moyen — Privatisation**', value=f'{event_tckmean_privat:,.0f} €'.replace(","," "), delta=f'{div_p:,.0f} cvts'.replace(',', ' '), delta_arrow='off', delta_color='gray')
    col_3.metric("**CA — Hors Privatisation**", value=f'{ca_event_exploitation:,.0f} €'.replace(",", " "), delta=f'{pct_privatision:.0%}', delta_arrow='off')
    col_3.metric('**Ticket moyen — Hors Privatisation**', value=f'{event_tckmean_exploit:,.0f} €'.replace(',',' '), delta=f'{div_e:,.0f} cvts'.replace(',', ' '), delta_arrow='off', delta_color='gray')

    st.subheader('Répartition du chiffre d\'affaires Events — Par site', divider='blue')
    col_a, col_b = st.columns(2)
    with col_a:
        fig_events = px.bar(var, x='Site', y='Ca_ht', template='simple_white', labels={'Ca_ht': "<b>Chiffre d'affaire HT</b>"})
        fig_events.update_traces(textposition='outside', texttemplate='%{value:.3s} €')
        st.plotly_chart(clean_chart_layout(fig_events), use_container_width=True)
    with col_b:
        fig_events_b = px.scatter(df_events.query('année == @année_n'), x='Date', y='Ca_ht', color='Site', size='Nb_de_cvts', template='simple_white')
        st.plotly_chart(clean_chart_layout(fig_events_b), use_container_width=True)

    with st.expander(" Cliquer pour afficher la base de données brute Événements"):
        st.dataframe(df_events, hide_index=True)

# ==========================================
#  TAB 2 : VUE PAR SITE & PRÉVISIONS
# ==========================================
with tab2:
    pv = ["Restaurant", "Guinguette", "Le petit baigneur"]
    st.header('Sélection du point de vente', text_alignment='center')
    aa, ab, ac = st.columns(3)
    with ab:
        site_selectionne = st.pills('', options=pv, default="Guinguette", width=500)

    année_n = df_ventes['année'].max()
    année_n_1 = année_n - 1
    nom_filtre_df = "LPB" if site_selectionne == "Le petit baigneur" else site_selectionne
    
    dernier_semaine_n_site = df_ventes.query('année == @année_n & Site == @nom_filtre_df')['iso_semaine'].max()
    current_mois = int(df_ventes.query('année == @année_n')['mois'].max())
    if pd.isna(dernier_semaine_n_site):
        dernier_semaine_n_site = df_ventes['iso_semaine'].max()

    df_site_n = df_ventes.query('année == @année_n & Site == @nom_filtre_df').copy()
    df_site_n_1_ytd = df_ventes.query('année == @année_n_1 & Site == @nom_filtre_df & iso_semaine <= @dernier_semaine_n_site')

    ca_site_n = df_site_n['Ca_ht'].sum()
    ca_site_n_1_ytd = df_site_n_1_ytd['Ca_ht'].sum()
    delta_ca_site = ca_site_n - ca_site_n_1_ytd

    df_rh['année'] = df_rh['Date'].dt.year
    ms_c_montant = df_rh.query('année == @année_n & Site == @nom_filtre_df')['Montant'].sum()
    ms_c = ms_c_montant / ca_site_n if ca_site_n else 0
    delta_msc = 0.35 - ms_c

    nb_cvt_n = df_site_n['Nb_de_cvts'].sum()
    delta_cvt = nb_cvt_n - df_site_n_1_ytd['Nb_de_cvts'].sum()
    ticket_moyen_n = ca_site_n / nb_cvt_n if nb_cvt_n else 0
    delta_ticket_moyen = ticket_moyen_n - (ca_site_n_1_ytd / df_site_n_1_ytd['Nb_de_cvts'].sum() if df_site_n_1_ytd['Nb_de_cvts'].sum() else 0)

    food_ca = df_site_n['Food_ht'].sum()
    food_cogs = food_ca / ca_site_n if ca_site_n else 0
    bev_ca = df_site_n['Bev_ht'].sum()
    bev_cogs = bev_ca / ca_site_n if ca_site_n else 0

    ca_site_du_mois = df_site_n.query('mois == @current_mois')['Ca_ht'].sum()
    bp_site_mensuel = DATA_BP26.get(current_mois, {}).get(site_selectionne, 0)
    
    ecart_bp_site_mensuel = ca_site_du_mois - bp_site_mensuel
    bp_site_hebdo = bp_site_mensuel / 4
    semaines_ecoulees_site = max(1, df_site_n.query('mois == @current_mois')['iso_semaine'].nunique())
    ecart_bp_site_hebdo = (ca_site_du_mois / semaines_ecoulees_site) - bp_site_hebdo

    st.subheader(f'Performances : {site_selectionne}', divider='blue')

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("CA HT Site (YTD)", fmt_euro(ca_site_n), delta=fmt_euro(delta_ca_site), delta_description='vs N-1 ISO')
    col2.metric('Ratio MS/C Site', f'{ms_c:.1%}', delta=f'{delta_msc:.1%}', delta_color='normal')
    col3.metric("Écart BP Site M-Courant", value=fmt_euro(ca_site_du_mois), delta=fmt_euro(ecart_bp_site_mensuel), delta_description=f"Cible : {fmt_euro(bp_site_mensuel)}")
    col4.metric("Rythme Hebdo Site", value=fmt_euro(ca_site_du_mois / semaines_ecoulees_site), delta=fmt_euro(ecart_bp_site_hebdo), delta_description=f"Cible : {fmt_euro(bp_site_hebdo)}")

    a, b, c, d = st.columns(4)
    a.metric('Couverts (YTD)', fmt_qty(nb_cvt_n), delta=fmt_qty(delta_cvt), delta_description='vs N-1 ISO')
    b.metric('Ticket Moyen (YTD)', fmt_euro_2d(ticket_moyen_n), delta=fmt_euro_2d(delta_ticket_moyen), delta_description='vs N-1 ISO')
    c.metric("Part CA Food TTC", fmt_euro(food_ca), delta=f'{food_cogs:.0%}', delta_arrow='off', delta_color='off', delta_description='Du CA du site')
    d.metric("Part CA Bev TTC", fmt_euro(bev_ca), delta=f'{bev_cogs:.0%}', delta_arrow='off', delta_color='off', delta_description='Du CA du site')

    st.write("")
    with st.container(border=True):
        st.markdown(f"#####  Suivi d'avancement des objectifs : {site_selectionne} (Mois {current_mois})")
        col_sm_site, col_sh_site = st.columns(2)
        with col_sm_site:
            if ecart_bp_site_mensuel < 0:
                st.warning(f"Il reste **{fmt_euro(abs(ecart_bp_site_mensuel))}** de CA HT à générer pour atteindre le BP mensuel de ce site.")
            else:
                st.success(f"Objectif mensuel dépassé de **{fmt_euro(ecart_bp_site_mensuel)}** sur ce site !")
        with col_sh_site:
            if ecart_bp_site_hebdo < 0:
                st.warning(f"Ce site est en retard de **{fmt_euro(abs(ecart_bp_site_hebdo))}** en moyenne hebdomadaire sur ce mois.")
            else:
                st.success(f"Rythme hebdomadaire en avance de **{fmt_euro(ecart_bp_site_hebdo)}** sur ce site !")

    # Graphique Comparatif CA Évolutif & BP
    st.subheader(f'Évolution et Objectifs Mensuels YoY : {site_selectionne}', divider='blue')
    df_site_global = df_ventes.query("Site == @nom_filtre_df").copy()
    
    mois_saison = [5, 6, 7, 8, 9]
    chart_data = pd.DataFrame({'mois': mois_saison})
    ca_n_1 = df_site_global.query("année == @année_n_1").groupby('mois')['Ca_ht'].sum()
    ca_n = df_site_global.query("année == @année_n").groupby('mois')['Ca_ht'].sum()
    
    chart_data['CA N-1'] = chart_data['mois'].map(ca_n_1).fillna(0)
    chart_data['CA Actuel'] = chart_data['mois'].map(ca_n).fillna(0)
    chart_data['Objectif BP'] = chart_data['mois'].apply(lambda m: DATA_BP26.get(m, {}).get(site_selectionne, 0))
    
    df_bars = chart_data.melt(id_vars=['mois'], value_vars=['CA N-1', 'CA Actuel'], var_name='Année', value_name='Chiffre d\'affaires')
    df_bars['mois'] = df_bars['mois'].astype(str)
    
    fig_pv = px.bar(
        df_bars, x='mois', y='Chiffre d\'affaires', 
        color='Année', barmode='group',
        template='simple_white', color_discrete_map={'CA N-1': '#808495', 'CA Actuel': '#E63946'},
        labels={'Chiffre d\'affaires': "<b>Chiffre d'affaires HT (€)</b>", 'mois': '<b>Mois</b>'}
    )
    # AJUSTEMENT 1 : Nettoyage du template texte sans l'espace brisé
    fig_pv.update_traces(texttemplate='<b>%{value:.3s}€</b>', textposition='outside')
    fig_pv.add_scatter(x=chart_data['mois'].astype(str), y=chart_data['Objectif BP'], mode='lines+markers', name='Objectif BP', line=dict(color='gold', width=3, dash='dash'), marker=dict(size=8, symbol='diamond'))
    
    # AJUSTEMENT 2 : Ajout de la vue de survol unifiée (x unified)
    fig_pv.update_layout(hovermode='x unified')
    st.plotly_chart(clean_chart_layout(fig_pv), use_container_width=True)

    # --- NOUVELLE SECTION : MODÈLE PREDICTIF PROPHET DE LA SEMAINE PROCHAINE ---
    st.subheader(f" Prévision : Estimation du CA restant de la semaine ({site_selectionne})", divider='blue')
    
    df_prophet = df_site_global[['Date', 'Ca_ht', 'Temp_Max', 'Pluie_mm']].copy().dropna(subset=['Date', 'Ca_ht'])
    df_prophet.columns = ['ds', 'y', 'Temp_Max', 'Pluie_mm']
    df_prophet = df_prophet.sort_values('ds').reset_index(drop=True)
    
    if len(df_prophet) > 14:
        with st.spinner("Analyse des tendances et récupération des prévisions météo pour Talloires-Montmin (74290)..."):
            model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
            model.add_regressor('Temp_Max')
            model.add_regressor('Pluie_mm')
            model.fit(df_prophet)
            
            df_futur_meteo = get_weather_forecast()
            
            # AJUSTEMENT 4 : Caler la fin des prévisions au Dimanche en cours (ou prochain dimanche)
            aujourdhui = datetime.date.today()
            jours_restants_dimanche = (6 - aujourdhui.weekday()) % 7
            prochain_dimanche = pd.to_datetime(aujourdhui + datetime.timedelta(days=jours_restants_dimanche))
            df_futur_meteo = df_futur_meteo[df_futur_meteo['ds'] <= prochain_dimanche].copy()
            
            if not df_futur_meteo.empty:
                forecast = model.predict(df_futur_meteo)
                forecast['yhat'] = forecast['yhat'].clip(lower=0)
                
                # AJUSTEMENT 3 : Logique d'ouvertures/fermetures par établissement
                forecast['num_mois'] = forecast['ds'].dt.month
                forecast['jour_semaine'] = forecast['ds'].dt.dayofweek  # 0=Lundi, 1=Mardi... 6=Dimanche
                
                def appliquer_calendrier_saisonnier(row):
                    m = row['num_mois']
                    j = row['jour_semaine']
                    
                    # RÈGLE JUIN : Tout fermé Lundi & Mardi
                    if m == 6 and j in [0, 1]:
                        return 0.0
                    
                    # RÈGLE JUILLET : Seul le restaurant est fermé Lundi & Mardi
                    if m == 7 and j in [0, 1] and site_selectionne == "Restaurant":
                        return 0.0
                        
                    return row['yhat']
                
                forecast['yhat'] = forecast.apply(appliquer_calendrier_saisonnier, axis=1)
                ca_total_estime = forecast['yhat'].sum()
                
                p1, p2 = st.columns([1, 2])
                with p1:
                    st.write("")
                    st.metric(
                        label=" Estimation CA Restant (Jusqu'à Dimanche)",
                        value=fmt_euro(ca_total_estime),
                        delta="Ajusté selon Calendrier de Site",
                        delta_color="off"
                    )
                    st.markdown("""
                        **Détails de la prévision :**
                        * Le modèle intègre vos fermetures (Lundi/Mardi en Juin pour tous ; Lundi/Mardi en Juillet uniquement pour le Restaurant).
                        * Les prévisions météo et les volumes historiques sont arrêtés précisément au **Dimanche soir**.
                    """)
                    
                    with st.expander(" Voir la météo prévisionnelle utilisée"):
                        df_meteo_brute_display = df_futur_meteo.copy()
                        df_meteo_brute_display['ds'] = df_meteo_brute_display['ds'].dt.strftime('%d/%m')
                        df_meteo_brute_display.columns = ["Date", "Temp Max (°C)", "Pluie (mm)"]
                        st.dataframe(df_meteo_brute_display, hide_index=True)

                with p2:
                    forecast['Jour'] = forecast['ds'].dt.strftime('%A %d/%m')
                    trad = {'Monday': 'Lun', 'Tuesday': 'Mar', 'Wednesday': 'Mer', 'Thursday': 'Jeu', 'Friday': 'Ven', 'Saturday': 'Sam', 'Sunday': 'Dim'}
                    for eng, fr in trad.items():
                        forecast['Jour'] = forecast['Jour'].str.replace(eng, fr)
                        
                    fig_forecast = px.bar(
                        forecast, x='Jour', y='yhat',
                        title="Répartition estimée du CA jour par jour (Ajustée)",
                        labels={'yhat': 'CA HT Estimé (€)', 'Jour': 'Jour de la semaine'},
                        template='simple_white',
                        color_discrete_sequence=['gold'] # CORRECTION BUG : '#gold' -> 'gold' ou '#FFD700'
                    )
                    fig_forecast.update_traces(texttemplate='<b>%{value:.0f} €</b>', textposition='outside')
                    st.plotly_chart(clean_chart_layout(fig_forecast), use_container_width=True)
            else:
                st.info("Aucun jour prédictible restant pour la semaine en cours.")
    else:
        st.info("Historique de données insuffisant sur ce point de vente spécifique pour générer une prédiction fiable.")

# ==========================================
#  TAB NEW : ONGLET DONNÉES (FILTRES PRÉCIS)
# ==========================================
with tab_donnees:
    st.header(" Exploration et Recherche de Données", divider='blue')
    st.write("Filtrez les ventes par dates et points de vente")
    
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        min_date_val = df_ventes['Date'].min().date() if not df_ventes.empty else datetime.date.today()
        max_date_val = df_ventes['Date'].max().date() if not df_ventes.empty else datetime.date.today()
        
        selected_dates = st.date_input(
            " Sélectionner la plage de dates :",
            value=(min_date_val, max_date_val),
            min_value=min_date_val,
            max_value=max_date_val
        )
    with col_f2:
        options_recherche_site = ["Tous les établissements", "Restaurant", "Guinguette", "LPB"]
        selected_search_site = st.selectbox(" Choisir le point de vente :", options=options_recherche_site)
    
    if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
        start_d, end_d = pd.to_datetime(selected_dates[0]), pd.to_datetime(selected_dates[1])
        
        df_filtre_data = df_ventes.copy()
        df_filtre_data['Date'] = pd.to_datetime(df_filtre_data['Date'])
        df_filtre_data = df_filtre_data[(df_filtre_data['Date'] >= start_d) & (df_filtre_data['Date'] <= end_d)]
        
        if selected_search_site != "Tous les établissements":
            df_filtre_data = df_filtre_data[df_filtre_data['Site'] == selected_search_site]
            
        sum_ca_filtré = df_filtre_data['Ca_ht'].sum()
        sum_cvt_filtré = df_filtre_data['Nb_de_cvts'].sum()
        tck_filtré = sum_ca_filtré / sum_cvt_filtré if sum_cvt_filtré else 0
        
        sf1, sf2, sf3 = st.columns(3)
        sf1.metric("CA HT sur sélection", fmt_euro(sum_ca_filtré))
        sf2.metric("Couverts cumulés", fmt_qty(sum_cvt_filtré))
        sf3.metric("Ticket moyen sur sélection", fmt_euro_2d(tck_filtré))
        
        df_display = df_filtre_data.copy()
        df_display['Date'] = df_display['Date'].dt.strftime('%d/%m/%Y')
        
        cols_ordonnees = ['Date', 'Site', 'Ca_ht', 'Ca_ttc', 'Nb_de_cvts', 'ticket_moyen', 'Food_ht', 'Bev_ht', 'Espece', 'Cb', 'Temp_Max', 'Pluie_mm']
        cols_finales = [c for c in cols_ordonnees if c in df_display.columns] + [c for c in df_display.columns if c not in cols_ordonnees]
        
        st.write("")
        st.dataframe(df_display[cols_finales], hide_index=True, use_container_width=True)
    else:
        st.info("Veuillez sélectionner une date de début et de fin dans le calendrier ci-dessus.")

# ==========================================
#  TAB 3 : COMPTE FOURNISSEUR
# ==========================================
with tab3:
    df_bl['Mois'] = df_bl['Date'].dt.strftime('%Y-%m')
    df_facture['Mois'] = df_facture['Date'].dt.strftime('%Y-%m')
    bl = df_bl.groupby(['Mois','Fournisseur'])['Montant_ht'].sum().reset_index()
    facture = df_facture.groupby(['Mois','Fournisseur'])['Montant_ht'].sum().reset_index()
    
    compte_fournisseur = pd.merge(bl, facture, how='outer', on=['Mois','Fournisseur']).fillna(0)
    compte_fournisseur.columns = ['Mois','Fournisseur', 'Montant BL HT', 'Montant Facture HT']
    compte_fournisseur['Solde HT'] = compte_fournisseur['Montant BL HT'] - compte_fournisseur['Montant Facture HT']
    compte_fournisseur['Taille_Treemap'] = compte_fournisseur['Solde HT'].abs() + 0.01

    st.header('Indicateurs Comptes Fournisseurs', divider='blue')
    total_bl = compte_fournisseur['Montant BL HT'].sum()
    total_facture = compte_fournisseur['Montant Facture HT'].sum()
    solde_total = total_bl - total_facture

    c1, c2, c3 = st.columns(3)
    c1.metric('Total BL HT', f'{total_bl:,.0f} €'.replace(',', ' '))
    c2.metric('Total Factures HT', f'{total_facture:,.0f} €'.replace(',', ' '))
    c3.metric('SOLDE GLOBAL EN COURS', f'{solde_total:,.0f} €'.replace(',', ' '), delta_color="inverse")

    st.subheader('Synthèse Compte Fournisseur (Poids des écarts de soldes)', divider='blue')
    if not compte_fournisseur.empty and compte_fournisseur['Taille_Treemap'].sum() > 0.5:
        fig_fournisseur = px.treemap(compte_fournisseur, path=[px.Constant("Tous les fournisseurs"), 'Fournisseur'], values='Taille_Treemap', color='Solde HT', custom_data=['Fournisseur', 'Solde HT', 'Montant BL HT', 'Montant Facture HT'], color_continuous_scale='RdBu', color_continuous_midpoint=0)
        fig_fournisseur.update_traces(texttemplate="<b>%{label}</b><br>%{customdata[1]:,.0f} €", textposition="middle center", hovertemplate="<b>%{customdata[0]}</b><br>Solde : %{customdata[1]:,.0f} €")
        st.plotly_chart(clean_chart_layout(fig_fournisseur), use_container_width=True)
    else:
        st.info("Aucun écart de solde à afficher pour le moment.")

    st.header('Détail individuel par Compte', divider='blue')
    cols = st.columns(2)
    with cols[0]: sel_fournisseur = st.selectbox('**Sélectionner un fournisseur :**', options=sorted(compte_fournisseur['Fournisseur'].unique()))
    with cols[1]:
        solde_compte = compte_fournisseur.query('Fournisseur == @sel_fournisseur')['Solde HT'].sum()
        st.metric('Solde Restant dû', value=f'{solde_compte:,.0f} €')
    
    df_res = compte_fournisseur.query('Fournisseur == @sel_fournisseur').drop(columns='Taille_Treemap')
    st.dataframe(df_res, hide_index=True, use_container_width=True)

# ==========================================
#  TAB 4 : SUIVI DU CASH
# ==========================================
with tab4:
    recette = df_ventes.query('année == @année_n')['Espece'].sum()
    df_cash["mois"] = df_cash['Date'].dt.month
    depot = df_cash.query('mois > 4')['Montant'].sum()
    df_cash_visuel = df_cash.query('mois > 4').copy()
    df_cash_visuel['Date dépôt'] = df_cash_visuel['Date'].dt.date
    df_cash_visuel = df_cash_visuel[['Date dépôt', 'Montant','Numero_ticket']]

    st.header('Suivi Flux Espèces', divider='blue')
    cols = st.columns(3)
    with cols[0]: st.metric('CA global TTC — Espèces', value=f'{recette:,.0f} €'.replace(',', ' '))
    with cols[1]: st.metric('Total Espèces Déposées', value=f'{depot:,.0f} €'.replace(',', ' '))
    with cols[2]: st.metric('Solde Théorique Coffre', value=f'{(recette-depot):,.0f} €'.replace(',', ' '))

    st.subheader('Historique des dépôt de cash', divider = 'blue')
    with st.expander('**Historique des Dépôts**'):
        st.dataframe(df_cash_visuel, hide_index=True, use_container_width=True)

    st.subheader('Audit de Cohérence (Enveloppes vs Caisse)', divider='blue')
    audit_cash_caisse = df_caisse[['Date', 'Site', 'Espece']].copy()
    audit_cash_caisse['Date'] = pd.to_datetime(audit_cash_caisse['Date']).dt.date
    df_enveloppe_clean = df_enveloppe.copy()
    df_enveloppe_clean['Date'] = pd.to_datetime(df_enveloppe_clean['Date']).dt.date

    audit_cash_caisse_agg = audit_cash_caisse.groupby(['Date', 'Site'], as_index=False)['Espece'].sum()
    df_enveloppe_agg = df_enveloppe_clean.groupby(['Date', 'Site'], as_index=False)['Montant'].sum()

    audit_cash = pd.merge(audit_cash_caisse_agg, df_enveloppe_agg, on=['Date', 'Site'], how='left')
    audit_cash['Montant'] = audit_cash['Montant'].fillna(0)
    audit_cash['Ecarts'] = (audit_cash['Espece'] - audit_cash['Montant']).round(2)
    df_historique_cash = audit_cash.sort_values(by='Date', ascending=False).head(10)

    with st.expander(" Afficher l'historique d'audit des 10 derniers jours", expanded=False):
        st.dataframe(df_historique_cash, hide_index=True, use_container_width=True)

# ==========================================
#  TAB 5 : MASSE SALARIALE
# ==========================================
with tab5:
    st.header('Suivi Analyse de la Masse Salariale', divider='blue')
    df_rh_analyse = df_rh.copy()
    df_ventes_rh = df_ventes.query("année == @année_n").copy()

    date_rh = df_rh_analyse['Date'].dt
    df_rh_analyse = df_rh_analyse.assign(année=date_rh.year, mois=date_rh.month, iso_semaine=date_rh.isocalendar().week)
    df_rh_analyse_année_n = df_rh_analyse.query('année == @année_n')
    
    rh_synthese = df_rh_analyse_année_n.groupby(['iso_semaine','Site'])['Montant'].sum().reset_index()
    rh_ventes_synthese = df_ventes_rh.groupby(['iso_semaine', 'Site'])['Ca_ht'].sum().reset_index()
    globale_rh = pd.merge(rh_ventes_synthese, rh_synthese, how='outer', on=['iso_semaine','Site'])
    globale_rh['Ratio (%)'] = ((globale_rh['Montant'] / globale_rh['Ca_ht'] ) * 100 ).round(2)
    globale_rh['Valeur cible'] = 35
    globale_rh.columns = ['Semaine ISO', 'Site', "Chiffre d'affaire HT", "Masse salariale chargée", 'Ratio (%)', "Valeur cible"]

    st.subheader('Sélection du périmètre d\'analyse', divider='blue')
    site_rh = st.pills('', options=globale_rh['Site'].unique(), default='Restaurant')
    var_rh = globale_rh.query('Site == @site_rh').groupby(['Semaine ISO', 'Site']).agg({"Chiffre d'affaire HT": 'sum', "Masse salariale chargée": 'sum', "Ratio (%)": 'mean', "Valeur cible": 'mean'}).round().reset_index()

    ca_rh = var_rh["Chiffre d'affaire HT"].sum()
    msc_rh = var_rh["Masse salariale chargée"].sum()
    ratio_rh = msc_rh / ca_rh if ca_rh else 0
    delta_rh = 0.35 - ratio_rh
    ecart_rh_val = ca_rh * delta_rh * -1

    cols = st.columns(3)
    cols[0].metric("CA HT Cumulé (Période)", value=f'{ca_rh:,.0f} €'.replace(",", " "), delta=f'{ca_rh / ca_année_n :.0%}' if ca_année_n else '0%')
    cols[1].metric("Masse salariale chargée", value=f'{msc_rh:,.0f} €'.replace(",", " "), delta=f'{ecart_rh_val:,.0f} €'.replace(",", " "), delta_color='inverse')
    cols[2].metric("Ratio Réel MS/C", value=f'{ratio_rh:.2%}', delta=f'{delta_rh:.2%}', delta_arrow='off')

    fig_rh = px.bar(var_rh, x='Semaine ISO', y='Ratio (%)', color='Site', template='plotly_white')
    fig_rh.add_scatter(x=var_rh['Semaine ISO'], y=var_rh['Valeur cible'], name='Objectif Limite 35%')
    st.plotly_chart(clean_chart_layout(fig_rh), use_container_width=True)

# ==========================================
#  TAB 9 : ARCHIVES
# ==========================================
with tab9:
    with st.expander(' Consulter l\'historique complet de la Master Data (Ventes)'):
        df_ventes['Date'] = df_ventes['Date'].dt.date
        st.dataframe(df_ventes, hide_index=True, use_container_width=True)
