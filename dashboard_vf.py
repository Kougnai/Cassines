import gspread
from google.oauth2.service_account import Credentials
import pandas as pd
import streamlit as st
import plotly.express as px
import requests
from prophet import Prophet
import datetime

## ---- CONFIGURATION DE LA PAGE ---- 
st.set_page_config(page_title="Dashboard Cassines", layout="wide")
st.markdown("""
    <style>
    /* 1. Arrondir les angles de TOUS les éléments (boutons, inputs, metrics) */
    .stButton>button, .stMetric, .stTabs [data-baseweb="tab"], div[data-testid="stExpander"] {
        border-radius: 12px !important;
        border: none !important;
    }

    /* 2. Style "Card" pour les Metrics (tes KPIs) */
    [data-testid="stMetric"] {
        background-color: #151921; /* Gris un peu plus clair que le fond */
        padding: 20px !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
    }

    /* 3. Modernisation et Centrage des Onglets (Tabs) */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: transparent;
        justify-content: center; /* Centre la liste des onglets */
        width: 100%;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #151921;
        color: #808495;
        padding: 10px 20px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #E63946 !important; /* Ton rouge Cassines */
        color: white !important;
    }

    /* 4. Cacher le header Streamlit inutile (Deploy, Menu) */
    header {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* 5. Ajuster les marges pour que ça respire */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 95%;
    }
    </style>
""", unsafe_allow_html=True)
st.title('Les Cassines', text_alignment='center')
st.header('Tableau de bord', text_alignment='center')

@st.cache_data(ttl=900)
def get_data():
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds_info = dict(st.secrets["gcp_service_account"])
    
    if "private_key" in creds_info:
        creds_info["private_key"] = creds_info["private_key"].replace("\\n", "\n")
 
    creds = Credentials.from_service_account_info(creds_info, scopes=scope)
    client = gspread.authorize(creds)
    spreadsheet = client.open("Cassines_bdd")
    
    onglets = ['Ventes', 'Caisse', 'Events', 'Rh', 'Cash','Bon_livraison', 'Facture', 'Stock', 'Enveloppe', 'Bp26']
    data = {nom: pd.DataFrame(spreadsheet.worksheet(nom).get_all_records(value_render_option='FORMATTED_VALUE')) for nom in onglets}
    return data

# Fonction pour ajouter des données directement dans Google Sheets
def append_to_sheet(worksheet_name, row_data):
    try:
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds_info = dict(st.secrets["gcp_service_account"])
        if "private_key" in creds_info:
            creds_info["private_key"] = creds_info["private_key"].replace("\\n", "\n")
        creds = Credentials.from_service_account_info(creds_info, scopes=scope)
        client = gspread.authorize(creds)
        spreadsheet = client.open("Cassines_bdd")
        worksheet = spreadsheet.worksheet(worksheet_name)
        worksheet.append_row(row_data)
        return True
    except Exception as e:
        st.error(f"Erreur lors de l'envoi vers Google Sheets : {e}")
        return False

# --- METEO ---
@st.cache_data
def add_weather_data(df):
    lat, lon = 45.86, 6.17
    start = df['Date'].min().strftime('%Y-%m-%d')
    end = df['Date'].max().strftime('%Y-%m-%d')
    url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={start}&end_date={end}&daily=temperature_2m_max,precipitation_sum,cloudcover_mean,windspeed_10m_max&timezone=Europe%2FParis"
    response = requests.get(url).json()
    df_meteo = pd.DataFrame(response['daily'])
    df_meteo['time'] = pd.to_datetime(df_meteo['time'])
    df_meteo.columns = ['Date', 'Temp_Max', 'Pluie_mm', 'Nuages_%', 'Vent_max']
    return pd.merge(df, df_meteo, on='Date', how='left')

## ---- CHARGEMENT DES DONNÉES -----
dfs = get_data()
df_ventes, df_caisse, df_events, df_rh, df_cash, df_bl, df_facture, df_stock, df_enveloppe, df_bp = (
    dfs['Ventes'], dfs['Caisse'], dfs['Events'], dfs['Rh'], dfs['Cash'], 
    dfs['Bon_livraison'], dfs['Facture'], dfs['Stock'], dfs['Enveloppe'], dfs['Bp26']
)

## ---- NETTOYAGE ET CONVERSION ---
onglets_list = [df_cash, df_caisse, df_events, df_rh, df_ventes, df_bl, df_facture, df_stock, df_enveloppe, df_bp]
col_num = ['Ca_ttc', 'Taxes_20', 'Taxes_10', 'Taxes_5.5','Ca_ht','Cb','Espece', 
            'Cheque', 'Autres_ht', 'Privatisation_ht', 'Food_ht', 'Bev_ht',
             'Nb_de_cvts', 'Autres', 'Tips', 'Autre_ht', 'Montant', 'Montant_ht','Quantité', 'Prix d\'achat', 'Total', 'Restaurant', 'Guinguette', 'LPB', 'Taxes', 'Montant_ttc']

for df in onglets_list:
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
        df['Date'] = df['Date'].dt.normalize()
    
    cols_presentes = [c for c in col_num if c in df.columns]
    for col in cols_presentes:
        df[col] = df[col].astype(str).replace(r'\s+', '', regex=True).str.replace(',', '.')
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

##### ---- AJOUT DES DONNÉES MÉTÉO ----
df_ventes = add_weather_data(df_ventes)

## ----- DICTIONNAIRE BP26 EN DUR -----
# Structure indexée par le numéro de mois (5 à 9)
DATA_BP26 = {
    5: {"Restaurant": 120000, "Guinguette": 120000, "Le petit baigneur": 0, "Total": 240000},
    6: {"Restaurant": 150000, "Guinguette": 160000, "Le petit baigneur": 10000, "Total": 320000},
    7: {"Restaurant": 200000, "Guinguette": 250000, "Le petit baigneur": 45000, "Total": 495000},
    8: {"Restaurant": 200000, "Guinguette": 280000, "Le petit baigneur": 45000, "Total": 525000},
    9: {"Restaurant": 140000, "Guinguette": 100000, "Le petit baigneur": 0, "Total": 240000}
}

## ----- CRÉATION DES TABLES -----
tab1, tab2, tab3, tab4, tab5, tab9, tab_saisie = st.tabs([
    " Vue globale", " Vue par site", " Compte Fournisseur", " Cash", " Masse salariale", " Archives", " 📝 Saisie de Données"
])

# Formatages réutilisables
fmt_euro = lambda x: f"{x:,.0f} €".replace(",", " ")
fmt_euro_2d = lambda x: f"{x:,.2f} €".replace(",", " ")
fmt_qty = lambda x: f"{x:,.0f}".replace(",", " ")

# ==========================================
# 📊 TAB 1 : VUE GLOBALE
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

    # Chiffre d'affaires global
    ca_année_n = df_année_n['Ca_ht'].sum()
    ca_année_n_1_ytd = df_année_n_1_ytd['Ca_ht'].sum()
    delta_ca = ca_année_n - ca_année_n_1_ytd

    # Couverts & Ticket Moyen
    nb_cvts_année_n = df_année_n['Nb_de_cvts'].sum()
    nb_cvts_n_1_ytd = df_année_n_1_ytd['Nb_de_cvts'].sum()
    ticket_moyen_n = ca_année_n / nb_cvts_année_n if nb_cvts_année_n else 0
    ticket_moyen_n_1_ytd = ca_année_n_1_ytd / nb_cvts_n_1_ytd if nb_cvts_n_1_ytd else 0
    delta_ticket_moyen = ticket_moyen_n - ticket_moyen_n_1_ytd

    # RH
    df_rh['année'] = df_rh['Date'].dt.year 
    ms_c_année_n = df_rh.query("année == @année_n")['Montant'].sum() / ca_année_n if ca_année_n else 0
    delta_msc_c = ms_c_année_n - 0.35

    # --- CALCULS BP GLOBAL DEPUIS LES DONNÉES EN DUR ---
    bp_global_mensuel = DATA_BP26.get(current_mois, {"Total": 0})["Total"]
    if bp_global_mensuel == 0:  # Secours si mois hors cible (ex: début de saison)
        bp_global_mensuel = sum([v["Total"] for v in DATA_BP26.values()]) / len(DATA_BP26)
        
    ecart_bp = ca_année_n - bp_global_mensuel
    bp_hebdo = bp_global_mensuel / 4
    ecart_bp_hebdo = (ca_année_n / (dernier_semaine_n if dernier_semaine_n else 1)) - bp_hebdo

    st.subheader(f'KPI : {année_n} Semaine : N° {dernier_semaine_n}', divider='blue')

    cola, colb, colc, cold = st.columns(4)
    cola.metric("Chiffre d'affaire HT", value=fmt_euro(ca_année_n), delta=fmt_euro(delta_ca), delta_description="VS N-1 à la sem. ISO")
    colb.metric('MS/C', value=f'{ms_c_année_n:.0%}', delta=f'{delta_msc_c:.0%}', delta_color='inverse')
    colc.metric("Écart BP26 Mensuel", value=fmt_euro(bp_global_mensuel), delta=fmt_euro(ecart_bp), delta_description="Reste à faire global")
    cold.metric("Écart BP26 Hebdo", value=fmt_euro(bp_hebdo), delta=fmt_euro(ecart_bp_hebdo), delta_description="Rythme global requis")

    cola.metric('Nombre de couvert', value=fmt_qty(nb_cvts_année_n), delta=fmt_qty(nb_cvts_année_n - nb_cvts_n_1_ytd), delta_description="VS N-1 à la sem. ISO")
    colb.metric('Ticket moyen', value=fmt_euro_2d(ticket_moyen_n), delta=fmt_euro_2d(delta_ticket_moyen), delta_description="VS N-1 à la sem. ISO")

    st.write("")
    st.write("**🎯 Suivi de l'avancement des objectifs globaux (Reste à faire)**")
    col_mois, col_sem = st.columns(2)
    with col_mois:
        if ecart_bp < 0:
            st.warning(f"🔴 Il reste **{fmt_euro(abs(ecart_bp))}** à réaliser pour atteindre l'objectif mensuel")
        else:
            st.success(f"🍏 Objectif mensuel global dépassé de **{fmt_euro(ecart_bp)}** !")
    with col_sem:
        if ecart_bp_hebdo < 0:
            st.warning(f"🔴 Il reste **{fmt_euro(abs(ecart_bp_hebdo))}** à réaliser pour atteindre l'objectif hebdomadaire")
        else:
            st.success(f"🍏 Objectif hebdomadaire dépassé de **{fmt_euro(ecart_bp_hebdo)}** !")

    st.subheader('Évolution du Chiffre d\'affaire YoY', text_alignment='center', divider='blue')
    cols = st.columns(3)
    with cols[0]: mode = st.segmented_control('**Mode de vue**', options=['mois', 'iso_semaine'], default='mois')
    with cols[1]: year = st.pills('**Choisir l\'année**', options=df_ventes['année'].unique(), default=[2025, 2026], selection_mode='multi')
    with cols[2]: site = st.pills('**Point de vente**', options=df_ventes['Site'].unique(), default='Guinguette' , selection_mode='multi')

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
        st.plotly_chart(fig_total, use_container_width=True)

    st.subheader(f'Events : {année_n}', text_alignment='center', divider='blue')
    df_events['année'] = df_events['Date'].dt.year
    var = df_events.query('année == @année_n').groupby('Site')['Ca_ht'].sum().round().reset_index().sort_values('Ca_ht', ascending=False)
    ca_events = var['Ca_ht'].sum()
    ca_events_privat = df_events.query("année == @année_n and Ca_ht > 4000")['Ca_ht'].sum()
    ca_event_exploitation = df_events.query("année == @année_n and Ca_ht < 4000")['Ca_ht'].sum()
    pct_privatision = 1 - (ca_events_privat / ca_events) if ca_events else 0
    pct_exploitation = 1 - (ca_event_exploitation / ca_events) if ca_events else 0
    cvt_event = df_events.query('année == @année_n')['Nb_de_cvts'].sum()
    
    div_p = df_events.query("année == @année_n and Ca_ht > 6000")['Nb_de_cvts'].sum()
    event_tckmean_privat = ca_events_privat / div_p if div_p else 0
    div_e = df_events.query("année == @année_n and Ca_ht < 6000")['Nb_de_cvts'].sum()
    event_tckmean_exploit = ca_event_exploitation / div_e if div_e else 0

    col_1, col_2, col_3 = st.columns(3)
    col_1.metric("**Chiffre d'affaire HT**", value=f'{ca_events:,.0f} €'.replace(","," "), delta='100 %')
    col_1.metric('**Nombre de clients**', value=f'{cvt_event:,.0f}'.replace(",", " "), delta=f'{cvt_event/nb_cvts_année_n:.0%}' if nb_cvts_année_n else '0%', delta_arrow='off', delta_color='blue')
    col_2.metric("**Chiffre d'affaire - Privatisation**", value=f'{ca_events_privat:,.0f} €'.replace(",", " "), delta=f'{pct_exploitation:.0%}', delta_arrow='off')
    col_2.metric('**Ticket moyen - Privatisation**', value=f'{event_tckmean_privat:,.0f} €'.replace(","," "), delta=f'{div_p:,.0f} cvts'.replace(',', ' '), delta_arrow='off', delta_color='gray')
    col_3.metric("**Chiffre d'affaire - Hors Privatisation**", value=f'{ca_event_exploitation:,.0f} €'.replace(",", " "), delta=f'{pct_privatision:.0%}', delta_arrow='off')
    col_3.metric('**Ticket moyen - Hors Privatisation**', value=f'{event_tckmean_exploit:,.0f} €'.replace(',',' '), delta=f'{div_e:,.0f} cvts'.replace(',', ' '), delta_arrow='off', delta_color='gray')

    st.subheader('**Répartition du chiffre d\'affaire Events - Point de vente**', text_alignment='center', divider='blue')
    col_a, col_b = st.columns(2)
    with col_a:
        fig_events = px.bar(var, x='Site', y='Ca_ht', template='simple_white', labels={'Ca_ht': "<b>Chiffre d'affaire HT</b>"})
        fig_events.update_traces(textposition='outside', texttemplate='%{value:.3s} €')
        st.plotly_chart(fig_events, use_container_width=True)
    with col_b:
        fig_events_b = px.scatter(df_events.query('année == @année_n'), x='Date', y='Ca_ht', color='Site', size='Nb_de_cvts', template='simple_white')
        st.plotly_chart(fig_events_b, use_container_width=True)

    st.subheader('**Base de données - Évenement**', text_alignment='center', divider='blue')
    with st.expander(' Cliquer pour afficher la base de donnée') :
        st.dataframe(df_events, hide_index=True)

# ==========================================
# 🏪 TAB 2 : VUE PAR SITE (BP EN DUR CORRIGÉ)
# ==========================================
with tab2:
    # On harmonise les noms de l'en-tête pour être cohérent avec ton sélecteur existant
    pv = ["Restaurant", "Guinguette", "Le petit baigneur"]
    st.header('Quels sites ?', text_alignment='center')
    aa, ab, ac = st.columns(3)
    with ab:
        site_selectionne = st.pills('', options=pv, default="Guinguette", width=500)

    année_n = df_ventes['année'].max()
    année_n_1 = année_n - 1
    
    # Sécurité pour le nom de filtrage du DF ('LPB' ou 'Le petit baigneur')
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

    # Masse Salariale
    df_rh['année'] = df_rh['Date'].dt.year
    ms_c_montant = df_rh.query('année == @année_n & Site == @nom_filtre_df')['Montant'].sum()
    ms_c = ms_c_montant / ca_site_n if ca_site_n else 0
    delta_msc = 0.35 - ms_c

    # Couverts & Ticket Moyen
    nb_cvt_n = df_site_n['Nb_de_cvts'].sum()
    nb_cvt_n_1_ytd = df_site_n_1_ytd['Nb_de_cvts'].sum()
    delta_cvt = nb_cvt_n - nb_cvt_n_1_ytd

    ticket_moyen_n = ca_site_n / nb_cvt_n if nb_cvt_n else 0
    ticket_moyen_n_1_ytd = ca_site_n_1_ytd / nb_cvt_n_1_ytd if nb_cvt_n_1_ytd else 0
    delta_ticket_moyen = ticket_moyen_n - ticket_moyen_n_1_ytd

    # Food & Bev
    food_ca = df_site_n['Food_ht'].sum()
    food_cogs = food_ca / ca_site_n if ca_site_n else 0
    bev_ca = df_site_n['Bev_ht'].sum()
    bev_cogs = bev_ca / ca_site_n if ca_site_n else 0

    # --- 🎯 EXTRACTION DE LA PERF DEPUIS LES DONNÉES DU DEVIS EN DUR ---
    bp_site_mensuel = DATA_BP26.get(current_mois, {}).get(site_selectionne, 0)

    bp_site_hebdo = bp_site_mensuel / 4
    ecart_bp_site_mensuel = ca_site_n - bp_site_mensuel
    ecart_bp_site_hebdo = (ca_site_n / (dernier_semaine_n_site if dernier_semaine_n_site else 1)) - bp_site_hebdo

    "---"
    st.header(f'KPI : {site_selectionne}', text_alignment='center')

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("**Chiffre d'affaire HT**", fmt_euro(ca_site_n), delta=fmt_euro(delta_ca_site), delta_description='**vs N-1 à la sem. ISO**')
    col2.metric('**Masse salariale / chargée**', f'{ms_c:.1%}', delta=f'{delta_msc:.1%}', delta_color='normal')
    col3.metric("Écart BP26 Site Mensuel", value=fmt_euro(bp_site_mensuel), delta=fmt_euro(ecart_bp_site_mensuel), delta_description="Reste à faire sur ce site")
    col4.metric("Écart BP26 Site Hebdo", value=fmt_euro(bp_site_hebdo), delta=fmt_euro(ecart_bp_site_hebdo), delta_description="Rythme hebdomadaire requis")

    a, b, c, d = st.columns(4)
    a.metric('**Nb de couverts**', fmt_qty(nb_cvt_n), delta=fmt_qty(delta_cvt), delta_description='**vs N-1 à la sem. ISO**')
    b.metric('**Ticket moyen**', fmt_euro_2d(ticket_moyen_n), delta=fmt_euro_2d(delta_ticket_moyen), delta_description='**vs N-1 à la sem. ISO**')
    c.metric("**CA Food TTC**", fmt_euro(food_ca), delta=f'{food_cogs:.0%}', delta_arrow='off', delta_color='off', delta_description='**Du CA du site**')
    d.metric("**CA Bev TTC**", fmt_euro(bev_ca), delta=f'{bev_cogs:.0%}', delta_arrow='off', delta_color='off', delta_description='**Du CA du site**')

    # Bloc d'avancement par site
    st.write("")
    st.write(f"**🎯 Suivi de la performance de l'objectif de : {site_selectionne}**")
    col_sm_site, col_sh_site = st.columns(2)
    with col_sm_site:
        if ecart_bp_site_mensuel < 0:
            st.warning(f"🔴 Il reste **{fmt_euro(abs(ecart_bp_site_mensuel))}** de CA HT pour atteindre l'objectif mensuel")
        else:
            st.success(f"🍏 Objectif mensuel dépassé de **{fmt_euro(ecart_bp_site_mensuel)}** WELL DONE !")
    with col_sh_site:
        if ecart_bp_site_hebdo < 0:
            st.warning(f"🔴 Il reste **{fmt_euro(abs(ecart_bp_site_hebdo))}** pour atteindre l'objectif hebdomadaire")
        else:
            st.success(f"🍏 En avance de **{fmt_euro(ecart_bp_site_hebdo)}** au prévisionnel mensuel!")

    df_site_global = df_ventes.query("Site == @nom_filtre_df").copy()
    var_pv = df_site_global.query("année in [@année_n, @année_n_1]").groupby(['année', 'mois'])['Ca_ht'].sum().reset_index()
    var_pv['année'] = var_pv['année'].astype(str)

    st.write(f'Evolution du CA mensuel comparaison YoY : {site_selectionne}')
    fig_pv = px.bar(var_pv, x='mois', y='Ca_ht', template='simple_white', color='année', barmode='group', labels={"Ca_ht": "<b>Chiffre d'affaire HT (€)</b>", 'mois': '<b>Numéro de mois</b>'})
    fig_pv.update_traces(texttemplate='<b>%{value:.3s}€</b>', textposition='outside')
    st.plotly_chart(fig_pv, use_container_width=True)

    "---"
    #### ----- HISTORIQUE DE CORRÉLATION METEO ----- ####
    st.subheader("Corrélation Température vs Chiffre d'affaire", text_alignment='center')
    col_temp_1, col_temp_2 = st.columns(2)
    with col_temp_1:
        fig_temp = px.scatter(df_site_global, x='Temp_Max', y='Ca_ht', size='Nb_de_cvts', color='Site', trendline='ols', template='simple_white', labels={'Temp_Max': '<b>Température °C</b>', "Ca_ht": "<b>Chiffre d'affaire (€)</b>"})
        st.plotly_chart(fig_temp, use_container_width=True)

    with col_temp_2:
        group_temp = df_site_global.copy()
        bins_temp = [-float('inf'), 15, 25, 30, float('inf')]
        labels_temp = ['0-15°C', '16-25°C', '26-30°C', '+ 31°C']
        group_temp['tranche_temp'] = pd.cut(group_temp['Temp_Max'], bins=bins_temp, labels=labels_temp, right=True)
        temp = group_temp.groupby(['Site', 'tranche_temp'], observed=False)['Ca_ht'].sum().reset_index()
        fig_temp1 = px.bar(temp, x='tranche_temp', y='Ca_ht', color='Site', text_auto='.2s', template='simple_white')
        fig_temp1.update_traces(textposition='outside', texttemplate='<b>%{value:.3s}€</b>')
        st.plotly_chart(fig_temp1, use_container_width=True)

    mean_temp = group_temp.groupby(['Site', 'tranche_temp'], observed=False)['Ca_ht'].mean().reset_index().round()
    mean_temp.columns = ['Site', 'Catégorie temp', 'Chiffre d\'affaire moyen']
    st.subheader('**Chiffre d\'affaire moyen - Température**', text_alignment='center')
    st.dataframe(mean_temp, hide_index=True)

    "---"
    st.subheader("Corrélation Pluie vs Chiffre d'affaire", text_alignment='center')
    col_pluie_1, col_pluie_2 = st.columns(2)
    with col_pluie_1:
        fig_pluie = px.scatter(df_site_global, x='Pluie_mm', y='Ca_ht', size='Nb_de_cvts', color='Site', trendline='ols', template='simple_white')
        st.plotly_chart(fig_pluie, use_container_width=True)

    with col_pluie_2:
        group_pluie = df_site_global.copy()
        bins_pluie = [-float('inf'), 10, 20, 30, float('inf')]
        labels_pluie = ['0-10 mm', '11-20 mm', '21-30 mm', '+ 31 mm']
        group_pluie['tranche_pluie'] = pd.cut(group_pluie['Pluie_mm'], bins=bins_pluie, labels=labels_pluie, right=True)
        pluie = group_pluie.groupby(['Site', 'tranche_pluie'], observed=False)['Ca_ht'].sum().reset_index()
        fig_pluie1 = px.bar(pluie, x='tranche_pluie', y='Ca_ht', color='Site', text_auto='.2s', template='simple_white')
        fig_pluie1.update_traces(textposition='outside', texttemplate='<b>%{value:.3s}€</b>')
        st.plotly_chart(fig_pluie1, use_container_width=True)

    mean_pluie = group_pluie.groupby(['Site', 'tranche_pluie'], observed=False)['Ca_ht'].mean().reset_index().round()
    mean_pluie.columns = ['Site', 'Catégorie pluie', 'Chiffre d\'affaire moyen']
    st.subheader('**Chiffre d\'affaire moyen - Pluie**', text_alignment='center')
    st.dataframe(mean_pluie, hide_index=True)

# ==========================================
# 📦 TAB 3 : COMPTE FOURNISSEUR
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

    st.header('Indicateur compte fournisseur', divider='blue')
    total_bl = compte_fournisseur['Montant BL HT'].sum()
    total_facture = compte_fournisseur['Montant Facture HT'].sum()
    solde_total = total_bl - total_facture

    c1, c2, c3 = st.columns(3)
    c1.metric('Total BL HT', f'{total_bl:,.0f} €'.replace(',', ' '))
    c2.metric('Total Facture HT', f'{total_facture:,.0f} €'.replace(',', ' '))
    c3.metric('SOLDE GLOBAL', f'{solde_total:,.0f} €'.replace(',', ' '))

    "---"
    st.subheader('Synthèse compte fournisseur (Poids du solde)')
    if not compte_fournisseur.empty and compte_fournisseur['Taille_Treemap'].sum() > 0.5:
        fig_fournisseur = px.treemap(compte_fournisseur, path=[px.Constant("Tous les fournisseurs"), 'Fournisseur'], values='Taille_Treemap', color='Solde HT', custom_data=['Fournisseur', 'Solde HT', 'Montant BL HT', 'Montant Facture HT'], color_continuous_scale='RdBu', color_continuous_midpoint=0)
        fig_fournisseur.update_traces(texttemplate="<b>%{label}</b><br>%{customdata[1]:,.0f} €", textposition="middle center", hovertemplate="<b>%{customdata[0]}</b><br>Solde : %{customdata[1]:,.0f} €")
        st.plotly_chart(fig_fournisseur, use_container_width=True)
    else:
        st.info("Aucun écart de solde à afficher.")

    "---"
    st.header('Détail par compte', divider='blue')
    cols = st.columns(2)
    with cols[0]: sel_fournisseur = st.selectbox('**Quels fournisseurs ?**', options=sorted(compte_fournisseur['Fournisseur'].unique()))
    with cols[1]:
        solde_compte = compte_fournisseur.query('Fournisseur == @sel_fournisseur')['Solde HT'].sum()
        st.metric('**Solde**', value=f'{solde_compte:,.0f} €')
    
    df_res = compte_fournisseur.query('Fournisseur == @sel_fournisseur').drop(columns='Taille_Treemap')
    st.dataframe(df_res, hide_index=True)   

# ==========================================
# 💰 TAB 4 : SUIVI DU CASH
# ==========================================
with tab4:
    recette = df_ventes.query('année == @année_n')['Espece'].sum()
    df_cash["mois"] = df_cash['Date'].dt.month
    depot = df_cash.query('mois > 4')['Montant'].sum()
    fond_caisse = df_cash.query('mois < 4')['Montant'].sum()
    df_cash_visuel = df_cash.query('mois > 4').copy()
    df_cash_visuel['Date dépôt'] = df_cash_visuel['Date'].dt.date
    df_cash_visuel = df_cash_visuel[['Date dépôt', 'Montant','Numero_ticket']]

    st.header('**Suivi espèces**', divider='blue')
    cols = st.columns(4)
    with cols[0]: st.metric('**Chiffre d\'affaire TTC - Espèces**', value=f'{recette:,.0f} €'.replace(',', ' '))
    with cols[1]: st.metric('**Espèces déposer**', value=f'{depot:,.0f} €'.replace(',', ' '))
    with cols[2]: st.metric('**Solde du coffre**', value=f'{(recette-depot):,.0f} €'.replace(',', ' '))
    with cols[3]:
        st.write('**Historique dépôt**')
        st.dataframe(df_cash_visuel, hide_index=True)

    st.subheader('**Audit des envellopes**', divider='blue')
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

    with st.expander(" Historique des 10 derniers jours (Audit Cash)", expanded=False):
        st.dataframe(df_historique_cash, hide_index=True, use_container_width=True)

# ==========================================
# 👥 TAB 5 : MASSE SALARIALE
# ==========================================
with tab5:
    st.header('Masse salariale', divider='blue')
    df_rh_analyse = df_rh.copy()
    df_ventes_rh = df_ventes.query("année == @année_n").copy()

    date_rh = df_rh_analyse['Date'].dt
    df_rh_analyse = df_rh_analyse.assign(année = date_rh.year, mois = date_rh.month, iso_semaine = date_rh.isocalendar().week)
    df_rh_analyse_année_n = df_rh_analyse.query('année == @année_n')
    
    rh_synthese = df_rh_analyse_année_n.groupby(['iso_semaine','Site'])['Montant'].sum().reset_index()
    rh_ventes_synthese = df_ventes_rh.groupby(['iso_semaine', 'Site'])['Ca_ht'].sum().reset_index()
    globale_rh = pd.merge(rh_ventes_synthese, rh_synthese, how='outer', on=['iso_semaine','Site'])
    globale_rh['Ratio (%)'] = ((globale_rh['Montant'] / globale_rh['Ca_ht'] ) * 100 ).round(2)
    globale_rh['Valeur cible'] = 35
    globale_rh.columns = ['Semaine ISO', 'Site', "Chiffre d'affaire HT", "Masse salariale chargée", 'Ratio (%)', "Valeur cible"]

    st.subheader('**Sélectionner le point de vente**')
    site_rh = st.pills('', options=globale_rh['Site'].unique(), default='Restaurant')
    var_rh = globale_rh.query('Site == @site_rh').groupby(['Semaine ISO', 'Site']).agg({"Chiffre d'affaire HT" : 'sum', "Masse salariale chargée" : 'sum', "Ratio (%)" : 'mean', "Valeur cible" : 'mean'}).round().reset_index()

    ca_rh = var_rh["Chiffre d'affaire HT"].sum()
    msc_rh = var_rh["Masse salariale chargée"].sum()
    ratio_rh = msc_rh / ca_rh if ca_rh else 0
    delta_rh = 0.35 - ratio_rh
    ecart_rh_val = ca_rh * delta_rh * -1

    cols = st.columns(3)
    cols[0].metric("Chiffre d'affaire HT", value=f'{ca_rh:,.0f} €'.replace(",", " "), delta=f'{ca_rh / ca_année_n :.0%}' if ca_année_n else '0%')
    cols[1].metric("Masse salariale chargée", value=f'{msc_rh:,.0f} €'.replace(",", " "), delta=f'{ecart_rh_val:,.0f} €'.replace(",", " "), delta_color='inverse')
    cols[2].metric("Ratio MS/C", value=f'{ratio_rh:.2%}', delta=f'{delta_rh:.2%}', delta_arrow='off')

    fig_rh = px.bar(var_rh, x='Semaine ISO', y='Ratio (%)', color='Site', template='plotly_white')
    fig_rh.add_scatter(x=var_rh['Semaine ISO'], y=var_rh['Valeur cible'], name='Objectif 35%')
    st.plotly_chart(fig_rh, use_container_width=True)

# ==========================================
# 🗄️ TAB 9 : ARCHIVES
# ==========================================
with tab9:
    with st.expander('**Historique Master Data**'):
        df_ventes['Date'] = df_ventes['Date'].dt.date
        st.dataframe(df_ventes, hide_index=True, use_container_width=True)

# ==========================================
# 📝 TAB : SAISIE DE DONNÉES (BL COMPLÈTÉ + EVENTS)
# ==========================================
with tab_saisie:
    st.header("📝 Centre d'enregistrement", divider='red')
    st.write("Ajoutez de nouvelles entrées directement dans votre base de données Google Sheets.")
    
    form_selection = st.radio("**Choisissez le formulaire à remplir :**", ["Bon de livraison", "Événements (Events)"], horizontal=True)
    st.write("---")
    
    liste_sites_form = ["Restaurant", "Guinguette", "LPB"]
    
    if form_selection == "Bon de livraison":
        st.subheader("📦 Nouveau Bon de Livraison")
        with st.form("form_bl", clear_on_submit=True):
            col_bl1, col_bl2 = st.columns(2)
            with col_bl1:
                bl_date = st.date_input("Date du BL", datetime.date.today())
                bl_site = st.selectbox("Site concerné", options=["Tous les sites"] + liste_sites_form)
                bl_fournisseur = st.text_input("Nom du Fournisseur (ex: Metro, Pomona, David boissons...)")
                bl_num = st.text_input("Numéro de Bon de Livraison (Numero_bl)")
            with col_bl2:
                bl_categorie = st.selectbox("Catégorie", options=["Food", "Bev", "Autres"])
                bl_montant_ht = st.number_input("Montant HT (€)", min_value=0.0, step=10.0, format="%.2f")
                bl_taxes = st.number_input("Taxes / TVA (€)", min_value=0.0, step=2.0, format="%.2f")
                bl_montant_ttc = st.number_input("Montant TTC (€)", min_value=0.0, step=10.0, format="%.2f")
            
            bl_note = st.text_area("Note pour le suivi / Commentaires", placeholder="Exemple : Écart de prix constaté, produit manquant remplacé...")
                
            submit_bl = st.form_submit_button("💾 Enregistrer le Bon de Livraison")
            
            if submit_bl:
                if bl_fournisseur.strip() == "":
                    st.error("❌ Le nom du fournisseur est indispensable.")
                else:
                    date_str = bl_date.strftime("%d/%m/%Y")
                    # Ligne complète alignée avec ton onglet 'Bon_livraison'
                    nouvelle_ligne_bl = [
                        date_str, bl_site, bl_num, bl_fournisseur, bl_categorie,
                        bl_montant_ht, bl_taxes, bl_montant_ttc, "False", "", bl_note, "A Traiter"
                    ]
                    
                    with st.spinner("Mise à jour du fichier Google Sheet..."):
                        succes = append_to_sheet("Bon_livraison", nouvelle_ligne_bl)
                    if succes:
                        st.success(f"🎉 Bon de livraison {bl_fournisseur} enregistré ({bl_montant_ht} € HT) !")
                        st.cache_data.clear()
                        
    elif form_selection == "Événements (Events)":
        st.subheader("🎉 Nouvel Événement (Events)")
        with st.form("form_event", clear_on_submit=True):
            col_ev1, col_ev2 = st.columns(2)
            with col_ev1:
                ev_date = st.date_input("Date de l'événement", datetime.date.today())
                ev_site = st.selectbox("Point de vente associé", options=liste_sites_form)
                ev_ca = st.number_input("Chiffre d'affaires HT (€)", min_value=0.0, step=50.0, format="%.2f")
            with col_ev2:
                ev_cvts = st.number_input("Nombre de couverts / clients", min_value=0, step=1)
                ev_comm = st.text_input("Nom / Commentaire de l'événement")
                
            submit_event = st.form_submit_button("💾 Enregistrer l'Événement")
            
            if submit_event:
                date_str = ev_date.strftime("%d/%m/%Y")
                nouvelle_ligne_ev = [date_str, ev_site, "", "", "", "", "", "", "", ev_ca, ev_cvts, "", "", "", "", "", ev_comm, ""]
                
                with st.spinner("Envoi vers Google Sheets..."):
                    succes = append_to_sheet("Events", nouvelle_ligne_ev)
                if succes:
                    st.success(f"🎉 Événement '{ev_comm}' enregistré avec succès !")
                    st.cache_data.clear()
