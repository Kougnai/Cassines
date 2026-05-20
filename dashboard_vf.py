import gspread
from google.oauth2.service_account import Credentials
import pandas as pd
import streamlit as st
import plotly.express as px
import requests
from prophet import Prophet

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
        justify-content: center; /* <--- AJOUTÉ : Centre la liste des onglets */
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
    # 1. Définition des accès
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    
    # 2. Récupération sécurisée des secrets
    # On transforme en dict pour pouvoir manipuler la private_key
    creds_info = dict(st.secrets["gcp_service_account"])
    
    # 3. NETTOYAGE DE LA CLÉ (Correction de l'erreur base64/sauts de ligne)
    if "private_key" in creds_info:
        creds_info["private_key"] = creds_info["private_key"].replace("\\n", "\n")
    
    # 4. Authentification avec la méthode moderne
    creds = Credentials.from_service_account_info(creds_info, scopes=scope)
    
    # 5. Connexion à gspread
    client = gspread.authorize(creds)
    
    # 6. Ouverture du Spreadsheet
    spreadsheet = client.open("Cassines_bdd")
    
    onglets = ['Ventes', 'Caisse', 'Events', 'Rh', 'Cash','Bon_livraison', 'Facture', 'Stock', 'Enveloppe'] # Selectionner les onglets
    
    # 7. Extraction des données
    data = {nom: pd.DataFrame(spreadsheet.worksheet(nom).get_all_records(value_render_option='FORMATTED_VALUE')) for nom in onglets}
    
    return data


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
df_ventes, df_caisse, df_events, df_rh, df_cash, df_bl, df_facture, df_stock, df_enveloppe = dfs['Ventes'],dfs['Caisse'], dfs['Events'], dfs['Rh'], dfs['Cash'], dfs['Bon_livraison'], dfs['Facture'], dfs['Stock'], dfs['Enveloppe']

## ---- NETTOYAGE ET CONVERSION ---
onglets_list = [df_cash, df_caisse, df_events, df_rh, df_ventes, df_bl, df_facture, df_stock, df_enveloppe]
col_num = ['Ca_ttc', 'Taxes_20', 'Taxes_10', 'Taxes_5.5','Ca_ht','Cb','Espece', 
            'Cheque', 'Autres_ht', 'Privatisation_ht', 'Food_ht', 'Bev_ht',
             'Nb_de_cvts', 'Autres', 'Tips', 'Autre_ht', 'Montant', 'Montant_ht','Quantité', 'Prix d\'achat', 'Total']

## --- MISE EN FORME DES COLONNES 
for df in onglets_list:
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
        df['Date'] = df['Date'].dt.normalize()
    
    cols_presentes = [c for c in col_num if c in df.columns]
    for col in cols_presentes:
        # Nettoyage manuel des espaces avant le replace
        df[col] = df[col].astype(str).replace(r'\s+', '', regex=True).str.replace(',', '.')
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

##### ---- AJOUT DES DONNÉES MÉTÉO ----
df_ventes = add_weather_data(df_ventes)

## ----- CRÉATION DES TABLES -----
tab1, tab2, tab3, tab4, tab5, tab9 = st.tabs(["🌍 Vue globale", "📍 Vue par site", "📄 Compte Fournisseur", "💶 Cash",'👨‍🍳 Masse salariale','📚 Archives'] )

with tab1: ## VUE GLOBALE 

    ### Analyse globale  KPI

    # --- CALCULS KPIs ---
    # En une seule passe pour optimiser les performances
    df_ventes = df_ventes.assign(
        année=df_ventes['Date'].dt.year,
        mois=df_ventes['Date'].dt.month,
        iso_semaine=df_ventes['Date'].dt.isocalendar().week,
        ticket_moyen=df_ventes['Ca_ht'] / df_ventes['Nb_de_cvts'],
        jour_année=df_ventes['Date'].dt.day_of_year,
        jour_semaine=df_ventes['Date'].dt.day_of_week,
    )

    # --- DATES & FILTRES (MODIFIÉ POUR SEMAINE ISO) ---
    année_n = df_ventes['année'].max()
    année_n_1 = année_n - 1

    # Trouver la dernière semaine ISO enregistrée en année N
    dernier_semaine_n = df_ventes.query('année == @année_n')['iso_semaine'].max()

    # Données Année N
    df_année_n = df_ventes.query('année == @année_n').copy()

    # Données Année N-1 filtrées à la SEMAINE ISO
    df_année_n_1_ytd = df_ventes.query('année == @année_n_1 & iso_semaine <= @dernier_semaine_n')

    # --- CALCULS DES METRICS ---
    # Chiffre d'affaires
    ca_année_n = df_année_n['Ca_ht'].sum()
    ca_année_n_1_ytd = df_année_n_1_ytd['Ca_ht'].sum()
    delta_ca = ca_année_n - ca_année_n_1_ytd

    # Couverts & Ticket Moyen
    nb_cvts_année_n = df_année_n['Nb_de_cvts'].sum()
    nb_cvts_n_1_ytd = df_année_n_1_ytd['Nb_de_cvts'].sum()

    ticket_moyen_n = ca_année_n / nb_cvts_année_n if nb_cvts_année_n else 0
    ticket_moyen_n_1_ytd = ca_année_n_1_ytd / nb_cvts_n_1_ytd if nb_cvts_n_1_ytd else 0
    delta_ticket_moyen = ticket_moyen_n - ticket_moyen_n_1_ytd

    # RH & COGS
    df_rh['année'] = df_rh['Date'].dt.year 
    ms_c_année_n = df_rh.query("année == @année_n")['Montant'].sum() / ca_année_n if ca_année_n else 0
    delta_msc_c = ms_c_année_n - 0.35

    food_ca_année_n = df_année_n['Food_ht'].sum()
    food_cogs = food_ca_année_n / ca_année_n if ca_année_n else 0

    bev_ca_année_n = df_année_n['Bev_ht'].sum()
    bev_cogs = bev_ca_année_n / ca_année_n if ca_année_n else 0

    # --- AFFICHAGE STREAMLIT ---
    st.subheader(f'KPI : {année_n} Semaine : N° {dernier_semaine_n}', divider='blue')

    # Fonctions de formatage locales rapides
    fmt_euro = lambda x: f"{x:,.0f} €".replace(",", " ")
    fmt_euro_2d = lambda x: f"{x:,.2f} €".replace(",", " ")
    fmt_qty = lambda x: f"{x:,.0f}".replace(",", " ")

    cola, colb, colc, cold = st.columns(4)

    # Ligne 1
    cola.metric("Chiffre d'affaire HT", value=fmt_euro(ca_année_n), delta=fmt_euro(delta_ca), delta_description="VS N-1 à la sem. ISO")
    colb.metric('MS/C', value=f'{ms_c_année_n:.0%}', delta=f'{delta_msc_c:.0%}', delta_color='inverse')
    colc.metric('Food HT', value=fmt_euro(food_ca_année_n), delta=f'{food_cogs:.0%}', delta_color='off', delta_arrow='off', delta_description="du CA")
    cold.metric('Bev HT', value=fmt_euro(bev_ca_année_n), delta=f'{bev_cogs:.0%}', delta_color='off', delta_arrow='off', delta_description="du CA")

    # Ligne 2
    cola.metric('Nombre de couvert', value=fmt_qty(nb_cvts_année_n), delta=fmt_qty(nb_cvts_année_n - nb_cvts_n_1_ytd), delta_description="VS N-1 à la sem. ISO")
    colb.metric('Ticket moyen', value=fmt_euro_2d(ticket_moyen_n), delta=fmt_euro_2d(delta_ticket_moyen), delta_description="VS N-1 à la sem. ISO")
    colc.metric('Food COGS', value='32%', delta='4%', delta_arrow='up', delta_color='inverse', delta_description='Statique')
    cold.metric('Bev COGS', value='27%', delta='2%', delta_arrow='up', delta_color='inverse', delta_description='Statique')

    ######### EVOLUTION DU CHIFFRE D'AFFAIRE YOY GLOBALE

    st.subheader('Évolution du Chiffre d\'affaire YoY', text_alignment='center', divider='blue')

    cols = st.columns(3)
    with cols[0]:
        mode = st.segmented_control('**Mode de vue**', options=['mois', 'iso_semaine'], default='mois')
    with cols[1]:
        year = st.pills('**Choissir l\'année**', options=df_ventes['année'].unique(), default=[2025, 2026], selection_mode='multi')
    with cols[2]:
        site = st.pills('**Point de vente**', options=df_ventes['Site'].unique(),default='Guinguette' , selection_mode='multi')

    ca_month = df_ventes.query('année == @year and Site == @site').groupby(['année', mode])['Ca_ht'].sum().reset_index().sort_values(['année', mode], ascending=True)

    if not ca_month.empty:
        idx_max = ca_month['Ca_ht'].idxmax()
        ca_max = ca_month.loc[idx_max, 'Ca_ht']
        max_x = ca_month.loc[idx_max, mode]
        max_annee = ca_month.loc[idx_max, 'année']

        # 3. Création du graphique de base
        ca_month_copy = ca_month.copy()
        ca_month_copy['année'] = ca_month_copy['année'].astype(str)
        fig_total = px.line(
            ca_month_copy,
            x=mode,
            y='Ca_ht',
            color='année',
            template='simple_white',
            title=f'Évolution du CA par {mode} : {", ".join(map(str, year))}',
            labels={mode: f'<b>{mode.capitalize()}</b>', 'Ca_ht': '<b>CA HT</b>', 'année': '<b>Année</b>', 'iso_semaine' : '<b>Semaine ISO</b>'}
        )

        # 4. Ajout du point "Record" dynamique
        fig_total.add_scatter(
            x=[max_x],
            y=[ca_max],
            mode='markers+text',
            name='Record Historique',
            text=[f" Record : {ca_max:,.0f} €".replace(',', ' ')],
            textposition="top center",
            marker=dict(color='gold', size=12, symbol='star'),
            showlegend=False
        )

        fig_total.update_layout(hovermode='x unified')
        st.plotly_chart(fig_total, use_container_width=True)

    ####### PARTIE EVENEMENT POUR L'ANNÉE EN COURS 

    st.subheader(f'Events : {année_n}', text_alignment='center', divider='blue')
    st.write("")
    
    ## ---- IMPORTATION  ET MISE EN FORME DU DF -----
    df_events['année'] = df_events['Date'].dt.year

    ## --- PRÉPARATION DES DES VARIABLES ET DES DONNÉES -----
    var = df_events.query('année == @année_n').groupby('Site')['Ca_ht'].sum().round().reset_index().sort_values('Ca_ht', ascending=False)
    ca_events = var['Ca_ht'].sum()
    ca_events_privat = df_events.query("année == @année_n and Ca_ht > 6000")['Ca_ht'].sum()
    ca_event_exploitation = df_events.query("année == @année_n and Ca_ht < 6000")['Ca_ht'].sum()
    pct_privatision =  1 - (ca_events_privat /ca_events)
    pct_exploitation = 1 - (ca_event_exploitation / ca_events)
    cvt_event = df_events.query('année == @année_n')['Nb_de_cvts'].sum()
    event_tckmean_privat = ca_events_privat / df_events.query("année == @année_n and Ca_ht > 6000")['Nb_de_cvts'].sum() 
    cvt_event_privat = df_events.query("année == @année_n and Ca_ht > 6000")['Nb_de_cvts'].sum() 
    event_tckmean_exploit = ca_event_exploitation / df_events.query("année == @année_n and Ca_ht < 6000")['Nb_de_cvts'].sum()
    cvt_event_exploit = df_events.query("année == @année_n and Ca_ht < 6000")['Nb_de_cvts'].sum()

    ## --- AFFICHAGE DES METRICS SUR STREAMLIT ----- 
    col_1, col_2, col_3 = st.columns(3)
    col_1.metric("**Chiffre d'affaire HT**", value=f'{ca_events:,.0f} €'.replace(","," "), delta='100 %')
    col_1.metric('**Nombre de clients**', value=f'{cvt_event:,.0f}'.replace(",", " "), delta=f'{cvt_event/nb_cvts_année_n:.0%}', delta_arrow='off', delta_color='blue', delta_description='Total de couverts')
    col_2.metric("**Chiffre d'affaire - Privatisation**", value=f'{ca_events_privat:,.0f} €'.replace(",", " "), delta=f'{pct_exploitation:.0%}', delta_arrow='off')
    col_2.metric('**Ticket moyen - Privatisation**', value=f'{event_tckmean_privat:,.0f} €'.replace(","," "), delta=f'{cvt_event_privat:,.0f} cvts'.replace(',', ' '), delta_arrow='off', delta_color='gray')
    col_3.metric("**Chiffre d'affaire - Hors Privatisation**", value=f'{ca_event_exploitation:,.0f} €'.replace(",", " "), delta=f'{pct_privatision:.0%}', delta_arrow='off')
    col_3.metric('**Ticket moyen - Hors Privatisation**', value=f'{event_tckmean_exploit:,.0f} €'.replace(',',' '), delta=f'{cvt_event_exploit:,.0f} cvts'.replace(',', ' '), delta_arrow='off', delta_color='gray')

    st.write("")
    ### --- VENTILATION DU CHIFFRE D'AFFAIRE - EVENEMENTS

    st.subheader('**Répartition du chiffre d\'affaire Events - Point de vente**', text_alignment='center', divider='blue')

    ## ---- RÉPARATION PAR POINT DE VENTE ---- 
    col_a, col_b = st.columns(2)
    with col_a:
        fig_events = px.bar(
            var,
            x='Site',
            y='Ca_ht',
            template='simple_white',
            labels={'Ca_ht': "<b>Chiffre d'affaire HT</b>", 'Site':'<b>Point de vente</b>'},
            range_y=[0, df_events['Ca_ht'].sum()]
        )
        fig_events.update_traces(
            textposition = 'outside',
            texttemplate = '%{value:.3s} €'
        )
        st.plotly_chart(fig_events, use_container_width=True)

    ## ---- VUE EN SCATTER DES EVENEMENT PAR POINT DE VENTE ---- 
    with col_b:
        fig_events_b = px.scatter(
            df_events.query('année == @année_n'),
            x='Date',
            y='Ca_ht',
            color='Site',
            size='Nb_de_cvts',
            trendline='lowess',
            template='simple_white',
            labels={'Date' : '<b>Date</b>', "Ca_ht" : '<b>Chiffre d\'affaire HT</b>', 'Site': '<b>Point de vente</b>', 'Nb_de_cvts' : '<b>Nombre de couverts</b>'}
        )
        st.plotly_chart(fig_events_b, use_container_width=True)

    ### --- AFICHAGE DE LA BASE DE DONNÉES DU FICHIER EVENEMENT 
    st.subheader('**Base de données - Évenement**', text_alignment='center', divider='blue')
    with st.expander(' Cliquer pour afficher la base de donnée') :
            st.dataframe(df_events, hide_index=True)
            
with tab2: ## VUE PAR SITE 
        
   # --- FILTRE DYNAMIQUE DES SITES ---
    pv = df_ventes['Site'].unique()

    st.header('Quels sites ?', text_alignment='center')
   
    aa, ab, ac = st.columns(3)
    with ab:
        # Changement du défaut pour être robuste si 'Guinguette' n'est pas présent
        default_site = 'Guinguette' if 'Guinguette' in pv else pv[0]
        site = st.pills('', options=pv, default=default_site, width=500)

    # --- PRÉPARATION DES DONNÉES DYNAMIQUES & À LA SEMAINE ISO ---
    année_n = df_ventes['année'].max()
    année_n_1 = année_n - 1

    # Trouver la dernière semaine ISO enregistrée pour CE SITE en année N
    dernier_semaine_n_site = df_ventes.query('année == @année_n & Site == @site')['iso_semaine'].max()

    # Si le site n'a pas encore de ventes en année N, on évite un plantage
    if pd.isna(dernier_semaine_n_site):
        dernier_semaine_n_site = df_ventes['iso_semaine'].max()

    # Données Année N pour le site
    df_site_n = df_ventes.query('année == @année_n & Site == @site').copy()

    # Données Année N-1 pour le site filtrées à la SEMAINE ISO
    df_site_n_1_ytd = df_ventes.query('année == @année_n_1 & Site == @site & iso_semaine <= @dernier_semaine_n_site')

    # --- CALCULS DES METRICS ---
    # Chiffre d'affaires (Courant vs Semaine ISO)
    ca_site_n = df_site_n['Ca_ht'].sum()
    ca_site_n_1_ytd = df_site_n_1_ytd['Ca_ht'].sum()
    delta_ca_site = ca_site_n - ca_site_n_1_ytd

    # Masse Salariale
    df_rh['année'] = df_rh['Date'].dt.year
    ms_c_montant = df_rh.query('année == @année_n & Site == @site')['Montant'].sum()
    ms_c = ms_c_montant / ca_site_n if ca_site_n else 0
    valeur_cible_msc = 0.35
    # Inversion pour le delta (si ms_c < cible = positif/vert)
    delta_msc = valeur_cible_msc - ms_c

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

    "---"

    # --- AFFICHAGE STREAMLIT ---
    st.header(f'KPI : {site}', text_alignment='center')

    # Fonctions de formatage locales
    fmt_euro = lambda x: f"{x:,.0f} €".replace(",", " ")
    fmt_euro_2d = lambda x: f"{x:,.2f} €".replace(",", " ")
    fmt_qty = lambda x: f"{x:,.0f}".replace(",", " ")

    ## ----- PARTIE 1/2 DES KPI ----
    col1, col2, col3, col4 = st.columns(4)

    # Chiffre d'affaires comparé à la semaine ISO
    col1.metric("**Chiffre d'affaire HT**", fmt_euro(ca_site_n), delta=fmt_euro(delta_ca_site), delta_description='**vs N-1 à la sem. ISO**')
    # Masse salariale
    col2.metric('**Masse salariale / chargée**', f'{ms_c:.1%}', delta=f'{delta_msc:.1%}', delta_color='normal')
    # COGS Statiques (en attendant dynamisation)
    col3.metric('**Food COGS**', value='32%', delta='4%', delta_arrow='up', delta_color='inverse', delta_description='Statique')
    col4.metric('**Bev COGS**', value='27%', delta='2%', delta_arrow='up', delta_color='inverse', delta_description='Statique')

    ## ---- PARTIE 2/2 DES KPI ----
    a, b, c, d = st.columns(4)

    # Nombre de couverts comparé à la semaine ISO
    a.metric('**Nb de couverts**', fmt_qty(nb_cvt_n), delta=fmt_qty(delta_cvt), delta_description='**vs N-1 à la sem. ISO**')
    # Ticket moyen comparé à la semaine ISO
    b.metric('**Ticket moyen**', fmt_euro_2d(ticket_moyen_n), delta=fmt_euro_2d(delta_ticket_moyen), delta_description='**vs N-1 à la sem. ISO**')
    # Répartition CA
    c.metric("**CA Food HT**", fmt_euro(food_ca), delta=f'{food_cogs:.0%}', delta_arrow='off', delta_color='off', delta_description='**Du CA du site**')
    d.metric("**CA Bev HT**", fmt_euro(bev_ca), delta=f'{bev_cogs:.0%}', delta_arrow='off', delta_color='off', delta_description='**Du CA du site**')

   # --- PRÉPARATION DES DONNÉES GLOBALES POUR LES GRAPHES (SANS FILTRE À DATE) ---

    # Extraction de la totalité historique pour le site sélectionné (Année N, N-1, etc.)
    df_site_global = df_ventes.query("Site == @site").copy()

    # Groupement mensuel complet pour le graphique YoY
    var_pv = df_site_global.query("année in [@année_n, @année_n_1]").groupby(['année', 'mois'])['Ca_ht'].sum().reset_index()
    var_pv['année'] = var_pv['année'].astype(str)

    #  ------ GRAPHIQUE DE COMPARAISON CHIFFRE D'AFFAIRE PAR SITE YOY -----
    st.write(f'Evolution du CA mensuel comparaison YoY : {site}')
    fig_pv = px.bar(
        var_pv, x='mois', y='Ca_ht',
        template='simple_white', color='année', barmode='group',
        labels={"Ca_ht": "<b>Chiffre d'affaire HT (€)</b>", 'mois': '<b>Numéro de mois</b>'}
    )
    fig_pv.update_traces(
        texttemplate='<b>%{value:.3s}€</b>', textposition='outside'
    )
    st.plotly_chart(fig_pv, use_container_width=True)

    "---"

    #### ----- TEMPÉRATURE ----- ####
    st.subheader("Corrélation Température vs Chiffre d'affaire", text_alignment='center')

    col_temp_1, col_temp_2 = st.columns(2)

    with col_temp_1:
    # Nuage de points global sans distinction d'année
        fig_temp = px.scatter(
        df_site_global,
        x='Temp_Max',
        y='Ca_ht',
        size='Nb_de_cvts',
        color='Site',
        trendline='ols',
        template='simple_white',
        labels={'Temp_Max': '<b>Température °C</b>', "Ca_ht": "<b>Chiffre d'affaire (€)</b>", "Nb_de_cvts": '<b>Nombre de couverts</b>', 'Site': "<b>Site</b>"}    
        )
        st.plotly_chart(fig_temp, use_container_width=True)

    with col_temp_2:
        ## Graphique en barres par tranche de température global
        group_temp = df_site_global.copy()
        
        bins_temp = [-float('inf'), 15, 25, 30, float('inf')]
        labels_temp = ['0-15°C', '16-25°C', '26-30°C', '+ 31°C']

        group_temp['tranche_temp'] = pd.cut(
            group_temp['Temp_Max'], 
            bins=bins_temp, 
            labels=labels_temp,
            right=True 
            )
        
        temp = group_temp.groupby(['Site', 'tranche_temp'], observed=False)['Ca_ht'].sum().reset_index()

        fig_temp1 = px.bar(
            temp,
            x='tranche_temp',
            y='Ca_ht',
            color='Site',
            text_auto='.2s',
            template='simple_white',
            labels={'Ca_ht': '<b>Chiffre d\'affaire (€)</b>', 'tranche_temp': '<b>Catégorie de température</b>', 'Site': "<b>Site</b>"},
            range_y=[0, temp['Ca_ht'].max() * 1.2 if not temp.empty else 100]
            )
        fig_temp1.update_traces(
            textposition='outside',
            texttemplate='<b>%{value:.3s}€</b>'
            )
        st.plotly_chart(fig_temp1, use_container_width=True)

    # Tableau résumé Température global
    mean_temp = group_temp.groupby(['Site', 'tranche_temp'], observed=False)['Ca_ht'].mean().reset_index().round()
    mean_temp.columns = ['Site', 'Catégorie temp', 'Chiffre d\'affaire moyen']
    st.subheader('**Chiffre d\'affaire moyen - Température**', text_alignment='center')
    st.dataframe(mean_temp, hide_index=True)

    "---"  ### ------ Corrélation PLUIE vs CA ----- ###

    st.subheader("Corrélation Pluie vs Chiffre d'affaire", text_alignment='center')

    col_pluie_1, col_pluie_2 = st.columns(2)

    with col_pluie_1:
        # Nuage de points pluie global
        fig_pluie = px.scatter(
            df_site_global,
            x='Pluie_mm',
            y='Ca_ht',
            size='Nb_de_cvts',
            color='Site',
            trendline='ols',
            template='simple_white',
            labels={'Pluie_mm': '<b>Pluviométrie en mm</b>', "Ca_ht": "<b>Chiffre d'affaire (€)</b>", "Nb_de_cvts": '<b>Nombre de couverts</b>', 'Site': "<b>Site</b>"}    
        )
        st.plotly_chart(fig_pluie, use_container_width=True)

    with col_pluie_2:
        ## --- Graphique bar de Pluie global
        group_pluie = df_site_global.copy()

        bins_pluie = [-float('inf'), 10, 20, 30, float('inf')]
        labels_pluie = ['0-10 mm', '11-20 mm', '21-30 mm', '+ 31 mm']

        group_pluie['tranche_pluie'] = pd.cut(
            group_pluie['Pluie_mm'], 
            bins=bins_pluie, 
            labels=labels_pluie,
            right=True
        )

        pluie = group_pluie.groupby(['Site', 'tranche_pluie'], observed=False)['Ca_ht'].sum().reset_index()

        fig_pluie1 = px.bar(
            pluie,
            x='tranche_pluie',
            y='Ca_ht',
            color='Site',
            text_auto='.2s',
            template='simple_white',
            labels={'Ca_ht': '<b>Chiffre d\'affaire (€)</b>', 'tranche_pluie': '<b>Catégorie de pluie</b>', 'Site': "<b>Site</b>"},
            range_y=[0, pluie['Ca_ht'].max() * 1.3 if not pluie.empty else 100]
       )
        fig_pluie1.update_traces(
            textposition='outside',
            texttemplate='<b>%{value:.3s}€</b>'
        )
        st.plotly_chart(fig_pluie1, use_container_width=True)

        ### ---- Tableau résumé Pluie global
        mean_pluie = group_pluie.groupby(['Site', 'tranche_pluie'], observed=False)['Ca_ht'].mean().reset_index().round()
        mean_pluie.columns = ['Site', 'Catégorie pluie', 'Chiffre d\'affaire moyen']
    st.subheader('**Chiffre d\'affaire moyen - Pluie**', text_alignment='center')
    st.dataframe(mean_pluie, hide_index=True)

with tab3: ## VUE COMPTE FOURNISSEUR 
    # --- Consolidation ---
    df_bl['Mois'] = df_bl['Date'].dt.strftime('%Y-%m')
    df_facture['Mois'] = df_facture['Date'].dt.strftime('%Y-%m')
    bl = df_bl.groupby(['Mois','Fournisseur'])['Montant_ht'].sum().reset_index()
    facture = df_facture.groupby(['Mois','Fournisseur'])['Montant_ht'].sum().reset_index()
    
    # Merge et renommage propre
    compte_fournisseur = pd.merge(bl, facture, how='outer', on=['Mois','Fournisseur']).fillna(0)
    compte_fournisseur.columns = ['Mois','Fournisseur', 'Montant BL HT', 'Montant Facture HT']
    compte_fournisseur['Solde HT'] = compte_fournisseur['Montant BL HT'] - compte_fournisseur['Montant Facture HT']
    
    # Création d'une colonne pour la taille (toujours positive)
    # On ajoute un micro-montant (0.01) pour que même le 0 soit (très peu) visible
    compte_fournisseur['Taille_Treemap'] = compte_fournisseur['Solde HT'].abs() + 0.01

    # --- KPI ---
    st.header('Indicateur compte fournisseur', divider='blue')
    total_bl = compte_fournisseur['Montant BL HT'].sum()
    total_facture = compte_fournisseur['Montant Facture HT'].sum()
    solde_total = total_bl - total_facture

    c1, c2, c3 = st.columns(3)
    c1.metric('Total BL HT', f'{total_bl:,.0f} €'.replace(',', ' '))
    c2.metric('Total Facture HT', f'{total_facture:,.0f} €'.replace(',', ' '))
    c3.metric('SOLDE GLOBAL', f'{solde_total:,.0f} €'.replace(',', ' '))

    "---"
    # --- TREEMAP ---
    st.subheader('Synthèse compte fournisseur (Poids du solde)')

    # On vérifie si on a au moins une ligne à afficher
    if not compte_fournisseur.empty and compte_fournisseur['Taille_Treemap'].sum() > 0.5:
        fig_fournisseur = px.treemap(
            compte_fournisseur,
            path=[px.Constant("Tous les fournisseurs"), 'Fournisseur'],
            values='Taille_Treemap',
            color='Solde HT',
            # On ajoute 'Fournisseur' dans les custom_data pour l'affichage texte
            custom_data=['Fournisseur', 'Solde HT', 'Montant BL HT', 'Montant Facture HT'],
            color_continuous_scale='RdBu',
            color_continuous_midpoint=0
        )

        # MISE À JOUR : Affichage du texte direct + formatage du survol
        fig_fournisseur.update_traces(
            # textinfo définit ce qui est écrit DANS la case
            # 'label' est le nom du fournisseur, 'value' est la taille, mais on veut le vrai solde
            texttemplate="<b>%{label}</b><br>%{customdata[1]:,.0f} €",
            textposition="middle center",
            # Le hovertemplate reste pour le détail complet au survol
            hovertemplate="<b>%{customdata[0]}</b><br>Solde : %{customdata[1]:,.0f} €<br>BL : %{customdata[2]:,.0f} €<br>Factures : %{customdata[3]:,.0f} €"
        )

        # Optionnel : Ajuster la taille de la police pour que ce soit lisible
        fig_fournisseur.update_layout(margin=dict(t=30, l=10, r=10, b=10))
        
        st.plotly_chart(fig_fournisseur, use_container_width=True)
    else:
        st.info("Aucun écart de solde à afficher sur le graphique.")

    "---"
    # --- RECHERCHE ---
    st.header('Détail par compte', divider='blue')
    cols = st.columns(2)
    with cols[0]:
        sel_fournisseur = st.selectbox('**Quels fournisseurs ?**', options=sorted(compte_fournisseur['Fournisseur'].unique()))
    with cols[1]:
        solde_compte = compte_fournisseur.query('Fournisseur == @sel_fournisseur')['Solde HT'].sum()
        st.metric('**Solde**', value=f'{solde_compte:,.0f} €')
    
    df_res = compte_fournisseur.query('Fournisseur == @sel_fournisseur').drop(columns='Taille_Treemap')
    st.dataframe(df_res, hide_index=True)   
with tab4: ## VUE SUIVIT DU CASH
    ##### ------- SUIVIT ESPCES ----- 

    ## --- PRÉPARATION DES VARIABLES ---- 

    ## --- CHIFFRE D'AFFAIRE --- 
    recette = df_ventes.query('année == @année_n')['Espece'].sum()
    df_cash["mois"] = df_cash['Date'].dt.month
    depot = df_cash.query('mois > 4')['Montant'].sum()
    fond_caisse = df_cash.query('mois < 4')['Montant'].sum()
    df_cash_visuel = df_cash.query('mois > 4').copy()
    df_cash_visuel['Date dépôt'] = df_cash_visuel['Date'].dt.date
    df_cash_visuel = df_cash_visuel[['Date dépôt', 'Montant','Numero_ticket']]

    ### ----- INDICATEUR DE CASH CHIFFRE D'AFFAIRE ---- 

    st.header('**Suivi espèces**', divider='blue')

    cols = st.columns(4)
    with cols[0]:
        st.metric('**Chiffre d\'affaire TTC - Espèces**', value=f'{recette:,.0f} €'.replace(',', ' '))
    with cols[1]:
        st.metric('**Espèces déposer**', value=f'{depot:,.0f} €'.replace(',', ' '))
    with cols[2]:
        st.metric('**Solde du coffre**', value=f'{(recette-depot):,.0f} €'.replace(',', ' '))
    with cols[3]:
        st.write('**Historique dépôt**')
        st.dataframe(df_cash_visuel, hide_index=True)

    ## ---- CHECK DES ENVELOPPES ---
    st.subheader('**Audit des envellopes**', divider='blue')

    ## --- PRÉPARATION DES VARIABLES ---
    # 1. Préparation et nettoyage des dates
    audit_cash_caisse = df_caisse[['Date', 'Site', 'Espece']].copy()
    audit_cash_caisse['Date'] = pd.to_datetime(audit_cash_caisse['Date']).dt.date

    df_enveloppe_clean = df_enveloppe.copy()
    df_enveloppe_clean['Date'] = pd.to_datetime(df_enveloppe_clean['Date']).dt.date

    # 2. AGRÉGATION PRÉALABLE (La clé du problème)
    # On somme les espèces par jour/site pour être sûr de n'avoir qu'une ligne par couple
    audit_cash_caisse_agg = audit_cash_caisse.groupby(['Date', 'Site'], as_index=False)['Espece'].sum()

    # On fait de même pour les enveloppes
    df_enveloppe_agg = df_enveloppe_clean.groupby(['Date', 'Site'], as_index=False)['Montant'].sum()

    # 3. FUSION (Merge) 
    # Maintenant, le merge est du 1-pour-1, donc pas de doublons
    audit_cash = pd.merge(audit_cash_caisse_agg, df_enveloppe_agg, on=['Date', 'Site'], how='left')

    # 4. Calculs finaux
    audit_cash['Montant'] = audit_cash['Montant'].fillna(0) # Gérer les cas sans enveloppe
    audit_cash['Ecarts'] = (audit_cash['Espece'] - audit_cash['Montant']).round(2)
    audit_cash = audit_cash.sort_values(by='Date', ascending=True)

    # On prépare les données (plus récent en haut pour l'audit)
    df_historique_cash = audit_cash.sort_values(by='Date', ascending=False).head(10)

    # Création de l'expander
    with st.expander("📅 Historique des 10 derniers jours (Audit Cash)", expanded=False):
        st.write("Comparaison entre le CA théorique Espèces et le comptage des enveloppes.")
        
        # Affichage du tableau
        st.dataframe(
            df_historique_cash, 
            hide_index=True, 
            use_container_width=True,
            column_config={
                "Date": st.column_config.DateColumn("Date", format="DD/MM/YYYY"),
                "Ecart": st.column_config.NumberColumn("Écart (€)", format="%.2f €")
            }
        )
with tab5: ## VUE MASSE SALARIALE
    st.header('Masse salariale', divider='blue')

    ## ---- IMPORTATION DES DONNÉES ---
    df_rh_analyse = df_rh.copy()
    df_ventes_rh = df_ventes.query("année == @année_n").copy()

    ## --- FORMATAGE DES DONNÉES ---
    date_rh = df_rh_analyse['Date'].dt
    df_rh_analyse = df_rh_analyse.assign(
        année = date_rh.year,
        mois = date_rh.month,
        iso_semaine = date_rh.isocalendar().week
    )
    df_rh_analyse_année_n = df_rh_analyse.query('année == @année_n')
    
    ## ---- CALCUL D'AGGRÉGATION  ----- 
    rh_synthese = df_rh_analyse_année_n.groupby(['iso_semaine','Site'])['Montant'].sum().reset_index()
    rh_ventes_synthese = df_ventes_rh.groupby(['iso_semaine', 'Site'])['Ca_ht'].sum().reset_index()
    globale_rh = pd.merge(rh_ventes_synthese, rh_synthese, how='outer', on=['iso_semaine','Site'])
    globale_rh['Ratio (%)'] = ((globale_rh['Montant'] / globale_rh['Ca_ht'] ) * 100 ) . round(2)
    globale_rh['Valeur cible'] = 35
    globale_rh.columns = ['Semaine ISO', 'Site', "Chiffre d'affaire HT", "Masse salariale chargée", 'Ratio (%)', "Valeur cible"]

    ## --- CRÉATION DES FILTRES DE VISUALISATION ---- 
    st.subheader('**Sélectionner le point de vente**')
    site_rh = st.pills('', options=globale_rh['Site'].unique(), default='Restaurant')
    st.write("")
    var_rh = globale_rh.query('Site == @site_rh').groupby(['Semaine ISO', 'Site']).agg({
        "Chiffre d'affaire HT" : 'sum',
        "Masse salariale chargée" : 'sum',
        "Ratio (%)" : 'mean',
        "Valeur cible" : 'mean'
    }).round().reset_index()

    ## ----- CREATION DES INDICATEURS -----
    ca_rh = var_rh["Chiffre d'affaire HT"].sum()
    msc_rh = var_rh["Masse salariale chargée"].sum()
    ratio_rh = msc_rh / ca_rh
    ratio_cible = 0.35
    delta_rh = ratio_cible - ratio_rh
    ecart_rh_val = ca_rh * delta_rh * -1

    ### ---- AFFICHAGE DES INDICATEURS ---- 
    cols = st.columns(3)
    cols[0].metric("Chiffre d'affaire HT", value=f'{ca_rh:,.0f} €'.replace(",", " "), delta=f'{ca_rh / ca_année_n :.0%}', delta_arrow='off', delta_color='off', delta_description=f"Chiffre d'affaire : {année_n}")
    cols[1].metric("Masse salariale chargée", value=f'{msc_rh:,.0f} €'.replace(",", " "), delta=f'{ecart_rh_val:,.0f} €'.replace(",", " "), delta_color='inverse')
    cols[2].metric("Ratio MS/C", value=f'{ratio_rh:.2%}', delta=f'{delta_rh:.2%}', delta_arrow='off')

    ### ----- GRAPHIQUE DE L'EVOLUTION ----
    fig_rh = px.bar(
        var_rh,
        x='Semaine ISO',
        y='Ratio (%)',
        color='Site',
        template='plotly_white',
        labels={"Semaine ISO" : "<b>Semaine ISO</b>", 'Ratio (%)' : '<b>Ratio MS/C (%)</b>', 'Site' : '<b>Point de vente</b>'}
    )
    fig_rh.add_scatter(
        x=var_rh['Semaine ISO'],
        y=var_rh['Valeur cible'],
        name='Objectif 35%'
    )
    st.subheader('**Vue à la semaine**', divider='blue')
    st.plotly_chart(fig_rh, use_container_width=True)
with tab9: ## ARCHIVES 
    with st.expander('**Historique Master Data**'):
        df_ventes['Date'] = df_ventes['Date'].dt.date
        st.dataframe(df_ventes, hide_index=True, use_container_width=True)
