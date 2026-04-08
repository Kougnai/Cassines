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

    /* 3. Modernisation des Onglets (Tabs) */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: transparent;
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

@st.cache_data
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
    onglets = ['Ventes', 'Caisse', 'Events', 'Rh', 'Cash', 'Tips','Bon_livraison', 'Facture', 'Stock', 'Enveloppe']
    
    # 7. Extraction des données
    data = {nom: pd.DataFrame(spreadsheet.worksheet(nom).get_all_records(value_render_option='FORMATTED_VALUE')) for nom in onglets}
    
    return data
@st.cache_resource
def train_and_eval_prophet(df, site_name):
    # 1. On prend les données du site
    df_p = df[df['Site'] == site_name].copy()
    df_p = df_p.groupby('Date')['Ca_ht'].sum().reset_index()
    df_p.columns = ['ds', 'y']
    df_p = df_p[df_p['y'] > 10] # On vire les jours fermés

    if len(df_p) < 14: return None, 0, 0

    # 2. On entraîne le modèle sur TOUT
    m = Prophet(yearly_seasonality=True, weekly_seasonality=True)
    m.add_country_holidays(country_name='FR')
    m.fit(df_p)

    # 3. Calcul de fiabilité tout bête : on compare les 7 derniers jours réels
    # aux 7 derniers jours que le modèle vient de "re-prédire"
    last_7_days = df_p.tail(7)
    forecast = m.predict(last_7_days[['ds']])
    
    # Écart moyen en € (MAE)
    mae = abs(last_7_days['y'].values - forecast['yhat'].values).mean()
    # % d'erreur (MAPE)
    mape = (abs(last_7_days['y'].values - forecast['yhat'].values) / last_7_days['y'].values).mean()

    return m, mae, mape
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
df_ventes, df_caisse, df_events, df_rh, df_cash, df_tips, df_bl, df_facture, df_stock, df_enveloppe = dfs['Ventes'],dfs['Caisse'], dfs['Events'], dfs['Rh'], dfs['Cash'], dfs['Tips'], dfs['Bon_livraison'], dfs['Facture'], dfs['Stock'], dfs['Enveloppe']

## ---- NETTOYAGE ET CONVERSION ---
onglets_list = [df_cash, df_caisse, df_events, df_rh, df_tips, df_ventes, df_bl, df_facture, df_stock, df_enveloppe]
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
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs(["🌍 Vue globale", "📍 Vue par site", "📄 Compte Fournisseur", "💶 Cash",'👨‍🍳 Masse salariale', '✏️ Audit données', '📊 Stock', '🔮 Prévision', '📚 Archives'] )

with tab1: ## VUE GLOBALE 
    ### Analyse globale  KPI

    # --- CALCULS KIPS ---
    
    df_ventes = df_ventes.assign(
    année = df_ventes['Date'].dt.year,
    mois = df_ventes['Date'].dt.month,
    iso_semaine = df_ventes['Date'].dt.isocalendar().week,
    ticket_moyen = (df_ventes['Ca_ht'] / df_ventes['Nb_de_cvts']),
    jour_année = df_ventes['Date'].dt.day_of_year,
    jour_semaine = df_ventes['Date'].dt.day_of_week,
    )
    

    #### --------- TOUTE LES VARIABLES DU DASHBOARD ------

    année_n = df_ventes['année'].max()
    df_année_n = df_ventes.query('année == @année_n').copy()
    ca_année_n = df_année_n['Ca_ht'].sum().round()
    ca_année_n_1 = df_ventes.query('année == 2024')['Ca_ht'].sum().round().copy()
    ms_c_année_n = df_rh['Montant'].sum().round() / ca_année_n
    ms_c_cible = 0.35
    delta_msc_c = ms_c_année_n - ms_c_cible
    food_ca_année_n = df_année_n['Food_ht'].sum()
    food_cogs = food_ca_année_n / ca_année_n
    bev_ca_année_n = df_année_n['Bev_ht'].sum()
    bev_cogs = bev_ca_année_n / ca_année_n
    nb_cvts_année_n = df_année_n['Nb_de_cvts'].sum().round()

    ############################

    ## Affichage des données
    st.subheader(f'KPI : {année_n}', text_alignment='center', divider='blue')

    cola, colb, colc, cold = st.columns(4)
    #Chiffre d'affaire HT
    cola.metric("Chiffre d'affaire HT", value=f'{ca_année_n:,.0f} €'.replace(",", " "), delta=f'{ca_année_n-ca_année_n_1:,.0f} €'.replace(',', ' '), delta_description="VS N-1", help='Contient le chiffre d\'affaire pour privatisation')
    # Ms/c 
    colb.metric('MS/C',value=f'{ms_c_année_n:.2%}', delta=f'{delta_msc_c:.2%}', delta_color='inverse')
    #Food HT
    colc.metric('Food HT',value=f'{food_ca_année_n:,.0f} €'.replace(",", " "), delta=f'{food_cogs:.2%}', delta_color='off',delta_arrow='off', delta_description="Du chiffre d'affaire")
    #bev HT
    cold.metric('Bev HT',value=f'{bev_ca_année_n:,.0f} €'.replace(",", " "), delta=f'{bev_cogs:.2%}', delta_color='off', delta_arrow='off', delta_description="Du chiffre d'affaire")
    #Nombre de couvert
    cola.metric('Nombre de couvert', value=f'{nb_cvts_année_n:,.0f}'.replace(",", " "), delta='100 %', delta_color='blue', delta_arrow='off')
    #Ticket moyen
    colb.metric('Ticket moyen', value=f'{ca_année_n/nb_cvts_année_n:,.2f} €'.replace(",", " "), delta='Global', delta_color='orange', delta_arrow='off')
    # Food Cost
    colc.metric('**Food COGS**', value='32%', delta=f'{abs(28-32)} %', delta_arrow='up', delta_color='inverse', delta_description='Pas dynamique')
    # Bev cost
    cold.metric('**Bev COGS**', value='27%', delta=f'{abs(25-27)} %', delta_arrow='up', delta_color='inverse',delta_description='Pas dynamique')
    st.write('')

    ######### EVOLUTION DU CHIFFRE D'AFFAIRE YOY GLOBALE

    st.subheader('Évolution du Chiffre d\'affaire YoY', text_alignment='center', divider='blue')

    cols = st.columns(3)
    with cols[0]:
        mode = st.segmented_control('**Mode de vue**', options=['mois', 'iso_semaine'], default='mois')
    with cols[1]:
        year = st.pills('**Choissir l\'année**', options=df_ventes['année'].unique(), default=[2024, 2025], selection_mode='multi')
    with cols[2]:
        site = st.pills('**Point de vente**', options=df_ventes['Site'].unique(),default='Guinguette' , selection_mode='multi')

    ca_month = df_ventes.query('année == @year and Site == @site').groupby(['année', mode])['Ca_ht'].sum().reset_index().sort_values(['année', mode], ascending=True)

    if not ca_month.empty:
        idx_max = ca_month['Ca_ht'].idxmax()
        ca_max = ca_month.loc[idx_max, 'Ca_ht']
        max_x = ca_month.loc[idx_max, mode]
        max_annee = ca_month.loc[idx_max, 'année']

        # 3. Création du graphique de base
        fig_total = px.line(
            ca_month,
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
            text=[f"🏆 Record : {ca_max:,.0f} €".replace(',', ' ')],
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
    ca_events = df_events['Ca_ht'].sum()
    ca_events_privat = df_events.query("Ca_ht > 6000")['Ca_ht'].sum()
    ca_event_exploitation = df_events.query("Ca_ht < 6000")['Ca_ht'].sum()
    pct_privatision =  1 - (ca_events_privat /ca_events)
    pct_exploitation = 1 - (ca_event_exploitation / ca_events)
    cvt_event = df_events['Nb_de_cvts'].sum()
    event_tckmean_privat = ca_events_privat / df_events.query("Ca_ht > 6000")['Nb_de_cvts'].sum() 
    cvt_event_privat = df_events.query("Ca_ht > 6000")['Nb_de_cvts'].sum() 
    event_tckmean_exploit = ca_event_exploitation / df_events.query("Ca_ht < 6000")['Nb_de_cvts'].sum()
    cvt_event_exploit = df_events.query("Ca_ht < 6000")['Nb_de_cvts'].sum()

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
            df_events,
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
    with st.expander('📑 Cliquer pour afficher la base de donnée') :
            st.dataframe(df_events, hide_index=True)
with tab2: ## VUE PAR SITE 
        
    #  ---- Filtre dynamique ----- 
    pv = df_ventes['Site'].unique()

    st.header('Quels sites voulez-vous ?', text_alignment='center')
    aa, ab, ac = st.columns(3)
    with ab:
        site = st.pills('', options=pv, default='Guinguette', width=500)

    ##### ------ Préparation des données avec filtre 
    df_filtrer = df_ventes.query("année== [@année_n, @année_n -1] and Site == @site").copy() # < --- CHANGER LA DATE POUR 2026, 2025
    df_2025 = df_ventes.query('année ==@année_n and Site == @site').copy()
    var_pv = df_filtrer.groupby(["année","mois"])['Ca_ht'].sum().reset_index()
    var_pv['année'] = var_pv['année'].astype(str)
    ca_2025 = df_ventes.query("année == @année_n and Site == @site").groupby('année')['Ca_ht'].sum().reset_index()
    ca_2025 = ca_2025['Ca_ht'].sum()
    ca_2024 = df_ventes.query('année == 2024 and Site==@site')['Ca_ht'].sum()
    ms_c = df_rh.query('Site == @site')['Montant'].sum() / ca_2025
    valeur_cible_msc = 0.35
    delta_reel = valeur_cible_msc - ms_c
    nb_cvt = df_2025['Nb_de_cvts'].sum()
    food_ca = df_2025['Food_ht'].sum()
    bev_ca = df_2025['Bev_ht'].sum()

    "---" 

    ## ----- PARTIE 1/2 DES KPI ----
    st.header(f'KPI : {site}', text_alignment='center')

    col1, col2, col3, col4 = st.columns(4)
    # Chiffre d'affaire
    col1.metric("**Chiffre d'affaire HT**", f'{ca_2025:,.0f} €'.replace(",", " "), delta=f'{(ca_2025-ca_2024):,.0f} €'.replace(",", " "), delta_description='**vs N-1**')
    # Masse salarial Chargé
    col2.metric('**Masse salariale / chargée**', f'{ms_c:.0%}', delta=f'{delta_reel:.2%} ', delta_arrow='auto' )
    # Food Cost
    col3.metric('**Food COGS**', value='32%', delta=f'{abs(28-32)} %', delta_arrow='up', delta_color='inverse', delta_description='Pas dynamique')
    # Bev cost 
    col4.metric('**Bev COGS**', value='27%', delta=f'{abs(25-27)} %', delta_arrow='up', delta_color='inverse',delta_description='Pas dynamique')
    
    ## ---- PARTI 2/2 DES KPI ----
    a, b, c, d = st.columns(4)

    a.metric('**Nb de couverts**', f'{nb_cvt:,.0f}'.replace(",", " "), delta=f'{nb_cvt/nb_cvts_année_n:.0%}', delta_arrow='off', delta_color='blue', delta_description='Total de couverts')
    b.metric('**Ticket moyen**', f'{ca_2025/nb_cvt:,.2f} €', delta='Pas de données N-1', delta_arrow='off', delta_color='gray')
    c.metric("**CA Food HT**", f'{food_ca:,.0f} €'.replace(",", " "), delta=f'{food_ca/ca_2025:.0%}', delta_arrow='off', delta_color='off', delta_description='**Du CA**')
    d.metric("**CA Bev HT**", f'{bev_ca:,.0f} €'.replace(',', ' '), delta=f'{bev_ca/ca_2025:.0%}', delta_arrow='off', delta_color='off', delta_description='**Du CA**')

    "---"
    #  ------ GRAPHIQUE DE COMPARAISON CHIFFRE D'AFFAIRE PAR SITE YOY -----
    st.write(f'Evolution du CA mensuel comparaison YoY (N-1) : {site}')
    fig_pv = px.bar(
        var_pv, x='mois', y='Ca_ht',
        template='simple_white',color='année', barmode='group',
        labels={"Ca_ht": "<b>Chiffre d'affaire HT (€)</b>", 'mois':'<b>Numéro de mois</b>'}
    )
    fig_pv.update_traces(
        texttemplate = '<b>%{value:.3s}€</b>', textposition = 'outside'
    )
    st.plotly_chart(fig_pv, use_container_width=True)

    "---"
    #### ----- TEMPÉRATURE ----- ####
    st.subheader('Corrélation Température vs Chiffre d\'affaire', text_alignment='center')

    col_temp_1, col_temp_2 = st.columns(2)
    
    with col_temp_1:
        # Graphique de Corrélation Température vs Chiffre d\'affaire'
        fig_temp = px.scatter(
            df_filtrer,
            x='Temp_Max',
            y='Ca_ht',
            size='Nb_de_cvts',
            color='Site',
            trendline='ols',
            template='simple_white',
            labels={'Temp_Max' : '<b>Température °C</b>', "Ca_ht" : "<b>Chiffre d'affaire (€)</b>", "Nb_de_cvts" : '<b>Nombre de couverts</b>',  'Site' : "<b>Point de vente</b>"}    
        )
        st.plotly_chart(fig_temp, use_container_width=True)
    
    with col_temp_2 :

        ## Graphique de group de température 

        # Création des bins et groupby
        group_temp = df_2025.copy()
        bins = [-float('inf'), 15, 25, 30, float('inf')]
        labels = ['0-15°C', '16-25°C', '26-30°C', '+ 31°C']

        group_temp['tranche_temp'] = pd.cut(
            group_temp['Temp_Max'], 
            bins=bins, 
            labels=labels,
            right=True 
        )
        temp = group_temp.groupby(['Site','tranche_temp'])['Ca_ht'].sum().reset_index()

        fig_temp1 = px.bar(
            temp,
            x='tranche_temp',
            y='Ca_ht',
            color='Site',
            text_auto='.2s',
            template='simple_white',
            labels={'Ca_ht':'<b>Chiffre d\'affaire (€)</b>', 'tranche_temp':'<b>Catégorie de température</b>', 'Site' : "<b>Point de vente</b>"},
            range_y=[0,temp['Ca_ht'].max() * 1.2]
        )
        fig_temp1.update_traces(
            textposition = 'outside',
            texttemplate = '<b>%{value:.3s}€</b>'
        )
        st.plotly_chart(fig_temp1, use_container_width=True)

    mean_temp = group_temp.groupby(['Site', 'tranche_temp'])['Ca_ht'].mean().reset_index().round()
    mean_temp.columns = ['Site', 'Catégorie temp', 'Chiffre d\'affaire moyen']
    st.subheader('**Chiffre d\'affaire moyen - Température**', text_alignment='center')
    st.dataframe(mean_temp, hide_index=True)

    "---"  ### ------ Corrélation PLUIE vs CA ----- ###

    # --- Graphique de corrélation --- 
    st.subheader('Corrélation Pluie vs Chiffre d\'affaire', text_alignment='center')

    col_pluie_1, col_pluie_2 = st.columns(2)
    
    with col_pluie_1 :
        fig_pluie =px.scatter(
            df_filtrer,
            x='Pluie_mm',
            y='Ca_ht',
            size='Nb_de_cvts',
            color='Site',
            trendline='ols',
            template='simple_white',
            labels={'Pluie_mm' : '<b>Pluviométrie en mm</b>', "Ca_ht" : "<b>Chiffre d'affaire (€)</b>", "Nb_de_cvts" : '<b>Nombre de couverts</b>',  'Site' : "<b>Point de vente</b>"}    

        )
        st.plotly_chart(fig_pluie, use_container_width=True)
    
    with col_pluie_2:

        ## --- Graphique bar de Pluie
        group_pluie = df_2025.copy()

        bins = [-float('inf'), 10, 20, 30, float('inf')]
        labels = ['0-10 mm', '11-20 mm', '21-30 mm', '+ 31 mm']

        group_pluie['tranche_pluie'] = pd.cut(
            group_temp['Pluie_mm'], 
            bins=bins, 
            labels=labels,
            right=True)

        pluie = group_pluie.groupby(['Site','tranche_pluie'])['Ca_ht'].sum().reset_index()

        fig_pluie1 = px.bar(
            pluie,
            x='tranche_pluie',
            y='Ca_ht',
            color='Site',
            text_auto='.2s',
            template='simple_white',
            labels={'Ca_ht':'<b>Chiffre d\'affaire (€)</b>', 'tranche_pluie':'<b>Catégorie de pluie</b>',  'Site' : "<b>Point de vente</b>"},
            range_y=[0,pluie['Ca_ht'].max() * 1.5]
        )

        fig_pluie1.update_traces(
            textposition = 'outside',
            texttemplate = '<b>%{value:.3s}€</b>'
        )

        st.plotly_chart(fig_pluie1, use_container_width=True)

    ### ---- Tableau des moyenne 
    mean_pluie = group_pluie.groupby(['Site','tranche_pluie'])['Ca_ht'].mean().reset_index().round()
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
        sel_fournisseur = st.selectbox('**Quels fournisseurs ?**', options=compte_fournisseur['Fournisseur'].unique())
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
    depot = df_cash['Montant'].sum()
    df_cash_visuel = df_cash.copy()
    df_cash_visuel['Date dépôt'] = df_cash_visuel['Date'].dt.date
    df_cash_visuel = df_cash_visuel[['Date dépôt', 'Montant','Numero_ticket']]

    ## --- TIPS ----
    df_tips['Date'] = df_tips['Date'].dt.date
    tips_recu = df_ventes.query('année == @année_n').groupby('Site')['Tips'].sum().reset_index().copy()
    tips_donner = df_tips.groupby('Site')['Montant'].sum().reset_index().copy()
    tips_consolider = pd.merge(tips_recu, tips_donner, how='left', on='Site')
    tips_consolider['Solde'] = tips_consolider['Tips'] - tips_consolider['Montant']
    tips_consolider.columns = ['Site', 'Tips Récuperer', 'Tips Donner', 'Solde']
    

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
    
    st.header('**Suivi des tips**', divider='blue')
    cols = st.columns(2)
    with cols[0]:
        st.subheader('**Synthèse des Tips**')
        st.dataframe(tips_consolider, hide_index=True)
    with cols[1]:
        st.subheader('**Historique des Tips**')
        st.dataframe(df_tips, hide_index=True, use_container_width=True)

    ## ---- CHECK DES ENVELOPPES ---
    st.subheader('**Audit des envellopes**', divider='blue')

    ## --- PRÉPARATION DES VARIABLES ---
    audit_cash_caisse = df_caisse[['Date', 'Site', 'Espece', 'Tips']].copy()
    audit_cash_caisse['Date'] = audit_cash_caisse['Date'].dt.date
    df_enveloppe['Date'] = pd.to_datetime(df_enveloppe['Date']).dt.date
    audit_cash = pd.concat([audit_cash_caisse, df_enveloppe])
    audit_cash = audit_cash.groupby(['Date', 'Site']).agg({'Espece': 'sum', 'Tips':'sum', 'Montant':'sum'}).round(2).reset_index()
    audit_cash['Ecarts'] = audit_cash['Espece'] + audit_cash['Tips'] - audit_cash['Montant']
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
with tab6: ## VUE AUDIT DES DONNÉES 
    st.subheader("Audit des données Caisse et Events", divider='blue')
    
    # 1. Copies propres
    df_audit_event = df_events.query("année == 2025").copy()
    df_audit_caisse = df_caisse.copy()

    # 2. Harmonisation des dates (on s'assure que c'est du datetime pur avant le .date)
    df_audit_event['Date'] = pd.to_datetime(df_audit_event['Date'], errors='coerce').dt.date
    df_audit_caisse['Date'] = pd.to_datetime(df_audit_caisse['Date'], errors='coerce').dt.date

    # 3. Consolidation
    df_historique = pd.concat([df_audit_caisse, df_audit_event], ignore_index=True)
    
    # On supprime 'Client' s'il existe pour ne pas gêner le groupby numérique
    if 'Client' in df_historique.columns:
        df_historique = df_historique.drop(columns='Client')

    # 4. NETTOYAGE DE SÉCURITÉ (Anti-TypeError)
    # On force toutes les colonnes de calcul en numérique
    cols_calcul = [
        'Ca_ttc', 'Virement', 'Cb', 'Espece', 'Cheque', 'Autres', 'Tips', 
        'Taxes_20', 'Taxes_10', 'Taxes_5.5', 'Ca_ht', 'Nb_de_cvts', 
        'Food_ht', 'Bev_ht', 'Privatisation_ht', 'Autre_ht'
    ]

    for col in cols_calcul:
        if col in df_historique.columns:
            # On force la conversion : tout ce qui n'est pas un nombre devient NaN, puis 0
            df_historique[col] = pd.to_numeric(df_historique[col], errors='coerce').fillna(0)

    # 5. GROUPBY SÉCURISÉ
    # On ne demande l'agrégation que pour les colonnes qui existent vraiment dans df_historique
    dict_agg_final = {c: 'sum' for c in cols_calcul if c in df_historique.columns}

    if not df_historique.empty:
        df_audit_master = (df_historique.groupby(['Date', 'Site'])
                    .agg(dict_agg_final)
                    .round()
                    .reset_index())
        
        # Tri par date décroissante pour voir le plus récent
        df_audit_master = df_audit_master.sort_values('Date', ascending=False)
        
       # On crée l'expander avec le titre dynamique (le nombre de lignes)
        with st.expander(f"🔍 Détail de l'audit ({len(df_audit_master)} lignes combinées)", expanded=False):
            st.write("Voici les données brutes après fusion des sources :")
            st.dataframe(df_audit_master, hide_index=True, use_container_width=True)
    else:
        st.warning("Aucune donnée à auditer.")
    
    st.subheader('Ligne à ajouter à la Master Data', divider='blue')

    audit_ventes = df_ventes.copy()
    audit_ventes['Date'] = pd.to_datetime(audit_ventes['Date']).dt.date
    date_max = audit_ventes['Date'].max()
    data_ajouter = df_audit_master.query('Date > @date_max')

    st.dataframe(data_ajouter, hide_index=True)    
with tab7: ## VUE STOCK

    st.header('**Suivi du stock**', divider = 'blue', text_alignment='center')

    ## --- IMPORTATION DES DONNÉES ET MISE EN FORME ---
    df_stock['Date_heure'] = pd.to_datetime(df_stock['Date_heure'], dayfirst=True)
    df_stock = df_stock.assign(
        année = df_stock['Date_heure'].dt.year,
        mois = df_stock['Date_heure'].dt.month,
        iso_semaine = df_stock['Date_heure'].dt.isocalendar().week
    )
    df_stock.columns = df_stock.columns.str.strip()

    ### ----- VARIABLE DES KPI ---- 
    
    achat = df_stock.query('`Type de mouvement`== ("Entrée")')['Total'].sum()
    achat_food = df_stock.query('`Type de mouvement` == ("Entrée") and Departement == ("Food")')['Total'].sum()
    achat_bev = df_stock.query('`Type de mouvement` == ("Entrée") and Departement == ("Boisson")')['Total'].sum()

    consommer = df_stock.query('`Type de mouvement` == ("Sortie")')['Total'].sum()
    sortie_food = df_stock.query('`Type de mouvement` == ("Sortie") and Departement == ("Food")')['Total'].sum()
    sortie_bev = df_stock.query('`Type de mouvement` == ("Sortie") and Departement == ("Boisson")')['Total'].sum()
    
    stock_delta = achat - consommer
    stock_delta_food = achat_food - sortie_food
    stock_delta_bev = achat_bev - sortie_bev

    ### --- AFFICHAGE DES INDICATEURS STREAMLIT ----

    st.subheader('**Stock en cours**', divider='red')
    cols = st.columns(3)
    cols[0].metric('**Stock en cours HT**', value=f'{stock_delta:,.0f} €'.replace(',', ' '))
    cols[1].metric('**Stock en cours - Food HT**', value=f'{stock_delta_food:,.0f} €'.replace(',',' '))
    cols[2].metric('**Stock en cours - Bev HT**', value=f'{stock_delta_bev:,.0f} €'.replace(',',' '))
    st.subheader('**Total des achats HT**', divider='red')
    cols = st.columns(3)
    cols[0].metric('**Total des achats HT**', value=f'{achat:,.0f} €'.replace(',', ' '))
    cols[1].metric('**Total achat - Food HT**', value=f'{achat_food:,.0f} €'.replace(',',' '))
    cols[2].metric('**Total achat - Bev HT**', value=f'{achat_bev:,.0f} €'.replace(',',' '))
    st.subheader('**Total des sorties de stock HT**', divider='red')
    cols = st.columns(3)
    cols[0].metric('**Total sortie HT**', value=f'{consommer:,.0f} €'.replace(',', ' '))
    cols[1].metric('**Total sortie - Food HT**', value=f'{sortie_food:,.0f} €'.replace(',',' '))
    cols[2].metric('**Total sortie - Bev HT**', value=f'{sortie_bev:,.0f} €'.replace(',',' '))

    ### --- APPLICATION DES FILTRES PAR POINT DE VENTE --- 
    st.subheader('**Food & Bev - COGS**', divider='red')
    cols = st.columns(3)
    with cols[0]:
        cogs_site = st.pills('**Quels sites ?**', options=df_ventes['Site'].unique())
        ## ---- CALCUL DU FOOD ET BEV COGS -----
        ca_cogs = df_ventes.query('année == @année_n and Site == @cogs_site')['Ca_ht'].sum()
        food_cost = (df_stock.query('Destination == @cogs_site and Departement == ("Food")')['Total'].sum() /
                     ca_cogs
        )
        food_cogs_cible = 35
        bev_cost = (df_stock.query('Destination == @cogs_site and Departement == ("Boisson")')['Total'].sum() /
                     ca_cogs
        )
        bev_cost_cible = 25

    with cols[1]:
        st.metric('**Food COGS**', value=f'{food_cost:.2%}', delta=f'{food_cost - food_cogs_cible:,.2f}', delta_color='inverse')
    with cols[2]:
        st.metric('**Bev COGS**', value=f'{bev_cost:.2%}', delta=f'{bev_cost - bev_cost_cible:,.2f}', delta_color='inverse')
with tab8: ## PRÉVISION PROPHET 7 JOURS
    st.header("🔮 Prévisions d'Exploitation (7 jours)", divider='blue')

    sites_dispo = df_caisse['Site'].unique()
    sel_site = st.pills("Choisir le point de vente", options=sites_dispo, default='Restaurant')

    if st.button(f"🚀 Lancer l'IA pour {sel_site}"):
        with st.spinner("Analyse des cycles et synchronisation météo..."):
            
            # 1. Entraînement du modèle (utilise l'historique météo de df_caisse)
            model, mae, mape = train_and_eval_prophet(df_caisse, sel_site)

            if model is None:
                st.warning("Historique insuffisant pour ce site.")
            else:
                # 3. Préparation du dataframe 'future'
                future = model.make_future_dataframe(periods=7)
                
                # 4. Prédiction de 7 jours
                forecast = model.predict(future)
                
                # --- AFFICHAGE ---
                col1, col2, col3 = st.columns(3)
                fiabilite = max(0, 100 - (mape * 100))
                
                col1.metric("Fiabilité Score", f"{fiabilite:.1f} %", help="Basé sur l'erreur MAPE")
                col2.metric("Marge d'erreur (MAE)", f"{mae:.0f} €", help="Écart moyen par jour")
                col3.metric("CA Estimé (7j)", f"{forecast.tail(7)['yhat'].sum():,.0f} €".replace(",", " "))

                # Graphique
                from prophet.plot import plot_plotly
                fig = plot_plotly(model, forecast)
                fig.update_layout(template="plotly_dark", title=f"Tendance 7 jours : {sel_site}")
                st.plotly_chart(fig, use_container_width=True)
with tab9:
    with st.expander('**Historique Master Data**'):
        df_ventes['Date'] = df_ventes['Date'].dt.date
        st.dataframe(df_ventes, hide_index=True, use_container_width=True)
