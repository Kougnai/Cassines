import datetime
import calendar
import gspread
from google.oauth2.service_account import Credentials
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ==========================================
# 1. CONFIGURATION SYSTEM & EXECUTIVE CSS
# ==========================================
st.set_page_config(
    page_title="Executive Dashboard — Les Cassines",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
    .stApp {
        background-color: #0E1117;
        color: #E0E6ED;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 96%;
    }
    .brand-header {
        font-size: 1.6rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: #FFFFFF;
        margin-bottom: 0.2rem;
    }
    .brand-sub {
        font-size: 0.85rem;
        color: #8B949E;
        margin-bottom: 1.5rem;
    }
    [data-testid="stMetric"] {
        background: #161B22 !important;
        border: 1px solid #30363D !important;
        border-radius: 8px !important;
        padding: 16px 20px !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2) !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.8rem !important;
        font-weight: 600 !important;
        color: #8B949E !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.6rem !important;
        font-weight: 700 !important;
        color: #F0F6FC !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background-color: transparent;
        border-bottom: 1px solid #30363D;
        padding-bottom: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 40px;
        background-color: transparent;
        border: 1px solid transparent;
        border-radius: 6px;
        color: #8B949E;
        font-weight: 500;
        font-size: 0.88rem;
        padding: 0 16px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #21262D !important;
        border-color: #30363D !important;
        color: #58A6FF !important;
        font-weight: 600;
    }
    #MainMenu, header, footer { visibility: hidden; }
    </style>
""", unsafe_allow_html=True)


# ==========================================
# 2. PIPELINE DE DONNÉES & UTILS
# ==========================================
DATA_BP26 = {
    5: {"Restaurant": 120000, "Guinguette": 120000, "Le petit baigneur": 0, "Total": 240000},
    6: {"Restaurant": 150000, "Guinguette": 160000, "Le petit baigneur": 10000, "Total": 320000},
    7: {"Restaurant": 200000, "Guinguette": 250000, "Le petit baigneur": 45000, "Total": 495000},
    8: {"Restaurant": 200000, "Guinguette": 280000, "Le petit baigneur": 45000, "Total": 525000},
    9: {"Restaurant": 140000, "Guinguette": 100000, "Le petit baigneur": 0, "Total": 240000}
}

@st.cache_data(ttl=900, show_spinner=False)
def load_data():
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds_info = dict(st.secrets["gcp_service_account"])
    
    if "private_key" in creds_info:
        creds_info["private_key"] = creds_info["private_key"].replace("\\n", "\n")
 
    creds = Credentials.from_service_account_info(creds_info, scopes=scope)
    client = gspread.authorize(creds)
    spreadsheet = client.open("Cassines_bdd")
    
    onglets = ['Ventes', 'Caisse', 'Events', 'Rh', 'Cash', 'Stock', 'Enveloppe', 'Bp26']
    raw_dfs = {nom: pd.DataFrame(spreadsheet.worksheet(nom).get_all_records(value_render_option='FORMATTED_VALUE')) for nom in onglets}
    
    col_num = ['Ca_ttc', 'Ca_ht', 'Cb', 'Espece', 'Cheque', 'Food_ht', 'Bev_ht', 'Nb_de_cvts', 'Montant', 'Total']
    
    for df in raw_dfs.values():
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce').dt.normalize()
        
        for c in [col for col in col_num if col in df.columns]:
            df[c] = (
                df[c].astype(str)
                .str.replace(r'\s+', '', regex=True)
                .str.replace(',', '.')
            )
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
            
    df_v = raw_dfs['Ventes']
    df_v['année'] = df_v['Date'].dt.year
    df_v['mois'] = df_v['Date'].dt.month
    df_v['iso_semaine'] = df_v['Date'].dt.isocalendar().week
    df_v['ticket_moyen'] = (df_v['Ca_ht'] / df_v['Nb_de_cvts'].replace(0, pd.NA)).fillna(0)
    
    return raw_dfs

def fmt_eur(val, decimals=0):
    if pd.isna(val): return "0 €"
    return f"{val:,.{decimals}f} €".replace(",", " ").replace(".", ",")

def fmt_qty(val):
    if pd.isna(val): return "0"
    return f"{val:,.0f}".replace(",", " ")

def build_chart_theme(fig):
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color="#8B949E", size=12),
        margin=dict(l=10, r=10, t=35, b=10),
        xaxis=dict(showgrid=False, zeroline=False, color="#8B949E"),
        yaxis=dict(showgrid=True, gridcolor="#21262D", zeroline=False, color="#8B949E"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#C9D1D9"))
    )
    return fig

# --- CALCUL DE L'OBJECTIF BP À DATE PRORATISÉ (Jours d'ouverture) ---
def get_bp_target_to_date(site, max_date, year=2026):
    bp_key = "Le petit baigneur" if site in ["LPB", "Le petit baigneur"] else site
    bp_total = 0.0
    
    for m in range(5, max_date.month + 1):
        if m not in DATA_BP26:
            continue
        m_target = DATA_BP26[m].get(bp_key, 0)
        num_days_in_month = calendar.monthrange(year, m)[1]
        
        # Détermination du nombre de jours d'ouverture théoriques du mois
        if site == "Restaurant":
            # Fermé Lundis (0) et Mardis (1) -> Ouvert 5j/7
            open_days_in_month = sum(1 for day in range(1, num_days_in_month + 1) if datetime.date(year, m, day).weekday() not in [0, 1])
        else:
            # Guinguette / LPB : 7j/7
            open_days_in_month = num_days_in_month

        if m < max_date.month:
            # Mois complet
            bp_total += m_target
        else:
            # Mois en cours : calcul des jours ouverts écoulés jusqu'à max_date
            limit_day = min(max_date.day, num_days_in_month)
            if site == "Restaurant":
                open_days_elapsed = sum(1 for day in range(1, limit_day + 1) if datetime.date(year, m, day).weekday() not in [0, 1])
            else:
                open_days_elapsed = limit_day

            daily_target = m_target / open_days_in_month if open_days_in_month > 0 else 0
            bp_total += daily_target * open_days_elapsed
            
    return bp_total


# ==========================================
# 3. INITIALISATION
# ==========================================
try:
    dfs = load_data()
    df_ventes, df_rh, df_events, df_cash = dfs['Ventes'], dfs['Rh'], dfs['Events'], dfs['Cash']
except Exception as e:
    st.error(f"Erreur d'accès aux données : {e}")
    st.stop()

st.markdown('<div class="brand-header">LES CASSINES</div>', unsafe_allow_html=True)
st.markdown('<div class="brand-sub">Tableau de bord de pilotage financier & opérationnel</div>', unsafe_allow_html=True)

tab_global, tab_site, tab_data, tab_rh, tab_cash, tab_archives = st.tabs([
    " Vue Globale", 
    " Performance Etablissements", 
    " Données Ventes", 
    " Masse Salariale", 
    " Flux Cash", 
    " Archives"
])

# ==========================================
# TAB 1 : VUE GLOBALE
# ==========================================
with tab_global:
    # 1. Filtres Exécutifs Globaux
    c_s, c_m = st.columns([3, 2])
    with c_s:
        sites_opt = list(df_ventes['Site'].dropna().unique())
        selected_sites = st.multiselect("Périmètre Établissements :", options=sites_opt, default=sites_opt)

    st.write("")
    
    df_base = df_ventes.query("Site in @selected_sites")
    
    # 2. Logiciel DAF : Calcul YTD / À Date pour N vs N-1
    annee_max = int(df_base['année'].max()) if not df_base.empty else 2026
    annee_prev = annee_max - 1
    
    df_curr_all = df_base.query("année == @annee_max")
    
    if not df_curr_all.empty:
        max_date_n = df_curr_all['Date'].max()
        df_curr = df_curr_all[df_curr_all['Date'] <= max_date_n]
        
        max_date_n1 = pd.Timestamp(year=annee_prev, month=max_date_n.month, day=max_date_n.day)
        df_prev = df_base.query("année == @annee_prev and Date <= @max_date_n1")
    else:
        df_curr, df_prev = df_curr_all, df_base.query("année == @annee_prev")
        max_date_n = datetime.datetime.now()

    # Calculs vectorisés KPI
    ca_ht, ca_prev = df_curr['Ca_ht'].sum(), df_prev['Ca_ht'].sum()
    cvts, cvts_prev = df_curr['Nb_de_cvts'].sum(), df_prev['Nb_de_cvts'].sum()
    tck = ca_ht / cvts if cvts else 0
    tck_prev = ca_prev / cvts_prev if cvts_prev else 0
    
    df_rh['année'] = df_rh['Date'].dt.year
    ms_m = df_rh.query("année == @annee_max and Site in @selected_sites")['Montant'].sum()
    ratio_msc = (ms_m / ca_ht) if ca_ht else 0

    # Calcul de l'objectif BP à date global
    obj_bp_todate_g = sum(get_bp_target_to_date(site, max_date_n, year=annee_max) for site in selected_sites)
    ecart_bp_todate_g = ca_ht - obj_bp_todate_g
    pct_bp_todate_g = (ca_ht / obj_bp_todate_g) if obj_bp_todate_g else 0

    st.caption(f" *Comparaison à date arrêtée au **{max_date_n.strftime('%d/%m/%Y')}** (N vs N-1)*")
    
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric(f"CA HT {annee_max} (YTD)", fmt_eur(ca_ht), delta=fmt_eur(ca_ht - ca_prev), delta_description=f"vs {annee_prev} à date")
    k2.metric(
        "Écart BP 2026 à Date", 
        fmt_eur(ecart_bp_todate_g), 
        delta=f"{pct_bp_todate_g:.1%}", 
        delta_description=f"sur Cible BP : {fmt_eur(obj_bp_todate_g)}",
        help="Prorata calculé au jour près selon le calendrier d'ouverture (Resto 5j/7, Guinguette/LPB 7j/7)"
    )
    k3.metric("Fréquentation", fmt_qty(cvts), delta=fmt_qty(cvts - cvts_prev), delta_description=f"vs {annee_prev} à date")
    k4.metric("Ticket Moyen HT", fmt_eur(tck, 2), delta=fmt_eur(tck - tck_prev, 2), delta_description=f"vs {annee_prev} à date")
    k5.metric("Ratio MS / CA HT", f"{ratio_msc:.1%}", delta=f"{(0.35 - ratio_msc):.1%}", delta_color="inverse", help='Basé sur les données comptable mensuel')

    st.markdown("---")

    # 3. SUIVI DES OBJECTIFS & ÉCARTS BP 2026 (KPIs Annuel/Saison, Mois & Semaine)
    st.subheader(" Suivi des Cibles & Écarts BP 2026")
    
    months_bp = [5, 6, 7, 8, 9]
    
    def get_bp_target_for_sites(m, sites_list):
        tot = 0
        for site in sites_list:
            target_key = "Le petit baigneur" if site in ["LPB", "Le petit baigneur"] else site
            tot += DATA_BP26.get(m, {}).get(target_key, 0)
        return tot

    # A. Saison / Annuel Global (Mai à Septembre)
    obj_saison_g = sum(get_bp_target_for_sites(m, selected_sites) for m in months_bp)
    ca_saison_26_g = df_curr_all.query("mois in @months_bp")['Ca_ht'].sum()
    delta_saison_g = ca_saison_26_g - obj_saison_g
    pct_saison_g = (ca_saison_26_g / obj_saison_g) if obj_saison_g else 0

    # B. Mensuel (Dernier mois enregistré dans les données courantes)
    max_m26_g = int(df_curr_all['mois'].max()) if not df_curr_all.empty else 5
    ca_m_26_g = df_curr_all[df_curr_all['mois'] == max_m26_g]['Ca_ht'].sum()
    obj_m_26_g = get_bp_target_for_sites(max_m26_g, selected_sites)
    delta_m_g = ca_m_26_g - obj_m_26_g
    pct_m_g = (ca_m_26_g / obj_m_26_g) if obj_m_26_g else 0

    # C. Hebdomadaire (Dernière semaine ISO enregistrée)
    max_w26_g = int(df_curr_all['iso_semaine'].max()) if not df_curr_all.empty else 18
    df_w26_g = df_curr_all[df_curr_all['iso_semaine'] == max_w26_g]
    ca_w_26_g = df_w26_g['Ca_ht'].sum()
    
    month_of_w_g = int(df_w26_g['mois'].iloc[-1]) if not df_w26_g.empty else max_m26_g
    obj_w_26_g = get_bp_target_for_sites(month_of_w_g, selected_sites) / 4.33
    delta_w_g = ca_w_26_g - obj_w_26_g
    pct_w_g = (ca_w_26_g / obj_w_26_g) if obj_w_26_g else 0

    # Cartes KPI
    k_g1, k_g2, k_g3 = st.columns(3)

    k_g1.metric(
        label="Objectif Saison Global (BP 2026)",
        value=fmt_eur(ca_saison_26_g),
        delta=f"{fmt_eur(delta_saison_g)} ({pct_saison_g:.1%})",
        delta_description=f"sur Cible : {fmt_eur(obj_saison_g)}"
    )

    k_g2.metric(
        label=f"Objectif Mensuel (Mois {max_m26_g:02d})",
        value=fmt_eur(ca_m_26_g),
        delta=f"{fmt_eur(delta_m_g)} ({pct_m_g:.1%})",
        delta_description=f"sur Cible : {fmt_eur(obj_m_26_g)}"
    )

    k_g3.metric(
        label=f"Objectif Hebdo (Semaine {max_w26_g})",
        value=fmt_eur(ca_w_26_g),
        delta=f"{fmt_eur(delta_w_g)} ({pct_w_g:.1%})",
        delta_description=f"sur Target Proratisée : {fmt_eur(obj_w_26_g)}"
    )

    st.markdown("---")
    
    # 4. TRAJECTOIRE (Option Semaine vs Mois)
    st.subheader(" Trajectoire de l'Activité")
    
    f_site, f_gran, f_years = st.columns([2, 2, 3])
    with f_site:
        site_traj_opt = ["Tous (Global)"] + list(df_ventes['Site'].dropna().unique())
        selected_traj_site = st.selectbox("Périmètre Trajectoire :", options=site_traj_opt, index=0)
    with f_gran:
        granularity = st.radio("Maillage temporel :", options=["Mensuel", "Hebdomadaire (Semaine)"], horizontal=True)
    with f_years:
        all_years = sorted(list(df_ventes['année'].dropna().unique()), reverse=True)
        default_years = [annee_max, annee_prev] if len(all_years) >= 2 else all_years
        selected_traj_years = st.multiselect("Années comparées :", options=all_years, default=default_years)

    df_traj = df_ventes if selected_traj_site == "Tous (Global)" else df_ventes.query("Site == @selected_traj_site")
    df_traj = df_traj.query("année in @selected_traj_years")

    if not df_traj.empty and len(selected_traj_years) > 0:
        group_col = 'mois' if granularity == "Mensuel" else 'iso_semaine'
        df_chart_traj = df_traj.groupby(['année', group_col])['Ca_ht'].sum().reset_index()
        df_chart_traj['année_str'] = df_chart_traj['année'].astype(str)
        
        fig_line = px.line(
            df_chart_traj, 
            x=group_col, 
            y='Ca_ht', 
            color='année_str',
            labels={group_col: 'Mois' if granularity == "Mensuel" else 'Semaine (ISO)', 'Ca_ht': 'CA HT (€)', 'année_str': 'Exercice'},
            markers=True,
            color_discrete_sequence=px.colors.qualitative.Safe
        )

        fig_line.update_layout(hovermode="x unified")
        fig_line.update_xaxes(tickmode='linear', tick0=1, dtick=1 if granularity == "Mensuel" else 2)
        st.plotly_chart(build_chart_theme(fig_line), use_container_width=True)

    # 5. TABLEAU DES ÉCARTS RÉEL 2026 VS OBJECTIF BP
    st.markdown("---")
    st.subheader(" Suivi de la Performance vs Objectifs BP 2026")
    
    df_bp_comp = pd.DataFrame({'Mois': months_bp})
    
    df_v_26 = df_ventes.query("année == 2026 and Site in @selected_sites")
    df_bp_comp['CA Réel 2026'] = df_bp_comp['Mois'].map(df_v_26.groupby('mois')['Ca_ht'].sum()).fillna(0)

    df_bp_comp['Objectif BP 2026'] = df_bp_comp['Mois'].apply(lambda m: get_bp_target_for_sites(m, selected_sites))
    df_bp_comp['Écart (€)'] = df_bp_comp['CA Réel 2026'] - df_bp_comp['Objectif BP 2026']
    df_bp_comp['Atteinte (%)'] = (df_bp_comp['CA Réel 2026'] / df_bp_comp['Objectif BP 2026'].replace(0, pd.NA)).fillna(0)
    
    df_bp_disp = df_bp_comp.copy()
    df_bp_disp['CA Réel 2026'] = df_bp_disp['CA Réel 2026'].apply(fmt_eur)
    df_bp_disp['Objectif BP 2026'] = df_bp_disp['Objectif BP 2026'].apply(fmt_eur)
    df_bp_disp['Écart (€)'] = df_bp_disp['Écart (€)'].apply(fmt_eur)
    df_bp_disp['Atteinte (%)'] = df_bp_disp['Atteinte (%)'].apply(lambda x: f"{x:.1%}")
    
    st.dataframe(df_bp_disp, use_container_width=True, hide_index=True)

# ==========================================
# TAB 2 : PERFORMANCE ÉTABLISSEMENTS
# ==========================================
with tab_site:
    st.subheader("Analyse Détaillée par Point de Vente")
    
    # --- 1. SELECTION ET BASE INITIALE ---
    selected_site = st.radio("Sélectionner l'établissement :", options=["Guinguette", "Restaurant", "LPB"], horizontal=True)
    target_site = "LPB" if selected_site == "LPB" else selected_site
    bp_key = "Le petit baigneur" if selected_site == "LPB" else selected_site
    
    df_s_all = df_ventes.query("Site == @target_site").copy()
    annee_max_s = int(df_s_all['année'].max()) if not df_s_all.empty else 2026
    
    df_s_26_all = df_s_all.query("année == @annee_max_s")
    if not df_s_26_all.empty:
        max_date_s = df_s_26_all['Date'].max()
        df_s_26 = df_s_26_all[df_s_26_all['Date'] <= max_date_s]
        
        max_date_s1 = pd.Timestamp(year=annee_max_s - 1, month=max_date_s.month, day=max_date_s.day)
        df_s_25 = df_s_all.query("année == (@annee_max_s - 1) and Date <= @max_date_s1")
    else:
        df_s_26 = df_s_26_all
        df_s_25 = df_s_all.query("année == (@annee_max_s - 1)")
        max_date_s = datetime.datetime.now()
    
    ca_s26, ca_s25 = df_s_26['Ca_ht'].sum(), df_s_25['Ca_ht'].sum()
    cvt_s26, cvt_s25 = df_s_26['Nb_de_cvts'].sum(), df_s_25['Nb_de_cvts'].sum()
    
    # Calcul de l'objectif BP à date par site
    obj_bp_todate_site = get_bp_target_to_date(target_site, max_date_s, year=annee_max_s)
    ecart_bp_todate_site = ca_s26 - obj_bp_todate_site
    pct_bp_todate_site = (ca_s26 / obj_bp_todate_site) if obj_bp_todate_site else 0

    st.caption(f" *Comparaison YTD arrêtée au **{max_date_s.strftime('%d/%m/%Y')}***")
    s1, s2, s3, s4 = st.columns(4)
    s1.metric(f"CA HT {annee_max_s} (YTD)", fmt_eur(ca_s26), delta=fmt_eur(ca_s26 - ca_s25), delta_description=f"vs {annee_max_s - 1} à date")
    s2.metric(
        "Écart BP 2026 à Date", 
        fmt_eur(ecart_bp_todate_site), 
        delta=f"{pct_bp_todate_site:.1%}", 
        delta_description=f"sur Cible BP : {fmt_eur(obj_bp_todate_site)}",
        help=f"Prorata au jour près ({'5j/7' if target_site == 'Restaurant' else '7j/7'})"
    )
    s3.metric("Couverts (YTD)", fmt_qty(cvt_s26), delta=fmt_qty(cvt_s26 - cvt_s25), delta_description=f"vs {annee_max_s - 1} à date")
    s4.metric("Ticket Moyen", fmt_eur(ca_s26/cvt_s26 if cvt_s26 else 0, 2))

    st.write("")
    st.markdown("##### Comparatif Évolution Mensuelle vs Objectifs BP 2026")
    
    months = [5, 6, 7, 8, 9]
    df_bp_site = pd.DataFrame({'mois': months})
    df_bp_site['CA 2026'] = df_bp_site['mois'].map(df_s_all.query("année == 2026").groupby('mois')['Ca_ht'].sum()).fillna(0)
    df_bp_site['CA 2025'] = df_bp_site['mois'].map(df_s_all.query("année == 2025").groupby('mois')['Ca_ht'].sum()).fillna(0)
    df_bp_site['Objectif BP'] = df_bp_site['mois'].apply(lambda m: DATA_BP26.get(m, {}).get(bp_key, 0))

    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(x=df_bp_site['mois'], y=df_bp_site['CA 2026'], name='CA Réel 2026', marker_color='#58A6FF'))
    fig_bar.add_trace(go.Bar(x=df_bp_site['mois'], y=df_bp_site['CA 2025'], name='CA Réel 2025', marker_color="#2C45C4"))
    fig_bar.add_trace(go.Scatter(x=df_bp_site['mois'], y=df_bp_site['Objectif BP'], name='Cible BP 2026', line=dict(color='#D29922', width=3, dash='dash')))
    
    fig_bar.update_layout(barmode='group', hovermode='x unified')
    st.plotly_chart(build_chart_theme(fig_bar), use_container_width=True)

    # --- 2. TABLEAU DE SUIVI DES ÉCARTS RÉEL 2026 VS OBJECTIF BP ---
    st.markdown("---")
    st.subheader("Suivi de la Performance vs Objectifs BP 2026")
    
    df_bp_comp_site = pd.DataFrame({'Mois': months})
    df_v_26_site = df_s_all.query("année == 2026")
    
    df_bp_comp_site['CA Réel 2026'] = df_bp_comp_site['Mois'].map(df_v_26_site.groupby('mois')['Ca_ht'].sum()).fillna(0)
    df_bp_comp_site['Objectif BP 2026'] = df_bp_comp_site['Mois'].apply(lambda m: DATA_BP26.get(m, {}).get(bp_key, 0))
    df_bp_comp_site['Écart (€)'] = df_bp_comp_site['CA Réel 2026'] - df_bp_comp_site['Objectif BP 2026']
    df_bp_comp_site['Atteinte (%)'] = (df_bp_comp_site['CA Réel 2026'] / df_bp_comp_site['Objectif BP 2026'].replace(0, pd.NA)).fillna(0)
    
    df_bp_disp_site = df_bp_comp_site.copy()
    df_bp_disp_site['CA Réel 2026'] = df_bp_disp_site['CA Réel 2026'].apply(fmt_eur)
    df_bp_disp_site['Objectif BP 2026'] = df_bp_disp_site['Objectif BP 2026'].apply(fmt_eur)
    df_bp_disp_site['Écart (€)'] = df_bp_disp_site['Écart (€)'].apply(fmt_eur)
    df_bp_disp_site['Atteinte (%)'] = df_bp_disp_site['Atteinte (%)'].apply(lambda x: f"{x:.1%}")
    
    st.dataframe(df_bp_disp_site, use_container_width=True, hide_index=True)

    # --- 3. SUIVI ET KPI DES OBJECTIFS ---
    st.markdown("---")
    st.markdown("##### Suivi des Objectifs & Écarts BP 2026 (Saison, Mois & Semaine)")

    # A. Saison Global (Mai à Septembre)
    obj_saison = sum(DATA_BP26.get(m, {}).get(bp_key, 0) for m in months)
    ca_saison_26 = df_s_all.query("année == 2026 and mois in @months")['Ca_ht'].sum()
    delta_saison = ca_saison_26 - obj_saison
    pct_saison = (ca_saison_26 / obj_saison) if obj_saison else 0

    # B. Mensuel (Dernier mois enregistré)
    max_m26 = int(df_s_26_all['mois'].max()) if not df_s_26_all.empty else 5
    ca_m_26 = df_s_26_all[df_s_26_all['mois'] == max_m26]['Ca_ht'].sum()
    obj_m_26 = DATA_BP26.get(max_m26, {}).get(bp_key, 0)
    delta_m = ca_m_26 - obj_m_26
    pct_m = (ca_m_26 / obj_m_26) if obj_m_26 else 0

    # C. Hebdomadaire (Dernière semaine ISO enregistrée)
    max_w26 = int(df_s_26_all['iso_semaine'].max()) if not df_s_26_all.empty else 18
    df_w26 = df_s_26_all[df_s_26_all['iso_semaine'] == max_w26]
    ca_w_26 = df_w26['Ca_ht'].sum()
    
    month_of_w = int(df_w26['mois'].iloc[-1]) if not df_w26.empty else max_m26
    obj_w_26 = DATA_BP26.get(month_of_w, {}).get(bp_key, 0) / 4.33
    delta_w = ca_w_26 - obj_w_26
    pct_w = (ca_w_26 / obj_w_26) if obj_w_26 else 0

    # Affichage des 3 Cartes KPI
    k_obj1, k_obj2, k_obj3 = st.columns(3)

    k_obj1.metric(
        label="Objectif Saison Global (BP 2026)",
        value=fmt_eur(ca_saison_26),
        delta=f"{fmt_eur(delta_saison)} ({pct_saison:.1%})",
        delta_description=f"sur Cible : {fmt_eur(obj_saison)}"
    )

    k_obj2.metric(
        label=f"Objectif Mensuel (Mois M{max_m26:02d})",
        value=fmt_eur(ca_m_26),
        delta=f"{fmt_eur(delta_m)} ({pct_m:.1%})",
        delta_description=f"sur Cible : {fmt_eur(obj_m_26)}"
    )

    k_obj3.metric(
        label=f"Objectif Hebdo (Semaine {max_w26})",
        value=fmt_eur(ca_w_26),
        delta=f"{fmt_eur(delta_w)} ({pct_w:.1%})",
        delta_description=f"sur Target Proratisée : {fmt_eur(obj_w_26)}"
    )

# ==========================================
# TAB 3 : DONNÉES VENTES (HISTORIQUE & KPI COMPARATIFS N vs N-1)
# ==========================================
with tab_data:
    st.subheader("Explorateur de Données — Ventes")
    st.caption("Consultation globale des ventes et analyse comparative N vs N-1.")
    
    min_date_db = df_ventes['Date'].min().date() if not df_ventes.empty else datetime.date(2020, 1, 1)
    max_date_db = df_ventes['Date'].max().date() if not df_ventes.empty else datetime.date.today()
    
    default_start = max(datetime.date(2026, 1, 1), min_date_db)
    default_end = min(datetime.date(2026, 12, 31), max_date_db)
    
    if default_start > default_end:
        default_start = min_date_db

    f_d1, f_d2 = st.columns(2)
    with f_d1:
        date_selection = st.date_input(
            "Plage de dates (Période N) :",
            value=(default_start, default_end),
            min_value=min_date_db,
            max_value=max_date_db
        )
    with f_d2:
        site_data_filter = st.selectbox("Établissement :", options=["Tous"] + list(df_ventes['Site'].dropna().unique()), key="data_site_filter")
        
    if isinstance(date_selection, tuple) and len(date_selection) == 2:
        t_start, t_end = pd.to_datetime(date_selection[0]), pd.to_datetime(date_selection[1])
        
        df_export = df_ventes[(df_ventes['Date'] >= t_start) & (df_ventes['Date'] <= t_end)]
        t_start_n1 = t_start - pd.DateOffset(years=1)
        t_end_n1 = t_end - pd.DateOffset(years=1)
        df_export_n1 = df_ventes[(df_ventes['Date'] >= t_start_n1) & (df_ventes['Date'] <= t_end_n1)]
        
        if site_data_filter != "Tous":
            df_export = df_export[df_export['Site'] == site_data_filter]
            df_export_n1 = df_export_n1[df_export_n1['Site'] == site_data_filter]

        ca_n = df_export['Ca_ht'].sum()
        ca_n1 = df_export_n1['Ca_ht'].sum()
        delta_ca_eur = ca_n - ca_n1
        delta_ca_pct = (delta_ca_eur / ca_n1) if ca_n1 else 0

        cvt_n = df_export['Nb_de_cvts'].sum()
        cvt_n1 = df_export_n1['Nb_de_cvts'].sum()
        delta_cvt_qty = cvt_n - cvt_n1
        delta_cvt_pct = (delta_cvt_qty / cvt_n1) if cvt_n1 else 0

        tm_n = ca_n / cvt_n if cvt_n else 0
        tm_n1 = ca_n1 / cvt_n1 if cvt_n1 else 0
        delta_tm = tm_n - tm_n1

        st.caption(f" *Comparaison vs période équivalente N-1 (**{t_start_n1.strftime('%d/%m/%Y')}** au **{t_end_n1.strftime('%d/%m/%Y')}**)*")
        
        k1, k2, k3, k4 = st.columns(4)
        k1.metric(label=f"CA HT ({t_start.year})", value=fmt_eur(ca_n), delta=f"{fmt_eur(delta_ca_eur)} ({delta_ca_pct:+.1%})")
        k2.metric(label=f"Couverts ({t_start.year})", value=fmt_qty(cvt_n), delta=f"{fmt_qty(delta_cvt_qty)} ({delta_cvt_pct:+.1%})")
        k3.metric(label="Ticket Moyen HT", value=fmt_eur(tm_n, 2), delta=fmt_eur(delta_tm, 2))
        k4.metric(label="CA HT N-1 (Rappel)", value=fmt_eur(ca_n1), delta_color="off")

        st.markdown("---")
        
        st.markdown("##### Détail des Transactions")
        df_disp = df_export.copy()
        df_disp['Date'] = df_disp['Date'].dt.strftime('%d/%m/%Y')
        cols_display = ['Date', 'Site', 'Ca_ht', 'Ca_ttc', 'Nb_de_cvts', 'ticket_moyen', 'Food_ht', 'Bev_ht', 'Espece', 'Cb']
        
        st.dataframe(
            df_disp[[c for c in cols_display if c in df_disp.columns]],
            use_container_width=True,
            hide_index=True
        )

# ==========================================
# TAB 4 : MASSE SALARIALE (ANALYSES & DOUBLE AXE)
# ==========================================
with tab_rh:
    st.subheader("Suivi & Contrôle de la Masse Salariale")
    
    rh_site_opt = ["Tous les établissements"] + list(df_rh['Site'].dropna().unique())
    selected_rh_site = st.selectbox("Filtrer par site :", options=rh_site_opt, index=0)
    
    df_rh_f = df_rh.query("année == 2026")
    df_v_rh = df_ventes.query("année == 2026")
    
    if selected_rh_site != "Tous les établissements":
        df_rh_f = df_rh_f[df_rh_f['Site'] == selected_rh_site]
        df_v_rh = df_v_rh[df_v_rh['Site'] == selected_rh_site]

    ms_mensuel = df_rh_f.groupby(df_rh_f['Date'].dt.month)['Montant'].sum()
    ca_mensuel = df_v_rh.groupby('mois')['Ca_ht'].sum()
    
    df_rh_combo = pd.DataFrame({'mois': range(1, 13)})
    df_rh_combo['CA_HT'] = df_rh_combo['mois'].map(ca_mensuel).fillna(0)
    df_rh_combo['MS'] = df_rh_combo['mois'].map(ms_mensuel).fillna(0)
    df_rh_combo['Ratio_MS'] = (df_rh_combo['MS'] / df_rh_combo['CA_HT'].replace(0, pd.NA)).fillna(0)
    df_rh_combo = df_rh_combo[(df_rh_combo['CA_HT'] > 0) | (df_rh_combo['MS'] > 0)]

    tot_ca = df_rh_combo['CA_HT'].sum()
    tot_ms = df_rh_combo['MS'].sum()
    ratio_cumule = tot_ms / tot_ca if tot_ca else 0
    
    r1, r2, r3 = st.columns(3)
    r1.metric("Masse Salariale Cumulée 2026", fmt_eur(tot_ms))
    r2.metric("Ratio MS / CA HT Global", f"{ratio_cumule:.1%}")
    r3.metric("Écart Cible (35%)", f"{(0.35 - ratio_cumule):.1%}", delta_color="inverse")

    st.write("")
    
    fig_dual = go.Figure()
    fig_dual.add_trace(go.Bar(
        x=df_rh_combo['mois'],
        y=df_rh_combo['CA_HT'],
        name="Chiffre d'Affaires HT (€)",
        marker_color='#58A6FF',
        opacity=0.85
    ))
    fig_dual.add_trace(go.Scatter(
        x=df_rh_combo['mois'],
        y=df_rh_combo['Ratio_MS'],
        name="Ratio MS / CA (%)",
        yaxis="y2",
        mode="lines+markers+text",
        text=df_rh_combo['Ratio_MS'].apply(lambda x: f"{x:.1%}"),
        textposition="top center",
        line=dict(color='#F2994A', width=3),
        marker=dict(size=8)
    ))

    fig_dual.update_layout(
        hovermode="x unified",
        xaxis=dict(title="Mois", tickmode='linear', tick0=1, dtick=1),
        yaxis=dict(title="CA HT (€)", showgrid=False),
        yaxis2=dict(
            title="Ratio MS (%)",
            overlaying="y",
            side="right",
            tickformat=".0%",
            showgrid=True,
            gridcolor="#21262D"
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(build_chart_theme(fig_dual), use_container_width=True)

    st.markdown("---")
    st.markdown("##### Détail des Écritures RH 2026")
    st.dataframe(df_rh_f, use_container_width=True, hide_index=True)

# ==========================================
# TAB 5 : FLUX DE CASH
# ==========================================
with tab_cash:
    st.subheader("Audit & Pilotage du Coffre / Espèces")
    
    df_v_26 = df_ventes[df_ventes['Date'].dt.year == 2026].copy()
    recette_esp_tot = df_v_26['Espece'].sum()
    
    df_cash['Date_clean'] = pd.to_datetime(df_cash['Date'], dayfirst=True, errors='coerce')
    df_cash_26 = df_cash[df_cash['Date_clean'].dt.year == 2026].copy()

    if 'Semaine' in df_cash_26.columns:
        df_cash_26['iso_semaine'] = pd.to_numeric(
            df_cash_26['Semaine'].astype(str).str.extract(r'(\d+)')[0], 
            errors='coerce'
        )
    else:
        df_cash_26['iso_semaine'] = df_cash_26['Date_clean'].dt.isocalendar().week

    fdc_coffre = 4978
    depots_esp_tot = df_cash_26['Montant'].sum() - fdc_coffre
    solde_coffre_tot = recette_esp_tot - depots_esp_tot
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Encaissements Espèces Global (2026)", fmt_eur(recette_esp_tot))
    c2.metric("Dépôts Enveloppes / Banque Total", fmt_eur(depots_esp_tot))
    c3.metric("Solde Théorique Coffre Global", fmt_eur(solde_coffre_tot))

    st.markdown("---")

    st.markdown("##### Suivi de l'Encaisse & Coffre par Établissement")
    
    sites_cash = list(df_ventes['Site'].dropna().unique())
    summary_cash_data = []
    
    for site in sites_cash:
        recette_site = df_v_26[df_v_26['Site'] == site]['Espece'].sum()
        
        site_cash_keys = [site]
        if site in ["LPB", "Le petit baigneur"]:
            site_cash_keys = ["LPB", "Le petit baigneur"]
            
        if 'Site' in df_cash_26.columns:
            depots_site = df_cash_26[df_cash_26['Site'].isin(site_cash_keys)]['Montant'].sum()
        else:
            depots_site = 0.0
            
        solde_site = recette_site - depots_site
        
        summary_cash_data.append({
            'Établissement': site,
            'Recettes Espèces 2026': recette_site,
            'Dépôts Banque / Enveloppes': depots_site,
            'Solde Théorique Coffre': solde_site
        })
        
    df_cash_summary = pd.DataFrame(summary_cash_data)
    
    cols_sites = st.columns(len(sites_cash))
    for idx, row in df_cash_summary.iterrows():
        with cols_sites[idx]:
            st.markdown(f"**{row['Établissement']}**")
            st.metric("Encaissements Espèces", fmt_eur(row['Recettes Espèces 2026']))
            st.metric("Dépôts / Enveloppes", fmt_eur(row['Dépôts Banque / Enveloppes']))
            st.metric("Solde Coffre Site", fmt_eur(row['Solde Théorique Coffre']))

    with st.expander(" Consulter le tableau récapitulatif par site"):
        df_disp_cash = df_cash_summary.copy()
        df_disp_cash['Recettes Espèces 2026'] = df_disp_cash['Recettes Espèces 2026'].apply(fmt_eur)
        df_disp_cash['Dépôts Banque / Enveloppes'] = df_disp_cash['Dépôts Banque / Enveloppes'].apply(fmt_eur)
        df_disp_cash['Solde Théorique Coffre'] = df_disp_cash['Solde Théorique Coffre'].apply(fmt_eur)
        st.dataframe(df_disp_cash, use_container_width=True, hide_index=True)

    st.markdown("---")

    st.markdown("##### Suivi des Écarts Hebdomadaires (Caisse vs Dépôts)")

    v_hebdo = df_v_26.groupby(['Site', 'iso_semaine'])['Espece'].sum().reset_index()
    v_hebdo.rename(columns={'Espece': 'Espèces Caisse (€)'}, inplace=True)

    if 'Site' in df_cash_26.columns:
        df_cash_prep = df_cash_26.copy()
        df_cash_prep['Site'] = df_cash_prep['Site'].replace({"Le petit baigneur": "LPB"})
        
        c_hebdo = df_cash_prep.groupby(['Site', 'iso_semaine'])['Montant'].sum().reset_index()
        c_hebdo.rename(columns={'Montant': 'Dépôts Enveloppes (€)'}, inplace=True)
    else:
        c_hebdo = pd.DataFrame(columns=['Site', 'iso_semaine', 'Dépôts Enveloppes (€)'])

    df_hebdo_cash = pd.merge(v_hebdo, c_hebdo, on=['Site', 'iso_semaine'], how='outer').fillna(0)
    df_hebdo_cash = df_hebdo_cash[df_hebdo_cash['iso_semaine'] > 0]
    df_hebdo_cash['iso_semaine'] = df_hebdo_cash['iso_semaine'].astype(int)
    
    df_hebdo_cash['Écart Semaine (€)'] = df_hebdo_cash['Espèces Caisse (€)'] - df_hebdo_cash['Dépôts Enveloppes (€)']
    df_hebdo_cash.sort_values(by=['Site', 'iso_semaine'], ascending=[True, False], inplace=True)

    site_cash_filter = st.selectbox("Filtrer l'analyse hebdomadaire par site :", options=["Tous"] + sites_cash)
    
    df_hebdo_disp = df_hebdo_cash.copy()
    if site_cash_filter != "Tous":
        df_hebdo_disp = df_hebdo_disp[df_hebdo_disp['Site'] == site_cash_filter]

    df_hebdo_disp['Semaine ISO'] = df_hebdo_disp['iso_semaine'].apply(lambda s: f"Semaine {s:02d}")
    df_hebdo_disp['Espèces Caisse (€)'] = df_hebdo_disp['Espèces Caisse (€)'].apply(fmt_eur)
    df_hebdo_disp['Dépôts Enveloppes (€)'] = df_hebdo_disp['Dépôts Enveloppes (€)'].apply(fmt_eur)
    df_hebdo_disp['Écart Semaine (€)'] = df_hebdo_disp['Écart Semaine (€)'].apply(fmt_eur)

    cols_hebdo_view = ['Site', 'Semaine ISO', 'Espèces Caisse (€)', 'Dépôts Enveloppes (€)', 'Écart Semaine (€)']

    with st.expander(" Consulter le détail des écarts hebdomadaires (Caisse vs Dépôts par Semaine CA)", expanded=True):
        st.dataframe(df_hebdo_disp[cols_hebdo_view], use_container_width=True, hide_index=True)

    st.markdown("---")
    with st.expander(" Historique complet des mouvements de Cash / Enveloppes 2026"):
        st.dataframe(df_cash_26, use_container_width=True, hide_index=True)

# ==========================================
# TAB 6 : ARCHIVES COMPLÈTES
# ==========================================
with tab_archives:
    st.subheader("Archives Brutes Master Data")
    with st.expander("Consulter la base globale des ventes (Historique Complet)"):
        st.dataframe(df_ventes, use_container_width=True, hide_index=True)
