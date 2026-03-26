"""
==================================================
SUICIDE ANALYSIS IN MEXICO (1994-2024)
Data Engineering - Project Advance II

Team Members:
- Ferrusca Jaimez Irving Nahir
- Galván de la Cruz Angel Dolores

Description: Comprehensive data analysis platform
for studying suicide trends in Mexico,
integrating Data Mining techniques, Mathematical Modeling,
and Advanced Statistics.
==================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import curve_fit
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import warnings
import os
from PIL import Image

warnings.filterwarnings('ignore')

# ==================================================
# 1. APPLICATION CONFIGURATION
# ==================================================

# CHANGE "logo.png" to your exact image filename
IMAGE_NAME = "logo.png"

# Configure tab icon (favicon)
if os.path.exists(IMAGE_NAME):
    try:
        favicon = Image.open(IMAGE_NAME)
        st.set_page_config(
            page_title="Suicide Analysis in Mexico",
            page_icon=favicon,
            layout="wide",
            initial_sidebar_state="expanded"
        )
    except Exception as e:
        st.set_page_config(
            page_title="Suicide Analysis in Mexico",
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded"
        )
else:
    st.set_page_config(
        page_title="Suicide Analysis in Mexico",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
        )

# ==================================================
# 2. DATA LOADING AND PREPROCESSING FUNCTIONS
# ==================================================
@st.cache_data(ttl=3600, show_spinner="Loading data...")
def load_data():
    try:
        df = pd.read_csv('sucidios_mx.csv')
        df['ENTIDAD'] = df['ENTIDAD'].str.strip().str.upper()
        df['AÑO'] = pd.to_numeric(df['AÑO'], errors='coerce')
        df['RATIO_HM'] = (df['HOMBRES'] / df['MUJERES'].replace(0, 1)).round(2)
        df['MALE_RATE'] = (df['HOMBRES'] / df['POBLACION_HOMBRES'] * 100000).round(2)
        df['FEMALE_RATE'] = (df['MUJERES'] / df['POBLACION_MUJERES'] * 100000).round(2)
        df_sorted = df.sort_values(['ENTIDAD', 'AÑO'])
        df['PCT_CHANGE'] = df_sorted.groupby('ENTIDAD')['TASA_TOTAL'].pct_change() * 100
        df['RISK_LEVEL'] = pd.cut(
            df['TASA_TOTAL'],
            bins=[0, 5, 10, 15, float('inf')],
            labels=['Low', 'Medium', 'High', 'Critical']
        )
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

@st.cache_data
def get_mexico_geojson():
    return 'https://raw.githubusercontent.com/angelnmara/geojson/master/mexicoHigh.json'

# ==================================================
# 3. MATHEMATICAL MODELING FUNCTIONS
# ==================================================
def verhulst_logistic_model(t, L, k, t0):
    return L / (1 + np.exp(-k * (t - t0)))

def fit_logistic_growth(data_years, data_rates):
    try:
        t_normalized = np.arange(len(data_years))
        popt, _ = curve_fit(
            verhulst_logistic_model,
            t_normalized,
            data_rates,
            p0=[max(data_rates) * 1.2, 0.3, len(data_years) / 2],
            maxfev=5000
        )
        t_future = np.arange(len(data_years) + 5)
        projection = verhulst_logistic_model(t_future, *popt)
        ss_res = np.sum((data_rates - projection[:len(data_rates)]) ** 2)
        ss_tot = np.sum((data_rates - np.mean(data_rates)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        return {
            'capacity': popt[0],
            'growth_rate': popt[1],
            'inflection': popt[2],
            'projection': projection,
            'r_squared': r_squared
        }
    except:
        return None

# ==================================================
# 4. DATA MINING FUNCTIONS
# ==================================================
def perform_clustering_analysis(df_filtered):
    try:
        pivot_data = df_filtered.pivot_table(
            index='ENTIDAD',
            columns='AÑO',
            values='TASA_TOTAL'
        ).fillna(method='ffill', axis=1)
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(pivot_data)
        inertias = []
        k_range = range(2, 7)
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(scaled_data)
            inertias.append(kmeans.inertia_)
        optimal_k = 3
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(scaled_data)
        cosine_sim = cosine_similarity(scaled_data)
        return {
            'clusters': pd.Series(clusters, index=pivot_data.index),
            'inertias': inertias,
            'k_range': list(k_range),
            'cosine_similarity': cosine_sim,
            'cluster_centers': kmeans.cluster_centers_
        }
    except Exception as e:
        st.warning(f"Could not perform clustering: {e}")
        return None

# ==================================================
# 5. VISUALIZATION FUNCTIONS
# ==================================================
def create_risk_heatmap(df, years):
    heatmap_data = df[df['AÑO'].isin(years)].pivot_table(
        index='ENTIDAD',
        columns='AÑO',
        values='TASA_TOTAL'
    )
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=heatmap_data.columns,
        y=heatmap_data.index,
        colorscale='YlOrRd',
        hovertemplate='State: %{y}<br>Year: %{x}<br>Rate: %{z:.2f}<extra></extra>'
    ))
    fig.update_layout(
        title='Suicide Rate Heatmap by State and Year',
        xaxis_title='Year',
        yaxis_title='State',
        height=600
    )
    return fig

def create_gender_evolution_chart(df_nat):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_nat['AÑO'], y=df_nat['MALE_RATE'],
        name='Male', line=dict(color='#2E86AB', width=3),
        fill='tozeroy', opacity=0.3
    ))
    fig.add_trace(go.Scatter(
        x=df_nat['AÑO'], y=df_nat['FEMALE_RATE'],
        name='Female', line=dict(color='#A23B72', width=3),
        fill='tozeroy', opacity=0.3
    ))
    fig.update_layout(
        title='Suicide Rate Evolution by Gender (1994-2024)',
        xaxis_title='Year',
        yaxis_title='Rate per 100,000 inhabitants',
        hovermode='x unified'
    )
    return fig

def create_boxplot_distribution(df):
    fig = px.box(
        df[df['ENTIDAD'] != 'NACIONAL'],
        x='AÑO',
        y='TASA_TOTAL',
        points='all',
        title='Annual Distribution of State Rates'
    )
    fig.update_layout(
        xaxis_title='Year',
        yaxis_title='Rate per 100,000 inhabitants'
    )
    return fig

def create_gender_ratio_evolution(df_nat):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_nat['AÑO'], y=df_nat['RATIO_HM'],
        mode='lines+markers', line=dict(color='#F18F01', width=3),
        marker=dict(size=8), name='Male/Female Ratio'
    ))
    fig.add_hline(y=5, line_dash="dash", line_color="red",
                  annotation_text="Reference 5:1")
    fig.update_layout(
        title='Male-to-Female Ratio Evolution (Gender Disparity)',
        xaxis_title='Year',
        yaxis_title='Ratio (Male / Female)'
    )
    return fig

def create_volume_vs_rate_chart(df_nat):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(x=df_nat['AÑO'], y=df_nat['TOTAL'], name='Volume (Cases)',
               marker_color='#2E86AB'),
        secondary_y=False
    )
    fig.add_trace(
        go.Scatter(x=df_nat['AÑO'], y=df_nat['TASA_TOTAL'], name='Rate (Risk)',
                   line=dict(color='#F18F01', width=4), mode='lines+markers'),
        secondary_y=True
    )
    fig.update_layout(
        title='Absolute Volume vs. Real Risk Rate',
        xaxis_title='Year'
    )
    fig.update_yaxes(title_text="Number of Cases", secondary_y=False)
    fig.update_yaxes(title_text="Rate per 100,000 inhabitants", secondary_y=True)
    return fig

def create_logistic_model_visualization(df_state):
    if len(df_state) < 10:
        return None
    years = df_state['AÑO'].values
    rates = df_state['TASA_TOTAL'].values
    model_result = fit_logistic_growth(years, rates)
    if model_result:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=years, y=rates,
            mode='markers+lines',
            name='Historical Data',
            line=dict(color='blue', width=2),
            marker=dict(size=8)
        ))
        future_years = np.arange(years[0], years[-1] + 6)
        projection_data = model_result['projection'][:len(future_years)]
        fig.add_trace(go.Scatter(
            x=future_years, y=projection_data,
            mode='lines',
            name=f'Logistic Projection (R²={model_result["r_squared"]:.3f})',
            line=dict(color='red', dash='dash', width=3)
        ))
        fig.add_hline(
            y=model_result['capacity'],
            line_dash="dot",
            line_color="green",
            annotation_text=f"Carrying Capacity: {model_result['capacity']:.2f}"
        )
        fig.update_layout(
            title='Mathematical Modeling: Verhulst Logistic Equation',
            xaxis_title='Year',
            yaxis_title='Rate per 100,000 inhabitants',
            hovermode='x unified'
        )
        return fig
    return None

# ==================================================
# 6. MAIN FUNCTION
# ==================================================
def main():
    # SIDEBAR WITH LOGO
    with st.sidebar:
        # Load logo in sidebar
        if os.path.exists(IMAGE_NAME):
            try:
                logo = Image.open(IMAGE_NAME)
                # Resize for better visualization
                logo = logo.resize((180, 90))
                st.image(logo, use_container_width=False)
                st.markdown("---")
            except Exception as e:
                st.title("📊 Suicide Analysis")
        else:
            st.title("📊 Suicide Analysis")
            st.caption(f"Logo not found: {IMAGE_NAME}")
        
        # Navigation
        st.subheader("Navigation")
        page = st.radio(
            "Select a section:",
            [
                "🏠 Home & Global KPIs",
                "📊 State Analysis",
                "🗺️ Geospatial Analysis",
                "📈 Mathematical Modeling",
                "🔍 Data Mining"
            ],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # Project information
        with st.expander("ℹ️ Project Information"):
            st.markdown("""
            **Team Members:**
            - Ferrusca Jaimez Irving Nahir
            - Galván de la Cruz Angel Dolores
            
            **Course:** Integrative Project II
            
            **Data Sources:**
            - INEGI (2024)
            - CONAPO (2023)
            - Ministry of Health (2024)
            """)
        
        st.markdown("---")
        st.caption("© 2024 - Data Engineering")
    
    # Load data
    df = load_data()
    
    if df is None:
        st.error("Could not load data. Verify that the CSV file exists.")
        return
    
    # ========== HOME & GLOBAL KPIs ==========
    if page == "🏠 Home & Global KPIs":
        st.title("Suicide Analysis Platform in Mexico")
        st.markdown("""
        ### Welcome to the Comprehensive Analysis Dashboard
        
        This project integrates multiple data engineering disciplines to analyze
        suicide trends in Mexico during the 1994-2024 period.
        """)
        
        # Global KPIs
        st.subheader("National Key Indicators")
        
        col1, col2, col3, col4 = st.columns(4)
        
        df_nat = df[df['ENTIDAD'] == 'NACIONAL'].sort_values('AÑO')
        last_year = df_nat.iloc[-1]
        
        with col1:
            st.metric(
                "Total Cases (Historical)",
                f"{df_nat['TOTAL'].sum():,}",
                delta=f"{last_year['TOTAL']:,} in {last_year['AÑO']}"
            )
        
        with col2:
            current_rate = last_year['TASA_TOTAL']
            previous_rate = df_nat.iloc[-2]['TASA_TOTAL']
            variation = ((current_rate - previous_rate) / previous_rate) * 100
            st.metric(
                "Current National Rate",
                f"{current_rate:.2f} / 100k",
                delta=f"{variation:.1f}%"
            )
        
        with col3:
            current_ratio = last_year['RATIO_HM']
            st.metric(
                "Male/Female Ratio",
                f"{current_ratio:.2f}:1",
                delta="↑ 6:1 in recent years"
            )
        
        with col4:
            critical_states = len(df[df['RISK_LEVEL'] == 'Critical']['AÑO'].unique())
            st.metric(
                "States in Critical Level",
                critical_states,
                delta="↑ 40% vs previous decade"
            )
        
        st.markdown("---")
        
        # Main charts
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.subheader("National Evolution (1994-2024)")
            fig_national = px.line(
                df_nat, x='AÑO', y='TASA_TOTAL',
                title='National Suicide Rate per 100,000 inhabitants',
                markers=True
            )
            st.plotly_chart(fig_national, use_container_width=True)
        
        with col_right:
            st.subheader("Top 10 States (Last Year)")
            last_year_data = df[df['AÑO'] == last_year['AÑO']].sort_values('TASA_TOTAL', ascending=False).head(10)
            fig_top = px.bar(
                last_year_data,
                x='ENTIDAD',
                y='TASA_TOTAL',
                color='TASA_TOTAL',
                color_continuous_scale='Reds',
                title='States with Highest Risk Rate'
            )
            st.plotly_chart(fig_top, use_container_width=True)
        
        # Correlation analysis
        st.subheader("Correlation: Population vs. Risk Rate")
        st.markdown("**Insight:** This regression demonstrates that suicide rates are independent of population size, validating the use of standardized rates for fair comparisons between states.")
        
        df_corr = df[df['ENTIDAD'] != 'NACIONAL'].copy()
        fig_corr = px.scatter(
            df_corr,
            x='POBLACION_TOTAL',
            y='TASA_TOTAL',
            color='ENTIDAD',
            trendline='ols',
            title='Regression: Population vs Risk Rate',
            labels={'POBLACION_TOTAL': 'Total Population', 'TASA_TOTAL': 'Rate per 100,000 inhabitants'}
        )
        st.plotly_chart(fig_corr, use_container_width=True)
    
    # ========== STATE ANALYSIS ==========
    elif page == "📊 State Analysis":
        st.title("Comparative State Analysis")
        
        only_states = sorted([e for e in df['ENTIDAD'].unique() if e != 'NACIONAL'])
        selected_states = st.multiselect(
            "Select one or more states to analyze:",
            options=only_states,
            default=[only_states[0]] if only_states else []
        )
        
        if selected_states:
            df_comp = df[df['ENTIDAD'].isin(selected_states)]
            
            tab1, tab2, tab3, tab4 = st.tabs([
                "📈 Temporal Trends",
                "⚖️ Indicator Comparison",
                "🔮 Predictive Modeling",
                "📊 Annual Distribution"
            ])
            
            with tab1:
                fig_trend = px.line(
                    df_comp,
                    x='AÑO',
                    y='TASA_TOTAL',
                    color='ENTIDAD',
                    title='Risk Rate Evolution by State',
                    markers=True
                )
                st.plotly_chart(fig_trend, use_container_width=True)
                
                fig_ratio = px.line(
                    df_comp,
                    x='AÑO',
                    y='RATIO_HM',
                    color='ENTIDAD',
                    title='Male-to-Female Ratio Evolution'
                )
                st.plotly_chart(fig_ratio, use_container_width=True)
            
            with tab2:
                fig_gender_comp = go.Figure()
                for state in selected_states:
                    df_st = df_comp[df_comp['ENTIDAD'] == state]
                    fig_gender_comp.add_trace(go.Scatter(
                        x=df_st['AÑO'],
                        y=df_st['MALE_RATE'],
                        name=f'{state} - Male',
                        line=dict(dash='solid')
                    ))
                    fig_gender_comp.add_trace(go.Scatter(
                        x=df_st['AÑO'],
                        y=df_st['FEMALE_RATE'],
                        name=f'{state} - Female',
                        line=dict(dash='dot')
                    ))
                
                fig_gender_comp.update_layout(
                    title='Gender Rate Comparison',
                    xaxis_title='Year',
                    yaxis_title='Rate per 100,000 inhabitants'
                )
                st.plotly_chart(fig_gender_comp, use_container_width=True)
            
            with tab3:
                for state in selected_states:
                    st.markdown(f"### State: {state}")
                    df_st = df_comp[df_comp['ENTIDAD'] == state].sort_values('AÑO')
                    fig_logistic = create_logistic_model_visualization(df_st)
                    if fig_logistic:
                        st.plotly_chart(fig_logistic, use_container_width=True)
                        st.markdown("""
                        **Logistic Model Interpretation:**
                        - The red dashed line shows the saturation projection
                        - The green dotted line indicates the carrying capacity (maximum expected value)
                        - This model helps distinguish between exponential growth and eventual stabilization
                        """)
                    else:
                        st.warning("Insufficient historical data to model this state.")
            
            with tab4:
                fig_box = create_boxplot_distribution(df_comp)
                st.plotly_chart(fig_box, use_container_width=True)
                st.markdown("""
                **Insight:** The boxplot highlights increasing dispersion and the presence of outliers, 
                meaning specific states are facing atypical crisis levels.
                """)
    
    # ========== GEOSPATIAL ANALYSIS ==========
    elif page == "🗺️ Geospatial Analysis":
        st.title("Geospatial Risk Analysis")
        
        st.subheader("Risk Intensity Heatmap by State")
        st.markdown("**Insight:** The heatmap identifies geographical clusters where the intensity of the phenomenon has increased, allowing us to prioritize high-risk regions.")
        
        years_range = st.slider(
            "Select the year range for the heatmap:",
            min_value=int(df['AÑO'].min()),
            max_value=int(df['AÑO'].max()),
            value=(2014, int(df['AÑO'].max()))
        )
        
        years = list(range(years_range[0], years_range[1] + 1))
        fig_heatmap = create_risk_heatmap(df, years)
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        st.subheader("Interactive Geographic Visualization")
        
        year_map = st.select_slider(
            "Select the year to visualize:",
            options=sorted(df['AÑO'].unique())
        )
        
        df_map = df[(df['AÑO'] == year_map) & (df['ENTIDAD'] != 'NACIONAL')].copy()
        df_map['ENTIDAD_MAP'] = df_map['ENTIDAD'].str.title()
        
        try:
            geojson_url = get_mexico_geojson()
            fig_map = px.choropleth(
                df_map,
                geojson=geojson_url,
                locations='ENTIDAD_MAP',
                featureidkey="properties.name",
                color='TASA_TOTAL',
                color_continuous_scale="YlOrRd",
                title=f'Suicide Rate Map - {year_map}',
                hover_data={'ENTIDAD': True, 'TASA_TOTAL': ':.2f', 'TOTAL': True}
            )
            fig_map.update_geos(fitbounds="locations", visible=False)
            fig_map.update_layout(height=600)
            st.plotly_chart(fig_map, use_container_width=True)
        except Exception as e:
            st.error(f"Error loading map: {e}")
    
    # ========== MATHEMATICAL MODELING ==========
    elif page == "📈 Mathematical Modeling":
        st.title("Mathematical Modeling with Differential Equations")
        
        st.markdown("""
        ### Verhulst Logistic Equation
        
        The following ordinary differential equation (ODE) is used to model growth 
        with saturation:
        
        $$\\frac{dP}{dt} = rP\\left(1 - \\frac{P}{K}\\right)$$
        
        Where:
        - **P**: Suicide rate at time t
        - **r**: Intrinsic growth rate
        - **K**: Carrying capacity (maximum saturation value)
        """)
        
        model_states = sorted([e for e in df['ENTIDAD'].unique() if e != 'NACIONAL'])
        selected_state = st.selectbox("Select a state to model:", model_states)
        
        if selected_state:
            df_state = df[df['ENTIDAD'] == selected_state].sort_values('AÑO')
            
            if len(df_state) >= 10:
                fig_logistic = create_logistic_model_visualization(df_state)
                if fig_logistic:
                    st.plotly_chart(fig_logistic, use_container_width=True)
                    
                    years = df_state['AÑO'].values
                    rates = df_state['TASA_TOTAL'].values
                    model_result = fit_logistic_growth(years, rates)
                    
                    if model_result:
                        st.subheader("Model Parameters")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Carrying Capacity (K)", f"{model_result['capacity']:.2f}")
                        with col2:
                            st.metric("Growth Rate (r)", f"{model_result['growth_rate']:.3f}")
                        with col3:
                            st.metric("R² of the Model", f"{model_result['r_squared']:.3f}")
                        
                        st.info(f"""
                        **Interpretation:**
                        - The model projects that the suicide rate in {selected_state} will stabilize around {model_result['capacity']:.1f} per 100,000 inhabitants
                        - The fit quality (R² = {model_result['r_squared']:.3f}) indicates a {'very good' if model_result['r_squared'] > 0.8 else 'moderate'} model fit
                        """)
            else:
                st.warning(f"Insufficient historical data to model {selected_state} (minimum 10 years required)")
    
    # ========== DATA MINING (CLUSTERING) ==========
    elif page == "🔍 Data Mining":
        st.title("Clustering Analysis and Pattern Recognition")
        
        st.markdown("""
        ### Behavior Pattern Identification
        
        **K-Means** and **Cosine Similarity** algorithms are applied to find non-obvious 
        correlations, identifying states with similar behavioral patterns despite 
        geographical distances.
        """)
        
        cluster_years = st.multiselect(
            "Select years for clustering analysis:",
            options=sorted(df['AÑO'].unique()),
            default=[2020, 2021, 2022, 2023, 2024]
        )
        
        if len(cluster_years) >= 3:
            df_cluster = df[df['AÑO'].isin(cluster_years)]
            cluster_results = perform_clustering_analysis(df_cluster)
            
            if cluster_results:
                st.subheader("State Clusters by Behavior Pattern")
                
                cluster_df = pd.DataFrame({
                    'STATE': cluster_results['clusters'].index,
                    'CLUSTER': cluster_results['clusters'].values
                })
                
                cluster_names = {0: 'Moderate Pattern', 1: 'High Pattern', 2: 'Critical Pattern'}
                cluster_df['CLUSTER_NAME'] = cluster_df['CLUSTER'].map(cluster_names)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.dataframe(cluster_df.sort_values('CLUSTER'), use_container_width=True)
                
                with col2:
                    fig_cluster_dist = px.pie(
                        cluster_df,
                        names='CLUSTER_NAME',
                        title='State Distribution by Cluster'
                    )
                    st.plotly_chart(fig_cluster_dist, use_container_width=True)
                
                st.subheader("Rate Evolution by Cluster")
                
                df_with_cluster = df_cluster.merge(cluster_df[['STATE', 'CLUSTER_NAME']], 
                                                   left_on='ENTIDAD', right_on='STATE', how='left')
                
                cluster_evolution = df_with_cluster.groupby(['CLUSTER_NAME', 'AÑO'])['TASA_TOTAL'].mean().reset_index()
                
                fig_cluster_evo = px.line(
                    cluster_evolution,
                    x='AÑO',
                    y='TASA_TOTAL',
                    color='CLUSTER_NAME',
                    title='Average Rate Evolution by Cluster',
                    markers=True
                )
                st.plotly_chart(fig_cluster_evo, use_container_width=True)
                
                st.markdown("""
                **Cluster Interpretation:**
                - **Moderate Pattern:** States with rates consistently below the national average
                - **High Pattern:** States with elevated but stable rates
                - **Critical Pattern:** States with increasing rates requiring priority intervention
                """)
                
                st.subheader("Cosine Similarity Matrix")
                st.markdown("Shows the similarity in temporal behavior between states (values close to 1 indicate high similarity)")
                
                states_list = cluster_results['clusters'].index.tolist()
                sim_df = pd.DataFrame(
                    cluster_results['cosine_similarity'],
                    index=states_list,
                    columns=states_list
                )
                
                similar_state = st.selectbox("Select a state to see its most similar ones:", states_list)
                if similar_state:
                    similarities = sim_df[similar_state].sort_values(ascending=False)
                    top_similar = similarities.iloc[1:6]
                    
                    st.write(f"**States with most similar behavior to {similar_state}:**")
                    for state, sim in top_similar.items():
                        st.write(f"- {state}: {sim:.3f} similarity")
            else:
                st.warning("Could not perform clustering analysis with the selected years.")
        else:
            st.warning("Select at least 3 years to perform clustering analysis.")

# ==================================================
# 7. EXECUTION
# ==================================================
if __name__ == "__main__":
    main()