import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="NCAA Volleyball Dashboard",
    page_icon="🏐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to enhance the look and feel
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #273469;
        text-align: center;
        margin-bottom: 1rem;
        padding-bottom: 1rem;
        border-bottom: 2px solid #273469;
        font-weight: 700;
    }
    
    .sub-header {
        font-size: 1.8rem;
        color: #273469;
        margin-top: 2rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #aaa;
    }
    
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        color: #273469;
    }
    
    .metric-label {
        font-size: 1rem;
        color: #666;
    }
    
    .highlight {
        background-color: #f0f7ff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #273469;
        margin: 1rem 0;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f7ff;
        border-radius: 4px 4px 0px 0px;
        padding: 10px 20px;
        border: none;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #273469;
        color: white;
    }
    
    footer {
        margin-top: 3rem;
        text-align: center;
        color: #666;
        font-size: 0.8rem;
    }
    
    /* Simple volleyball court using CSS */
    .volleyball-court {
        background-color: #f0f7ff;
        border: 3px solid #273469;
        border-radius: 5px;
        width: 100%;
        height: 300px;
        position: relative;
        margin-bottom: 20px;
    }
    
    .volleyball-net {
        position: absolute;
        top: 50%;
        left: 0;
        right: 0;
        height: 5px;
        background-color: #C75D68;
        transform: translateY(-50%);
    }
    
    .volleyball-line-1, .volleyball-line-2 {
        position: absolute;
        left: 0;
        right: 0;
        height: 2px;
        background-color: #273469;
        border-style: dashed;
    }
    
    .volleyball-line-1 {
        top: 33%;
    }
    
    .volleyball-line-2 {
        top: 67%;
    }
    
    .volleyball-icon {
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        font-size: 50px;
    }
</style>
""", unsafe_allow_html=True)

# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv("data/volleyball_ncaa_div1_2022_23.csv")
    
    # Fix potential data type issues - convert Conference to string
    df['Conference'] = df['Conference'].astype(str)
    
    # Fix any NaN values in numeric columns
    numeric_cols = ['hitting_pctg', 'kills_per_set', 'blocks_per_set', 
                    'aces_per_set', 'win_loss_pctg', 'opp_hitting_pctg']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(0)
    
    return df

try:
    vb = load_data()
    
    # Handle missing values and other potential data issues
    if 'Conference' not in vb.columns:
        st.error("Dataset is missing the 'Conference' column!")
        st.stop()
    
    # Remove "nan" or empty values from Conference if they exist
    vb = vb[vb['Conference'] != 'nan']
    vb = vb[vb['Conference'] != '']
    
    # Sidebar
    with st.sidebar:
        # Simple volleyball court with HTML/CSS instead of Plotly
        st.markdown("""
        <div class="volleyball-court">
            <div class="volleyball-net"></div>
            <div class="volleyball-line-1"></div>
            <div class="volleyball-line-2"></div>
            <div class="volleyball-icon">🏐</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### Dashboard Settings")
        
        # Conference filter - Convert all values to strings before sorting
        all_conferences = sorted(vb['Conference'].astype(str).unique().tolist())
        
        # Filter out any problematic conference values
        all_conferences = [conf for conf in all_conferences if conf != 'nan' and conf != 'None']
        
        selected_conferences = st.multiselect(
            "Select Conferences to Analyze:",
            options=all_conferences,
            default=all_conferences[:5] if len(all_conferences) >= 5 else all_conferences
        )
        
        # Metric selection
        selected_metric = st.selectbox(
            "Primary Analysis Metric:",
            options=["kills_per_set", "blocks_per_set", "hitting_pctg", "win_loss_pctg", "aces_per_set"],
            index=0,
            format_func=lambda x: {
                "kills_per_set": "Kills per Set",
                "blocks_per_set": "Blocks per Set",
                "hitting_pctg": "Hitting Percentage",
                "win_loss_pctg": "Win-Loss Percentage",
                "aces_per_set": "Aces per Set"
            }[x]
        )
        
        # Top N teams slider
        top_n = st.slider("Number of Top Teams to Display:", min_value=5, max_value=20, value=10, step=1)
        
        st.markdown("---")
        
        # Add some volleyball facts in the sidebar
        st.markdown("### NCAA Volleyball Facts")
        facts = [
            "The NCAA Women's Volleyball Championship was first held in 1981.",
            "Stanford has won the most NCAA Division I women's volleyball championships.",
            "Rally scoring was implemented in NCAA volleyball in 2001.",
            "Liberos were introduced to NCAA volleyball in 2002.",
            "The NCAA volleyball net height for women is 7 feet, 4 1/8 inches."
        ]
        st.info(facts[np.random.randint(0, len(facts))])
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("This dashboard analyzes NCAA Division I Women's Volleyball stats for the 2022-23 season.")

    # Filter data based on sidebar selections
    if selected_conferences:
        filtered_vb = vb[vb['Conference'].isin(selected_conferences)]
    else:
        filtered_vb = vb

    # Main content area
    st.markdown('<h1 class="main-header">NCAA Division I Women\'s Volleyball Dashboard</h1>', unsafe_allow_html=True)

    # Top metrics overview
    metric_cols = st.columns(4)

    # Calculate key metrics
    avg_hitting = filtered_vb['hitting_pctg'].mean()
    avg_kills = filtered_vb['kills_per_set'].mean()
    avg_blocks = filtered_vb['blocks_per_set'].mean()
    avg_win_pct = filtered_vb['win_loss_pctg'].mean()

    with metric_cols[0]:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{avg_hitting:.3f}</div>
            <div class="metric-label">Avg. Hitting %</div>
        </div>
        """, unsafe_allow_html=True)

    with metric_cols[1]:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{avg_kills:.2f}</div>
            <div class="metric-label">Avg. Kills/Set</div>
        </div>
        """, unsafe_allow_html=True)

    with metric_cols[2]:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{avg_blocks:.2f}</div>
            <div class="metric-label">Avg. Blocks/Set</div>
        </div>
        """, unsafe_allow_html=True)

    with metric_cols[3]:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{avg_win_pct:.1%}</div>
            <div class="metric-label">Avg. Win Percentage</div>
        </div>
        """, unsafe_allow_html=True)

    # Create tabs for different visualizations
    tab1, tab2, tab3 = st.tabs(["Team Rankings", "Conference Analysis", "Team Performance"])

    with tab1:
        st.markdown('<h2 class="sub-header">Top Teams Analysis</h2>', unsafe_allow_html=True)
        
        # Sort data based on selected metric
        metric_name_map = {
            "kills_per_set": "Kills per Set",
            "blocks_per_set": "Blocks per Set",
            "hitting_pctg": "Hitting Percentage",
            "win_loss_pctg": "Win-Loss Percentage",
            "aces_per_set": "Aces per Set"
        }
        
        # Get color scale based on metric
        color_scales = {
            "kills_per_set": "blues",
            "blocks_per_set": "greens",
            "hitting_pctg": "reds",
            "win_loss_pctg": "purples",
            "aces_per_set": "oranges"
        }
        
        top_teams = filtered_vb[['Team', 'Conference', selected_metric]].sort_values(
            by=selected_metric, ascending=False
        ).head(top_n)
        
        # Two columns layout
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Create bar chart with Plotly
            fig = px.bar(
                top_teams,
                x=selected_metric,
                y='Team',
                orientation='h',
                color=selected_metric,
                color_continuous_scale=color_scales[selected_metric],
                text=selected_metric,
                labels={selected_metric: metric_name_map[selected_metric], 'Team': ''},
                height=600,
                hover_data=['Conference']
            )
            
            # Customize layout
            fig.update_layout(
                title=f'Top {top_n} Teams by {metric_name_map[selected_metric]}',
                xaxis_title=metric_name_map[selected_metric],
                yaxis={'categoryorder': 'total ascending'},
                hoverlabel=dict(bgcolor="white", font_size=14),
                title_font=dict(size=24, color="#273469"),
                margin=dict(l=0, r=20, t=80, b=20),
            )
            
            # Format the text values displayed on the bars
            if selected_metric == 'hitting_pctg' or selected_metric == 'win_loss_pctg':
                fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
            else:
                fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
            
            # Display the plotly chart
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### Team Details")
            st.dataframe(
                top_teams.rename(columns={
                    selected_metric: metric_name_map[selected_metric]
                }),
                use_container_width=True,
                hide_index=True
            )
            
            # Add a metric interpretation
            st.markdown('<div class="highlight">', unsafe_allow_html=True)
            if selected_metric == 'kills_per_set':
                st.markdown("**Insight:** Teams with higher kills per set generally demonstrate more offensive firepower. The NCAA D1 average is typically around 13-14 kills per set.")
            elif selected_metric == 'blocks_per_set':
                st.markdown("**Insight:** Strong blocking teams often have tall middle blockers. Top NCAA D1 teams average around 2.5-3.0 blocks per set.")
            elif selected_metric == 'hitting_pctg':
                st.markdown("**Insight:** Hitting percentage measures attack efficiency. Elite teams typically maintain hitting percentages above .250.")
            elif selected_metric == 'win_loss_pctg':
                st.markdown("**Insight:** Win percentage directly reflects team success, but doesn't always correlate with specific statistical strengths.")
            elif selected_metric == 'aces_per_set':
                st.markdown("**Insight:** Serving aces provide easy points and disrupt opponent passing. Top serving teams average 1.5-2.0 aces per set.")
            st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        st.markdown('<h2 class="sub-header">Conference Analysis</h2>', unsafe_allow_html=True)
        
        # Create conference aggregated data
        try:
            conf_data = filtered_vb.groupby('Conference').agg({
                'hitting_pctg': 'mean',
                'kills_per_set': 'mean',
                'blocks_per_set': 'mean',
                'aces_per_set': 'mean',
                'win_loss_pctg': 'mean',
                'Team': 'count'
            }).reset_index().rename(columns={'Team': 'Number_of_Teams'})
            
            conf_data = conf_data.sort_values(by=selected_metric, ascending=False)
            
            # Two columns layout
            col1, col2 = st.columns([3, 2])
            
            with col1:
                # Create a radar chart to compare conferences
                fig = go.Figure()
                
                # Normalize the data for radar chart
                radar_data = conf_data.copy()
                for col in ['hitting_pctg', 'kills_per_set', 'blocks_per_set', 'aces_per_set', 'win_loss_pctg']:
                    min_val = radar_data[col].min()
                    max_val = radar_data[col].max()
                    if max_val > min_val:  # Prevent division by zero
                        radar_data[col] = (radar_data[col] - min_val) / (max_val - min_val)
                    else:
                        radar_data[col] = 0.5  # Default value if all values are the same
                
                # Add traces for top 5 conferences (or fewer if not enough)
                max_conferences = min(5, len(radar_data))
                for i, (idx, row) in enumerate(radar_data.head(max_conferences).iterrows()):
                    fig.add_trace(go.Scatterpolar(
                        r=[row['hitting_pctg'], row['kills_per_set'], row['blocks_per_set'], 
                           row['aces_per_set'], row['win_loss_pctg'], row['hitting_pctg']],
                        theta=['Hitting %', 'Kills/Set', 'Blocks/Set', 'Aces/Set', 'Win %', 'Hitting %'],
                        fill='toself',
                        name=row['Conference'],
                        line_color=px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]
                    ))
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 1]
                        )
                    ),
                    title="Top Conferences Statistical Comparison",
                    showlegend=True,
                    title_font=dict(size=24, color="#273469"),
                    height=600
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Create box plot using Plotly
                fig = px.box(
                    filtered_vb, 
                    x='Conference', 
                    y='win_loss_pctg',
                    color='Conference',
                    title="Win Percentage Distribution by Conference",
                    labels={'win_loss_pctg': 'Win Percentage', 'Conference': ''},
                    height=600
                )
                
                fig.update_layout(
                    xaxis={'categoryorder': 'mean descending'},
                    showlegend=False,
                    title_font=dict(size=24, color="#273469"),
                    yaxis=dict(range=[0, 1]),
                    xaxis_tickangle=-45
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Conference stats table
            st.markdown("### Conference Performance Metrics")
            
            # Format the table
            display_conf_data = conf_data.copy()
            display_conf_data['hitting_pctg'] = display_conf_data['hitting_pctg'].apply(lambda x: f"{x:.3f}")
            display_conf_data['kills_per_set'] = display_conf_data['kills_per_set'].apply(lambda x: f"{x:.2f}")
            display_conf_data['blocks_per_set'] = display_conf_data['blocks_per_set'].apply(lambda x: f"{x:.2f}")
            display_conf_data['aces_per_set'] = display_conf_data['aces_per_set'].apply(lambda x: f"{x:.2f}")
            display_conf_data['win_loss_pctg'] = display_conf_data['win_loss_pctg'].apply(lambda x: f"{x:.1%}")
            
            # Rename columns for display
            display_conf_data = display_conf_data.rename(columns={
                'hitting_pctg': 'Hitting %',
                'kills_per_set': 'Kills/Set',
                'blocks_per_set': 'Blocks/Set',
                'aces_per_set': 'Aces/Set',
                'win_loss_pctg': 'Win %',
                'Number_of_Teams': 'Teams'
            })
            
            st.dataframe(display_conf_data, use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(f"Error with Conference Analysis: {e}")
            st.info("This error might be due to data issues with your Conference column. Please check your data.")

    with tab3:
        st.markdown('<h2 class="sub-header">Team Performance Analysis</h2>', unsafe_allow_html=True)
        
        # Safely check if all required columns exist
        required_cols = ['hitting_pctg', 'blocks_per_set', 'aces_per_set', 'opp_hitting_pctg']
        if all(col in filtered_vb.columns for col in required_cols):
            # Calculate dominance score
            filtered_vb['dominance_score'] = (
                filtered_vb['hitting_pctg'] * 0.4 +
                filtered_vb['blocks_per_set'] * 0.2 +
                filtered_vb['aces_per_set'] * 0.2 -
                filtered_vb['opp_hitting_pctg'] * 0.2
            )
            
            # Sort by dominance score
            top_dominant = filtered_vb[['Team', 'Conference', 'dominance_score', 'win_loss_pctg', 'hitting_pctg']].sort_values(
                by='dominance_score', ascending=False
            ).head(top_n)
            
            col1, col2 = st.columns([3, 2])
            
            with col1:
                # Create a scatter plot to show relationship between dominance and win percentage
                fig = px.scatter(
                    filtered_vb,
                    x='dominance_score',
                    y='win_loss_pctg',
                    color='Conference',
                    size='hitting_pctg',
                    size_max=15,
                    hover_name='Team',
                    labels={
                        'dominance_score': 'Dominance Score',
                        'win_loss_pctg': 'Win Percentage',
                        'hitting_pctg': 'Hitting Percentage'
                    },
                    title='Team Dominance Score vs. Win Percentage',
                    height=600
                )
                
                # Add a trendline
                fig.update_layout(
                    title_font=dict(size=24, color="#273469"),
                    legend_title_text='Conference',
                    xaxis=dict(title='Dominance Score', tickformat='.3f'),
                    yaxis=dict(title='Win Percentage', tickformat='.0%')
                )
                
                # Add a trendline
                fig.update_traces(marker=dict(line=dict(width=1, color='DarkSlateGrey')))
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Create a heat map showing correlation between metrics
                corr_metrics = ['win_loss_pctg', 'hitting_pctg', 'blocks_per_set', 'kills_per_set', 'aces_per_set', 'opp_hitting_pctg']
                corr_data = filtered_vb[corr_metrics].corr()
                
                # Use matplotlib/seaborn for the heatmap
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(
                    corr_data, 
                    annot=True, 
                    cmap='coolwarm', 
                    vmin=-1, 
                    vmax=1, 
                    fmt='.2f',
                    linewidths=0.5,
                    ax=ax
                )
                ax.set_title('Correlation Between Key Metrics', fontsize=18, color='#273469')
                plt.tight_layout()
                
                st.pyplot(fig)
                
                st.markdown("""
                <div class="highlight">
                <strong>Interpretation:</strong> The correlation matrix shows how different volleyball metrics relate to each other. 
                Strong positive correlations (close to 1.0) indicate that as one metric increases, the other tends to increase as well. 
                Negative correlations suggest an inverse relationship between metrics.
                </div>
                """, unsafe_allow_html=True)
            
            # Top dominant teams
            st.markdown("### Top Dominant Teams Analysis")
            
            # Add a description of the dominance score
            st.markdown("""
            <div class="highlight">
            The <strong>Dominance Score</strong> is a composite metric that combines:
            <ul>
                <li>40% weighting on hitting percentage</li>
                <li>20% weighting on blocks per set</li>
                <li>20% weighting on aces per set</li>
                <li>20% negative weighting on opponent hitting percentage</li>
            </ul>
            This provides a holistic measure of team performance beyond simple win percentage.
            </div>
            """, unsafe_allow_html=True)
            
            # Create custom dominance rating chart
            fig = go.Figure()
            
            # Add bars
            fig.add_trace(go.Bar(
                x=top_dominant['dominance_score'],
                y=top_dominant['Team'],
                orientation='h',
                marker=dict(
                    color=top_dominant['dominance_score'],
                    colorscale='Viridis',
                    line=dict(color='rgba(58, 71, 80, 0.8)', width=1)
                ),
                hovertemplate='<b>%{y}</b><br>Dominance Score: %{x:.3f}<br>Win Pct: %{customdata:.1%}<extra></extra>',
                customdata=top_dominant['win_loss_pctg']
            ))
            
            # Add conference labels as annotations
            for i, (_, row) in enumerate(top_dominant.iterrows()):
                fig.add_annotation(
                    x=row['dominance_score'],
                    y=row['Team'],
                    text=row['Conference'],
                    showarrow=False,
                    xshift=10,
                    align="left",
                    xanchor="left",
                    font=dict(size=10),
                    bgcolor="rgba(255, 255, 255, 0.8)",
                    bordercolor="rgba(0, 0, 0, 0.3)",
                    borderwidth=1,
                    borderpad=2
                )
            
            # Update layout
            fig.update_layout(
                title='Top Dominant Teams (Feature Engineered Score)',
                xaxis_title='Dominance Score',
                yaxis={'categoryorder': 'total ascending'},
                height=600,
                title_font=dict(size=24, color="#273469"),
                margin=dict(l=10, r=150, t=80, b=40),
            )
            
            # Display the plot
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Missing required columns for dominance score calculation. Please check your data.")
            st.info(f"Required columns: {required_cols}")
            st.info(f"Available columns: {filtered_vb.columns.tolist()}")

    # Footer
    st.markdown("""
    <footer>
        <p>NCAA Women's Volleyball Dashboard | Data from 2022-23 Season</p>
        <p>Created with Streamlit and Plotly</p>
    </footer>
    """, unsafe_allow_html=True)

except Exception as e:
    st.error(f"An error occurred while loading the dashboard: {e}")
    st.info("This could be due to data formatting issues. Please check your CSV file.")
    st.markdown("""
    ### Troubleshooting Tips:
    1. Make sure your CSV file is in the correct location (`data/volleyball_ncaa_div1_2022_23.csv`)
    2. Check that all required columns exist: Team, Conference, hitting_pctg, kills_per_set, blocks_per_set, etc.
    3. Ensure numeric columns contain only numbers (no text values)
    4. Look for missing values in your dataset
    """)