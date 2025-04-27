import pandas as pd
import streamlit as st
# IMPORTANT: This must be the first Streamlit command
st.set_page_config(
    page_title="NCAA Volleyball Dashboard",
    page_icon="🏐",
    layout="wide",
    initial_sidebar_state="expanded"
)
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import numpy as np

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
    
    .filter-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
        border: 1px solid #e9ecef;
    }
    
    .volleyball-image-container {
        width: 100%;
        margin-bottom: 20px;
        text-align: center;
    }
    
    .volleyball-image {
        max-width: 100%;
        border-radius: 5px;
        border: 2px solid #273469;
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
    
    # Sidebar for general dashboard settings and information
    with st.sidebar:
        # Try multiple approaches to display the image
        # Approach 1: Using Streamlit's native image function
        try:
            from PIL import Image
            import os
            # Try multiple possible paths
            image_paths = [
                "data/volleyballWomens.png",
                "static/volleyballWomens.png",
                "assets/volleyballWomens.png",
                "images/volleyballWomens.png",
                "./volleyballWomens.png"
            ]
            
            image_loaded = False
            for path in image_paths:
                if os.path.exists(path):
                    volleyball_image = Image.open(path)
                    st.image(volleyball_image, caption="Women's Volleyball", use_column_width=True)
                    image_loaded = True
                    break
            
            if not image_loaded:
                # Fallback to basic text if image fails to load
                st.markdown("🏐 **NCAA Women's Volleyball**")
                st.info("Image could not be found. Please check the file path.")
        except Exception as e:
            st.error(f"Error loading image: {e}")
            # Fallback to basic text if image fails to load
            st.markdown("🏐 **NCAA Women's Volleyball**")
        
        # Conference filter - moved the main analysis controls to the Team Rankings tab
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
        
        # Move the metric selection and top N teams to this tab
        st.markdown('<div class="filter-container">', unsafe_allow_html=True)
        filter_cols = st.columns([1, 1])
        with filter_cols[0]:
            # Metric selection - moved from sidebar
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
            
        with filter_cols[1]:
            # Top N teams slider - moved from sidebar
            top_n = st.slider("Number of Top Teams to Display:", min_value=5, max_value=20, value=10, step=1)
        st.markdown('</div>', unsafe_allow_html=True)
        
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
                fig.update_traces()
            else:
                fig.update_traces()
            
            # Display the plotly chart
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### Team Details")
            # For older versions of Streamlit, use the regular dataframe method without hide_index
            display_df = top_teams.rename(columns={
                selected_metric: metric_name_map[selected_metric]
            }).reset_index(drop=True)
            st.dataframe(display_df, use_container_width=True)
            
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
        
        try:
            # Create conference aggregated data
            conf_data = filtered_vb.groupby('Conference').agg({
                'hitting_pctg': 'mean',
                'kills_per_set': 'mean',
                'blocks_per_set': 'mean',
                'aces_per_set': 'mean',
                'win_loss_pctg': 'mean',
                'Team': 'count'
            }).reset_index().rename(columns={'Team': 'Number_of_Teams'})
            
            # Use the selected metric from the Team Rankings tab for sorting
            conf_data = conf_data.sort_values(by=selected_metric, ascending=False)
            
            # Create a better radar chart that uses actual values
            fig = go.Figure()
            
            # Define colors
            plotly_colors = px.colors.qualitative.Plotly
            
            # Get top 5 conferences (or fewer if not enough)
            max_conferences = min(5, len(conf_data))
            
            # For each conference, add a trace with PROPERLY SCALED values
            for i, (_, row) in enumerate(conf_data.head(max_conferences).iterrows()):
                # Better scaling method:
                # Instead of min-max normalization, we divide by predefined maximums
                # This preserves the relative differences between values
                
                # Define reasonable max values for scaling (slightly above actual max values)
                max_values = {
                    'kills_per_set': 15.0,      # Max is around 13, so 15 gives headroom
                    'hitting_pctg': 0.3,        # Max is around 0.23, so 0.3 gives headroom
                    'blocks_per_set': 3.0,      # Max is around 2.3, so 3 gives headroom
                    'aces_per_set': 2.0,        # Max is around 1.6, so 2 gives headroom
                    'win_loss_pctg': 1.0        # This is already 0-1 scale
                }
                
                # Create scaled values that maintain proper relationships
                scaled_values = [
                    row['kills_per_set'] / max_values['kills_per_set'],
                    row['hitting_pctg'] / max_values['hitting_pctg'],
                    row['blocks_per_set'] / max_values['blocks_per_set'],
                    row['aces_per_set'] / max_values['aces_per_set'],
                    row['win_loss_pctg'] / max_values['win_loss_pctg'],
                    row['kills_per_set'] / max_values['kills_per_set']
                ]
                
                fig.add_trace(go.Scatterpolar(
                    r=scaled_values,
                    theta=['Kills/Set', 'Hitting %', 'Blocks/Set', 'Aces/Set', 'Win %', 'Kills/Set'],
                    fill='toself',
                    name=row['Conference'],
                    line_color=plotly_colors[i % len(plotly_colors)],
                    # Add hover data with actual values
                    hovertemplate='<b>%{theta}</b><br>' +
                                 '<b>Conference:</b> ' + row['Conference'] + '<br>' +
                                 '<b>Value:</b> %{customdata:.2f}<extra></extra>',
                    customdata=[
                        row['kills_per_set'],
                        row['hitting_pctg'],
                        row['blocks_per_set'],
                        row['aces_per_set'],
                        row['win_loss_pctg'],
                        row['kills_per_set']
                    ]
                ))
            
            # Add reference circles with value labels
            # These concentric circles provide context for the scales
            reference_circle_values = [0.2, 0.4, 0.6, 0.8, 1.0]
            for rv in reference_circle_values:
                # Add invisible trace for reference circles
                fig.add_trace(go.Scatterpolar(
                    r=[rv, rv, rv, rv, rv, rv],
                    theta=['Kills/Set', 'Hitting %', 'Blocks/Set', 'Aces/Set', 'Win %', 'Kills/Set'],
                    mode='lines',
                    line=dict(color='rgba(200, 200, 200, 0.2)'),
                    showlegend=False,
                    hoverinfo='skip'
                ))
            
            # Update layout with tick labels that show actual values
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1],
                        tickvals=reference_circle_values,
                        ticktext=[
                            # Format actual values at each reference point
                            f"K:{rv*max_values['kills_per_set']:.1f} H:{rv*max_values['hitting_pctg']:.3f} B:{rv*max_values['blocks_per_set']:.1f}",
                            f"K:{rv*max_values['kills_per_set']:.1f} H:{rv*max_values['hitting_pctg']:.3f} B:{rv*max_values['blocks_per_set']:.1f}",
                            f"K:{rv*max_values['kills_per_set']:.1f} H:{rv*max_values['hitting_pctg']:.3f} B:{rv*max_values['blocks_per_set']:.1f}",
                            f"K:{rv*max_values['kills_per_set']:.1f} H:{rv*max_values['hitting_pctg']:.3f} B:{rv*max_values['blocks_per_set']:.1f}",
                            f"K:{rv*max_values['kills_per_set']:.1f} H:{rv*max_values['hitting_pctg']:.3f} B:{rv*max_values['blocks_per_set']:.1f}"
                        ]
                    ),
                    angularaxis=dict(
                        direction="clockwise"
                    )
                ),
                title="Conference Statistical Comparison (Properly Scaled)",
                showlegend=True,
                title_font=dict(size=24, color="#273469"),
                height=600
            )
            
            # Display the radar chart
            st.plotly_chart(fig, use_container_width=True)
            
            # Add an explanation note about the radar chart
            st.markdown("""
            <div class="highlight">
            <strong>About the radar chart:</strong> This chart shows how each conference performs across five key metrics. 
            The values are properly scaled to maintain the true relationships between the numbers in the data table below. 
            Hover over each point to see the exact values for each metric.
            </div>
            """, unsafe_allow_html=True)
            
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
            }).reset_index(drop=True)
            
            # Use regular dataframe method for older Streamlit versions
            st.dataframe(display_conf_data, use_container_width=True)
            
            # Create customized insights based on actual data
            top_conference = conf_data.iloc[0]['Conference']
            highest_hitting = conf_data.sort_values('hitting_pctg', ascending=False).iloc[0]
            highest_blocks = conf_data.sort_values('blocks_per_set', ascending=False).iloc[0]
            highest_aces = conf_data.sort_values('aces_per_set', ascending=False).iloc[0]
            lowest_win = conf_data.sort_values('win_loss_pctg').iloc[0]
            
            # Add conference insights
            st.markdown(f"""
            <div class="highlight">
            <strong>Conference Insights:</strong><br>
            • {highest_blocks['Conference']} leads in blocks per set ({highest_blocks['blocks_per_set']:.2f}) and has a {highest_blocks['win_loss_pctg']:.1%} win percentage.<br>
            • {highest_hitting['Conference']} has the highest hitting percentage ({highest_hitting['hitting_pctg']:.3f}) among the displayed conferences.<br>
            • {highest_aces['Conference']} records the most aces per set ({highest_aces['aces_per_set']:.2f}) despite having fewer teams ({highest_aces['Number_of_Teams']}).<br>
            • {lowest_win['Conference']} has the lowest win percentage ({lowest_win['win_loss_pctg']:.1%}) among the selected conferences.
            </div>
            """, unsafe_allow_html=True)
            
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
                filtered_vb['blocks_per_set'] * 0.1 +
                filtered_vb['kills_per_set'] * 0.4 -
                filtered_vb['opp_hitting_pctg'] * 0.1
            )
            
            # Sort by dominance score - use the top_n value from the Team Rankings tab
            top_dominant = filtered_vb[['Team', 'Conference', 'dominance_score', 'win_loss_pctg', 'hitting_pctg']].sort_values(
                by='dominance_score', ascending=False
            ).head(top_n)
            
            # Create a heat map showing correlation between metrics
            corr_metrics = ['win_loss_pctg', 'hitting_pctg', 'blocks_per_set', 'kills_per_set', 'aces_per_set', 'opp_hitting_pctg']
            corr_data = filtered_vb[corr_metrics].corr()
            
            # Use matplotlib/seaborn for the heatmap
            fig, ax = plt.subplots(figsize=(10, 8))
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
                <li>10% weighting on blocks per set</li>
                <li>40% weighting on hits per set</li>
                <li>10% negative weighting on opponent hitting percentage</li>
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
            
            # Add conference labels as annotations with improved positioning
            for i, (_, row) in enumerate(top_dominant.iterrows()):
                fig.add_annotation(
                    x=row['dominance_score'] * 0.85,  # Position at 85% of the bar length
                    y=row['Team'],
                    text=row['Conference'],
                    showarrow=False,
                    align="center",
                    yshift=0,  # Center vertically in the bar
                    font=dict(
                        size=10,
                        color="white"  # White text for better visibility on blue bars
                    ),
                    bgcolor="rgba(39, 52, 105, 0.7)",  # Semi-transparent dark blue background
                    bordercolor="rgba(255, 255, 255, 0.5)",
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
    5. Check your Streamlit version with `streamlit --version` in your terminal
    """)