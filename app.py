import streamlit as st
import pandas as pd
import numpy as np
import datetime
import plotly.graph_objects as go
import plotly.express as px
import json
import copy
import time
import functools
import concurrent.futures
from collections import defaultdict
from typing import Dict, List, Any, Tuple, Optional, Union
from data_processor import process_data, analyze_historical_data, extract_historical_arrival_patterns, extract_historical_mos_distribution
from simulation_engine import run_simulation
from optimization import optimize_schedule
from utils import ensure_config_compatibility, apply_custom_paths_to_configs

# Configure the page
st.set_page_config(page_title="Training Schedule Optimizer", layout="wide")

# Cache decorators for performance optimization
@st.cache_data(ttl=3600)
def cached_process_data(data: pd.DataFrame) -> pd.DataFrame:
    """Cached version of process_data for performance"""
    return process_data(data)

@st.cache_data(ttl=3600)
def cached_analyze_historical_data(processed_data: pd.DataFrame) -> Dict[str, Any]:
    """Cached version of analyze_historical_data for performance"""
    return analyze_historical_data(processed_data)

@st.cache_data(ttl=3600)
def cached_extract_patterns(processed_data: pd.DataFrame) -> Tuple[Dict[str, Any], Dict[str, float]]:
    """Cached function to extract historical patterns"""
    arrival_patterns = extract_historical_arrival_patterns(processed_data)
    mos_distribution = extract_historical_mos_distribution(processed_data)
    return arrival_patterns, mos_distribution

@st.cache_data
def run_cached_simulation(schedule_str: str, configs_str: str, **kwargs) -> Dict[str, Any]:
    """Run simulation with caching based on string representations of inputs"""
    # Convert string representations back to objects
    schedule = json.loads(schedule_str)
    configs = json.loads(configs_str)
    
    # Prepare simulation inputs
    simulation_inputs = {
        'schedule': schedule,
        'course_configs': configs,
        **kwargs
    }
    
    # Run the simulation
    return run_simulation(simulation_inputs)

def main():
    st.title("Training Schedule Optimization System")
    
    # Initialize structured session state
    initialize_session_state()
    
    # Sidebar navigation with improved UX
    display_sidebar_navigation()
    
    # Determine which page to display
    page = st.session_state.current_page
    
    # Display the selected page
    if page == "Upload Data":
        display_upload_page()
    elif page == "Career Path Builder":
        display_career_path_builder()
    elif page == "Course Configuration":
        display_config_page()
    elif page == "Schedule Builder":
        display_schedule_builder()
    elif page == "Simulation":
        display_simulation_page()
    elif page == "Optimization":
        display_optimization_page()
    
    # Add debug panel
    if st.session_state.show_debug:
        display_debug_panel()

def initialize_session_state():
    """Initialize structured session state for the application"""
    # Navigation state
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "Upload Data"
    
    # Debug mode toggle
    if 'show_debug' not in st.session_state:
        st.session_state.show_debug = False
    
    # Data state
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
    if 'historical_analysis' not in st.session_state:
        st.session_state.historical_analysis = None
    if 'historical_arrival_patterns' not in st.session_state:
        st.session_state.historical_arrival_patterns = None
    if 'historical_mos_distribution' not in st.session_state:
        st.session_state.historical_mos_distribution = None
    
    # Configuration state
    if 'course_configs' not in st.session_state:
        st.session_state.course_configs = {}
    else:
        # Apply backward compatibility for existing configurations
        for course, config in st.session_state.course_configs.items():
            ensure_config_compatibility(config)
    
    # Career path state
    if 'custom_paths' not in st.session_state:
        st.session_state.custom_paths = {
            '18A': {'path': [], 'flexible_courses': [], 'or_groups': [], 'flexible_constraints': {}},
            '18B': {'path': [], 'flexible_courses': [], 'or_groups': [], 'flexible_constraints': {}},
            '18C': {'path': [], 'flexible_courses': [], 'or_groups': [], 'flexible_constraints': {}},
            '18D': {'path': [], 'flexible_courses': [], 'or_groups': [], 'flexible_constraints': {}},
            '18E': {'path': [], 'flexible_courses': [], 'or_groups': [], 'flexible_constraints': {}},
            'General': {'path': [], 'flexible_courses': [], 'or_groups': [], 'flexible_constraints': {}}
        }
    
    # Schedule state
    if 'future_schedule' not in st.session_state:
        st.session_state.future_schedule = []
    
    # Simulation state
    if 'simulation_results' not in st.session_state:
        st.session_state.simulation_results = None
    if 'comparison_simulations' not in st.session_state:
        st.session_state.comparison_simulations = []
    
    # Optimization state
    if 'optimization_results' not in st.session_state:
        st.session_state.optimization_results = None
    
    # Navigation control
    if 'navigate_to' in st.session_state:
        st.session_state.current_page = st.session_state.navigate_to
        del st.session_state.navigate_to

def display_sidebar_navigation():
    """Display sidebar navigation with improved UX"""
    st.sidebar.title("Navigation")
    
    # Main navigation
    pages = ["Upload Data", "Career Path Builder", "Course Configuration", 
             "Schedule Builder", "Simulation", "Optimization"]
    
    # Highlight current page and show progress
    progress_idx = pages.index(st.session_state.current_page) if st.session_state.current_page in pages else 0
    
    # Ensure progress_pct is within [0.0, 1.0]
    # The error happens because we're dividing by (len(pages) - 1) which could be 0
    # or the calculation might result in a value > 1.0
    if len(pages) <= 1:
        progress_pct = 0.0
    else:
        progress_pct = min(1.0, max(0.0, progress_idx / (len(pages) - 1)))
    
    st.sidebar.progress(progress_pct)
    
    # Navigation buttons
    for i, page in enumerate(pages):
        # Visual indicator of current page and completion
        if i <= progress_idx:
            prefix = "✅ " if i < progress_idx else "🔍 "
        else:
            prefix = "⏳ "
        
        # Disable pages that require previous steps
        disabled = False
        if page in ["Career Path Builder", "Course Configuration"] and st.session_state.data is None:
            disabled = True
        elif page in ["Schedule Builder"] and not st.session_state.course_configs:
            disabled = True
        elif page == "Simulation" and not st.session_state.future_schedule:
            disabled = True
        elif page == "Optimization" and st.session_state.simulation_results is None:
            disabled = True
        
        # Create the button
        if st.sidebar.button(
            f"{prefix} {page}", 
            key=f"nav_{page}",
            disabled=disabled,
            use_container_width=True
        ):
            st.session_state.current_page = page
            st.rerun()
    
    # Debug mode toggle
    st.sidebar.divider()
    debug_toggle = st.sidebar.checkbox("Show Debug Panel", value=st.session_state.show_debug)
    if debug_toggle != st.session_state.show_debug:
        st.session_state.show_debug = debug_toggle
        st.rerun()
    
    # Reset application button
    if st.sidebar.button("Reset Application", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
    
    # App information
    st.sidebar.divider()
    st.sidebar.info(
        "This application helps military training planners create, simulate, and optimize "
        "training schedules for special forces Q-course pipelines."
    )

def display_upload_page():
    st.header("Upload Training Data")
    
    # Add format information and example before upload
    with st.expander("Data Format Requirements", expanded=True):
        st.markdown("""
        ### Required Data Format
        Your CSV file should include the following columns:
        | Column Name | Description | Example |
        |-------------|-------------|---------|
        | FY | Fiscal Year | 2025 |
        | Course Number | Course identifier | 2E-F253/011-F95 |
        | Course Title | Name of the course | SF QUAL (ORIENTATION) |
        | NAME | Student name | SMITH JOHN A |
        | SSN | Social Security Number (unique ID) | 123456789 |
        | Arrival Date | Student arrival date (MM/DD/YYYY) | 5/24/2025 |
        | CLS | Class number | 003 |
        | Cls Start Date | Class start date (MM/DD/YYYY) | 5/27/2025 |
        | Cls End Date | Class end date (MM/DD/YYYY) | 5/30/2025 |
        | Res Stat | Reservation status | R |
        | Reserve Reason | Reservation reason code | |
        | Reserve Reason Description | Detailed reservation reason | |
        | Input Stat | Input status code | I |
        | Input Date | Input date (MM/DD/YYYY) | 5/27/2025 |
        | Input Reason | Input reason code | |
        | Input Reason Description | Detailed input reason | |
        | Out Stat | Output status code | G |
        | Output Date | Output date (MM/DD/YYYY) | 5/30/2025 |
        | CP Pers Type | Personnel type | E |
        | Group Type | Student group type | ADE |
        | Training MOS | Military Occupational Specialty (optional) | 18B |
        
        **Important Status Codes:**
        
        **INPUT STAT:**
        - H = HOLD (SHOWED DID NOT START OR DID NOT GRAD)
        - I = NEW INPUT
        - J = RETRAINEE IN, FROM ANOTHER COURSE OF INSTRUCTION
        - N = NO SHOW
        - Q = RECYCLE IN, FROM ANOTHER CLASS, SAME COURSE
        - U = SHOWED, DID NOT BEGIN TRNG (POST APPROP REASON CODE)
        
        **OUT STAT:**
        - G = GRADUATE, SUCCESSFULLY COMPLETED CLASS
        - K = RETRAINEE OUT, TO ANOTHER COURSE OF INSTRUCTION
        - L = RECYCLE OUT, TO ANOTHER CLASS, SAME COURSE
        - Z = NON-SUCCESSFUL COMPLETION
        
        **RES STAT (Reservation Status):**
        - C = CANCELLED RESERVATION
        - R = VALID RESERVATION
        - M = MEP RESERVATION
        - W = WAITING FOR RESERVATION
        
        **CP Pers Type:**
        - O = Officer
        - E = Enlisted
        
        **Group Type Examples:**
        - ADE = Active Duty Enlisted
        - OF = Officer
        - NG = National Guard
        """, unsafe_allow_html=False)
        
        # Option to download a sample CSV
        sample_data = """FY,Course Number,Course Title,NAME,SSN,Arrival Date,CLS,Cls Start Date,Cls End Date,Res Stat,Reserve Reason,Reserve Reason Description,Input Stat,Input Date,Input Reason,Input Reason Description,Out Stat,Output Date,CP Pers Type,Group Type,Training MOS
2025,2E-F253/011-F95,SF QUAL (ORIENTATION),SEABRIGHT NICK J,123456789,5/24/2025,003,5/27/2025,5/30/2025,R,,,I,5/27/2025,,,G,5/30/2025,E,ADE,18B
2025,2E-F254/011-F96,SF QUAL (SMALL UNIT TACTICS),SEABRIGHT NICK J,123456789,5/30/2025,003,6/2/2025,7/11/2025,R,,,I,6/2/2025,,,G,7/11/2025,E,ADE,18B
2025,3A-F38/012-F27,SERE HIGH RISK (LEVEL C),SEABRIGHT NICK J,123456789,7/11/2025,010,7/14/2025,8/1/2025,R,,,I,7/14/2025,,,G,8/1/2025,E,ADE,18B
2025,011-18B30-C45,SF QUAL (SF WEAPONS SERGEANT) ALC,SEABRIGHT NICK J,123456789,8/1/2025,003,8/4/2025,10/24/2025,R,,,I,8/4/2025,,,G,10/24/2025,E,ADE,18B
2025,600-C44 (ARSOF),ARSOF BASIC LEADER,SMITH JANE A,987654321,4/25/2025,005,4/28/2025,5/19/2025,R,,,I,4/28/2025,,,G,5/19/2025,O,OF,18A
2024,2E-F133/011-SQIW,SF ADV RECON TGT ANALY,JOHNSON ROBERT T,456789123,10/7/2024,001,10/10/2024,12/9/2024,R,,,I,10/10/2024,,,G,12/9/2024,E,NG,18C"""
        
        st.download_button(
            label="Download Sample CSV",
            data=sample_data,
            file_name="sample_training_data.csv",
            mime="text/csv"
        )
    
    uploaded_file = st.file_uploader("Upload training data CSV", type=["csv"])
    
    if uploaded_file is not None:
        try:
            # Display loading indicator
            with st.spinner("Loading and processing data..."):
                data = pd.read_csv(uploaded_file)
                st.session_state.data = data
                
                # Display data preview
                st.subheader("Data Preview")
                st.dataframe(data.head())
                
                # Process data with progress indicator
                progress_bar = st.progress(0)
                st.write("Processing data...")
                
                # Step 1: Process data (25%)
                processed_data = cached_process_data(data)
                st.session_state.processed_data = processed_data
                progress_bar.progress(25)
                
                # Step 2: Historical analysis (50%)
                historical_analysis = cached_analyze_historical_data(processed_data)
                st.session_state.historical_analysis = historical_analysis
                progress_bar.progress(50)
                
                # Step 3: Extract historical arrival patterns (75%)
                try:
                    historical_arrival_patterns = extract_historical_arrival_patterns(processed_data)
                    st.session_state.historical_arrival_patterns = historical_arrival_patterns
                except Exception as e:
                    st.warning(f"Could not extract arrival patterns: {e}")
                    st.session_state.historical_arrival_patterns = None
                progress_bar.progress(75)

                # Step 4: Extract historical MOS distribution with robust error handling (100%)
                try:
                    # Default distribution to use if anything goes wrong
                    default_distribution = {'18A': 0.2, '18B': 0.2, '18C': 0.2, '18D': 0.2, '18E': 0.2}
                    
                    # First check if processed_data has the necessary column
                    if 'TrainingMOS' in processed_data.columns:
                        # Count non-null values by MOS
                        valid_mos_mask = processed_data['TrainingMOS'].notna() & (processed_data['TrainingMOS'] != 'None')
                        
                        if valid_mos_mask.sum() > 0:
                            # Get counts of each MOS
                            mos_counts = processed_data.loc[valid_mos_mask, 'TrainingMOS'].value_counts()
                            total_students = mos_counts.sum()
                            
                            # Create a clean distribution dictionary
                            mos_distribution = default_distribution.copy()
                            
                            # Process each MOS value
                            for mos, count in mos_counts.items():
                                if isinstance(mos, str):
                                    # Normalize to standard MOS
                                    if mos in ['18A', '18B', '18C', '18D', '18E']:
                                        standard_mos = mos
                                    elif 'OFFICER' in mos.upper() or '18A' in mos.upper():
                                        standard_mos = '18A'
                                    elif 'WEAPON' in mos.upper() or '18B' in mos.upper():
                                        standard_mos = '18B'
                                    elif 'ENGINEER' in mos.upper() or '18C' in mos.upper():
                                        standard_mos = '18C'
                                    elif 'MEDIC' in mos.upper() or '18D' in mos.upper():
                                        standard_mos = '18D'
                                    elif 'COMM' in mos.upper() or '18E' in mos.upper():
                                        standard_mos = '18E'
                                    else:
                                        # Deterministic assignment for unknown values
                                        hash_val = sum(ord(c) for c in mos) % 5
                                        standard_mos = ['18A', '18B', '18C', '18D', '18E'][hash_val]
                                    
                                    # Calculate percentage and add to distribution
                                    percentage = float(count / total_students)
                                    mos_distribution[standard_mos] = percentage
                            
                            # Normalize distribution to sum to 1.0
                            total = sum(mos_distribution.values())
                            if total > 0:
                                normalized_distribution = {k: v/total for k, v in mos_distribution.items()}
                                st.session_state.historical_mos_distribution = normalized_distribution
                            else:
                                st.session_state.historical_mos_distribution = default_distribution
                        else:
                            st.warning("No valid TrainingMOS values found. Using default distribution.")
                            st.session_state.historical_mos_distribution = default_distribution
                    else:
                        st.warning("TrainingMOS column not found. Using default distribution.")
                        st.session_state.historical_mos_distribution = default_distribution
                except Exception as e:
                    st.warning(f"Error extracting MOS distribution: {e}")
                    st.session_state.historical_mos_distribution = default_distribution
                
                progress_bar.progress(100)

                # Display data statistics and quality metrics
                st.subheader("Data Statistics")
                
                # Create metrics in columns
                col1, col2, col3 = st.columns(3)
                
                unique_courses = len(data['Course Title'].unique())
                unique_students = len(data['SSN'].unique())
                
                with col1:
                    st.metric("Unique Courses", unique_courses)
                
                with col2:
                    st.metric("Unique Students", unique_students)
                
                with col3:
                    total_entries = len(data)
                    st.metric("Total Records", total_entries)
                
                # Display extracted patterns
                with st.expander("Historical Patterns"):
                    if st.session_state.historical_arrival_patterns:
                        st.write("### Historical Arrival Patterns")
                        st.write(f"Average days before class start: {st.session_state.historical_arrival_patterns.get('avg_days_before', 'N/A')}")
                        
                        # Show monthly distribution
                        monthly_data = st.session_state.historical_arrival_patterns.get('monthly_distribution', {})
                        if monthly_data:
                            months = list(monthly_data.keys())
                            values = list(monthly_data.values())
                            fig = px.bar(
                                x=months,
                                y=values,
                                title="Historical Monthly Arrival Distribution",
                                labels={"x": "Month", "y": "Percentage of Students"},
                                color_discrete_sequence=['#3366CC']
                            )
                            fig.update_layout(xaxis={'categoryorder':'array', 'categoryarray':['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']})
                            st.plotly_chart(fig, use_container_width=True)
                    
                    # Display MOS distribution with additional safety checks
                    if 'historical_mos_distribution' in st.session_state and st.session_state.historical_mos_distribution:
                        try:
                            st.write("### Historical MOS Distribution")
                            mos_data = []
                            
                            # Get distribution and ensure it only contains the standard MOS keys
                            mos_distribution = st.session_state.historical_mos_distribution
                            standard_mos = ['18A', '18B', '18C', '18D', '18E']
                            
                            for mos in standard_mos:
                                if mos in mos_distribution and isinstance(mos_distribution[mos], (int, float)):
                                    mos_data.append({"MOS": mos, "Percentage": float(mos_distribution[mos]) * 100})
                            
                            # Create visualization only if we have data
                            if mos_data:
                                fig = px.pie(
                                    mos_data,
                                    values="Percentage",
                                    names="MOS",
                                    title="Historical MOS Distribution",
                                    color_discrete_sequence=px.colors.qualitative.Bold
                                )
                                fig.update_traces(textposition='inside', textinfo='percent+label')
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.info("No valid MOS distribution data available.")
                        except Exception as e:
                            st.error(f"Error displaying MOS distribution: {e}")
                            st.info("Unable to display MOS distribution chart.")
                
                st.success("✅ Data successfully loaded and processed!")
                
                # Navigation hint
                st.info("You can now proceed to the Career Path Builder to define course relationships or directly to Course Configuration.")
                
                # Direct navigation buttons
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Go to Career Path Builder", use_container_width=True):
                        st.session_state.current_page = "Career Path Builder"
                        st.rerun()
                with col2:
                    if st.button("Go to Course Configuration", use_container_width=True):
                        st.session_state.current_page = "Course Configuration"
                        st.rerun()
                
        except Exception as e:
            st.error(f"Error processing data: {e}")
    else:
        st.info("Please upload a CSV file with the training data.")

def calculate_data_quality(data: pd.DataFrame, processed_data: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate data quality metrics
    Args:
        data: Raw input data
        processed_data: Processed data
    Returns:
        Dictionary of quality metrics
    """
    quality_metrics = {}
    
    # Overall completeness
    total_cells = data.size
    missing_cells = data.isnull().sum().sum()
    quality_metrics['completeness'] = 1 - (missing_cells / total_cells) if total_cells > 0 else 0
    
    # Column-level completeness
    missing_columns = {}
    for col in data.columns:
        missing_count = data[col].isnull().sum()
        missing_columns[col] = missing_count / len(data) if len(data) > 0 else 0
    quality_metrics['missing_columns'] = missing_columns
    
    # Duplicate detection
    if 'SSN' in data.columns:
        duplicate_students = data['SSN'].duplicated().sum()
        quality_metrics['duplicate_students'] = duplicate_students
    
    # Invalid dates
    if 'Cls Start Date' in data.columns and 'Cls End Date' in data.columns:
        try:
            start_dates = pd.to_datetime(data['Cls Start Date'], errors='coerce')
            end_dates = pd.to_datetime(data['Cls End Date'], errors='coerce')
            invalid_dates = ((end_dates < start_dates) | start_dates.isnull() | end_dates.isnull()).sum()
            quality_metrics['invalid_dates'] = invalid_dates
        except:
            quality_metrics['invalid_dates'] = "Unable to determine"
    
    return quality_metrics

def visualize_prerequisites(course_configs: Dict[str, Any]) -> go.Figure:
    """
    Create a network visualization of course prerequisite relationships
    Args:
        course_configs: Dictionary of course configurations
    Returns:
        Plotly figure with the network visualization
    """
    try:
        import networkx as nx
        
        # Create a directed graph
        G = nx.DiGraph()
        
        # Add nodes (courses)
        for course in course_configs:
            G.add_node(course)
        
        # Add edges (prerequisites)
        for course, config in course_configs.items():
            # AND prerequisites
            if 'prerequisites' in config:
                prereqs = []
                if isinstance(config['prerequisites'], list):
                    prereqs = config['prerequisites']
                elif isinstance(config['prerequisites'], dict):
                    prereqs = config['prerequisites'].get('courses', [])
                
                for prereq in prereqs:
                    if prereq in course_configs:  # Only add if both are in the configs
                        G.add_edge(prereq, course, type='AND')
            
            # OR prerequisites
            if 'or_prerequisites' in config:
                for i, or_group in enumerate(config['or_prerequisites']):
                    # Create a virtual node for the OR group
                    or_node = f"OR_{course}_{i}"
                    G.add_node(or_node, is_virtual=True)
                    G.add_edge(or_node, course, type='OR')
                    
                    for prereq in or_group:
                        if prereq in course_configs:  # Only add if both are in the configs
                            G.add_edge(prereq, or_node, type='OR_member')
        
        # Use a layout that shows the hierarchy
        pos = nx.spring_layout(G, k=0.3, iterations=50)
        
        # Prepare the visualization
        edge_trace = []
        node_trace = []
        
        # Add edges
        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_type = edge[2].get('type', 'default')
            
            # Use different styles for different edge types
            if edge_type == 'AND':
                line_color = 'rgba(66, 133, 244, 0.8)'  # Blue for AND
                line_width = 2
                line_dash = 'solid'
            elif edge_type == 'OR':
                line_color = 'rgba(219, 68, 55, 0.8)'  # Red for OR
                line_width = 2
                line_dash = 'dash'
            else:  # OR_member
                line_color = 'rgba(219, 68, 55, 0.5)'  # Light red for OR members
                line_width = 1
                line_dash = 'dot'
            
            edge_trace.append(
                go.Scatter(
                    x=[x0, x1, None],
                    y=[y0, y1, None],
                    line=dict(width=line_width, color=line_color, dash=line_dash),
                    hoverinfo='none',
                    mode='lines'
                )
            )
        
        # Add nodes
        node_x = []
        node_y = []
        node_text = []
        node_color = []
        node_size = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            # Check if it's a virtual node
            is_virtual = G.nodes[node].get('is_virtual', False)
            
            if is_virtual:
                # Format for OR group nodes
                node_text.append("OR")
                node_color.append('rgba(219, 68, 55, 0.7)')  # Red for OR nodes
                node_size.append(10)  # Smaller size for virtual nodes
            else:
                # Regular course nodes
                node_text.append(node)
                node_color.append('rgba(66, 133, 244, 0.7)')  # Blue for regular nodes
                node_size.append(20)
        
        # Create node trace
        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            text=node_text,
            mode='markers+text',
            hoverinfo='text',
            marker=dict(
                color=node_color,
                size=node_size,
                line=dict(width=2, color='rgba(50, 50, 50, 0.8)')
            ),
            textposition="top center"
        )
        
        # Create the figure
        fig = go.Figure(data=edge_trace + [node_trace])
        
        # Update layout
        fig.update_layout(
            title='Course Prerequisite Relationships',
            titlefont_size=16,
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=600,
            width=800
        )
        
        return fig
    
    except Exception as e:
        print(f"Error creating prerequisite visualization: {e}")
        
        # Return a simple error figure
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating visualization: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig

def display_career_path_builder():
    st.header("Career Path Builder")
    st.write("Build and edit career paths for each MOS to automatically set prerequisites.")
    
    # Check if data is uploaded first
    if 'processed_data' not in st.session_state or st.session_state.processed_data is None:
        if 'data' not in st.session_state or st.session_state.data is None:
            st.warning("Please upload data first before using the Career Path Builder.")
            st.write("Go to the 'Upload Data' page to upload your training data.")
            return
    
    # Initialize custom paths in session state if not present
    if 'custom_paths' not in st.session_state:
        st.session_state.custom_paths = {
            '18A': {'path': [], 'flexible_courses': [], 'or_groups': [], 'flexible_constraints': {}, 'or_group_positions': {}},
            '18B': {'path': [], 'flexible_courses': [], 'or_groups': [], 'flexible_constraints': {}, 'or_group_positions': {}},
            '18C': {'path': [], 'flexible_courses': [], 'or_groups': [], 'flexible_constraints': {}, 'or_group_positions': {}},
            '18D': {'path': [], 'flexible_courses': [], 'or_groups': [], 'flexible_constraints': {}, 'or_group_positions': {}},
            '18E': {'path': [], 'flexible_courses': [], 'or_groups': [], 'flexible_constraints': {}, 'or_group_positions': {}}
        }
    
    # Add General path option
    if 'General' not in st.session_state.custom_paths:
        st.session_state.custom_paths['General'] = {
            'path': [], 
            'flexible_courses': [], 
            'or_groups': [], 
            'flexible_constraints': {},
            'or_group_positions': {}
        }
    
    # Get all available courses
    all_courses = []
    # First try to get courses from course_configs
    if 'course_configs' in st.session_state and st.session_state.course_configs:
        all_courses.extend(list(st.session_state.course_configs.keys()))
    # Then try to get courses from processed_data if it exists
    if 'processed_data' in st.session_state and st.session_state.processed_data is not None:
        data_courses = list(st.session_state.processed_data['Course Title'].unique())
        all_courses.extend([c for c in data_courses if c not in all_courses])
    # If we still don't have courses, check raw data
    if not all_courses and 'data' in st.session_state and st.session_state.data is not None:
        data_courses = list(st.session_state.data['Course Title'].unique())
        all_courses.extend(data_courses)
    # Remove duplicates and sort
    all_courses = sorted(list(set(all_courses)))
    
    # Debug information
    st.write(f"Found {len(all_courses)} unique courses")
    with st.expander("Available Courses"):
        for i, course in enumerate(all_courses):
            st.write(f"{i+1}. {course}")
    
    if not all_courses:
        st.error("No courses found. Please upload data or configure courses first.")
        return
    
    # Create tabs for each MOS and General path
    path_tabs = st.tabs(['General', '18A (Officer)', '18B (Weapons)', '18C (Engineer)', '18D (Medical)', '18E (Communications)'])
    
    # Process each tab
    for i, mos in enumerate(['General', '18A', '18B', '18C', '18D', '18E']):
        with path_tabs[i]:
            st.write(f"### {mos} Career Path")
            
            # Get the current path
            path_data = st.session_state.custom_paths[mos]
            current_path = path_data.get('path', [])
            flexible_courses = path_data.get('flexible_courses', [])
            or_groups = path_data.get('or_groups', []) if 'or_groups' in path_data else []
            flexible_constraints = path_data.get('flexible_constraints', {}) if 'flexible_constraints' in path_data else {}
            
            # Ensure the structure exists
            if 'or_groups' not in path_data:
                path_data['or_groups'] = []
                or_groups = []
            
            if 'flexible_constraints' not in path_data:
                path_data['flexible_constraints'] = {}
                flexible_constraints = {}
                
            # Track path position for OR groups
            if 'or_group_positions' not in path_data:
                path_data['or_group_positions'] = {}
            or_group_positions = path_data.get('or_group_positions', {})
            
            # Debug information for this MOS
            st.write(f"Path has {len(current_path)} courses, {len(or_groups)} OR groups, {len(flexible_courses)} flexible courses")
            
            # Build Career Path
            st.write("### Build Career Path")
            st.write("Add courses in sequence to create a career path. Courses will automatically get prerequisites based on their order.")
            
            # Add course to path
            col1, col2 = st.columns([3, 1])
            with col1:
                # Get courses that aren't already in any path or OR group
                all_used_courses = set(current_path)
                for group in or_groups:
                    all_used_courses.update(group)
                all_used_courses.update(flexible_courses)
                
                available_courses = [c for c in all_courses if c not in all_used_courses]
                if not available_courses:
                    st.warning("All courses have been added to either the path or flexible courses.")
                    new_course = ""
                else:
                    new_course = st.selectbox(
                        "Add Course to Path",
                        options=[""] + available_courses,
                        key=f"add_course_{mos}"
                    )
            with col2:
                add_button = st.button("Add", key=f"add_button_{mos}")
                if add_button and new_course:
                    current_path.append(new_course)
                    st.session_state.custom_paths[mos]['path'] = current_path
                    st.success(f"Added {new_course} to path")
                    st.rerun()
            
            # Display and edit the current path
            if current_path:
                st.write("### Current Path Order")
                
                # Build a combined path that includes OR groups in their positions
                combined_path = []
                combined_types = []  # Track if each item is a regular course or OR group
                
                # Add regular courses and OR groups in order
                regular_course_positions = list(range(len(current_path)))
                all_or_positions = [(or_group_positions.get(f"group_{i}", 0), i) for i in range(len(or_groups))]
                
                # Sort all positions
                all_positions = [(pos, 'regular', idx) for idx, pos in enumerate(regular_course_positions)]
                all_positions.extend([(pos, 'or_group', idx) for pos, idx in all_or_positions])
                all_positions.sort()  # Sort by position
                
                # Build the combined path
                for pos, item_type, idx in all_positions:
                    if item_type == 'regular':
                        combined_path.append(current_path[idx])
                        combined_types.append('regular')
                    else:  # or_group
                        group_courses = " OR ".join(or_groups[idx]) if or_groups[idx] else "Empty Group"
                        combined_path.append(f"[OR Group {idx+1}: {group_courses}]")
                        combined_types.append('or_group')
                
                # Display combined path
                path_df = pd.DataFrame({
                    "Position": range(1, len(combined_path)+1),
                    "Course/Group": combined_path,
                    "Type": combined_types
                })
                
                # Display styled dataframe
                st.dataframe(
                    path_df,
                    hide_index=True, 
                    use_container_width=True
                )
                
                # Manual reordering controls for regular courses
                st.write("### Reorder Courses")
                col1, col2, col3 = st.columns([3, 2, 1])
                
                with col1:
                    selected_course = st.selectbox(
                        "Select course to move", 
                        options=current_path,
                        key=f"move_course_{mos}"
                    )
                
                with col2:
                    # Get current position (1-based for user interface)
                    current_position = current_path.index(selected_course) + 1 if selected_course in current_path else 1
                    
                    # New position input
                    new_position = st.number_input(
                        "Move to position", 
                        min_value=1, 
                        max_value=len(current_path),
                        value=current_position,
                        key=f"new_position_{mos}"
                    )
                
                with col3:
                    # Apply move button
                    if st.button("Move", key=f"apply_move_{mos}"):
                        # Remove from current position (0-based index)
                        current_path.remove(selected_course)
                        # Insert at new position (adjust for 0-based indexing)
                        current_path.insert(new_position - 1, selected_course)
                        st.session_state.custom_paths[mos]['path'] = current_path
                        st.success(f"Moved {selected_course} to position {new_position}")
                        st.rerun()
                
                # Option to remove courses
                st.write("### Remove Course")
                course_to_remove = st.selectbox(
                    "Remove Course from Path",
                    options=[""] + current_path,
                    key=f"remove_course_{mos}"
                )
                if st.button("Remove", key=f"remove_button_{mos}") and course_to_remove:
                    current_path.remove(course_to_remove)
                    st.session_state.custom_paths[mos]['path'] = current_path
                    st.success(f"Removed {course_to_remove} from path")
                    st.rerun()
            else:
                st.info("No courses in path yet. Add courses using the dropdown above.")
            
            # OR Groups Section - Courses where student must take one from the group
            st.write("### OR Course Groups")
            st.write("Add groups of courses where a student must complete ONE course from EACH group.")
            
            # Add a new OR group button above the groups
            if st.button("Add New OR Group", key=f"add_or_group_{mos}"):
                new_group_id = f"group_{len(or_groups)}"
                or_groups.append([])
                # Default position is at the end of the current path
                or_group_positions[new_group_id] = len(current_path)
                st.session_state.custom_paths[mos]['or_groups'] = or_groups
                st.session_state.custom_paths[mos]['or_group_positions'] = or_group_positions
                st.success("Added new OR Group")
                st.rerun()
                
            # Display existing OR groups
            for i, group in enumerate(or_groups):
                group_id = f"group_{i}"
                st.write(f"**OR Group {i+1}:**")
                
                # Use columns for group editing
                cols = st.columns([3, 2, 1])
                
                with cols[0]:
                    # We need to include all courses as options except those already in the path or in other groups
                    used_in_other_groups = set()
                    for j, other_group in enumerate(or_groups):
                        if i != j:  # Don't exclude courses from this group
                            used_in_other_groups.update(other_group)
                    
                    available_or_courses = [c for c in all_courses if c not in current_path 
                                          and c not in flexible_courses 
                                          and c not in used_in_other_groups]
                    
                    # Make sure current group members are in options
                    for course in group:
                        if course not in available_or_courses:
                            available_or_courses.append(course)
                    
                    # Store selections in session state to prevent reloading
                    group_key = f"or_group_{mos}_{i}"
                    if group_key not in st.session_state:
                        st.session_state[group_key] = group.copy()
                    
                    # Display multiselect for group courses
                    updated_group = st.multiselect(
                        f"Select courses for Group {i+1}:",
                        options=available_or_courses,
                        default=group,  # Use the actual group directly, not session state
                        key=group_key
                    )
                
                with cols[1]:
                    # Position selection for the OR group in the path
                    position_options = list(range(len(current_path) + 1))
                    position_labels = ["Start of path"] + [f"After: {current_path[j]}" for j in range(len(current_path))]
                    
                    # Get the current position with a default
                    current_position = or_group_positions.get(group_id, 0)
                    
                    # Ensure index is valid
                    position_index = min(current_position, len(position_options)-1) if position_options else 0
                    
                    # Show position selector
                    position_key = f"position_{mos}_{i}"
                    if position_key not in st.session_state:
                        st.session_state[position_key] = position_index
                        
                    selected_position = st.selectbox(
                        "Position in path:",
                        options=position_options,
                        format_func=lambda x: position_labels[x] if x < len(position_labels) else "End of path",
                        index=position_index,
                        key=f"position_{mos}_{i}"
                    )
                
                with cols[2]:
                    # Option to delete this group
                    if st.button(f"Delete", key=f"delete_or_group_{mos}_{i}"):
                        or_groups.pop(i)
                        # Also remove the position
                        if group_id in or_group_positions:
                            del or_group_positions[group_id]
                        st.session_state.custom_paths[mos]['or_groups'] = or_groups
                        st.session_state.custom_paths[mos]['or_group_positions'] = or_group_positions
                        st.success(f"Deleted OR Group {i+1}")
                        st.rerun()
                    
                    # Button to update the group
                    if st.button(f"Update", key=f"update_or_group_{mos}_{i}"):
                        or_groups[i] = updated_group  # Get value directly from the widget result
                        or_group_positions[group_id] = selected_position
                        st.session_state.custom_paths[mos]['or_groups'] = or_groups
                        st.session_state.custom_paths[mos]['or_group_positions'] = or_group_positions
                        st.success(f"Updated Group {i+1}")
                        st.rerun()
                            
            # Save OR groups
            st.session_state.custom_paths[mos]['or_groups'] = or_groups
            st.session_state.custom_paths[mos]['or_group_positions'] = or_group_positions
            
            # Flexible Courses Section
            st.write("### Flexible Courses")
            st.write("These courses are part of the career path but have flexible timing constraints.")
            
            # Add flexible course
            col1, col2 = st.columns([3, 1])
            with col1:
                # Get courses that aren't already in any path
                all_used_courses = set(current_path)
                for group in or_groups:
                    all_used_courses.update(group)
                all_used_courses.update(flexible_courses)
                
                available_flex_courses = [c for c in all_courses if c not in all_used_courses]
                if not available_flex_courses:
                    st.warning("All courses have been added to either the path or flexible courses.")
                    new_flexible = ""
                else:
                    new_flexible = st.selectbox(
                        "Add Flexible Course",
                        options=[""] + available_flex_courses,
                        key=f"add_flexible_{mos}"
                    )
            with col2:
                if st.button("Add", key=f"add_flexible_button_{mos}") and new_flexible:
                    flexible_courses.append(new_flexible)
                    st.session_state.custom_paths[mos]['flexible_courses'] = flexible_courses
                    # Initialize constraints for this course
                    if new_flexible not in flexible_constraints:
                        flexible_constraints[new_flexible] = {
                            'must_be_before': [],
                            'must_be_after': []
                        }
                    st.session_state.custom_paths[mos]['flexible_constraints'] = flexible_constraints
                    st.success(f"Added {new_flexible} as flexible course")
                    st.rerun()
            
            # Display flexible courses and constraints
            if flexible_courses:
                st.write("### Flexible Course Constraints")
                st.write("Set timing constraints for when flexible courses can be taken:")
                
                for i, course in enumerate(flexible_courses):
                    st.write(f"**{course}**")
                    # Create constraint columns
                    constraint_cols = st.columns(2)
                    
                    # Initialize constraints if they don't exist
                    if course not in flexible_constraints:
                        flexible_constraints[course] = {
                            'must_be_before': [],
                            'must_be_after': []
                        }
                    
                    # Create session state keys for constraints
                    before_key = f"before_{mos}_{i}"
                    after_key = f"after_{mos}_{i}"
                    
                    # Initialize session state for constraints
                    if before_key not in st.session_state:
                        st.session_state[before_key] = flexible_constraints[course].get('must_be_before', [])
                    
                    if after_key not in st.session_state:
                        st.session_state[after_key] = flexible_constraints[course].get('must_be_after', [])
                    
                    # Before constraints
                    with constraint_cols[0]:
                        selected_before = st.multiselect(
                            f"Must be taken BEFORE these courses:",
                            options=current_path,
                            default=st.session_state[before_key],
                            key=before_key
                        )
                        st.session_state[before_key] = selected_before
                    
                    # After constraints
                    with constraint_cols[1]:
                        selected_after = st.multiselect(
                            f"Must be taken AFTER these courses:",
                            options=current_path,
                            default=st.session_state[after_key],
                            key=after_key
                        )
                        st.session_state[after_key] = selected_after
                    
                    # Update button for this flexible course
                    if st.button("Update Constraints", key=f"update_constraints_{mos}_{i}"):
                        flexible_constraints[course]['must_be_before'] = selected_before
                        flexible_constraints[course]['must_be_after'] = selected_after
                        st.session_state.custom_paths[mos]['flexible_constraints'] = flexible_constraints
                        st.success(f"Updated constraints for {course}")
                        st.rerun()
                
                # Display list of flexible courses
                st.write("### Current flexible courses:")
                flexible_df = pd.DataFrame({
                    "course": flexible_courses
                })
                st.dataframe(flexible_df, hide_index=True, use_container_width=True)
                
                # Remove flexible course
                flexible_to_remove = st.selectbox(
                    "Remove Flexible Course",
                    options=[""] + flexible_courses,
                    key=f"remove_flexible_{mos}"
                )
                if st.button("Remove", key=f"remove_flexible_button_{mos}") and flexible_to_remove:
                    flexible_courses.remove(flexible_to_remove)
                    # Also remove from constraints
                    if flexible_to_remove in flexible_constraints:
                        del flexible_constraints[flexible_to_remove]
                    st.session_state.custom_paths[mos]['flexible_courses'] = flexible_courses
                    st.session_state.custom_paths[mos]['flexible_constraints'] = flexible_constraints
                    st.success(f"Removed {flexible_to_remove} from flexible courses")
                    st.rerun()
            else:
                st.info("No flexible courses defined yet.")
            
            # Save flexible constraints
            st.session_state.custom_paths[mos]['flexible_constraints'] = flexible_constraints
            
            # Move between path and flexible
            if current_path and flexible_courses:
                st.write("### Move Courses")
                col1, col2 = st.columns(2)
                with col1:
                    # Move from path to flexible
                    path_to_flexible = st.selectbox(
                        "Move from Path to Flexible",
                        options=[""] + current_path,
                        key=f"path_to_flexible_{mos}"
                    )
                    if st.button("Move to Flexible", key=f"path_to_flexible_button_{mos}") and path_to_flexible:
                        current_path.remove(path_to_flexible)
                        flexible_courses.append(path_to_flexible)
                        # Initialize constraints for this course
                        if path_to_flexible not in flexible_constraints:
                            flexible_constraints[path_to_flexible] = {
                                'must_be_before': [],
                                'must_be_after': []
                            }
                        st.session_state.custom_paths[mos]['path'] = current_path
                        st.session_state.custom_paths[mos]['flexible_courses'] = flexible_courses
                        st.session_state.custom_paths[mos]['flexible_constraints'] = flexible_constraints
                        st.success(f"Moved {path_to_flexible} to flexible courses")
                        st.rerun()
                with col2:
                    # Move from flexible to path
                    flexible_to_path = st.selectbox(
                        "Move from Flexible to Path",
                        options=[""] + flexible_courses,
                        key=f"flexible_to_path_{mos}"
                    )
                    if st.button("Move to Path", key=f"flexible_to_path_button_{mos}") and flexible_to_path:
                        flexible_courses.remove(flexible_to_path)
                        current_path.append(flexible_to_path)
                        # Remove from constraints when moving to the main path
                        if flexible_to_path in flexible_constraints:
                            del flexible_constraints[flexible_to_path]
                        st.session_state.custom_paths[mos]['path'] = current_path
                        st.session_state.custom_paths[mos]['flexible_courses'] = flexible_courses
                        st.session_state.custom_paths[mos]['flexible_constraints'] = flexible_constraints
                        st.success(f"Moved {flexible_to_path} to path")
                        st.rerun()
            
            # Show the current path visualization
            if current_path or flexible_courses or or_groups:
                st.write("### Career Path Visualization")
                
                # Create a Plotly figure for the visualization
                fig = go.Figure()
                
                # Calculate positions for the visualization based on the combined path
                visual_positions = {}
                current_pos = 0
                
                # Create a combined ordered list of all path elements
                combined_items = []
                
                # Add regular courses with their positions
                for idx, course in enumerate(current_path):
                    combined_items.append(('regular', idx, current_pos))
                    visual_positions[course] = current_pos
                    current_pos += 1
                
                # Add OR groups with their positions
                for i in range(len(or_groups)):
                    group_id = f"group_{i}"
                    position = or_group_positions.get(group_id, 0)
                    # Insert at the right position
                    combined_items.append(('or_group', i, position))
                
                # Sort all items by their position
                combined_items.sort(key=lambda x: x[2])
                
                # Plot the main path courses
                for course in current_path:
                    pos = visual_positions.get(course, 0)
                    fig.add_trace(go.Scatter(
                        x=[pos],
                        y=[0],
                        mode='markers+text',
                        text=[course],
                        textposition='top center',
                        marker=dict(size=20, color='rgba(66, 133, 244, 0.8)'),
                        name=course
                    ))
                
                # Add connecting lines between courses in the path
                for i in range(len(current_path) - 1):
                    pos1 = visual_positions.get(current_path[i], 0)
                    pos2 = visual_positions.get(current_path[i+1], 0)
                    
                    if pos2 > pos1:
                        fig.add_annotation(
                            x=pos1+0.5,
                            y=0,
                            ax=pos1,
                            ay=0,
                            xref='x',
                            yref='y',
                            axref='x',
                            ayref='y',
                            showarrow=True,
                            arrowhead=2,
                            arrowsize=1,
                            arrowwidth=2,
                            arrowcolor='rgba(66, 133, 244, 0.8)'
                        )
                
                # Plot OR groups
                for group_idx, group in enumerate(or_groups):
                    if not group:  # Skip empty groups
                        continue
                    
                    # Get position for this OR group
                    group_id = f"group_{group_idx}"
                    position = or_group_positions.get(group_id, 0)
                    
                    # Position OR groups below the main path
                    y_position = -1 - group_idx % 3  # Stack OR groups if there are many
                    
                    # Add a label for the OR group
                    fig.add_annotation(
                        x=position,
                        y=y_position - 0.5,
                        text=f"OR Group {group_idx+1}",
                        showarrow=False,
                        font=dict(size=12),
                        xanchor="center"
                    )
                    
                    # Add courses in the OR group
                    for c_idx, course in enumerate(group):
                        x_offset = (c_idx - (len(group) - 1) / 2) * 0.3  # Center the group
                        fig.add_trace(go.Scatter(
                            x=[position + x_offset],
                            y=[y_position],
                            mode='markers+text',
                            text=[course],
                            textposition='bottom center',
                            marker=dict(size=15, color='rgba(219, 68, 55, 0.8)'),
                            name=f"OR{group_idx+1}: {course}"
                        ))
                    
                    # Add a vertical line from the OR group to the path
                    fig.add_shape(
                        type="line",
                        x0=position,
                        y0=y_position,
                        x1=position,
                        y1=-0.2,
                        line=dict(color="rgba(219, 68, 55, 0.8)", width=2, dash="dot"),
                    )
                
                # Plot flexible courses with constraints
                for f_idx, course in enumerate(flexible_courses):
                    # Position flexible courses above the main path
                    y_position = 1 + f_idx % 3  # Stack flexible courses if there are many
                    x_position = f_idx // 3  # Spread out horizontally for better visibility
                    
                    fig.add_trace(go.Scatter(
                        x=[x_position],
                        y=[y_position],
                        mode='markers+text',
                        text=[course],
                        textposition='top center',
                        marker=dict(size=15, color='rgba(76, 175, 80, 0.8)', symbol='diamond'),
                        name=f"Flex: {course}"
                    ))
                    
                    # Add constraint lines if present
                    if course in flexible_constraints:
                        constraints = flexible_constraints[course]
                        before_courses = constraints.get('must_be_before', [])
                        after_courses = constraints.get('must_be_after', [])
                        
                        # Draw lines to before courses (must be taken before these)
                        for before_course in before_courses:
                            if before_course in current_path:
                                target_idx = current_path.index(before_course)
                                target_pos = visual_positions.get(before_course, 0)
                                fig.add_annotation(
                                    x=target_pos,
                                    y=0,
                                    ax=x_position,
                                    ay=y_position,
                                    xref='x',
                                    yref='y',
                                    axref='x',
                                    ayref='y',
                                    showarrow=True,
                                    arrowhead=2,
                                    arrowsize=1,
                                    arrowwidth=1,
                                    arrowcolor='rgba(76, 175, 80, 0.5)'
                                )
                        
                        # Draw lines from after courses (must be taken after these)
                        for after_course in after_courses:
                            if after_course in current_path:
                                source_idx = current_path.index(after_course)
                                source_pos = visual_positions.get(after_course, 0)
                                fig.add_annotation(
                                    x=source_pos,
                                    y=0,
                                    ax=x_position,
                                    ay=y_position,
                                    xref='x',
                                    yref='y',
                                    axref='x',
                                    ayref='y',
                                    showarrow=True,
                                    arrowhead=2,
                                    arrowsize=1,
                                    arrowwidth=1,
                                    arrowcolor='rgba(76, 175, 80, 0.5)'
                                )
                
                # Update layout
                fig.update_layout(
                    showlegend=False,
                    xaxis=dict(showticklabels=False, zeroline=False, showgrid=False),
                    yaxis=dict(showticklabels=False, zeroline=False, showgrid=False),
                    margin=dict(l=50, r=20, t=20, b=20),
                    height=max(400, 150 + len(or_groups) * 50 + len(flexible_courses) * 40),
                    width=max(800, (len(current_path) + len(or_groups)) * 100)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Add a text representation of the path
                with st.expander("Path Summary"):
                    if current_path:
                        st.write("**Main Path (in order):**")
                        st.write(" → ".join(current_path))
                    
                    if or_groups:
                        st.write("**OR Groups (take one course from each group):**")
                        for i, group in enumerate(or_groups):
                            if group:
                                group_id = f"group_{i}"
                                position = or_group_positions.get(group_id, 0)
                                position_desc = "At start of path" if position == 0 else f"After {current_path[position-1]}" if position <= len(current_path) else "At end of path"
                                st.write(f"Group {i+1} ({position_desc}): {' OR '.join(group)}")
                    
                    if flexible_courses:
                        st.write("**Flexible Courses (with constraints):**")
                        for course in flexible_courses:
                            constraints_text = []
                            if course in flexible_constraints:
                                before = flexible_constraints[course].get('must_be_before', [])
                                after = flexible_constraints[course].get('must_be_after', [])
                                
                                if after:
                                    constraints_text.append(f"After: {', '.join(after)}")
                                if before:
                                    constraints_text.append(f"Before: {', '.join(before)}")
                            
                            constraint_str = f" ({' & '.join(constraints_text)})" if constraints_text else ""
                            st.write(f"• {course}{constraint_str}")
            else:
                st.info("No courses in path yet. Add courses using the options above.")
    
    # Apply paths to configurations button
    st.subheader("Apply Career Paths to Configurations")
    st.write("This will update course prerequisites based on your custom career paths.")
    
    col1, col2 = st.columns(2)
    with col1:
        clear_existing = st.checkbox("Clear existing prerequisites", value=True,
                                   help="If checked, existing prerequisites will be cleared before applying the career paths.")
    with col2:
        update_mos_paths = st.checkbox("Update MOS paths", value=True,
                                     help="If checked, each course will be marked as part of the appropriate MOS path.")
    
    if st.button("Apply Career Paths to Prerequisites"):
        if 'course_configs' not in st.session_state or not st.session_state.course_configs:
            st.error("No course configurations found. Please go to the Course Configuration page first.")
        else:
            # Apply career paths to configurations
            try:
                with st.spinner("Applying career paths to configurations..."):
                    updated_configs, changes = apply_custom_paths_to_configs(
                        st.session_state.course_configs,
                        st.session_state.custom_paths
                    )
                
                if changes:
                    st.session_state.career_path_changes = changes
                    st.success(f"Made {len(changes)} updates to course configurations based on career paths!")
                    
                    # Show summary of changes
                    with st.expander("View Changes", expanded=True):
                        # Count changes by type
                        change_counts = defaultdict(int)
                        for change in changes:
                            change_counts[change['action']] += 1
                        
                        # Display counts
                        st.write("### Changes Summary")
                        for action, count in change_counts.items():
                            if action == 'set_prerequisites':
                                st.write(f"• Set prerequisites for {count} courses")
                            elif action == 'added_to_mos_path':
                                st.write(f"• Added {count} courses to MOS paths")
                            elif action == 'added_as_flexible_course':
                                st.write(f"• Added {count} flexible courses to MOS paths")
                            elif action == 'added_or_prerequisite':
                                st.write(f"• Added {count} OR prerequisites")
                            elif action == 'set_flexible_prerequisites':
                                st.write(f"• Set {count} flexible course constraints")
                            elif action == 'added_flexible_as_prerequisite':
                                st.write(f"• Added {count} flexible courses as prerequisites")
                        
                        # Option to see all changes
                        if st.checkbox("Show all changes"):
                            for change in changes:
                                course = change['course']
                                mos = change['mos']
                                action = change['action']
                                
                                if action == 'set_prerequisites':
                                    prereqs = change['prerequisites']
                                    st.write(f"• Set prerequisites for {course} ({mos}): {', '.join(prereqs)}")
                                elif action == 'added_to_mos_path':
                                    st.write(f"• Added {course} to {mos} career path")
                                elif action == 'added_as_flexible_course':
                                    st.write(f"• Added {course} as a flexible course for {mos}")
                                elif action == 'added_or_prerequisite':
                                    or_group = change['or_group']
                                    st.write(f"• Added OR prerequisite to {course} ({mos}): {' OR '.join(or_group)}")
                                elif action == 'set_flexible_prerequisites':
                                    prereqs = change['prerequisites']
                                    st.write(f"• Set prerequisites for flexible course {course} ({mos}): {', '.join(prereqs)}")
                                elif action == 'added_flexible_as_prerequisite':
                                    prereq = change['prerequisite_added']
                                    st.write(f"• Added flexible course {prereq} as prerequisite for {course} ({mos})")
                    
                    # Option to apply changes
                    if st.button("Confirm and Update Configurations"):
                        st.session_state.course_configs = updated_configs
                        st.success("Course configurations updated successfully!")
                        st.rerun()
                else:
                    st.info("No changes needed. Course configurations already match the career paths.")
            except Exception as e:
                st.error(f"Error applying career paths: {str(e)}")
                import traceback
                st.error(traceback.format_exc())
    
    # Add navigation to next page
    st.write("---")
    st.write("Next Steps:")
    st.info("Once you've defined your career paths and applied them to your configurations, proceed to the Course Configuration page to review and further customize your course settings.")
    
    # Add a direct link button
    if st.button("Go to Course Configuration", use_container_width=True):
        # Store a flag in session state
        st.session_state.current_page = "Course Configuration"
        st.rerun()

def display_config_page():
    st.header("Course Configuration")
    
    if 'processed_data' not in st.session_state or st.session_state.processed_data is None:
        st.warning("Please upload and process data first.")
        
        # Navigation button to upload page
        if st.button("Go to Upload Data", use_container_width=True):
            st.session_state.current_page = "Upload Data"
            st.rerun()
        return
    
    # Get unique courses
    unique_courses = st.session_state.processed_data['Course Title'].unique()
    
    # Add visualization of course relationships
    with st.expander("Course Prerequisite Relationships Overview", expanded=False):
        if st.session_state.course_configs:
            st.write("This visualization shows how courses are related based on current configurations:")
            prereq_fig = visualize_prerequisites(st.session_state.course_configs)
            st.plotly_chart(prereq_fig, use_container_width=True)
        else:
            st.info("No course configurations available yet. Configure courses to see relationships.")
    
    # Course selection
    selected_course = st.selectbox("Select Course to Configure", unique_courses)
    
    # Create or load configuration for the selected course
    if selected_course not in st.session_state.course_configs:
        # Initialize with default values or from historical data if available
        historical_data = st.session_state.historical_analysis.get(selected_course, {})
        # Ensure default values are all integers
        default_class_size = int(historical_data.get('avg_class_size', 50))
        default_classes_per_year = int(historical_data.get('classes_per_year', 4))
        st.session_state.course_configs[selected_course] = {
            'prerequisites': {
                'type': 'AND',  # Default to AND logic (all prerequisites required)
                'courses': []   # List of required courses (for AND)
            },
            'or_prerequisites': [],  # List of lists, each inner list is a set of OR prerequisites
            'mos_paths': {
                '18A': [],  # Officer path
                '18B': [],  # Weapons Sergeant path
                '18C': [],  # Engineer Sergeant path
                '18D': [],  # Medical Sergeant path
                '18E': []   # Communications Sergeant path
            },
            'required_for_all_mos': False,  # Whether this course is required for all MOS paths
            'max_capacity': default_class_size,
            'classes_per_year': default_classes_per_year,
            'reserved_seats': {
                'OF': 0,
                'ADE': 0,
                'NG': 0
            },
            'officer_enlisted_ratio': None,  # Default to No Ratio
            'use_even_mos_ratio': False  # Default to not using even MOS ratio
        }
    
    config = st.session_state.course_configs[selected_course]
    
    # Ensure numeric values are integers to avoid type conflicts
    if not isinstance(config['max_capacity'], int):
        config['max_capacity'] = int(config['max_capacity'])
    if not isinstance(config['classes_per_year'], int):
        config['classes_per_year'] = int(config['classes_per_year'])
    
    # Backward compatibility check for older config format
    if isinstance(config.get('prerequisites'), list):
        # Convert old format to new format
        config['prerequisites'] = {
            'type': 'AND',
            'courses': config['prerequisites']
        }
        config['or_prerequisites'] = []
    
    # Add MOS paths if they don't exist
    if 'mos_paths' not in config:
        config['mos_paths'] = {
            '18A': [],
            '18B': [],
            '18C': [],
            '18D': [],
            '18E': []
        }
        config['required_for_all_mos'] = False
    
    # Configure prerequisite type
    st.subheader("Prerequisites")
    with st.expander("How Prerequisites Work"):
        st.markdown("""
        ### Prerequisite Types:
        1. **All Required (AND)**: Student must complete ALL of the specified courses before taking this course.
        2. **Any Required (OR)**: Student must complete ANY ONE of the specified courses before taking this course.
        3. **Complex (AND/OR)**: For advanced prerequisite relationships. You can define:
           - Courses that ALL must be completed (AND logic)
           - Groups of courses where ANY ONE from EACH group must be completed (OR logic within groups, AND logic between groups)
        4. **MOS-Specific Paths**: Define different prerequisites for each MOS training path (18A, 18B, 18C, 18D, 18E)
        
        #### Example:
        If you want to require either "Course A" OR "Course B", AND either "Course C" OR "Course D":
        1. Leave the "Student must complete ALL of these courses" field empty
        2. Create Group 1 with "Course A" and "Course B"
        3. Create Group 2 with "Course C" and "Course D"
        """)
    
    prerequisite_type = st.radio(
        "Prerequisite Configuration Method",
        ["General Prerequisites", "MOS-Specific Paths"]
    )
    
    prerequisite_options = [c for c in unique_courses if c != selected_course]
    
    # Use session state to prevent the dropdown selection from resetting
    if 'temp_prerequisites' not in st.session_state:
        # Initialize with current config
        if prerequisite_type == "General Prerequisites":
            st.session_state.temp_prerequisites = config['prerequisites']['courses']
        else:
            st.session_state.temp_prerequisites = {}
            for mos in ['18A', '18B', '18C', '18D', '18E']:
                st.session_state.temp_prerequisites[mos] = config['mos_paths'].get(mos, [])
    
    if prerequisite_type == "General Prerequisites":
        # Use the existing AND/OR prerequisite system
        prereq_logic = st.radio(
            "Prerequisite Logic",
            ["All Required (AND)", "Any Required (OR)", "Complex (AND/OR)"],
            index=0 if config['prerequisites']['type'] == 'AND' and not config['or_prerequisites'] else
                1 if config['prerequisites']['type'] == 'OR' and not config['or_prerequisites'] else 2
        )
        
        if prereq_logic == "All Required (AND)":
            # Simple AND logic - all courses must be taken
            config['prerequisites']['type'] = 'AND'
            selected_prerequisites = st.multiselect(
                "Student must complete ALL of these courses:",
                prerequisite_options,
                default=st.session_state.temp_prerequisites,
                key="and_prereqs"
            )
            st.session_state.temp_prerequisites = selected_prerequisites
            
            # Update button to commit changes
            if st.button("Update Prerequisites", key="update_and_prereqs"):
                config['prerequisites']['courses'] = st.session_state.temp_prerequisites
                config['or_prerequisites'] = []  # Clear any OR prerequisites
                st.success("Prerequisites updated!")
        
        elif prereq_logic == "Any Required (OR)":
            # Simple OR logic - any course can be taken
            config['prerequisites']['type'] = 'OR'
            selected_prerequisites = st.multiselect(
                "Student must complete ANY ONE of these courses:",
                prerequisite_options,
                default=st.session_state.temp_prerequisites,
                key="or_prereqs"
            )
            st.session_state.temp_prerequisites = selected_prerequisites
            
            # Update button to commit changes
            if st.button("Update Prerequisites", key="update_or_prereqs"):
                config['prerequisites']['courses'] = st.session_state.temp_prerequisites
                config['or_prerequisites'] = []  # Clear any complex OR prerequisites
                st.success("Prerequisites updated!")
        
        else:  # Complex AND/OR
            # For complex logic, we'll use the or_prerequisites structure
            st.write("Define complex prerequisite relationships:")
            
            # First, define any required courses (AND logic)
            if 'temp_and_prereqs' not in st.session_state:
                st.session_state.temp_and_prereqs = config['prerequisites']['courses'] if config['prerequisites']['type'] == 'AND' else []
            
            and_prerequisites = st.multiselect(
                "Student must complete ALL of these courses (leave empty if none):",
                prerequisite_options,
                default=st.session_state.temp_and_prereqs,
                key="complex_and_prereqs"
            )
            st.session_state.temp_and_prereqs = and_prerequisites
            
            # Then, define sets of OR prerequisites
            st.write("AND the student must complete at least one course from EACH of the following groups:")
            
            # Initialize or_prerequisites if needed
            if 'temp_or_groups' not in st.session_state:
                st.session_state.temp_or_groups = config['or_prerequisites'].copy() if config['or_prerequisites'] else [[]]
            
            # Display existing OR groups
            or_groups = st.session_state.temp_or_groups.copy()
            updated_or_groups = []
            
            for i, group in enumerate(or_groups):
                col1, col2 = st.columns([4, 1])
                with col1:
                    selected_or_courses = st.multiselect(
                        f"Group {i+1}: Student must complete ANY ONE of these courses:",
                        [c for c in prerequisite_options if c not in and_prerequisites],
                        default=group,
                        key=f"or_group_{i}"
                    )
                    if selected_or_courses:  # Only add non-empty groups
                        updated_or_groups.append(selected_or_courses)
                
                with col2:
                    if st.button(f"Remove Group {i+1}", key=f"remove_group_{i}"):
                        pass  # We'll filter out this group by not adding it to updated_or_groups
            
            # Add a button to add a new OR group
            if st.button("Add Another OR Group"):
                updated_or_groups.append([])  # Add an empty group
            
            st.session_state.temp_or_groups = updated_or_groups
            
            # Update button to commit changes
            if st.button("Update Prerequisites", key="update_complex_prereqs"):
                config['prerequisites'] = {
                    'type': 'AND',
                    'courses': st.session_state.temp_and_prereqs
                }
                config['or_prerequisites'] = st.session_state.temp_or_groups
                st.success("Complex prerequisites updated!")
    
    else:  # MOS-Specific Paths
        st.write("Configure prerequisites for each MOS training path:")
        
        # Option to make this course required for all MOS paths
        required_for_all = st.checkbox("This course is required for all MOS paths",
                                      value=config.get('required_for_all_mos', False))
        config['required_for_all_mos'] = required_for_all
        
        if required_for_all:
            st.info("This course will be required for all students regardless of MOS path.")
        
        # Initialize the temp state for MOS paths if needed
        if 'temp_mos_paths' not in st.session_state:
            st.session_state.temp_mos_paths = {}
            for mos in ['18A', '18B', '18C', '18D', '18E']:
                st.session_state.temp_mos_paths[mos] = config['mos_paths'].get(mos, [])
        
        # Configure MOS-specific prerequisites
        mos_tabs = st.tabs(['18A (Officer)', '18B (Weapons)', '18C (Engineer)', '18D (Medical)', '18E (Communications)'])
        
        for i, mos in enumerate(['18A', '18B', '18C', '18D', '18E']):
            with mos_tabs[i]:
                st.write(f"Configure prerequisites for {mos} path:")
                
                # Check if this course is part of this MOS path
                is_in_path = st.checkbox(f"This course is part of the {mos} training path",
                                        value=len(config['mos_paths'].get(mos, [])) > 0 or selected_course in config['mos_paths'].get(mos, []))
                
                if is_in_path:
                    # If it's part of the path, configure prerequisites for this MOS
                    mos_prereqs = st.multiselect(
                        f"Prerequisites for {mos} path:",
                        prerequisite_options,
                        default=st.session_state.temp_mos_paths.get(mos, []),
                        key=f"mos_prereqs_{mos}"
                    )
                    st.session_state.temp_mos_paths[mos] = mos_prereqs
                else:
                    # If not part of this MOS path, clear prerequisites
                    st.session_state.temp_mos_paths[mos] = []
        
        # Update button to commit changes
        if st.button("Update MOS Paths", key="update_mos_paths"):
            for mos in ['18A', '18B', '18C', '18D', '18E']:
                config['mos_paths'][mos] = st.session_state.temp_mos_paths.get(mos, [])
            st.success("MOS paths updated!")
    
    # Display the current prerequisite structure for clarity
    st.subheader("Current Prerequisites Configuration")
    
    if prerequisite_type == "General Prerequisites":
        if prereq_logic == "All Required (AND)" and config['prerequisites']['courses']:
            st.write("Student must complete ALL of these courses:")
            st.write(", ".join(config['prerequisites']['courses']))
        elif prereq_logic == "Any Required (OR)" and config['prerequisites']['courses']:
            st.write("Student must complete ANY ONE of these courses:")
            st.write(", ".join(config['prerequisites']['courses']))
        else:
            if config['prerequisites']['courses']:
                st.write("Student must complete ALL of these courses:")
                st.write(", ".join(config['prerequisites']['courses']))
            
            if config['or_prerequisites']:
                st.write("AND at least one course from EACH of these groups:")
                for i, group in enumerate(config['or_prerequisites']):
                    if group:  # Only show non-empty groups
                        st.write(f"Group {i+1}: {' OR '.join(group)}")
    else:  # MOS-Specific Paths
        if config['required_for_all_mos']:
            st.write("This course is required for all MOS paths.")
        
        st.write("MOS-specific prerequisites:")
        for mos in ['18A', '18B', '18C', '18D', '18E']:
            prereqs = config['mos_paths'].get(mos, [])
            if prereqs:
                st.write(f"{mos}: {', '.join(prereqs)}")
            else:
                if config['required_for_all_mos']:
                    st.write(f"{mos}: No additional prerequisites")
                else:
                    st.write(f"{mos}: Not in this training path")
    
    # Configure capacity
    st.subheader("Class Capacity")
    
    max_capacity = st.number_input(
        "Maximum number of students per class",
        min_value=1,
        value=int(config['max_capacity'])  # Explicitly convert to int
    )
    config['max_capacity'] = int(max_capacity)  # Ensure it's stored as int
    
    # Configure classes per year
    st.subheader("Schedule Frequency")
    
    classes_per_year = st.number_input(
        "Number of classes per fiscal year (Oct 1 - Sep 30)",
        min_value=1,
        value=int(config['classes_per_year'])  # Explicitly convert to int
    )
    config['classes_per_year'] = int(classes_per_year)  # Ensure it's stored as int
    
    # Configure reserved seats
    st.subheader("Reserved Seats")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Ensure reserved seats are integers
        if not isinstance(config['reserved_seats']['OF'], int):
            config['reserved_seats']['OF'] = int(config['reserved_seats']['OF'])
        
        of_seats = st.number_input(
            "OF (Officer) Seats",
            min_value=0,
            max_value=int(max_capacity),
            value=int(config['reserved_seats']['OF'])
        )
        config['reserved_seats']['OF'] = int(of_seats)
    
    with col2:
        # Ensure reserved seats are integers
        if not isinstance(config['reserved_seats']['ADE'], int):
            config['reserved_seats']['ADE'] = int(config['reserved_seats']['ADE'])
        
        ade_seats = st.number_input(
            "ADE (Active Duty Enlisted) Seats",
            min_value=0,
            max_value=int(max_capacity),
            value=int(config['reserved_seats']['ADE'])
        )
        config['reserved_seats']['ADE'] = int(ade_seats)
    
    with col3:
        # Ensure reserved seats are integers
        if not isinstance(config['reserved_seats']['NG'], int):
            config['reserved_seats']['NG'] = int(config['reserved_seats']['NG'])
        
        ng_seats = st.number_input(
            "NG (National Guard) Seats",
            min_value=0,
            max_value=int(max_capacity),
            value=int(config['reserved_seats']['NG'])
        )
        config['reserved_seats']['NG'] = int(ng_seats)
    
    # Validate reserved seats
    total_reserved = of_seats + ade_seats + ng_seats
    if total_reserved > max_capacity:
        st.warning(f"Total reserved seats ({total_reserved}) exceeds maximum capacity ({max_capacity})")
    
    # Officer to Enlisted ratio
    st.subheader("Officer to Enlisted Ratio")
    
    with st.expander("About Officer-Enlisted Ratio"):
        st.markdown("""
        **No Ratio**: The class will accept any mix of officers and enlisted personnel up to capacity.
        
        **1:4 Ratio**: For every 1 officer, the class should have 4 enlisted personnel. The system will try to maintain this ratio when assigning students.
        
        **Custom Ratio**: Enter your own ratio in the format "1:5" (1 officer to 5 enlisted).
        
        Note: Ratios are maintained as students are added to classes. If a student would cause the ratio to deviate too far from the target, they may not be accepted into the class.
        """)
    
    ratio_options = ["No Ratio", "1:1", "1:2", "1:3", "1:4", "1:5", "1:8", "1:10", "Custom"]
    
    # Get current ratio setting
    current_ratio = config['officer_enlisted_ratio']
    current_index = 0  # Default to "No Ratio"
    
    # Determine which option to select by default
    if current_ratio:  # If not None or empty
        if current_ratio in ratio_options:
            current_index = ratio_options.index(current_ratio)
        else:
            current_index = ratio_options.index("Custom")
    
    selected_ratio = st.selectbox(
        "Select ratio",
        ratio_options,
        index=current_index
    )
    
    if selected_ratio == "Custom":
        custom_ratio = st.text_input("Enter custom ratio (format: 1:4)",
                                    value=current_ratio if current_ratio and current_ratio not in ratio_options else "1:4")
        config['officer_enlisted_ratio'] = custom_ratio
    
    elif selected_ratio == "No Ratio":
        config['officer_enlisted_ratio'] = None  # Store None for no ratio
        st.info("No officer-to-enlisted ratio will be enforced for this course.")
    
    else:
        config['officer_enlisted_ratio'] = selected_ratio
    
    # Add MOS Allocation Settings
    st.subheader("MOS Allocation Settings")
    
    use_even_mos_ratio = st.checkbox(
        "Use equal allocation for all MOS paths",
        value=config.get('use_even_mos_ratio', False),
        help="When enabled, the system will automatically allocate seats evenly across all MOS paths (18A, 18B, 18C, 18D, 18E)"
    )
    config['use_even_mos_ratio'] = use_even_mos_ratio
    
    if use_even_mos_ratio:
        st.info("Classes will have seats allocated equally among all MOS paths. This setting will override individual MOS allocations in the Schedule Builder.")
    
    # Show historical metrics for this course
    if selected_course in st.session_state.historical_analysis:
        st.subheader("Historical Data")
        
        hist_data = st.session_state.historical_analysis[selected_course]
        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
        
        with metrics_col1:
            st.metric("Pass Rate", f"{hist_data.get('pass_rate', 0):.1%}")
        
        with metrics_col2:
            st.metric("Avg. Duration", f"{hist_data.get('avg_duration', 0):.1f} days")
        
        with metrics_col3:
            st.metric("Recycle Rate", f"{hist_data.get('recycle_rate', 0):.1%}")
    
    # Save configuration button
    if st.button("Save Configuration", use_container_width=True):
        st.session_state.course_configs[selected_course] = config
        st.success(f"Configuration for {selected_course} saved successfully!")
    
    # After the "Save Configuration" button, add save/load functionality for all configurations
    st.subheader("Save/Load All Course Configurations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Save All Configurations", use_container_width=True):
            # Convert to JSON
            config_json = json.dumps(st.session_state.course_configs, indent=2, default=lambda o: str(o) if isinstance(o, (datetime.date, datetime.datetime)) else o)
            
            # Provide download button
            st.download_button(
                label="Download Configurations JSON",
                data=config_json,
                file_name="course_configurations.json",
                mime="application/json",
                use_container_width=True
            )
    
    with col2:
        uploaded_configs = st.file_uploader("Upload Configurations JSON", type=["json"], key="config_uploader")
        
        if uploaded_configs is not None:
            try:
                loaded_configs = json.load(uploaded_configs)
                
                # Apply backward compatibility
                for course, config in loaded_configs.items():
                    ensure_config_compatibility(config)
                
                # Confirm before overwriting
                if st.checkbox("Overwrite existing configurations?", key="overwrite_configs"):
                    st.session_state.course_configs = loaded_configs
                    st.success("Configurations loaded successfully!")
                    st.rerun()
                else:
                    # Merge with existing configs
                    st.session_state.course_configs.update(loaded_configs)
                    st.success("Configurations merged successfully!")
                    st.rerun()
            
            except Exception as e:
                st.error(f"Error loading configurations: {e}")
    
    # Navigation buttons
    st.write("---")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("← Back to Career Path Builder", use_container_width=True):
            st.session_state.current_page = "Career Path Builder"
            st.rerun()
    
    with col2:
        if st.button("Continue to Schedule Builder →", use_container_width=True):
            st.session_state.current_page = "Schedule Builder"
            st.rerun()

def display_schedule_builder():
    st.header("Future Schedule Builder")
    
    if not st.session_state.course_configs:
        st.warning("Please configure courses before building a schedule.")
        
        if st.button("Go to Course Configuration", use_container_width=True):
            st.session_state.current_page = "Course Configuration"
            st.rerun()
        return
    
    # Schedule settings
    st.subheader("Schedule Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        start_date = st.date_input("Schedule Start Date", value=datetime.date.today())
    
    with col2:
        end_date = st.date_input("Schedule End Date", value=datetime.date.today() + datetime.timedelta(days=365))
    
    if start_date >= end_date:
        st.error("End date must be after start date.")
        return
    
    # Initialize schedule if empty
    if not st.session_state.future_schedule:
        st.session_state.future_schedule = []
    
    # Available courses
    st.subheader("Add Course to Schedule")
    
    course_title = st.selectbox("Select Course", list(st.session_state.course_configs.keys()))
    
    # Show historical duration for selected course
    if course_title in st.session_state.historical_analysis:
        hist_data = st.session_state.historical_analysis[course_title]
        avg_duration = hist_data.get('avg_duration', 14)  # Default to 14 days if not available
        st.info(f"Historical average duration for {course_title}: {avg_duration:.1f} days")
    
    # Quick date setter buttons
    with st.expander("Quick Date Setter"):
        col1, col2, col3, col4 = st.columns(4)
        
        # Get historical duration for this course if available
        default_duration = 14  # Default if no historical data
        
        if course_title in st.session_state.historical_analysis:
            hist_duration = st.session_state.historical_analysis[course_title].get('avg_duration')
            if hist_duration:
                default_duration = int(hist_duration)
        
        with col1:
            if st.button("1 Week", use_container_width=True):
                st.session_state.temp_start = start_date
                st.session_state.temp_end = start_date + datetime.timedelta(days=7)
                st.rerun()
        
        with col2:
            if st.button("2 Weeks", use_container_width=True):
                st.session_state.temp_start = start_date
                st.session_state.temp_end = start_date + datetime.timedelta(days=14)
                st.rerun()
        
        with col3:
            if st.button("1 Month", use_container_width=True):
                st.session_state.temp_start = start_date
                st.session_state.temp_end = start_date + datetime.timedelta(days=30)
                st.rerun()
        
        with col4:
            if st.button(f"Historical ({default_duration} days)", use_container_width=True):
                st.session_state.temp_start = start_date
                st.session_state.temp_end = start_date + datetime.timedelta(days=default_duration)
                st.rerun()
    
    # Date and size inputs
    col1, col2, col3 = st.columns(3)
    
    with col1:
        default_start = st.session_state.temp_start if 'temp_start' in st.session_state else start_date
        class_start_date = st.date_input("Class Start Date", value=default_start)
        
        # Clear the temporary value after using it
        if 'temp_start' in st.session_state:
            del st.session_state.temp_start
    
    with col2:
        # Set default end date based on historical duration if available
        default_duration = 14  # Default duration in days
        
        if course_title in st.session_state.historical_analysis:
            hist_duration = st.session_state.historical_analysis[course_title].get('avg_duration')
            if hist_duration:
                default_duration = int(hist_duration)
        
        default_end = st.session_state.temp_end if 'temp_end' in st.session_state else (class_start_date + datetime.timedelta(days=default_duration))
        class_end_date = st.date_input("Class End Date", value=default_end)
        
        # Clear the temporary value after using it
        if 'temp_end' in st.session_state:
            del st.session_state.temp_end
    
    with col3:
        class_size = st.number_input(
            "Class Size",
            min_value=1,
            max_value=st.session_state.course_configs[course_title]['max_capacity'],
            value=st.session_state.course_configs[course_title]['max_capacity']
        )
    
    # Show duration between selected dates
    if class_start_date and class_end_date:
        duration_days = (class_end_date - class_start_date).days
        st.info(f"Selected duration: {duration_days} days")
        
        # Add warning if the duration is very different from historical average
        if course_title in st.session_state.historical_analysis:
            hist_duration = st.session_state.historical_analysis[course_title].get('avg_duration')
            if hist_duration:
                if duration_days < hist_duration * 0.7 or duration_days > hist_duration * 1.3:
                    st.warning(f"The selected duration ({duration_days} days) is significantly different from the historical average ({hist_duration:.1f} days).")
    
    # Add multiple class scheduling option
    st.subheader("Multiple Class Scheduling")
    
    with st.expander("Add Multiple Classes for Same Course", expanded=False):
        st.write("Quickly add multiple instances of the same course throughout the year")
        
        # Date preset options
        preset_options = [
            "Quarterly (4 classes per year)",
            "Bimonthly (6 classes per year)",
            "Monthly (12 classes per year)",
            "Custom"
        ]
        date_preset = st.selectbox("Quick Date Preset", preset_options, index=3)  # Default to Custom
        
        # Number of classes to add
        if date_preset == "Quarterly (4 classes per year)":
            num_classes = 4
            preset_months = [1, 4, 7, 10]  # Jan, Apr, Jul, Oct
        elif date_preset == "Bimonthly (6 classes per year)":
            num_classes = 6
            preset_months = [1, 3, 5, 7, 9, 11]  # Jan, Mar, May, Jul, Sep, Nov
        elif date_preset == "Monthly (12 classes per year)":
            num_classes = 12
            preset_months = list(range(1, 13))  # All months
        else:  # Custom
            num_classes = st.number_input("Number of Classes to Add", min_value=1, max_value=10, value=3)
            preset_months = None
        
        # Calculate default duration
        default_duration = 14  # Default if no historical data
        
        if course_title in st.session_state.historical_analysis:
            hist_duration = st.session_state.historical_analysis[course_title].get('avg_duration')
            if hist_duration:
                default_duration = int(hist_duration)
        
        # Initialize lists to store all dates
        all_start_dates = []
        all_end_dates = []
        
        # Generate dates based on preset or custom
        if preset_months:
            current_year = start_date.year
            for month in preset_months:
                # Create date for first day of month
                month_start = datetime.date(current_year, month, 1)
                
                # If this date is before the schedule start, use next year
                if month_start < start_date:
                    month_start = datetime.date(current_year + 1, month, 1)
                
                # Set end date based on duration
                month_end = month_start + datetime.timedelta(days=default_duration)
                
                all_start_dates.append(month_start)
                all_end_dates.append(month_end)
        
        # Create columns for each class
        col_count = 3  # Show 3 classes per row
        rows = (num_classes + col_count - 1) // col_count  # Calculate how many rows we need
        
        for row in range(rows):
            cols = st.columns(col_count)
            
            for col_idx in range(col_count):
                class_idx = row * col_count + col_idx
                
                if class_idx < num_classes:
                    with cols[col_idx]:
                        st.write(f"**Class {class_idx + 1}**")
                        
                        # Default dates (if not using presets)
                        if not preset_months:
                            # Space classes throughout the year
                            days_in_year = 365
                            days_offset = int(class_idx * (days_in_year / num_classes))
                            default_start = start_date + datetime.timedelta(days=days_offset)
                            default_end = default_start + datetime.timedelta(days=default_duration)
                        else:
                            # Use the preset dates
                            default_start = all_start_dates[class_idx] if class_idx < len(all_start_dates) else start_date
                            default_end = all_end_dates[class_idx] if class_idx < len(all_end_dates) else (default_start + datetime.timedelta(days=default_duration))
                        
                        # Date inputs for this class
                        class_start = st.date_input(f"Start Date", value=default_start, key=f"multi_start_{class_idx}")
                        class_end = st.date_input(f"End Date", value=default_end, key=f"multi_end_{class_idx}")
                        
                        # Validate dates
                        if class_start >= class_end:
                            st.error("End date must be after start date")
                        else:
                            all_start_dates.append(class_start)
                            all_end_dates.append(class_end)
        
        # Use the same size for all classes
        use_same_size = st.checkbox("Use Same Size for All Classes", value=True)
        
        multi_class_size = st.number_input(
            "Class Size for All",
            min_value=1,
            max_value=st.session_state.course_configs[course_title]['max_capacity'],
            value=st.session_state.course_configs[course_title]['max_capacity'],
            key="multi_class_size"
        )
        
        # MOS allocation for multiple classes
        course_config = st.session_state.course_configs.get(course_title, {})
        use_even_mos_ratio = course_config.get('use_even_mos_ratio', False)
        
        if use_even_mos_ratio:
            st.info(f"Using equal allocation for all MOS paths. Each MOS will receive {multi_class_size // 5} seats.")
            
            multi_mos_allocation = {
                '18A': multi_class_size // 5,
                '18B': multi_class_size // 5,
                '18C': multi_class_size // 5,
                '18D': multi_class_size // 5,
                '18E': multi_class_size // 5,
            }
            
            # Handle any remainder
            remainder = multi_class_size - (multi_class_size // 5 * 5)
            for i, mos in enumerate(['18A', '18B', '18C', '18D', '18E']):
                if i < remainder:
                    multi_mos_allocation[mos] += 1
        else:
            # Manual MOS allocation
            st.write("Specify MOS allocation to use for all classes:")
            multi_cols = st.columns(5)
            multi_mos_allocation = {}
            multi_total_allocated = 0
            
            with multi_cols[0]:
                multi_mos_18a = st.number_input("18A (Officer)", min_value=0, max_value=int(multi_class_size), value=0, key="multi_18a")
                multi_mos_allocation['18A'] = multi_mos_18a
                multi_total_allocated += multi_mos_18a
            
            with multi_cols[1]:
                multi_mos_18b = st.number_input("18B (Weapons)", min_value=0, max_value=int(multi_class_size), value=0, key="multi_18b")
                multi_mos_allocation['18B'] = multi_mos_18b
                multi_total_allocated += multi_mos_18b
            
            with multi_cols[2]:
                multi_mos_18c = st.number_input("18C (Engineer)", min_value=0, max_value=int(multi_class_size), value=0, key="multi_18c")
                multi_mos_allocation['18C'] = multi_mos_18c
                multi_total_allocated += multi_mos_18c
            
            with multi_cols[3]:
                multi_mos_18d = st.number_input("18D (Medical)", min_value=0, max_value=int(multi_class_size), value=0, key="multi_18d")
                multi_mos_allocation['18D'] = multi_mos_18d
                multi_total_allocated += multi_mos_18d
            
            with multi_cols[4]:
                multi_mos_18e = st.number_input("18E (Communications)", min_value=0, max_value=int(multi_class_size), value=0, key="multi_18e")
                multi_mos_allocation['18E'] = multi_mos_18e
                multi_total_allocated += multi_mos_18e
            
            # Validate MOS allocation
            if multi_total_allocated > multi_class_size:
                st.error(f"Total MOS allocation ({multi_total_allocated}) exceeds class size ({multi_class_size}).")
            elif multi_total_allocated < multi_class_size:
                st.warning(f"Total MOS allocation ({multi_total_allocated}) is less than class size ({multi_class_size}). {multi_class_size - multi_total_allocated} seats are unallocated.")
            else:
                st.success(f"MOS allocation complete ({multi_total_allocated}/{multi_class_size} seats allocated)")
        
        # Add all classes button
        if st.button("Add All Classes", use_container_width=True):
            with st.spinner("Adding classes..."):
                classes_added = 0
                
                # Make sure we only use the date pairs corresponding to the number of classes
                valid_date_pairs = []
                for i in range(min(num_classes, len(all_start_dates))):
                    if i < len(all_start_dates) and i < len(all_end_dates):
                        valid_date_pairs.append((all_start_dates[i], all_end_dates[i]))
                
                for i, (start_date, end_date) in enumerate(valid_date_pairs):
                    if start_date < end_date:  # Validate dates
                        new_class = {
                            "course_title": course_title,
                            "start_date": start_date.strftime("%Y-%m-%d"),
                            "end_date": end_date.strftime("%Y-%m-%d"),
                            "size": int(multi_class_size) if use_same_size else int(class_size),
                            "mos_allocation": multi_mos_allocation.copy(),
                            "id": len(st.session_state.future_schedule) + 1 + i
                        }
                        st.session_state.future_schedule.append(new_class)
                        classes_added += 1
                
                if classes_added > 0:
                    st.success(f"Added {classes_added} {course_title} classes to the schedule")
                    st.rerun()  # Refresh to show updated schedule
    
    # Add Training MOS allocation
    st.subheader("Training MOS Allocation for Single Class")
    
    # Check if the course is configured to use even MOS ratio
    course_config = st.session_state.course_configs.get(course_title, {})
    use_even_mos_ratio = course_config.get('use_even_mos_ratio', False)
    
    if use_even_mos_ratio:
        st.info(f"Using equal allocation for all MOS paths as configured in Course Configuration. Each MOS will receive {class_size // 5} seats.")
        
        # Create an even distribution across all MOS paths
        mos_allocation = {
            '18A': class_size // 5,
            '18B': class_size // 5,
            '18C': class_size // 5,
            '18D': class_size // 5,
            '18E': class_size // 5,
        }
        
        # Handle any remainder
        remainder = class_size - (class_size // 5 * 5)
        for i, mos in enumerate(['18A', '18B', '18C', '18D', '18E']):
            if i < remainder:
                mos_allocation[mos] += 1
        
        # Display the allocation
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("18A (Officer)", mos_allocation['18A'])
        
        with col2:
            st.metric("18B (Weapons)", mos_allocation['18B'])
        
        with col3:
            st.metric("18C (Engineer)", mos_allocation['18C'])
        
        with col4:
            st.metric("18D (Medical)", mos_allocation['18D'])
        
        with col5:
            st.metric("18E (Communications)", mos_allocation['18E'])
        
        total_allocated = sum(mos_allocation.values())
        st.success(f"MOS allocation complete ({total_allocated}/{class_size} seats allocated)")
    
    else:
        # Original MOS allocation code for manual input
        st.write("Specify the number of seats reserved for each MOS path:")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        mos_allocation = {}
        total_allocated = 0
        
        with col1:
            mos_18a = st.number_input("18A (Officer)", min_value=0, max_value=int(class_size), value=0)
            mos_allocation['18A'] = mos_18a
            total_allocated += mos_18a
        
        with col2:
            mos_18b = st.number_input("18B (Weapons)", min_value=0, max_value=int(class_size), value=0)
            mos_allocation['18B'] = mos_18b
            total_allocated += mos_18b
        
        with col3:
            mos_18c = st.number_input("18C (Engineer)", min_value=0, max_value=int(class_size), value=0)
            mos_allocation['18C'] = mos_18c
            total_allocated += mos_18c
        
        with col4:
            mos_18d = st.number_input("18D (Medical)", min_value=0, max_value=int(class_size), value=0)
            mos_allocation['18D'] = mos_18d
            total_allocated += mos_18d
        
        with col5:
            mos_18e = st.number_input("18E (Communications)", min_value=0, max_value=int(class_size), value=0)
            mos_allocation['18E'] = mos_18e
            total_allocated += mos_18e
        
        # Validate MOS allocation
        if total_allocated > class_size:
            st.error(f"Total MOS allocation ({total_allocated}) exceeds class size ({class_size}).")
        elif total_allocated < class_size:
            st.warning(f"Total MOS allocation ({total_allocated}) is less than class size ({class_size}). {class_size - total_allocated} seats are unallocated.")
        else:
            st.success(f"MOS allocation complete ({total_allocated}/{class_size} seats allocated)")
    
    # Validate dates
    if class_start_date >= class_end_date:
        st.error("Class end date must be after start date.")
    else:
        # Add class button
        if st.button("Add Class to Schedule", use_container_width=True):
            new_class = {
                "course_title": course_title,
                "start_date": class_start_date.strftime("%Y-%m-%d"),
                "end_date": class_end_date.strftime("%Y-%m-%d"),
                "size": int(class_size),  # Ensure it's stored as int
                "mos_allocation": mos_allocation,
                "id": len(st.session_state.future_schedule) + 1
            }
            st.session_state.future_schedule.append(new_class)
            st.success(f"Added {course_title} from {class_start_date} to {class_end_date}")
            st.rerun()  # Refresh to show updated schedule
    
    # Display current schedule
    st.subheader("Current Schedule")
    
    if st.session_state.future_schedule:
        # Create a more efficient display with caching
        @st.cache_data(ttl=5)
        def prepare_schedule_display(schedule):
            """Prepare schedule data for display with caching for performance"""
            schedule_df = pd.DataFrame(schedule)
            
            # Convert date strings to datetime for plotting
            try:
                schedule_df['start_date'] = pd.to_datetime(schedule_df['start_date'])
                schedule_df['end_date'] = pd.to_datetime(schedule_df['end_date'])
                return schedule_df
            except Exception as e:
                st.error(f"Error preparing schedule data: {e}")
                return pd.DataFrame()
        
        schedule_df = prepare_schedule_display(st.session_state.future_schedule)
        
        if not schedule_df.empty:
            # Filter options
            with st.expander("Chart Options"):
                # Filter by course
                unique_courses = schedule_df['course_title'].unique()
                
                # Store filter selections in session state to prevent losing selection on interaction
                if 'selected_schedule_courses' not in st.session_state:
                    st.session_state.selected_schedule_courses = []
                
                selected_courses = st.multiselect(
                    "Show only these courses (leave empty to show all):",
                    options=unique_courses,
                    default=st.session_state.selected_schedule_courses,
                    key="schedule_course_filter"
                )
                
                # Save selection to session state
                st.session_state.selected_schedule_courses = selected_courses
                
                # Date range filter
                col1, col2 = st.columns(2)
                
                with col1:
                    min_date = schedule_df['start_date'].min().date()
                    start_date_filter = st.date_input("From date", min_date, key="gantt_start_filter")
                
                with col2:
                    max_date = schedule_df['end_date'].max().date()
                    end_date_filter = st.date_input("To date", max_date, key="gantt_end_filter")
                
                # Display options
                show_class_ids = st.checkbox("Show Class IDs", value=True)
                show_mos_allocation = st.checkbox("Show MOS Allocation in Tooltips", value=True)
            
            # Apply filters
            filtered_df = schedule_df.copy()
            
            if selected_courses:
                filtered_df = filtered_df[filtered_df['course_title'].isin(selected_courses)]
            
            filtered_df = filtered_df[
                (filtered_df['start_date'] >= pd.Timestamp(start_date_filter)) &
                (filtered_df['end_date'] <= pd.Timestamp(end_date_filter))
            ]
            
            # Check if we have data to display
            if filtered_df.empty:
                st.warning("No classes match the selected filters. Try adjusting the date range or course selection.")
            else:
                # Sort by course title first, then by start date
                filtered_df = filtered_df.sort_values(['course_title', 'start_date'])
                
                # Create an enhanced Gantt chart using plotly
                fig = go.Figure()
                
                # Get unique courses for coloring and positioning
                unique_courses = filtered_df['course_title'].unique()
                
                # Use a different color palette if there are many courses
                if len(unique_courses) <= 10:
                    colors = px.colors.qualitative.Plotly[:len(unique_courses)]
                else:
                    colors = px.colors.qualitative.Alphabet[:len(unique_courses)]
                
                color_map = {course: color for course, color in zip(unique_courses, colors)}
                
                # Add bars for each class
                for i, row in filtered_df.iterrows():
                    course = row['course_title']
                    class_id = row['id']
                    
                    # Calculate date range for the bar
                    start_date = row['start_date']
                    end_date = row['end_date']
                    duration_days = (end_date - start_date).days
                    
                    # Get MOS allocation if available and requested
                    mos_info = ""
                    if show_mos_allocation and 'mos_allocation' in row:
                        mos_alloc = row['mos_allocation']
                        mos_info = "<br>MOS: " + ", ".join([f"{mos}: {count}" for mos, count in mos_alloc.items() if count > 0])
                    
                    # Prepare hover text
                    hover_text = f"Class {class_id}: {course}<br>Start: {start_date.strftime('%Y-%m-%d')}<br>End: {end_date.strftime('%Y-%m-%d')}<br>Size: {row['size']}{mos_info}"
                    
                    # Add bar for class using scatter with lines
                    fig.add_trace(go.Scatter(
                        x=[start_date, end_date],
                        y=[course, course],
                        mode='lines',
                        line=dict(color=color_map[course], width=20),
                        name=f"{course} (Class {class_id})",
                        text=hover_text,
                        hoverinfo='text'
                    ))
                    
                    # Add text label for class ID if requested
                    if show_class_ids:
                        fig.add_annotation(
                            x=start_date + pd.Timedelta(days=duration_days/2),
                            y=course,
                            text=f"Class {class_id}",
                            showarrow=False,
                            font=dict(color="black", size=10)
                        )
                
                # Configure layout
                fig.update_layout(
                    title="Course Schedule",
                    xaxis=dict(
                        title="Date",
                        type='date',
                        tickformat='%Y-%m-%d',
                        rangeselector=dict(
                            buttons=list([
                                dict(count=1, label="1m", step="month", stepmode="backward"),
                                dict(count=6, label="6m", step="month", stepmode="backward"),
                                dict(count=1, label="YTD", step="year", stepmode="todate"),
                                dict(count=1, label="1y", step="year", stepmode="backward"),
                                dict(step="all")
                            ])
                        ),
                        rangeslider=dict(visible=True)
                    ),
                    yaxis=dict(
                        title="Course",
                        categoryorder='array',
                        categoryarray=list(unique_courses)
                    ),
                    hoverlabel=dict(
                        bgcolor="white",
                        font_size=12,
                        font_family="Arial"
                    ),
                    height=max(400, len(unique_courses) * 70),
                    margin=dict(l=150, r=50, t=50, b=50)
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Schedule table view with new columns
        st.subheader("Schedule Table")
        
        if not schedule_df.empty:
            # Create a display version of the dataframe with formatted MOS allocation
            display_df = schedule_df.copy()
            
            # Format MOS allocation as a string for display
            if 'mos_allocation' in display_df.columns:
                display_df['MOS Allocation'] = display_df['mos_allocation'].apply(
                    lambda x: ", ".join([f"{mos}: {count}" for mos, count in x.items() if count > 0]) if isinstance(x, dict) else ""
                )
            
            # Format dates for display
            display_df['Start Date'] = display_df['start_date'].dt.strftime('%Y-%m-%d')
            display_df['End Date'] = display_df['end_date'].dt.strftime('%Y-%m-%d')
            
            # Calculate duration
            display_df['Duration (days)'] = (display_df['end_date'] - display_df['start_date']).dt.days
            
            # Select columns to display
            display_columns = ['id', 'course_title', 'Start Date', 'End Date', 'Duration (days)', 'MOS Allocation', 'size']
            
            # Create an interactive table with filtering capabilities
            st.dataframe(
                display_df[display_columns].rename(columns={
                    'id': 'ID',
                    'course_title': 'Course Title',
                    'size': 'Class Size'
                }),
                hide_index=True,
                use_container_width=True
            )
        
        # Remove class button
        class_to_remove = st.selectbox(
            "Select class to remove",
            options=schedule_df['id'].tolist(),
            format_func=lambda x: f"ID: {x} - {schedule_df[schedule_df['id'] == x]['course_title'].iloc[0]}"
        )
        
        if st.button("Remove Class", use_container_width=True):
            st.session_state.future_schedule = [c for c in st.session_state.future_schedule if c['id'] != class_to_remove]
            st.success(f"Removed class with ID {class_to_remove}")
            st.rerun()
    else:
        st.info("No classes scheduled yet. Add classes using the form above.")
    
    # Save/load schedule
    st.subheader("Save/Load Schedule")
    
    with st.expander("Schedule File Format Information"):
        st.markdown("""
        ### Schedule File Format
        The schedule is saved as a JSON file with the following structure:
        ```json
        {
          "schedule": [
            {
              "course_title": "SF QUAL (ORIENTATION)",
              "start_date": "2025-05-27",
              "end_date": "2025-05-30",
              "size": 200,
              "id": 1,
              "mos_allocation": {
                "18A": 40,
                "18B": 40,
                "18C": 40,
                "18D": 40,
                "18E": 40
              }
            }
          ],
          "configurations": {
            "SF QUAL (ORIENTATION)": {
              "prerequisites": {
                "type": "AND",
                "courses": []
              },
              "or_prerequisites": [],
              "mos_paths": {
                "18A": [],
                "18B": [],
                "18C": [],
                "18D": [],
                "18E": []
              },
              "required_for_all_mos": true,
              "max_capacity": 200,
              "classes_per_year": 12,
              "reserved_seats": {
                "OF": 40,
                "ADE": 140,
                "NG": 20
              },
              "officer_enlisted_ratio": "1:4"
            }
          }
        }
        ```
        When uploading a schedule, the file must follow this format or be a simplified version with just the schedule array.
        """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Save Schedule", use_container_width=True):
            # Create a complete package with schedule and configurations
            save_data = {
                'schedule': st.session_state.future_schedule,
                'configurations': st.session_state.course_configs
            }
            
            # Convert to JSON with custom serializer for dates
            schedule_json = json.dumps(save_data, default=lambda o: o.isoformat() if isinstance(o, (datetime.date, datetime.datetime)) else o)
            
            st.download_button(
                label="Download Schedule JSON",
                data=schedule_json,
                file_name="training_schedule.json",
                mime="application/json",
                use_container_width=True
            )
    
    with col2:
        uploaded_schedule = st.file_uploader("Upload Schedule JSON", type=["json"])
        
        if uploaded_schedule is not None:
            try:
                loaded_data = json.load(uploaded_schedule)
                
                # Check if this is a complete package or just a schedule
                if isinstance(loaded_data, dict) and 'schedule' in loaded_data and 'configurations' in loaded_data:
                    # Apply backward compatibility for loaded configurations
                    for course, config in loaded_data['configurations'].items():
                        ensure_config_compatibility(config)
                    
                    st.session_state.course_configs.update(loaded_data['configurations'])
                    st.session_state.future_schedule = loaded_data['schedule']
                else:
                    # Assume it's just a schedule without configurations
                    st.session_state.future_schedule = loaded_data
                
                st.success("Schedule loaded successfully!")
                st.rerun()
            
            except Exception as e:
                st.error(f"Error loading schedule: {e}")
    
    # Navigation buttons
    st.write("---")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("← Back to Course Configuration", use_container_width=True):
            st.session_state.current_page = "Course Configuration"
            st.rerun()
    
    with col2:
        if st.button("Continue to Simulation →", use_container_width=True):
            st.session_state.current_page = "Simulation"
            st.rerun()

def display_simulation_page():
    st.header("Training Schedule Simulation")
    
    if not st.session_state.future_schedule:
        st.warning("Please build a schedule before running simulations.")
        
        if st.button("Go to Schedule Builder", use_container_width=True):
            st.session_state.current_page = "Schedule Builder"
            st.rerun()
        return
    
    # Option to use historical data for simulation settings
    use_historical_data = st.checkbox(
        "Use historical data for simulation settings",
        value=True,
        help="When enabled, the simulation will use patterns from historical data for student arrivals, MOS distribution, and pass rates."
    )
    
    # Option to use current pipeline state from historical data
    if use_historical_data and 'processed_data' in st.session_state and st.session_state.processed_data is not None:
        use_historical_state = st.checkbox(
            "Use current pipeline state from historical data",
            value=True,
            help="When enabled, the simulation will start with students already in the pipeline based on historical data."
        )
        
        if use_historical_state:
            historical_cutoff = st.date_input(
                "Historical cutoff date",
                value=datetime.datetime.now() - datetime.timedelta(days=30),
                help="Consider this date as 'now' - students who were in training as of this date will be included in the simulation's starting state."
            )
            
            adjust_mos_distribution = st.checkbox(
                "Adjust incoming MOS distribution to balance pipeline",
                value=True,
                help="When enabled, the simulation will adjust the MOS distribution of new students to help balance the overall pipeline."
            )
    else:
        use_historical_state = False
        historical_cutoff = None
        adjust_mos_distribution = False
    
    # Simulation settings
    st.subheader("Simulation Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        num_students = st.number_input("Number of students to simulate", min_value=10, value=100)
    
    with col2:
        num_iterations = st.number_input("Number of simulation iterations", min_value=1, value=10)
    
    # Student arrival settings
    st.subheader("Student Arrival Settings")
    
    # Historical arrival patterns
    historical_arrival_patterns = None
    historical_mos_distribution = None
    
    if use_historical_data and 'processed_data' in st.session_state and st.session_state.processed_data is not None:
        try:
            # Get historical patterns from session state if available
            if 'historical_arrival_patterns' in st.session_state:
                historical_arrival_patterns = st.session_state.historical_arrival_patterns
            
            if 'historical_mos_distribution' in st.session_state:
                historical_mos_distribution = st.session_state.historical_mos_distribution
            
            # If not in session state, try to extract them now
            if not historical_arrival_patterns:
                historical_arrival_patterns = extract_historical_arrival_patterns(st.session_state.processed_data)
                st.session_state.historical_arrival_patterns = historical_arrival_patterns
            
            if not historical_mos_distribution:
                historical_mos_distribution = extract_historical_mos_distribution(st.session_state.processed_data)
                st.session_state.historical_mos_distribution = historical_mos_distribution
            
            # Display historical patterns
            with st.expander("Historical Data Patterns"):
                if historical_arrival_patterns:
                    st.write("### Historical Arrival Patterns")
                    st.write(f"Average days before class start: {historical_arrival_patterns.get('avg_days_before', 'N/A')}")
                    
                    # Show monthly distribution
                    monthly_data = historical_arrival_patterns.get('monthly_distribution', {})
                    if monthly_data:
                        months = list(monthly_data.keys())
                        values = list(monthly_data.values())
                        fig = px.bar(
                            x=months,
                            y=values,
                            title="Historical Monthly Arrival Distribution",
                            labels={"x": "Month", "y": "Percentage of Students"},
                            color_discrete_sequence=['#3366CC']
                        )
                        fig.update_layout(xaxis={'categoryorder':'array', 'categoryarray':['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']})
                        st.plotly_chart(fig, use_container_width=True)
                
                if historical_mos_distribution:
                    st.write("### Historical MOS Distribution")
                    try:
                        # Filter out any metadata or non-standard keys
                        standard_mos_keys = ['18A', '18B', '18C', '18D', '18E']
                        filtered_distribution = {mos: historical_mos_distribution[mos] 
                                              for mos in standard_mos_keys 
                                              if mos in historical_mos_distribution and 
                                                 isinstance(historical_mos_distribution[mos], (int, float))}
                        
                        # Check if we have valid data to display
                        if filtered_distribution and sum(filtered_distribution.values()) > 0:
                            mos_data = []
                            for mos, percentage in filtered_distribution.items():
                                mos_data.append({"MOS": mos, "Percentage": percentage * 100})
                            
                            fig = px.pie(
                                mos_data,
                                values="Percentage",
                                names="MOS",
                                title="Historical MOS Distribution",
                                color_discrete_sequence=px.colors.qualitative.Bold
                            )
                            fig.update_traces(textposition='inside', textinfo='percent+label')
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("Insufficient MOS distribution data available.")
                    except Exception as e:
                        st.error(f"Error displaying MOS distribution: {e}")
                        st.info("Default MOS distribution will be used for simulations.")
        
        except Exception as e:
            st.warning(f"Error extracting historical patterns: {e}")
            historical_arrival_patterns = None
            historical_mos_distribution = None
    
    # Only show these options if not using historical data or if historical data extraction failed
    if not use_historical_data or not historical_arrival_patterns:
        arrival_method = st.radio(
            "Student arrival method",
            ["Before each class", "Continuous throughout the year"]
        )
        
        if arrival_method == "Before each class":
            arrival_days_before = st.number_input(
                "Days before class start that students arrive",
                min_value=0,
                max_value=30,
                value=3,
                help="Students will arrive this many days before their first class starts"
            )
        else:
            # For continuous arrivals, allow setting a pattern
            st.write("Students will arrive throughout the simulation period:")
            
            col1, col2 = st.columns(2)
            
            with col1:
                monthly_distribution = st.selectbox(
                    "Monthly arrival pattern",
                    ["Even", "Summer heavy", "Winter heavy", "Custom"]
                )
            
            with col2:
                arrival_randomness = st.slider(
                    "Arrival randomness",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.3,
                    help="Higher values mean more random/uneven arrivals"
                )
            
            if monthly_distribution == "Custom":
                st.write("Set the relative percentage of students arriving each month:")
                
                months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
                month_cols = st.columns(6)
                custom_distribution = {}
                
                for i, month in enumerate(months):
                    col_idx = i % 6
                    with month_cols[col_idx]:
                        custom_distribution[month] = st.slider(
                            month,
                            min_value=0,
                            max_value=100,
                            value=100 // 12,  # Default to even distribution
                            help=f"Relative percentage for {month}"
                        )
                
                # Normalize to make sure they sum to 100%
                total = sum(custom_distribution.values())
                if total > 0:
                    for month in custom_distribution:
                        custom_distribution[month] = custom_distribution[month] / total
    
    # Advanced settings expander
    with st.expander("Advanced Simulation Settings"):
        # MOS Distribution
        st.write("MOS Distribution (percentage of students in each path):")
        
        # Safe MOS distribution handling
        default_mos_distribution = {'18A': 0.2, '18B': 0.2, '18C': 0.2, '18D': 0.2, '18E': 0.2}

        if use_historical_data:
            try:
                # First check if we have historical distribution data
                if historical_mos_distribution is None:
                    st.warning("No historical MOS distribution data available. Using default distribution.")
                    mos_distribution = default_mos_distribution.copy()
                else:
                    # Validate the structure of historical_mos_distribution
                    standard_mos = ['18A', '18B', '18C', '18D', '18E']
                    valid_distribution = True
                    
                    # Check if it's a dictionary with the expected keys
                    if not isinstance(historical_mos_distribution, dict):
                        valid_distribution = False
                        st.error(f"Invalid MOS distribution type: {type(historical_mos_distribution)}")
                    else:
                        # Ensure all standard MOS keys exist with numeric values
                        safe_distribution = {}
                        for mos in standard_mos:
                            if mos not in historical_mos_distribution:
                                safe_distribution[mos] = 0.2
                                valid_distribution = False
                            else:
                                value = historical_mos_distribution[mos]
                                if not isinstance(value, (int, float)):
                                    safe_distribution[mos] = 0.2
                                    valid_distribution = False
                                    st.warning(f"Non-numeric value for {mos}: {value}")
                                else:
                                    safe_distribution[mos] = float(value)
                        
                        # Normalize to ensure sum is 1.0
                        total = sum(safe_distribution.values())
                        if total == 0:
                            safe_distribution = default_mos_distribution.copy()
                        else:
                            safe_distribution = {k: v/total for k, v in safe_distribution.items()}
                    
                    if valid_distribution:
                        st.info("Using historical MOS distribution data")
                        mos_distribution = safe_distribution
                    else:
                        st.warning("Found issues with historical MOS distribution. Using corrected values.")
                        mos_distribution = safe_distribution
                    
                    # Display the distribution with sliders
                    mos_cols = st.columns(5)
                    
                    with mos_cols[0]:
                        mos_distribution['18A'] = st.slider("18A (Officer)", 0, 100,
                                                        int(mos_distribution['18A'] * 100)) / 100
                    
                    with mos_cols[1]:
                        mos_distribution['18B'] = st.slider("18B (Weapons)", 0, 100,
                                                        int(mos_distribution['18B'] * 100)) / 100
                    
                    with mos_cols[2]:
                        mos_distribution['18C'] = st.slider("18C (Engineer)", 0, 100,
                                                        int(mos_distribution['18C'] * 100)) / 100
                    
                    with mos_cols[3]:
                        mos_distribution['18D'] = st.slider("18D (Medical)", 0, 100,
                                                        int(mos_distribution['18D'] * 100)) / 100
                    
                    with mos_cols[4]:
                        mos_distribution['18E'] = st.slider("18E (Communications)", 0, 100,
                                                        int(mos_distribution['18E'] * 100)) / 100
            except Exception as e:
                st.error(f"Error processing MOS distribution: {e}")
                st.warning("Using default MOS distribution instead.")
                mos_distribution = default_mos_distribution.copy()
        else:
            # Manual MOS distribution
            mos_distribution = {}
            mos_cols = st.columns(5)
            
            with mos_cols[0]:
                mos_distribution['18A'] = st.slider("18A (Officer)", 0, 100, 20) / 100
            
            with mos_cols[1]:
                mos_distribution['18B'] = st.slider("18B (Weapons)", 0, 100, 20) / 100
            
            with mos_cols[2]:
                mos_distribution['18C'] = st.slider("18C (Engineer)", 0, 100, 20) / 100
            
            with mos_cols[3]:
                mos_distribution['18D'] = st.slider("18D (Medical)", 0, 100, 20) / 100
            
            with mos_cols[4]:
                mos_distribution['18E'] = st.slider("18E (Communications)", 0, 100, 20) / 100
        
        # Rest of advanced settings
        randomize_factor = st.slider("Randomization Factor", min_value=0.0, max_value=1.0, value=0.1,
                                    help="How much to randomize historical rates in simulation")
        
        st.write("Course Pass Rates:")
        
        # Organize courses into tabs by type to make it more manageable
        # First, group courses by common prefixes or categories
        course_categories = {
            "SF QUAL": [],
            "SERE": [],
            "ARSOF": [],
            "Other": []
        }
        
        for course in st.session_state.course_configs.keys():
            if "SF QUAL" in course:
                course_categories["SF QUAL"].append(course)
            elif "SERE" in course:
                course_categories["SERE"].append(course)
            elif "ARSOF" in course:
                course_categories["ARSOF"].append(course)
            else:
                course_categories["Other"].append(course)
        
        # Create tabs for each category
        course_tabs = st.tabs(list(course_categories.keys()))
        
        # Initialize override rates dictionary
        override_rates = {}
        
        # Fill in each tab with its courses
        for i, (category, courses) in enumerate(course_categories.items()):
            with course_tabs[i]:
                if not courses:
                    st.write("No courses in this category.")
                    continue
                
                # Create multiple columns for more compact display
                cols = st.columns(3)
                
                for j, course in enumerate(sorted(courses)):
                    col_idx = j % 3
                    with cols[col_idx]:
                        hist_rate = st.session_state.historical_analysis.get(course, {}).get('pass_rate', 0.8)
                        override_rates[course] = st.slider(
                            f"{course[:30]}...",
                            min_value=0.0,
                            max_value=1.0,
                            value=float(hist_rate),
                            format="%.2f"
                        )
    
    # Run simulation button
    if st.button("Run Simulation", use_container_width=True):
        with st.spinner("Running simulation... This may take a while."):
            # Create a progress bar
            progress_bar = st.progress(0)
            progress_text = st.empty()
            
            # Prepare inputs for simulation
            course_configs_copy = copy.deepcopy(st.session_state.course_configs)
            
            # Apply backward compatibility to all configurations
            for course, config in course_configs_copy.items():
                ensure_config_compatibility(config)
            
            # Convert to string representations for caching
            schedule_str = json.dumps(st.session_state.future_schedule, sort_keys=True, default=lambda o: str(o))
            configs_str = json.dumps(course_configs_copy, sort_keys=True, default=lambda o: str(o))
            
            simulation_inputs = {
                'schedule_str': schedule_str,
                'configs_str': configs_str,
                'historical_data': st.session_state.historical_analysis,
                'num_students': num_students,
                'num_iterations': num_iterations,
                'use_historical_data': use_historical_data,
                'historical_arrival_patterns': historical_arrival_patterns,
                'historical_mos_distribution': mos_distribution,
                'randomize_factor': randomize_factor,
                'override_rates': override_rates
            }
            
            # Add inputs for historical state if using it
            if use_historical_data and use_historical_state:
                simulation_inputs.update({
                    'use_historical_state': use_historical_state,
                    'historical_cutoff_date': historical_cutoff,
                    'adjust_mos_distribution': adjust_mos_distribution,
                    'processed_data': st.session_state.processed_data
                })
            
            # If not using historical data, add manual settings
            if not use_historical_data or not historical_arrival_patterns:
                if arrival_method == "Before each class":
                    simulation_inputs['arrival_method'] = 'before_class'
                    simulation_inputs['arrival_days_before'] = arrival_days_before
                else:
                    simulation_inputs['arrival_method'] = 'continuous'
                    simulation_inputs['monthly_distribution'] = monthly_distribution
                    simulation_inputs['arrival_randomness'] = arrival_randomness
                    if monthly_distribution == "Custom":
                        simulation_inputs['custom_distribution'] = custom_distribution
            
            # Run simulation with progress updates
            for i in range(10):
                # Update progress
                progress_bar.progress(i * 10)
                progress_text.text(f"Running simulation iteration {i+1}/10...")
                time.sleep(0.1)  # Simulate work
            
            # Run the actual simulation
            simulation_results = run_cached_simulation(**simulation_inputs)
            st.session_state.simulation_results = simulation_results
            
            # Complete the progress bar
            progress_bar.progress(100)
            progress_text.text("Simulation completed!")
            
            st.success("Simulation completed successfully!")
    
    # Display simulation results if available
    if st.session_state.simulation_results:
        results = st.session_state.simulation_results
        
        # Key metrics overview
        st.subheader("Simulation Results Overview")
        
        metric_cols = st.columns(4)
        
        with metric_cols[0]:
            st.metric("Avg. Completion Time", f"{results['avg_completion_time']:.1f} days")
        
        with metric_cols[1]:
            st.metric("Student Throughput", f"{results['throughput']:.1f}/month")
        
        with metric_cols[2]:
            st.metric("Avg. Wait Time", f"{results['avg_wait_time']:.1f} days")
        
        with metric_cols[3]:
            st.metric("Resource Utilization", f"{results['resource_utilization']:.1%}")
        
        # Add save/load functionality for simulation results
        st.subheader("Save/Load Simulation Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Save Results", use_container_width=True):
                # Create a complete results package with simulation results, settings, and schedule
                results_package = {
                    'simulation_results': st.session_state.simulation_results,
                    'simulation_settings': {
                        'use_historical_data': use_historical_data,
                        'num_students': num_students,
                        'num_iterations': num_iterations,
                        'randomize_factor': randomize_factor,
                        'schedule': st.session_state.future_schedule
                        # Include other relevant settings
                    },
                    'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
                # Convert to JSON (handle numpy values and datetime objects)
                results_json = json.dumps(results_package, default=lambda o: o.isoformat() if isinstance(o, (datetime.date, datetime.datetime, np.integer, np.floating)) else str(o), indent=2)
                
                st.download_button(
                    label="Download Simulation Results",
                    data=results_json,
                    file_name=f"simulation_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )
        
        with col2:
            uploaded_results = st.file_uploader("Upload Results JSON", type=["json"], key="results_uploader")
            
            if uploaded_results is not None:
                try:
                    loaded_package = json.load(uploaded_results)
                    
                    # Extract the components
                    loaded_results = loaded_package.get('simulation_results', {})
                    loaded_settings = loaded_package.get('simulation_settings', {})
                    timestamp = loaded_package.get('timestamp', 'Unknown')
                    
                    st.session_state.loaded_simulation_results = loaded_results
                    
                    # Display information about the loaded results
                    st.success(f"Results from {timestamp} loaded successfully!")
                    
                    if st.button("View Loaded Results", use_container_width=True):
                        st.session_state.simulation_results = loaded_results
                        st.rerun()
                
                except Exception as e:
                    st.error(f"Error loading results: {e}")
        
        # Tabs for different visualizations
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Bottlenecks", "Student Flow", "Class Utilization", "MOS Analysis", "Detailed Metrics", "Compare Simulations"])
        
        with tab1:
            st.subheader("Bottleneck Analysis")
            
            # Bottleneck chart
            bottleneck_df = pd.DataFrame(results['bottlenecks'])
            
            if not bottleneck_df.empty:
                # Sort by wait time (descending)
                bottleneck_df = bottleneck_df.sort_values('wait_time', ascending=False)
                
                fig = px.bar(
                    bottleneck_df, 
                    x='course', 
                    y='wait_time',
                    title="Average Wait Time Before Each Course (Days)",
                    color='wait_time', 
                    color_continuous_scale='RdYlGn_r',  # Red for high wait times (bad), green for low
                    height=500
                )
                
                fig.update_layout(
                    xaxis_title="Course",
                    yaxis_title="Average Wait Time (days)",
                    coloraxis_colorbar_title="Wait Time"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Bottleneck table
                st.dataframe(
                    bottleneck_df.rename(columns={
                        'course': 'Course',
                        'wait_time': 'Average Wait Time (days)'
                    }),
                    hide_index=True,
                    use_container_width=True
                )
            else:
                st.info("No bottleneck data available.")
        
        with tab2:
            st.subheader("Student Flow Analysis")
            
            # Convert student progression data to DataFrame if it exists
            if 'student_progression' in results:
                progression_df = pd.DataFrame(results['student_progression'])
                
                if not progression_df.empty:
                    # Convert time column to datetime if needed
                    if 'time' in progression_df.columns:
                        progression_df['time'] = pd.to_datetime(progression_df['time'])
                    
                    # Student progression visualization
                    fig = px.line(
                        progression_df, 
                        x='time', 
                        y='count', 
                        color='stage',
                        title="Student Progression Over Time", 
                        height=500,
                        color_discrete_map={
                            'Waiting': 'orange',
                            'In Class': 'blue',
                            'Graduated': 'green'
                        }
                    )
                    
                    fig.update_layout(
                        xaxis_title="Time",
                        yaxis_title="Number of Students",
                        legend_title="Student Status"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No student progression data available.")
            
            # Completion time distribution
            if 'completion_times' in results:
                completion_df = pd.DataFrame(results['completion_times'])
                
                if not completion_df.empty:
                    fig = px.histogram(
                        completion_df, 
                        x='days',
                        title="Distribution of Student Completion Times",
                        labels={'days': 'Days to Complete All Courses'},
                        color_discrete_sequence=['#3366CC'],
                        height=400,
                        nbins=30
                    )
                    
                    fig.update_layout(
                        xaxis_title="Days to Complete Training",
                        yaxis_title="Number of Students"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No completion time data available.")
        
        with tab3:
            st.subheader("Class Utilization")
            
            # Class utilization chart
            if 'class_utilization' in results:
                util_df = pd.DataFrame(results['class_utilization'])
                
                if not util_df.empty:
                    # Sort by utilization (ascending to highlight low utilization)
                    util_df = util_df.sort_values('utilization')
                    
                    fig = px.bar(
                        util_df, 
                        x='course', 
                        y='utilization',
                        title="Class Capacity Utilization",
                        color='utilization',
                        color_continuous_scale='RdYlGn',  # Red for low utilization (bad), green for high
                        range_color=[0, 1],
                        height=500
                    )
                    
                    fig.update_layout(
                        xaxis_title="Course",
                        yaxis_title="Utilization Rate",
                        coloraxis_colorbar_title="Utilization"
                    )
                    
                    # Add a horizontal line at 60% utilization
                    fig.add_shape(
                        type="line",
                        x0=-0.5,
                        x1=len(util_df) - 0.5,
                        y0=0.6,
                        y1=0.6,
                        line=dict(
                            color="red",
                            width=2,
                            dash="dash"
                        )
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Classes with low utilization
                    st.write("Classes with Low Utilization (<60%)")
                    low_util = util_df[util_df['utilization'] < 0.6]
                    
                    if not low_util.empty:
                        st.dataframe(
                            low_util.rename(columns={
                                'course': 'Course',
                                'utilization': 'Utilization',
                                'capacity': 'Capacity',
                                'enrolled': 'Enrolled'
                            }),
                            hide_index=True,
                            use_container_width=True
                        )
                    else:
                        st.write("No classes with low utilization found.")
            else:
                st.info("No class utilization data available.")
        
        with tab4:
            st.subheader("MOS Path Analysis")
            
            if 'mos_metrics' in results:
                mos_data = results['mos_metrics']
                
                # Create DataFrame for MOS metrics
                mos_df = pd.DataFrame([
                    {
                        'MOS': mos,
                        'Count': data['count'],
                        'Avg Completion Time (days)': data['avg_completion_time'],
                        'Avg Wait Time (days)': data['avg_wait_time']
                    }
                    for mos, data in mos_data.items()
                ])
                
                # Display MOS metrics table
                st.dataframe(mos_df, hide_index=True, use_container_width=True)
                
                # Create visualization for MOS metrics
                fig = px.bar(
                    mos_df,
                    x='MOS',
                    y=['Avg Completion Time (days)', 'Avg Wait Time (days)'],
                    barmode='group',
                    title='Average Completion and Wait Times by MOS',
                    height=400,
                    color_discrete_map={
                        'Avg Completion Time (days)': '#3366CC',
                        'Avg Wait Time (days)': '#FF9900'
                    }
                )
                
                fig.update_layout(
                    xaxis_title="Military Occupational Specialty (MOS)",
                    yaxis_title="Days",
                    legend_title="Metric"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # MOS utilization across classes
                st.subheader("MOS Utilization by Class")
                
                if 'class_mos_utilization' in results:
                    mos_util_data = []
                    
                    for class_util in results['class_mos_utilization']:
                        course = class_util['course']
                        for mos, util in class_util['mos_utilization'].items():
                            mos_util_data.append({
                                'Course': course,
                                'MOS': mos,
                                'Utilization': util
                            })
                    
                    if mos_util_data:
                        mos_util_df = pd.DataFrame(mos_util_data)
                        
                        # Create heatmap of MOS utilization
                        fig = px.density_heatmap(
                            mos_util_df,
                            x='Course',
                            y='MOS',
                            z='Utilization',
                            title='MOS Seat Utilization by Course',
                            color_continuous_scale='YlGnBu',
                            height=500
                        )
                        
                        fig.update_layout(
                            xaxis_title="Course",
                            yaxis_title="MOS",
                            coloraxis_colorbar_title="Utilization Rate"
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Add a table view
                        with st.expander("View MOS Utilization Data Table"):
                            st.dataframe(
                                mos_util_df,
                                hide_index=True,
                                use_container_width=True
                            )
            else:
                st.info("No MOS-specific data available in simulation results. Try running the simulation again with MOS paths enabled.")
        
        with tab5:
            st.subheader("Detailed Metrics")
            
            # Detailed metrics table
            if 'detailed_metrics' in results:
                detailed_metrics = pd.DataFrame(results['detailed_metrics'])
                
                if not detailed_metrics.empty:
                    st.dataframe(detailed_metrics, use_container_width=True)
                    
                    # Export results button
                    csv = detailed_metrics.to_csv(index=False)
                    st.download_button(
                        label="Download Detailed Results CSV",
                        data=csv,
                        file_name="simulation_results.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
            else:
                st.info("No detailed metrics available in the simulation results.")
        
        with tab6:
            st.subheader("Compare Simulations")
            
            # List of simulations to compare
            if 'comparison_simulations' not in st.session_state:
                st.session_state.comparison_simulations = []
            
            # Add current simulation to comparison
            if st.button("Add Current Simulation to Comparison", use_container_width=True):
                # Create a snapshot of current simulation
                simulation_snapshot = {
                    'results': copy.deepcopy(st.session_state.simulation_results),
                    'name': f"Simulation {len(st.session_state.comparison_simulations) + 1}",
                    'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
                st.session_state.comparison_simulations.append(simulation_snapshot)
                st.success("Current simulation added to comparison!")
                st.rerun()
            
            # Upload a saved simulation for comparison
            uploaded_comparison = st.file_uploader("Upload Simulation for Comparison", type=["json"], key="comparison_uploader")
            
            if uploaded_comparison is not None:
                try:
                    loaded_package = json.load(uploaded_comparison)
                    
                    # Extract the results
                    loaded_results = loaded_package.get('simulation_results', {})
                    timestamp = loaded_package.get('timestamp', 'Unknown')
                    
                    # Create a name for the uploaded simulation
                    simulation_name = st.text_input("Name for this simulation",
                                                 value=f"Uploaded Simulation {len(st.session_state.comparison_simulations) + 1}")
                    
                    if st.button("Add to Comparison", use_container_width=True):
                        simulation_snapshot = {
                            'results': loaded_results,
                            'name': simulation_name,
                            'timestamp': timestamp
                        }
                        
                        st.session_state.comparison_simulations.append(simulation_snapshot)
                        st.success(f"Uploaded simulation '{simulation_name}' added to comparison!")
                        st.rerun()
                
                except Exception as e:
                    st.error(f"Error loading simulation for comparison: {e}")
            
            # Display comparison if we have simulations to compare
            if st.session_state.comparison_simulations:
                st.write("### Simulations for Comparison")
                
                # Create comparison table
                comparison_data = []
                
                for sim in st.session_state.comparison_simulations:
                    results = sim['results']
                    comparison_data.append({
                        'Simulation': sim['name'],
                        'Timestamp': sim['timestamp'],
                        'Avg Completion Time': f"{results.get('avg_completion_time', 0):.1f} days",
                        'Avg Wait Time': f"{results.get('avg_wait_time', 0):.1f} days",
                        'Throughput': f"{results.get('throughput', 0):.1f}/month",
                        'Resource Utilization': f"{results.get('resource_utilization', 0):.1%}"
                    })
                
                # Display comparison table
                st.dataframe(pd.DataFrame(comparison_data), hide_index=True, use_container_width=True)
                
                # Create comparison charts
                metrics_to_compare = ["Avg Completion Time", "Avg Wait Time", "Throughput", "Resource Utilization"]
                metric_to_viz = st.selectbox("Select metric to visualize", metrics_to_compare)
                
                # Create visualization based on selected metric
                if metric_to_viz == "Avg Completion Time":
                    data = [{'Simulation': sim['name'], 'Value': sim['results'].get('avg_completion_time', 0)}
                           for sim in st.session_state.comparison_simulations]
                    fig = px.bar(
                        data, x='Simulation', y='Value',
                        title="Average Completion Time Comparison (days)",
                        labels={'Value': 'Days'},
                        color='Simulation'
                    )
                
                elif metric_to_viz == "Avg Wait Time":
                    data = [{'Simulation': sim['name'], 'Value': sim['results'].get('avg_wait_time', 0)}
                           for sim in st.session_state.comparison_simulations]
                    fig = px.bar(
                        data, x='Simulation', y='Value',
                        title="Average Wait Time Comparison (days)",
                        labels={'Value': 'Days'},
                        color='Simulation'
                    )
                
                elif metric_to_viz == "Throughput":
                    data = [{'Simulation': sim['name'], 'Value': sim['results'].get('throughput', 0)}
                           for sim in st.session_state.comparison_simulations]
                    fig = px.bar(
                        data, x='Simulation', y='Value',
                        title="Throughput Comparison (students/month)",
                        labels={'Value': 'Students/Month'},
                        color='Simulation'
                    )
                
                else:  # Resource Utilization
                    data = [{'Simulation': sim['name'], 'Value': sim['results'].get('resource_utilization', 0)}
                           for sim in st.session_state.comparison_simulations]
                    fig = px.bar(
                        data, x='Simulation', y='Value',
                        title="Resource Utilization Comparison",
                        labels={'Value': 'Utilization'},
                        range_y=[0, 1],
                        color='Simulation'
                    )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Option to clear comparison data
                if st.button("Clear Comparison Data", use_container_width=True):
                    st.session_state.comparison_simulations = []
                    st.success("Comparison data cleared!")
                    st.rerun()
        
        # Navigation buttons
        st.write("---")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("← Back to Schedule Builder", use_container_width=True):
                st.session_state.current_page = "Schedule Builder"
                st.rerun()
        
        with col2:
            if st.button("Continue to Optimization →", use_container_width=True):
                st.session_state.current_page = "Optimization"
                st.rerun()

def display_optimization_page():
    st.header("Schedule Optimization")
    
    if st.session_state.simulation_results is None:
        st.warning("Please run a simulation before attempting optimization.")
        
        if st.button("Go to Simulation Page", use_container_width=True):
            st.session_state.current_page = "Simulation"
            st.rerun()
        return
    
    # Optimization settings
    st.subheader("Optimization Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        optimization_goal = st.selectbox(
            "Primary Optimization Goal",
            ["Minimize Average Completion Time",
             "Maximize Student Throughput",
             "Minimize Wait Times",
             "Maximize Resource Utilization"]
        )
    
    with col2:
        optimization_iterations = st.number_input("Optimization Iterations", min_value=1, max_value=100, value=20)
    
    # Advanced settings
    with st.expander("Advanced Optimization Settings"):
        constraint_weight = st.slider(
            "Constraint Satisfaction Weight",
            min_value=0.1,
            max_value=1.0,
            value=0.7,
            help="Higher values prioritize meeting constraints over optimization goal"
        )
        
        allow_capacity_changes = st.checkbox(
            "Allow Capacity Changes",
            value=True,
            help="Allow the optimizer to suggest changes to class capacities"
        )
        
        allow_duration_changes = st.checkbox(
            "Allow Duration Changes",
            value=False,
            help="Allow the optimizer to suggest changes to class durations"
        )
        
        allow_prerequisite_changes = st.checkbox(
            "Allow Prerequisite Changes",
            value=False,
            help="Allow the optimizer to suggest changes to prerequisite relationships"
        )
        
        allow_mos_allocation_changes = st.checkbox(
            "Allow MOS Allocation Changes",
            value=True,
            help="Allow the optimizer to suggest changes to MOS allocations"
        )
    
    # Run optimization button
    if st.button("Run Optimization", use_container_width=True):
        with st.spinner("Optimizing schedule... This may take a few minutes."):
            # Create a progress bar
            progress_bar = st.progress(0)
            progress_text = st.empty()
            
            # Prepare optimization inputs
            course_configs_copy = copy.deepcopy(st.session_state.course_configs)
            
            # Apply backward compatibility
            for course, config in course_configs_copy.items():
                ensure_config_compatibility(config)
            
            optimization_inputs = {
                'current_schedule': st.session_state.future_schedule,
                'course_configs': course_configs_copy,
                'simulation_results': st.session_state.simulation_results,
                'historical_data': st.session_state.historical_analysis,
                'optimization_goal': optimization_goal,
                'iterations': optimization_iterations,
                'constraint_weight': constraint_weight,
                'allow_capacity_changes': allow_capacity_changes,
                'allow_duration_changes': allow_duration_changes,
                'allow_prerequisite_changes': allow_prerequisite_changes,
                'allow_mos_allocation_changes': allow_mos_allocation_changes
            }
            
            # Update progress periodically to show activity
            for i in range(5):
                progress_bar.progress(i * 10)
                progress_text.text(f"Analyzing bottlenecks and constraints...")
                time.sleep(0.1)  # Simulate work
            
            # Run optimization with progress updates
            try:
                # Run the actual optimization
                optimization_results = optimize_schedule(optimization_inputs)
                
                # Final progress updates
                for i in range(5, 10):
                    progress_bar.progress(i * 10)
                    progress_text.text(f"Applying optimizations and validating...")
                    time.sleep(0.1)  # Simulate work
                
                st.session_state.optimization_results = optimization_results
                
                # Complete the progress
                progress_bar.progress(100)
                progress_text.text("Optimization completed!")
                
                st.success("Optimization completed successfully!")
            
            except Exception as e:
                st.error(f"Error during optimization: {str(e)}")
                # Show traceback for debugging
                import traceback
                st.error(traceback.format_exc())
    
    # Display optimization results if available
    if 'optimization_results' in st.session_state and st.session_state.optimization_results:
        results = st.session_state.optimization_results
        
        # Key improvements
        st.subheader("Optimization Results")
        
        # Metrics comparison
        metrics_comparison = results['metrics_comparison']
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            delta = metrics_comparison['completion_time']['improvement']
            st.metric("Completion Time",
                     f"{metrics_comparison['completion_time']['optimized']:.1f} days",
                     f"{delta:.1f} days")
        
        with col2:
            delta = metrics_comparison['throughput']['improvement']
            st.metric("Student Throughput",
                     f"{metrics_comparison['throughput']['optimized']:.1f}/month",
                     f"{delta:.1f}/month")
        
        with col3:
            delta = metrics_comparison['wait_time']['improvement']
            st.metric("Avg. Wait Time",
                     f"{metrics_comparison['wait_time']['optimized']:.1f} days",
                     f"{delta:.1f} days")
        
        with col4:
            delta = metrics_comparison['utilization']['improvement'] * 100
            st.metric("Resource Utilization",
                     f"{metrics_comparison['utilization']['optimized']:.1%}",
                     f"{delta:.1f}%")
        
        # Recommended changes
        st.subheader("Recommended Schedule Changes")
        
        changes = results['recommended_changes']
        
        # Tabs for different types of changes
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["Schedule Changes", "Capacity Changes", "MOS Allocation", "Prerequisite Changes", "Other Recommendations"])
        
        with tab1:
            if 'schedule_changes' in changes and changes['schedule_changes']:
                schedule_changes = pd.DataFrame(changes['schedule_changes'])
                
                # Convert date strings to datetime for proper display
                if 'original_start' in schedule_changes.columns:
                    schedule_changes['original_start'] = pd.to_datetime(schedule_changes['original_start'])
                if 'original_end' in schedule_changes.columns:
                    schedule_changes['original_end'] = pd.to_datetime(schedule_changes['original_end'])
                if 'new_start' in schedule_changes.columns:
                    schedule_changes['new_start'] = pd.to_datetime(schedule_changes['new_start'])
                if 'new_end' in schedule_changes.columns:
                    schedule_changes['new_end'] = pd.to_datetime(schedule_changes['new_end'])
                
                st.dataframe(schedule_changes, hide_index=True, use_container_width=True)
                
                # Visualization of schedule changes
                if not schedule_changes.empty and 'course' in schedule_changes.columns:
                    try:
                        # Create a proper timeline chart
                        fig = go.Figure()
                        
                        # Add original schedule
                        for i, row in schedule_changes.iterrows():
                            # Original schedule in lighter color
                            fig.add_trace(go.Bar(
                                x=[row['original_end'] - row['original_start']],
                                y=[f"{row['course']} (Original)"],
                                orientation='h',
                                base=[row['original_start']],
                                name=f"{row['course']} (Original)",
                                marker_color='rgba(200, 200, 200, 0.6)',
                                hoverinfo='text',
                                text=f"Original: {row['course']}<br>Start: {row['original_start'].strftime('%Y-%m-%d')}<br>End: {row['original_end'].strftime('%Y-%m-%d')}"
                            ))
                            
                            # New schedule in bolder color
                            fig.add_trace(go.Bar(
                                x=[row['new_end'] - row['new_start']],
                                y=[f"{row['course']} (New)"],
                                orientation='h',
                                base=[row['new_start']],
                                name=f"{row['course']} (New)",
                                marker_color='rgba(66, 133, 244, 0.8)',
                                hoverinfo='text',
                                text=f"New: {row['course']}<br>Start: {row['new_start'].strftime('%Y-%m-%d')}<br>End: {row['new_end'].strftime('%Y-%m-%d')}"
                            ))
                        
                        # Add title and format x-axis as date
                        fig.update_layout(
                            title="Schedule Changes (Original vs. Optimized)",
                            xaxis_title="Date",
                            yaxis_title="Course",
                            height=max(400, len(schedule_changes) * 60),
                            barmode='overlay',
                            xaxis=dict(
                                type='date',
                                tickformat='%Y-%m-%d'
                            )
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error creating schedule changes visualization: {e}")
            else:
                st.write("No schedule changes recommended.")
        
        with tab2:
            if 'capacity_changes' in changes and changes['capacity_changes']:
                capacity_changes = pd.DataFrame(changes['capacity_changes'])
                
                if not capacity_changes.empty:
                    st.dataframe(capacity_changes, hide_index=True, use_container_width=True)
                    
                    # Visualization of capacity changes
                    try:
                        fig = px.bar(
                            capacity_changes,
                            x="course",
                            y=["original_capacity", "recommended_capacity"],
                            barmode="group",
                            title="Class Capacity Recommendations",
                            labels={
                                "value": "Capacity",
                                "course": "Course",
                                "variable": "Type"
                            },
                            color_discrete_map={
                                "original_capacity": "lightgrey",
                                "recommended_capacity": "#4CAF50"
                            }
                        )
                        
                        fig.update_layout(
                            xaxis_title="Course",
                            yaxis_title="Capacity",
                            legend_title="Type",
                            height=max(400, len(capacity_changes) * 40)
                        )
                        
                        # Rename legend items
                        fig.for_each_trace(lambda t: t.update(
                            name=t.name.replace("original_capacity", "Original Capacity").replace("recommended_capacity", "Recommended Capacity")
                        ))
                        
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error creating capacity changes visualization: {e}")
            else:
                st.write("No capacity changes recommended.")
        
        with tab3:
            if 'mos_allocation_changes' in changes and changes['mos_allocation_changes']:
                mos_allocation_changes = pd.DataFrame(changes['mos_allocation_changes'])
                
                if not mos_allocation_changes.empty:
                    st.dataframe(mos_allocation_changes, hide_index=True, use_container_width=True)
                    
                    # Visualization of MOS allocation changes
                    if len(mos_allocation_changes) > 0:
                        try:
                            # Create a long-form dataframe for easier plotting
                            plot_data = []
                            
                            for _, row in mos_allocation_changes.iterrows():
                                course = row['course']
                                class_id = row['class_id']
                                
                                for mos in ['18A', '18B', '18C', '18D', '18E']:
                                    # Original allocation
                                    original = row.get(f'original_{mos}', 0)
                                    plot_data.append({
                                        'Course': f"{course} (ID:{class_id})",
                                        'MOS': mos,
                                        'Type': 'Original',
                                        'Seats': original
                                    })
                                    
                                    # Recommended allocation
                                    recommended = row.get(f'recommended_{mos}', 0)
                                    plot_data.append({
                                        'Course': f"{course} (ID:{class_id})",
                                        'MOS': mos,
                                        'Type': 'Recommended',
                                        'Seats': recommended
                                    })
                            
                            plot_df = pd.DataFrame(plot_data)
                            
                            fig = px.bar(
                                plot_df,
                                x='MOS',
                                y='Seats',
                                color='Type',
                                facet_col='Course',
                                title="MOS Allocation Changes",
                                barmode='group',
                                color_discrete_map={
                                    'Original': 'lightgrey',
                                    'Recommended': '#3366CC'
                                }
                            )
                            
                            # Adjust layout for better readability
                            fig.update_layout(
                                height=400 * (len(mos_allocation_changes) // 2 + 1),
                                margin=dict(t=100, b=50)
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.error(f"Error creating MOS allocation visualization: {e}")
            else:
                st.write("No MOS allocation changes recommended.")
        
        with tab4:
            if 'prerequisite_changes' in changes and changes['prerequisite_changes']:
                prerequisite_changes = pd.DataFrame(changes['prerequisite_changes'])
                
                if not prerequisite_changes.empty:
                    st.dataframe(prerequisite_changes, hide_index=True, use_container_width=True)
                    
                    # Add a visualization if possible
                    if len(prerequisite_changes) > 0:
                        try:
                            # Create a network diagram of before/after prerequisite structure
                            st.write("### Prerequisite Relationship Changes")
                            st.info("This visualization is not implemented yet. Consider using NetworkX to create a before/after graph of course relationships.")
                        except Exception as e:
                            st.error(f"Error creating prerequisite visualization: {e}")
            else:
                st.write("No prerequisite changes recommended.")
        
        with tab5:
            other_recommendations = changes.get('other_recommendations', [])
            
            if other_recommendations:
                for i, rec in enumerate(other_recommendations, 1):
                    st.write(f"{i}. {rec}")
            else:
                st.write("No other recommendations.")
        
        # Apply optimization button
        if st.button("Apply Optimized Schedule", use_container_width=True):
            # Update future schedule with optimized version
            st.session_state.future_schedule = results['optimized_schedule']
            
            # Update course configs if needed
            capacity_changes = changes.get('capacity_changes', [])
            for change in capacity_changes:
                course = change.get('course')
                recommended_capacity = change.get('recommended_capacity')
                
                if course and recommended_capacity and course in st.session_state.course_configs:
                    st.session_state.course_configs[course]['max_capacity'] = recommended_capacity
            
            st.success("Applied optimized schedule! Go to Schedule Builder to view the updated schedule.")
            st.info("You may want to run a new simulation to verify the improvements.")
        
        # Navigation buttons
        st.write("---")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("← Back to Simulation", use_container_width=True):
                st.session_state.current_page = "Simulation"
                st.rerun()
        
        with col2:
            if st.button("Go to Schedule Builder →", use_container_width=True):
                st.session_state.current_page = "Schedule Builder"
                st.rerun()

def display_debug_panel():
    """Display a debug panel with session state and performance metrics"""
    st.sidebar.divider()
    st.sidebar.subheader("Debug Panel")
    
    # Session state explorer
    if st.sidebar.checkbox("Show Session State"):
        st.sidebar.json({k: str(v)[:100] + "..." if isinstance(v, (list, dict)) and len(str(v)) > 100 else v 
                         for k, v in st.session_state.items()})
    
    # Memory usage
    if st.sidebar.checkbox("Show Memory Usage"):
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        st.sidebar.metric("Memory Usage", f"{memory_info.rss / (1024 * 1024):.1f} MB")
    
    # Performance metrics
    if st.sidebar.checkbox("Show Performance Metrics"):
        st.sidebar.write("Performance metrics will be shown here")

if __name__ == "__main__":
    main()