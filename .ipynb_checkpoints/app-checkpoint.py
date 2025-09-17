import streamlit as st
import pandas as pd
import numpy as np
import datetime
import plotly.graph_objects as go
import plotly.express as px
import json
import copy
from collections import defaultdict

from data_processor import process_data, analyze_historical_data, extract_historical_arrival_patterns, extract_historical_mos_distribution
from simulation_engine import run_simulation
from optimization import optimize_schedule
from utils import ensure_config_compatibility

st.set_page_config(page_title="Training Schedule Optimizer", layout="wide")

def main():
    st.title("Training Schedule Optimization System")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Upload Data", "Course Configuration", "Schedule Builder", 
                                      "Simulation", "Optimization"])
    
    # Initialize session state for data persistence between pages
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
    if 'historical_analysis' not in st.session_state:
        st.session_state.historical_analysis = None
    if 'course_configs' not in st.session_state:
        st.session_state.course_configs = {}
    else:
        # Apply backward compatibility for existing configurations
        for course, config in st.session_state.course_configs.items():
            # Convert empty string or invalid ratio to None for "No Ratio"
            if 'officer_enlisted_ratio' in config:
                if config['officer_enlisted_ratio'] == "" or not config['officer_enlisted_ratio']:
                    config['officer_enlisted_ratio'] = None
                    
            # Handle prerequisites compatibility
            if 'prerequisites' in config and isinstance(config['prerequisites'], list):
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
    
    if 'future_schedule' not in st.session_state:
        st.session_state.future_schedule = []
    if 'simulation_results' not in st.session_state:
        st.session_state.simulation_results = None
    
    if page == "Upload Data":
        display_upload_page()
    elif page == "Course Configuration":
        display_config_page()
    elif page == "Schedule Builder":
        display_schedule_builder()
    elif page == "Simulation":
        display_simulation_page()
    elif page == "Optimization":
        display_optimization_page()
    
    # Add a reset button to the sidebar for debugging
    if st.sidebar.button("Reset Application"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.experimental_rerun()

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
        | CLS | Class number | 003 |
        | Cls Start Date | Class start date (MM/DD/YYYY) | 5/27/2025 |
        | Cls End Date | Class end date (MM/DD/YYYY) | 5/30/2025 |
        | Arrival Date | Student arrival date (MM/DD/YYYY) | 5/24/2025 |
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
        """)
        
        # Option to download a sample CSV
        sample_data = """FY,Course Number,Course Title,NAME,SSN,CLS,Cls Start Date,Cls End Date,Arrival Date,Res Stat,Reserve Reason,Reserve Reason Description,Input Stat,Input Date,Input Reason,Input Reason Description,Out Stat,Output Date,CP Pers Type,Group Type,Training MOS
        2025,2E-F253/011-F95,SF QUAL (ORIENTATION),SEABRIGHT NICK J,123456789,003,5/27/2025,5/30/2025,5/24/2025,R,,,I,5/27/2025,,,G,5/30/2025,E,ADE,18B
        2025,2E-F254/011-F96,SF QUAL (SMALL UNIT TACTICS),SEABRIGHT NICK J,123456789,003,6/2/2025,7/11/2025,5/30/2025,R,,,I,6/2/2025,,,G,7/11/2025,E,ADE,18B
        2025,3A-F38/012-F27,SERE HIGH RISK (LEVEL C),SEABRIGHT NICK J,123456789,010,7/14/2025,8/1/2025,7/11/2025,R,,,I,7/14/2025,,,G,8/1/2025,E,ADE,18B
        2025,011-18B30-C45,SF QUAL (SF WEAPONS SERGEANT) ALC,SEABRIGHT NICK J,123456789,003,8/4/2025,10/24/2025,8/1/2025,R,,,I,8/4/2025,,,E,ADE,18B
        2025,600-C44 (ARSOF),ARSOF BASIC LEADER,SMITH JANE A,987654321,005,4/28/2025,5/19/2025,4/25/2025,R,,,I,4/28/2025,,,G,5/19/2025,O,OF,18A
        2024,2E-F133/011-SQIW,SF ADV RECON TGT ANALY,JOHNSON ROBERT T,456789123,001,10/10/2024,12/9/2024,10/7/2024,R,,,I,10/10/2024,,,G,12/9/2024,E,NG,18C"""
        
        st.download_button(
            label="Download Sample CSV",
            data=sample_data,
            file_name="sample_training_data.csv",
            mime="text/csv"
        )
    
    uploaded_file = st.file_uploader("Upload training data CSV", type=["csv"])
    
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            st.session_state.data = data
            
            # Display data preview
            st.subheader("Data Preview")
            st.dataframe(data.head())
            
            # Process data
            processed_data = process_data(data)
            st.session_state.processed_data = processed_data
            
            # Display basic statistics
            st.subheader("Data Statistics")
            unique_courses = len(data['Course Title'].unique())
            unique_students = len(data['SSN'].unique())
            st.write(f"Number of unique courses: {unique_courses}")
            st.write(f"Number of unique students: {unique_students}")
            
            # Historical analysis
            historical_analysis = analyze_historical_data(processed_data)
            st.session_state.historical_analysis = historical_analysis
            
            # Extract historical arrival patterns and MOS distribution
            historical_arrival_patterns = extract_historical_arrival_patterns(processed_data)
            historical_mos_distribution = extract_historical_mos_distribution(processed_data)
            
            st.session_state.historical_arrival_patterns = historical_arrival_patterns
            st.session_state.historical_mos_distribution = historical_mos_distribution
            
            # Display extracted patterns
            with st.expander("Historical Patterns"):
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
                            labels={"x": "Month", "y": "Percentage of Students"}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                if historical_mos_distribution:
                    st.write("### Historical MOS Distribution")
                    mos_data = []
                    for mos, percentage in historical_mos_distribution.items():
                        mos_data.append({"MOS": mos, "Percentage": percentage * 100})
                    
                    fig = px.pie(
                        mos_data,
                        values="Percentage",
                        names="MOS",
                        title="Historical MOS Distribution"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            st.success("Data successfully loaded and processed!")
            
        except Exception as e:
            st.error(f"Error processing data: {e}")
    else:
        st.info("Please upload a CSV file with the training data.")

def display_config_page():
    st.header("Course Configuration")
    
    if 'processed_data' not in st.session_state or st.session_state.processed_data is None:
        st.warning("Please upload and process data first.")
        return
    
    # Get unique courses
    unique_courses = st.session_state.processed_data['Course Title'].unique()
    
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
            'officer_enlisted_ratio': None  # Default to No Ratio
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
                default=config['prerequisites']['courses']
            )
            config['prerequisites']['courses'] = selected_prerequisites
            config['or_prerequisites'] = []  # Clear any OR prerequisites
            
        elif prereq_logic == "Any Required (OR)":
            # Simple OR logic - any course can be taken
            config['prerequisites']['type'] = 'OR'
            selected_prerequisites = st.multiselect(
                "Student must complete ANY ONE of these courses:", 
                prerequisite_options, 
                default=config['prerequisites']['courses']
            )
            config['prerequisites']['courses'] = selected_prerequisites
            config['or_prerequisites'] = []  # Clear any complex OR prerequisites
            
        else:  # Complex AND/OR
            # For complex logic, we'll use the or_prerequisites structure
            st.write("Define complex prerequisite relationships:")
            
            # First, define any required courses (AND logic)
            and_prerequisites = st.multiselect(
                "Student must complete ALL of these courses (leave empty if none):", 
                prerequisite_options, 
                default=config['prerequisites']['courses'] if config['prerequisites']['type'] == 'AND' else []
            )
            config['prerequisites'] = {
                'type': 'AND',
                'courses': and_prerequisites
            }
            
            # Then, define sets of OR prerequisites
            st.write("AND the student must complete at least one course from EACH of the following groups:")
            
            # Initialize or_prerequisites if needed
            if 'or_prerequisites' not in config or not config['or_prerequisites']:
                config['or_prerequisites'] = [[]]  # Start with one empty group
            
            # Display existing OR groups
            or_groups = config['or_prerequisites'].copy()
            updated_or_groups = []
            
            for i, group in enumerate(or_groups):
                col1, col2 = st.columns([4, 1])
                
                with col1:
                    selected_or_courses = st.multiselect(
                        f"Group {i+1}: Student must complete ANY ONE of these courses:",
                        [c for c in prerequisite_options if c not in and_prerequisites],
                        default=group
                    )
                    if selected_or_courses:  # Only add non-empty groups
                        updated_or_groups.append(selected_or_courses)
                
                with col2:
                    if st.button(f"Remove Group {i+1}", key=f"remove_group_{i}"):
                        pass  # We'll filter out this group by not adding it to updated_or_groups
            
            # Add a button to add a new OR group
            if st.button("Add Another OR Group"):
                updated_or_groups.append([])  # Add an empty group
            
            config['or_prerequisites'] = updated_or_groups
        
    else:  # MOS-Specific Paths
        st.write("Configure prerequisites for each MOS training path:")
        
        # Option to make this course required for all MOS paths
        required_for_all = st.checkbox("This course is required for all MOS paths", 
                                      value=config.get('required_for_all_mos', False))
        config['required_for_all_mos'] = required_for_all
        
        if required_for_all:
            st.info("This course will be required for all students regardless of MOS path.")
        
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
                        default=config['mos_paths'].get(mos, [])
                    )
                    config['mos_paths'][mos] = mos_prereqs
                else:
                    # If not part of this MOS path, clear prerequisites
                    config['mos_paths'][mos] = []
    
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
    if st.button("Save Configuration"):
        st.session_state.course_configs[selected_course] = config
        st.success(f"Configuration for {selected_course} saved successfully!")

def display_schedule_builder():
    st.header("Future Schedule Builder")
    
    if not st.session_state.course_configs:
        st.warning("Please configure courses before building a schedule.")
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
            if st.button("1 Week"):
                st.session_state.temp_start = start_date
                st.session_state.temp_end = start_date + datetime.timedelta(days=7)
                st.experimental_rerun()
        
        with col2:
            if st.button("2 Weeks"):
                st.session_state.temp_start = start_date
                st.session_state.temp_end = start_date + datetime.timedelta(days=14)
                st.experimental_rerun()
        
        with col3:
            if st.button("1 Month"):
                st.session_state.temp_start = start_date
                st.session_state.temp_end = start_date + datetime.timedelta(days=30)
                st.experimental_rerun()
        
        with col4:
            if st.button(f"Historical ({default_duration} days)"):
                st.session_state.temp_start = start_date
                st.session_state.temp_end = start_date + datetime.timedelta(days=default_duration)
                st.experimental_rerun()
    
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
        class_size = st.number_input("Class Size", 
                                     min_value=1, 
                                     max_value=st.session_state.course_configs[course_title]['max_capacity'],
                                     value=st.session_state.course_configs[course_title]['max_capacity'])
    
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
    
    # Add Training MOS allocation
    st.subheader("Training MOS Allocation")
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
        if st.button("Add Class to Schedule"):
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
    
    # Display current schedule
    st.subheader("Current Schedule")
    
    if st.session_state.future_schedule:
        schedule_df = pd.DataFrame(st.session_state.future_schedule)
        
        # Display raw data for debugging
        with st.expander("Debug: Schedule Data"):
            st.write("Raw Schedule Data:")
            st.write(schedule_df)
        
        # Convert date strings to datetime for plotting
        try:
            schedule_df['start_date'] = pd.to_datetime(schedule_df['start_date'])
            schedule_df['end_date'] = pd.to_datetime(schedule_df['end_date'])
            
            # Filter options
            with st.expander("Chart Options"):
                # Filter by course
                unique_courses = schedule_df['course_title'].unique()
                st.write(f"Available courses: {len(unique_courses)}")
                
                selected_courses = st.multiselect(
                    "Show only these courses (leave empty to show all):",
                    options=unique_courses,
                    default=[]
                )
                
                # Date range filter
                col1, col2 = st.columns(2)
                with col1:
                    min_date = schedule_df['start_date'].min().date()
                    start_date_filter = st.date_input("From date", min_date, key="gantt_start_filter")
                with col2:
                    max_date = schedule_df['end_date'].max().date()
                    end_date_filter = st.date_input("To date", max_date, key="gantt_end_filter")
            
            # Apply filters
            filtered_df = schedule_df.copy()
            
            if selected_courses:
                filtered_df = filtered_df[filtered_df['course_title'].isin(selected_courses)]
            
            filtered_df = filtered_df[
                (filtered_df['start_date'] >= pd.Timestamp(start_date_filter)) &
                (filtered_df['end_date'] <= pd.Timestamp(end_date_filter))
            ]
            
            # Debug information
            with st.expander("Debug: Filtered Data"):
                st.write(f"Filtered schedule data (rows: {len(filtered_df)})")
                st.write(filtered_df)
            
            # Check if we have data to display
            if filtered_df.empty:
                st.warning("No classes match the selected filters. Try adjusting the date range or course selection.")
            else:
                # Sort by course title first, then by start date
                filtered_df = filtered_df.sort_values(['course_title', 'start_date'])
                
                # Create a custom Gantt chart using plotly
                fig = go.Figure()
                
                # Get unique courses for coloring and positioning
                unique_courses = filtered_df['course_title'].unique()
                st.write(f"Courses to display: {len(unique_courses)}")
                
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
                    
                    # Get MOS allocation if available
                    mos_info = ""
                    if 'mos_allocation' in row:
                        mos_alloc = row['mos_allocation']
                        mos_info = "<br>MOS: " + ", ".join([f"{mos}: {count}" for mos, count in mos_alloc.items() if count > 0])
                    
                    # Add bar for class using scatter with lines
                    fig.add_trace(go.Scatter(
                        x=[start_date, end_date],
                        y=[course, course],
                        mode='lines',
                        line=dict(color=color_map[course], width=20),
                        name=f"{course} (Class {class_id})",
                        text=f"Class {class_id}: {course}<br>Start: {start_date.strftime('%Y-%m-%d')}<br>End: {end_date.strftime('%Y-%m-%d')}<br>Size: {row['size']}{mos_info}",
                        hoverinfo='text'
                    ))
                    
                    # Add text label for class ID
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
                
        except Exception as e:
            st.error(f"Error generating Gantt chart: {e}")
            st.write("Please check your schedule data format.")
        
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
            
            # Select columns to display
            display_columns = ['id', 'course_title', 'start_date', 'end_date', 'MOS Allocation', 'size']
            
            st.dataframe(display_df[display_columns])
        
        # Remove class button
        class_to_remove = st.selectbox("Select class to remove", 
                                      options=schedule_df['id'].tolist(),
                                      format_func=lambda x: f"ID: {x} - {schedule_df[schedule_df['id'] == x]['course_title'].iloc[0]}")
        
        if st.button("Remove Class"):
            st.session_state.future_schedule = [c for c in st.session_state.future_schedule if c['id'] != class_to_remove]
            st.success(f"Removed class with ID {class_to_remove}")
            st.experimental_rerun()
    else:
        st.info("No classes scheduled yet. Add classes using the form above.")
    
    # Add a button to add test data if needed
    with st.expander("Debug Tools"):
        if st.button("Add Test Data"):
            test_data = [
                {
                    "course_title": "Test Course 1",
                    "start_date": (datetime.date.today()).strftime("%Y-%m-%d"),
                    "end_date": (datetime.date.today() + datetime.timedelta(days=14)).strftime("%Y-%m-%d"),
                    "size": 30,
                    "mos_allocation": {'18A': 5, '18B': 10, '18C': 5, '18D': 5, '18E': 5},
                    "id": 999
                },
                {
                    "course_title": "Test Course 2",
                    "start_date": (datetime.date.today() + datetime.timedelta(days=7)).strftime("%Y-%m-%d"),
                    "end_date": (datetime.date.today() + datetime.timedelta(days=21)).strftime("%Y-%m-%d"),
                    "size": 25,
                    "mos_allocation": {'18A': 5, '18B': 5, '18C': 5, '18D': 5, '18E': 5},
                    "id": 998
                }
            ]
            st.session_state.future_schedule.extend(test_data)
            st.success("Test data added. Please refresh the page.")
        
        if st.button("Debug: Fix Schedule Date Formats"):
            fixed_schedule = []
            for class_item in st.session_state.future_schedule:
                # Ensure dates are in string format YYYY-MM-DD
                try:
                    start_date = pd.to_datetime(class_item['start_date']).strftime('%Y-%m-%d')
                    end_date = pd.to_datetime(class_item['end_date']).strftime('%Y-%m-%d')
                    
                    fixed_item = {
                        'id': class_item['id'],
                        'course_title': class_item['course_title'],
                        'start_date': start_date,
                        'end_date': end_date,
                        'size': int(class_item['size'])
                    }
                    
                    # Copy MOS allocation if it exists
                    if 'mos_allocation' in class_item:
                        fixed_item['mos_allocation'] = class_item['mos_allocation']
                    
                    fixed_schedule.append(fixed_item)
                except Exception as e:
                    st.error(f"Error fixing class {class_item.get('id')}: {e}")
            
            st.session_state.future_schedule = fixed_schedule
            st.success("Schedule date formats fixed. Please refresh the page.")
    
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
        if st.button("Save Schedule"):
            # Create a complete package with schedule and configurations
            save_data = {
                'schedule': st.session_state.future_schedule,
                'configurations': st.session_state.course_configs
            }
            schedule_json = json.dumps(save_data)
            st.download_button(
                label="Download Schedule JSON",
                data=schedule_json,
                file_name="training_schedule.json",
                mime="application/json"
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
                        # Handle officer_enlisted_ratio compatibility
                        if 'officer_enlisted_ratio' in config:
                            if config['officer_enlisted_ratio'] == "" or not config['officer_enlisted_ratio']:
                                config['officer_enlisted_ratio'] = None
                        
                        # Handle prerequisites compatibility
                        if 'prerequisites' in config and isinstance(config['prerequisites'], list):
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
                    
                    st.session_state.course_configs.update(loaded_data['configurations'])
                    st.session_state.future_schedule = loaded_data['schedule']
                else:
                    # Assume it's just a schedule without configurations
                    st.session_state.future_schedule = loaded_data
                
                st.success("Schedule loaded successfully!")
                st.experimental_rerun()
            except Exception as e:
                st.error(f"Error loading schedule: {e}")

def display_simulation_page():
    st.header("Training Schedule Simulation")
    
    if not st.session_state.future_schedule:
        st.warning("Please build a schedule before running simulations.")
        return
    
    # Option to use historical data for simulation settings
    use_historical_data = st.checkbox(
        "Use historical data for simulation settings", 
        value=True,
        help="When enabled, the simulation will use patterns from historical data for student arrivals, MOS distribution, and pass rates."
    )
    
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
                            labels={"x": "Month", "y": "Percentage of Students"}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                if historical_mos_distribution:
                    st.write("### Historical MOS Distribution")
                    mos_data = []
                    for mos, percentage in historical_mos_distribution.items():
                        mos_data.append({"MOS": mos, "Percentage": percentage * 100})
                    
                    fig = px.pie(
                        mos_data,
                        values="Percentage",
                        names="MOS",
                        title="Historical MOS Distribution"
                    )
                    st.plotly_chart(fig, use_container_width=True)
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
        
        # Use historical MOS distribution if available and selected
        if use_historical_data and historical_mos_distribution:
            st.info("Using historical MOS distribution data")
            mos_distribution = historical_mos_distribution.copy()
            
            # Display the historical distribution but allow adjustments
            mos_cols = st.columns(5)
            
            with mos_cols[0]:
                mos_distribution['18A'] = st.slider("18A (Officer)", 0, 100, 
                                                  int(historical_mos_distribution.get('18A', 0.2) * 100)) / 100
            with mos_cols[1]:
                mos_distribution['18B'] = st.slider("18B (Weapons)", 0, 100, 
                                                  int(historical_mos_distribution.get('18B', 0.2) * 100)) / 100
            with mos_cols[2]:
                mos_distribution['18C'] = st.slider("18C (Engineer)", 0, 100, 
                                                  int(historical_mos_distribution.get('18C', 0.2) * 100)) / 100
            with mos_cols[3]:
                mos_distribution['18D'] = st.slider("18D (Medical)", 0, 100, 
                                                  int(historical_mos_distribution.get('18D', 0.2) * 100)) / 100
            with mos_cols[4]:
                mos_distribution['18E'] = st.slider("18E (Communications)", 0, 100, 
                                                  int(historical_mos_distribution.get('18E', 0.2) * 100)) / 100
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
        
        # Normalize MOS distribution
        total = sum(mos_distribution.values())
        if total > 0:
            normalized_mos_distribution = {mos: value / total for mos, value in mos_distribution.items()}
        else:
            normalized_mos_distribution = {'18A': 0.2, '18B': 0.2, '18C': 0.2, '18D': 0.2, '18E': 0.2}
        
        # Rest of advanced settings
        randomize_factor = st.slider("Randomization Factor", min_value=0.0, max_value=1.0, value=0.1, 
                                    help="How much to randomize historical rates in simulation")
        
        st.write("Course Pass Rates:")
        override_rates = {}
        
        # Create 3 columns to fit more courses per row
        cols = st.columns(3)
        for i, course in enumerate(st.session_state.course_configs.keys()):
            col_idx = i % 3
            with cols[col_idx]:
                hist_rate = st.session_state.historical_analysis.get(course, {}).get('pass_rate', 0.8)
                override_rates[course] = st.slider(f"{course[:20]}...", 
                                                min_value=0.0, 
                                                max_value=1.0, 
                                                value=float(hist_rate))
    
    # Run simulation button
    if st.button("Run Simulation"):
        with st.spinner("Running simulation..."):
            # Prepare inputs for simulation
            simulation_inputs = {
                'schedule': st.session_state.future_schedule,
                'course_configs': st.session_state.course_configs,
                'historical_data': st.session_state.historical_analysis,
                'num_students': num_students,
                'num_iterations': num_iterations,
                'use_historical_data': use_historical_data,
                'historical_arrival_patterns': historical_arrival_patterns,
                'historical_mos_distribution': normalized_mos_distribution,
                'randomize_factor': randomize_factor,
                'override_rates': override_rates
            }
            
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
            
            # Run simulation
            simulation_results = run_simulation(simulation_inputs)
            st.session_state.simulation_results = simulation_results
            
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
        
        # Tabs for different visualizations
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["Bottlenecks", "Student Flow", "Class Utilization", "MOS Analysis", "Detailed Metrics"])
        
        with tab1:
            st.subheader("Bottleneck Analysis")
            
            # Bottleneck chart
            bottleneck_df = pd.DataFrame(results['bottlenecks'])
            fig = px.bar(bottleneck_df, x='course', y='wait_time', 
                        title="Average Wait Time Before Each Course (Days)",
                        color='wait_time', height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # Bottleneck table
            st.dataframe(bottleneck_df.sort_values('wait_time', ascending=False))
        
        with tab2:
            st.subheader("Student Flow Analysis")
            
            # Student progression visualization
            fig = px.line(results['student_progression'], x='time', y='count', color='stage',
                         title="Student Progression Over Time", height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # Completion time distribution
            fig = px.histogram(results['completion_times'], x='days',
                              title="Distribution of Student Completion Times",
                              labels={'days': 'Days to Complete All Courses'},
                              height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.subheader("Class Utilization")
            
            # Class utilization chart
            util_df = pd.DataFrame(results['class_utilization'])
            fig = px.bar(util_df, x='course', y='utilization', 
                        title="Class Capacity Utilization",
                        color='utilization', height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # Classes with low utilization
            st.write("Classes with Low Utilization (<60%)")
            low_util = util_df[util_df['utilization'] < 0.6].sort_values('utilization')
            if not low_util.empty:
                st.dataframe(low_util)
            else:
                st.write("No classes with low utilization found.")
        
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
                st.dataframe(mos_df)
                
                # Create visualization for MOS metrics
                fig = px.bar(
                    mos_df, 
                    x='MOS', 
                    y=['Avg Completion Time (days)', 'Avg Wait Time (days)'],
                    barmode='group',
                    title='Average Completion and Wait Times by MOS',
                    height=400
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
                            height=500
                        )
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No MOS-specific data available in simulation results. Try running the simulation again with MOS paths enabled.")
        
        with tab5:
            st.subheader("Detailed Metrics")
            
            # Detailed metrics table
            detailed_metrics = pd.DataFrame(results['detailed_metrics'])
            st.dataframe(detailed_metrics)
            
            # Export results button
            csv = detailed_metrics.to_csv(index=False)
            st.download_button(
                label="Download Detailed Results CSV",
                data=csv,
                file_name="simulation_results.csv",
                mime="text/csv"
            )
        
        # If using historical data, show comparison between historical and simulated patterns
        if use_historical_data and historical_arrival_patterns:
            with st.expander("Historical vs. Simulated Patterns"):
                st.write("### Comparison of Historical and Simulated Patterns")
                
                # Create two columns for comparison
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("#### Historical Patterns")
                    
                    if historical_arrival_patterns:
                        st.write(f"Average arrival days before class: {historical_arrival_patterns.get('avg_days_before', 'N/A')}")
                        
                        # Show historical monthly distribution
                        if 'monthly_distribution' in historical_arrival_patterns:
                            monthly_data = historical_arrival_patterns['monthly_distribution']
                            months = list(monthly_data.keys())
                            values = list(monthly_data.values())
                            
                            fig = px.bar(
                                x=months, 
                                y=values,
                                title="Historical Monthly Arrival Distribution",
                                labels={"x": "Month", "y": "Percentage"}
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    
                    if historical_mos_distribution:
                        st.write("#### Historical MOS Distribution")
                        mos_data = []
                        for mos, percentage in historical_mos_distribution.items():
                            mos_data.append({"MOS": mos, "Percentage": percentage * 100})
                        
                        fig = px.pie(
                            mos_data,
                            values="Percentage",
                            names="MOS",
                            title="Historical MOS Distribution"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.write("#### Simulated Patterns")
                    
                    # Get simulated arrival patterns from results
                    if 'arrival_patterns' in results:
                        arrival_patterns = results['arrival_patterns']
                        
                        st.write(f"Average arrival days before class: {arrival_patterns.get('avg_days_before', 'N/A')}")
                        
                        # Show simulated monthly distribution
                        if 'monthly_distribution' in arrival_patterns:
                            monthly_data = arrival_patterns['monthly_distribution']
                            months = list(monthly_data.keys())
                            values = list(monthly_data.values())
                            
                            fig = px.bar(
                                x=months, 
                                y=values,
                                title="Simulated Monthly Arrival Distribution",
                                labels={"x": "Month", "y": "Percentage"}
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    
                    # Get simulated MOS distribution from results
                    if 'mos_metrics' in results:
                        mos_metrics = results['mos_metrics']
                        
                        mos_data = []
                        total_students = sum(metrics.get('count', 0) for metrics in mos_metrics.values())
                        
                        for mos, metrics in mos_metrics.items():
                            count = metrics.get('count', 0)
                            percentage = count / total_students * 100 if total_students > 0 else 0
                            mos_data.append({"MOS": mos, "Percentage": percentage})
                        
                        fig = px.pie(
                            mos_data,
                            values="Percentage",
                            names="MOS",
                            title="Simulated MOS Distribution"
                        )
                        st.plotly_chart(fig, use_container_width=True)

def display_optimization_page():
    st.header("Schedule Optimization")
    
    if st.session_state.simulation_results is None:
        st.warning("Please run a simulation before attempting optimization.")
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
    if st.button("Run Optimization"):
        with st.spinner("Optimizing schedule... This may take a few minutes."):
            # Prepare optimization inputs
            course_configs_copy = copy.deepcopy(st.session_state.course_configs)
            
            # Apply backward compatibility
            for course, config in course_configs_copy.items():
                if 'officer_enlisted_ratio' in config:
                    if config['officer_enlisted_ratio'] == "" or not config['officer_enlisted_ratio']:
                        config['officer_enlisted_ratio'] = None
            
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
            
            # Run optimization
            optimization_results = optimize_schedule(optimization_inputs)
            st.session_state.optimization_results = optimization_results
            
            st.success("Optimization completed successfully!")
    
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
            schedule_changes = pd.DataFrame(changes['schedule_changes'])
            if not schedule_changes.empty:
                st.dataframe(schedule_changes)
                
                # Visualization of schedule changes
                fig = px.timeline(
                    schedule_changes, 
                    x_start="original_start", 
                    x_end="original_end", 
                    y="course",
                    color="course",
                    opacity=0.5,
                    title="Schedule Changes (Original vs. Optimized)"
                )
                
                # Add optimized schedule
                for _, row in schedule_changes.iterrows():
                    fig.add_trace(
                        px.timeline(
                            pd.DataFrame([{
                                "course": row["course"],
                                "start": row["new_start"],
                                "end": row["new_end"]
                            }]),
                            x_start="start",
                            x_end="end",
                            y="course",
                            color="course"
                        ).data[0]
                    )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.write("No schedule changes recommended.")
        
        with tab2:
            capacity_changes = pd.DataFrame(changes['capacity_changes'])
            if not capacity_changes.empty:
                st.dataframe(capacity_changes)
                
                # Visualization of capacity changes
                fig = px.bar(
                    capacity_changes,
                    x="course",
                    y=["original_capacity", "recommended_capacity"],
                    barmode="group",
                    title="Class Capacity Recommendations"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.write("No capacity changes recommended.")
        
        with tab3:
            mos_allocation_changes = pd.DataFrame(changes.get('mos_allocation_changes', []))
            if not mos_allocation_changes.empty:
                st.dataframe(mos_allocation_changes)
                
                # Visualization of MOS allocation changes
                if len(mos_allocation_changes) > 0:
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
                        barmode='group'
                    )
                    
                    # Adjust layout for better readability
                    fig.update_layout(
                        height=400 * (len(mos_allocation_changes) // 2 + 1),
                        margin=dict(t=100, b=50)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.write("No MOS allocation changes recommended.")
        
        with tab4:
            prerequisite_changes = pd.DataFrame(changes['prerequisite_changes'])
            if not prerequisite_changes.empty:
                st.dataframe(prerequisite_changes)
            else:
                st.write("No prerequisite changes recommended.")
        
        with tab5:
            other_recommendations = changes['other_recommendations']
            for i, rec in enumerate(other_recommendations, 1):
                st.write(f"{i}. {rec}")
        
        # Apply optimization button
        if st.button("Apply Optimized Schedule"):
            # Update future schedule with optimized version
            st.session_state.future_schedule = results['optimized_schedule']
            
            # Update course configs if needed
            for course, capacity in changes.get('capacity_changes', {}).items():
                if course in st.session_state.course_configs:
                    st.session_state.course_configs[course]['max_capacity'] = capacity['recommended_capacity']
            
            st.success("Applied optimized schedule! Go to Schedule Builder to view the updated schedule.")
            st.info("You may want to run a new simulation to verify the improvements.")

if __name__ == "__main__":
    main()