import streamlit as st
import pandas as pd
import numpy as np
import datetime
import plotly.figure_factory as ff
import plotly.express as px
from data_processor import process_data, analyze_historical_data
from simulation_engine import run_simulation
from optimization import optimize_schedule
import json

st.set_page_config(page_title="Q-Course Training Schedule Optimizer", layout="wide")

def main():
    st.title("Q-Course Training Schedule Optimization System")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Upload Data", "Course Configuration", "Schedule Builder", 
                                      "Simulation", "Optimization"])
    
    # Initialize session state for data persistence between pages
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'course_configs' not in st.session_state:
        st.session_state.course_configs = {}
    else:
        # Apply backward compatibility for existing configurations
        for course, config in st.session_state.course_configs.items():
            # Convert empty string or invalid ratio to None for "No Ratio"
            if 'officer_enlisted_ratio' in config:
                if config['officer_enlisted_ratio'] == "" or not config['officer_enlisted_ratio']:
                    config['officer_enlisted_ratio'] = None
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

def display_upload_page():
    st.header("Upload Training Data")
    
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
            'max_capacity': default_class_size,
            'classes_per_year': default_classes_per_year,
            'reserved_seats': {
                'OF': 0,
                'ADE': 0,
                'NG': 0
            },
            'officer_enlisted_ratio': '1:4'
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
    
    # Configure prerequisites
    st.subheader("Prerequisites")

    with st.expander("How Prerequisites Work"):
        st.markdown("""
        ### Prerequisite Types:
        
        1. **All Required (AND)**: Student must complete ALL of the specified courses before taking this course.
        
        2. **Any Required (OR)**: Student must complete ANY ONE of the specified courses before taking this course.
        
        3. **Complex (AND/OR)**: For advanced prerequisite relationships. You can define:
           - Courses that ALL must be completed (AND logic)
           - Groups of courses where ANY ONE from EACH group must be completed (OR logic within groups, AND logic between groups)
           
        #### Example:
        If you want to require either "Course A" OR "Course B", AND either "Course C" OR "Course D":
        1. Leave the "Student must complete ALL of these courses" field empty
        2. Create Group 1 with "Course A" and "Course B"
        3. Create Group 2 with "Course C" and "Course D"
        """)
    
    # Radio button to select prerequisite logic type
    prereq_logic = st.radio(
        "Prerequisite Logic", 
        ["All Required (AND)", "Any Required (OR)", "Complex (AND/OR)"],
        index=0 if config['prerequisites']['type'] == 'AND' and not config['or_prerequisites'] else 
              1 if config['prerequisites']['type'] == 'OR' and not config['or_prerequisites'] else 2
    )
    
    prerequisite_options = [c for c in unique_courses if c != selected_course]
    
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
    
    # Display the current prerequisite structure for clarity
    st.subheader("Current Prerequisites Configuration")
    
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
    
    col1, col2, col3 = st.columns(3)
    with col1:
        class_start_date = st.date_input("Class Start Date", value=start_date)
    with col2:
        duration = st.number_input("Duration (days)", min_value=1, value=30)
    with col3:
        class_size = st.number_input("Class Size", 
                                  min_value=1, 
                                  max_value=int(st.session_state.course_configs[course_title]['max_capacity']),
                                  value=int(st.session_state.course_configs[course_title]['max_capacity']))
    
    class_end_date = class_start_date + datetime.timedelta(days=duration)
    
    # Add class button
    if st.button("Add Class to Schedule"):
        new_class = {
            "course_title": course_title,
            "start_date": class_start_date.strftime("%Y-%m-%d"),
            "end_date": class_end_date.strftime("%Y-%m-%d"),
            "size": class_size,
            "id": len(st.session_state.future_schedule) + 1
        }
        
        st.session_state.future_schedule.append(new_class)
        st.success(f"Added {course_title} starting on {class_start_date}")
    
    # Display current schedule
    st.subheader("Current Schedule")
    
    if st.session_state.future_schedule:
        schedule_df = pd.DataFrame(st.session_state.future_schedule)
        
        # Convert date strings to datetime for plotting
        schedule_df['start_date'] = pd.to_datetime(schedule_df['start_date'])
        schedule_df['end_date'] = pd.to_datetime(schedule_df['end_date'])
        
        # Create Gantt chart data
        gantt_data = []
        for i, row in schedule_df.iterrows():
            gantt_data.append(dict(
                Task=row['course_title'], 
                Start=row['start_date'], 
                Finish=row['end_date'], 
                Resource=f"Class {row['id']}"
            ))
        
        fig = ff.create_gantt(gantt_data, index_col='Resource', show_colorbar=True)
        st.plotly_chart(fig, use_container_width=True)
        
        # Schedule table view
        st.dataframe(schedule_df[['id', 'course_title', 'start_date', 'end_date', 'size']])
        
        # Remove class button
        class_to_remove = st.selectbox("Select class to remove", 
                                      options=schedule_df['id'].tolist(),
                                      format_func=lambda x: f"ID: {x} - {schedule_df[schedule_df['id'] == x]['course_title'].iloc[0]}")
        
        if st.button("Remove Class"):
            st.session_state.future_schedule = [c for c in st.session_state.future_schedule if c['id'] != class_to_remove]
            st.success(f"Removed class with ID {class_to_remove}")
            st.rerun()
    else:
        st.info("No classes scheduled yet. Add classes using the form above.")
    
    # Save/load schedule
    st.subheader("Save/Load Schedule")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Save Schedule"):
            schedule_json = json.dumps(st.session_state.future_schedule)
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
                loaded_schedule = json.load(uploaded_schedule)
                
                # Load and process course configurations if included
                if isinstance(loaded_schedule, dict) and 'schedule' in loaded_schedule and 'configurations' in loaded_schedule:
                    # Apply backward compatibility for loaded configurations
                    for course, config in loaded_schedule['configurations'].items():
                        if 'officer_enlisted_ratio' in config:
                            if config['officer_enlisted_ratio'] == "" or not config['officer_enlisted_ratio']:
                                config['officer_enlisted_ratio'] = None
                    
                    st.session_state.course_configs.update(loaded_schedule['configurations'])
                    st.session_state.future_schedule = loaded_schedule['schedule']
                else:
                    # Assume it's just a schedule without configurations
                    st.session_state.future_schedule = loaded_schedule
                
                st.success("Schedule loaded successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"Error loading schedule: {e}")

def display_simulation_page():
    st.header("Training Schedule Simulation")
    
    if not st.session_state.future_schedule:
        st.warning("Please build a schedule before running simulations.")
        return
    
    # Simulation settings
    st.subheader("Simulation Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        num_students = st.number_input("Number of students to simulate", min_value=10, value=100)
    
    with col2:
        num_iterations = st.number_input("Number of simulation iterations", min_value=1, value=10)
    
    # Advanced settings expander
    with st.expander("Advanced Simulation Settings"):
        randomize_factor = st.slider("Randomization Factor", min_value=0.0, max_value=1.0, value=0.1, 
                                    help="How much to randomize historical rates in simulation")
        
        st.write("Override Historical Pass Rates:")
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
                'randomize_factor': randomize_factor,
                'override_rates': override_rates
            }
            
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
        tab1, tab2, tab3, tab4 = st.tabs(["Bottlenecks", "Student Flow", "Class Utilization", "Detailed Metrics"])
        
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
    
    # Run optimization button
    if st.button("Run Optimization"):
        with st.spinner("Optimizing schedule... This may take a few minutes."):
            # Prepare optimization inputs
            optimization_inputs = {
                'current_schedule': st.session_state.future_schedule,
                'course_configs': st.session_state.course_configs,
                'simulation_results': st.session_state.simulation_results,
                'historical_data': st.session_state.historical_analysis,
                'optimization_goal': optimization_goal,
                'iterations': optimization_iterations,
                'constraint_weight': constraint_weight,
                'allow_capacity_changes': allow_capacity_changes,
                'allow_duration_changes': allow_duration_changes,
                'allow_prerequisite_changes': allow_prerequisite_changes
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
        tab1, tab2, tab3, tab4 = st.tabs(["Schedule Changes", "Capacity Changes", "Prerequisite Changes", "Other Recommendations"])
        
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
            prerequisite_changes = pd.DataFrame(changes['prerequisite_changes'])
            if not prerequisite_changes.empty:
                st.dataframe(prerequisite_changes)
            else:
                st.write("No prerequisite changes recommended.")
        
        with tab4:
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