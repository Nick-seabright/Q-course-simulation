import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict
import streamlit as st
import copy

"""
Data Processing Module Training Schedule Optimizer

Status Code Reference:
--------------------------
INPUT STAT:
- H = HOLD (SHOWED DID NOT START OR DID NOT GRAD)
- I = NEW INPUT
- J = RETRAINEE IN, FROM ANOTHER COURSE OF INSTRUCTION
- N = NO SHOW
- Q = RECYCLE IN, FROM ANOTHER CLASS, SAME COURSE
- U = SHOWED, DID NOT BEGIN TRNG (POST APPROP REASON CODE)

OUT STAT:
- G = GRADUATE, SUCCESSFULLY COMPLETED CLASS
- K = RETRAINEE OUT, TO ANOTHER COURSE OF INSTRUCTION
- L = RECYCLE OUT, TO ANOTHER CLASS, SAME COURSE
- Z = NON-SUCCESSFUL COMPLETION

RES STAT:
- C = CANCELLED RESERVATION
- R = VALID RESERVATION
- M = MEP RESERVATION
- W = WAITING FOR RESERVATION
"""

def safe_st_warning(message):
    """Safe wrapper for st.warning that handles case where streamlit isn't available"""
    try:
        st.warning(message)
    except:
        print(f"WARNING: {message}")

def safe_st_error(message):
    """Safe wrapper for st.error that handles case where streamlit isn't available"""
    try:
        st.error(message)
    except:
        print(f"ERROR: {message}")

def process_data(raw_data):
    """
    Process raw training data to extract relevant information
    Args:
        raw_data (pd.DataFrame): Raw training data
    Returns:
        pd.DataFrame: Processed data
    """
    # Make a copy to avoid modifying the original
    data = raw_data.copy()
    
    # Check for required columns
    required_columns = ['FY', 'Course Title', 'SSN', 'Cls Start Date', 'Cls End Date']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        error_msg = f"Missing required columns: {', '.join(missing_columns)}"
        safe_st_error(error_msg)
        raise ValueError(error_msg)
    
    # Convert date columns to datetime - use column names, not positions
    date_columns = ['Cls Start Date', 'Cls End Date', 'Input Date', 'Output Date']
    
    # Add Arrival Date to date columns if it exists
    if 'Arrival Date' in data.columns:
        date_columns.append('Arrival Date')
    
    for col in date_columns:
        if col in data.columns:
            data[col] = pd.to_datetime(data[col], errors='coerce')
    
    # Group by student (SSN) to identify each student's first class
    if 'SSN' in data.columns:
        student_groups = data.groupby('SSN')
        
        # If Arrival Date doesn't exist or has missing values, add/update it
        if 'Arrival Date' not in data.columns:
            # Create the column
            data['Arrival Date'] = pd.NaT
            safe_st_warning("Arrival Date column not found. Estimating arrival as 3 weeks before each student's first class.")
        
        # Process each student
        for ssn, student_data in student_groups:
            # Sort by class start date to find first class
            if 'Cls Start Date' in student_data.columns:
                sorted_classes = student_data.sort_values('Cls Start Date')
                if len(sorted_classes) > 0:
                    first_class_start = sorted_classes.iloc[0]['Cls Start Date']
                    
                    # Check each row for this student
                    for idx in sorted_classes.index:
                        # If arrival date is missing or invalid (after class start)
                        if pd.isna(data.loc[idx, 'Arrival Date']) or data.loc[idx, 'Arrival Date'] > data.loc[idx, 'Cls Start Date']:
                            # Set arrival date to 3 weeks (21 days) before first class
                            data.loc[idx, 'Arrival Date'] = first_class_start - pd.Timedelta(days=21)
        
        # Perform a final check to ensure all arrival dates are before class start dates
        invalid_arrivals = data[data['Arrival Date'] > data['Cls Start Date']]
        if len(invalid_arrivals) > 0:
            safe_st_warning(f"Found {len(invalid_arrivals)} records with arrival dates after class start dates. These have been corrected.")
            # Fix any remaining invalid arrival dates
            data.loc[invalid_arrivals.index, 'Arrival Date'] = data.loc[invalid_arrivals.index, 'Cls Start Date'] - pd.Timedelta(days=1)
        
        # Calculate days between arrival and class start
        data['Days Before Class'] = (data['Cls Start Date'] - data['Arrival Date']).dt.days
    
    # Calculate class duration
    data['Duration'] = (data['Cls End Date'] - data['Cls Start Date']).dt.days
    
    # Determine student status based on the correct status codes
    # Determine if a student started a class (I=New Input, J=Retrainee In, Q=Recycle In)
    if 'Input Stat' in data.columns:
        data['Started'] = data['Input Stat'].isin(['I', 'J', 'Q'])
    else:
        data['Started'] = True  # Assume started if no input status
    
    # Determine if a student graduated (G=Graduate)
    if 'Out Stat' in data.columns:
        data['Graduated'] = data['Out Stat'] == 'G'
    else:
        data['Graduated'] = False  # Assume not graduated if no output status
    
    # Determine if a student recycled (L=Recycle Out)
    if 'Out Stat' in data.columns:
        data['Recycled'] = data['Out Stat'] == 'L'
    else:
        data['Recycled'] = False
    
    # Determine if a student was a no-show (N=No Show)
    if 'Input Stat' in data.columns:
        data['NoShow'] = data['Input Stat'] == 'N'
    else:
        data['NoShow'] = False
    
    # Determine if a student was retrainee out (K=Retrainee Out)
    if 'Out Stat' in data.columns:
        data['RetraineeOut'] = data['Out Stat'] == 'K'
    else:
        data['RetraineeOut'] = False
    
    # Determine if a reservation was cancelled (C=Cancelled Reservation)
    if 'Res Stat' in data.columns:
        data['Cancelled'] = data['Res Stat'] == 'C'
    else:
        data['Cancelled'] = False
    
    # Determine if a student failed or had non-successful completion (Z=Non-Successful Completion)
    if 'Out Stat' in data.columns:
        data['Failed'] = data['Out Stat'] == 'Z'
    else:
        data['Failed'] = False
    
    # Extract personnel type (Officer vs Enlisted)
    if 'CP Pers Type' in data.columns:
        data['PersonnelType'] = data['CP Pers Type'].map({'O': 'Officer', 'E': 'Enlisted'})
    else:
        data['PersonnelType'] = 'Unknown'
    
    # Extract group type (Active Duty, National Guard, etc.)
    if 'Group Type' in data.columns:
        data['GroupType'] = data['Group Type']
    else:
        data['GroupType'] = 'Unknown'
    
    # Extract or infer Training MOS if available
    process_training_mos(data)
    
    return data

def process_training_mos(data):
    """
    Process Training MOS data, inferring it if not explicitly provided
    Args:
        data (pd.DataFrame): Training data to process
    """
    # Check if Training MOS column exists
    if 'Training MOS' in data.columns:
        # Use existing column
        data['TrainingMOS'] = data['Training MOS']
    else:
        # Try to infer from other data
        data['TrainingMOS'] = None
        
        # Officers are typically 18A
        if 'PersonnelType' in data.columns:
            data.loc[data['PersonnelType'] == 'Officer', 'TrainingMOS'] = '18A'
        
        # Try to infer enlisted MOS from course titles if available
        if 'Course Title' in data.columns:
            # Look for keywords in course titles
            for idx, row in data.iterrows():
                if pd.isna(data.loc[idx, 'TrainingMOS']) and data.loc[idx, 'PersonnelType'] == 'Enlisted':
                    course_title = str(row.get('Course Title', '')).upper()
                    if 'WEAPONS' in course_title or 'WEAP SGT' in course_title:
                        data.loc[idx, 'TrainingMOS'] = '18B'
                    elif 'ENGINEER' in course_title or 'ENG SGT' in course_title:
                        data.loc[idx, 'TrainingMOS'] = '18C'
                    elif 'MEDICAL' in course_title or 'MED SGT' in course_title or 'MEDIC' in course_title:
                        data.loc[idx, 'TrainingMOS'] = '18D'
                    elif 'COMM' in course_title or 'SIGNAL' in course_title or 'COMMO SGT' in course_title:
                        data.loc[idx, 'TrainingMOS'] = '18E'

def analyze_historical_data(processed_data):
    """
    Analyze historical training data to extract statistics by course
    Args:
        processed_data (pd.DataFrame): Processed training data
    Returns:
        dict: Dictionary of course statistics
    """
    # Group by course title
    grouped_data = processed_data.groupby('Course Title')
    course_stats = {}
    
    for course, group in grouped_data:
        # Skip courses with very few entries
        if len(group) < 3:
            continue
        
        # Calculate pass rate (G=Graduate)
        if group['Started'].sum() > 0:
            pass_rate = group['Graduated'].sum() / group['Started'].sum()
        else:
            pass_rate = 0
        
        # Calculate recycle rate (L=Recycle Out)
        if group['Started'].sum() > 0:
            recycle_rate = group['Recycled'].sum() / group['Started'].sum()
        else:
            recycle_rate = 0
        
        # Calculate no-show rate (N=No Show)
        reservation_count = len(group)
        if reservation_count > 0:
            no_show_rate = group['NoShow'].sum() / reservation_count
        else:
            no_show_rate = 0
        
        # Calculate cancellation rate (C=Cancelled Reservation)
        if reservation_count > 0:
            cancellation_rate = group['Cancelled'].sum() / reservation_count
        else:
            cancellation_rate = 0
        
        # Calculate failure rate (Z=Non-Successful Completion)
        if group['Started'].sum() > 0:
            failure_rate = group['Failed'].sum() / group['Started'].sum()
        else:
            failure_rate = 0
        
        # Calculate average class size
        classes = group.groupby(['CLS', 'Cls Start Date'])
        class_sizes = classes.size().values
        if len(class_sizes) > 0:
            avg_class_size = int(np.mean(class_sizes))
        else:
            avg_class_size = 0
        
        # Calculate average duration
        avg_duration = group['Duration'].mean()
        
        # Estimate classes per year based on historical data
        # Group by fiscal year and count unique class identifiers
        group['FY'] = group['FY'].astype(str)
        fy_classes = group.groupby('FY').apply(lambda x: x[['CLS']].drop_duplicates().shape[0])
        if len(fy_classes) > 0:
            classes_per_year = int(fy_classes.mean())
        else:
            classes_per_year = 0
        
        # Personnel composition
        officer_count = (group['PersonnelType'] == 'Officer').sum()
        enlisted_count = (group['PersonnelType'] == 'Enlisted').sum()
        total = officer_count + enlisted_count
        if total > 0:
            officer_ratio = officer_count / total
            enlisted_ratio = enlisted_count / total
        else:
            officer_ratio = 0
            enlisted_ratio = 0
        
        # Group type composition
        group_counts = group['GroupType'].value_counts()
        total_students = group_counts.sum()
        group_composition = {group_type: count / total_students for group_type, count in group_counts.items()}
        
        # MOS composition
        if 'TrainingMOS' in group.columns:
            mos_counts = group['TrainingMOS'].value_counts()
            total_mos = mos_counts.sum()
            mos_composition = {mos: count / total_mos for mos, count in mos_counts.items()}
        else:
            mos_composition = {}
        
        # Store all statistics
        course_stats[course] = {
            'pass_rate': pass_rate,
            'recycle_rate': recycle_rate,
            'no_show_rate': no_show_rate,
            'cancellation_rate': cancellation_rate,
            'failure_rate': failure_rate,
            'avg_class_size': avg_class_size,
            'avg_duration': avg_duration,
            'classes_per_year': classes_per_year,
            'officer_ratio': officer_ratio,
            'enlisted_ratio': enlisted_ratio,
            'group_composition': group_composition,
            'mos_composition': mos_composition
        }
    
    return course_stats

def extract_historical_arrival_patterns(processed_data):
    """
    Extract historical student arrival patterns from processed data
    Args:
        processed_data (pd.DataFrame): Processed training data
    Returns:
        dict: Dictionary of arrival patterns
    """
    # Use Arrival Date directly if available, otherwise use Days Before Class
    if 'Arrival Date' in processed_data.columns:
        # Filter to only include valid data
        valid_data = processed_data.dropna(subset=['Arrival Date', 'Cls Start Date'])
        
        # Calculate days before class start that students arrive
        valid_data['days_before'] = (valid_data['Cls Start Date'] - valid_data['Arrival Date']).dt.days
        
        # Filter out negative values (arrival after class start) and unreasonably large values
        valid_arrivals = valid_data[(valid_data['days_before'] >= 0) & (valid_data['days_before'] <= 90)]
    else:
        # Use the calculated Days Before Class field
        valid_data = processed_data.dropna(subset=['Days Before Class'])
        valid_data['days_before'] = valid_data['Days Before Class']
        valid_arrivals = valid_data[(valid_data['days_before'] >= 0) & (valid_data['days_before'] <= 90)]
    
    if len(valid_arrivals) == 0:
        return None
    
    # Calculate average days before class start
    avg_days_before = valid_arrivals['days_before'].mean()
    
    # Extract monthly distribution of arrivals
    valid_arrivals['arrival_month'] = valid_arrivals['Arrival Date'].dt.month_name()
    monthly_counts = valid_arrivals['arrival_month'].value_counts()
    total_students = monthly_counts.sum()
    
    monthly_distribution = {}
    for month, count in monthly_counts.items():
        monthly_distribution[month] = count / total_students if total_students > 0 else 0
    
    return {
        'avg_days_before': avg_days_before,
        'monthly_distribution': monthly_distribution
    }

def extract_historical_mos_distribution(processed_data):
    """
    Extract historical MOS distribution from processed data
    Args:
        processed_data (pd.DataFrame): Processed training data
    Returns:
        dict: Dictionary of MOS distribution
    """
    # Define the MOS column to use
    mos_column = None
    
    # Check if MOS column exists (might be called "Training MOS" or similar)
    if 'TrainingMOS' in processed_data.columns:
        mos_column = 'TrainingMOS'
    else:
        # Try to find another suitable column
        possible_columns = ['Training MOS', 'MOS', 'TrainingMOS', 'TMOS']
        for col in possible_columns:
            if col in processed_data.columns:
                mos_column = col
                break
    
    # If no explicit MOS column found, try to infer from other data
    if not mos_column:
        # For example, Officers (CP Pers Type = 'O') might be 18A
        if 'CP Pers Type' in processed_data.columns:
            # Create synthetic MOS distribution based on personnel type
            officers = (processed_data['CP Pers Type'] == 'O').sum()
            enlisted = (processed_data['CP Pers Type'] == 'E').sum()
            total = officers + enlisted
            
            if total > 0:
                # Officers are 18A, distribute enlisted evenly among other MOS
                mos_distribution = {
                    '18A': officers / total,
                    '18B': enlisted / total / 4,
                    '18C': enlisted / total / 4,
                    '18D': enlisted / total / 4,
                    '18E': enlisted / total / 4
                }
                return mos_distribution
            
        # Default distribution if we can't extract from data
        return {'18A': 0.2, '18B': 0.2, '18C': 0.2, '18D': 0.2, '18E': 0.2}
    
    # If we have an MOS column, calculate the distribution
    mos_counts = processed_data[mos_column].value_counts()
    total_students = mos_counts.sum()
    
    # Map to standard MOS codes if needed
    mos_mapping = {
        'OFFICER': '18A',
        'WEAPONS': '18B',
        'ENGINEER': '18C',
        'MEDICAL': '18D',
        'COMMUNICATIONS': '18E'
    }
    
    mos_distribution = {}
    for mos, count in mos_counts.items():
        # Try to map the MOS to standard code
        standard_mos = mos
        for key, value in mos_mapping.items():
            if isinstance(mos, str) and key in mos.upper():
                standard_mos = value
                break
        
        # If not one of our standard MOS codes, group with the closest match
        if standard_mos not in ['18A', '18B', '18C', '18D', '18E']:
            # Default to distributing evenly among enlisted MOS if officer/enlisted info available
            if 'CP Pers Type' in processed_data.columns:
                # Check if this MOS is typically for officers
                mos_entries = processed_data[processed_data[mos_column] == mos]
                officer_count = (mos_entries['CP Pers Type'] == 'O').sum()
                enlisted_count = (mos_entries['CP Pers Type'] == 'E').sum()
                
                if officer_count > enlisted_count:
                    standard_mos = '18A'  # Assume officer MOS
                else:
                    # Distribute evenly among enlisted MOS for now
                    # In a real implementation, you might want more sophisticated mapping
                    standard_mos = np.random.choice(['18B', '18C', '18D', '18E'])
            else:
                # If no officer/enlisted info, just distribute evenly
                standard_mos = np.random.choice(['18A', '18B', '18C', '18D', '18E'])
        
        # Add to distribution
        if standard_mos in mos_distribution:
            mos_distribution[standard_mos] += count / total_students if total_students > 0 else 0
        else:
            mos_distribution[standard_mos] = count / total_students if total_students > 0 else 0
    
    # Ensure all standard MOS codes are present
    for mos in ['18A', '18B', '18C', '18D', '18E']:
        if mos not in mos_distribution:
            mos_distribution[mos] = 0
    
    return mos_distribution

def infer_prerequisites(processed_data):
    """
    Infer potential prerequisites based on historical student progression
    Args:
        processed_data (pd.DataFrame): Processed training data
    Returns:
        dict: Dictionary of likely prerequisites for each course
    """
    # Group data by student (SSN)
    student_groups = processed_data.groupby('SSN')
    
    # Track course sequences
    course_sequences = {}
    for ssn, student_data in student_groups:
        # Only consider graduated courses
        completed_courses = student_data[student_data['Graduated']].sort_values('Cls Start Date')
        
        # Skip if student completed fewer than 2 courses
        if len(completed_courses) < 2:
            continue
        
        # Get course sequence
        course_sequence = completed_courses['Course Title'].tolist()
        
        # Update course sequences
        for i in range(1, len(course_sequence)):
            current_course = course_sequence[i]
            if current_course not in course_sequences:
                course_sequences[current_course] = {}
            
            # Count all previous courses as potential prerequisites
            for j in range(i):
                prev_course = course_sequence[j]
                if prev_course not in course_sequences[current_course]:
                    course_sequences[current_course][prev_course] = 0
                course_sequences[current_course][prev_course] += 1
    
    # Calculate likelihood of prerequisites
    prerequisites = {}
    for course, potential_prereqs in course_sequences.items():
        # Total number of students who took this course
        total_students = sum(potential_prereqs.values())
        
        # Consider strong prerequisites (courses most students took)
        strong_prereqs = []
        potential_or_groups = []
        remaining_prereqs = []
        
        for prereq, count in potential_prereqs.items():
            ratio = count / total_students
            if ratio >= 0.9:  # Very strong prerequisite
                strong_prereqs.append(prereq)
            elif ratio >= 0.5:  # Potential OR relationship
                remaining_prereqs.append((prereq, ratio))
        
        # Look for potential OR relationships among remaining prerequisites
        # Two courses with similar percentages that add up to nearly 100% might be OR prerequisites
        while remaining_prereqs:
            prereq1, ratio1 = remaining_prereqs.pop(0)
            or_group = [prereq1]
            
            for prereq2, ratio2 in remaining_prereqs[:]:
                # If the two courses together cover almost all students (possible OR relationship)
                if 0.9 <= ratio1 + ratio2 <= 1.1 and abs(ratio1 - ratio2) < 0.3:
                    or_group.append(prereq2)
                    remaining_prereqs.remove((prereq2, ratio2))
            
            if len(or_group) > 1:
                potential_or_groups.append(or_group)
            elif len(or_group) == 1:
                # If just one course and it's taken by most students, consider it required
                if ratio1 >= 0.7:
                    strong_prereqs.append(prereq1)
        
        # Create the prerequisite structure
        if strong_prereqs or potential_or_groups:
            prereq_structure = {
                'prerequisites': {
                    'type': 'AND',
                    'courses': strong_prereqs
                },
                'or_prerequisites': potential_or_groups
            }
            prerequisites[course] = prereq_structure
    
    return prerequisites

def infer_mos_paths(processed_data):
    """
    Infer MOS-specific training paths from historical data
    Args:
        processed_data (pd.DataFrame): Processed training data
    Returns:
        dict: Dictionary of course to MOS path mappings
    """
    if 'TrainingMOS' not in processed_data.columns:
        return {}  # Can't infer MOS paths without MOS data
    
    # Group by course and MOS
    course_mos_groups = processed_data.groupby(['Course Title', 'TrainingMOS'])
    
    # Count students in each course by MOS
    course_mos_counts = course_mos_groups.size().reset_index(name='count')
    
    # Calculate total students per course
    course_totals = course_mos_counts.groupby('Course Title')['count'].sum().to_dict()
    
    # Calculate percentage of each MOS in each course
    course_mos_counts['percentage'] = course_mos_counts.apply(
        lambda row: row['count'] / course_totals[row['Course Title']] if row['Course Title'] in course_totals else 0,
        axis=1
    )
    
    # Initialize course to MOS path mappings
    mos_paths = {}
    
    # Standard MOS codes
    mos_codes = ['18A', '18B', '18C', '18D', '18E']
    
    # For each course, determine which MOS paths it belongs to
    for course in course_totals.keys():
        mos_paths[course] = {}
        course_data = course_mos_counts[course_mos_counts['Course Title'] == course]
        
        for mos in mos_codes:
            # Get percentage of this MOS in the course
            mos_data = course_data[course_data['TrainingMOS'] == mos]
            if not mos_data.empty:
                percentage = mos_data.iloc[0]['percentage']
                # If more than 10% of students with this MOS take this course,
                # consider it part of this MOS path
                if percentage >= 0.1:
                    mos_paths[course][mos] = True
                else:
                    mos_paths[course][mos] = False
            else:
                mos_paths[course][mos] = False
    
    # Convert the dictionary format to match the expected format in the app
    formatted_mos_paths = {}
    for course, mos_dict in mos_paths.items():
        formatted_mos_paths[course] = {
            'mos_paths': {
                mos: [] for mos, included in mos_dict.items() if included
            },
            'required_for_all_mos': all(mos_dict.values())
        }
    
    return formatted_mos_paths

def analyze_student_progression(processed_data):
    """
    Analyze student progression through courses over time
    Args:
        processed_data (pd.DataFrame): Processed training data
    Returns:
        dict: Dictionary of progression metrics
    """
    # Group data by student (SSN)
    student_groups = processed_data.groupby('SSN')
    
    # Track metrics
    completion_times = []
    wait_times = []
    progression_paths = []
    
    for ssn, student_data in student_groups:
        # Sort by class start date
        student_courses = student_data.sort_values('Cls Start Date')
        
        # Only consider students who completed at least one course
        graduated_courses = student_courses[student_courses['Graduated']]
        if len(graduated_courses) == 0:
            continue
        
        # Calculate completion time (from first class start to last class end)
        if len(graduated_courses) > 1:
            first_start = graduated_courses['Cls Start Date'].min()
            last_end = graduated_courses['Cls End Date'].max()
            completion_time = (last_end - first_start).days
            completion_times.append(completion_time)
        
        # Calculate wait times between courses
        if len(graduated_courses) > 1:
            sorted_courses = graduated_courses.sort_values('Cls Start Date')
            for i in range(1, len(sorted_courses)):
                prev_end = sorted_courses.iloc[i-1]['Cls End Date']
                curr_start = sorted_courses.iloc[i]['Cls Start Date']
                wait_time = (curr_start - prev_end).days
                # Only count positive wait times
                if wait_time > 0:
                    wait_times.append({
                        'student': ssn,
                        'prev_course': sorted_courses.iloc[i-1]['Course Title'],
                        'next_course': sorted_courses.iloc[i]['Course Title'],
                        'wait_time': wait_time
                    })
        
        # Record progression path
        if len(graduated_courses) > 0:
            course_sequence = graduated_courses['Course Title'].tolist()
            progression_paths.append({
                'student': ssn,
                'sequence': course_sequence
            })
    
    # Calculate average completion time
    avg_completion_time = np.mean(completion_times) if completion_times else 0
    
    # Calculate average wait time
    avg_wait_time = np.mean([w['wait_time'] for w in wait_times]) if wait_times else 0
    
    # Find common wait bottlenecks
    wait_bottlenecks = defaultdict(list)
    for wait in wait_times:
        key = (wait['prev_course'], wait['next_course'])
        wait_bottlenecks[key].append(wait['wait_time'])
    
    bottlenecks = [
        {
            'prev_course': prev,
            'next_course': next_,
            'avg_wait_time': np.mean(times),
            'count': len(times)
        }
        for (prev, next_), times in wait_bottlenecks.items()
    ]
    
    # Sort bottlenecks by average wait time
    bottlenecks.sort(key=lambda x: x['avg_wait_time'], reverse=True)
    
    # Find common progression paths
    path_counts = defaultdict(int)
    for path in progression_paths:
        path_tuple = tuple(path['sequence'])
        path_counts[path_tuple] += 1
    
    common_paths = [
        {
            'sequence': list(path),
            'count': count
        }
        for path, count in path_counts.items()
    ]
    
    # Sort by frequency
    common_paths.sort(key=lambda x: x['count'], reverse=True)
    
    return {
        'avg_completion_time': avg_completion_time,
        'avg_wait_time': avg_wait_time,
        'bottlenecks': bottlenecks,
        'common_paths': common_paths[:10]  # Top 10 most common paths
    }

def extract_current_students_from_history(processed_data, cutoff_date=None):
    """
    Extract students currently in the training pipeline from historical data
    
    Args:
        processed_data (pd.DataFrame): Processed historical training data
        cutoff_date (datetime, optional): Date to use as "now" for determining current students
        
    Returns:
        list: List of student objects with their current state
    """
    # Import here to avoid circular imports
    from simulation_engine import Student
    
    if cutoff_date is None:
        # Use the latest date in the data as the cutoff if not specified
        cutoff_date = processed_data['Cls End Date'].max()
    
    # Group by student
    student_groups = processed_data.groupby('SSN')
    
    current_students = []
    student_id = 0
    
    for ssn, student_data in student_groups:
        # Sort courses by date
        student_courses = student_data.sort_values('Cls Start Date')
        
        # Skip students who have fully completed training
        if student_courses['Graduated'].all():
            continue
            
        # Find the student's current status
        completed_courses = []
        current_course = None
        current_class_end = None
        waiting_for = None
        
        for _, course_data in student_courses.iterrows():
            course_title = course_data['Course Title']
            start_date = course_data['Cls Start Date']
            end_date = course_data['Cls End Date']
            
            # If this course ended before the cutoff, mark as completed if graduated
            if end_date < cutoff_date and course_data['Graduated']:
                completed_courses.append(course_title)
            
            # If this course spans the cutoff date, it's the current course
            elif start_date <= cutoff_date and end_date >= cutoff_date:
                current_course = course_title
                current_class_end = end_date
                
            # If this course starts after cutoff, it's a future course
            elif start_date > cutoff_date:
                # Student might be waiting for this course
                if not current_course and not waiting_for:
                    waiting_for = course_title
        
        # If student has completed courses or is in a course, add to current students
        if completed_courses or current_course or waiting_for:
            # Extract student details
            training_mos = student_data['TrainingMOS'].iloc[0] if 'TrainingMOS' in student_data.columns else None
            personnel_type = student_data['PersonnelType'].iloc[0] if 'PersonnelType' in student_data.columns else 'Enlisted'
            group_type = student_data['GroupType'].iloc[0] if 'GroupType' in student_data.columns else 'ADE'
            
            # Get earliest arrival date
            arrival_date = student_data['Arrival Date'].min() if 'Arrival Date' in student_data.columns else student_courses['Cls Start Date'].min()
            
            # Create student object with current state
            student = Student(
                id=student_id,
                entry_time=student_courses['Cls Start Date'].min(),
                arrival_date=arrival_date,
                personnel_type=personnel_type,
                group_type=group_type,
                training_mos=training_mos
            )
            
            # Set student's state
            student.completed_courses = completed_courses
            
            if current_course:
                student.current_course = current_course
                student.current_class_end = current_class_end
                student.record_status_change('in_class', cutoff_date, current_course)
            elif waiting_for:
                student.start_waiting(waiting_for, cutoff_date)
            
            current_students.append(student)
            student_id += 1
    
    return current_students

def analyze_career_paths(processed_data, min_students=3):
    """
    Analyze historical data to identify typical career paths
    Works with or without MOS information in the dataset.
    
    Args:
        processed_data (pd.DataFrame): Processed training data
        min_students (int): Minimum number of students required to consider a path valid
        
    Returns:
        dict: Dictionary with career paths and flexible courses
    """
    import copy
    
    # Check if MOS information is available
    has_mos_data = ('TrainingMOS' in processed_data.columns and 
                   not processed_data['TrainingMOS'].isna().all() and
                   processed_data['TrainingMOS'].notna().sum() > min_students)
    
    print(f"MOS data available: {has_mos_data}")
    
    # Group data by student
    student_groups = processed_data.groupby('SSN')
    total_students = len(student_groups)
    
    print(f"Analyzing {total_students} student records...")
    
    # Track data by MOS and general
    all_mos_paths = defaultdict(list)
    all_courses = set()
    course_frequencies = defaultdict(lambda: defaultdict(int))
    course_positions = defaultdict(lambda: defaultdict(list))
    student_counts = defaultdict(int)
    
    # Add a general category
    student_counts["General"] = 0
    
    # Process each student
    for i, (ssn, student_data) in enumerate(student_groups):
        if i % 1000 == 0:
            print(f"Processing student {i}/{total_students}...")
            
        # Check if the student has graduation data
        has_graduation_data = False
        if 'Graduated' in student_data.columns:
            has_graduation_data = any(student_data['Graduated'])
        elif 'Out Stat' in student_data.columns:
            has_graduation_data = any(student_data['Out Stat'] == 'G')
        
        # Skip students with no graduation records
        if not has_graduation_data and ('Graduated' in student_data.columns or 'Out Stat' in student_data.columns):
            continue
            
        # Get completed courses in chronological order
        try:
            if 'Graduated' in student_data.columns:
                completed_courses = student_data[student_data['Graduated']].sort_values('Cls Start Date')
            elif 'Out Stat' in student_data.columns:
                completed_courses = student_data[student_data['Out Stat'] == 'G'].sort_values('Cls Start Date')
            else:
                # No graduation info, use all courses
                completed_courses = student_data.sort_values('Cls Start Date')
            
            # Skip if too few courses
            if len(completed_courses) < 2:
                continue
                
            # Extract the sequence of courses
            course_sequence = completed_courses['Course Title'].tolist()
            
            # Skip if no courses found
            if not course_sequence:
                continue
            
            # Add to the all courses set
            all_courses.update(course_sequence)
            
            # Get the student's MOS if available
            if has_mos_data and 'TrainingMOS' in student_data.columns:
                mos = student_data['TrainingMOS'].iloc[0]
                if pd.notna(mos) and mos in ['18A', '18B', '18C', '18D', '18E']:
                    # Add to MOS-specific paths
                    all_mos_paths[mos].append(course_sequence)
                    student_counts[mos] += 1
                    
                    # Track course frequencies and positions for this MOS
                    for j, course in enumerate(course_sequence):
                        course_frequencies[mos][course] += 1
                        course_positions[mos][course].append(j)
            
            # Add to general paths regardless of MOS
            all_mos_paths["General"].append(course_sequence)
            student_counts["General"] += 1
            
            # Track course frequencies and positions for general
            for j, course in enumerate(course_sequence):
                course_frequencies["General"][course] += 1
                course_positions["General"][course].append(j)
                
        except Exception as e:
            print(f"Error processing student {ssn}: {e}")
            continue
    
    # Process data into career paths
    career_paths = {}
    
    # Process each MOS (and General)
    for mos, paths in all_mos_paths.items():
        # Skip if too few students for this MOS/track
        if len(paths) < min_students:
            print(f"Skipping {mos}: only {len(paths)} students (minimum {min_students})")
            continue
            
        print(f"Processing {mos}: {len(paths)} students")
            
        # Calculate percentage of students who took each course
        course_percentages = {}
        for course, count in course_frequencies[mos].items():
            course_percentages[course] = count / len(paths)
        
        # Calculate average position for each course
        avg_positions = {}
        position_variability = {}
        
        for course, positions in course_positions[mos].items():
            avg_positions[course] = sum(positions) / len(positions)
            # Calculate standard deviation of positions
            std_dev = np.std(positions) if len(positions) > 1 else 0
            position_variability[course] = std_dev
        
        # Set thresholds based on data availability
        if len(paths) >= 50:
            # With lots of data, we can be more selective
            core_threshold = 0.7 if mos != "General" else 0.4
            variability_threshold = 1.5
            flexible_threshold = 0.3 if mos != "General" else 0.2
        else:
            # With limited data, use more relaxed thresholds
            core_threshold = 0.6 if mos != "General" else 0.3
            variability_threshold = 2.0
            flexible_threshold = 0.2 if mos != "General" else 0.15
        
        # Identify core courses (taken by most students in consistent order)
        core_courses = []
        flexible_courses = []
        
        for course, percentage in course_percentages.items():
            # Core courses are taken by many students and have low position variability
            if percentage >= core_threshold and position_variability[course] < variability_threshold:
                core_courses.append((course, avg_positions[course]))
            # Flexible courses are taken by some students but have higher position variability
            elif percentage >= flexible_threshold:
                flexible_courses.append(course)
        
        # Sort core courses by average position to create the typical path
        core_courses.sort(key=lambda x: x[1])
        typical_path = [course for course, _ in core_courses]
        
        # Find most common first courses
        first_courses = defaultdict(int)
        for path in paths:
            if path:
                first_courses[path[0]] += 1
        
        # Find most common last courses
        last_courses = defaultdict(int)
        for path in paths:
            if path:
                last_courses[path[-1]] += 1
        
        # Make sure the most common first course is at the beginning
        most_common_first = max(first_courses.items(), key=lambda x: x[1])[0] if first_courses else None
        if most_common_first and most_common_first in typical_path and typical_path[0] != most_common_first:
            typical_path.remove(most_common_first)
            typical_path.insert(0, most_common_first)
        
        # Make sure the most common last course is at the end
        most_common_last = max(last_courses.items(), key=lambda x: x[1])[0] if last_courses else None
        if most_common_last and most_common_last in typical_path and typical_path[-1] != most_common_last:
            typical_path.remove(most_common_last)
            typical_path.append(most_common_last)
        
        # Store the results
        career_paths[mos] = {
            'typical_path': typical_path,
            'flexible_courses': flexible_courses,
            'course_percentages': course_percentages,
            'student_count': len(paths),
            'avg_positions': avg_positions,
            'position_variability': position_variability,
            'all_paths': paths
        }
    
    # If we have enough data but no MOS-specific paths, try to identify tracks
    if not has_mos_data and "General" in career_paths and career_paths["General"]["student_count"] >= 50:
        try:
            print("Attempting to identify distinct tracks from general data...")
            
            # Try clustering to find distinct tracks
            general_paths = career_paths["General"]["all_paths"]
            all_courses_list = list(all_courses)
            
            # Create a matrix of course presence (1 if student took course, 0 otherwise)
            course_matrix = np.zeros((len(general_paths), len(all_courses_list)))
            for i, path in enumerate(general_paths):
                for course in path:
                    if course in all_courses_list:
                        j = all_courses_list.index(course)
                        course_matrix[i, j] = 1
            
            # If there are many courses, try to reduce dimensionality
            if len(all_courses_list) > 100:
                # Filter to courses taken by at least 5% of students
                course_counts = np.sum(course_matrix, axis=0)
                frequent_courses = course_counts >= len(general_paths) * 0.05
                course_matrix = course_matrix[:, frequent_courses]
                filtered_courses = [all_courses_list[i] for i in range(len(all_courses_list)) if frequent_courses[i]]
                all_courses_list = filtered_courses
            
            # Import here to avoid dependency if not needed
            from sklearn.cluster import KMeans
            
            # Try different numbers of clusters
            best_silhouette = -1
            best_clusters = None
            best_n_clusters = 0
            
            try:
                from sklearn.metrics import silhouette_score
                
                for n_clusters in range(2, min(6, len(general_paths) // min_students)):
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    cluster_labels = kmeans.fit_predict(course_matrix)
                    
                    # Calculate silhouette score
                    silhouette_avg = silhouette_score(course_matrix, cluster_labels)
                    print(f"For n_clusters = {n_clusters}, silhouette score is {silhouette_avg}")
                    
                    if silhouette_avg > best_silhouette:
                        best_silhouette = silhouette_avg
                        best_clusters = cluster_labels
                        best_n_clusters = n_clusters
            except Exception as e:
                print(f"Error in silhouette analysis: {e}")
                # Fallback to 3 clusters
                kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
                best_clusters = kmeans.fit_predict(course_matrix)
                best_n_clusters = 3
            
            # If we found reasonable clusters
            if best_clusters is not None:
                print(f"Found {best_n_clusters} distinct tracks in the data")
                
                # Process each cluster into a track
                for cluster_id in range(best_n_clusters):
                    # Get paths in this cluster
                    cluster_paths = [general_paths[i] for i in range(len(general_paths)) if best_clusters[i] == cluster_id]
                    
                    # Skip if too few paths
                    if len(cluster_paths) < min_students:
                        continue
                    
                    # Calculate frequencies and positions for this cluster
                    cluster_frequencies = defaultdict(int)
                    cluster_positions = defaultdict(list)
                    
                    for path in cluster_paths:
                        for i, course in enumerate(path):
                            cluster_frequencies[course] += 1
                            cluster_positions[course].append(i)
                    
                    # Calculate course percentages
                    cluster_percentages = {course: count/len(cluster_paths) 
                                          for course, count in cluster_frequencies.items()}
                    
                    # Calculate average positions
                    cluster_avg_positions = {course: sum(positions)/len(positions) 
                                           for course, positions in cluster_positions.items()}
                    
                    # Calculate position variability
                    cluster_variability = {course: np.std(positions) if len(positions) > 1 else 0 
                                         for course, positions in cluster_positions.items()}
                    
                    # Identify core courses for this cluster (more selective criteria)
                    cluster_core = []
                    for course, percentage in cluster_percentages.items():
                        if percentage >= 0.6 and cluster_variability[course] < 1.5:
                            cluster_core.append((course, cluster_avg_positions[course]))
                    
                    # Sort by position
                    cluster_core.sort(key=lambda x: x[1])
                    cluster_path = [course for course, _ in cluster_core]
                    
                    # Only add if we found a meaningful path
                    if len(cluster_path) >= 3:
                        # Find the most distinctive courses for this cluster
                        general_percentages = career_paths["General"]["course_percentages"]
                        
                        distinctive_courses = []
                        for course, percentage in cluster_percentages.items():
                            general_pct = general_percentages.get(course, 0)
                            # Courses much more common in this cluster than overall
                            if percentage > general_pct * 1.5 and percentage >= 0.5:
                                distinctive_courses.append((course, percentage/general_pct))
                        
                        # Sort by distinctiveness
                        distinctive_courses.sort(key=lambda x: x[1], reverse=True)
                        
                        # Create a name for the track
                        track_name = f"Track {cluster_id+1}"
                        if distinctive_courses:
                            # Use most distinctive course for the name
                            track_name = f"Track: {distinctive_courses[0][0]}"
                        
                        # Add to career paths
                        career_paths[track_name] = {
                            'typical_path': cluster_path,
                            'flexible_courses': [c for c, _ in distinctive_courses if c not in cluster_path],
                            'course_percentages': cluster_percentages,
                            'student_count': len(cluster_paths),
                            'avg_positions': cluster_avg_positions,
                            'position_variability': cluster_variability,
                            'all_paths': cluster_paths
                        }
                
        except Exception as e:
            print(f"Error in track identification: {e}")
            import traceback
            print(traceback.format_exc())
    
    return career_paths
    
def apply_career_paths_to_configs(course_configs, edited_paths, clear_existing=True, update_mos_paths=True):
    """
    Apply edited career paths to course configurations by setting prerequisites
    Works with both MOS-specific paths and general/track paths
    
    Args:
        course_configs (dict): Dictionary of course configurations
        edited_paths (dict): Dictionary of edited career paths by MOS or track
        clear_existing (bool): Whether to clear existing prerequisites
        update_mos_paths (bool): Whether to update MOS paths
        
    Returns:
        dict: Updated course configurations
        list: List of changes made
    """
    import copy
    
    # Make a copy to avoid modifying the original
    updated_configs = copy.deepcopy(course_configs)
    changes_made = []
    
    # Process each path
    for path_name, path_data in edited_paths.items():
        typical_path = path_data['typical_path']
        
        # Skip if the path is too short
        if len(typical_path) < 2:
            continue
        
        # Determine if this is a MOS-specific path or a general track
        is_mos_path = path_name in ['18A', '18B', '18C', '18D', '18E', 'General']
        
        # Update each course in the typical path
        for i, course in enumerate(typical_path):
            # Skip if course doesn't exist in configs
            if course not in updated_configs:
                # Create default configuration for this course
                updated_configs[course] = {
                    'prerequisites': {
                        'type': 'AND',
                        'courses': []
                    },
                    'or_prerequisites': [],
                    'mos_paths': {
                        '18A': [],
                        '18B': [],
                        '18C': [],
                        '18D': [],
                        '18E': []
                    },
                    'required_for_all_mos': False,
                    'max_capacity': 50,
                    'classes_per_year': 4,
                    'reserved_seats': {
                        'OF': 0,
                        'ADE': 0,
                        'NG': 0
                    },
                    'officer_enlisted_ratio': None,
                    'use_even_mos_ratio': False
                }
            
            # Clear existing prerequisites if requested
            if clear_existing:
                if 'prerequisites' not in updated_configs[course]:
                    updated_configs[course]['prerequisites'] = {
                        'type': 'AND',
                        'courses': []
                    }
                elif isinstance(updated_configs[course]['prerequisites'], list):
                    updated_configs[course]['prerequisites'] = {
                        'type': 'AND',
                        'courses': []
                    }
                else:
                    updated_configs[course]['prerequisites']['courses'] = []
            
            # Set new prerequisites based on the path
            if i > 0:
                # Previous course in path becomes the prerequisite
                prereq_course = typical_path[i-1]
                
                if clear_existing:
                    # Replace existing prerequisites
                    updated_configs[course]['prerequisites']['courses'] = [prereq_course]
                else:
                    # Add to existing prerequisites without duplicates
                    if isinstance(updated_configs[course]['prerequisites'], list):
                        current_prereqs = updated_configs[course]['prerequisites']
                        updated_configs[course]['prerequisites'] = {
                            'type': 'AND',
                            'courses': current_prereqs
                        }
                    else:
                        current_prereqs = updated_configs[course]['prerequisites'].get('courses', [])
                    
                    if prereq_course not in current_prereqs:
                        current_prereqs.append(prereq_course)
                        updated_configs[course]['prerequisites']['courses'] = current_prereqs
                
                # Record the change
                changes_made.append({
                    'course': course,
                    'path': path_name,
                    'prerequisites': [prereq_course],
                    'action': 'set_prerequisites'
                })
            
            # Update MOS paths if requested and this is a MOS-specific path
            if update_mos_paths and is_mos_path and path_name in ['18A', '18B', '18C', '18D', '18E']:
                if 'mos_paths' not in updated_configs[course]:
                    updated_configs[course]['mos_paths'] = {
                        '18A': [],
                        '18B': [],
                        '18C': [],
                        '18D': [],
                        '18E': []
                    }
                
                # Mark this course as part of this MOS path
                updated_configs[course]['mos_paths'][path_name] = []
                
                # Record the change
                changes_made.append({
                    'course': course,
                    'path': path_name,
                    'action': 'added_to_mos_path'
                })
            
            # If this is the General path and update_mos_paths is enabled,
            # mark the course as required for all MOS paths
            if update_mos_paths and path_name == 'General':
                updated_configs[course]['required_for_all_mos'] = True
                
                # Record the change
                changes_made.append({
                    'course': course,
                    'path': path_name,
                    'action': 'marked_required_for_all'
                })
            
            # For track paths that aren't MOS-specific, add as prerequisites but don't
            # modify MOS paths or required_for_all_mos
            if not is_mos_path:
                # Already handled prerequisites above, no additional action needed
                pass
    
    return updated_configs, changes_made