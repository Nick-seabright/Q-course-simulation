import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict
import streamlit as st
from typing import Dict, List, Any, Tuple, Optional, Union

"""
Data Processing Module for Training Schedule Optimizer

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

def safe_st_warning(message: str) -> None:
    """
    Safe wrapper for st.warning that handles case where streamlit isn't available
    
    Args:
        message: Warning message to display
    """
    try:
        st.warning(message)
    except:
        print(f"WARNING: {message}")

def safe_st_error(message: str) -> None:
    """
    Safe wrapper for st.error that handles case where streamlit isn't available
    
    Args:
        message: Error message to display
    """
    try:
        st.error(message)
    except:
        print(f"ERROR: {message}")

@st.cache_data(ttl=3600)
def process_data(raw_data: pd.DataFrame) -> pd.DataFrame:
    """
    Process raw training data to extract relevant information
    
    Args:
        raw_data: Raw training data
        
    Returns:
        Processed data with additional derived fields
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
    
    # Track date conversion statistics
    date_stats = {}
    
    for col in date_columns:
        if col in data.columns:
            total_rows = len(data)
            # Store original values before conversion
            original_values = data[col].copy()
            
            # Try to convert to datetime with error handling
            try:
                data[col] = pd.to_datetime(data[col], errors='coerce')
                
                # Calculate conversion stats
                null_after = data[col].isnull().sum()
                null_before = original_values.isnull().sum()
                failed_conversions = null_after - null_before
                
                # Log statistics
                date_stats[col] = {
                    'total': total_rows,
                    'null_before': null_before,
                    'null_after': null_after,
                    'failed_conversions': failed_conversions,
                    'success_rate': (total_rows - null_after) / total_rows if total_rows > 0 else 0
                }
                
                # Warn about failed conversions
                if failed_conversions > 0:
                    safe_st_warning(f"Could not convert {failed_conversions} values in {col} to dates.")
            except Exception as e:
                safe_st_error(f"Error converting {col} to datetime: {str(e)}")
    
    # Group by student (SSN) to identify each student's first class
    if 'SSN' in data.columns:
        student_groups = data.groupby('SSN')
        
        # If Arrival Date doesn't exist or has missing values, add/update it
        if 'Arrival Date' not in data.columns:
            # Create the column
            data['Arrival Date'] = pd.NaT
            safe_st_warning("Arrival Date column not found. Estimating arrival as 3 weeks before each student's first class.")
        
        # Track number of students processed
        students_processed = 0
        students_with_missing_arrival = 0
        
        # Process each student
        for ssn, student_data in student_groups:
            students_processed += 1
            
            # Sort by class start date to find first class
            if 'Cls Start Date' in student_data.columns:
                sorted_classes = student_data.sort_values('Cls Start Date')
                
                if len(sorted_classes) > 0:
                    first_class_start = sorted_classes.iloc[0]['Cls Start Date']
                    
                    # Check each row for this student
                    for idx in sorted_classes.index:
                        # If arrival date is missing or invalid (after class start)
                        if pd.isna(data.loc[idx, 'Arrival Date']) or data.loc[idx, 'Arrival Date'] > data.loc[idx, 'Cls Start Date']:
                            students_with_missing_arrival += 1
                            
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
    
    # Validate durations and fix any negative values
    invalid_durations = data[data['Duration'] < 0]
    if len(invalid_durations) > 0:
        safe_st_warning(f"Found {len(invalid_durations)} records with negative durations. Setting these to 1 day.")
        data.loc[invalid_durations.index, 'Duration'] = 1
    
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

def process_training_mos(data: pd.DataFrame) -> None:
    """
    Process Training MOS data, inferring it if not explicitly provided
    
    Args:
        data: Training data to process
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
        
        # Create mapping dictionary for more maintainable code
        mos_keywords = {
            '18B': ['WEAPONS', 'WEAP SGT', 'WEAPON'],
            '18C': ['ENGINEER', 'ENG SGT', 'ENGINEERING'],
            '18D': ['MEDICAL', 'MED SGT', 'MEDIC', 'MEDICINE'],
            '18E': ['COMM', 'SIGNAL', 'COMMO SGT', 'COMMUNICATION']
        }
        
        # Try to infer enlisted MOS from course titles if available
        if 'Course Title' in data.columns:
            # Track inference statistics
            inference_counts = defaultdict(int)
            
            for idx, row in data.iterrows():
                if pd.isna(data.loc[idx, 'TrainingMOS']) and data.loc[idx, 'PersonnelType'] == 'Enlisted':
                    course_title = str(row.get('Course Title', '')).upper()
                    
                    # Use the mapping dictionary for inference
                    for mos, keywords in mos_keywords.items():
                        if any(keyword in course_title for keyword in keywords):
                            data.loc[idx, 'TrainingMOS'] = mos
                            inference_counts[mos] += 1
                            break

@st.cache_data(ttl=3600)
def analyze_historical_data(processed_data: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Analyze historical training data to extract statistics by course
    
    Args:
        processed_data: Processed training data
        
    Returns:
        Dictionary of course statistics
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
            max_class_size = int(np.max(class_sizes))
            min_class_size = int(np.min(class_sizes))
        else:
            avg_class_size = 0
            max_class_size = 0
            min_class_size = 0
        
        # Calculate average duration
        durations = group['Duration'].dropna()
        
        if len(durations) > 0:
            avg_duration = np.mean(durations)
            max_duration = np.max(durations)
            min_duration = np.min(durations)
        else:
            avg_duration = 0
            max_duration = 0
            min_duration = 0
        
        # Estimate classes per year based on historical data
        # Group by fiscal year and count unique class identifiers
        group['FY'] = group['FY'].astype(str)
        fy_classes = group.groupby('FY').apply(lambda x: x[['CLS']].drop_duplicates().shape[0])
        
        if len(fy_classes) > 0:
            classes_per_year = int(fy_classes.mean())
            max_classes_per_year = int(fy_classes.max())
        else:
            classes_per_year = 0
            max_classes_per_year = 0
        
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
        
        # Calculate confidence intervals for metrics
        confidence_intervals = {}
        
        # Calculate 95% confidence interval for pass rate using bootstrap
        if group['Started'].sum() > 10:  # Only calculate if we have enough data
            try:
                bootstrap_samples = 1000
                pass_samples = []
                
                for _ in range(bootstrap_samples):
                    sample = group.sample(n=len(group), replace=True)
                    if sample['Started'].sum() > 0:
                        sample_pass_rate = sample['Graduated'].sum() / sample['Started'].sum()
                        pass_samples.append(sample_pass_rate)
                
                pass_samples = np.array(pass_samples)
                confidence_intervals['pass_rate'] = {
                    'lower': np.percentile(pass_samples, 2.5),
                    'upper': np.percentile(pass_samples, 97.5)
                }
            except Exception as e:
                confidence_intervals['pass_rate'] = {'error': str(e)}
        
        # Store all statistics
        course_stats[course] = {
            'pass_rate': pass_rate,
            'recycle_rate': recycle_rate,
            'no_show_rate': no_show_rate,
            'cancellation_rate': cancellation_rate,
            'failure_rate': failure_rate,
            'avg_class_size': avg_class_size,
            'max_class_size': max_class_size,
            'min_class_size': min_class_size,
            'avg_duration': avg_duration,
            'max_duration': max_duration,
            'min_duration': min_duration,
            'classes_per_year': classes_per_year,
            'max_classes_per_year': max_classes_per_year,
            'officer_ratio': officer_ratio,
            'enlisted_ratio': enlisted_ratio,
            'group_composition': group_composition,
            'mos_composition': mos_composition,
            'confidence_intervals': confidence_intervals,
            'sample_size': len(group)
        }
    
    return course_stats

@st.cache_data(ttl=3600)
def extract_historical_arrival_patterns(processed_data: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """
    Extract historical student arrival patterns from processed data
    
    Args:
        processed_data: Processed training data
        
    Returns:
        Dictionary of arrival patterns or None if insufficient data
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
    median_days_before = valid_arrivals['days_before'].median()
    std_days_before = valid_arrivals['days_before'].std()
    
    # Extract monthly distribution of arrivals
    valid_arrivals['arrival_month'] = valid_arrivals['Arrival Date'].dt.month_name()
    monthly_counts = valid_arrivals['arrival_month'].value_counts()
    total_students = monthly_counts.sum()
    
    monthly_distribution = {}
    for month, count in monthly_counts.items():
        monthly_distribution[month] = count / total_students if total_students > 0 else 0
    
    # Add year-over-year analysis if data spans multiple years
    years_analysis = None
    
    if 'Arrival Date' in valid_arrivals.columns:
        valid_arrivals['arrival_year'] = valid_arrivals['Arrival Date'].dt.year
        years = valid_arrivals['arrival_year'].unique()
        
        if len(years) > 1:
            years_analysis = {}
            
            for year in years:
                year_data = valid_arrivals[valid_arrivals['arrival_year'] == year]
                
                if len(year_data) > 0:
                    year_monthly_counts = year_data['arrival_month'].value_counts()
                    year_total = year_monthly_counts.sum()
                    
                    year_distribution = {}
                    for month, count in year_monthly_counts.items():
                        year_distribution[month] = count / year_total if year_total > 0 else 0
                    
                    years_analysis[int(year)] = {
                        'monthly_distribution': year_distribution,
                        'avg_days_before': year_data['days_before'].mean(),
                        'total_arrivals': len(year_data)
                    }
    
    return {
        'avg_days_before': avg_days_before,
        'median_days_before': median_days_before,
        'std_days_before': std_days_before,
        'monthly_distribution': monthly_distribution,
        'years_analysis': years_analysis,
        'sample_size': len(valid_arrivals)
    }

@st.cache_data(ttl=3600)
def extract_historical_mos_distribution(processed_data: pd.DataFrame) -> Dict[str, float]:
    """
    Extract historical MOS distribution from processed data
    
    Args:
        processed_data: Processed training data
        
    Returns:
        Dictionary of MOS distribution
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
                    # Use a deterministic assignment based on string hash
                    # to ensure consistent mapping instead of random choice
                    hash_value = hash(str(mos)) % 4
                    standard_mos = ['18B', '18C', '18D', '18E'][hash_value]
            else:
                # If no officer/enlisted info, use hash for deterministic mapping
                hash_value = hash(str(mos)) % 5
                standard_mos = ['18A', '18B', '18C', '18D', '18E'][hash_value]
        
        # Add to distribution
        if standard_mos in mos_distribution:
            mos_distribution[standard_mos] += count / total_students if total_students > 0 else 0
        else:
            mos_distribution[standard_mos] = count / total_students if total_students > 0 else 0
    
    # Ensure all standard MOS codes are present
    for mos in ['18A', '18B', '18C', '18D', '18E']:
        if mos not in mos_distribution:
            mos_distribution[mos] = 0
    
    # Add metadata about the distribution
    mos_distribution['_metadata'] = {
        'source_column': mos_column,
        'total_students': int(total_students),
        'original_codes': len(mos_counts),
        'mapped_codes': len(set(mos_distribution.keys()) - {'_metadata'})
    }
    
    return mos_distribution

@st.cache_data(ttl=7200)
def infer_prerequisites(processed_data: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Infer potential prerequisites based on historical student progression
    
    Args:
        processed_data: Processed training data
        
    Returns:
        Dictionary of likely prerequisites for each course
    """
    # Group data by student (SSN)
    student_groups = processed_data.groupby('SSN')
    
    # Track course sequences
    course_sequences = {}
    student_count = 0
    
    for ssn, student_data in student_groups:
        student_count += 1
        
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
                'or_prerequisites': potential_or_groups,
                'confidence': min(1.0, total_students / 10),  # Confidence based on sample size
                'sample_size': total_students
            }
            prerequisites[course] = prereq_structure
    
    return prerequisites

@st.cache_data(ttl=7200)
def infer_mos_paths(processed_data: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Infer MOS-specific training paths from historical data
    
    Args:
        processed_data: Processed training data
        
    Returns:
        Dictionary of course to MOS path mappings
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
        
        # Calculate MOS composition stats
        mos_stats = {}
        for mos in mos_codes:
            # Get percentage of this MOS in the course
            mos_data = course_data[course_data['TrainingMOS'] == mos]
            
            if not mos_data.empty:
                percentage = mos_data.iloc[0]['percentage']
                count = mos_data.iloc[0]['count']
                
                mos_stats[mos] = {
                    'percentage': percentage,
                    'count': count,
                    'included': percentage >= 0.1  # If more than 10% of students with this MOS take this course
                }
            else:
                mos_stats[mos] = {
                    'percentage': 0,
                    'count': 0,
                    'included': False
                }
        
        # Determine if the course is required for all MOS
        all_mos_present = all(mos_stats[mos]['included'] for mos in mos_codes)
        significant_representation = sum(mos_stats[mos]['count'] for mos in mos_codes) >= 10
        
        mos_paths[course] = {
            'mos_paths': {
                mos: [] for mos in mos_codes if mos_stats[mos]['included']
            },
            'mos_stats': mos_stats,
            'required_for_all_mos': all_mos_present and significant_representation,
            'sample_size': course_totals[course]
        }
    
    # Find patterns in progression by MOS
    if 'SSN' in processed_data.columns:
        # Group by student
        student_groups = processed_data.groupby('SSN')
        
        # Track course sequences by MOS
        mos_sequences = {mos: [] for mos in mos_codes}
        
        for ssn, student_data in student_groups:
            # Get student's MOS
            student_mos = student_data['TrainingMOS'].iloc[0] if not student_data['TrainingMOS'].isnull().all() else None
            
            if student_mos in mos_codes:
                # Get graduated courses in sequence
                completed_courses = student_data[student_data['Graduated']].sort_values('Cls Start Date')
                
                if len(completed_courses) >= 2:
                    course_sequence = completed_courses['Course Title'].tolist()
                    mos_sequences[student_mos].append(course_sequence)
        
        # Analyze sequences to enhance MOS path information
        for mos, sequences in mos_sequences.items():
            if sequences:
                # Count how often courses appear in sequence
                course_positions = {}
                
                for sequence in sequences:
                    for pos, course in enumerate(sequence):
                        if course not in course_positions:
                            course_positions[course] = []
                        course_positions[course].append(pos)
                
                # Calculate average position for each course
                avg_positions = {}
                for course, positions in course_positions.items():
                    if positions:
                        avg_positions[course] = sum(positions) / len(positions)
                
                # Add position information to the MOS paths
                for course in avg_positions:
                    if course in mos_paths:
                        mos_path_info = mos_paths[course]
                        if mos in mos_path_info['mos_paths']:
                            mos_path_info['mos_stats'][mos]['avg_position'] = avg_positions[course]
                            mos_path_info['mos_stats'][mos]['frequency'] = len(course_positions[course]) / len(sequences)
    
    return mos_paths

@st.cache_data(ttl=3600)
def analyze_student_progression(processed_data: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze student progression through courses over time
    
    Args:
        processed_data: Processed training data
        
    Returns:
        Dictionary of progression metrics
    """
    # Group data by student (SSN)
    student_groups = processed_data.groupby('SSN')
    
    # Track metrics
    completion_times = []
    wait_times = []
    progression_paths = []
    mos_completion_times = defaultdict(list)
    
    for ssn, student_data in student_groups:
        # Sort by class start date
        student_courses = student_data.sort_values('Cls Start Date')
        
        # Only consider students who completed at least one course
        graduated_courses = student_courses[student_courses['Graduated']]
        
        if len(graduated_courses) == 0:
            continue
        
        # Get student's MOS if available
        student_mos = None
        if 'TrainingMOS' in student_courses.columns:
            mos_values = student_courses['TrainingMOS'].dropna()
            if not mos_values.empty:
                student_mos = mos_values.iloc[0]
        
        # Calculate completion time (from first class start to last class end)
        if len(graduated_courses) > 1:
            first_start = graduated_courses['Cls Start Date'].min()
            last_end = graduated_courses['Cls End Date'].max()
            completion_time = (last_end - first_start).days
            
            completion_times.append(completion_time)
            
            # Track completion time by MOS
            if student_mos:
                mos_completion_times[student_mos].append(completion_time)
        
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
                        'wait_time': wait_time,
                        'student_mos': student_mos
                    })
        
        # Record progression path
        if len(graduated_courses) > 0:
            course_sequence = graduated_courses['Course Title'].tolist()
            
            progression_paths.append({
                'student': ssn,
                'sequence': course_sequence,
                'mos': student_mos
            })
    
    # Calculate average completion time
    avg_completion_time = np.mean(completion_times) if completion_times else 0
    median_completion_time = np.median(completion_times) if completion_times else 0
    
    # Calculate average wait time
    avg_wait_time = np.mean([w['wait_time'] for w in wait_times]) if wait_times else 0
    median_wait_time = np.median([w['wait_time'] for w in wait_times]) if wait_times else 0
    
    # Calculate MOS-specific completion times
    mos_stats = {}
    for mos, times in mos_completion_times.items():
        if times:
            mos_stats[mos] = {
                'avg_completion_time': np.mean(times),
                'median_completion_time': np.median(times),
                'min_completion_time': np.min(times),
                'max_completion_time': np.max(times),
                'sample_size': len(times)
            }
    
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
            'median_wait_time': np.median(times),
            'max_wait_time': np.max(times),
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
    
    # Find MOS-specific common paths
    mos_paths = {}
    for mos in set(p['mos'] for p in progression_paths if p['mos']):
        mos_path_counts = defaultdict(int)
        
        for path in progression_paths:
            if path['mos'] == mos:
                path_tuple = tuple(path['sequence'])
                mos_path_counts[path_tuple] += 1
        
        mos_common_paths = [
            {
                'sequence': list(path),
                'count': count
            }
            for path, count in mos_path_counts.items()
        ]
        
        # Sort by frequency
        mos_common_paths.sort(key=lambda x: x['count'], reverse=True)
        
        if mos_common_paths:
            mos_paths[mos] = mos_common_paths[:5]  # Top 5 most common paths for this MOS
    
    # Generate timeline data for visualization
    timeline_data = []
    for ssn, student_data in student_groups:
        student_courses = student_data.sort_values('Cls Start Date')
        
        # Get student's MOS if available
        student_mos = None
        if 'TrainingMOS' in student_courses.columns:
            mos_values = student_courses['TrainingMOS'].dropna()
            if not mos_values.empty:
                student_mos = mos_values.iloc[0]
        
        for _, course_data in student_courses.iterrows():
            if pd.notna(course_data['Cls Start Date']) and pd.notna(course_data['Cls End Date']):
                timeline_data.append({
                    'student': ssn,
                    'course': course_data['Course Title'],
                    'start': course_data['Cls Start Date'],
                    'end': course_data['Cls End Date'],
                    'status': 'graduated' if course_data['Graduated'] else 'failed',
                    'mos': student_mos
                })
    
    return {
        'avg_completion_time': avg_completion_time,
        'median_completion_time': median_completion_time,
        'avg_wait_time': avg_wait_time,
        'median_wait_time': median_wait_time,
        'bottlenecks': bottlenecks,
        'common_paths': common_paths[:10],  # Top 10 most common paths
        'mos_stats': mos_stats,
        'mos_common_paths': mos_paths,
        'timeline_data': timeline_data,
        'sample_size': len(student_groups)
    }

def extract_current_students_from_history(processed_data: pd.DataFrame, cutoff_date=None) -> List[Any]:
    """
    Extract students currently in the training pipeline from historical data
    
    Args:
        processed_data: Processed historical training data
        cutoff_date: Date to use as "now" for determining current students
        
    Returns:
        List of student objects with their current state
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
    
    # Track statistics
    stats = {
        'total_students': 0,
        'students_in_course': 0,
        'students_waiting': 0,
        'students_completed': 0,
        'students_by_mos': defaultdict(int)
    }
    
    for ssn, student_data in student_groups:
        stats['total_students'] += 1
        
        # Sort courses by date
        student_courses = student_data.sort_values('Cls Start Date')
        
        # Skip students who have fully completed training
        if student_courses['Graduated'].all():
            stats['students_completed'] += 1
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
                stats['students_in_course'] += 1
            
            # If this course starts after cutoff, it's a future course
            elif start_date > cutoff_date:
                # Student might be waiting for this course
                if not current_course and not waiting_for:
                    waiting_for = course_title
                    stats['students_waiting'] += 1
        
        # If student has completed courses or is in a course, add to current students
        if completed_courses or current_course or waiting_for:
            # Extract student details
            training_mos = student_data['TrainingMOS'].iloc[0] if 'TrainingMOS' in student_data.columns else None
            personnel_type = student_data['PersonnelType'].iloc[0] if 'PersonnelType' in student_data.columns else 'Enlisted'
            group_type = student_data['GroupType'].iloc[0] if 'GroupType' in student_data.columns else 'ADE'
            
            # Track MOS statistics
            if training_mos:
                stats['students_by_mos'][training_mos] += 1
            
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
    
    # Add statistics to the first student's metadata if any students exist
    if current_students:
        current_students[0].extraction_stats = stats
    
    return current_students

def calculate_data_quality_metrics(data: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate data quality metrics for the dataset
    
    Args:
        data: Raw or processed data
        
    Returns:
        Dictionary of data quality metrics
    """
    total_rows = len(data)
    metrics = {
        "total_rows": total_rows,
        "completeness": {},
        "missing_columns": {},
        "duplicates": {},
        "consistency": {}
    }
    
    # Completeness (% of non-null values)
    for col in data.columns:
        non_null = data[col].notna().sum()
        metrics["completeness"][col] = non_null / total_rows if total_rows > 0 else 0
        
        # Track columns with significant missing data
        missing_pct = 1 - metrics["completeness"][col]
        if missing_pct > 0.05:  # More than 5% missing
            metrics["missing_columns"][col] = missing_pct
    
    # Overall completeness score
    metrics["overall_completeness"] = sum(metrics["completeness"].values()) / len(data.columns) if data.columns.size > 0 else 0
    
    # Duplicate detection
    if 'SSN' in data.columns:
        dup_ssn = data['SSN'].duplicated().sum()
        metrics["duplicates"]["SSN"] = dup_ssn
        
        # Analyze if duplicates are valid course progressions or errors
        if dup_ssn > 0:
            dup_students = data[data['SSN'].duplicated(keep=False)]
            valid_progressions = 0
            potential_errors = 0
            
            for ssn, records in dup_students.groupby('SSN'):
                # Check if records have different course titles (valid progression)
                if records['Course Title'].nunique() > 1:
                    valid_progressions += 1
                else:
                    # If same course, check if one is a recycle
                    if 'Out Stat' in records.columns and 'L' in records['Out Stat'].values:
                        valid_progressions += 1
                    else:
                        potential_errors += 1
            
            metrics["duplicates"]["valid_progressions"] = valid_progressions
            metrics["duplicates"]["potential_errors"] = potential_errors
    
    # Consistency checks
    if 'Cls Start Date' in data.columns and 'Cls End Date' in data.columns:
        valid_dates = data.dropna(subset=['Cls Start Date', 'Cls End Date'])
        invalid_duration = (valid_dates['Cls End Date'] < valid_dates['Cls Start Date']).sum()
        metrics["consistency"]["invalid_class_duration"] = invalid_duration
    
    # Arrival date consistency
    if 'Arrival Date' in data.columns and 'Cls Start Date' in data.columns:
        valid_dates = data.dropna(subset=['Arrival Date', 'Cls Start Date'])
        invalid_arrival = (valid_dates['Arrival Date'] > valid_dates['Cls Start Date']).sum()
        metrics["consistency"]["invalid_arrival_dates"] = invalid_arrival
    
    # Status code consistency
    if 'Input Stat' in data.columns and 'Out Stat' in data.columns:
        # Check for records with output but no input
        input_missing = ((data['Input Stat'].isna()) & (data['Out Stat'].notna())).sum()
        metrics["consistency"]["missing_input_with_output"] = input_missing
        
        # Check for graduates with no input
        grad_no_input = ((data['Input Stat'].isna()) & (data['Out Stat'] == 'G')).sum()
        metrics["consistency"]["graduates_without_input"] = grad_no_input
    
    # MOS consistency checks if available
    if 'TrainingMOS' in data.columns:
        # Check for students with different MOS values across records
        if 'SSN' in data.columns:
            mos_consistency = {}
            for ssn, records in data.groupby('SSN'):
                mos_values = records['TrainingMOS'].dropna().unique()
                if len(mos_values) > 1:
                    mos_consistency[ssn] = list(mos_values)
            
            metrics["consistency"]["students_with_multiple_mos"] = len(mos_consistency)
    
    # Invalid MOS values
    if 'TrainingMOS' in data.columns:
        valid_mos = ['18A', '18B', '18C', '18D', '18E']
        invalid_mos = data[~data['TrainingMOS'].isin(valid_mos) & data['TrainingMOS'].notna()].shape[0]
        metrics["consistency"]["invalid_mos_values"] = invalid_mos
    
    # Assign an overall data quality score (0-100)
    quality_factors = [
        metrics["overall_completeness"],
        1 - (sum(metrics["missing_columns"].values()) / len(metrics["missing_columns"]) if metrics["missing_columns"] else 0),
        1 - (metrics["duplicates"].get("potential_errors", 0) / total_rows if total_rows > 0 else 0),
        1 - (sum(metrics["consistency"].values()) / total_rows if total_rows > 0 else 0)
    ]
    
    metrics["data_quality_score"] = int(np.mean([f * 100 for f in quality_factors]))
    
    return metrics