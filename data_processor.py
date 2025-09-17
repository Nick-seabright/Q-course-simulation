import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict

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
    
    # Convert date columns to datetime
    date_columns = ['Cls Start Date', 'Cls End Date', 'Input Date', 'Output Date']
    
    # Add Arrival Date to date columns if it exists
    if 'Arrival Date' in data.columns:
        date_columns.append('Arrival Date')
    
    for col in date_columns:
        data[col] = pd.to_datetime(data[col], errors='coerce')
    
    # If Arrival Date doesn't exist, estimate it based on Input Date
    if 'Arrival Date' not in data.columns:
        # Estimate arrival as 3 days before class start by default
        data['Arrival Date'] = data['Cls Start Date'] - pd.Timedelta(days=3)
        st.warning("Arrival Date column not found. Estimating arrival as 3 days before class start.")
    
    # Calculate days between arrival and class start
    data['Days Before Class'] = (data['Cls Start Date'] - data['Arrival Date']).dt.days
    
    # Calculate class duration
    data['Duration'] = (data['Cls End Date'] - data['Cls Start Date']).dt.days
    
    # Determine if a student started a class (I=New Input, J=Retrainee In, Q=Recycle In)
    data['Started'] = data['Input Stat'].isin(['I', 'J', 'Q'])
    
    # Determine if a student graduated (G=Graduate)
    data['Graduated'] = data['Out Stat'] == 'G'
    
    # Determine if a student recycled (L=Recycle Out)
    data['Recycled'] = data['Out Stat'] == 'L'
    
    # Determine if a student was a no-show (N=No Show)
    data['NoShow'] = data['Input Stat'] == 'N'
    
    # Determine if a student was retrainee out (K=Retrainee Out)
    data['RetraineeOut'] = data['Out Stat'] == 'K'
    
    # Determine if a reservation was cancelled (C=Cancelled Reservation)
    data['Cancelled'] = data['Res Stat'] == 'C'
    
    # Determine if a student failed or had non-successful completion (Z=Non-Successful Completion)
    data['Failed'] = data['Out Stat'] == 'Z'
    
    # Extract personnel type (Officer vs Enlisted)
    data['PersonnelType'] = data['CP Pers Type'].map({'O': 'Officer', 'E': 'Enlisted'})
    
    # Extract group type (Active Duty, National Guard, etc.)
    data['GroupType'] = data['Group Type']
    
    # Try to extract or infer Training MOS
    # Check if there's an explicit Training MOS column
    mos_column = None
    possible_columns = ['Training MOS', 'MOS', 'TrainingMOS', 'TMOS']
    
    for col in possible_columns:
        if col in data.columns:
            mos_column = col
            break
    
    # If no explicit MOS column, try to infer from course titles or other data
    if not mos_column:
        # Create a new column for inferred MOS
        data['InferredMOS'] = None
        
        # Infer MOS from personnel type and course titles
        for i, row in data.iterrows():
            inferred_mos = None
            
            # Officers are typically 18A
            if row.get('PersonnelType') == 'Officer':
                inferred_mos = '18A'
            else:
                # Try to infer from course title
                course_title = str(row.get('Course Title', '')).upper()
                
                if 'WEAPONS' in course_title or 'WEAP SGT' in course_title:
                    inferred_mos = '18B'
                elif 'ENGINEER' in course_title or 'ENG SGT' in course_title:
                    inferred_mos = '18C'
                elif 'MEDICAL' in course_title or 'MED SGT' in course_title or 'MEDIC' in course_title:
                    inferred_mos = '18D'
                elif 'COMM' in course_title or 'SIGNAL' in course_title or 'COMMO SGT' in course_title:
                    inferred_mos = '18E'
            
            data.at[i, 'InferredMOS'] = inferred_mos
        
        # Use the inferred MOS column
        data['TrainingMOS'] = data['InferredMOS']
    else:
        # Use the existing MOS column but standardize the values
        data['TrainingMOS'] = data[mos_column]
        
        # Map non-standard MOS codes to standard ones
        mos_mapping = {
            'OFFICER': '18A',
            'WEAPONS': '18B',
            'ENGINEER': '18C',
            'MEDICAL': '18D',
            'COMMUNICATIONS': '18E'
        }
        
        for i, row in data.iterrows():
            current_mos = str(row['TrainingMOS']).upper()
            
            # Check if we need to map this MOS
            for key, value in mos_mapping.items():
                if key in current_mos:
                    data.at[i, 'TrainingMOS'] = value
                    break
            
            # If still not a standard MOS, make a best guess
            if data.at[i, 'TrainingMOS'] not in ['18A', '18B', '18C', '18D', '18E']:
                if row.get('PersonnelType') == 'Officer':
                    data.at[i, 'TrainingMOS'] = '18A'
                else:
                    # Default to a random enlisted MOS if we can't determine
                    data.at[i, 'TrainingMOS'] = np.random.choice(['18B', '18C', '18D', '18E'])
    
    return data

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
        
        if not mos_column:
            # If no explicit MOS column, try to infer from other data
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