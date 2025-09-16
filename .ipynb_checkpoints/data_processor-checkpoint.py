import pandas as pd
import numpy as np
from datetime import datetime

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
    for col in date_columns:
        data[col] = pd.to_datetime(data[col], errors='coerce')
    
    # Calculate class duration
    data['Duration'] = (data['Cls End Date'] - data['Cls Start Date']).dt.days
    
    # Determine if a student started a class
    data['Started'] = data['Input Stat'].isin(['I', 'J', 'Q'])
    
    # Determine if a student graduated
    data['Graduated'] = data['Out Stat'] == 'G'
    
    # Determine if a student recycled
    data['Recycled'] = data['Out Stat'] == 'L'
    
    # Determine if a student was a no-show
    data['NoShow'] = data['Input Stat'] == 'N'
    
    # Determine if a reservation was cancelled
    data['Cancelled'] = data['Input Stat'] == 'C'
    
    # Extract personnel type (Officer vs Enlisted)
    data['PersonnelType'] = data['CP Pers Type'].map({'O': 'Officer', 'E': 'Enlisted'})
    
    # Extract group type (Active Duty, National Guard, etc.)
    data['GroupType'] = data['Group Type']
    
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
            
        # Calculate pass rate
        if group['Started'].sum() > 0:
            pass_rate = group['Graduated'].sum() / group['Started'].sum()
        else:
            pass_rate = 0
            
        # Calculate recycle rate
        if group['Started'].sum() > 0:
            recycle_rate = group['Recycled'].sum() / group['Started'].sum()
        else:
            recycle_rate = 0
            
        # Calculate no-show rate
        reservation_count = len(group)
        if reservation_count > 0:
            no_show_rate = group['NoShow'].sum() / reservation_count
        else:
            no_show_rate = 0
            
        # Calculate cancellation rate
        if reservation_count > 0:
            cancellation_rate = group['Cancelled'].sum() / reservation_count
        else:
            cancellation_rate = 0
            
        # Calculate average class size
        classes = group.groupby(['CLS', 'Cls Start Date'])
        class_sizes = classes.size().values
        if len(class_sizes) > 0:
            avg_class_size = int(np.mean(class_sizes))  # Convert to integer
        else:
            avg_class_size = 0
            
        # Calculate average duration
        avg_duration = group['Duration'].mean()
        
        # Estimate classes per year based on historical data
        # Group by fiscal year and count unique class identifiers
        group['FY'] = group['FY'].astype(str)
        fy_classes = group.groupby('FY').apply(lambda x: x[['CLS']].drop_duplicates().shape[0])
        if len(fy_classes) > 0:
            classes_per_year = fy_classes.mean()
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
        group_composition = {group_type: count / len(group) for group_type, count in group_counts.items()}
        
        # Store all statistics
        course_stats[course] = {
            'pass_rate': pass_rate,
            'recycle_rate': recycle_rate,
            'no_show_rate': no_show_rate,
            'cancellation_rate': cancellation_rate,
            'avg_class_size': avg_class_size,
            'avg_duration': avg_duration,
            'classes_per_year': classes_per_year,
            'officer_ratio': officer_ratio,
            'enlisted_ratio': enlisted_ratio,
            'group_composition': group_composition
        }
    
    return course_stats

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