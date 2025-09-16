import pandas as pd
import numpy as np
import datetime

def calculate_fiscal_year(date):
    """
    Calculate the fiscal year for a given date.
    The fiscal year runs from October 1 to September 30.
    
    Args:
        date (datetime): The date to calculate fiscal year for
        
    Returns:
        int: Fiscal year
    """
    if isinstance(date, str):
        date = pd.to_datetime(date)
    
    if date.month >= 10:  # October through December
        return date.year + 1
    else:  # January through September
        return date.year

def parse_ratio(ratio_str):
    """
    Parse a ratio string (e.g., '1:4') into a tuple of integers
    
    Args:
        ratio_str (str): String representation of a ratio
        
    Returns:
        tuple: Tuple of (numerator, denominator)
    """
    try:
        numerator, denominator = ratio_str.split(':')
        return (int(numerator), int(denominator))
    except (ValueError, AttributeError):
        # Default to 1:4 if parsing fails
        return (1, 4)

def format_duration(days):
    """
    Format duration in days to a human-readable string
    
    Args:
        days (int): Duration in days
        
    Returns:
        str: Formatted duration string
    """
    if days < 7:
        return f"{days} days"
    elif days < 30:
        weeks = days // 7
        remaining_days = days % 7
        if remaining_days == 0:
            return f"{weeks} weeks"
        else:
            return f"{weeks} weeks, {remaining_days} days"
    elif days < 365:
        months = days // 30
        remaining_days = days % 30
        if remaining_days == 0:
            return f"{months} months"
        else:
            return f"{months} months, {remaining_days} days"
    else:
        years = days // 365
        remaining_days = days % 365
        if remaining_days == 0:
            return f"{years} years"
        else:
            months = remaining_days // 30
            if months == 0:
                return f"{years} years"
            else:
                return f"{years} years, {months} months"

def generate_date_range(start_date, end_date, fiscal_year=False):
    """
    Generate a list of dates between start_date and end_date
    
    Args:
        start_date (datetime): Start date
        end_date (datetime): End date
        fiscal_year (bool): If True, generate dates by fiscal year
        
    Returns:
        list: List of date strings
    """
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)
    
    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date)
    
    # Generate date range
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    if fiscal_year:
        # Group by fiscal year
        fiscal_years = {}
        for date in date_range:
            fy = calculate_fiscal_year(date)
            if fy not in fiscal_years:
                fiscal_years[fy] = []
            fiscal_years[fy].append(date.strftime('%Y-%m-%d'))
        
        return fiscal_years
    else:
        return [date.strftime('%Y-%m-%d') for date in date_range]

def calculate_class_conflicts(schedule):
    """
    Calculate conflicts between classes in a schedule
    
    Args:
        schedule (list): List of class dictionaries
        
    Returns:
        list: List of conflict dictionaries
    """
    conflicts = []
    
    # Convert schedule to DataFrame
    schedule_df = pd.DataFrame(schedule)
    
    # Convert date strings to datetime
    schedule_df['start_date'] = pd.to_datetime(schedule_df['start_date'])
    schedule_df['end_date'] = pd.to_datetime(schedule_df['end_date'])
    
    # Check each pair of classes for conflicts
    for i, class1 in schedule_df.iterrows():
        for j, class2 in schedule_df.iterrows():
            if i >= j:  # Skip self-comparison and duplicates
                continue
            
            # Check if classes overlap in time
            if (class1['start_date'] <= class2['end_date'] and class1['end_date'] >= class2['start_date']):
                # Check if they share prerequisites (which would indicate a conflict)
                course1 = class1['course_title']
                course2 = class2['course_title']
                
                # Calculate overlap days
                overlap_start = max(class1['start_date'], class2['start_date'])
                overlap_end = min(class1['end_date'], class2['end_date'])
                overlap_days = (overlap_end - overlap_start).days + 1
                
                conflicts.append({
                    'course1': course1,
                    'course2': course2,
                    'start1': class1['start_date'].strftime('%Y-%m-%d'),
                    'end1': class1['end_date'].strftime('%Y-%m-%d'),
                    'start2': class2['start_date'].strftime('%Y-%m-%d'),
                    'end2': class2['end_date'].strftime('%Y-%m-%d'),
                    'overlap_days': overlap_days
                })
    
    # Sort conflicts by overlap days (descending)
    conflicts.sort(key=lambda x: x['overlap_days'], reverse=True)
    
    return conflicts

def validate_schedule(schedule, course_configs):
    """
    Validate a schedule against course configurations and constraints
    
    Args:
        schedule (list): List of class dictionaries
        course_configs (dict): Dictionary of course configurations
        
    Returns:
        dict: Validation results
    """
    issues = []
    warnings = []
    
    # Convert schedule to DataFrame
    schedule_df = pd.DataFrame(schedule)
    
    # Convert date strings to datetime
    schedule_df['start_date'] = pd.to_datetime(schedule_df['start_date'])
    schedule_df['end_date'] = pd.to_datetime(schedule_df['end_date'])
    
    # Check for date ranges
    invalid_dates = schedule_df[schedule_df['start_date'] >= schedule_df['end_date']]
    if not invalid_dates.empty:
        for _, row in invalid_dates.iterrows():
            issues.append(f"Invalid date range for {row['course_title']}: start date is on or after end date")
    
    # Check for class conflicts
    conflicts = calculate_class_conflicts(schedule)
    for conflict in conflicts:
        # Determine if these courses have prerequisites between them
        course1 = conflict['course1']
        course2 = conflict['course2']
        
        # Skip reporting conflicts for unrelated courses
        is_related = False
        
        # Check if one is a prerequisite of the other
        if course1 in course_configs and course2 in course_configs.get(course1, {}).get('prerequisites', []):
            is_related = True
        
        if course2 in course_configs and course1 in course_configs.get(course2, {}).get('prerequisites', []):
            is_related = True
        
        if is_related:
            issues.append(
                f"Schedule conflict: {course1} and {course2} overlap by {conflict['overlap_days']} days, " +
                f"but one is a prerequisite of the other"
            )
        else:
            warnings.append(
                f"Potential resource conflict: {course1} and {course2} overlap by {conflict['overlap_days']} days"
            )
    
    # Check classes per fiscal year
    course_fy_counts = defaultdict(lambda: defaultdict(int))
    
    for _, row in schedule_df.iterrows():
        course = row['course_title']
        fy = calculate_fiscal_year(row['start_date'])
        course_fy_counts[course][fy] += 1
    
    for course, fy_counts in course_fy_counts.items():
        if course in course_configs:
            expected_classes = course_configs[course].get('classes_per_year', 0)
            
            for fy, count in fy_counts.items():
                if expected_classes > 0 and count > expected_classes:
                    warnings.append(
                        f"FY{fy}: {course} has {count} classes scheduled, but configuration specifies {expected_classes} per year"
                    )
    
    # Check class sizes
    for _, row in schedule_df.iterrows():
        course = row['course_title']
        size = row['size']
        
        if course in course_configs:
            max_capacity = course_configs[course].get('max_capacity', 0)
            
            if max_capacity > 0 and size > max_capacity:
                issues.append(
                    f"{course} starting {row['start_date'].strftime('%Y-%m-%d')} has size {size}, " +
                    f"but maximum capacity is {max_capacity}"
                )
    
    return {
        'valid': len(issues) == 0,
        'issues': issues,
        'warnings': warnings
    }