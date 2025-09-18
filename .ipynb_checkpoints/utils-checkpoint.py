import pandas as pd
import numpy as np
import datetime
import json
from collections import defaultdict
from typing import Dict, List, Any, Tuple, Optional, Union

def ensure_config_compatibility(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure backward compatibility with older configuration formats
    
    Args:
        config: Course configuration dictionary
        
    Returns:
        Updated configuration with compatible format
    """
    # Handle officer-enlisted ratio compatibility
    if 'officer_enlisted_ratio' in config:
        if config['officer_enlisted_ratio'] == "" or not config['officer_enlisted_ratio']:
            config['officer_enlisted_ratio'] = None
    
    # Handle prerequisites compatibility (if you've updated that structure)
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
    
    # Add even MOS ratio setting if it doesn't exist
    if 'use_even_mos_ratio' not in config:
        config['use_even_mos_ratio'] = False
    
    return config

def calculate_fiscal_year(date: Union[str, datetime.datetime, pd.Timestamp]) -> int:
    """
    Calculate the fiscal year for a given date.
    The fiscal year runs from October 1 to September 30.
    
    Args:
        date: The date to calculate fiscal year for
        
    Returns:
        Fiscal year
    """
    if isinstance(date, str):
        try:
            date = pd.to_datetime(date)
        except:
            # Return current year if date parsing fails
            return datetime.datetime.now().year
    
    if date.month >= 10:  # October through December
        return date.year + 1
    else:  # January through September
        return date.year

def parse_ratio(ratio_str: Optional[str]) -> Optional[Tuple[int, int]]:
    """
    Parse a ratio string (e.g., '1:4') into a tuple of integers
    
    Args:
        ratio_str: String representation of a ratio
        
    Returns:
        Tuple of (numerator, denominator) or None if invalid
    """
    if not ratio_str:  # Handle None or empty string
        return None
    
    try:
        numerator, denominator = ratio_str.split(':')
        return (int(numerator), int(denominator))
    except (ValueError, AttributeError, TypeError):
        # Log the error for debugging
        print(f"Error parsing ratio string: '{ratio_str}'")
        # Default to 1:4 if parsing fails
        return (1, 4)

def format_duration(days: Union[int, float]) -> str:
    """
    Format duration in days to a human-readable string
    
    Args:
        days: Duration in days
        
    Returns:
        Formatted duration string
    """
    try:
        days = int(days)
    except (ValueError, TypeError):
        return "Invalid duration"
    
    if days < 0:
        return "Invalid duration"
    elif days == 0:
        return "0 days"
    elif days < 7:
        return f"{days} day{'s' if days != 1 else ''}"
    elif days < 30:
        weeks = days // 7
        remaining_days = days % 7
        
        if remaining_days == 0:
            return f"{weeks} week{'s' if weeks != 1 else ''}"
        else:
            return f"{weeks} week{'s' if weeks != 1 else ''}, {remaining_days} day{'s' if remaining_days != 1 else ''}"
    elif days < 365:
        months = days // 30
        remaining_days = days % 30
        
        if remaining_days == 0:
            return f"{months} month{'s' if months != 1 else ''}"
        else:
            return f"{months} month{'s' if months != 1 else ''}, {remaining_days} day{'s' if remaining_days != 1 else ''}"
    else:
        years = days // 365
        remaining_days = days % 365
        
        if remaining_days == 0:
            return f"{years} year{'s' if years != 1 else ''}"
        else:
            months = remaining_days // 30
            if months == 0:
                return f"{years} year{'s' if years != 1 else ''}"
            else:
                return f"{years} year{'s' if years != 1 else ''}, {months} month{'s' if months != 1 else ''}"

def generate_date_range(start_date: Union[str, datetime.datetime, pd.Timestamp], 
                        end_date: Union[str, datetime.datetime, pd.Timestamp], 
                        fiscal_year: bool = False) -> Union[List[str], Dict[int, List[str]]]:
    """
    Generate a list of dates between start_date and end_date
    
    Args:
        start_date: Start date
        end_date: End date
        fiscal_year: If True, generate dates by fiscal year
        
    Returns:
        List of date strings or dictionary of fiscal years to date strings
    """
    # Validate input dates
    try:
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
    except:
        return []  # Return empty list if date parsing fails
    
    if start_date > end_date:
        return []  # Return empty list if start date is after end date
    
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

def calculate_class_conflicts(schedule: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Calculate conflicts between classes in a schedule
    
    Args:
        schedule: List of class dictionaries
        
    Returns:
        List of conflict dictionaries
    """
    if not schedule:
        return []  # Return empty list if schedule is empty
    
    conflicts = []
    
    try:
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
                    # Calculate overlap days
                    overlap_start = max(class1['start_date'], class2['start_date'])
                    overlap_end = min(class1['end_date'], class2['end_date'])
                    overlap_days = (overlap_end - overlap_start).days + 1
                    
                    # Check for MOS allocation overlap
                    mos_overlap = False
                    
                    if 'mos_allocation' in class1 and 'mos_allocation' in class2:
                        # Find MOS paths that overlap between the classes
                        common_mos = set(class1['mos_allocation'].keys()) & set(class2['mos_allocation'].keys())
                        
                        # If there are common MOS paths with allocations > 0, there's a potential MOS conflict
                        for mos in common_mos:
                            if class1['mos_allocation'].get(mos, 0) > 0 and class2['mos_allocation'].get(mos, 0) > 0:
                                mos_overlap = True
                                break
                    else:
                        # If MOS allocation not specified, assume overlap
                        mos_overlap = True
                    
                    if mos_overlap:
                        conflicts.append({
                            'course1': class1['course_title'],
                            'course2': class2['course_title'],
                            'start1': class1['start_date'].strftime('%Y-%m-%d'),
                            'end1': class1['end_date'].strftime('%Y-%m-%d'),
                            'start2': class2['start_date'].strftime('%Y-%m-%d'),
                            'end2': class2['end_date'].strftime('%Y-%m-%d'),
                            'overlap_days': overlap_days,
                            'conflict_severity': 'high' if overlap_days > 5 else 'medium' if overlap_days > 1 else 'low'
                        })
    
    except Exception as e:
        print(f"Error calculating class conflicts: {e}")
        return []
    
    # Sort conflicts by overlap days (descending)
    conflicts.sort(key=lambda x: x['overlap_days'], reverse=True)
    
    return conflicts

def validate_schedule(schedule: List[Dict[str, Any]], course_configs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Validate a schedule against course configurations and constraints
    
    Args:
        schedule: List of class dictionaries
        course_configs: Dictionary of course configurations
        
    Returns:
        Dictionary with validation results
    """
    issues = []
    warnings = []
    
    # Check for empty schedule
    if not schedule:
        return {
            'valid': True,
            'issues': [],
            'warnings': ["Schedule is empty. No validation performed."]
        }
    
    try:
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
            if course1 in course_configs and course2 in get_prerequisites(course_configs, course1):
                is_related = True
            
            if course2 in course_configs and course1 in get_prerequisites(course_configs, course2):
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
        
        # Check MOS allocations
        for _, row in schedule_df.iterrows():
            if 'mos_allocation' in row and row['mos_allocation']:
                course = row['course_title']
                total_allocation = sum(row['mos_allocation'].values())
                
                # Check if total MOS allocation exceeds class size
                if total_allocation > row['size']:
                    issues.append(
                        f"{course} starting {row['start_date'].strftime('%Y-%m-%d')} has total MOS allocation {total_allocation}, " +
                        f"but class size is {row['size']}"
                    )
                
                # Check if any MOS has zero allocation
                for mos, count in row['mos_allocation'].items():
                    if count < 0:
                        issues.append(
                            f"{course} starting {row['start_date'].strftime('%Y-%m-%d')} has negative allocation for {mos}: {count}"
                        )
        
        # Check prerequisite order violations
        prereq_violations = check_prerequisite_order(schedule, course_configs)
        issues.extend(prereq_violations)
    
    except Exception as e:
        issues.append(f"Error validating schedule: {e}")
    
    return {
        'valid': len(issues) == 0,
        'issues': issues,
        'warnings': warnings
    }

def get_prerequisites(course_configs: Dict[str, Dict[str, Any]], course: str) -> List[str]:
    """
    Get all prerequisites for a course (including from OR groups and MOS paths)
    
    Args:
        course_configs: Dictionary of course configurations
        course: Course to get prerequisites for
        
    Returns:
        List of prerequisite courses
    """
    if course not in course_configs:
        return []
    
    config = course_configs[course]
    all_prereqs = set()
    
    # Get standard prerequisites
    if 'prerequisites' in config:
        if isinstance(config['prerequisites'], list):
            all_prereqs.update(config['prerequisites'])
        elif isinstance(config['prerequisites'], dict):
            all_prereqs.update(config['prerequisites'].get('courses', []))
    
    # Get OR prerequisites
    if 'or_prerequisites' in config:
        for group in config['or_prerequisites']:
            all_prereqs.update(group)

    # Get MOS-specific prerequisites
    if 'mos_paths' in config:
        for mos, prereqs in config['mos_paths'].items():
            all_prereqs.update(prereqs)
    
    return list(all_prereqs)  # Remove duplicates by converting set back to list

def analyze_mos_enrollment(schedule: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze MOS enrollment patterns across the schedule
    
    Args:
        schedule: List of class dictionaries
        
    Returns:
        Dictionary with MOS enrollment statistics
    """
    # Handle empty schedule
    if not schedule:
        return {
            'mos_stats': {
                '18A': {'total': 0, 'avg_per_class': 0, 'max_per_class': 0, 'min_per_class': 0},
                '18B': {'total': 0, 'avg_per_class': 0, 'max_per_class': 0, 'min_per_class': 0},
                '18C': {'total': 0, 'avg_per_class': 0, 'max_per_class': 0, 'min_per_class': 0},
                '18D': {'total': 0, 'avg_per_class': 0, 'max_per_class': 0, 'min_per_class': 0},
                '18E': {'total': 0, 'avg_per_class': 0, 'max_per_class': 0, 'min_per_class': 0}
            },
            'course_mos_ratios': {}
        }
    
    # Track MOS enrollment
    mos_enrollment = {
        '18A': [],
        '18B': [],
        '18C': [],
        '18D': [],
        '18E': []
    }
    
    # Track course-specific MOS enrollment
    course_mos_enrollment = defaultdict(lambda: {
        '18A': 0,
        '18B': 0,
        '18C': 0,
        '18D': 0,
        '18E': 0,
        'total': 0
    })
    
    # Process each class
    for class_info in schedule:
        course = class_info['course_title']
        
        if 'mos_allocation' in class_info and class_info['mos_allocation']:
            for mos, count in class_info['mos_allocation'].items():
                if mos in mos_enrollment:
                    mos_enrollment[mos].append(count)
                    course_mos_enrollment[course][mos] += count
                    course_mos_enrollment[course]['total'] += count
    
    # Calculate statistics
    mos_stats = {}
    
    for mos, counts in mos_enrollment.items():
        if counts:
            mos_stats[mos] = {
                'total': sum(counts),
                'avg_per_class': np.mean(counts),
                'max_per_class': np.max(counts),
                'min_per_class': np.min(counts),
                'std_per_class': np.std(counts),
                'num_classes_with_mos': len(counts)
            }
        else:
            mos_stats[mos] = {
                'total': 0,
                'avg_per_class': 0,
                'max_per_class': 0,
                'min_per_class': 0,
                'std_per_class': 0,
                'num_classes_with_mos': 0
            }
    
    # Calculate course-specific MOS ratios
    course_mos_ratios = {}
    
    for course, counts in course_mos_enrollment.items():
        total = counts['total']
        
        if total > 0:
            course_mos_ratios[course] = {
                mos: counts[mos] / total for mos in ['18A', '18B', '18C', '18D', '18E']
            }
            course_mos_ratios[course]['total'] = total
            
            # Calculate balance score (0-1, where 1 is perfectly balanced)
            ideal_ratio = 0.2  # Each MOS gets 20% in a perfect balance
            deviations = [abs(counts[mos] / total - ideal_ratio) for mos in ['18A', '18B', '18C', '18D', '18E']]
            balance_score = 1 - (sum(deviations) / 2)  # Normalize to 0-1 range
            course_mos_ratios[course]['balance_score'] = balance_score
        else:
            course_mos_ratios[course] = {
                '18A': 0, '18B': 0, '18C': 0, '18D': 0, '18E': 0, 'total': 0, 'balance_score': 0
            }
    
    # Calculate overall MOS balance
    total_seats = sum(mos_stats[mos]['total'] for mos in mos_stats)
    
    if total_seats > 0:
        overall_distribution = {mos: mos_stats[mos]['total'] / total_seats for mos in mos_stats}
        ideal_ratio = 0.2  # Each MOS gets 20% in a perfect balance
        deviations = [abs(overall_distribution[mos] - ideal_ratio) for mos in overall_distribution]
        overall_balance_score = 1 - (sum(deviations) / 2)  # Normalize to 0-1 range
    else:
        overall_distribution = {mos: 0 for mos in mos_stats}
        overall_balance_score = 0
    
    return {
        'mos_stats': mos_stats,
        'course_mos_ratios': course_mos_ratios,
        'overall_distribution': overall_distribution,
        'overall_balance_score': overall_balance_score
    }

def format_mos_allocation(mos_allocation: Dict[str, int]) -> str:
    """
    Format MOS allocation for display
    
    Args:
        mos_allocation: Dictionary of MOS allocations
        
    Returns:
        Formatted string representation
    """
    if not mos_allocation:
        return "None"
    
    parts = []
    
    for mos, count in mos_allocation.items():
        if count > 0:
            parts.append(f"{mos}: {count}")
    
    if not parts:
        return "None"
    
    return ", ".join(parts)

def parse_mos_allocation(allocation_str: str) -> Dict[str, int]:
    """
    Parse MOS allocation from string
    
    Args:
        allocation_str: String representation of MOS allocation
        
    Returns:
        Dictionary of MOS allocations
    """
    if not allocation_str or allocation_str.lower() == "none":
        return {}
    
    allocation = {
        '18A': 0,
        '18B': 0,
        '18C': 0,
        '18D': 0,
        '18E': 0
    }
    
    try:
        parts = allocation_str.split(",")
        
        for part in parts:
            if ":" not in part:
                continue
            
            mos, count_str = part.split(":")
            mos = mos.strip()
            count = int(count_str.strip())
            
            if mos in allocation:
                allocation[mos] = count
    
    except Exception as e:
        print(f"Error parsing MOS allocation '{allocation_str}': {e}")
        # Return empty allocation if parsing fails
        return {}
    
    return allocation

def calculate_schedule_metrics(schedule: List[Dict[str, Any]], course_configs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate overall metrics for a schedule
    
    Args:
        schedule: List of class dictionaries
        course_configs: Dictionary of course configurations
        
    Returns:
        Dictionary of schedule metrics
    """
    # Handle empty schedule
    if not schedule:
        return {
            'total_classes': 0,
            'total_seats': 0,
            'duration_days': 0,
            'classes_per_month': 0,
            'avg_class_size': 0,
            'mos_distribution': {
                '18A': 0, '18B': 0, '18C': 0, '18D': 0, '18E': 0
            },
            'seats_per_course': {},
            'program_throughput': 0
        }
    
    try:
        # Convert to DataFrame for easier manipulation
        schedule_df = pd.DataFrame(schedule)
        
        # Convert dates to datetime
        schedule_df['start_date'] = pd.to_datetime(schedule_df['start_date'])
        schedule_df['end_date'] = pd.to_datetime(schedule_df['end_date'])
        
        # Calculate total classes and seats
        total_classes = len(schedule_df)
        total_seats = schedule_df['size'].sum()
        
        # Calculate schedule duration
        min_date = schedule_df['start_date'].min()
        max_date = schedule_df['end_date'].max()
        duration_days = (max_date - min_date).days + 1
        
        # Calculate classes per month
        months = duration_days / 30.0
        classes_per_month = total_classes / months if months > 0 else 0
        
        # Calculate average class size
        avg_class_size = total_seats / total_classes if total_classes > 0 else 0
        
        # Calculate MOS distribution
        mos_counts = {
            '18A': 0,
            '18B': 0,
            '18C': 0,
            '18D': 0,
            '18E': 0
        }
        
        for _, row in schedule_df.iterrows():
            if 'mos_allocation' in row and row['mos_allocation']:
                for mos, count in row['mos_allocation'].items():
                    if mos in mos_counts:
                        mos_counts[mos] += count
        
        # Calculate percentages
        total_allocated = sum(mos_counts.values())
        
        mos_distribution = {
            mos: count / total_allocated if total_allocated > 0 else 0
            for mos, count in mos_counts.items()
        }
        
        # Calculate throughput metrics
        # Group classes by course
        courses = schedule_df.groupby('course_title')
        
        # Calculate seats per course
        seats_per_course = {
            course: group['size'].sum() for course, group in courses
        }
        
        # Calculate potential bottlenecks
        course_throughputs = {}
        
        for course, config in course_configs.items():
            if course not in seats_per_course:
                continue
            
            # Get prerequisites
            prereqs = []
            
            if 'prerequisites' in config and isinstance(config['prerequisites'], dict):
                prereqs.extend(config['prerequisites'].get('courses', []))
            
            # Calculate maximum possible throughput based on this course and its prerequisites
            throughput = seats_per_course.get(course, 0)
            
            for prereq in prereqs:
                if prereq in seats_per_course:
                    # Throughput is limited by the minimum seats available in any prerequisite
                    throughput = min(throughput, seats_per_course[prereq])
            
            course_throughputs[course] = throughput
        
        # Overall program throughput is the minimum throughput of any required course
        program_throughput = min(course_throughputs.values()) if course_throughputs else 0
        
        # Calculate class density
        class_density = {}
        
        for month_start in pd.date_range(min_date, max_date, freq='MS'):
            month_end = month_start + pd.DateOffset(months=1) - pd.DateOffset(days=1)
            month_key = month_start.strftime('%Y-%m')
            
            # Count classes that overlap with this month
            month_classes = schedule_df[
                (schedule_df['start_date'] <= month_end) & 
                (schedule_df['end_date'] >= month_start)
            ]
            
            class_density[month_key] = len(month_classes)
        
        # Find most congested periods
        if class_density:
            avg_density = np.mean(list(class_density.values()))
            peak_months = {month: count for month, count in class_density.items() if count > avg_density * 1.5}
        else:
            avg_density = 0
            peak_months = {}
        
        return {
            'total_classes': total_classes,
            'total_seats': total_seats,
            'duration_days': duration_days,
            'classes_per_month': classes_per_month,
            'avg_class_size': avg_class_size,
            'mos_distribution': mos_distribution,
            'seats_per_course': seats_per_course,
            'program_throughput': program_throughput,
            'class_density': class_density,
            'avg_density': avg_density,
            'peak_months': peak_months
        }
    
    except Exception as e:
        print(f"Error calculating schedule metrics: {e}")
        
        return {
            'total_classes': 0,
            'total_seats': 0,
            'duration_days': 0,
            'classes_per_month': 0,
            'avg_class_size': 0,
            'mos_distribution': {
                '18A': 0, '18B': 0, '18C': 0, '18D': 0, '18E': 0
            },
            'seats_per_course': {},
            'program_throughput': 0,
            'error': str(e)
        }

def optimize_mos_allocation(class_info: Dict[str, Any], mos_demand: Optional[Dict[str, float]] = None) -> Dict[str, int]:
    """
    Optimize MOS allocation for a class based on demand
    
    Args:
        class_info: Class information
        mos_demand: Dictionary of MOS demand percentages
        
    Returns:
        Optimized MOS allocation
    """
    if not mos_demand:
        # Default demand distribution if not provided
        mos_demand = {
            '18A': 0.2,
            '18B': 0.2,
            '18C': 0.2,
            '18D': 0.2,
            '18E': 0.2
        }
    
    total_size = class_info['size']
    
    # Calculate allocation based on demand
    allocation = {}
    remaining = total_size
    
    # First pass: allocate seats based on demand percentages
    for mos, demand in mos_demand.items():
        seats = int(total_size * demand)
        allocation[mos] = seats
        remaining -= seats
    
    # Distribute remaining seats to MOS with highest fractional parts
    if remaining > 0:
        fractional_parts = {
            mos: (total_size * demand) - int(total_size * demand)
            for mos, demand in mos_demand.items()
        }
        
        # Sort MOS by fractional part (descending)
        sorted_mos = sorted(fractional_parts.items(), key=lambda x: x[1], reverse=True)
        
        # Distribute remaining seats
        for i in range(remaining):
            if i < len(sorted_mos):
                mos = sorted_mos[i][0]
                allocation[mos] += 1
    
    # Handle negative remaining (should not happen normally)
    elif remaining < 0:
        # Remove seats from MOS with most seats
        sorted_allocation = sorted(allocation.items(), key=lambda x: x[1], reverse=True)
        
        for i in range(-remaining):
            if i < len(sorted_allocation):
                mos = sorted_allocation[i][0]
                if allocation[mos] > 0:
                    allocation[mos] -= 1
    
    return allocation

def compare_schedules(original_schedule: List[Dict[str, Any]], new_schedule: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compare two schedules and identify changes
    
    Args:
        original_schedule: Original schedule
        new_schedule: New schedule
        
    Returns:
        Dictionary with comparison results
    """
    # Handle empty schedules
    if not original_schedule and not new_schedule:
        return {
            'added_classes': [],
            'removed_classes': [],
            'modified_classes': [],
            'original_metrics': {'total_classes': 0, 'total_seats': 0},
            'new_metrics': {'total_classes': 0, 'total_seats': 0},
            'net_change_classes': 0,
            'net_change_seats': 0
        }
    
    try:
        # Convert to DataFrames
        if original_schedule:
            original_df = pd.DataFrame(original_schedule)
            original_df['start_date'] = pd.to_datetime(original_df['start_date'])
            original_df['end_date'] = pd.to_datetime(original_df['end_date'])
        else:
            original_df = pd.DataFrame(columns=['id', 'course_title', 'start_date', 'end_date', 'size'])
        
        if new_schedule:
            new_df = pd.DataFrame(new_schedule)
            new_df['start_date'] = pd.to_datetime(new_df['start_date'])
            new_df['end_date'] = pd.to_datetime(new_df['end_date'])
        else:
            new_df = pd.DataFrame(columns=['id', 'course_title', 'start_date', 'end_date', 'size'])
        
        # Identify added classes
        if not original_df.empty and not new_df.empty:
            original_ids = set(original_df['id'])
            new_ids = set(new_df['id'])
            
            added_ids = new_ids - original_ids
            removed_ids = original_ids - new_ids
            modified_ids = set()
            
            # Identify modified classes
            for class_id in original_ids & new_ids:
                original_class = original_df[original_df['id'] == class_id].iloc[0]
                new_class = new_df[new_df['id'] == class_id].iloc[0]
                
                # Check if any key attributes changed
                if (original_class['start_date'] != new_class['start_date'] or
                    original_class['end_date'] != new_class['end_date'] or
                    original_class['size'] != new_class['size']):
                    modified_ids.add(class_id)
        else:
            # If either schedule is empty, all classes are added or removed
            added_ids = set(new_df['id']) if not new_df.empty else set()
            removed_ids = set(original_df['id']) if not original_df.empty else set()
            modified_ids = set()
        
        # Create detailed change records
        added_classes = []
        removed_classes = []
        modified_classes = []
        
        for class_id in added_ids:
            new_class = new_df[new_df['id'] == class_id].iloc[0]
            
            added_classes.append({
                'id': int(class_id),
                'course_title': new_class['course_title'],
                'start_date': new_class['start_date'].strftime('%Y-%m-%d'),
                'end_date': new_class['end_date'].strftime('%Y-%m-%d'),
                'size': int(new_class['size'])
            })
        
        for class_id in removed_ids:
            original_class = original_df[original_df['id'] == class_id].iloc[0]
            
            removed_classes.append({
                'id': int(class_id),
                'course_title': original_class['course_title'],
                'start_date': original_class['start_date'].strftime('%Y-%m-%d'),
                'end_date': original_class['end_date'].strftime('%Y-%m-%d'),
                'size': int(original_class['size'])
            })
        
        for class_id in modified_ids:
            original_class = original_df[original_df['id'] == class_id].iloc[0]
            new_class = new_df[new_df['id'] == class_id].iloc[0]
            
            modified_classes.append({
                'id': int(class_id),
                'course_title': new_class['course_title'],
                'original_start': original_class['start_date'].strftime('%Y-%m-%d'),
                'original_end': original_class['end_date'].strftime('%Y-%m-%d'),
                'original_size': int(original_class['size']),
                'new_start': new_class['start_date'].strftime('%Y-%m-%d'),
                'new_end': new_class['end_date'].strftime('%Y-%m-%d'),
                'new_size': int(new_class['size'])
            })
        
        # Calculate summary metrics
        original_metrics = {
            'total_classes': len(original_df),
            'total_seats': original_df['size'].sum() if not original_df.empty else 0
        }
        
        new_metrics = {
            'total_classes': len(new_df),
            'total_seats': new_df['size'].sum() if not new_df.empty else 0
        }
        
        # Calculate advanced comparison metrics
        if not original_df.empty and not new_df.empty:
            # Calculate total class days in each schedule
            original_days = sum((row['end_date'] - row['start_date']).days + 1 for _, row in original_df.iterrows())
            new_days = sum((row['end_date'] - row['start_date']).days + 1 for _, row in new_df.iterrows())
            
            # Calculate average class size
            original_avg_size = original_df['size'].mean() if len(original_df) > 0 else 0
            new_avg_size = new_df['size'].mean() if len(new_df) > 0 else 0
            
            # Calculate schedule density
            original_span = (original_df['end_date'].max() - original_df['start_date'].min()).days + 1 if len(original_df) > 0 else 0
            new_span = (new_df['end_date'].max() - new_df['start_date'].min()).days + 1 if len(new_df) > 0 else 0
            
            original_density = original_days / original_span if original_span > 0 else 0
            new_density = new_days / new_span if new_span > 0 else 0
            
            advanced_metrics = {
                'original_days': original_days,
                'new_days': new_days,
                'day_change': new_days - original_days,
                'day_change_pct': (new_days - original_days) / original_days * 100 if original_days > 0 else 0,
                'original_avg_size': original_avg_size,
                'new_avg_size': new_avg_size,
                'size_change': new_avg_size - original_avg_size,
                'size_change_pct': (new_avg_size - original_avg_size) / original_avg_size * 100 if original_avg_size > 0 else 0,
                'original_density': original_density,
                'new_density': new_density,
                'density_change': new_density - original_density,
                'density_change_pct': (new_density - original_density) / original_density * 100 if original_density > 0 else 0
            }
        else:
            advanced_metrics = {}
        
        return {
            'added_classes': added_classes,
            'removed_classes': removed_classes,
            'modified_classes': modified_classes,
            'original_metrics': original_metrics,
            'new_metrics': new_metrics,
            'advanced_metrics': advanced_metrics,
            'net_change_classes': new_metrics['total_classes'] - original_metrics['total_classes'],
            'net_change_seats': new_metrics['total_seats'] - original_metrics['total_seats']
        }
    
    except Exception as e:
        print(f"Error comparing schedules: {e}")
        
        return {
            'added_classes': [],
            'removed_classes': [],
            'modified_classes': [],
            'original_metrics': {'total_classes': 0, 'total_seats': 0},
            'new_metrics': {'total_classes': 0, 'total_seats': 0},
            'net_change_classes': 0,
            'net_change_seats': 0,
            'error': str(e)
        }

def json_serializer(obj: Any) -> Any:
    """
    Custom JSON serializer for objects not serializable by default json code
    
    Args:
        obj: Python object to serialize
        
    Returns:
        JSON serializable object
    """
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    
    if isinstance(obj, (datetime.date, datetime.datetime)):
        return obj.isoformat()
    
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    
    # For anything else that's not JSON serializable
    return str(obj)

def suggest_schedule_improvements(schedule: List[Dict[str, Any]], bottlenecks: Optional[List[Dict[str, Any]]] = None) -> List[str]:
    """
    Suggest potential improvements to a schedule based on common patterns
    
    Args:
        schedule: List of class dictionaries
        bottlenecks: List of bottleneck courses with wait times
        
    Returns:
        List of suggestions for schedule improvements
    """
    suggestions = []
    
    # Handle empty schedule
    if not schedule:
        return ["Add classes to the schedule to begin optimization."]
    
    try:
        # Convert to DataFrame
        schedule_df = pd.DataFrame(schedule)
        schedule_df['start_date'] = pd.to_datetime(schedule_df['start_date'])
        schedule_df['end_date'] = pd.to_datetime(schedule_df['end_date'])
        
        # Group by course
        course_groups = schedule_df.groupby('course_title')
        
        # Check for courses with few classes
        for course, group in course_groups:
            if len(group) == 1:
                suggestions.append(f"Consider adding more {course} classes as there is currently only one.")
        
        # Check for large gaps between classes
        for course, group in course_groups:
            if len(group) >= 2:
                sorted_group = group.sort_values('start_date')
                
                for i in range(len(sorted_group) - 1):
                    gap = (sorted_group.iloc[i+1]['start_date'] - sorted_group.iloc[i]['end_date']).days
                    
                    if gap > 60:
                        suggestions.append(
                            f"Large gap of {gap} days between {course} classes ending {sorted_group.iloc[i]['end_date'].strftime('%Y-%m-%d')} " +
                            f"and starting {sorted_group.iloc[i+1]['start_date'].strftime('%Y-%m-%d')}."
                        )
        
        # Use bottleneck information if provided
        if bottlenecks:
            for bottleneck in bottlenecks:
                course = bottleneck.get('course')
                wait_time = bottleneck.get('wait_time')
                
                if course and wait_time and wait_time > 10:
                    course_classes = schedule_df[schedule_df['course_title'] == course]
                    
                    if len(course_classes) < 3:
                        suggestions.append(f"Consider adding more {course} classes to reduce the {wait_time:.1f} day wait time.")
                    else:
                        suggestions.append(f"Consider increasing capacity for {course} classes to reduce the {wait_time:.1f} day wait time.")
        
        # Check for overlapping classes that might cause resource conflicts
        conflicts = calculate_class_conflicts(schedule)
        
        if conflicts:
            suggestions.append(f"Found {len(conflicts)} potential class conflicts. Consider adjusting dates to reduce overlaps.")
            
            # Suggest specific fixes for the worst conflicts
            severe_conflicts = [c for c in conflicts if c.get('overlap_days', 0) > 5]
            
            if severe_conflicts:
                for conflict in severe_conflicts[:3]:  # Show top 3 severe conflicts
                    course1 = conflict['course1']
                    course2 = conflict['course2']
                    overlap = conflict['overlap_days']
                    
                    suggestions.append(
                        f"Severe conflict: {course1} and {course2} overlap by {overlap} days. " +
                        f"Consider rescheduling one of these classes."
                    )
        
        # Check MOS allocations
        total_mos_allocation = {
            '18A': 0,
            '18B': 0,
            '18C': 0,
            '18D': 0,
            '18E': 0
        }
        
        for class_info in schedule:
            if 'mos_allocation' in class_info and class_info['mos_allocation']:
                for mos, count in class_info['mos_allocation'].items():
                    if mos in total_mos_allocation:
                        total_mos_allocation[mos] += count
        
        # Check for imbalanced MOS allocations
        total_allocated = sum(total_mos_allocation.values())
        
        if total_allocated > 0:
            for mos, count in total_mos_allocation.items():
                percentage = count / total_allocated
                
                if percentage < 0.1:
                    suggestions.append(f"Low allocation for {mos} ({percentage:.1%} of total). Consider increasing seats for this MOS.")
                elif percentage > 0.3:
                    suggestions.append(f"High allocation for {mos} ({percentage:.1%} of total). Consider if this matches your training needs.")
        
        # Check schedule balance across the year
        if len(schedule_df) > 0:
            # Group classes by month
            schedule_df['month'] = schedule_df['start_date'].dt.to_period('M')
            monthly_counts = schedule_df.groupby('month').size()
            
            if len(monthly_counts) > 0:
                avg_classes_per_month = monthly_counts.mean()
                max_classes = monthly_counts.max()
                min_classes = monthly_counts.min()
                
                # Check for highly uneven distribution
                if max_classes > avg_classes_per_month * 2:
                    peak_month = monthly_counts.idxmax().strftime('%B %Y')
                    suggestions.append(f"Schedule is heavily concentrated in {peak_month} ({max_classes} classes). Consider spreading classes more evenly.")
                
                # Check for empty months
                if len(monthly_counts) < 12 and len(schedule_df) > 12:
                    suggestions.append("Some months have no classes scheduled. Consider a more consistent distribution throughout the year.")
    
    except Exception as e:
        suggestions.append(f"Error analyzing schedule for improvements: {e}")
    
    return suggestions

def check_prerequisite_order(schedule: List[Dict[str, Any]], course_configs: Dict[str, Dict[str, Any]]) -> List[str]:
    """
    Check if the schedule respects prerequisite ordering
    
    Args:
        schedule: List of class dictionaries
        course_configs: Dictionary of course configurations
        
    Returns:
        List of violations of prerequisite ordering
    """
    violations = []
    
    # Handle empty schedule
    if not schedule:
        return violations
    
    try:
        schedule_df = pd.DataFrame(schedule)
        schedule_df['start_date'] = pd.to_datetime(schedule_df['start_date'])
        schedule_df['end_date'] = pd.to_datetime(schedule_df['end_date'])
        
        # For each course, check if prerequisites start before it
        for _, class_info in schedule_df.iterrows():
            course = class_info['course_title']
            
            if course not in course_configs:
                continue
            
            start_date = class_info['start_date']
            prereqs = get_prerequisites(course_configs, course)
            
            for prereq in prereqs:
                # Find all classes for this prerequisite
                prereq_classes = schedule_df[schedule_df['course_title'] == prereq]
                
                if len(prereq_classes) == 0:
                    violations.append(f"Prerequisite {prereq} for {course} is not scheduled at all.")
                    continue
                
                # Check if at least one prerequisite class ends before this course starts
                if not any(prereq_classes['end_date'] < start_date):
                    violations.append(
                        f"No {prereq} class ends before {course} starts on {start_date.strftime('%Y-%m-%d')}, " +
                        f"but {prereq} is a prerequisite for {course}."
                    )
    
    except Exception as e:
        violations.append(f"Error checking prerequisite order: {e}")
    
    return violations

def apply_custom_paths_to_configs(course_configs: Dict[str, Dict[str, Any]], 
                                custom_paths: Dict[str, Dict[str, Any]]) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Apply custom career paths to course configurations by setting prerequisites
    
    Args:
        course_configs: Dictionary of course configurations
        custom_paths: Dictionary of custom career paths by MOS
        
    Returns:
        Updated course configurations and list of changes made
    """
    import copy
    # Make a copy to avoid modifying the original
    updated_configs = copy.deepcopy(course_configs)
    changes_made = []
    
    # Process each MOS path
    for mos, path_data in custom_paths.items():
        typical_path = path_data.get('path', [])
        flexible_courses = path_data.get('flexible_courses', [])
        or_groups = path_data.get('or_groups', []) if 'or_groups' in path_data else []
        flexible_constraints = path_data.get('flexible_constraints', {}) if 'flexible_constraints' in path_data else {}
        
        # Skip if the path is empty
        if not typical_path and not flexible_courses and not or_groups:
            continue
        
        # Process the main path to set prerequisites
        for i, course in enumerate(typical_path):
            # Skip if course doesn't exist in configs
            if course not in updated_configs:
                continue
            
            # Ensure prerequisites structure exists
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
            
            # Set new prerequisites based on the path
            if i > 0:
                # Previous course in path becomes the prerequisite
                updated_configs[course]['prerequisites']['courses'] = [typical_path[i-1]]
                changes_made.append({
                    'course': course,
                    'mos': mos,
                    'prerequisites': [typical_path[i-1]],
                    'action': 'set_prerequisites'
                })
            
            # Update MOS paths
            if 'mos_paths' not in updated_configs[course]:
                updated_configs[course]['mos_paths'] = {
                    '18A': [],
                    '18B': [],
                    '18C': [],
                    '18D': [],
                    '18E': []
                }
            
            # Set this course as part of this MOS path
            if mos != 'General' and mos in updated_configs[course]['mos_paths']:
                # This course is in the path for this MOS
                updated_configs[course]['mos_paths'][mos] = []
                # Record if this is a new addition
                if not any(change['course'] == course and change['mos'] == mos and change['action'] == 'added_to_mos_path' for change in changes_made):
                    changes_made.append({
                        'course': course,
                        'mos': mos,
                        'action': 'added_to_mos_path'
                    })
        
        # Process OR groups
        for group_idx, group in enumerate(or_groups):
            if not group:  # Skip empty groups
                continue
                
            # Find courses in the path that come after this OR group
            for i, course in enumerate(typical_path):
                # Skip if course doesn't exist in configs
                if course not in updated_configs:
                    continue
                
                # Determine if this course comes after any course in the OR group
                or_group_in_path = False
                max_group_position = -1
                
                for or_course in group:
                    if or_course in typical_path:
                        or_position = typical_path.index(or_course)
                        max_group_position = max(max_group_position, or_position)
                        or_group_in_path = True
                
                # If this course comes after the OR group, add the group as a prerequisite
                if or_group_in_path and i > max_group_position:
                    # Ensure or_prerequisites structure exists
                    if 'or_prerequisites' not in updated_configs[course]:
                        updated_configs[course]['or_prerequisites'] = []
                    
                    # Check if this exact group is already in the or_prerequisites
                    if not any(set(existing_group) == set(group) for existing_group in updated_configs[course]['or_prerequisites']):
                        updated_configs[course]['or_prerequisites'].append(group)
                        changes_made.append({
                            'course': course,
                            'mos': mos,
                            'or_group': group,
                            'action': 'added_or_prerequisite'
                        })
        
        # Process flexible courses
        for flex_course in flexible_courses:
            # Skip if course doesn't exist in configs
            if flex_course not in updated_configs:
                continue
            
            # Update MOS paths
            if 'mos_paths' not in updated_configs[flex_course]:
                updated_configs[flex_course]['mos_paths'] = {
                    '18A': [],
                    '18B': [],
                    '18C': [],
                    '18D': [],
                    '18E': []
                }
            
            # Set this course as part of this MOS path
            if mos != 'General' and mos in updated_configs[flex_course]['mos_paths']:
                # This course is in the path for this MOS
                updated_configs[flex_course]['mos_paths'][mos] = []
                # Record if this is a new addition
                if not any(change['course'] == flex_course and change['mos'] == mos and change['action'] == 'added_as_flexible_course' for change in changes_made):
                    changes_made.append({
                        'course': flex_course,
                        'mos': mos,
                        'action': 'added_as_flexible_course'
                    })
            
            # Process flexible course constraints
            if flex_course in flexible_constraints:
                constraints = flexible_constraints[flex_course]
                
                # Process "must be after" constraints (prerequisites for the flexible course)
                after_courses = constraints.get('must_be_after', [])
                if after_courses:
                    # Ensure prerequisites structure exists
                    if 'prerequisites' not in updated_configs[flex_course]:
                        updated_configs[flex_course]['prerequisites'] = {
                            'type': 'AND',
                            'courses': []
                        }
                    elif isinstance(updated_configs[flex_course]['prerequisites'], list):
                        updated_configs[flex_course]['prerequisites'] = {
                            'type': 'AND',
                            'courses': []
                        }
                    
                    # Set the prerequisites
                    updated_configs[flex_course]['prerequisites']['courses'] = after_courses
                    changes_made.append({
                        'course': flex_course,
                        'mos': mos,
                        'prerequisites': after_courses,
                        'action': 'set_flexible_prerequisites'
                    })
                
                # Process "must be before" constraints (this flexible course is a prerequisite)
                before_courses = constraints.get('must_be_before', [])
                for before_course in before_courses:
                    if before_course not in updated_configs:
                        continue
                    
                    # Ensure prerequisites structure exists
                    if 'prerequisites' not in updated_configs[before_course]:
                        updated_configs[before_course]['prerequisites'] = {
                            'type': 'AND',
                            'courses': []
                        }
                    elif isinstance(updated_configs[before_course]['prerequisites'], list):
                        updated_configs[before_course]['prerequisites'] = {
                            'type': 'AND',
                            'courses': []
                        }
                    
                    # Add the flexible course as a prerequisite
                    if flex_course not in updated_configs[before_course]['prerequisites']['courses']:
                        updated_configs[before_course]['prerequisites']['courses'].append(flex_course)
                        changes_made.append({
                            'course': before_course,
                            'mos': mos,
                            'prerequisite_added': flex_course,
                            'action': 'added_flexible_as_prerequisite'
                        })
    
    return updated_configs, changes_made