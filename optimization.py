import pandas as pd
import numpy as np
import copy
import datetime
from collections import defaultdict
from simulation_engine import run_simulation

def optimize_schedule(inputs):
    """
    Optimize training schedule based on simulation results
    
    Args:
        inputs (dict): Optimization inputs
            - current_schedule: Current schedule
            - course_configs: Course configurations
            - simulation_results: Current simulation results
            - historical_data: Historical training data
            - optimization_goal: Primary optimization goal
            - iterations: Number of optimization iterations
            - constraint_weight: Weight for constraint satisfaction
            - allow_capacity_changes: Whether to allow capacity changes
            - allow_duration_changes: Whether to allow duration changes
            - allow_prerequisite_changes: Whether to allow prerequisite changes
    
    Returns:
        dict: Optimization results
    """
    # Extract inputs
    current_schedule = inputs['current_schedule']
    course_configs = inputs['course_configs']
    simulation_results = inputs['simulation_results']
    historical_data = inputs['historical_data']
    optimization_goal = inputs['optimization_goal']
    iterations = inputs['iterations']
    constraint_weight = inputs['constraint_weight']
    allow_capacity_changes = inputs['allow_capacity_changes']
    allow_duration_changes = inputs['allow_duration_changes']
    allow_prerequisite_changes = inputs['allow_prerequisite_changes']
    
    # Make a copy of the current schedule for optimization
    best_schedule = copy.deepcopy(current_schedule)
    best_course_configs = copy.deepcopy(course_configs)
    
    # Current best metrics
    best_metrics = {
        'completion_time': simulation_results['avg_completion_time'],
        'wait_time': simulation_results['avg_wait_time'],
        'throughput': simulation_results['throughput'],
        'utilization': simulation_results['resource_utilization']
    }
    
    # Define optimization functions based on goal
    if optimization_goal == "Minimize Average Completion Time":
        optimization_fn = lambda metrics: -metrics['completion_time']
    elif optimization_goal == "Maximize Student Throughput":
        optimization_fn = lambda metrics: metrics['throughput']
    elif optimization_goal == "Minimize Wait Times":
        optimization_fn = lambda metrics: -metrics['wait_time']
    elif optimization_goal == "Maximize Resource Utilization":
        optimization_fn = lambda metrics: metrics['utilization']
    else:
        # Default to a balanced approach
        optimization_fn = lambda metrics: (
            -0.3 * metrics['completion_time'] + 
            0.3 * metrics['throughput'] - 
            0.2 * metrics['wait_time'] + 
            0.2 * metrics['utilization']
        )
    
    best_score = optimization_fn(best_metrics)
    
    # Track changes made
    schedule_changes = []
    capacity_changes = []
    prerequisite_changes = []
    other_recommendations = []
    
    # Extract bottlenecks from simulation results
    bottlenecks = {b['course']: b['wait_time'] for b in simulation_results['bottlenecks']}
    
    # Get courses with low utilization
    low_utilization_courses = [
        u['course'] for u in simulation_results['class_utilization']
        if u['utilization'] < 0.6
    ]
    
    # Optimization iterations
    for i in range(iterations):
        # Make a copy of the current best schedule and configs
        candidate_schedule = copy.deepcopy(best_schedule)
        candidate_configs = copy.deepcopy(best_course_configs)
        
        # Choose an optimization strategy based on iteration
        strategy = i % 4
        
        if strategy == 0:
            # Strategy 1: Adjust class dates for bottleneck courses
            changes = adjust_class_dates(candidate_schedule, bottlenecks)
            if changes:
                schedule_changes.extend(changes)
        
        elif strategy == 1 and allow_capacity_changes:
            # Strategy 2: Adjust class capacities
            changes = adjust_class_capacities(candidate_schedule, candidate_configs, 
                                             bottlenecks, low_utilization_courses)
            if changes:
                capacity_changes.extend(changes)
        
        elif strategy == 2:
            # Strategy 3: Add or remove classes
            changes = adjust_class_frequency(candidate_schedule, bottlenecks, low_utilization_courses)
            if changes:
                schedule_changes.extend(changes)
                
                # Update other recommendations
                for change in changes:
                    if change.get('action') == 'add':
                        other_recommendations.append(
                            f"Add a {change['course']} class starting on {change['new_start']} to reduce wait times."
                        )
                    elif change.get('action') == 'remove':
                        other_recommendations.append(
                            f"Remove the {change['course']} class starting on {change['original_start']} due to low utilization."
                        )
        
        elif strategy == 3 and allow_prerequisite_changes:
            # Strategy 4: Adjust prerequisites
            changes = adjust_prerequisites(candidate_configs, bottlenecks)
            if changes:
                prerequisite_changes.extend(changes)
        
        # Run simulation with the candidate schedule
        simulation_inputs = {
            'schedule': candidate_schedule,
            'course_configs': candidate_configs,
            'historical_data': historical_data,
            'num_students': 100,  # Use consistent number for comparison
            'num_iterations': 3    # Fewer iterations for speed during optimization
        }
        
        candidate_results = run_simulation(simulation_inputs)
        
        # Extract metrics
        candidate_metrics = {
            'completion_time': candidate_results['avg_completion_time'],
            'wait_time': candidate_results['avg_wait_time'],
            'throughput': candidate_results['throughput'],
            'utilization': candidate_results['resource_utilization']
        }
        
        # Calculate score
        candidate_score = optimization_fn(candidate_metrics)
        
        # Check if candidate is better
        if candidate_score > best_score:
            best_schedule = candidate_schedule
            best_course_configs = candidate_configs
            best_metrics = candidate_metrics
            best_score = candidate_score
    
    # Run final simulation with best schedule
    final_simulation_inputs = {
        'schedule': best_schedule,
        'course_configs': best_course_configs,
        'historical_data': historical_data,
        'num_students': 100,
        'num_iterations': 5
    }
    
    final_results = run_simulation(final_simulation_inputs)
    
    # Calculate metrics improvements
    metrics_comparison = {
        'completion_time': {
            'original': simulation_results['avg_completion_time'],
            'optimized': final_results['avg_completion_time'],
            'improvement': simulation_results['avg_completion_time'] - final_results['avg_completion_time']
        },
        'wait_time': {
            'original': simulation_results['avg_wait_time'],
            'optimized': final_results['avg_wait_time'],
            'improvement': simulation_results['avg_wait_time'] - final_results['avg_wait_time']
        },
        'throughput': {
            'original': simulation_results['throughput'],
            'optimized': final_results['throughput'],
            'improvement': final_results['throughput'] - simulation_results['throughput']
        },
        'utilization': {
            'original': simulation_results['resource_utilization'],
            'optimized': final_results['resource_utilization'],
            'improvement': final_results['resource_utilization'] - simulation_results['resource_utilization']
        }
    }
    
    # Deduplicate and clean up changes
    schedule_changes = deduplicate_changes(schedule_changes)
    capacity_changes = deduplicate_changes(capacity_changes)
    prerequisite_changes = deduplicate_changes(prerequisite_changes)
    other_recommendations = list(set(other_recommendations))
    
    return {
        'optimized_schedule': best_schedule,
        'optimized_configs': best_course_configs,
        'metrics_comparison': metrics_comparison,
        'recommended_changes': {
            'schedule_changes': schedule_changes,
            'capacity_changes': capacity_changes,
            'prerequisite_changes': prerequisite_changes,
            'other_recommendations': other_recommendations
        }
    }

def adjust_class_dates(schedule, bottlenecks):
    """Adjust class dates to reduce bottlenecks"""
    changes = []
    
    # Convert schedule to DataFrame for easier manipulation
    schedule_df = pd.DataFrame(schedule)
    
    # Convert date strings to datetime
    schedule_df['start_date'] = pd.to_datetime(schedule_df['start_date'])
    schedule_df['end_date'] = pd.to_datetime(schedule_df['end_date'])
    
    # Sort by bottleneck severity
    bottleneck_courses = sorted(bottlenecks.items(), key=lambda x: x[1], reverse=True)
    
    for course, wait_time in bottleneck_courses:
        if wait_time < 5:  # Skip if wait time is less than 5 days
            continue
        
        # Get classes for this course
        course_classes = schedule_df[schedule_df['course_title'] == course].sort_values('start_date')
        
        if len(course_classes) < 2:
            continue
        
        # Try to add a class in the middle of the longest gap
        class_gaps = []
        for i in range(len(course_classes) - 1):
            gap_start = course_classes.iloc[i]['end_date']
            gap_end = course_classes.iloc[i+1]['start_date']
            gap_duration = (gap_end - gap_start).days
            
            if gap_duration > 30:  # Only consider gaps > 30 days
                class_gaps.append({
                    'start_idx': i,
                    'end_idx': i+1,
                    'gap_start': gap_start,
                    'gap_end': gap_end,
                    'duration': gap_duration
                })
        
        if not class_gaps:
            continue
        
        # Find the largest gap
        largest_gap = max(class_gaps, key=lambda x: x['duration'])
        
        # Calculate new class dates
        new_start = largest_gap['gap_start'] + pd.Timedelta(days=largest_gap['duration'] // 2)
        
        # Get duration from a typical class
        typical_duration = (course_classes.iloc[0]['end_date'] - course_classes.iloc[0]['start_date']).days
        new_end = new_start + pd.Timedelta(days=typical_duration)
        
        # Add change
        changes.append({
            'action': 'add',
            'course': course,
            'new_start': new_start.strftime('%Y-%m-%d'),
            'new_end': new_end.strftime('%Y-%m-%d'),
            'reason': f"Adding class to reduce {wait_time:.1f} day wait time"
        })
        
        # Update schedule
        new_class = {
            'course_title': course,
            'start_date': new_start.strftime('%Y-%m-%d'),
            'end_date': new_end.strftime('%Y-%m-%d'),
            'size': course_classes.iloc[0]['size'],
            'id': max([c['id'] for c in schedule]) + 1
        }
        
        schedule.append(new_class)
    
    return changes

def adjust_class_capacities(schedule, course_configs, bottlenecks, low_utilization_courses):
    """Adjust class capacities to optimize throughput"""
    changes = []
    
    # Increase capacity for bottleneck courses
    for course, wait_time in bottlenecks.items():
        if wait_time < 5:  # Skip if wait time is less than 5 days
            continue
        
        # Find classes for this course
        course_classes = [c for c in schedule if c['course_title'] == course]
        
        if not course_classes:
            continue
        
        # Get current capacity
        current_capacity = course_classes[0]['size']
        
        # Calculate new capacity (increase by 10-30% based on wait time)
        increase_pct = min(0.3, wait_time / 100)
        new_capacity = int(current_capacity * (1 + increase_pct))
        
        # Make sure it's at least 5 more
        new_capacity = max(new_capacity, current_capacity + 5)
        
        # Cap at a reasonable maximum
        new_capacity = min(new_capacity, 250)
        
        # Update all classes for this course
        for class_info in course_classes:
            class_info['size'] = new_capacity
        
        # Update course config
        if course in course_configs:
            original_capacity = course_configs[course].get('max_capacity', current_capacity)
            course_configs[course]['max_capacity'] = new_capacity
        else:
            original_capacity = current_capacity
        
        # Record change
        changes.append({
            'course': course,
            'original_capacity': original_capacity,
            'recommended_capacity': new_capacity,
            'reason': f"Increased capacity to reduce {wait_time:.1f} day wait time"
        })
    
    # Decrease capacity for low utilization courses
    for course in low_utilization_courses:
        # Skip if also a bottleneck course (don't decrease capacity for bottlenecks)
        if course in bottlenecks and bottlenecks[course] > 5:
            continue
        
        # Find classes for this course
        course_classes = [c for c in schedule if c['course_title'] == course]
        
        if not course_classes:
            continue
        
        # Get current capacity
        current_capacity = course_classes[0]['size']
        
        # Reduce capacity by 10-20%
        new_capacity = int(current_capacity * 0.85)
        
        # Make sure it's at least 5 less
        new_capacity = min(new_capacity, current_capacity - 5)
        
        # Ensure minimum reasonable class size
        new_capacity = max(new_capacity, 10)
        
        # Update all classes for this course
        for class_info in course_classes:
            class_info['size'] = new_capacity
        
        # Update course config
        if course in course_configs:
            original_capacity = course_configs[course].get('max_capacity', current_capacity)
            course_configs[course]['max_capacity'] = new_capacity
        else:
            original_capacity = current_capacity
        
        # Record change
        changes.append({
            'course': course,
            'original_capacity': original_capacity,
            'recommended_capacity': new_capacity,
            'reason': "Decreased capacity due to low utilization"
        })
    
    return changes

def adjust_class_frequency(schedule, bottlenecks, low_utilization_courses):
    """Add or remove classes to optimize throughput"""
    changes = []
    
    # Convert schedule to DataFrame for easier manipulation
    schedule_df = pd.DataFrame(schedule)
    
    # Convert date strings to datetime
    schedule_df['start_date'] = pd.to_datetime(schedule_df['start_date'])
    schedule_df['end_date'] = pd.to_datetime(schedule_df['end_date'])
    
    # Add classes for bottleneck courses with high wait times
    for course, wait_time in bottlenecks.items():
        if wait_time < 10:  # Only consider severe bottlenecks
            continue
        
        # Get classes for this course
        course_classes = schedule_df[schedule_df['course_title'] == course].sort_values('start_date')
        
        if len(course_classes) < 1:
            continue
        
        # Get the typical class duration
        typical_duration = (course_classes.iloc[0]['end_date'] - course_classes.iloc[0]['start_date']).days
        
        # Determine where to add a new class
        if len(course_classes) >= 2:
            # Find largest gap between classes
            largest_gap = 0
            largest_gap_start = None
            
            for i in range(len(course_classes) - 1):
                gap = (course_classes.iloc[i+1]['start_date'] - course_classes.iloc[i]['end_date']).days
                if gap > largest_gap:
                    largest_gap = gap
                    largest_gap_start = course_classes.iloc[i]['end_date']
            
            if largest_gap > typical_duration * 1.5:
                # Add a class in the middle of the gap
                new_start = largest_gap_start + pd.Timedelta(days=largest_gap // 2)
                new_end = new_start + pd.Timedelta(days=typical_duration)
                
                # Record change
                changes.append({
                    'action': 'add',
                    'course': course,
                    'new_start': new_start.strftime('%Y-%m-%d'),
                    'new_end': new_end.strftime('%Y-%m-%d'),
                    'reason': f"Added class to reduce {wait_time:.1f} day wait time"
                })
                
                # Add to schedule
                new_class = {
                    'course_title': course,
                    'start_date': new_start.strftime('%Y-%m-%d'),
                    'end_date': new_end.strftime('%Y-%m-%d'),
                    'size': course_classes.iloc[0]['size'],
                    'id': max([c['id'] for c in schedule]) + 1
                }
                
                schedule.append(new_class)
        else:
            # Only one class exists, add another one after a reasonable gap
            last_class_end = course_classes.iloc[-1]['end_date']
            new_start = last_class_end + pd.Timedelta(days=30)  # Add a month after the last class
            new_end = new_start + pd.Timedelta(days=typical_duration)
            
            # Record change
            changes.append({
                'action': 'add',
                'course': course,
                'new_start': new_start.strftime('%Y-%m-%d'),
                'new_end': new_end.strftime('%Y-%m-%d'),
                'reason': f"Added class to reduce {wait_time:.1f} day wait time"
            })
            
            # Add to schedule
            new_class = {
                'course_title': course,
                'start_date': new_start.strftime('%Y-%m-%d'),
                'end_date': new_end.strftime('%Y-%m-%d'),
                'size': course_classes.iloc[0]['size'],
                'id': max([c['id'] for c in schedule]) + 1
            }
            
            schedule.append(new_class)
    
    # Remove classes for low utilization courses
    classes_to_remove = []
    
    for course in low_utilization_courses:
        # Skip if also a bottleneck course
        if course in bottlenecks and bottlenecks[course] > 5:
            continue
        
        # Get classes for this course
        course_classes = schedule_df[schedule_df['course_title'] == course].sort_values('start_date')
        
        if len(course_classes) <= 1:
            # Don't remove if only one class exists
            continue
        
        # Consider removing every other class to reduce frequency
        for i in range(1, len(course_classes), 2):
            class_info = course_classes.iloc[i]
            
            # Record change
            changes.append({
                'action': 'remove',
                'course': course,
                'original_start': class_info['start_date'].strftime('%Y-%m-%d'),
                'original_end': class_info['end_date'].strftime('%Y-%m-%d'),
                'reason': "Removed class due to low utilization"
            })
            
            # Mark for removal
            classes_to_remove.append(class_info['id'])
    
    # Remove classes from schedule
    schedule[:] = [c for c in schedule if c['id'] not in classes_to_remove]
    
    return changes

def adjust_prerequisites(course_configs, bottlenecks):
    """Adjust course prerequisites to optimize flow"""
    changes = []
    
    # Identify bottleneck courses with long wait times
    bottleneck_courses = [course for course, wait_time in bottlenecks.items() if wait_time > 15]
    
    for course in bottleneck_courses:
        if course not in course_configs:
            continue
        
        # Check if course has many prerequisites
        prerequisites = course_configs[course].get('prerequisites', [])
        
        if len(prerequisites) <= 1:
            continue
        
        # Identify prerequisites that might be optional
        for prereq in prerequisites:
            # Record change
            changes.append({
                'course': course,
                'original_prerequisites': prerequisites.copy(),
                'removed_prerequisite': prereq,
                'reason': f"Consider making {prereq} optional for {course} to reduce bottleneck"
            })
            
            # Create a copy of prerequisites without this one
            new_prerequisites = [p for p in prerequisites if p != prereq]
            
            # Update course config
            course_configs[course]['prerequisites'] = new_prerequisites
            
            # Only remove one prerequisite per course
            break
    
    return changes

def deduplicate_changes(changes):
    """Remove duplicate changes"""
    seen = set()
    unique_changes = []
    
    for change in changes:
        # Create a hashable representation of the change
        change_tuple = tuple(sorted(change.items()))
        
        if change_tuple not in seen:
            seen.add(change_tuple)
            unique_changes.append(change)
    
    return unique_changes