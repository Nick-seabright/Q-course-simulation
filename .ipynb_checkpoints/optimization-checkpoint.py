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
            - allow_mos_allocation_changes: Whether to allow MOS allocation changes
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
    allow_mos_allocation_changes = inputs.get('allow_mos_allocation_changes', True)
    
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
    mos_allocation_changes = []
    other_recommendations = []
    
    # Track optimization progress
    progress_updates = []
    
    # Extract bottlenecks from simulation results
    bottlenecks = {b['course']: b['wait_time'] for b in simulation_results['bottlenecks']}
    
    # Get courses with low utilization
    low_utilization_courses = [
        u['course'] for u in simulation_results['class_utilization']
        if u['utilization'] < 0.6
    ]
    
    # Extract MOS-specific bottlenecks if available
    mos_bottlenecks = {}
    if 'mos_metrics' in simulation_results:
        for mos, metrics in simulation_results['mos_metrics'].items():
            mos_bottlenecks[mos] = metrics.get('avg_wait_time', 0)
    
    # Optimization iterations
    for i in range(iterations):
        # Make a copy of the current best schedule and configs
        candidate_schedule = copy.deepcopy(best_schedule)
        candidate_configs = copy.deepcopy(best_course_configs)
        
        # Choose an optimization strategy based on problem characteristics
        strategy = select_optimization_strategy(i, bottlenecks, low_utilization_courses, 
                                             allow_capacity_changes, allow_duration_changes,
                                             allow_prerequisite_changes, allow_mos_allocation_changes)
        
        if strategy == 0:
            # Strategy 1: Adjust class dates for bottleneck courses
            changes = adjust_class_dates(candidate_schedule, bottlenecks)
            if changes:
                schedule_changes.extend(changes)
        
        elif strategy == 1 and allow_capacity_changes:
            # Strategy 2: Adjust class capacities
            changes = adjust_class_capacities(candidate_schedule, candidate_configs,
                                             bottlenecks, low_utilization_courses, mos_bottlenecks)
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
        
        elif strategy == 4 and allow_mos_allocation_changes:
            # Strategy 5: Adjust MOS allocations
            changes = adjust_mos_allocations(candidate_schedule, mos_bottlenecks)
            if changes:
                mos_allocation_changes.extend(changes)
        
        # Run simulation with the candidate schedule
        simulation_inputs = {
            'schedule': candidate_schedule,
            'course_configs': candidate_configs,
            'historical_data': historical_data,
            'num_students': 100,  # Use consistent number for comparison
            'num_iterations': 3    # Fewer iterations for speed during optimization
        }
        
        try:
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
                
                # Record progress
                progress_updates.append({
                    'iteration': i,
                    'strategy': strategy,
                    'improvement': candidate_score - best_score,
                    'metrics': candidate_metrics.copy()
                })
        except Exception as e:
            print(f"Error during simulation in optimization iteration {i}: {e}")
            # Continue to next iteration
            continue
    
    # Run final simulation with best schedule
    final_simulation_inputs = {
        'schedule': best_schedule,
        'course_configs': best_course_configs,
        'historical_data': historical_data,
        'num_students': 100,
        'num_iterations': 5
    }
    
    try:
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
    except Exception as e:
        print(f"Error during final simulation in optimization: {e}")
        # If final simulation fails, use best metrics from iterations
        metrics_comparison = {
            'completion_time': {
                'original': simulation_results['avg_completion_time'],
                'optimized': best_metrics['completion_time'],
                'improvement': simulation_results['avg_completion_time'] - best_metrics['completion_time']
            },
            'wait_time': {
                'original': simulation_results['avg_wait_time'],
                'optimized': best_metrics['wait_time'],
                'improvement': simulation_results['avg_wait_time'] - best_metrics['wait_time']
            },
            'throughput': {
                'original': simulation_results['throughput'],
                'optimized': best_metrics['throughput'],
                'improvement': best_metrics['throughput'] - simulation_results['throughput']
            },
            'utilization': {
                'original': simulation_results['resource_utilization'],
                'optimized': best_metrics['utilization'],
                'improvement': best_metrics['utilization'] - simulation_results['resource_utilization']
            }
        }
        final_results = {
            'avg_completion_time': best_metrics['completion_time'],
            'avg_wait_time': best_metrics['wait_time'],
            'throughput': best_metrics['throughput'],
            'resource_utilization': best_metrics['utilization'],
            'bottlenecks': simulation_results['bottlenecks'],  # Use original bottlenecks
            'class_utilization': simulation_results['class_utilization']  # Use original utilization
        }
    
    # Deduplicate and clean up changes
    schedule_changes = deduplicate_changes(schedule_changes)
    capacity_changes = deduplicate_changes(capacity_changes)
    prerequisite_changes = deduplicate_changes(prerequisite_changes)
    mos_allocation_changes = deduplicate_changes(mos_allocation_changes)
    other_recommendations = list(set(other_recommendations))
    
    return {
        'optimized_schedule': best_schedule,
        'optimized_configs': best_course_configs,
        'metrics_comparison': metrics_comparison,
        'recommended_changes': {
            'schedule_changes': schedule_changes,
            'capacity_changes': capacity_changes,
            'prerequisite_changes': prerequisite_changes,
            'mos_allocation_changes': mos_allocation_changes,
            'other_recommendations': other_recommendations
        },
        'progress_updates': progress_updates
    }

def select_optimization_strategy(iteration, bottlenecks, low_utilization_courses, 
                               allow_capacity_changes, allow_duration_changes, 
                               allow_prerequisite_changes, allow_mos_allocation_changes):
    """
    Choose optimization strategy based on problem characteristics
    Args:
        iteration (int): Current iteration
        bottlenecks (dict): Dictionary of bottleneck courses with wait times
        low_utilization_courses (list): List of courses with low utilization
        allow_capacity_changes (bool): Whether capacity changes are allowed
        allow_duration_changes (bool): Whether duration changes are allowed
        allow_prerequisite_changes (bool): Whether prerequisite changes are allowed
        allow_mos_allocation_changes (bool): Whether MOS allocation changes are allowed
    Returns:
        int: Strategy index to use
    """
    # If severe bottlenecks exist, prioritize addressing them
    severe_bottlenecks = [course for course, wait_time in bottlenecks.items() if wait_time > 20]
    if severe_bottlenecks and iteration < 3:
        return 0  # Adjust class dates
    
    # If many low utilization courses, prioritize capacity adjustments
    if len(low_utilization_courses) > len(bottlenecks) and allow_capacity_changes and iteration < 5:
        return 1  # Adjust capacities
    
    # Otherwise use round-robin with allowed strategies
    allowed_strategies = [0]  # Always allow date adjustments
    if allow_capacity_changes:
        allowed_strategies.append(1)
    allowed_strategies.append(2)  # Always allow frequency adjustments
    if allow_prerequisite_changes:
        allowed_strategies.append(3)
    if allow_mos_allocation_changes:
        allowed_strategies.append(4)
    
    return allowed_strategies[iteration % len(allowed_strategies)]

def adjust_class_dates(schedule, bottlenecks):
    """Adjust class dates to reduce bottlenecks"""
    changes = []
    
    # Convert schedule to DataFrame for easier manipulation
    schedule_df = pd.DataFrame(schedule)
    
    # Handle empty schedule
    if schedule_df.empty:
        return changes
    
    # Convert date strings to datetime
    try:
        schedule_df['start_date'] = pd.to_datetime(schedule_df['start_date'])
        schedule_df['end_date'] = pd.to_datetime(schedule_df['end_date'])
    except Exception as e:
        print(f"Error converting dates in adjust_class_dates: {e}")
        return changes
    
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
        
        # Copy MOS allocation if available
        if 'mos_allocation' in course_classes.iloc[0]:
            new_class['mos_allocation'] = course_classes.iloc[0]['mos_allocation']
        
        schedule.append(new_class)
    
    return changes

def adjust_class_capacities(schedule, course_configs, bottlenecks, low_utilization_courses, mos_bottlenecks=None):
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
        
        # Adjust MOS allocation proportionally if available
        for class_info in course_classes:
            if 'mos_allocation' in class_info and class_info['mos_allocation']:
                original_allocation = class_info['mos_allocation'].copy()
                original_total = sum(original_allocation.values())
                
                if original_total > 0:
                    # Scale each MOS allocation proportionally to the new capacity
                    new_allocation = {}
                    for mos, count in original_allocation.items():
                        # If this MOS has high wait times, increase its allocation more
                        mos_wait_time = mos_bottlenecks.get(mos, 0) if mos_bottlenecks else 0
                        mos_increase_pct = min(0.4, (wait_time + mos_wait_time) / 100)
                        mos_scale = 1 + mos_increase_pct
                        new_count = int(count * mos_scale)
                        new_allocation[mos] = new_count
                    
                    # Adjust to match new total capacity
                    new_total = sum(new_allocation.values())
                    if new_total > 0:
                        scale_factor = new_capacity / new_total
                        for mos in new_allocation:
                            new_allocation[mos] = int(new_allocation[mos] * scale_factor)
                        
                        # Assign any remaining seats
                        remaining = new_capacity - sum(new_allocation.values())
                        if remaining > 0:
                            # Distribute remaining seats to MOS with highest wait times
                            mos_wait_sorted = sorted(
                                [(mos, mos_bottlenecks.get(mos, 0)) for mos in new_allocation],
                                key=lambda x: x[1], reverse=True
                            )
                            for i in range(remaining):
                                if i < len(mos_wait_sorted):
                                    new_allocation[mos_wait_sorted[i][0]] += 1
                            
                            class_info['mos_allocation'] = new_allocation
            
            # Update class size
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
        
        # Adjust MOS allocation proportionally if available
        for class_info in course_classes:
            if 'mos_allocation' in class_info and class_info['mos_allocation']:
                original_allocation = class_info['mos_allocation'].copy()
                original_total = sum(original_allocation.values())
                
                if original_total > 0:
                    # Scale each MOS allocation proportionally to the new capacity
                    new_allocation = {}
                    for mos, count in original_allocation.items():
                        new_count = int(count * (new_capacity / current_capacity))
                        new_allocation[mos] = new_count
                    
                    # Adjust to match new total capacity
                    new_total = sum(new_allocation.values())
                    if new_total > 0:
                        scale_factor = new_capacity / new_total
                        for mos in new_allocation:
                            new_allocation[mos] = int(new_allocation[mos] * scale_factor)
                        
                        # Handle any remaining seats
                        remaining = new_capacity - sum(new_allocation.values())
                        for i in range(remaining):
                            # Add remaining seats to largest MOS allocations
                            largest_mos = max(new_allocation.items(), key=lambda x: x[1])[0]
                            new_allocation[largest_mos] += 1
                        
                        class_info['mos_allocation'] = new_allocation
            
            # Update class size
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
    
    # Handle empty schedule
    if schedule_df.empty:
        return changes
    
    # Convert date strings to datetime
    try:
        schedule_df['start_date'] = pd.to_datetime(schedule_df['start_date'])
        schedule_df['end_date'] = pd.to_datetime(schedule_df['end_date'])
    except Exception as e:
        print(f"Error converting dates in adjust_class_frequency: {e}")
        return changes
    
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
                
                # Copy MOS allocation if available
                if 'mos_allocation' in course_classes.iloc[0]:
                    new_class['mos_allocation'] = course_classes.iloc[0]['mos_allocation'].copy()
                
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
            
            # Copy MOS allocation if available
            if 'mos_allocation' in course_classes.iloc[0]:
                new_class['mos_allocation'] = course_classes.iloc[0]['mos_allocation'].copy()
            
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
        if 'prerequisites' in course_configs[course]:
            prerequisites = course_configs[course]['prerequisites'].get('courses', [])
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
                course_configs[course]['prerequisites']['courses'] = new_prerequisites
                
                # Only remove one prerequisite per course
                break
        
        # Check if course has MOS-specific prerequisites that could be optimized
        if 'mos_paths' in course_configs[course]:
            mos_paths = course_configs[course]['mos_paths']
            for mos, prereqs in mos_paths.items():
                if len(prereqs) <= 1:
                    continue
                
                # Identify MOS-specific prerequisites that might be optional
                for prereq in prereqs:
                    # Record change
                    changes.append({
                        'course': course,
                        'mos': mos,
                        'original_mos_prerequisites': prereqs.copy(),
                        'removed_prerequisite': prereq,
                        'reason': f"Consider making {prereq} optional for {mos} students taking {course} to reduce bottleneck"
                    })
                    
                    # Create a copy of prerequisites without this one
                    new_prerequisites = [p for p in prereqs if p != prereq]
                    
                    # Update course config
                    course_configs[course]['mos_paths'][mos] = new_prerequisites
                    
                    # Only remove one prerequisite per MOS path
                    break
    
    return changes

def adjust_mos_allocations(schedule, mos_bottlenecks):
    """Adjust MOS allocations to optimize throughput for bottlenecked MOS paths"""
    changes = []
    
    # Skip if no MOS bottlenecks data
    if not mos_bottlenecks:
        return changes
    
    # Sort MOS by wait time (highest first)
    sorted_mos = sorted(mos_bottlenecks.items(), key=lambda x: x[1], reverse=True)
    
    # Identify which MOS paths have the highest bottlenecks
    bottleneck_mos = [mos for mos, wait_time in sorted_mos if wait_time > 10]
    
    # Skip if no significant MOS bottlenecks
    if not bottleneck_mos:
        return changes
    
    # Process each class with MOS allocations
    for class_info in schedule:
        if 'mos_allocation' not in class_info or not class_info['mos_allocation']:
            continue
        
        original_allocation = class_info['mos_allocation'].copy()
        total_size = class_info['size']
        
        # Skip if class is small or has no allocation
        if total_size < 10 or sum(original_allocation.values()) == 0:
            continue
        
        # Determine if this class can help bottlenecked MOS paths
        relevant_mos = [mos for mos in bottleneck_mos if mos in original_allocation]
        if not relevant_mos:
            continue
        
        # Create a new allocation that increases seats for bottlenecked MOS paths
        new_allocation = original_allocation.copy()
        
        # Increase allocation for bottlenecked MOS by up to 30%
        # and decrease allocation for non-bottlenecked MOS
        for mos, wait_time in sorted_mos:
            if mos not in original_allocation:
                continue
            
            current_allocation = original_allocation[mos]
            if mos in bottleneck_mos:
                # Increase by 10-30% based on wait time
                increase_pct = min(0.3, wait_time / 50)
                new_allocation[mos] = int(current_allocation * (1 + increase_pct))
            else:
                # Decrease non-bottlenecked MOS, but ensure at least 1 seat
                new_allocation[mos] = max(1, int(current_allocation * 0.9))
        
        # Adjust to ensure total seats match class size
        new_total = sum(new_allocation.values())
        if new_total > total_size:
            # If over capacity, reduce non-bottlenecked MOS first
            excess = new_total - total_size
            
            # Sort non-bottlenecked MOS by wait time (lowest first)
            non_bottleneck_mos = [(mos, count) for mos, count in new_allocation.items()
                                 if mos not in bottleneck_mos]
            non_bottleneck_mos.sort(key=lambda x: mos_bottlenecks.get(x[0], 0))
            
            # Reduce seats for non-bottlenecked MOS
            for mos, count in non_bottleneck_mos:
                if excess <= 0:
                    break
                
                # Reduce this MOS allocation, ensuring at least 1 seat
                reduction = min(excess, count - 1)
                if reduction > 0:
                    new_allocation[mos] -= reduction
                    excess -= reduction
            
            # If still over capacity, reduce bottlenecked MOS starting with lowest wait time
            if excess > 0:
                bottleneck_mos_list = [(mos, wait_time) for mos, wait_time in sorted_mos
                                       if mos in bottleneck_mos and mos in new_allocation]
                
                # Sort by wait time (lowest first)
                bottleneck_mos_list.sort(key=lambda x: x[1])
                
                for mos, _ in bottleneck_mos_list:
                    if excess <= 0:
                        break
                    
                    # Reduce this MOS allocation, ensuring at least 1 seat
                    reduction = min(excess, new_allocation[mos] - 1)
                    if reduction > 0:
                        new_allocation[mos] -= reduction
                        excess -= reduction
        
        elif new_total < total_size:
            # If under capacity, add seats to bottlenecked MOS based on wait time
            remaining = total_size - new_total
            
            # Calculate total wait time for bottlenecked MOS
            total_wait_time = sum(mos_bottlenecks.get(mos, 0) for mos in bottleneck_mos
                                 if mos in new_allocation)
            
            if total_wait_time > 0:
                # Distribute remaining seats proportionally to wait time
                for mos in bottleneck_mos:
                    if mos not in new_allocation:
                        continue
                    
                    wait_time = mos_bottlenecks.get(mos, 0)
                    proportion = wait_time / total_wait_time
                    
                    # Add seats based on proportion of total wait time
                    seats_to_add = max(1, int(remaining * proportion))
                    new_allocation[mos] += min(seats_to_add, remaining)
                    remaining -= seats_to_add
                    
                    if remaining <= 0:
                        break
            
            # If still under capacity, add remaining seats to any MOS
            if remaining > 0:
                for mos in sorted(new_allocation.keys()):
                    new_allocation[mos] += 1
                    remaining -= 1
                    
                    if remaining <= 0:
                        break
        
        # Check if the allocation has changed significantly
        changed = False
        change_details = {}
        
        for mos in set(original_allocation.keys()) | set(new_allocation.keys()):
            original = original_allocation.get(mos, 0)
            new = new_allocation.get(mos, 0)
            
            if abs(original - new) >= 2 or (original > 0 and abs(original - new) / original > 0.1):
                changed = True
                change_details[f'original_{mos}'] = original
                change_details[f'recommended_{mos}'] = new
        
        if changed:
            # Record the change
            change = {
                'course': class_info['course_title'],
                'class_id': class_info['id'],
                'original_size': class_info['size'],
                'recommended_size': class_info['size'],
                'reason': "Rebalanced MOS allocation to reduce bottlenecks"
            }
            
            # Add all MOS allocation details
            change.update(change_details)
            changes.append(change)
            
            # Update the class allocation
            class_info['mos_allocation'] = new_allocation
    
    return changes

def deduplicate_changes(changes):
    """Remove duplicate changes"""
    seen = set()
    unique_changes = []
    
    for change in changes:
        # Create a hashable representation of the change
        if isinstance(change, dict):
            # Convert dictionary to a tuple of (key, value) pairs
            change_items = sorted((str(k), str(v)) for k, v in change.items())
            change_tuple = tuple(change_items)
        else:
            change_tuple = str(change)
        
        if change_tuple not in seen:
            seen.add(change_tuple)
            unique_changes.append(change)
    
    return unique_changes

def validate_schedule(schedule, course_configs):
    """
    Validate a schedule against course configurations and constraints
    Args:
        schedule (list): List of class dictionaries
        course_configs (dict): Dictionary of course configurations
    Returns:
        dict: Validation results with issues and warnings
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
    
    # Convert schedule to DataFrame
    try:
        schedule_df = pd.DataFrame(schedule)
        schedule_df['start_date'] = pd.to_datetime(schedule_df['start_date'])
        schedule_df['end_date'] = pd.to_datetime(schedule_df['end_date'])
    except Exception as e:
        return {
            'valid': False,
            'issues': [f"Error processing schedule data: {e}"],
            'warnings': []
        }
    
    # Check for invalid date ranges
    invalid_dates = schedule_df[schedule_df['start_date'] >= schedule_df['end_date']]
    for _, row in invalid_dates.iterrows():
        issues.append(f"Invalid date range for {row['course_title']}: start date is on or after end date")
    
    # Check for class overlaps within same course
    for course, group in schedule_df.groupby('course_title'):
        if len(group) > 1:
            # Sort by start date
            sorted_classes = group.sort_values('start_date')
            for i in range(len(sorted_classes) - 1):
                current_end = sorted_classes.iloc[i]['end_date']
                next_start = sorted_classes.iloc[i+1]['start_date']
                if current_end >= next_start:
                    issues.append(
                        f"Class overlap in {course}: Class ending {current_end.strftime('%Y-%m-%d')} " +
                        f"overlaps with class starting {next_start.strftime('%Y-%m-%d')}"
                    )
    
    # Check prerequisites
    for course, config in course_configs.items():
        # Get all classes for this course
        course_classes = schedule_df[schedule_df['course_title'] == course]
        if course_classes.empty:
            continue
        
        # Get prerequisites
        if 'prerequisites' in config and isinstance(config['prerequisites'], dict):
            prereqs = config['prerequisites'].get('courses', [])
            # Check if prerequisites are scheduled before this course
            for prereq in prereqs:
                prereq_classes = schedule_df[schedule_df['course_title'] == prereq]
                if prereq_classes.empty:
                    warnings.append(f"Prerequisite {prereq} for {course} is not scheduled")
                    continue
                
                # For each class of this course, check if there's a prerequisite class that ends before it starts
                for _, class_row in course_classes.iterrows():
                    class_start = class_row['start_date']
                    valid_prereq_classes = prereq_classes[prereq_classes['end_date'] < class_start]
                    if valid_prereq_classes.empty:
                        warnings.append(
                            f"No {prereq} class ends before {course} class starting on {class_start.strftime('%Y-%m-%d')}"
                        )
    
    # Check MOS allocations
    for _, row in schedule_df.iterrows():
        if 'mos_allocation' in row and row['mos_allocation']:
            mos_allocation = row['mos_allocation']
            total_allocation = sum(mos_allocation.values())
            if total_allocation > row['size']:
                issues.append(
                    f"{row['course_title']} starting {row['start_date'].strftime('%Y-%m-%d')} has total MOS allocation " +
                    f"{total_allocation}, but class size is {row['size']}"
                )
    
    return {
        'valid': len(issues) == 0,
        'issues': issues,
        'warnings': warnings
    }

def safe_run_simulation(simulation_inputs):
    """
    Safely run simulation with error handling
    Args:
        simulation_inputs (dict): Simulation inputs
    Returns:
        dict: Simulation results or default results if error occurs
    """
    try:
        return run_simulation(simulation_inputs)
    except Exception as e:
        print(f"Simulation error during optimization: {e}")
        # Return a default result structure that won't crash the optimizer
        return {
            'avg_completion_time': float('inf'),
            'avg_wait_time': float('inf'),
            'throughput': 0,
            'resource_utilization': 0,
            'bottlenecks': [],
            'class_utilization': []
        }