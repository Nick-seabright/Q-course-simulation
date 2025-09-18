import pandas as pd
import numpy as np
import copy
import datetime
import functools
import concurrent.futures
from collections import defaultdict
from typing import Dict, List, Any, Tuple, Optional, Union
import streamlit as st

@st.cache_data(ttl=600)
def optimize_schedule(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Optimize training schedule based on simulation results
    
    Args:
        inputs: Optimization inputs
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
        Optimization results including the optimized schedule and recommended changes
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
    
    # Use parallel optimization if multiple iterations
    if iterations > 1 and allow_parallel_optimization():
        # Parallel optimization for multiple iterations
        optimized_results = run_parallel_optimization(
            current_schedule, course_configs, simulation_results, historical_data,
            optimization_goal, iterations, constraint_weight,
            allow_capacity_changes, allow_duration_changes,
            allow_prerequisite_changes, allow_mos_allocation_changes,
            bottlenecks, low_utilization_courses, mos_bottlenecks,
            optimization_fn, best_score, best_metrics
        )
        
        # Extract results from parallel optimization
        best_schedule = optimized_results['best_schedule']
        best_course_configs = optimized_results['best_course_configs']
        best_metrics = optimized_results['best_metrics']
        best_score = optimized_results['best_score']
        schedule_changes = optimized_results['schedule_changes']
        capacity_changes = optimized_results['capacity_changes'] 
        prerequisite_changes = optimized_results['prerequisite_changes']
        mos_allocation_changes = optimized_results['mos_allocation_changes']
        other_recommendations = optimized_results['other_recommendations']
        progress_updates = optimized_results['progress_updates']
    else:
        # Sequential optimization for single iteration or when parallel is disabled
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
                from simulation_engine import run_simulation
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
        from simulation_engine import run_simulation
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

def allow_parallel_optimization() -> bool:
    """
    Check if parallel optimization is enabled and available
    
    Returns:
        True if parallel optimization should be used, False otherwise
    """
    # Check if multiprocessing is available
    try:
        import concurrent.futures
        return True
    except ImportError:
        return False
    
    # Could also add system-specific checks here, e.g., CPU count or memory

def run_parallel_optimization(
    current_schedule: List[Dict[str, Any]],
    course_configs: Dict[str, Dict[str, Any]],
    simulation_results: Dict[str, Any],
    historical_data: Dict[str, Dict[str, Any]],
    optimization_goal: str,
    iterations: int,
    constraint_weight: float,
    allow_capacity_changes: bool,
    allow_duration_changes: bool,
    allow_prerequisite_changes: bool,
    allow_mos_allocation_changes: bool,
    bottlenecks: Dict[str, float],
    low_utilization_courses: List[str],
    mos_bottlenecks: Dict[str, float],
    optimization_fn,
    best_score: float,
    best_metrics: Dict[str, float]
) -> Dict[str, Any]:
    """
    Run optimization in parallel using multiple processes
    
    Args:
        current_schedule: Current training schedule
        course_configs: Course configurations
        simulation_results: Current simulation results
        historical_data: Historical training data
        optimization_goal: Primary optimization goal
        iterations: Number of optimization iterations
        constraint_weight: Weight for constraint satisfaction
        allow_capacity_changes: Whether to allow capacity changes
        allow_duration_changes: Whether to allow duration changes
        allow_prerequisite_changes: Whether to allow prerequisite changes
        allow_mos_allocation_changes: Whether to allow MOS allocation changes
        bottlenecks: Dictionary of bottleneck courses with wait times
        low_utilization_courses: List of courses with low utilization
        mos_bottlenecks: Dictionary of MOS-specific bottlenecks
        optimization_fn: Function to calculate optimization score
        best_score: Current best score
        best_metrics: Current best metrics
        
    Returns:
        Dictionary of optimization results
    """
    # Make copies for thread safety
    best_schedule = copy.deepcopy(current_schedule)
    best_course_configs = copy.deepcopy(course_configs)
    
    # Generate candidate schedules with different strategies
    candidates = []
    
    # Track changes by candidate
    all_changes = {
        'schedule_changes': [],
        'capacity_changes': [],
        'prerequisite_changes': [],
        'mos_allocation_changes': [],
        'other_recommendations': []
    }
    
    # Create candidates using different strategies
    for i in range(iterations):
        # Choose strategy based on problem characteristics and iteration
        strategy = select_optimization_strategy(i, bottlenecks, low_utilization_courses,
                                             allow_capacity_changes, allow_duration_changes,
                                             allow_prerequisite_changes, allow_mos_allocation_changes)
        
        # Apply the selected strategy to create a candidate
        candidate_schedule = copy.deepcopy(current_schedule)
        candidate_configs = copy.deepcopy(course_configs)
        changes = []
        
        if strategy == 0:
            # Strategy 1: Adjust class dates for bottleneck courses
            changes = adjust_class_dates(candidate_schedule, bottlenecks)
            if changes:
                all_changes['schedule_changes'].extend(changes)
        
        elif strategy == 1 and allow_capacity_changes:
            # Strategy 2: Adjust class capacities
            changes = adjust_class_capacities(candidate_schedule, candidate_configs,
                                            bottlenecks, low_utilization_courses, mos_bottlenecks)
            if changes:
                all_changes['capacity_changes'].extend(changes)
        
        elif strategy == 2:
            # Strategy 3: Add or remove classes
            changes = adjust_class_frequency(candidate_schedule, bottlenecks, low_utilization_courses)
            if changes:
                all_changes['schedule_changes'].extend(changes)
                # Update other recommendations
                for change in changes:
                    if change.get('action') == 'add':
                        all_changes['other_recommendations'].append(
                            f"Add a {change['course']} class starting on {change['new_start']} to reduce wait times."
                        )
                    elif change.get('action') == 'remove':
                        all_changes['other_recommendations'].append(
                            f"Remove the {change['course']} class starting on {change['original_start']} due to low utilization."
                        )
        
        elif strategy == 3 and allow_prerequisite_changes:
            # Strategy 4: Adjust prerequisites
            changes = adjust_prerequisites(candidate_configs, bottlenecks)
            if changes:
                all_changes['prerequisite_changes'].extend(changes)
        
        elif strategy == 4 and allow_mos_allocation_changes:
            # Strategy 5: Adjust MOS allocations
            changes = adjust_mos_allocations(candidate_schedule, mos_bottlenecks)
            if changes:
                all_changes['mos_allocation_changes'].extend(changes)
        
        # Add candidate to the list if changes were made
        if changes:
            candidates.append({
                'schedule': candidate_schedule,
                'configs': candidate_configs,
                'strategy': strategy,
                'iteration': i
            })
    
    # Skip parallel evaluation if no candidates
    if not candidates:
        return {
            'best_schedule': best_schedule,
            'best_course_configs': best_course_configs,
            'best_metrics': best_metrics,
            'best_score': best_score,
            'schedule_changes': all_changes['schedule_changes'],
            'capacity_changes': all_changes['capacity_changes'],
            'prerequisite_changes': all_changes['prerequisite_changes'],
            'mos_allocation_changes': all_changes['mos_allocation_changes'],
            'other_recommendations': list(set(all_changes['other_recommendations'])),
            'progress_updates': []
        }
    
    # Define function to evaluate a candidate
    def evaluate_candidate(candidate_index):
        from simulation_engine import run_simulation
        
        candidate = candidates[candidate_index]
        
        # Create simulation inputs
        simulation_inputs = {
            'schedule': candidate['schedule'],
            'course_configs': candidate['configs'],
            'historical_data': historical_data,
            'num_students': 100,  # Use consistent number for comparison
            'num_iterations': 3    # Fewer iterations for speed during optimization
        }
        
        try:
            # Run simulation
            results = run_simulation(simulation_inputs)
            
            # Extract metrics
            metrics = {
                'completion_time': results['avg_completion_time'],
                'wait_time': results['avg_wait_time'],
                'throughput': results['throughput'],
                'utilization': results['resource_utilization']
            }
            
            # Calculate score
            score = optimization_fn(metrics)
            
            return {
                'index': candidate_index,
                'score': score,
                'metrics': metrics,
                'success': True
            }
        
        except Exception as e:
            # Return failure if simulation fails
            return {
                'index': candidate_index,
                'score': float('-inf'),
                'error': str(e),
                'success': False
            }
    
    # Run candidates in parallel
    progress_updates = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Submit all candidates for evaluation
        future_to_index = {
            executor.submit(evaluate_candidate, i): i 
            for i in range(len(candidates))
        }
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_index):
            try:
                result = future.result()
                
                if result['success']:
                    candidate_index = result['index']
                    candidate_score = result['score']
                    candidate_metrics = result['metrics']
                    
                    # Check if this candidate is better than the current best
                    if candidate_score > best_score:
                        best_schedule = candidates[candidate_index]['schedule']
                        best_course_configs = candidates[candidate_index]['configs']
                        best_metrics = candidate_metrics
                        best_score = candidate_score
                        
                        # Record progress
                        progress_updates.append({
                            'iteration': candidates[candidate_index]['iteration'],
                            'strategy': candidates[candidate_index]['strategy'],
                            'improvement': candidate_score - best_score,
                            'metrics': candidate_metrics.copy()
                        })
            
            except Exception as e:
                print(f"Error processing optimization result: {e}")
    
    # Return the best schedule and all collected changes
    return {
        'best_schedule': best_schedule,
        'best_course_configs': best_course_configs,
        'best_metrics': best_metrics,
        'best_score': best_score,
        'schedule_changes': all_changes['schedule_changes'],
        'capacity_changes': all_changes['capacity_changes'],
        'prerequisite_changes': all_changes['prerequisite_changes'],
        'mos_allocation_changes': all_changes['mos_allocation_changes'],
        'other_recommendations': list(set(all_changes['other_recommendations'])),
        'progress_updates': progress_updates
    }

def select_optimization_strategy(
    iteration: int, 
    bottlenecks: Dict[str, float], 
    low_utilization_courses: List[str],
    allow_capacity_changes: bool, 
    allow_duration_changes: bool,
    allow_prerequisite_changes: bool, 
    allow_mos_allocation_changes: bool
) -> int:
    """
    Choose optimization strategy based on problem characteristics
    
    Args:
        iteration: Current iteration
        bottlenecks: Dictionary of bottleneck courses with wait times
        low_utilization_courses: List of courses with low utilization
        allow_capacity_changes: Whether capacity changes are allowed
        allow_duration_changes: Whether duration changes are allowed
        allow_prerequisite_changes: Whether prerequisite changes are allowed
        allow_mos_allocation_changes: Whether MOS allocation changes are allowed
        
    Returns:
        Strategy index to use
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

def adjust_class_dates(
    schedule: List[Dict[str, Any]], 
    bottlenecks: Dict[str, float]
) -> List[Dict[str, Any]]:
    """
    Adjust class dates to reduce bottlenecks
    
    Args:
        schedule: List of scheduled classes
        bottlenecks: Dictionary of bottleneck courses with wait times
        
    Returns:
        List of schedule changes
    """
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

def adjust_class_capacities(
    schedule: List[Dict[str, Any]], 
    course_configs: Dict[str, Dict[str, Any]], 
    bottlenecks: Dict[str, float], 
    low_utilization_courses: List[str], 
    mos_bottlenecks: Optional[Dict[str, float]] = None
) -> List[Dict[str, Any]]:
    """
    Adjust class capacities to optimize throughput
    
    Args:
        schedule: List of scheduled classes
        course_configs: Dictionary of course configurations
        bottlenecks: Dictionary of bottleneck courses with wait times
        low_utilization_courses: List of courses with low utilization
        mos_bottlenecks: Dictionary of MOS-specific bottlenecks
        
    Returns:
        List of capacity changes
    """
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

def adjust_class_frequency(
    schedule: List[Dict[str, Any]], 
    bottlenecks: Dict[str, float], 
    low_utilization_courses: List[str]
) -> List[Dict[str, Any]]:
    """
    Add or remove classes to optimize throughput
    
    Args:
        schedule: List of scheduled classes
        bottlenecks: Dictionary of bottleneck courses with wait times
        low_utilization_courses: List of courses with low utilization
        
    Returns:
        List of schedule changes
    """
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

def adjust_prerequisites(
    course_configs: Dict[str, Dict[str, Any]], 
    bottlenecks: Dict[str, float]
) -> List[Dict[str, Any]]:
    """
    Adjust course prerequisites to optimize flow
    
    Args:
        course_configs: Dictionary of course configurations
        bottlenecks: Dictionary of bottleneck courses with wait times
        
    Returns:
        List of prerequisite changes
    """
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

def adjust_mos_allocations(
    schedule: List[Dict[str, Any]], 
    mos_bottlenecks: Dict[str, float]
) -> List[Dict[str, Any]]:
    """
    Adjust MOS allocations to optimize throughput for bottlenecked MOS paths
    
    Args:
        schedule: List of scheduled classes
        mos_bottlenecks: Dictionary of MOS bottlenecks
        
    Returns:
        List of MOS allocation changes
    """
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

def deduplicate_changes(changes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Remove duplicate changes
    
    Args:
        changes: List of change dictionaries
        
    Returns:
        List of deduplicated changes
    """
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

def progressive_optimize(
    schedule: List[Dict[str, Any]], 
    course_configs: Dict[str, Dict[str, Any]], 
    bottlenecks: Dict[str, float],
    low_utilization_courses: List[str],
    mos_bottlenecks: Dict[str, float],
    optimization_fn,
    historical_data: Dict[str, Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Apply a series of optimizations progressively, keeping improvements
    
    Args:
        schedule: Current schedule to optimize
        course_configs: Course configurations
        bottlenecks: Dictionary of bottleneck courses with wait times
        low_utilization_courses: List of courses with low utilization
        mos_bottlenecks: Dictionary of MOS-specific bottlenecks
        optimization_fn: Function to calculate optimization score
        historical_data: Historical training data
        
    Returns:
        Tuple of (optimized schedule, optimized configs, all changes)
    """
    from simulation_engine import run_simulation
    
    # Start with copies of the inputs
    current_schedule = copy.deepcopy(schedule)
    current_configs = copy.deepcopy(course_configs)
    all_changes = []
    
    # Run initial simulation to get baseline
    initial_results = run_simulation({
        'schedule': current_schedule,
        'course_configs': current_configs,
        'historical_data': historical_data,
        'num_students': 100,
        'num_iterations': 3
    })
    
    current_metrics = {
        'completion_time': initial_results['avg_completion_time'],
        'wait_time': initial_results['avg_wait_time'],
        'throughput': initial_results['throughput'],
        'utilization': initial_results['resource_utilization']
    }
    
    current_score = optimization_fn(current_metrics)
    
    # Define optimization steps
    optimization_steps = [
        {
            'name': "Adjust class dates",
            'function': adjust_class_dates,
            'args': [bottlenecks],
            'change_type': 'schedule_changes'
        },
        {
            'name': "Adjust class capacities",
            'function': adjust_class_capacities,
            'args': [current_configs, bottlenecks, low_utilization_courses, mos_bottlenecks],
            'change_type': 'capacity_changes'
        },
        {
            'name': "Adjust class frequency",
            'function': adjust_class_frequency,
            'args': [bottlenecks, low_utilization_courses],
            'change_type': 'schedule_changes'
        },
        {
            'name': "Adjust MOS allocations",
            'function': adjust_mos_allocations,
            'args': [mos_bottlenecks],
            'change_type': 'mos_allocation_changes'
        }
    ]
    
    # Apply each optimization step in sequence
    for step in optimization_steps:
        print(f"Applying optimization step: {step['name']}")
        
        # Create a copy of the current schedule
        candidate_schedule = copy.deepcopy(current_schedule)
        candidate_configs = copy.deepcopy(current_configs)
        
        # Apply the optimization function
        changes = step['function'](candidate_schedule, *step['args'])
        
        if not changes:
            print(f"  No changes from {step['name']}")
            continue
        
        # Run simulation with the candidate
        try:
            candidate_results = run_simulation({
                'schedule': candidate_schedule,
                'course_configs': candidate_configs,
                'historical_data': historical_data,
                'num_students': 100,
                'num_iterations': 3
            })
            
            # Extract metrics
            candidate_metrics = {
                'completion_time': candidate_results['avg_completion_time'],
                'wait_time': candidate_results['avg_wait_time'],
                'throughput': candidate_results['throughput'],
                'utilization': candidate_results['resource_utilization']
            }
            
            # Calculate score
            candidate_score = optimization_fn(candidate_metrics)
            
            # Check if this is an improvement
            if candidate_score > current_score:
                print(f"  Improved score from {current_score:.4f} to {candidate_score:.4f}")
                
                # Update current best
                current_schedule = candidate_schedule
                current_configs = candidate_configs
                current_score = candidate_score
                current_metrics = candidate_metrics
                
                # Store the changes
                all_changes.extend(changes)
            else:
                print(f"  No improvement: {current_score:.4f} vs {candidate_score:.4f}")
        
        except Exception as e:
            print(f"  Error in simulation: {e}")
    
    return current_schedule, current_configs, all_changes

def optimize_mos_allocation(
    class_info: Dict[str, Any], 
    mos_bottlenecks: Dict[str, float], 
    mos_completion_rates: Dict[str, float]
) -> Dict[str, int]:
    """
    Optimize MOS allocation based on bottlenecks and historical completion rates
    
    Args:
        class_info: Class information including current MOS allocation
        mos_bottlenecks: Dictionary of MOS-specific bottlenecks
        mos_completion_rates: Dictionary of MOS completion rates
        
    Returns:
        Optimized MOS allocation
    """
    # Get current allocation
    current_allocation = class_info.get('mos_allocation', {})
    
    if not current_allocation:
        return {}
    
    # Get class size
    class_size = class_info.get('size', 0)
    
    if class_size <= 0:
        return current_allocation
    
    # Create a new allocation
    new_allocation = current_allocation.copy()
    
    # Adjust allocation based on bottlenecks and completion rates
    total_factor = 0
    
    for mos in new_allocation:
        # Calculate an importance factor based on bottlenecks and completion rates
        bottleneck_factor = mos_bottlenecks.get(mos, 0) / 10  # Scale to 0-1 range
        
        # Adjust for completion rates (lower completion rates need more seats)
        completion_rate = mos_completion_rates.get(mos, 0.8)
        completion_factor = 1.0
        
        if completion_rate < 0.7:
            # Increase allocation for lower completion rates
            completion_factor = 1.0 + (0.7 - completion_rate)
        
        # Combined factor
        combined_factor = (1.0 + bottleneck_factor) * completion_factor
        new_allocation[mos] = combined_factor
        total_factor += combined_factor
    
    # Normalize to match class size
    if total_factor > 0:
        for mos in new_allocation:
            # Calculate seats based on factor
            new_allocation[mos] = int((new_allocation[mos] / total_factor) * class_size)
    
    # Ensure all allocations have at least 1 seat if original did
    for mos, count in current_allocation.items():
        if count > 0 and new_allocation.get(mos, 0) < 1:
            new_allocation[mos] = 1
    
    # Adjust to match class size exactly
    total_allocated = sum(new_allocation.values())
    
    if total_allocated != class_size:
        # Find difference
        diff = class_size - total_allocated
        
        if diff > 0:
            # Add seats to MOS with highest factors
            mos_factors = [(mos, mos_bottlenecks.get(mos, 0)) for mos in new_allocation]
            mos_factors.sort(key=lambda x: x[1], reverse=True)
            
            for i in range(diff):
                if i < len(mos_factors):
                    new_allocation[mos_factors[i][0]] += 1
        
        elif diff < 0:
            # Remove seats from MOS with lowest factors, ensuring min 1 seat
            mos_factors = [(mos, mos_bottlenecks.get(mos, 0)) for mos in new_allocation]
            mos_factors.sort(key=lambda x: x[1])
            
            for i in range(-diff):
                if i < len(mos_factors):
                    mos = mos_factors[i][0]
                    if new_allocation[mos] > 1:  # Ensure at least 1 seat
                        new_allocation[mos] -= 1
    
    return new_allocation

def check_prerequisite_order(
    schedule: List[Dict[str, Any]], 
    course_configs: Dict[str, Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Check if the schedule respects prerequisite ordering
    
    Args:
        schedule: List of scheduled classes
        course_configs: Dictionary of course configurations
        
    Returns:
        List of violations of prerequisite ordering
    """
    violations = []
    
    # Convert to dataframe for easier manipulation
    try:
        schedule_df = pd.DataFrame(schedule)
        schedule_df['start_date'] = pd.to_datetime(schedule_df['start_date'])
        schedule_df['end_date'] = pd.to_datetime(schedule_df['end_date'])
    except Exception as e:
        return [{'error': f"Failed to process schedule: {e}"}]
    
    # For each course, check if prerequisites start before it
    for _, class_info in schedule_df.iterrows():
        course = class_info['course_title']
        
        if course not in course_configs:
            continue
        
        start_date = class_info['start_date']
        
        # Get all prerequisites (direct and from OR groups)
        prereqs = []
        
        if 'prerequisites' in course_configs[course]:
            if isinstance(course_configs[course]['prerequisites'], dict):
                prereqs.extend(course_configs[course]['prerequisites'].get('courses', []))
        
        if 'or_prerequisites' in course_configs[course]:
            for group in course_configs[course]['or_prerequisites']:
                prereqs.extend(group)
        
        # Get MOS-specific prerequisites
        if 'mos_paths' in course_configs[course]:
            for mos, mos_prereqs in course_configs[course]['mos_paths'].items():
                prereqs.extend(mos_prereqs)
        
        # Remove duplicates
        prereqs = list(set(prereqs))
        
        # Check each prerequisite
        for prereq in prereqs:
            # Find all classes for this prerequisite
            prereq_classes = schedule_df[schedule_df['course_title'] == prereq]
            
            if len(prereq_classes) == 0:
                violations.append({
                    'course': course,
                    'prerequisite': prereq,
                    'type': 'missing',
                    'description': f"Prerequisite {prereq} for {course} is not scheduled at all."
                })
                continue
            
            # Check if at least one prerequisite class ends before this course starts
            if not any(prereq_classes['end_date'] < start_date):
                violations.append({
                    'course': course,
                    'prerequisite': prereq,
                    'type': 'timing',
                    'description': f"No {prereq} class ends before {course} starts on {start_date.strftime('%Y-%m-%d')}, but {prereq} is a prerequisite for {course}."
                })
    
    return violations

def check_facility_constraints(
    schedule: List[Dict[str, Any]], 
    facilities: Dict[str, Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Check if the schedule respects facility constraints
    
    Args:
        schedule: List of scheduled classes
        facilities: Dictionary of facility information
        
    Returns:
        List of facility constraint violations
    """
    violations = []
    
    # If no facility information, return empty list
    if not facilities:
        return []
    
    # Convert to dataframe for easier manipulation
    try:
        schedule_df = pd.DataFrame(schedule)
        schedule_df['start_date'] = pd.to_datetime(schedule_df['start_date'])
        schedule_df['end_date'] = pd.to_datetime(schedule_df['end_date'])
    except Exception as e:
        return [{'error': f"Failed to process schedule: {e}"}]
    
    # Create a mapping of which facilities each course requires
    course_facilities = {}
    
    for course_title, facility_info in facilities.items():
        course_facilities[course_title] = facility_info.get('required_facilities', [])
    
    # Track facility usage over time
    facility_usage = {facility: {} for facility in facilities}
    
    # Check each class
    for _, class_info in schedule_df.iterrows():
        course = class_info['course_title']
        start_date = class_info['start_date']
        end_date = class_info['end_date']
        class_size = class_info.get('size', 0)
        
        # Skip if course doesn't have facility requirements
        if course not in course_facilities:
            continue
        
        # Check each required facility
        for facility_name in course_facilities[course]:
            if facility_name not in facilities:
                violations.append({
                    'course': course,
                    'facility': facility_name,
                    'type': 'missing_facility',
                    'description': f"Required facility {facility_name} for {course} is not defined."
                })
                continue
            
            facility = facilities[facility_name]
            capacity = facility.get('capacity', 0)
            
            # Generate dates for the class
            class_dates = pd.date_range(start=start_date, end=end_date)
            
            # Check each date
            for date in class_dates:
                date_str = date.strftime('%Y-%m-%d')
                
                # Initialize usage for this date if needed
                if date_str not in facility_usage[facility_name]:
                    facility_usage[facility_name][date_str] = 0
                
                # Add this class's usage
                facility_usage[facility_name][date_str] += class_size
                
                # Check if capacity is exceeded
                if capacity > 0 and facility_usage[facility_name][date_str] > capacity:
                    violations.append({
                        'course': course,
                        'facility': facility_name,
                        'date': date_str,
                        'type': 'capacity_exceeded',
                        'description': f"Facility {facility_name} capacity exceeded on {date_str}: " +
                                       f"usage {facility_usage[facility_name][date_str]} > capacity {capacity}"
                    })
    
    return violations

def check_instructor_availability(
    schedule: List[Dict[str, Any]], 
    instructors: Dict[str, Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Check if the schedule respects instructor availability constraints
    
    Args:
        schedule: List of scheduled classes
        instructors: Dictionary of instructor information
        
    Returns:
        List of instructor availability violations
    """
    violations = []
    
    # If no instructor information, return empty list
    if not instructors:
        return []
    
    # Convert to dataframe for easier manipulation
    try:
        schedule_df = pd.DataFrame(schedule)
        schedule_df['start_date'] = pd.to_datetime(schedule_df['start_date'])
        schedule_df['end_date'] = pd.to_datetime(schedule_df['end_date'])
    except Exception as e:
        return [{'error': f"Failed to process schedule: {e}"}]
    
    # Create a mapping of which instructors each course requires
    course_instructors = {}
    
    for instructor_id, instructor_info in instructors.items():
        qualified_courses = instructor_info.get('qualified_courses', [])
        
        for course in qualified_courses:
            if course not in course_instructors:
                course_instructors[course] = []
            
            course_instructors[course].append(instructor_id)
    
    # Track instructor assignments over time
    instructor_assignments = {instructor_id: {} for instructor_id in instructors}
    
    # Check each class
    for _, class_info in schedule_df.iterrows():
        course = class_info['course_title']
        start_date = class_info['start_date']
        end_date = class_info['end_date']
        
        # Skip if course doesn't have instructor requirements
        if course not in course_instructors:
            violations.append({
                'course': course,
                'type': 'no_qualified_instructors',
                'description': f"No qualified instructors found for {course}."
            })
            continue
        
        # Get qualified instructors for this course
        qualified_instructors = course_instructors[course]
        
        # Generate dates for the class
        class_dates = pd.date_range(start=start_date, end=end_date)
        
        # Check if any instructor is available for all dates
        available_instructors = []
        
        for instructor_id in qualified_instructors:
            instructor = instructors[instructor_id]
            unavailable_dates = instructor.get('unavailable_dates', [])
            
            # Convert unavailable dates to datetime if they're strings
            if unavailable_dates and isinstance(unavailable_dates[0], str):
                unavailable_dates = [pd.to_datetime(date) for date in unavailable_dates]
            
            # Check if instructor is available for all class dates
            is_available = True
            
            for date in class_dates:
                date_str = date.strftime('%Y-%m-%d')
                
                # Check if already assigned to another class on this date
                if date_str in instructor_assignments[instructor_id]:
                    is_available = False
                    break
                
                # Check if date is in unavailable dates
                if date in unavailable_dates:
                    is_available = False
                    break
            
            if is_available:
                available_instructors.append(instructor_id)
        
        # If no instructors are available, add a violation
        if not available_instructors:
            violations.append({
                'course': course,
                'start_date': start_date.strftime('%Y-%m-%d'),
                'end_date': end_date.strftime('%Y-%m-%d'),
                'type': 'no_available_instructors',
                'description': f"No qualified instructors available for {course} from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}."
            })
        else:
            # Assign the first available instructor
            instructor_id = available_instructors[0]
            
            # Mark instructor as assigned for all class dates
            for date in class_dates:
                date_str = date.strftime('%Y-%m-%d')
                instructor_assignments[instructor_id][date_str] = course
    
    return violations

def apply_custom_paths_to_schedule(
    schedule: List[Dict[str, Any]], 
    custom_paths: Dict[str, Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Apply custom career paths to optimize a schedule
    
    Args:
        schedule: Current schedule to optimize
        custom_paths: Dictionary of custom career paths by MOS
        
    Returns:
        Tuple of (optimized schedule, changes made)
    """
    optimized_schedule = copy.deepcopy(schedule)
    changes = []
    
    # Convert schedule to DataFrame for easier manipulation
    try:
        schedule_df = pd.DataFrame(optimized_schedule)
        schedule_df['start_date'] = pd.to_datetime(schedule_df['start_date'])
        schedule_df['end_date'] = pd.to_datetime(schedule_df['end_date'])
    except Exception as e:
        return optimized_schedule, [{'error': f"Failed to process schedule: {e}"}]
    
    # Process each MOS path
    for mos, path_data in custom_paths.items():
        if mos == 'General':
            continue  # Skip general path
        
        typical_path = path_data.get('path', [])
        
        if not typical_path:
            continue  # Skip if path is empty
        
        # Check for missing courses in the schedule
        missing_courses = []
        
        for course in typical_path:
            if not any(schedule_df['course_title'] == course):
                missing_courses.append(course)
        
        # Suggest adding missing courses
        for course in missing_courses:
            # Find position in the path
            course_idx = typical_path.index(course)
            
            # Determine best timing based on position in path
            if course_idx == 0:
                # First course in path - add at the beginning
                ideal_start_date = schedule_df['start_date'].min() - pd.Timedelta(days=30)
                ideal_end_date = ideal_start_date + pd.Timedelta(days=14)  # Default 2-week course
            elif course_idx == len(typical_path) - 1:
                # Last course in path - add at the end
                ideal_start_date = schedule_df['end_date'].max() + pd.Timedelta(days=14)
                ideal_end_date = ideal_start_date + pd.Timedelta(days=14)
            else:
                # Middle course - try to find appropriate timing based on prerequisites and follow-on courses
                prev_course = typical_path[course_idx - 1]
                next_course = typical_path[course_idx + 1]
                
                prev_classes = schedule_df[schedule_df['course_title'] == prev_course]
                next_classes = schedule_df[schedule_df['course_title'] == next_course]
                
                if not prev_classes.empty and not next_classes.empty:
                    # Find a gap between a prev_course and a next_course
                    prev_end = prev_classes['end_date'].max()
                    next_start = next_classes['start_date'].min()
                    
                    if prev_end < next_start:
                        # Add in the middle of the gap
                        gap_days = (next_start - prev_end).days
                        ideal_start_date = prev_end + pd.Timedelta(days=gap_days//2)
                        ideal_end_date = ideal_start_date + pd.Timedelta(days=14)
                    else:
                        # Conflict - schedule after prev_course
                        ideal_start_date = prev_end + pd.Timedelta(days=7)
                        ideal_end_date = ideal_start_date + pd.Timedelta(days=14)
                elif not prev_classes.empty:
                    # Only prev_course exists
                    ideal_start_date = prev_classes['end_date'].max() + pd.Timedelta(days=7)
                    ideal_end_date = ideal_start_date + pd.Timedelta(days=14)
                elif not next_classes.empty:
                    # Only next_course exists
                    ideal_start_date = next_classes['start_date'].min() - pd.Timedelta(days=21)
                    ideal_end_date = ideal_start_date + pd.Timedelta(days=14)
                else:
                    # Neither exists - use default timing
                    ideal_start_date = schedule_df['start_date'].min() + pd.Timedelta(days=course_idx * 21)
                    ideal_end_date = ideal_start_date + pd.Timedelta(days=14)
            
            # Suggest adding the course
            changes.append({
                'action': 'add_course',
                'course': course,
                'mos': mos,
                'suggested_start': ideal_start_date.strftime('%Y-%m-%d'),
                'suggested_end': ideal_end_date.strftime('%Y-%m-%d'),
                'reason': f"Missing required course {course} for {mos} career path"
            })
        
        # Check if path courses are scheduled in the correct order
        for i in range(1, len(typical_path)):
            prev_course = typical_path[i-1]
            current_course = typical_path[i]
            
            prev_classes = schedule_df[schedule_df['course_title'] == prev_course]
            current_classes = schedule_df[schedule_df['course_title'] == current_course]
            
            if not prev_classes.empty and not current_classes.empty:
                prev_end = prev_classes['end_date'].max()
                current_start = current_classes['start_date'].min()
                
                if current_start < prev_end:
                    # Courses are out of order
                    # Suggest moving the current course to start after the prerequisite ends
                    ideal_start_date = prev_end + pd.Timedelta(days=7)
                    ideal_end_date = ideal_start_date + pd.Timedelta(days=(current_classes['end_date'].min() - current_classes['start_date'].min()).days)
                    
                    changes.append({
                        'action': 'reschedule',
                        'course': current_course,
                        'current_start': current_classes['start_date'].min().strftime('%Y-%m-%d'),
                        'current_end': current_classes['end_date'].min().strftime('%Y-%m-%d'),
                        'suggested_start': ideal_start_date.strftime('%Y-%m-%d'),
                        'suggested_end': ideal_end_date.strftime('%Y-%m-%d'),
                        'reason': f"Course {current_course} starts before its prerequisite {prev_course} ends"
                    })
    
    return optimized_schedule, changes