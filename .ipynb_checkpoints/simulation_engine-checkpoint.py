import pandas as pd
import numpy as np
import datetime
from collections import defaultdict

class Student:
    """Class representing a student in the simulation"""
    
    def __init__(self, id, entry_time, personnel_type='Enlisted', group_type='ADE'):
        self.id = id
        self.entry_time = entry_time
        self.current_time = entry_time
        self.personnel_type = personnel_type  # 'Officer' or 'Enlisted'
        self.group_type = group_type  # 'ADE', 'OF', 'NG', etc.
        self.completed_courses = []
        self.current_course = None
        self.current_class_end = None
        self.waiting_for = None
        self.wait_times = []
        self.total_wait_time = 0
        self.graduation_time = None
    
    def is_eligible_for(self, course, prerequisites):
        """Check if student is eligible for a course based on prerequisites"""
        if not prerequisites:
            return True
        
        for prereq in prerequisites:
            if prereq not in self.completed_courses:
                return False
        
        return True
    
    def start_course(self, course, end_time):
        """Assign student to a course"""
        self.current_course = course
        self.current_class_end = end_time
        self.waiting_for = None
    
    def finish_course(self, passed):
        """Process course completion"""
        if passed:
            self.completed_courses.append(self.current_course)
        self.current_course = None
        self.current_class_end = None
    
    def start_waiting(self, course, current_time):
        """Start waiting for a course"""
        self.waiting_for = course
        self.wait_start_time = current_time
    
    def finish_waiting(self, current_time):
        """End waiting period and update statistics"""
        if self.waiting_for and self.wait_start_time:
            wait_time = (current_time - self.wait_start_time).days
            self.wait_times.append(wait_time)
            self.total_wait_time += wait_time
        
        self.waiting_for = None
        self.wait_start_time = None
    
    def graduate(self, current_time):
        """Mark student as graduated"""
        self.graduation_time = current_time
    
    def days_since_entry(self, current_time):
        """Calculate days since student entered the program"""
        return (current_time - self.entry_time).days

class Class:
    """Class representing a training class in the simulation"""
    
    def __init__(self, course_title, start_date, end_date, max_capacity, reserved_seats=None, officer_enlisted_ratio=None):
        self.course_title = course_title
        self.start_date = start_date
        self.end_date = end_date
        self.max_capacity = max_capacity
        self.reserved_seats = reserved_seats or {}  # {'OF': 10, 'ADE': 30, 'NG': 5}
        self.officer_enlisted_ratio = officer_enlisted_ratio  # Format: '1:4'
        self.students = []
    
    def can_accept(self, student):
        """Check if class can accept a student based on capacity and restrictions"""
        # Check if class is full
        if len(self.students) >= self.max_capacity:
            return False
        
        # Check reserved seats
        if student.group_type in self.reserved_seats:
            # Count current students of this group type
            group_count = sum(1 for s in self.students if s.group_type == student.group_type)
            
            # If we've reached the reserved limit
            if group_count >= self.reserved_seats[student.group_type]:
                # Only accept if we have unreserved seats available
                reserved_total = sum(self.reserved_seats.values())
                if len(self.students) >= reserved_total:
                    return False
        
        # Check officer/enlisted ratio if specified
        if self.officer_enlisted_ratio:
            officer_ratio, enlisted_ratio = map(int, self.officer_enlisted_ratio.split(':'))
            
            # Count current officers and enlisted
            officer_count = sum(1 for s in self.students if s.personnel_type == 'Officer')
            enlisted_count = sum(1 for s in self.students if s.personnel_type == 'Enlisted')
            
            # Calculate target ratio
            total = officer_count + enlisted_count
            if total > 0:
                current_ratio = officer_count / enlisted_count if enlisted_count > 0 else float('inf')
                target_ratio = officer_ratio / enlisted_ratio
                
                # If adding this student would worsen the ratio
                if student.personnel_type == 'Officer' and current_ratio > target_ratio:
                    return False
                if student.personnel_type == 'Enlisted' and current_ratio < target_ratio:
                    return False
        
        return True
    
    def add_student(self, student):
        """Add a student to the class"""
        if self.can_accept(student):
            self.students.append(student)
            return True
        return False
    
    def get_utilization(self):
        """Calculate class utilization"""
        return len(self.students) / self.max_capacity if self.max_capacity > 0 else 0

def run_simulation(inputs):
    """
    Run a simulation of the training schedule
    
    Args:
        inputs (dict): Simulation inputs
            - schedule: List of scheduled classes
            - course_configs: Dictionary of course configurations
            - historical_data: Dictionary of historical data by course
            - num_students: Number of students to simulate
            - num_iterations: Number of simulation iterations
            - randomize_factor: How much to randomize historical rates
            - override_rates: Dictionary of override pass rates by course
    
    Returns:
        dict: Simulation results
    """
    # Extract inputs
    schedule = inputs['schedule']
    course_configs = inputs['course_configs']
    historical_data = inputs['historical_data']
    num_students = inputs['num_students']
    num_iterations = inputs['num_iterations']
    randomize_factor = inputs.get('randomize_factor', 0.1)
    override_rates = inputs.get('override_rates', {})
    
    # Process schedule
    classes = []
    for class_info in schedule:
        course_title = class_info['course_title']
        config = course_configs.get(course_title, {})
        
        # Get reserved seats if specified
        reserved_seats = config.get('reserved_seats', {})
        
        # Get officer/enlisted ratio if specified
        officer_enlisted_ratio = config.get('officer_enlisted_ratio', None)
        
        # Create class object
        class_obj = Class(
            course_title=course_title,
            start_date=pd.to_datetime(class_info['start_date']),
            end_date=pd.to_datetime(class_info['end_date']),
            max_capacity=class_info['size'],
            reserved_seats=reserved_seats,
            officer_enlisted_ratio=officer_enlisted_ratio
        )
        
        classes.append(class_obj)
    
    # Sort classes by start date
    classes.sort(key=lambda x: x.start_date)
    
    # Get simulation start and end dates
    simulation_start = min(c.start_date for c in classes)
    simulation_end = max(c.end_date for c in classes) + pd.Timedelta(days=365)  # Add a year to see completion
    
    # Prepare results collection
    all_results = []
    
    # Run multiple iterations
    for iteration in range(num_iterations):
        # Generate students
        students = []
        
        # Randomize student attributes based on historical data
        personnel_types = []
        group_types = []
        
        # Calculate overall personnel and group type distributions
        officer_ratio = 0.2  # Default if no data
        group_distribution = {'ADE': 0.7, 'OF': 0.2, 'NG': 0.1}  # Default if no data
        
        for course, data in historical_data.items():
            if 'officer_ratio' in data:
                officer_ratio = data['officer_ratio']
            if 'group_composition' in data:
                group_distribution = data['group_composition']
        
        # Create students with random attributes
        for i in range(num_students):
            # Randomly determine personnel type
            personnel_type = 'Officer' if np.random.random() < officer_ratio else 'Enlisted'
            
            # Randomly determine group type
            group_type = np.random.choice(
                list(group_distribution.keys()),
                p=list(group_distribution.values())
            )
            
            # Randomly determine entry time (distributed over first 180 days)
            entry_offset = np.random.randint(0, 180)
            entry_time = simulation_start + pd.Timedelta(days=entry_offset)
            
            # Create student
            student = Student(
                id=i,
                entry_time=entry_time,
                personnel_type=personnel_type,
                group_type=group_type
            )
            
            students.append(student)
        
        # Simulation state
        current_time = simulation_start
        active_students = []
        completed_students = []
        
        # Time step (1 day)
        time_step = pd.Timedelta(days=1)
        
        # Track metrics over time
        student_progression = []
        class_utilization = []
        wait_times_by_course = defaultdict(list)
        
        # Main simulation loop
        while current_time <= simulation_end:
            # Activate new students who are entering today
            new_students = [s for s in students if s.entry_time <= current_time and s not in active_students and s not in completed_students]
            active_students.extend(new_students)
            
            # Process students finishing courses today
            finishing_students = [s for s in active_students if s.current_class_end and s.current_class_end <= current_time]
            
            for student in finishing_students:
                course = student.current_course
                
                # Determine if student passes the course
                base_pass_rate = override_rates.get(course, historical_data.get(course, {}).get('pass_rate', 0.8))
                
                # Add randomness to pass rate
                pass_rate = max(0, min(1, base_pass_rate + np.random.uniform(-randomize_factor, randomize_factor)))
                
                passed = np.random.random() < pass_rate
                
                # Process course completion
                student.finish_course(passed)
                
                # If failed, may need to re-take the course
                if not passed:
                    # Determine if student recycles or drops out
                    recycle_rate = historical_data.get(course, {}).get('recycle_rate', 0.5)
                    recycles = np.random.random() < recycle_rate
                    
                    if recycles:
                        # Start waiting to retake the course
                        student.start_waiting(course, current_time)
                    else:
                        # Student drops out
                        active_students.remove(student)
                        completed_students.append(student)
                        student.graduate(current_time)
                
                # If passed, check if student has completed all required courses
                elif all(c in student.completed_courses for c in get_required_courses(course_configs)):
                    # Student graduates
                    active_students.remove(student)
                    completed_students.append(student)
                    student.graduate(current_time)
                else:
                    # Student needs to take more courses
                    # Find the next course based on prerequisites
                    next_course = find_next_course(student, course_configs)
                    
                    if next_course:
                        student.start_waiting(next_course, current_time)
                    else:
                        # No valid next course, student is done
                        active_students.remove(student)
                        completed_students.append(student)
                        student.graduate(current_time)
            
            # Process classes starting today
            starting_classes = [c for c in classes if c.start_date <= current_time and c.start_date + time_step > current_time]
            
            for class_obj in starting_classes:
                # Find eligible waiting students
                eligible_students = [
                    s for s in active_students 
                    if s.waiting_for == class_obj.course_title and 
                    s.is_eligible_for(class_obj.course_title, course_configs.get(class_obj.course_title, {}).get('prerequisites', []))
                ]
                
                # Sort by wait time (longest first)
                eligible_students.sort(key=lambda s: s.total_wait_time, reverse=True)
                
                # Try to add students to class
                for student in eligible_students:
                    if class_obj.add_student(student):
                        # Student added to class
                        student.finish_waiting(current_time)
                        student.start_course(class_obj.course_title, class_obj.end_date)
                        
                        # Record wait time for this course
                        wait_time = student.wait_times[-1] if student.wait_times else 0
                        wait_times_by_course[class_obj.course_title].append(wait_time)
                
                # Record class utilization
                class_utilization.append({
                    'course': class_obj.course_title,
                    'start_date': class_obj.start_date,
                    'end_date': class_obj.end_date,
                    'capacity': class_obj.max_capacity,
                    'enrolled': len(class_obj.students),
                    'utilization': class_obj.get_utilization()
                })
            
            # Record student progression metrics
            waiting_count = sum(1 for s in active_students if s.waiting_for)
            in_class_count = sum(1 for s in active_students if s.current_course)
            graduated_count = len(completed_students)
            
            student_progression.append({
                'time': current_time,
                'stage': 'Waiting',
                'count': waiting_count
            })
            
            student_progression.append({
                'time': current_time,
                'stage': 'In Class',
                'count': in_class_count
            })
            
            student_progression.append({
                'time': current_time,
                'stage': 'Graduated',
                'count': graduated_count
            })
            
            # Advance time
            current_time += time_step
        
        # Collect iteration results
        iteration_results = {
            'completion_times': [
                {'student_id': s.id, 'days': (s.graduation_time - s.entry_time).days if s.graduation_time else None}
                for s in completed_students if s.graduation_time
            ],
            'wait_times': [
                {'student_id': s.id, 'course': c, 'days': days}
                for s in completed_students
                for c, days in zip(s.completed_courses, s.wait_times)
            ],
            'student_progression': student_progression,
            'class_utilization': class_utilization,
            'bottlenecks': [
                {'course': course, 'wait_time': np.mean(times) if times else 0}
                for course, times in wait_times_by_course.items()
            ]
        }
        
        all_results.append(iteration_results)
    
    # Aggregate results across iterations
    aggregated_results = aggregate_simulation_results(all_results)
    
    return aggregated_results

def get_required_courses(course_configs):
    """Get list of all courses that don't have prerequisites defined"""
    required_courses = []
    
    for course, config in course_configs.items():
        if 'prerequisites' not in config or not config['prerequisites']:
            required_courses.append(course)
    
    return required_courses

def find_next_course(student, course_configs):
    """Find the next course a student should take based on prerequisites"""
    eligible_courses = []
    
    for course, config in course_configs.items():
        # Skip if student already completed this course
        if course in student.completed_courses:
            continue
        
        # Check prerequisites
        prerequisites = config.get('prerequisites', [])
        if student.is_eligible_for(course, prerequisites):
            eligible_courses.append(course)
    
    # Return the first eligible course if any
    return eligible_courses[0] if eligible_courses else None

def aggregate_simulation_results(all_results):
    """Aggregate results from multiple simulation iterations"""
    # Completion times
    all_completion_times = [result['completion_times'] for result in all_results]
    flattened_completion_times = [
        item for sublist in all_completion_times for item in sublist if item['days'] is not None
    ]
    
    avg_completion_time = np.mean([ct['days'] for ct in flattened_completion_times]) if flattened_completion_times else 0
    
    # Wait times
    all_wait_times = [result['wait_times'] for result in all_results]
    flattened_wait_times = [
        item for sublist in all_wait_times for item in sublist
    ]
    
    avg_wait_time = np.mean([wt['days'] for wt in flattened_wait_times]) if flattened_wait_times else 0
    
    # Bottlenecks
    all_bottlenecks = [result['bottlenecks'] for result in all_results]
    
    # Create a dictionary to aggregate bottleneck data by course
    bottleneck_data = defaultdict(list)
    for iteration_bottlenecks in all_bottlenecks:
        for bottleneck in iteration_bottlenecks:
            bottleneck_data[bottleneck['course']].append(bottleneck['wait_time'])
    
    # Calculate average wait time for each course
    aggregated_bottlenecks = [
        {'course': course, 'wait_time': np.mean(wait_times)}
        for course, wait_times in bottleneck_data.items()
    ]
    
    # Sort bottlenecks by wait time (descending)
    aggregated_bottlenecks.sort(key=lambda x: x['wait_time'], reverse=True)
    
    # Class utilization
    all_utilizations = [result['class_utilization'] for result in all_results]
    flattened_utilizations = [
        item for sublist in all_utilizations for item in sublist
    ]
    
    # Group utilization by course
    utilization_by_course = defaultdict(list)
    for util in flattened_utilizations:
        utilization_by_course[util['course']].append(util['utilization'])
    
    aggregated_utilization = [
        {'course': course, 'utilization': np.mean(utils)}
        for course, utils in utilization_by_course.items()
    ]
    
    # Sort by utilization (ascending to highlight low utilization)
    aggregated_utilization.sort(key=lambda x: x['utilization'])
    
    # Student progression
    # We'll just use the last iteration for this since it's a time series
    student_progression = all_results[-1]['student_progression']
    
    # Calculate throughput (graduates per month)
    first_result = all_results[0]
    if 'student_progression' in first_result:
        progression = first_result['student_progression']
        
        # Filter for 'Graduated' stage and get the final count
        graduated_progression = [p for p in progression if p['stage'] == 'Graduated']
        graduated_progression.sort(key=lambda x: x['time'])
        
        if graduated_progression:
            # Get the first and last timestamps
            first_time = pd.to_datetime(graduated_progression[0]['time'])
            last_time = pd.to_datetime(graduated_progression[-1]['time'])
            
            # Calculate months (assuming 30 days per month)
            months = (last_time - first_time).days / 30
            
            # Calculate throughput
            if months > 0:
                final_count = graduated_progression[-1]['count']
                throughput = final_count / months
            else:
                throughput = 0
        else:
            throughput = 0
    else:
        throughput = 0
    
    # Calculate resource utilization
    avg_utilization = np.mean([u['utilization'] for u in flattened_utilizations]) if flattened_utilizations else 0
    
    # Detailed metrics by course
    detailed_metrics = []
    
    # Courses
    all_courses = set()
    for result in all_results:
        for bottleneck in result['bottlenecks']:
            all_courses.add(bottleneck['course'])
    
    for course in all_courses:
        # Wait times for this course
        course_wait_times = [
            wt['days'] for result in all_results
            for wt in result['wait_times']
            if wt.get('course') == course
        ]
        
        # Utilization for this course
        course_utilization = [
            util['utilization'] for result in all_results
            for util in result['class_utilization']
            if util['course'] == course
        ]
        
        detailed_metrics.append({
            'course': course,
            'avg_wait_time': np.mean(course_wait_times) if course_wait_times else 0,
            'max_wait_time': np.max(course_wait_times) if course_wait_times else 0,
            'avg_utilization': np.mean(course_utilization) if course_utilization else 0,
            'bottleneck_score': np.mean(course_wait_times) * (1 - np.mean(course_utilization) if course_utilization else 0)
        })
    
    # Sort by bottleneck score (higher is worse)
    detailed_metrics.sort(key=lambda x: x['bottleneck_score'], reverse=True)
    
    return {
        'avg_completion_time': avg_completion_time,
        'avg_wait_time': avg_wait_time,
        'throughput': throughput,
        'resource_utilization': avg_utilization,
        'bottlenecks': aggregated_bottlenecks,
        'class_utilization': aggregated_utilization,
        'student_progression': student_progression,
        'completion_times': flattened_completion_times,
        'detailed_metrics': detailed_metrics
    }