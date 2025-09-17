import pandas as pd
import numpy as np
import datetime
from collections import defaultdict

class Student:
    """Class representing a student in the simulation"""
    
    def __init__(self, id, entry_time, personnel_type='Enlisted', group_type='ADE', training_mos=None, arrival_date=None):
        self.id = id
        self.entry_time = entry_time  # When student enters the system
        self.arrival_date = arrival_date if arrival_date is not None else entry_time  # When student arrives on campus
        self.current_time = entry_time
        self.personnel_type = personnel_type  # 'Officer' or 'Enlisted'
        self.group_type = group_type  # 'ADE', 'OF', 'NG', etc.
        self.training_mos = training_mos  # '18A', '18B', '18C', '18D', or '18E'
        self.completed_courses = []
        self.current_course = None
        self.current_class_end = None
        self.waiting_for = None
        self.wait_start_time = None
        self.wait_times = []
        self.total_wait_time = 0
        self.graduation_time = None
        self.first_class_start = None  # Track when the student's first class starts
    
    def is_eligible_for(self, course, course_configs):
        """Check if student is eligible for a course based on prerequisites and MOS path"""
        if course not in course_configs:
            return True  # No configuration means no prerequisites
        
        config = course_configs[course]
        
        # Check if this course is required for all MOS paths
        required_for_all = config.get('required_for_all_mos', False)
        
        # Check if student has a training MOS and this course is part of their path
        if self.training_mos and not required_for_all and 'mos_paths' in config:
            mos_paths = config.get('mos_paths', {})
            
            # If the course has MOS paths defined but student's MOS is not included,
            # they are not eligible for this course
            if mos_paths and self.training_mos in mos_paths and not mos_paths[self.training_mos]:
                return False
        
        # For MOS-specific prerequisites, check only prerequisites for student's MOS
        if self.training_mos and 'mos_paths' in config:
            mos_prereqs = config['mos_paths'].get(self.training_mos, [])
            
            # Check if student has completed all required prerequisites for their MOS
            for prereq in mos_prereqs:
                if prereq not in self.completed_courses:
                    return False
            
            # If there are no MOS-specific prereqs or student completed them all, 
            # they are eligible
            return True
        
        # If no MOS-specific paths are defined, fall back to general prerequisites
        prereqs = config.get('prerequisites', {'type': 'AND', 'courses': []})
        or_prereqs = config.get('or_prerequisites', [])
        
        # Check AND prerequisites
        if prereqs['type'] == 'AND':
            for prereq in prereqs['courses']:
                if prereq not in self.completed_courses:
                    return False
        elif prereqs['type'] == 'OR' and prereqs['courses']:
            # For OR logic, at least one course must be completed
            if not any(prereq in self.completed_courses for prereq in prereqs['courses']):
                return False
        
        # Check complex OR prerequisites (groups)
        for group in or_prereqs:
            if not group:  # Skip empty groups
                continue
            # For each group, at least one course must be completed
            if not any(prereq in self.completed_courses for prereq in group):
                return False
        
        return True
    
    def start_course(self, course, end_time):
        """Assign student to a course"""
        self.current_course = course
        self.current_class_end = end_time
        self.waiting_for = None
        
        # Track first class start date
        if self.first_class_start is None:
            self.first_class_start = self.current_time
    
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
    
    def __init__(self, course_title, start_date, end_date, max_capacity, 
                 reserved_seats=None, officer_enlisted_ratio=None, mos_allocation=None):
        self.course_title = course_title
        self.start_date = start_date
        self.end_date = end_date
        self.max_capacity = max_capacity
        self.reserved_seats = reserved_seats or {}  # {'OF': 10, 'ADE': 30, 'NG': 5}
        self.officer_enlisted_ratio = officer_enlisted_ratio  # Format: '1:4'
        self.mos_allocation = mos_allocation or {}  # {'18A': 10, '18B': 20, etc.}
        self.students = []
        
        # Track current enrollment by MOS
        self.current_mos_enrollment = {mos: 0 for mos in self.mos_allocation.keys()}
    
    def can_accept(self, student):
        """Check if class can accept a student based on capacity, restrictions, and MOS"""
        # Check if class is full
        if len(self.students) >= self.max_capacity:
            return False
        
        # Check MOS allocation if applicable
        if self.mos_allocation and student.training_mos:
            # If this MOS is not allocated for this class, reject
            if student.training_mos not in self.mos_allocation or self.mos_allocation[student.training_mos] <= 0:
                return False
                
            # Check if MOS allocation is full
            current_count = self.current_mos_enrollment.get(student.training_mos, 0)
            max_count = self.mos_allocation.get(student.training_mos, 0)
            if current_count >= max_count:
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
            try:
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
            except (ValueError, ZeroDivisionError, AttributeError):
                # If there's any error parsing the ratio, just ignore the ratio check
                pass
        
        return True
    
    def add_student(self, student):
        """Add a student to the class"""
        if self.can_accept(student):
            self.students.append(student)
            
            # Update MOS enrollment count
            if student.training_mos and student.training_mos in self.current_mos_enrollment:
                self.current_mos_enrollment[student.training_mos] += 1
                
            return True
        return False
    
    def get_utilization(self):
        """Calculate class utilization"""
        return len(self.students) / self.max_capacity if self.max_capacity > 0 else 0
    
    def get_mos_utilization(self):
        """Calculate utilization by MOS"""
        mos_util = {}
        for mos, allocation in self.mos_allocation.items():
            if allocation > 0:
                enrolled = self.current_mos_enrollment.get(mos, 0)
                mos_util[mos] = enrolled / allocation
        return mos_util

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
            - use_historical_data: Whether to use historical data patterns
            - historical_arrival_patterns: Historical arrival patterns if available
            - historical_mos_distribution: Historical MOS distribution if available
            - arrival_method: How students arrive ("before_class" or "continuous")
            - arrival_days_before: For "before_class" method, days before class starts
            - monthly_distribution: For "continuous" method, distribution pattern
            - arrival_randomness: For "continuous" method, randomness factor
            - custom_distribution: Custom monthly distribution if provided
            - mos_distribution: Dictionary of MOS distribution percentages
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
    use_historical_data = inputs.get('use_historical_data', False)
    randomize_factor = inputs.get('randomize_factor', 0.1)
    override_rates = inputs.get('override_rates', {})
    
    # Get arrival and MOS parameters
    # Default values
    arrival_method = 'before_class'
    arrival_days_before = 3
    monthly_distribution = 'Even'
    arrival_randomness = 0.3
    custom_distribution = None
    mos_distribution = {
        '18A': 0.2, '18B': 0.2, '18C': 0.2, '18D': 0.2, '18E': 0.2
    }
    
    # Override with historical data if available and requested
    if use_historical_data:
        historical_arrival_patterns = inputs.get('historical_arrival_patterns', None)
        if historical_arrival_patterns:
            # Use historical average days before class
            arrival_days_before = int(historical_arrival_patterns.get('avg_days_before', 3))
            
            # Use historical monthly distribution
            monthly_distribution = 'Historical'
            custom_distribution = historical_arrival_patterns.get('monthly_distribution', None)
        
        # Use historical MOS distribution if available
        historical_mos_distribution = inputs.get('historical_mos_distribution', None)
        if historical_mos_distribution:
            mos_distribution = historical_mos_distribution
    else:
        # Use manually specified settings
        arrival_method = inputs.get('arrival_method', 'before_class')
        arrival_days_before = inputs.get('arrival_days_before', 3)
        monthly_distribution = inputs.get('monthly_distribution', 'Even')
        arrival_randomness = inputs.get('arrival_randomness', 0.3)
        custom_distribution = inputs.get('custom_distribution', None)
        mos_distribution = inputs.get('mos_distribution', {
            '18A': 0.2, '18B': 0.2, '18C': 0.2, '18D': 0.2, '18E': 0.2
        })
    
    # Normalize MOS distribution to ensure it sums to 1
    total_mos = sum(mos_distribution.values())
    if total_mos > 0:
        mos_distribution = {mos: value/total_mos for mos, value in mos_distribution.items()}
    
    # Process schedule
    classes = []
    for class_info in schedule:
        course_title = class_info['course_title']
        config = course_configs.get(course_title, {})
        
        # Get reserved seats if specified
        reserved_seats = config.get('reserved_seats', {})
        
        # Get officer/enlisted ratio if specified
        officer_enlisted_ratio = config.get('officer_enlisted_ratio', None)
        
        # Get MOS allocation if specified
        mos_allocation = class_info.get('mos_allocation', {})
        
        # Create class object
        class_obj = Class(
            course_title=course_title,
            start_date=pd.to_datetime(class_info['start_date']),
            end_date=pd.to_datetime(class_info['end_date']),
            max_capacity=class_info['size'],
            reserved_seats=reserved_seats,
            officer_enlisted_ratio=officer_enlisted_ratio,
            mos_allocation=mos_allocation
        )
        
        classes.append(class_obj)
    
    # Sort classes by start date
    classes.sort(key=lambda x: x.start_date)
    
    # Get simulation start and end dates
    simulation_start = min(c.start_date for c in classes)
    simulation_end = max(c.end_date for c in classes) + pd.Timedelta(days=365)  # Add a year to see completion
    
    # Prepare results collection
    all_results = []
    all_arrival_patterns = []
    
    # Run multiple iterations
    for iteration in range(num_iterations):
        # Generate students
        students = []
        
        # Initialize tracking for arrival patterns
        arrival_dates_by_month = {month: 0 for month in range(1, 13)}
        total_days_before = []
        
        if arrival_method == "before_class":
            # Generate students who arrive before specific classes
            students = generate_before_class_students(
                classes, num_students, arrival_days_before, 
                mos_distribution, arrival_dates_by_month, total_days_before
            )
        else:
            # Generate students who arrive continuously throughout the year
            students = generate_continuous_students(
                simulation_start, simulation_end, num_students, 
                monthly_distribution, arrival_randomness, custom_distribution,
                mos_distribution, arrival_dates_by_month
            )
        
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
        class_mos_utilization = []
        
        # Main simulation loop
        while current_time <= simulation_end:
            # Activate new students who are entering today
            new_students = [s for s in students if s.arrival_date <= current_time and s not in active_students and s not in completed_students]
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
                
                # If passed, check if student has completed all required courses for their MOS
                elif has_completed_required_courses(student, course_configs):
                    # Student graduates
                    active_students.remove(student)
                    completed_students.append(student)
                    student.graduate(current_time)
                else:
                    # Student needs to take more courses
                    # Find the next course based on prerequisites and MOS path
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
                    s.is_eligible_for(class_obj.course_title, course_configs)
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
                
                # Record MOS utilization
                mos_utilization = class_obj.get_mos_utilization()
                if mos_utilization:
                    class_mos_utilization.append({
                        'course': class_obj.course_title,
                        'class_id': class_obj.start_date.strftime('%Y-%m-%d'),  # Use date as ID
                        'mos_utilization': mos_utilization
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
            
            # Track student arrivals by month
            for student in new_students:
                if student.arrival_date.month in arrival_dates_by_month:
                    arrival_dates_by_month[student.arrival_date.month] += 1
                
                # Track days before first class
                if student.first_class_start:
                    days_before = (student.first_class_start - student.arrival_date).days
                    if days_before >= 0:  # Only count valid days before
                        total_days_before.append(days_before)
            
            # Advance time
            current_time += time_step
        
        # Calculate arrival patterns for this iteration
        total_arrivals = sum(arrival_dates_by_month.values())
        monthly_distribution_result = {month: count/total_arrivals for month, count in arrival_dates_by_month.items()} if total_arrivals > 0 else {}
        
        # Convert month numbers to names for better readability
        month_names = {
            1: 'January', 2: 'February', 3: 'March', 4: 'April',
            5: 'May', 6: 'June', 7: 'July', 8: 'August',
            9: 'September', 10: 'October', 11: 'November', 12: 'December'
        }
        named_monthly_distribution = {month_names[month]: percentage for month, percentage in monthly_distribution_result.items()}
        
        iteration_arrival_patterns = {
            'avg_days_before': np.mean(total_days_before) if total_days_before else 0,
            'monthly_distribution': named_monthly_distribution
        }
        
        all_arrival_patterns.append(iteration_arrival_patterns)
        
        # Collect MOS-specific metrics
        mos_metrics = {}
        for mos in mos_distribution.keys():
            # Get all students with this MOS who graduated
            mos_students = [s for s in completed_students if s.training_mos == mos and s.graduation_time]
            
            if mos_students:
                # Calculate average completion time
                completion_times = [(s.graduation_time - s.entry_time).days for s in mos_students]
                avg_completion = np.mean(completion_times) if completion_times else 0
                
                # Calculate average wait time
                wait_times = [s.total_wait_time for s in mos_students]
                avg_wait = np.mean(wait_times) if wait_times else 0
                
                mos_metrics[mos] = {
                    'count': len(mos_students),
                    'avg_completion_time': avg_completion,
                    'avg_wait_time': avg_wait
                }
            else:
                mos_metrics[mos] = {
                    'count': 0,
                    'avg_completion_time': 0,
                    'avg_wait_time': 0
                }
        
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
            'class_mos_utilization': class_mos_utilization,
            'bottlenecks': [
                {'course': course, 'wait_time': np.mean(times) if times else 0}
                for course, times in wait_times_by_course.items()
            ],
            'mos_metrics': mos_metrics
        }
        
        all_results.append(iteration_results)
    
    # Aggregate results across iterations
    aggregated_results = aggregate_simulation_results(all_results, all_arrival_patterns)
    
    return aggregated_results

def generate_before_class_students(classes, num_students, arrival_days_before, mos_distribution, 
                                 arrival_dates_by_month, total_days_before):
    """
    Generate students who arrive before specific classes
    
    Args:
        classes (list): List of class objects
        num_students (int): Number of students to generate
        arrival_days_before (int): Days before class students arrive
        mos_distribution (dict): Distribution of MOS paths
        arrival_dates_by_month (dict): Dictionary to track arrivals by month
        total_days_before (list): List to track days before class
        
    Returns:
        list: List of student objects
    """
    students = []
    
    # Distribute students among classes based on capacity
    total_capacity = sum(c.max_capacity for c in classes)
    
    # Calculate how many students to assign to each class
    class_assignments = {}
    for class_obj in classes:
        # Percentage of total capacity this class represents
        capacity_percent = class_obj.max_capacity / total_capacity if total_capacity > 0 else 0
        # Number of students to assign to this class
        assigned_students = int(num_students * capacity_percent)
        class_assignments[class_obj] = assigned_students
    
    # Adjust to ensure we get exactly num_students total
    total_assigned = sum(class_assignments.values())
    if total_assigned < num_students:
        # Distribute remaining students to classes with highest capacity
        remaining = num_students - total_assigned
        classes_by_capacity = sorted(classes, key=lambda c: c.max_capacity, reverse=True)
        for i in range(remaining):
            if i < len(classes_by_capacity):
                class_assignments[classes_by_capacity[i]] += 1
    
    # Create students for each class
    student_id = 0
    for class_obj, num_assigned in class_assignments.items():
        for _ in range(num_assigned):
            # Determine arrival date (days before class starts)
            arrival_date = class_obj.start_date - pd.Timedelta(days=arrival_days_before)
            
            # Track arrivals by month
            if arrival_date.month in arrival_dates_by_month:
                arrival_dates_by_month[arrival_date.month] += 1
            
            # Track days before first class
            total_days_before.append(arrival_days_before)
            
            # Determine MOS based on distribution or class MOS allocation
            if class_obj.mos_allocation:
                # Weight the MOS distribution by the class allocation
                mos_weights = {}
                total_allocation = sum(class_obj.mos_allocation.values())
                
                if total_allocation > 0:
                    for mos, count in class_obj.mos_allocation.items():
                        mos_weights[mos] = count / total_allocation
                    
                    # Combine with global MOS distribution
                    for mos in mos_weights:
                        mos_weights[mos] = (mos_weights[mos] + mos_distribution.get(mos, 0)) / 2
                    
                    # Normalize
                    total_weight = sum(mos_weights.values())
                    if total_weight > 0:
                        mos_weights = {mos: w/total_weight for mos, w in mos_weights.items()}
                    
                    training_mos = np.random.choice(
                        list(mos_weights.keys()),
                        p=list(mos_weights.values())
                    )
                else:
                    training_mos = np.random.choice(
                        list(mos_distribution.keys()),
                        p=list(mos_distribution.values())
                    )
            else:
                training_mos = np.random.choice(
                    list(mos_distribution.keys()),
                    p=list(mos_distribution.values())
                )
            
            # Personnel type based on MOS (18A = Officer, others = Enlisted)
            personnel_type = 'Officer' if training_mos == '18A' else 'Enlisted'
            
            # Group type distribution (simplified)
            group_type = np.random.choice(['ADE', 'NG', 'OF'], p=[0.7, 0.2, 0.1])
            
            # Create student
            student = Student(
                id=student_id,
                entry_time=arrival_date,
                arrival_date=arrival_date,
                personnel_type=personnel_type,
                group_type=group_type,
                training_mos=training_mos
            )
            
            students.append(student)
            student_id += 1
    
    return students

def generate_continuous_students(simulation_start, simulation_end, num_students, 
                               monthly_distribution, arrival_randomness, custom_distribution,
                               mos_distribution, arrival_dates_by_month):
    """
    Generate students who arrive continuously throughout the simulation period
    
    Args:
        simulation_start (datetime): Start date of simulation
        simulation_end (datetime): End date of simulation
        num_students (int): Number of students to generate
        monthly_distribution (str): Type of monthly distribution
        arrival_randomness (float): Randomness factor for arrivals
        custom_distribution (dict): Custom monthly distribution if specified
        mos_distribution (dict): Distribution of MOS paths
        arrival_dates_by_month (dict): Dictionary to track arrivals by month
        
    Returns:
        list: List of student objects
    """
    students = []
    
    # Create monthly weights based on selected distribution
    monthly_weights = {}
    
    if monthly_distribution == "Even":
        for month in range(1, 13):
            monthly_weights[month] = 1.0 / 12
    
    elif monthly_distribution == "Summer heavy":
        for month in range(1, 13):
            if 5 <= month <= 8:  # May to August
                monthly_weights[month] = 0.15  # 60% of students in summer months
            else:
                monthly_weights[month] = 0.04  # 40% of students in other months
    
    elif monthly_distribution == "Winter heavy":
        for month in range(1, 13):
            if 11 <= month or month <= 2:  # November to February
                monthly_weights[month] = 0.15  # 60% of students in winter months
            else:
                monthly_weights[month] = 0.04  # 40% of students in other months
    
    elif monthly_distribution == "Historical" and custom_distribution:
        # Convert month names to numbers
        month_name_to_num = {
            'January': 1, 'February': 2, 'March': 3, 'April': 4,
            'May': 5, 'June': 6, 'July': 7, 'August': 8,
            'September': 9, 'October': 10, 'November': 11, 'December': 12
        }
        
        monthly_weights = {}
        for month_name, weight in custom_distribution.items():
            month_num = month_name_to_num.get(month_name, 0)
            if month_num > 0:
                monthly_weights[month_num] = weight
        
        # Fill in any missing months with small weights
        for month in range(1, 13):
            if month not in monthly_weights:
                monthly_weights[month] = 0.01
    
    else:
        # Default to even distribution
        for month in range(1, 13):
            monthly_weights[month] = 1.0 / 12
    
    # Normalize weights to ensure they sum to 1
    total_weight = sum(monthly_weights.values())
    if total_weight > 0:
        monthly_weights = {month: weight/total_weight for month, weight in monthly_weights.items()}
    
    # Generate arrival dates based on monthly weights
    arrival_dates = []
    months = list(monthly_weights.keys())
    weights = [monthly_weights[m] for m in months]
    
    # Assign a month to each student based on weights
    student_months = np.random.choice(months, size=num_students, p=weights)
    
    # For each student, generate a random day within their assigned month
    simulation_year = simulation_start.year
    simulation_month_start = simulation_start.month
    
    for i in range(num_students):
        assigned_month = student_months[i]
        
        # Determine which year this month falls in
        if assigned_month < simulation_month_start:
            year = simulation_year + 1
        else:
            year = simulation_year
        
        # Get the number of days in the month
        days_in_month = pd.Timestamp(year=year, month=assigned_month, day=1).days_in_month
        
        # Generate a random day
        day = np.random.randint(1, days_in_month + 1)
        
        # Create arrival date
        arrival_date = pd.Timestamp(year=year, month=assigned_month, day=day)
        
        # Add randomness if specified
        if arrival_randomness > 0:
            # Add or subtract up to 15 days based on randomness factor
            max_days_shift = int(15 * arrival_randomness)
            days_shift = np.random.randint(-max_days_shift, max_days_shift + 1)
            arrival_date += pd.Timedelta(days=days_shift)
        
        # Ensure arrival date is within simulation period
        if arrival_date < simulation_start:
            arrival_date = simulation_start
        elif arrival_date > simulation_end:
            arrival_date = simulation_end
        
        # Track arrivals by month
        if arrival_date.month in arrival_dates_by_month:
            arrival_dates_by_month[arrival_date.month] += 1
        
        arrival_dates.append(arrival_date)
        
        # Determine MOS based on distribution
        training_mos = np.random.choice(
            list(mos_distribution.keys()),
            p=list(mos_distribution.values())
        )
        
        # Personnel type based on MOS (18A = Officer, others = Enlisted)
        personnel_type = 'Officer' if training_mos == '18A' else 'Enlisted'
        
        # Group type distribution (simplified)
        group_type = np.random.choice(['ADE', 'NG', 'OF'], p=[0.7, 0.2, 0.1])
        
        # Create student
        student = Student(
            id=i,
            entry_time=arrival_date,
            arrival_date=arrival_date,
            personnel_type=personnel_type,
            group_type=group_type,
            training_mos=training_mos
        )
        
        students.append(student)
    
    return students

def has_completed_required_courses(student, course_configs):
    """
    Check if a student has completed all required courses for their MOS
    
    Args:
        student (Student): Student object
        course_configs (dict): Dictionary of course configurations
        
    Returns:
        bool: True if student has completed all required courses, False otherwise
    """
    # If student has no MOS, check if they've completed any course marked as required for all
    if not student.training_mos:
        # Check if any course is required for all but not completed
        for course, config in course_configs.items():
            if config.get('required_for_all_mos', False) and course not in student.completed_courses:
                return False
        return True
    
    # Check MOS-specific requirements
    required_courses = []
    
    for course, config in course_configs.items():
        # If course is required for all MOS paths, add it
        if config.get('required_for_all_mos', False):
            required_courses.append(course)
        
        # If course is specifically in this student's MOS path, add it
        if 'mos_paths' in config and student.training_mos in config['mos_paths']:
            # Only add if this MOS has this course in its path
            if config['mos_paths'][student.training_mos]:
                required_courses.append(course)
    
    # Check if student has completed all required courses
    for course in required_courses:
        if course not in student.completed_courses:
            return False
    
    return True

def find_next_course(student, course_configs):
    """Find the next course a student should take based on prerequisites and MOS path"""
    eligible_courses = []
    
    for course, config in course_configs.items():
        # Skip if student already completed this course
        if course in student.completed_courses:
            continue
        
        # Check if the course is relevant for this student's MOS
        if student.training_mos and 'mos_paths' in config:
            # Skip if the course has MOS paths defined and student's MOS is not included
            # and the course is not required for all MOS paths
            required_for_all = config.get('required_for_all_mos', False)
            if not required_for_all:
                mos_paths = config['mos_paths']
                # If the course is not in the student's MOS path, skip it
                if student.training_mos in mos_paths and not mos_paths[student.training_mos]:
                    continue
        
        # Check if student is eligible
        if student.is_eligible_for(course, course_configs):
            eligible_courses.append(course)
    
    # Sort eligible courses by priority:
    # 1. Courses that are specifically in the student's MOS path
    # 2. Courses that are required for all MOS paths
    # 3. Other eligible courses
    if eligible_courses and student.training_mos:
        def course_priority(course):
            config = course_configs.get(course, {})
            
            # Check if course is specifically in student's MOS path
            if 'mos_paths' in config and student.training_mos in config['mos_paths']:
                if config['mos_paths'][student.training_mos]:
                    return 0  # Highest priority
            
            # Check if course is required for all MOS paths
            if config.get('required_for_all_mos', False):
                return 1  # Medium priority
                
            # Otherwise, lowest priority
            return 2
        
        eligible_courses.sort(key=course_priority)
    
    # Return the highest priority eligible course if any
    return eligible_courses[0] if eligible_courses else None

def aggregate_simulation_results(all_results, all_arrival_patterns):
    """
    Aggregate results from multiple simulation iterations
    
    Args:
        all_results (list): List of results from each iteration
        all_arrival_patterns (list): List of arrival patterns from each iteration
        
    Returns:
        dict: Aggregated simulation results
    """
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
    
    # MOS utilization
    all_mos_utilizations = []
    for result in all_results:
        if 'class_mos_utilization' in result:
            all_mos_utilizations.extend(result['class_mos_utilization'])
    
    # Student progression
    # We'll just use the last iteration for this since it's a time series
    student_progression = all_results[-1]['student_progression'] if all_results else []
    
    # Calculate throughput (graduates per month)
    throughput = 0
    if all_results:
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
    
    # Calculate resource utilization
    avg_utilization = np.mean([u['utilization'] for u in flattened_utilizations]) if flattened_utilizations else 0
    
    # Aggregate MOS metrics
    aggregated_mos_metrics = {}
    
    for result in all_results:
        if 'mos_metrics' in result:
            mos_metrics = result['mos_metrics']
            
            for mos, metrics in mos_metrics.items():
                if mos not in aggregated_mos_metrics:
                    aggregated_mos_metrics[mos] = {
                        'count': 0,
                        'completion_times': [],
                        'wait_times': []
                    }
                
                aggregated_mos_metrics[mos]['count'] += metrics['count']
                
                if metrics['count'] > 0:
                    aggregated_mos_metrics[mos]['completion_times'].append(metrics['avg_completion_time'])
                    aggregated_mos_metrics[mos]['wait_times'].append(metrics['avg_wait_time'])
    
    # Calculate average metrics for each MOS
    for mos, metrics in aggregated_mos_metrics.items():
        metrics['avg_completion_time'] = np.mean(metrics['completion_times']) if metrics['completion_times'] else 0
        metrics['avg_wait_time'] = np.mean(metrics['wait_times']) if metrics['wait_times'] else 0
        
        # Clean up temporary lists
        del metrics['completion_times']
        del metrics['wait_times']
    
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
    
    # Aggregate arrival patterns
    aggregated_arrival_patterns = {
        'avg_days_before': np.mean([patterns['avg_days_before'] for patterns in all_arrival_patterns]),
        'monthly_distribution': {}
    }
    
    # Aggregate monthly distributions
    if all_arrival_patterns:
        # Get all month names from all patterns
        all_months = set()
        for pattern in all_arrival_patterns:
            all_months.update(pattern.get('monthly_distribution', {}).keys())
        
        # Calculate average percentage for each month
        for month in all_months:
            aggregated_arrival_patterns['monthly_distribution'][month] = np.mean(
                [pattern.get('monthly_distribution', {}).get(month, 0) for pattern in all_arrival_patterns]
            )
    
    return {
        'avg_completion_time': avg_completion_time,
        'avg_wait_time': avg_wait_time,
        'throughput': throughput,
        'resource_utilization': avg_utilization,
        'bottlenecks': aggregated_bottlenecks,
        'class_utilization': aggregated_utilization,
        'class_mos_utilization': all_mos_utilizations,
        'mos_metrics': aggregated_mos_metrics,
        'student_progression': student_progression,
        'completion_times': flattened_completion_times,
        'detailed_metrics': detailed_metrics,
        'arrival_patterns': aggregated_arrival_patterns
    }