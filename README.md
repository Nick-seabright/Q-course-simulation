# Q-Course Training Schedule Optimizer

A comprehensive application for simulating and optimizing military training schedules with support for MOS paths, historical data analysis, and advanced schedule optimization.

## Overview

The Q-Course Training Schedule Optimizer helps military training planners create, simulate, and optimize training schedules for special forces qualification pipelines. It supports different Military Occupational Specialty (MOS) paths (18A, 18B, 18C, 18D, 18E), complex course prerequisites, and resource constraints to create realistic simulations of student flow through the training pipeline.

![Training Schedule Optimizer](https://github.com/Nick-seabright/Q-course-simulation/raw/main/screenshot.png)

## Features

### Data Analysis
- Upload and analyze historical training data
- Extract patterns including pass rates, recycle rates, MOS distributions, and arrival patterns
- Visualize key metrics and historical trends with interactive charts
- Automatically assess data quality and identify potential issues

### Career Path Builder
- Create custom career paths for each MOS specialty
- Define mandatory sequential courses in a career path
- Support for "OR" relationships (complete one course from a group)
- Configure flexible courses with position constraints (must be taken between specific courses)
- Visualize career paths with an interactive diagram

### Course Configuration
- Define prerequisites for each course using flexible AND/OR logic
- Configure MOS-specific training paths
- Set capacity constraints, class frequency, reserved seats, and officer-enlisted ratios
- View historical data to inform configuration decisions
- Save and load configurations for sharing and backup

### Schedule Building
- Interactive timeline for visualizing and building schedules
- Specify MOS allocations for each class
- Quick date setting options based on historical course durations
- Bulk schedule creation for rapid schedule development
- Intelligent conflict detection and validation

### Simulation
- Simulate student progression through the training pipeline
- Model different arrival patterns based on historical data
- Track MOS-specific metrics and bottlenecks
- Analyze resource utilization and waiting times
- Compare multiple simulation scenarios

### Optimization
- Automatically suggest schedule improvements
- Optimize class capacities, dates, and MOS allocations
- Balance different optimization goals (completion time, throughput, wait times)
- Compare optimized schedule to original with detailed metrics
- Apply selective optimizations to your schedule

## Getting Started

### Prerequisites
- Python 3.8 or higher
- pip for package installation

### Installation

1. Clone this repository:
```
git clone https://github.com/Nick-seabright/Q-course-simulation.git
cd Q-course-simulation
```

2. Install required packages:
```
pip install -r requirements.txt
```

3. Run the application:
```
streamlit run app.py
```

## Usage Guide

### 1. Upload Data
Start by uploading your historical training data in CSV format. The application accepts data with various columns including student information, course details, and completion status.

### 2. Create Career Paths
Define career paths for each MOS specialty using the Career Path Builder:
- Add courses to the main sequential path
- Create OR groups where students can choose one course from a group
- Add flexible courses with before/after constraints
- Apply career paths to generate prerequisites automatically

### 3. Configure Courses
For each course in your training pipeline:
- Set course prerequisites using AND/OR logic or MOS-specific paths
- Define class capacity and frequency constraints
- Configure reserved seats and officer-enlisted ratios
- Review historical performance metrics

### 4. Build Schedule
Create your training schedule by:
- Selecting courses to add to the calendar
- Setting start and end dates for classes
- Configuring MOS allocations for each class
- Using bulk scheduling for recurring classes
- Reviewing the schedule visualization for conflicts or issues

### 5. Run Simulation
Simulate student flow through your schedule:
- Choose to use historical data patterns or custom settings
- Set simulation parameters (number of students, iterations)
- Review results including bottlenecks, waiting times, and MOS-specific metrics
- Compare different simulation scenarios

### 6. Optimize Schedule
Improve your schedule based on simulation results:
- Select optimization goals (completion time, throughput, wait times)
- Review recommended changes to class dates, capacities, and MOS allocations
- Apply suggested optimizations and re-run simulation to verify improvements

## Advanced Features

### Custom Prerequisite Logic
The application supports complex prerequisite relationships:
- Standard AND prerequisites (must complete ALL courses)
- OR prerequisites (must complete ANY ONE course from a group)
- Complex AND/OR combinations (must complete ALL courses from group A AND ANY ONE from group B)
- MOS-specific prerequisites (different requirements based on specialty)

### Historical Data Integration
Leverage historical data to inform your planning:
- Automatically extract pass rates, recycle rates, and durations
- Model student arrivals based on historical patterns
- Use historical MOS distributions to simulate realistic student flows
- Start simulations with students already in the pipeline based on current state

### Performance Analytics
Get detailed insights into your training pipeline:
- Identify bottlenecks and waiting periods
- Analyze resource utilization by course and MOS
- Track completion times across different specialties
- Visualize student flow through the training pipeline

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Developed for Special Forces qualification course planning
- Built with Streamlit, Pandas, NumPy, and Plotly

---

For questions or support, please contact nick.seabright@gmail.com