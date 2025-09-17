# Q-Course Training Schedule Optimizer

A Streamlit application for simulating and optimizing military training schedules with support for MOS paths, historical data analysis, and advanced schedule optimization.

## Overview

This application helps military training planners create, simulate, and optimize training schedules for special forces training pipelines. It supports different Military Occupational Specialty (MOS) paths (18A, 18B, 18C, 18D, 18E), course prerequisites, and resource constraints to create realistic simulations of student flow through the training pipeline.

![Schedule Optimizer Screenshot](https://example.com/screenshot.png)

## Features

### Data Analysis

- Upload and analyze historical training data
- Extract patterns including pass rates, recycle rates, MOS distributions, and arrival patterns
- Visualize key metrics and historical trends

### Course Configuration

- Define prerequisites for each course using flexible AND/OR logic
- Configure MOS-specific training paths
- Set capacity constraints, class frequency, reserved seats, and officer-enlisted ratios
- Use historical data to inform configurations

### Schedule Building

- Interactive timeline for visualizing and building schedules
- Specify MOS allocations for each class
- Quick date setting options based on historical course durations
- Visual feedback on schedule conflicts and capacity utilization

### Simulation

- Simulate student progression through the training pipeline
- Model different arrival patterns based on historical data
- Track MOS-specific metrics and bottlenecks
- Analyze resource utilization and waiting times

### Optimization

- Automatically suggest schedule improvements
- Optimize class capacities, dates, and MOS allocations
- Balance different optimization goals (completion time, throughput, wait times)
- Compare optimized schedule to original

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

Start by uploading your historical training data in CSV format. The application expects data with the following columns:
- FY (Fiscal Year)
- Course Number
- Course Title
- NAME
- SSN
- CLS (Class Number)
- Cls Start Date
- Cls End Date
- Input Stat
- Output Stat
- Training MOS (if available)

### 2. Configure Courses

For each course in your training pipeline:
1. Set course prerequisites using AND/OR logic or MOS-specific paths
2. Define class capacity and frequency constraints
3. Configure reserved seats and officer-enlisted ratios
4. Review historical performance metrics

### 3. Build Schedule

Create your training schedule by:
1. Selecting courses to add to the calendar
2. Setting start and end dates
3. Configuring MOS allocations for each class
4. Reviewing the schedule visualization for conflicts or issues

### 4. Run Simulation

Simulate student flow through your schedule:
1. Choose to use historical data patterns or custom settings
2. Set simulation parameters (number of students, iterations)
3. Review results including bottlenecks, waiting times, and MOS-specific metrics

### 5. Optimize Schedule

Improve your schedule based on simulation results:
1. Select optimization goals (completion time, throughput, wait times)
2. Review recommended changes
3. Apply suggested optimizations and re-run simulation to verify improvements

## Key Components

- **app.py**: Main Streamlit application
- **data_processor.py**: Functions for processing and analyzing training data
- **simulation_engine.py**: Simulation logic for student progression
- **optimization.py**: Schedule optimization algorithms
- **utils.py**: Utility functions for dates, validation, and analysis

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Developed for Special Forces qualification course planning
- Built with Streamlit, Pandas, NumPy, and Plotly

---

For questions or support, please contact [your.email@example.com](mailto:your.email@example.com)