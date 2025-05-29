# Cox Model Analysis Project

This project consists of two main steps:
1. Mock Healthcare Data Generation
2. Cox Model Analysis with/without Privacy Considerations

## cd to python_files folder

## Step 1: Data Generation

### Prerequisites
- Python 3.x
- Required libraries: pandas, numpy, sqlite3
- Input data files in `input_data/` directory:
  - sampled_block_group_centers_100_30.json
  - ExactTravelTimeDatafromAllMatrix.json
  - rucc_codes.csv

These files are existing data.

### Data Generation Process
1. Create SQLite database with required tables:
   - provider (healthcare providers)
   - geolocation (patient locations)
   - travel_time (travel times between locations)
   - demographics (patient demographic information)
   - rucc (Rural-Urban Commuting Codes)
   - encounter (medical encounters)
   - procedure_table (medical procedures)
   - diagnosis (patient diagnoses)

2. Generate mock data:
   - Load existing geographic data
   - Generate correlated demographic information
   - Create medical encounters and procedures
   - Add diagnoses based on Charlson Comorbidity Index
   - Export all tables to CSV files
   - Merge tables into a single dataset

### Running Data Generation
```bash
python generate_mock_data_v3.py
python merge_tables_v3.py
```
also can use --seed parameter
```bash
python generate_mock_data_v3.py --seed 42
```
Good seeds: 42, 999, 456, 789


## Part 2: Cox Analysis

### Files
- `cox_analysis_privacy.py`: Enhanced Cox analysis with privacy protections

### Features
- Privacy checks to exclude groups with n<20 (min_size)
- Visualization and analysis summary of results with and without privacy enforcement
- Comprehensive model statistics and comparisons
- Forest plots for coefficient visualization

### Running Analysis
```bash
python cox_analysis_privacy_v3.py
```

## Prat 3: Multiple Dataset Generation with Different settings
```bash
python generate_multiple_datasets.py
```