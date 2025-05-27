#!/usr/bin/env python
# coding: utf-8

import os
import json
import shutil
import sqlite3
from datetime import datetime
import pandas as pd
from generate_mock_data_v3 import (
    MedicalDataGenerator, 
    create_database, 
    load_existing_data, 
    generate_mock_data,
    export_tables_to_csv
)

def export_tables_to_custom_dir(conn, output_dir):
    """
    Export database tables to CSV files in the specified directory
    
    Parameters:
    conn: sqlite3.Connection, database connection
    output_dir: str, output directory
    """
    tables = [
        'geolocation', 'demographics', 'encounter', 
        'procedure_table', 'provider', 'rucc', 'travel_time','diagnosis'
    ]
    
    for table in tables:
        query = f"SELECT * FROM {table}"
        df = pd.read_sql_query(query, conn)
        df.to_csv(f'{output_dir}/{table}.csv', index=False)
        print(f"Exported {table}.csv with {len(df)} rows")

        print(f"\nFirst few rows of {table}:")
        print(df.head())
        print("\n" + "="*50 + "\n")

def generate_dataset(time_effects, output_dir, seed=42):
    """
    Generate a single dataset
    
    Parameters:
    time_effects: dict, travel time effect configuration
    output_dir: str, output directory
    seed: int, random seed
    """
    print(f"\nStarting dataset generation, output directory: {output_dir}")
    
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create database connection
    conn = create_database()
    
    try:
        # Load location data
        with open('input_data/sampled_block_group_centers_100_30.json', 'r') as f:
            patient_locations = json.load(f)
            
        clinic_locations = [
            [40.40655, -86.8321528],
            [40.7344392, -86.77769099999999],
            [40.2765035, -86.4970488],
            [39.9164485, -86.1557417],
            [39.7805894, -86.3405844],
            [39.7775523, -86.1837364],
            [39.79052859999999, -86.16338739999999],
            [39.7756075, -86.1761174],
            [39.9868449, -85.929307],
            [39.6379321, -86.1593584],
            [40.2247576, -85.4507319],
            [39.2893255, -86.7867983],
            [39.9075207, -85.3861367],
            [39.1606644, -86.55537140000001],
            [38.8599541, -86.51307659999999],
            [38.56829949999999, -86.47532799999999]
        ]
        
        with open('input_data/ExactTravelTimeDatafromAllMatrix.json', 'r') as f:
            travel_times = json.load(f)
        
        # Load base data
        load_existing_data(conn, patient_locations, clinic_locations, travel_times)
        
        # Create data generator instance
        medical_generator = MedicalDataGenerator(seed=seed)
        medical_generator.base_params['encounter']['time_effects'] = time_effects
        
        # Generate data
        generate_mock_data(conn)
        
        # Export data to specified directory
        export_tables_to_custom_dir(conn, output_dir)
        
        # Copy database file to output directory
        shutil.copy('healthcare.db', os.path.join(output_dir, 'healthcare.db'))
        
        print(f"Dataset generation completed: {output_dir}")
        
    except Exception as e:
        print(f"Error during dataset generation: {str(e)}")
        raise
    finally:
        conn.close()

def main():
    """Main function: Generate four datasets with different travel time effects"""
    
    # Define four different time effects
    time_effects_configs = {
        'strong': {
            0.5: 3.0,   # 30 min
            1.0: 1.5,   # 1 hour
            2.0: 0.1,   # 2 hours
            3.0: 0.01,  # 3 hours
            float('inf'): 0.001  # 3 hours above
        },
        'medium': {
            0.5: 2.0,   # 30 min
            1.0: 1.0,   # 1 hour
            2.0: 0.2,   # 2 hours
            3.0: 0.05,  # 3 hours
            float('inf'): 0.01  # 3 hours above
        },
        'weak': {
            0.5: 1.5,   # 30 min
            1.0: 1.2,   # 1 hour
            2.0: 0.8,   # 2 hours
            3.0: 0.5,   # 3 hours
            float('inf'): 0.3  # 3 hours above
        },
        'none': {
            0.5: 1.0,   # 30 min
            1.0: 1.0,   # 1 hour
            2.0: 1.0,   # 2 hours
            3.0: 1.0,   # 3 hours
            float('inf'): 1.0  # 3 hours above
        }
    }
    
    # Create main output directory
    base_output_dir = 'output_data/multiple_datasets'
    if not os.path.exists(base_output_dir):
        os.makedirs(base_output_dir)
    
    # Generate dataset for each configuration
    for effect_type, time_effects in time_effects_configs.items():
        output_dir = os.path.join(base_output_dir, f'{effect_type}_travel_effect')
        print(f"\nGenerating dataset with {effect_type} travel time effect...")
        generate_dataset(time_effects, output_dir)
        
        # Print dataset information
        print(f"\n{effect_type} dataset information:")
        print(f"Output directory: {output_dir}")
        print(f"Travel time effect configuration: {time_effects}")

if __name__ == "__main__":
    main()