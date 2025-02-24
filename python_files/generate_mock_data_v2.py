#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta
import json
import random

from comorbidity_info import CHARLSON_CONDITIONS, CONDITION_WEIGHTS

# In[12]:

total_weight = sum(CONDITION_WEIGHTS.values())
print(f"Total weight: {total_weight}")


# In[13]:


def create_database():
    """Create SQLite database and tables"""
    conn = sqlite3.connect('healthcare.db')
    cursor = conn.cursor()
    
    # Create tables
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS provider (
        provider_id TEXT PRIMARY KEY,
        latitude REAL,
        longitude REAL
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS geolocation (
        patient_id TEXT PRIMARY KEY,
        census_block TEXT,
        latitude REAL,
        longitude REAL
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS travel_time (
        census_block TEXT,
        provider_id TEXT,
        travel_time_type TEXT,
        travel_time_minutes REAL,
        PRIMARY KEY (census_block, provider_id, travel_time_type)
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS demographics (
        patient_id TEXT PRIMARY KEY,
        birth_date DATE,
        sex TEXT,
        race TEXT,
        ethnicity TEXT,
        education TEXT,
        income REAL
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS rucc (
        census_block TEXT PRIMARY KEY,
        rucc_code INTEGER
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS encounter (
        patient_id TEXT,
        encounter_id TEXT PRIMARY KEY,
        start_date DATE,
        end_date DATE
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS procedure_table (
        patient_id TEXT,
        encounter_id TEXT,
        provider_id TEXT,
        procedure_code TEXT,
        start_datetime DATETIME,
        end_datetime DATETIME,
        FOREIGN KEY (encounter_id) REFERENCES encounter(encounter_id)
    )
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS diagnosis (
        patient_id TEXT,
        encounter_id TEXT,
        diagnosis_code TEXT,
        vocabulary_id TEXT,
        diagnosis_date DATE,
        FOREIGN KEY (patient_id) REFERENCES demographics(patient_id),
        FOREIGN KEY (encounter_id) REFERENCES encounter(encounter_id)
    )
    ''')
    
    conn.commit()
    return conn

def load_existing_data(conn, patient_locations, clinic_locations, travel_times):
    """Load existing location and travel time data, and RUCC data"""
    cursor = conn.cursor()
    
    # clear all
    cursor.execute('DELETE FROM provider')
    cursor.execute('DELETE FROM geolocation')
    cursor.execute('DELETE FROM travel_time')
    cursor.execute('DELETE FROM demographics')
    cursor.execute('DELETE FROM rucc')
    cursor.execute('DELETE FROM encounter')
    cursor.execute('DELETE FROM procedure_table')
    cursor.execute('DELETE FROM diagnosis')
    
    # Load provider (clinic) data
    provider_data = []
    for i, (lat, lon) in enumerate(clinic_locations):
        provider_id = f'PR{str(i+1).zfill(3)}'
        provider_data.append((provider_id, lat, lon))
    
    cursor.executemany('INSERT INTO provider (provider_id, latitude, longitude) VALUES (?, ?, ?)',
                      provider_data)
    
    # Load patient location data
    patient_data = []
    for i, location in enumerate(patient_locations):  # patient_locations
        patient_id = f'P{str(i+1).zfill(3)}'       # P-ID: P001-P100
        census_block = f'CB{str(i).zfill(3)}'      # census block: CB000-CB099
        patient_data.append((
            patient_id,
            census_block,
            location['lat'],
            location['lon']
        ))
            
    cursor.executemany('INSERT INTO geolocation (patient_id, census_block, latitude, longitude) VALUES (?, ?, ?, ?)',
                      patient_data)
    
    # Load travel time data
    travel_time_data = []
    seen_combinations = set()
    
    for i, times in enumerate(travel_times):
        census_block = f'CB{str(i).zfill(3)}'
        for j, time in enumerate(times.values()):
            provider_id = f'PR{str(j+1).zfill(3)}'
            combination = (census_block, provider_id, 'DRIVING')
            
            if combination not in seen_combinations:
                seen_combinations.add(combination)
                travel_time_data.append((census_block, provider_id, 'DRIVING', float(time) * 60))
    
    cursor.executemany('INSERT INTO travel_time (census_block, provider_id, travel_time_type, travel_time_minutes) VALUES (?, ?, ?, ?)',
                      travel_time_data)

    # Load RUCC data from CSV
    rucc_df = pd.read_csv('input_data/rucc_codes.csv')
    rucc_data = [(row['census_block'], row['rucc_code']) 
                 for _, row in rucc_df.iterrows()]
    
    cursor.executemany('INSERT INTO rucc (census_block, rucc_code) VALUES (?, ?)',
                      rucc_data)
    
    conn.commit()

def generate_mock_data(conn, n_patients=100):
    """Generate mock demographic and medical data"""
    cursor = conn.cursor()
    
    # Generate demographics with correlations
    current_date = datetime.now()
    demographics_data = []
    for i in range(n_patients):
        patient_id = f'P{str(i+1).zfill(3)}'
        
        # Age between 45-80
        age = np.random.normal(62, 10)
        age = max(45, min(80, age))
        birth_date = current_date - timedelta(days=int(age*365.25))
        
        # Correlated demographics
        education_level = np.random.choice(
            ['High School', 'Some College', 'Bachelor', 'Graduate'],
            p=[0.3, 0.3, 0.25, 0.15]
        )
        
        # Income correlated with education
        base_income = {
            'High School': 40000,
            'Some College': 55000,
            'Bachelor': 70000,
            'Graduate': 85000
        }
        income = np.random.normal(base_income[education_level], 10000)
        
        demographics_data.append((
            patient_id,
            birth_date.strftime('%Y-%m-%d'),
            np.random.choice(['M', 'F']),
            np.random.choice(['White', 'Black', 'Asian', 'Other'], p=[0.7, 0.15, 0.1, 0.05]),
            np.random.choice(['Hispanic', 'Non-Hispanic'], p=[0.15, 0.85]),
            education_level,
            income
        ))
    
    cursor.executemany('''
    INSERT INTO demographics 
    (patient_id, birth_date, sex, race, ethnicity, education, income)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', demographics_data)
    
    # Generate encounters and procedures
    encounter_data = []
    procedure_data = []
    
    for i in range(n_patients):
        patient_id = f'P{str(i+1).zfill(3)}'
        
        # 1-3 encounters per patient
        n_encounters = np.random.randint(1, 4)
        
        for j in range(n_encounters):
            encounter_id = f'E{patient_id}_{j}'
            
            # Generate dates within last 2 years
            start_date = current_date - timedelta(days=np.random.randint(1, 730))
            end_date = start_date + timedelta(days=np.random.randint(1, 5))
            
            encounter_data.append((
                patient_id,
                encounter_id,
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d')
            ))
            
            # 30% chance of CRC screening procedure
            if np.random.random() < 0.3:
                procedure_code = np.random.choice(['45378', '45380', '45384', '45385'])
                provider_id = f'PR{str(np.random.randint(1, 17)).zfill(3)}'
                
                procedure_start = datetime.combine(start_date, 
                                                datetime.strptime(f"{np.random.randint(9,17)}:00", "%H:%M").time())
                procedure_end = procedure_start + timedelta(hours=np.random.randint(1, 4))
                
                procedure_data.append((
                    patient_id,
                    encounter_id,
                    provider_id,
                    procedure_code,
                    procedure_start.strftime('%Y-%m-%d %H:%M:%S'),
                    procedure_end.strftime('%Y-%m-%d %H:%M:%S')
                ))
    
    cursor.executemany('''
    INSERT INTO encounter 
    (patient_id, encounter_id, start_date, end_date)
    VALUES (?, ?, ?, ?)
    ''', encounter_data)
    
    cursor.executemany('''
    INSERT INTO procedure_table 
    (patient_id, encounter_id, provider_id, procedure_code, start_datetime, end_datetime)
    VALUES (?, ?, ?, ?, ?, ?)
    ''', procedure_data)

    # Generate diagnosis data with Charlson Comorbidities
    diagnosis_data = []
    for encounter in encounter_data:
        patient_id = encounter[0]
        encounter_id = encounter[1]
        start_date = datetime.strptime(encounter[2], '%Y-%m-%d')
        end_date = datetime.strptime(encounter[3], '%Y-%m-%d')
        
        # 30% pat with 1-3 diseases:
        if np.random.random() < 0.3:
            
            n_conditions = np.random.randint(1, 4)
            
            selected_conditions = random.choices(
                list(CONDITION_WEIGHTS.keys()),
                weights=list(CONDITION_WEIGHTS.values()),
                k=n_conditions
            )
            
            for condition in selected_conditions:
                # ICO version (80% ICD-10, 20% ICD-9)
                version = '10' if random.random() < 0.8 else '9'
                
                # get codes from CHARLSON_CONDITIONS
                codes = CHARLSON_CONDITIONS[condition][version]
                selected_code = random.choice(codes)
                
                # generate diagnosis time
                diag_datetime = start_date + timedelta(
                    hours=random.randint(0, 24)
                )
                
                diagnosis_data.append((
                    patient_id,
                    encounter_id,
                    selected_code,
                    f'ICD{version}CM',
                    diag_datetime.strftime('%Y-%m-%d')
                ))

    cursor.executemany('''
    INSERT INTO diagnosis 
    (patient_id, encounter_id, diagnosis_code, vocabulary_id, diagnosis_date)
    VALUES (?, ?, ?, ?, ?)
    ''', diagnosis_data)
    
    conn.commit()


# In[14]:


def verify_data(conn):
    """Verify the generated data and show key statistics"""
    cursor = conn.cursor()
    
    print("Table Record Counts:")
    print("-" * 50)
    for table in ['provider', 'geolocation', 'travel_time', 'demographics', 'rucc', 'encounter', 'procedure_table']:
        count = cursor.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0]
        print(f"{table}: {count} records")
    
    print("\nSample Patient Data:")
    print("-" * 50)
    query = """
    SELECT 
        g.patient_id,
        g.census_block,
        d.sex,
        d.birth_date,
        d.education,
        d.race,
        d.ethnicity,
        r.rucc_code,
        COUNT(DISTINCT e.encounter_id) as num_encounters,
        COUNT(DISTINCT p.procedure_code) as num_procedures,
        MIN(t.travel_time_minutes) as min_travel_time
    FROM geolocation g
    JOIN demographics d ON g.patient_id = d.patient_id
    JOIN rucc r ON g.census_block = r.census_block
    LEFT JOIN encounter e ON g.patient_id = e.patient_id
    LEFT JOIN procedure_table p ON e.encounter_id = p.encounter_id
    LEFT JOIN travel_time t ON g.census_block = t.census_block
    GROUP BY g.patient_id
    LIMIT 5
    """
    df = pd.read_sql_query(query, conn)
    print(df)
    
    print("\nKey Statistics:")
    print("-" * 50)

    # Charlson Comorbidities
    print("\nCharlson Comorbidities Distribution:")
    for condition, codes in CHARLSON_CONDITIONS.items():
        query = f"""
        SELECT COUNT(DISTINCT patient_id) as count
        FROM diagnosis
        WHERE (vocabulary_id = 'ICD9CM' AND diagnosis_code IN ('{"','".join(codes['9'])}'))
        OR (vocabulary_id = 'ICD10CM' AND diagnosis_code IN ('{"','".join(codes['10'])}'))
        """
        count = cursor.execute(query).fetchone()[0]
        print(f"{condition}: {count} patients")
    
    screening_query = """
    SELECT 
        COUNT(DISTINCT CASE WHEN procedure_code IN ('45378', '45380', '45384', '45385') 
              THEN patient_id END) * 100.0 / COUNT(DISTINCT patient_id) as screening_rate
    FROM procedure_table
    """
    screening_rate = pd.read_sql_query(screening_query, conn).iloc[0,0]
    print(f"CRC Screening Rate: {screening_rate:.1f}%")
    
    encounters_query = """
    SELECT AVG(encounter_count) as avg_encounters
    FROM (
        SELECT patient_id, COUNT(*) as encounter_count
        FROM encounter
        GROUP BY patient_id
    )
    """
    avg_encounters = pd.read_sql_query(encounters_query, conn).iloc[0,0]
    print(f"Average encounters per patient: {avg_encounters:.1f}")
    
    # Travel time 
    travel_time_query = """
    SELECT 
        MIN(travel_time_minutes) as min_time,
        AVG(travel_time_minutes) as avg_time,
        MAX(travel_time_minutes) as max_time
    FROM travel_time
    """
    travel_times = pd.read_sql_query(travel_time_query, conn)
    print("\nTravel Time Distribution (minutes):")
    print(f"Min: {travel_times.iloc[0,0]:.1f}")
    print(f"Avg: {travel_times.iloc[0,1]:.1f}")
    print(f"Max: {travel_times.iloc[0,2]:.1f}")



# In[15]:


def export_tables_to_csv(conn):
    """Export all tables from database to CSV files"""
    tables = [
        'geolocation', 'demographics', 'encounter', 
        'procedure_table', 'provider', 'rucc', 'travel_time','diagnosis'
    ]
    
    for table in tables:
        query = f"SELECT * FROM {table}"
        df = pd.read_sql_query(query, conn)
        df.to_csv(f'output_data/{table}.csv', index=False)
        print(f"Exported {table}.csv with {len(df)} rows")

        print(f"\nFirst few rows of {table}:")
        print(df.head())
        print("\n" + "="*50 + "\n")
        
def close_all_connections():
    import sqlite3
    sqlite3.connect('healthcare.db').close()

def ensure_fresh_database():
    import os
    if os.path.exists('healthcare.db'):
        try:
            os.remove('healthcare.db')
        except PermissionError:
            print("Could not remove existing database. Please close any programs that might be using it.")
            return False
    return True

def main():

    close_all_connections()
    
    if not ensure_fresh_database():
        return
    
    conn = create_database()

    try:
        # Load your existing data
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
        
        # Load existing data into database and generate mock data
        load_existing_data(conn, patient_locations, clinic_locations, travel_times)
        generate_mock_data(conn)

        # verify data
        verify_data(conn)
        
        # Export all tables to CSV
        export_tables_to_csv(conn)
        
    finally:
        conn.close()


# In[16]:


if __name__ == "__main__":
    main()


# In[ ]:




