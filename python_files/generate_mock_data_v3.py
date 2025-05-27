#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta
import json
import random
import argparse
from scipy.stats import norm, multivariate_normal
from typing import Dict, List, Tuple, Optional
import os
import shutil

from comorbidity_info import CHARLSON_CONDITIONS, CONDITION_WEIGHTS

# Default seed is 42, other good seeds: 999, 456, 789
DEFAULT_SEED = 42
# DEFAULT_SEED = 999

# Create command line argument parser
parser = argparse.ArgumentParser(description='Generate mock healthcare data with specified random seed')
parser.add_argument('--seed', type=int, default=DEFAULT_SEED, help='Random seed for data generation (default: 42)')
args = parser.parse_args()

# Set random seed
SEED = args.seed
np.random.seed(SEED)
random.seed(SEED)

print(f"Using random seed: {SEED}")

# generate demographics data
class DemographicsGenerator:
    def __init__(self, seed: int = 42, reference_date: datetime = datetime(2024, 1, 1)):
        np.random.seed(seed)
        self.current_date = reference_date
        
        # Indiana demographics reference data
        self.indiana_demographics = {
            'age_distribution': {
                'mean': 62,
                'std': 10,
                'bounds': (45, 80),
                'weights': {
                    '45-54': 0.25,
                    '55-64': 0.35,
                    '65-74': 0.25,
                    '75-80': 0.15
                }
            },
            'education_by_age': {
                '45-54': {'High School': 0.25, 'Some College': 0.35, 'Bachelor': 0.25, 'Graduate': 0.15},
                '55-64': {'High School': 0.30, 'Some College': 0.30, 'Bachelor': 0.25, 'Graduate': 0.15},
                '65-74': {'High School': 0.35, 'Some College': 0.25, 'Bachelor': 0.25, 'Graduate': 0.15},
                '75-80': {'High School': 0.40, 'Some College': 0.25, 'Bachelor': 0.20, 'Graduate': 0.15}
            },
            'income_parameters': {
                'base_values': {
                    'High School': 40000,
                    'Some College': 55000,
                    'Bachelor': 70000,
                    'Graduate': 85000
                },
                'age_adjustment': {
                    '45-54': 1.1,
                    '55-64': 1.0,
                    '65-74': 0.8,
                    '75-80': 0.7
                },
                'gender_factor': {
                    'M': 1.0,
                    'F': 0.85
                }
            },
            'race_ethnicity': {
                'distributions': {
                    'urban': {
                        'race': {'White': 0.65, 'Black': 0.20, 'Asian': 0.10, 'Other': 0.05},
                        'ethnicity': {'Hispanic': 0.18, 'Non-Hispanic': 0.82}
                    },
                    'rural': {
                        'race': {'White': 0.80, 'Black': 0.10, 'Asian': 0.05, 'Other': 0.05},
                        'ethnicity': {'Hispanic': 0.10, 'Non-Hispanic': 0.90}
                    }
                }
            }
        }
        
        # correlation matrix
        self.correlation_matrix = np.array([
            [1.0,  0.3,  0.4], # age
            [0.3,  1.0,  0.6], # education
            [0.4,  0.6,  1.0]  # income
        ])

    def _generate_correlated_variables(self, n_samples: int) -> np.ndarray:
        """generate correlated standard normal random variables"""
        return multivariate_normal.rvs(
            mean=[0, 0, 0],
            cov=self.correlation_matrix,
            size=n_samples
        )

    def _get_age_category(self, age: float) -> str:
        if 45 <= age < 55: return '45-54'
        elif 55 <= age < 65: return '55-64'
        elif 65 <= age < 75: return '65-74'
        else: return '75-80'

    def _generate_education_level(self, age: float, random_factor: float) -> str:
        age_category = self._get_age_category(age)
        probabilities = self.indiana_demographics['education_by_age'][age_category]
        
        adjusted_probs = {k: v * (1 + random_factor * 0.2) for k, v in probabilities.items()}
        total = sum(adjusted_probs.values())
        adjusted_probs = {k: v/total for k, v in adjusted_probs.items()}
        
        return np.random.choice(
            list(adjusted_probs.keys()),
            p=list(adjusted_probs.values())
        )

    def _generate_income(self, age: float, education: str, gender: str, random_factor: float) -> float:
        base_income = self.indiana_demographics['income_parameters']['base_values'][education]
        age_category = self._get_age_category(age)
        age_factor = self.indiana_demographics['income_parameters']['age_adjustment'][age_category]
        gender_factor = self.indiana_demographics['income_parameters']['gender_factor'][gender]
        
        income = base_income * age_factor * gender_factor
        income *= (1 + random_factor * 0.2)
        
        return max(20000, min(150000, income))

    def _generate_race_ethnicity(self, is_urban: bool) -> Tuple[str, str]:
        distribution = self.indiana_demographics['race_ethnicity']['distributions']
        dist_type = 'urban' if is_urban else 'rural'
        
        race = np.random.choice(
            list(distribution[dist_type]['race'].keys()),
            p=list(distribution[dist_type]['race'].values())
        )
        ethnicity = np.random.choice(
            list(distribution[dist_type]['ethnicity'].keys()),
            p=list(distribution[dist_type]['ethnicity'].values())
        )
        
        return race, ethnicity

    def _generate_gender(self, age: float) -> str:
        """generate gender based on age, considering gender distribution difference by age category"""
        age_category = self._get_age_category(age)
        
        # gender distribution by age category
        gender_distribution = {
            '45-54': 0.51,  # 51% female
            '55-64': 0.52,
            '65-74': 0.54,
            '75-80': 0.56   # older women have higher proportion
        }
        
        female_prob = gender_distribution[age_category]
        return np.random.choice(['M', 'F'], p=[1-female_prob, female_prob])

    def generate_demographics(self, n_patients: int, rucc_codes: List[int]) -> pd.DataFrame:
        correlated_vars = self._generate_correlated_variables(n_patients)
        
        demographics_data = []
        for i in range(n_patients):
            patient_id = f'P{str(i+1).zfill(3)}'
            
            # generate age using correlated random variables
            age = norm.ppf(norm.cdf(correlated_vars[i, 0])) * 10 + 62
            age = max(45, min(80, age))
            birth_date = self.current_date - timedelta(days=int(age*365.25))
            
            # generate gender (considering age influence)
            gender = self._generate_gender(age)
            
            # generate education
            education = self._generate_education_level(age, correlated_vars[i, 1])
            
            # generate income
            income = self._generate_income(age, education, gender, correlated_vars[i, 2])
            
            # determine if urban area based on RUCC code
            is_urban = rucc_codes[i] <= 3
            race, ethnicity = self._generate_race_ethnicity(is_urban)
            
            demographics_data.append({
                'patient_id': patient_id,
                'birth_date': birth_date.strftime('%Y-%m-%d'),
                'sex': gender,
                'race': race,
                'ethnicity': ethnicity,
                'education': education,
                'income': income
            })
        
        return pd.DataFrame(demographics_data)

# generate medical data
class MedicalDataGenerator:
    def __init__(self, seed: int = 42, reference_date: datetime = datetime(2024, 1, 1)):
        np.random.seed(seed)
        self.current_date = reference_date
        
        # negative binomial distribution parameters
        self.nb_params = {
            'alpha': 0.5,  # dispersion parameter
        }
        
        # base parameters
        self.base_params = {
            'encounter': {
                'base_lambda': 0.3,  # base encounter rate
                'age_weights': {
                    '45-60': 1.0,
                    '60-75': 1.3,
                    '75+': 1.6
                },
                'season_weights': {
                    1: 1.3, 2: 1.3, 12: 1.3,  # winter
                    6: 0.8, 7: 0.8, 8: 0.8,   # summer
                    3: 1.0, 4: 1.0, 5: 1.0,   # spring
                    9: 1.0, 10: 1.0, 11: 1.0  # autumn
                },
                'time_effects': {  
                    0.5: 2.0,   # 30 min
                    1.0: 1.0,   # 1 hour
                    2.0: 0.2,   # 2 hours
                    3.0: 0.05,  # 3 hours
                    float('inf'): 0.01  # 3 hours above
                }
            },
            'diagnosis': {
                'age_risk_factors': {
                    '45-60': 1.0,
                    '60-75': 1.3,
                    '75+': 1.6
                }
            }
        }
        
        # colonoscopy_related_conditions definition
        self.colonoscopy_related_conditions = {
            'mal': {  # malignant tumor
                'probability': 0.8,  # higher screening probability
                'procedure_weights': {
                    '45378': 0.2,  # diagnostic colonoscopy
                    '45380': 0.3,  # biopsy
                    '45384': 0.2,  # lesion removal
                    '45385': 0.3   # polyp removal
                }
            },
            'mst': {  # metastatic tumor
                'probability': 0.7,
                'procedure_weights': {
                    '45378': 0.3,
                    '45380': 0.4,
                    '45384': 0.2,
                    '45385': 0.1
                }
            },
            'WL': {   # weight loss
                'probability': 0.4,
                'procedure_weights': {
                    '45378': 0.7,
                    '45380': 0.2,
                    '45384': 0.05,
                    '45385': 0.05
                }
            },
            'pud': {  # peptic ulcer disease
                'probability': 0.5,
                'procedure_weights': {
                    '45378': 0.6,
                    '45380': 0.3,
                    '45384': 0.05,
                    '45385': 0.05
                }
            }
        }

    def _calculate_travel_time_effect(self, travel_time: float) -> float:
        for threshold, weight in sorted(self.base_params['encounter']['time_effects'].items()):
            if travel_time <= threshold:
                return weight
        return self.base_params['encounter']['time_effects'][float('inf')]

    def _calculate_monthly_lambda(self, patient: pd.Series, date: datetime, travel_time: float) -> float:
        base_lambda = self.base_params['encounter']['base_lambda']
        
        # calculate travel time effect
        time_effect = self._calculate_travel_time_effect(travel_time)
        
        # age effect
        age = (self.current_date - pd.to_datetime(patient['birth_date'])).days / 365.25
        if age <= 60:
            age_factor = self.base_params['encounter']['age_weights']['45-60']
        elif age <= 75:
            age_factor = self.base_params['encounter']['age_weights']['60-75']
        else:
            age_factor = self.base_params['encounter']['age_weights']['75+']
            
        # season effect
        season_factor = self.base_params['encounter']['season_weights'][date.month]
        
        return base_lambda * time_effect * age_factor * season_factor

    def _generate_nb_visits(self, mu: float) -> int:
        """generate the number of encounters using negative binomial distribution"""
        alpha = self.nb_params['alpha']
        p = 1 / (1 + mu * alpha)
        r = 1 / alpha
        return np.random.negative_binomial(r, p)

    def generate_encounters(self, demographics_df, start_date, end_date, conn):
        encounters = []
        
        # get travel time data
        travel_times_df = pd.read_sql_query("""
            SELECT g.patient_id, g.census_block, MIN(t.travel_time_minutes)/60.0 as min_travel_time
            FROM geolocation g
            JOIN travel_time t ON g.census_block = t.census_block
            GROUP BY g.patient_id, g.census_block
        """, conn)
        
        for _, patient in demographics_df.iterrows():
            patient_encounters = []
            current_date = start_date
            encounter_sequence = 0
            
            # get the travel time of the patient
            travel_time = travel_times_df[
                travel_times_df['patient_id'] == patient['patient_id']
            ]['min_travel_time'].iloc[0]
            
            # calculate the travel time effect
            time_effect = self._calculate_travel_time_effect(travel_time)
            
            while current_date <= end_date:
                monthly_lambda = self._calculate_monthly_lambda(patient, current_date, travel_time)
                
                # generate the number of encounters using negative binomial distribution
                n_visits = self._generate_nb_visits(monthly_lambda)
                
                if n_visits > 0:
                    if current_date.month == 12:
                        month_end = min(
                            end_date,
                            datetime(current_date.year + 1, 1, 1) - timedelta(days=1)
                        )
                    else:
                        month_end = min(
                            end_date,
                            datetime(current_date.year, current_date.month + 1, 1) - timedelta(days=1)
                        )
                    days_in_month = (month_end - current_date).days + 1
                    
                    visit_dates = sorted([
                        current_date + timedelta(days=np.random.randint(0, days_in_month))
                        for _ in range(n_visits)
                    ])
                    
                    for visit_date in visit_dates:
                        duration = np.random.geometric(p=0.5)
                        encounter_id = f"{patient['patient_id']}_{encounter_sequence}"
                        encounter_sequence += 1
                        
                        patient_encounters.append({
                            'patient_id': patient['patient_id'],
                            'encounter_id': encounter_id,
                            'encounter_sequence': encounter_sequence - 1,
                            'start_date': visit_date.strftime('%Y-%m-%d'),
                            'end_date': (visit_date + timedelta(days=duration)).strftime('%Y-%m-%d'),
                            'travel_time': travel_time,
                            'time_effect': time_effect
                        })
                
                if current_date.month == 12:
                    current_date = datetime(current_date.year + 1, 1, 1)
                else:
                    current_date = datetime(current_date.year, current_date.month + 1, 1)
            
            encounters.extend(patient_encounters)
        
        # check the uniqueness of encounter_id before converting to DataFrame
        df = pd.DataFrame(encounters)
        if df['encounter_id'].duplicated().any():
            print("warning: duplicate encounter_id")
            print(df[df['encounter_id'].duplicated(keep=False)])
        
        # return the DataFrame
        return df[['patient_id', 'encounter_id', 'start_date', 'end_date']]

    def generate_diagnoses(self, encounters_df: pd.DataFrame, demographics_df: pd.DataFrame) -> pd.DataFrame:
        diagnoses = []
        patient_diagnoses = {}
        
        for _, encounter in encounters_df.iterrows():
            patient = demographics_df[
                demographics_df['patient_id'] == encounter['patient_id']
            ].iloc[0]
            
            if encounter['patient_id'] not in patient_diagnoses:
                patient_diagnoses[encounter['patient_id']] = set()
            
            # calculate the age effect
            age = (self.current_date - pd.to_datetime(patient['birth_date'])).days / 365.25
            age_category = '45-60' if age <= 60 else ('60-75' if age <= 75 else '75+')
            age_risk = self.base_params['diagnosis']['age_risk_factors'][age_category]
            
            # diagnosis probability affected by travel time
            time_effect = encounter.get('time_effect', 1.0)
            base_prob = 0.3 * age_risk * time_effect
            
            if np.random.random() < base_prob:
                n_conditions = np.random.randint(1, 4)
                conditions = random.choices(
                    list(CONDITION_WEIGHTS.keys()),
                    weights=list(CONDITION_WEIGHTS.values()),
                    k=n_conditions
                )
                
                for condition in conditions:
                    version = '10' if random.random() < 0.8 else '9'
                    codes = CHARLSON_CONDITIONS[condition][version]
                    selected_code = random.choice(codes)
                    
                    start_date = datetime.strptime(encounter['start_date'], '%Y-%m-%d')
                    diag_datetime = start_date + timedelta(hours=random.randint(0, 24))
                    
                    diagnoses.append({
                        'patient_id': encounter['patient_id'],
                        'encounter_id': encounter['encounter_id'],
                        'diagnosis_code': selected_code,
                        'vocabulary_id': f'ICD{version}CM',
                        'diagnosis_date': diag_datetime.strftime('%Y-%m-%d')
                    })
                    
                    patient_diagnoses[encounter['patient_id']].add(condition)
        
        return pd.DataFrame(diagnoses)

    def generate_procedures(self, encounters_df: pd.DataFrame, demographics_df: pd.DataFrame, diagnoses_df: pd.DataFrame) -> pd.DataFrame:
        procedures = []
        
        for _, encounter in encounters_df.iterrows():
            patient = demographics_df[
                demographics_df['patient_id'] == encounter['patient_id']
            ].iloc[0]
            
            patient_diagnoses = diagnoses_df[
                (diagnoses_df['patient_id'] == encounter['patient_id']) &
                (diagnoses_df['encounter_id'] == encounter['encounter_id'])
            ]
            
            age = (self.current_date - pd.to_datetime(patient['birth_date'])).days / 365.25
            
            # base probability
            base_prob = 0.0
            if 45 <= age <= 75:
                base_prob = 0.15
            
            # affected by travel time
            time_effect = encounter.get('time_effect', 1.0)
            base_prob *= time_effect
            
            # adjust probability by diagnosis
            max_prob = base_prob
            selected_weights = None
            
            for _, diagnosis in patient_diagnoses.iterrows():
                condition = self._get_condition_from_code(diagnosis['diagnosis_code'])
                if condition in self.colonoscopy_related_conditions:
                    condition_info = self.colonoscopy_related_conditions[condition]
                    if condition_info['probability'] > max_prob:
                        max_prob = condition_info['probability']
                        selected_weights = condition_info['procedure_weights']
            
            if np.random.random() < max_prob:
                if selected_weights:
                    procedure_code = np.random.choice(
                        list(selected_weights.keys()),
                        p=list(selected_weights.values())
                    )
                else:
                    procedure_code = np.random.choice(
                        ['45378', '45380', '45384', '45385'],
                        p=[0.7, 0.2, 0.05, 0.05]
                    )
                
                provider_id = f'PR{str(np.random.randint(1, 17)).zfill(3)}'
                start_date = datetime.strptime(encounter['start_date'], '%Y-%m-%d')
                procedure_start = datetime.combine(
                    start_date,
                    datetime.strptime(f"{np.random.randint(9,17)}:00", "%H:%M").time()
                )
                
                duration_hours = 1
                if procedure_code in ['45384', '45385']:
                    duration_hours = np.random.randint(2, 4)
                
                procedure_end = procedure_start + timedelta(hours=duration_hours)
                
                procedures.append({
                    'patient_id': encounter['patient_id'],
                    'encounter_id': encounter['encounter_id'],
                    'provider_id': provider_id,
                    'procedure_code': procedure_code,
                    'start_datetime': procedure_start.strftime('%Y-%m-%d %H:%M:%S'),
                    'end_datetime': procedure_end.strftime('%Y-%m-%d %H:%M:%S')
                })
        
        return pd.DataFrame(procedures)

    def _get_condition_from_code(self, diagnosis_code: str) -> Optional[str]:
        for condition, code_dict in CHARLSON_CONDITIONS.items():
            # check ICD-9 code
            if diagnosis_code in code_dict['9']:
                return condition
            # check ICD-10 code
            if diagnosis_code in code_dict['10']:
                return condition
        return None

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
    for i, location in enumerate(patient_locations):
        patient_id = f'P{str(i+1).zfill(3)}'
        census_block = f'CB{str(i).zfill(3)}'
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

def generate_mock_data(conn, n_patients=100, reference_date=datetime(2024, 1, 1)):
    cursor = conn.cursor()
    
    # access RUCC codes
    rucc_codes = [row[0] for row in cursor.execute('SELECT rucc_code FROM rucc ORDER BY census_block')]
    
    # generate demographics data
    demographics_generator = DemographicsGenerator(SEED, reference_date)
    demographics_df = demographics_generator.generate_demographics(n_patients, rucc_codes)
    
    # generate medical data
    medical_generator = MedicalDataGenerator(SEED, reference_date)
    
    # set time range
    start_date = reference_date - timedelta(days=730)
    end_date = reference_date
    
    # generate encounters data, pass conn parameter
    encounters_df = medical_generator.generate_encounters(
        demographics_df, 
        start_date, 
        end_date,
        conn
    )
    
    # insert demographics data into database
    cursor.executemany('''
    INSERT INTO demographics 
    (patient_id, birth_date, sex, race, ethnicity, education, income)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', demographics_df.values.tolist())
    
    # insert encounters data into database
    try:
        cursor.executemany('''
        INSERT INTO encounter 
        (patient_id, encounter_id, start_date, end_date)
        VALUES (?, ?, ?, ?)
        ''', encounters_df[['patient_id', 'encounter_id', 'start_date', 'end_date']].values.tolist())
        conn.commit()
    except sqlite3.IntegrityError as e:
        print("error:")
        print(f"error message: {str(e)}")
        print("\nfirst 5 rows of encounter data:")
        print(encounters_df.head())
        raise
    
    # generate diagnoses data
    diagnoses_df = medical_generator.generate_diagnoses(
        encounters_df, 
        demographics_df
    )
    
    # insert diagnoses data into database
    cursor.executemany('''
    INSERT INTO diagnosis 
    (patient_id, encounter_id, diagnosis_code, vocabulary_id, diagnosis_date)
    VALUES (?, ?, ?, ?, ?)
    ''', diagnoses_df.values.tolist())
    
    # generate medical procedures
    procedures_df = medical_generator.generate_procedures(
        encounters_df, 
        demographics_df,
        diagnoses_df
    )
    
    # insert medical procedures into database
    cursor.executemany('''
    INSERT INTO procedure_table 
    (patient_id, encounter_id, provider_id, procedure_code, start_datetime, end_datetime)
    VALUES (?, ?, ?, ?, ?, ?)
    ''', procedures_df.values.tolist())
    
    conn.commit()

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

def export_tables_to_custom_dir(conn, output_dir):
    """
    将数据库表导出到指定目录的CSV文件
    
    参数:
    conn: sqlite3.Connection, 数据库连接
    output_dir: str, 输出目录
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
    生成单个数据集
    
    参数:
    time_effects: dict, 旅行时间效应配置
    output_dir: str, 输出目录
    seed: int, 随机种子
    """
    print(f"\n开始生成数据集，输出目录: {output_dir}")
    
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 创建数据库连接
    conn = create_database()
    
    try:
        # 加载位置数据
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
        
        # 加载基础数据
        load_existing_data(conn, patient_locations, clinic_locations, travel_times)
        
        # 创建数据生成器实例
        medical_generator = MedicalDataGenerator(seed=seed)
        medical_generator.base_params['encounter']['time_effects'] = time_effects
        
        # 生成数据
        generate_mock_data(conn)
        
        # 导出数据到指定目录
        export_tables_to_custom_dir(conn, output_dir)
        
        # 复制数据库文件到输出目录
        shutil.copy('healthcare.db', os.path.join(output_dir, 'healthcare.db'))
        
        print(f"数据集生成完成: {output_dir}")
        
    except Exception as e:
        print(f"生成数据集时出错: {str(e)}")
        raise
    finally:
        conn.close()

def main():
    """主函数：生成四个不同旅行时间影响的数据集"""
    
    # 定义四种不同的时间效应
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
    
    # 创建主输出目录
    base_output_dir = 'output_data/multiple_datasets'
    if not os.path.exists(base_output_dir):
        os.makedirs(base_output_dir)
    
    # 为每个配置生成数据集
    for effect_type, time_effects in time_effects_configs.items():
        output_dir = os.path.join(base_output_dir, f'{effect_type}_travel_effect')
        print(f"\n开始生成 {effect_type} 旅行时间影响的数据集...")
        generate_dataset(time_effects, output_dir)
        
        # 打印数据集信息
        print(f"\n{effect_type} 数据集信息:")
        print(f"输出目录: {output_dir}")
        print(f"旅行时间效应配置: {time_effects}")

if __name__ == "__main__":
    main() 