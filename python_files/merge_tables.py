#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import sqlite3
import numpy as np
from datetime import datetime, timedelta
import json

from comorbidity_info import CHARLSON_CONDITIONS



# In[10]:


def merge_healthcare_data(db_path='healthcare.db'):
    conn = sqlite3.connect(db_path)
    
    charlson_conditions = []
    for condition, codes in CHARLSON_CONDITIONS.items():
        icd9_codes = "','".join(codes['9'])
        icd10_codes = "','".join(codes['10'])
        
        condition_sql = f"""
        MAX(CASE 
            WHEN (vocabulary_id = 'ICD9CM' AND diagnosis_code IN ('{icd9_codes}'))
            OR (vocabulary_id = 'ICD10CM' AND diagnosis_code IN ('{icd10_codes}'))
            THEN 1 ELSE 0 
        END) as {condition}"""
        charlson_conditions.append(condition_sql)
    
    charlson_sql = ",\n        ".join(charlson_conditions)
    
    query = f"""
    WITH screening_events AS (
        SELECT 
            p.patient_id,
            MIN(p.start_datetime) as first_screening_date,
            CASE WHEN COUNT(*) > 0 THEN 1 ELSE 0 END as event
        FROM procedure_table p
        WHERE p.procedure_code IN ('45378', '45380', '45384', '45385')
        GROUP BY p.patient_id
    ),
    last_visits AS (
        SELECT 
            e.patient_id,
            MAX(e.end_date) as last_visit_date
        FROM encounter e
        WHERE NOT EXISTS (
            SELECT 1 
            FROM procedure_table p 
            WHERE p.patient_id = e.patient_id 
            AND p.procedure_code IN ('45378', '45380', '45384', '45385')
        )
        GROUP BY e.patient_id
    )

    SELECT 
        d.patient_id,
        COALESCE(se.event, 0) as event,
        CASE 
            WHEN se.event = 1 THEN 
                ROUND((JULIANDAY(se.first_screening_date) - 
                      JULIANDAY(d.birth_date))/365.25 - 45, 2)
            ELSE 
                ROUND((JULIANDAY(lv.last_visit_date) - 
                      JULIANDAY(d.birth_date))/365.25 - 45, 2)
        END as T,
        d.sex,
        d.race,
        d.ethnicity,
        MIN(tt.travel_time_minutes) as min_travel_time,
        r.rucc_code as SDOH,
        d.education,
        d.income,
        {charlson_sql}
    FROM demographics d
    LEFT JOIN screening_events se ON d.patient_id = se.patient_id
    LEFT JOIN last_visits lv ON d.patient_id = lv.patient_id
    LEFT JOIN geolocation g ON d.patient_id = g.patient_id
    LEFT JOIN travel_time tt ON g.census_block = tt.census_block
    LEFT JOIN rucc r ON g.census_block = r.census_block
    LEFT JOIN diagnosis diag ON d.patient_id = diag.patient_id
    GROUP BY d.patient_id;
    """
    
    combined_df = pd.read_sql_query(query, conn)
    
    # Save both formats
    combined_df.to_csv('output_data/combined_data.csv', index=False)
    combined_df.to_parquet('output_data/combined_data.parquet')
    
    print("\nCombined Data Summary:")
    print(f"Total patients: {len(combined_df)}")
    print("\nScreening events distribution:")
    print(combined_df['event'].value_counts())
    
    # Charlson
    charlson_cols = CHARLSON_CONDITIONS.keys()
    print("\nCharlson Comorbidities Summary:")
    for col in charlson_cols:
        print(f"{col}: {combined_df[col].sum()} patients")
    
    print("\nTravel time statistics:")
    print(combined_df['min_travel_time'].describe())
    
    conn.close()
    return combined_df


# In[11]:


if __name__ == "__main__":
   combined_data = merge_healthcare_data()
   print("\nSample of combined data:")
   print(combined_data.head())


# In[ ]:




