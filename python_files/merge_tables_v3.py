#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import sqlite3
import numpy as np
from datetime import datetime, timedelta
import json
from comorbidity_info import CHARLSON_CONDITIONS
from typing import List, Tuple
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor



# In[10]:


def perform_multiple_imputation(
    df: pd.DataFrame,
    n_imputations: int = 5,
    predictors: List[str] = None,
    categorical_cols: List[str] = None,
    comorbidity_cols: List[str] = None
) -> Tuple[pd.DataFrame, List[pd.DataFrame]]:
    """
    modified function: set all comorbidity variables to 0 for patients without healthcare encounter
    """
    if predictors is None:
        predictors = ['sex', 'race', 'ethnicity', 'education', 'income', 
                     'min_travel_time', 'SDOH']
    
    if categorical_cols is None:
        categorical_cols = ['sex', 'race', 'ethnicity', 'education']
    
    print(f"\nstart data processing:")
    print(f"predictors: {predictors}")
    print(f"categorical variables: {categorical_cols}")
    print(f"comorbidity variables: {comorbidity_cols}")
    
    # convert categorical variables to numerical
    df_encoded = df.copy()
    for col in categorical_cols:
        df_encoded[col] = pd.Categorical(df_encoded[col]).codes
    
    # identify patients without healthcare encounter
    mask = df_encoded['T'] == 0
    
    if mask.sum() > 0:
        print(f"\nfind {mask.sum()} patients without healthcare encounter")
        
        # set all comorbidity variables to 0 for patients without healthcare encounter
        for col in comorbidity_cols:
            df_encoded.loc[mask, col] = 0
        
        # keep event=0
        df_encoded.loc[mask, 'event'] = 0
        
        # set observation time
        df_encoded.loc[mask, 'T'] = (
            datetime(2024,1,1) - pd.to_datetime(df_encoded.loc[mask, 'birth_date'])
        ).dt.days/365.25 - 45
        
        # output the results
        print("\ncomorbidity distribution:")
        print(df_encoded.loc[mask, comorbidity_cols].mean())
        
        print("\ncomorbidity distribution:")
        print(df_encoded.loc[~mask, comorbidity_cols].mean())
    else:
        print("no patients without healthcare encounter")
    
    # since we don't need multiple imputation, return the processed data set directly
    return df_encoded, [df_encoded]  # return the same data set as the only imputation result

def save_imputed_datasets(
    original_df: pd.DataFrame,
    processed_df: pd.DataFrame,
    processed_datasets: List[pd.DataFrame],
    output_dir: str = 'output_data'
) -> None:
    """
    save the processed data set
    """
    # save the original data
    original_df.to_csv(f'{output_dir}/combined_data_original.csv', index=False)
    original_df.to_parquet(f'{output_dir}/combined_data_original.parquet')
    
    # save the processed data set
    processed_df.to_csv(f'{output_dir}/combined_data_processed.csv', index=False)
    processed_df.to_parquet(f'{output_dir}/combined_data_processed.parquet')

def report_imputation_results(
    original_df: pd.DataFrame,
    processed_df: pd.DataFrame,
    comorbidity_cols: List[str]
) -> None:
    """
    report the data processing results
    """
    print("\ndata processing report:")
    for col in comorbidity_cols:
        orig_mean = original_df[col].mean()
        proc_mean = processed_df[col].mean()
        print(f"{col}:")
        print(f"  original data: {orig_mean:.3f}")
        print(f"  processed data: {proc_mean:.3f}")
        
    # calculate the ratio of patients without healthcare encounter
    no_encounter_mask = original_df['T'] == (
        datetime(2024,1,1) - pd.to_datetime(original_df['birth_date'])
    ).dt.days/365.25 - 45
    no_encounter_ratio = no_encounter_mask.mean() * 100
    print(f"\nno healthcare encounter patients ratio: {no_encounter_ratio:.1f}%")
    
    # report the comorbidity distribution
    print("\ncomorbidity distribution:")
    print("original data:")
    print(original_df[comorbidity_cols].mean())
    print("\nprocessed data:")
    print(processed_df[comorbidity_cols].mean())

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
        GROUP BY e.patient_id
    ),
    reference_date AS (
        SELECT '2024-01-01' as end_date
    )

    SELECT 
        d.patient_id,
        d.birth_date,
        COALESCE(se.event, 0) as event,
        CASE 
            WHEN se.event = 1 THEN 
                ROUND((JULIANDAY(se.first_screening_date) - 
                      JULIANDAY(d.birth_date))/365.25 - 45, 2)
            WHEN lv.last_visit_date IS NOT NULL THEN
                ROUND((JULIANDAY(lv.last_visit_date) - 
                      JULIANDAY(d.birth_date))/365.25 - 45, 2)
            ELSE 
                ROUND((JULIANDAY('2024-01-01') - 
                      JULIANDAY(d.birth_date))/365.25 - 45, 2)
        END as T,
        d.sex,
        d.race,
        d.ethnicity,
        MIN(tt.travel_time_minutes)/60.0 as min_travel_time,
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
    CROSS JOIN reference_date rd
    GROUP BY d.patient_id;
    """
    
    combined_df = pd.read_sql_query(query, conn)
    
    # check the number of patients without healthcare encounter
    no_encounter_mask = combined_df['T'] == (datetime(2024,1,1) - pd.to_datetime(combined_df['birth_date'])).dt.days/365.25 - 45
    print(f"\nnumber of patients without healthcare encounter: {no_encounter_mask.sum()}")
    print(f"total number of patients: {len(combined_df)}")
    
    # perform multiple imputation
    comorbidity_cols = list(CHARLSON_CONDITIONS.keys())
    processed_df, processed_datasets = perform_multiple_imputation(
        combined_df,
        comorbidity_cols=comorbidity_cols
    )
    
    # save the data
    save_imputed_datasets(combined_df, processed_df, processed_datasets)
    
    # report the results
    print("\nverification results:")
    print(f"total number of patients: {len(combined_df)}")
    print(f"number of patients with screening record: {combined_df['event'].sum()}")
    print(f"number of patients without healthcare encounter: {len(combined_df[combined_df['T'] == (datetime(2024,1,1) - pd.to_datetime(combined_df['birth_date'])).dt.days/365.25 - 45])}")
    
    
    # report the imputation results
    report_imputation_results(combined_df, processed_df, comorbidity_cols)
    
    conn.close()
    return combined_df, processed_datasets


# In[11]:


if __name__ == "__main__":
    original_data, processed_datasets = merge_healthcare_data()
    print("\nSample of original data:")
    print(original_data.head())
    print("\nSample of first processed dataset:")
    print(processed_datasets[0].head())





