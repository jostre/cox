#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import sqlite3
import numpy as np

def check_missing_reason():
    """检查插补后的数据完整性"""
    print("\n=== 开始检查数据 ===")
    
    # 1. 读取原始数据和插补后的平均数据
    original_df = pd.read_csv('output_data/combined_data_original.csv')
    imputed_df = pd.read_csv('output_data/combined_data_imputed_mean.csv')
    
    print(f"\n总病人数: {len(original_df)}")
    
    # 2. 检查原始数据中的无就医记录病人
    no_encounter_original = original_df[(original_df['event'] == 0) & (original_df['T'] > 0)]
    print(f"\n原始数据中无就医记录病人数: {len(no_encounter_original)}")
    
    # 3. 检查插补后数据中的无就医记录病人
    no_encounter_imputed = imputed_df[(imputed_df['event'] == 0) & (imputed_df['T'] > 0)]
    print(f"插补后数据中无就医记录病人数: {len(no_encounter_imputed)}")
    
    # 4. 检查共病数据
    comorbidity_cols = [col for col in original_df.columns if col not in 
                       ['patient_id', 'birth_date', 'event', 'T', 'sex', 'race', 
                        'ethnicity', 'min_travel_time', 'SDOH', 'education', 'income']]
    
    print("\n原始数据中无就医记录病人的共病分布：")
    print(no_encounter_original[comorbidity_cols].mean())
    
    print("\n插补后数据中无就医记录病人的共病分布：")
    print(no_encounter_imputed[comorbidity_cols].mean())
    
    # 5. 检查缺失值
    print("\n原始数据中的缺失值：")
    print(original_df[comorbidity_cols].isnull().sum())
    
    print("\n插补后数据中的缺失值：")
    print(imputed_df[comorbidity_cols].isnull().sum())
    
    # 6. 检查T值的分布
    print("\n原始数据中T值的分布：")
    print(original_df['T'].describe())
    
    print("\n插补后数据中T值的分布：")
    print(imputed_df['T'].describe())
    
    print("\n=== 检查完成 ===")

if __name__ == "__main__":
    check_missing_reason()