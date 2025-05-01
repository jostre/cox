#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import sqlite3
import numpy as np

def check_missing_values():
    """检查数据中的缺失值"""
    print("\n=== 开始检查缺失值 ===")
    
    # 1. 读取合并后的数据
    print("\n1. 读取合并数据...")
    try:
        data = pd.read_parquet('output_data/combined_data.parquet')
        print(f"成功读取数据，共 {len(data)} 行")
    except Exception as e:
        print(f"读取数据失败: {e}")
        return
    
    # 2. 检查总体缺失情况
    print("\n2. 检查总体缺失情况...")
    missing = data.isnull().sum()
    missing_percent = (data.isnull().sum() / len(data)) * 100
    
    missing_df = pd.DataFrame({
        '缺失值数量': missing,
        '缺失比例(%)': missing_percent
    })
    
    # 只显示有缺失的列
    missing_df = missing_df[missing_df['缺失值数量'] > 0]
    if len(missing_df) > 0:
        print("\n存在缺失值的列：")
        print(missing_df)
    else:
        print("\n没有发现缺失值！")
    
    # 3. 按变量类型分组检查
    print("\n3. 按变量类型分组检查...")
    
    # 关键变量
    print("\n关键变量：")
    key_vars = ['event', 'T']
    print(data[key_vars].isnull().sum())
    
    # 人口统计学变量
    print("\n人口统计学变量：")
    demo_vars = ['sex', 'race', 'ethnicity', 'education', 'income']
    print(data[demo_vars].isnull().sum())
    
    # 旅行时间相关
    print("\n旅行时间相关：")
    travel_vars = ['min_travel_time']
    print(data[travel_vars].isnull().sum())
    
    # RUCC代码
    print("\nRUCC代码：")
    print(data[['SDOH']].isnull().sum())
    
    # Charlson疾病
    print("\nCharlson疾病变量：")
    charlson_vars = [col for col in data.columns if col in [
        'mi', 'chf', 'pvd', 'cevd', 'dementia', 'copd', 'rheumd', 'pud', 
        'mld', 'msld', 'diab', 'dia_w_c', 'hp', 'mrend', 'srend', 
        'aids', 'hiv', 'mst', 'mal', 'Obesity', 'WL', 'Alcohol', 'Drug', 'Psycho', 'Dep'
    ]]
    print(data[charlson_vars].isnull().sum())
    
    # 4. 检查数据库中的原始数据
    print("\n4. 检查数据库中的原始数据...")
    try:
        conn = sqlite3.connect('healthcare.db')
        
        # 检查travel_time表
        print("\n检查travel_time表：")
        travel_time_check = pd.read_sql_query("""
            SELECT COUNT(*) as total_rows,
                   COUNT(travel_time_minutes) as non_null_rows
            FROM travel_time
        """, conn)
        print(travel_time_check)
        
        # 检查RUCC表
        print("\n检查RUCC表：")
        rucc_check = pd.read_sql_query("""
            SELECT COUNT(*) as total_rows,
                   COUNT(rucc_code) as non_null_rows
            FROM rucc
        """, conn)
        print(rucc_check)
        
        # 检查诊断表
        print("\n检查诊断表：")
        diagnosis_check = pd.read_sql_query("""
            SELECT COUNT(*) as total_rows,
                   COUNT(diagnosis_code) as non_null_rows
            FROM diagnosis
        """, conn)
        print(diagnosis_check)
        
        conn.close()
    except Exception as e:
        print(f"检查数据库时出错: {e}")
    
    print("\n=== 检查完成 ===")

if __name__ == "__main__":
    check_missing_values()