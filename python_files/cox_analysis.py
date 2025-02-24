#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
from lifelines import CoxPHFitter
import matplotlib.pyplot as plt
import seaborn as sns


# In[8]:


combined_data = pd.read_parquet('output_data/combined_data.parquet')
print("Columns with NaN values:")
print(combined_data.isnull().sum())
combined_data_clean = combined_data.dropna()


# In[12]:


def run_cox_analysis(data):

    # Travel Time
    travel_covariates = ['min_travel_time']
    
    # Demographics
    demo_covariates = [col for col in data.columns if col.startswith(('sex_', 'race_', 'ethnicity_'))]
    
    # Comorbidity
    charlson_covariates = [
    'mi', 'chf', 'pvd', 'cevd', 'dementia', 'copd', 'rheumd', 'pud', 
    'mld', 'msld', 'diab', 'dia_w_c', 'hp', 'mrend', 'srend', 
    'aids', 'hiv', 'mst', 'mal', 'Obesity', 'WL', 'Alcohol', 'Drug', 'Psycho', 'Dep'
    ]
    
    # SDOH
    sdoh_covariates = [col for col in data.columns 
                      if col.startswith('education_')] + ['SDOH']
    
    covariates = travel_covariates + demo_covariates + charlson_covariates + sdoh_covariates

    cph = CoxPHFitter(penalizer=0.1)

    cox_data = data[covariates + ['T', 'event']]

    cph.fit(cox_data, duration_col='T', event_col='event')
    return cph

print("Event distribution:")
print(combined_data_clean['event'].value_counts())
print("\nTime variable statistics:")
print(combined_data_clean['T'].describe())

model = run_cox_analysis(combined_data_clean)

print("Cox Model Summary:")
model.print_summary()


# In[ ]:




