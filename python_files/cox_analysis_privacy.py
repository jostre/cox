#!/usr/bin/env python
# coding: utf-8

# In[79]:


import pandas as pd
from lifelines import CoxPHFitter
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


# In[57]:


combined_data = pd.read_parquet('output_data/combined_data_processed.parquet')
print("Columns with NaN values:")
print(combined_data.isnull().sum())
combined_data_clean = combined_data.dropna()


# In[70]:


def custom_print_summary(cph, all_covariates, excluded_covariates):
    """
    customized print summary function
    """
    original_summary = cph.summary

    new_summary = pd.DataFrame(index=all_covariates, 
                               columns=original_summary.columns, 
                               data=pd.NA)

    included_covariates = [cov for cov in all_covariates if cov not in excluded_covariates]
    for cov in included_covariates:
        if cov in original_summary.index:
            new_summary.loc[cov] = original_summary.loc[cov]

    print("=== Custom CoxPHFitter Summary ===")
    print(new_summary.to_string())
    print("\n=== Model Statistics ===")
    print(f"Concordance Index: {cph.concordance_index_:.3f}")
    print(f"Partial AIC: {cph.AIC_partial_:.1f}")
    print(f"Log-likelihood ratio test: {cph.log_likelihood_ratio_test().test_statistic:.2f}")


# In[71]:


def check_group_sizes(data, categorical_cols):
    """
    Check if any group has less than 20 samples
    Returns a dict of problematic groups and their counts
    """

    min_size = 20
    small_groups = {}
    
    # Check categorical variables
    for col in categorical_cols:
        counts = data[col].value_counts()
        small_cats = counts[counts < min_size]
        if len(small_cats) > 0:
            small_groups[col] = small_cats.to_dict()
            
    # Check binary disease indicators
    disease_cols = ['mi', 'chf', 'pvd', 'cevd', 'dementia', 'copd', 'rheumd', 'pud', 
    'mld', 'msld', 'diab', 'dia_w_c', 'hp', 'mrend', 'srend', 
    'aids', 'hiv', 'mst', 'mal', 'Obesity', 'WL', 'Alcohol', 'Drug', 'Psycho', 'Dep']
    for col in disease_cols:
        if col in data.columns:
            counts = data[col].value_counts()
            if 1 in counts and counts[1] < 20:
                small_groups[col] = {'positive_cases': counts[1]}
                
    return small_groups

def run_cox_analysis_with_privacy(data, enforce_privacy=True):
    """
    Run Cox analysis with optional privacy enforcement
    
    Parameters:
    data: DataFrame with the required columns
    enforce_privacy: If True, exclude groups with n<20
    
    Returns:
    model: Fitted CoxPHFitter model
    excluded_groups: Dict of groups excluded due to small sample size
    """
    # data preprocessing
    data = data.copy()
    
    # handle NaN values
    print("\nsize of data before handling NaN values:", len(data))
    data = data.dropna(subset=['T', 'event'])  # delete rows with missing survival time or event
    print("size of data after handling NaN values:", len(data))
    
    # dummy encoding for categorical variables
    categorical_cols = ['sex', 'race', 'ethnicity', 'education']
    data = pd.get_dummies(data, columns=categorical_cols, prefix=categorical_cols)
    
    # standardize travel time
    data['min_travel_time'] = (data['min_travel_time'] - data['min_travel_time'].mean()) / data['min_travel_time'].std()
    
    # add nonlinear term
    data['travel_time_squared'] = data['min_travel_time'] ** 2
    
    # Travel Time
    travel_covariates = ['min_travel_time', 'travel_time_squared']
    
    # Demographics
    demo_covariates = [col for col in data.columns if col.startswith(('sex', 'race', 'ethnicity', 'education'))]
    
    # Comorbidity
    charlson_covariates = [
    'mi', 'chf', 'pvd', 'cevd', 'dementia', 'copd', 'rheumd', 'pud', 
    'mld', 'msld', 'diab', 'dia_w_c', 'hp', 'mrend', 'srend', 
    'aids', 'hiv', 'mst', 'mal', 'Obesity', 'WL', 'Alcohol', 'Drug', 'Psycho', 'Dep'
    ]
    
    # SDOH
    data['SDOH'] = data['SDOH'].astype(float)
    sdoh_covariates = ['SDOH']
    
    all_covariates = travel_covariates + demo_covariates + charlson_covariates + sdoh_covariates
    
    if enforce_privacy:
        small_groups = check_group_sizes(data, demo_covariates)
        excluded_covariates = list(small_groups.keys())
        included_covariates = [cov for cov in all_covariates if cov not in excluded_covariates]
    else:
        small_groups = {}
        excluded_covariates = []
        included_covariates = all_covariates

    # Use all covariates plus outcome columns
    cox_data = data[included_covariates + ['T', 'event']].copy()
    
    # ensure no infinite values
    cox_data = cox_data.replace([np.inf, -np.inf], np.nan)
    cox_data = cox_data.dropna()
    
    print("\nsize of cox analysis data:", len(cox_data))
    print("used variables:", included_covariates)
    
    # Fit Cox model with decreased penalizer to get more significant effects
    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(cox_data, duration_col='T', event_col='event', robust=True)

    return cph, small_groups, excluded_covariates, all_covariates

def generate_analysis_report(data, model, excluded_groups=None):
    """Generate a comprehensive analysis report"""
    report = []
    
    # Basic statistics
    report.append("=== Basic Statistics ===")
    report.append(f"Total patients: {len(data)}")
    report.append(f"Event distribution:\n{data['event'].value_counts().to_string()}")
    report.append(f"\nTime variable statistics:\n{data['T'].describe().to_string()}")
    
    # Excluded groups (if privacy check was enforced)
    if excluded_groups:
        report.append("\n=== Excluded Groups (n<20) ===")
        for col, groups in excluded_groups.items():
            report.append(f"{col}: {groups}")
    
    # Model summary statistics
    report.append("\n=== Model Summary ===")
    report.append(f"Concordance: {model.concordance_index_:.3f}")
    report.append(f"Partial AIC: {model.AIC_partial_:.1f}")
    report.append(f"Log-likelihood ratio test: {model.log_likelihood_ratio_test().test_statistic:.2f}")
    
    # Significant covariates (p < 0.05)
    report.append("\n=== Significant Covariates (p < 0.05) ===")
    summary_df = model.print_summary().summary
    sig_covariates = summary_df[summary_df['p'] < 0.05]
    if len(sig_covariates) > 0:
        report.append(sig_covariates.to_string())
    else:
        report.append("No significant covariates found at p < 0.05 level")
    
    return "\n".join(report)



# In[72]:


# read data
data = pd.read_parquet('output_data/combined_data_processed.parquet')

# version 1：with privacy
print("\n=== version 1: with privacy check ===")
model_with_privacy, excluded_groups, excluded_covs, all_covs = run_cox_analysis_with_privacy(data, enforce_privacy=True)
custom_print_summary(model_with_privacy, all_covs, excluded_groups.keys())

print("\nCox Summary (privacy):")
model_with_privacy.print_summary()



# In[80]:


# version 2：no privacy 
print("\n=== version 2: without privacy check ===")
model_without_privacy, small_groups, excluded_covs, all_covs = run_cox_analysis_with_privacy(data, enforce_privacy=False)
print("\nCox Summary (no privacy):")

custom_print_summary(model_without_privacy, all_covs, [])

model_without_privacy.print_summary()


# In[81]:


def plot_forest_with_privacy(model_with_privacy, model_without_privacy, excluded_covariates=None, ax=None):
    """
    Create a forest plot displaying "NA" for variables excluded due to privacy checks.
    
    model_with_privacy: CoxPHFitter model with privacy checks applied.
	model_without_privacy: CoxPHFitter model without privacy checks.
	excluded_covariates: List of covariates excluded due to privacy reasons.
	ax: Matplotlib axis object (optional).
    """
    
    if ax is None:
        ax = plt.gca()
    
    if excluded_covariates is None:
        excluded_covariates = []
    
    all_covariates = model_without_privacy.summary.index.tolist()
    
    summary = model_with_privacy.summary
    plot_data = pd.DataFrame(index=all_covariates,
                           columns=['coef', 'exp(coef)', 'se(coef)', 
                                  'coef lower 95%', 'coef upper 95%'])
    
    for var in all_covariates:
        if var not in excluded_covariates and var in summary.index:
            plot_data.loc[var] = summary.loc[var]
    
    ypos = range(len(all_covariates))
    
    for i, var in enumerate(all_covariates):
        if not pd.isna(plot_data.loc[var, 'coef']):
            ax.plot([plot_data.loc[var, 'coef lower 95%'],
                    plot_data.loc[var, 'coef upper 95%']], 
                   [i, i], 'b-', alpha=0.7)
            ax.plot(plot_data.loc[var, 'coef'], i, 'bs')
        else:
            ax.text(0, i, 'NA', va='center', ha='center')
    
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_yticks(ypos)
    ax.set_yticklabels(all_covariates)
    ax.set_xlabel('Coefficient')
    ax.set_ylabel('Variable')
    ax.grid(True, alpha=0.3)
    
    return ax

model_with_privacy, small_groups, excluded_covariates, all_covariates = run_cox_analysis_with_privacy(data, enforce_privacy=True)
model_without_privacy, _, _, _ = run_cox_analysis_with_privacy(data, enforce_privacy=False)

plt.figure(figsize=(12, 6))

ax1 = plt.subplot(121)
plot_forest_with_privacy(model_with_privacy, model_without_privacy, excluded_covariates, ax=ax1)
plt.title('Forest Plot with Privacy Check')

ax2 = plt.subplot(122)
model_without_privacy.plot(ax=ax2)
plt.title('Forest Plot without Privacy Check')

plt.tight_layout()
plt.show()

# Print comparison results
print("\n=== Model Comparison ===")
print("With privacy check:")
print(f"Concordance Index: {model_with_privacy.concordance_index_:.3f}")
print(f"Log-likelihood ratio test: {model_with_privacy.log_likelihood_ratio_test().test_statistic:.2f}")
print("\nWithout privacy check:")
print(f"Concordance Index: {model_without_privacy.concordance_index_:.3f}")
print(f"Log-likelihood ratio test: {model_without_privacy.log_likelihood_ratio_test().test_statistic:.2f}")


# In[ ]:




