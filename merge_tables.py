# 检查无就医记录样本的travel time
if no_encounter_mask.sum() > 0:
    print("\n无就医记录样本的travel time分布:")
    print(combined_df[no_encounter_mask]['min_travel_time'].describe())

# 使用简单插补替代多重插补
comorbidity_cols = list(CHARLSON_CONDITIONS.keys())
imputed_df = perform_simple_imputation(
    df=combined_df,
    comorbidity_cols=comorbidity_cols
)

# 保存数据
# 由于简单插补只生成一个数据集，我们修改保存方式
imputed_df.to_csv('output_data/combined_data_imputed.csv', index=False)
imputed_df.to_parquet('output_data/combined_data_imputed.parquet')

# 报告结果
print("\n数据验证:")
print(f"总病人数: {len(combined_df)}")
print(f"有筛查记录的病人数: {combined_df['event'].sum()}")
print(f"没有就医记录的病人数: {len(combined_df[combined_df['T'] == (datetime(2024,1,1) - pd.to_datetime(combined_df['birth_date'])).dt.days/365.25 - 45])}")

print("\nTravel time分布:")
print(combined_df['min_travel_time'].describe())

print("\nEvent和T的关系:")
print(combined_df.groupby('event')['T'].describe())

print("\nTravel time和event的关系:")
print(combined_df.groupby('event')['min_travel_time'].describe())

conn.close()
return combined_df, imputed_df  # 返回原始数据和插补后的数据 