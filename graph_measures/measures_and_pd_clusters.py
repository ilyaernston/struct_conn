import pandas as pd
from sklearn import preprocessing

labels_df = pd.read_csv('/Users/elijah/Desktop/thesis/tests_2/measures&pd&metadata.csv')

labels_df = labels_df.rename(columns={'PD_cluster(k=8;bic=4228.460)':'PD_cluster_k8'})
labels_df = labels_df.rename(columns={'PD_cluster(k=6;bic=4375.533)':'PD_cluster_k6'})
labels_df = labels_df.rename(columns={'PD_cluster(k=7;bic=4245.672)':'PD_cluster_k7'})

norm_cols = ['sw_sigma', 'sw_omega', 'avg_rich_club', 'global_efficiency', 
               'avg_clustering', 'modularity', 'avg_path_length', 'avg_degree', 'density']
for col in norm_cols:
    labels_df[f'{col}']=(labels_df[f'{col}']-labels_df[f'{col}'].min())/(labels_df[f'{col}'].max()-labels_df[f'{col}'].min())

### MANOVA ###

from statsmodels.multivariate.manova import MANOVA


scale_columns = ['SevereHealthConditions', 'MajorHealthConditions', 'MinorHealthCondition', 'BrainHealth']
topo_scales = ['PD_cluster_k7', 'sw_sigma', 'sw_omega', 'avg_rich_club', 'global_efficiency', 
               'avg_clustering', 'modularity', 'avg_path_length', 'avg_degree', 'density']

# Create the MANOVA formula
formula = ' + '.join(topo_scales) + ' ~ MajorHealthConditions'

# Run MANOVA
maov = MANOVA.from_formula(formula, data=labels_df)
results = maov.mv_test()
print(results)

### LINEAR REGRESSION ###

import statsmodels.api as sm
import statsmodels.formula.api as smf


labels_df = pd.read_csv('/Users/elijah/Desktop/thesis/tests_2/all_labels.csv')
#labels_df = output_df
labels_df = labels_df.rename(columns={'PD_cluster(k=8;bic=4228.460)':'PD_cluster_k8'})
labels_df['PD_cluster_k8'] = labels_df['PD_cluster_k8'].astype('category')

indep_vars_cont = [
        'Age', 'Number of Children',
        'SevereHealthConditions', 'MajorHealthConditions', 'MinorHealthCondition', 
        'BrainHealth',
        'PSQI', 'OASIS', 'PCL-5', 'PHQ9', 'GAD7',
        'B5 Extraversion', 'B5 Agreeableness', 'B5 Coscientioness', 'B5 EmotionalStability', 'B5 Openness', 
        'SubjectiveHappiness', 
        'EpigeneticScore', 'AdjustedEpigeneticScore',
        'Number of Languages'
        ]
indep_vars_nomin = [
       'Gender', 'DominantHand', 'Native Language', 'EthnicalIdentity', 
       'BloodSuger', 'BloodPressure', 'Thyroids', 'Lipids', 
       'Depression', 'Anxiety', 'AttentionDisorders', 'CommunicationDisorders',
       'VisualAid', 'HearingAid', 
       'LongCovid',
       'Education', 'Native Language Family'
       ]
for var in indep_vars_nomin: labels_df[f'{var}'] = labels_df[f'{var}'].astype('category')

indep_vars_all = indep_vars_cont + indep_vars_nomin
indep_vars_health = ['SevereHealthConditions', 'MajorHealthConditions', 'MinorHealthCondition', 
        'BrainHealth',
        'BloodSuger', 'BloodPressure', 'Thyroids', 'Lipids', 
        'Depression', 'Anxiety', 'AttentionDisorders', 'CommunicationDisorders',
        'VisualAid', 'HearingAid', 
        'LongCovid',
        ]
indep_vars_B5 = ['B5 Extraversion', 'B5 Agreeableness', 
                 'B5 Coscientioness', 'B5 EmotionalStability', 'B5 Openness']
indep_vars_cogn = [
        'Number of Languages', 'Education', 'Native Language Family', 'Native Language'
        ]

formula = 'PD_cluster_k8 ~ ' + ' + '.join(indep_vars_health)

model = smf.mnlogit(formula, data=labels_df)
result = model.fit()
print(result.summary())