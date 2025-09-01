
import pandas as pd
import numpy as np
from scipy.stats import hypergeom
from statsmodels.stats.multitest import multipletests
import os

def hypergeometric_test(
        cluster_labels1: pd.DataFrame | np.ndarray | list,
        cluster_labels2: pd.DataFrame | np.ndarray | list,
        alpha: float = 0.01, 
        correction: str = 'bonferroni') -> pd.DataFrame:
    """
    Test enrichment of GMM clusters in network components using the hypergeometric test.
    
    Parameters:
    - cluster_labels1: List, array, or pd.Series of cluster assignments (e.g., 'white', 'gray', 'mixed')
    - cluster_labels2: List, array, or pd.Series of cluster2 assignments (e.g., 0, 1, 2)
    - alpha: Significance threshold (default: 0.01)
    - correction: Multiple testing correction method (default: 'bonferroni')
    
    Returns:
    - results: pd.DataFrame with columns ['component', 'cluster', 'overlap', 'pval', 'significant']
    """
    
    # Convert inputs to pandas Series
    if not isinstance(cluster_labels1, pd.Series):
        cluster_labels1 = pd.Series(cluster_labels1)
    if not isinstance(cluster_labels2, pd.Series):
        cluster_labels2 = pd.Series(cluster_labels2)
    
    # Get unique clusters and components
    clusters = cluster_labels1.unique()
    components = cluster_labels2.unique()
    
    # Total molecules
    N = len(cluster_labels1)
    
    results = []
    for component in components:
        component_mask = (cluster_labels2 == component)
        n = component_mask.sum()  # Size of network component
        
        for cluster in clusters:
            cluster_mask = (cluster_labels1 == cluster)
            D = cluster_mask.sum()  # Size of GMM cluster
            k = (component_mask & cluster_mask).sum()  # Overlap
            
            # Hypergeometric test (right-tailed: P(X >= k))
            pval = hypergeom.sf(k-1, N, D, n)
            
            results.append({
                'clustering_1': cluster,
                'clustering_2': component,
                'overlap': k,
                'pval': pval
            })
    
    results_df = pd.DataFrame(results)
    
    # Apply Bonferroni correction
    n_tests = len(results_df)
    if correction == 'bonferroni':
        results_df['significant'] = results_df['pval'] < (alpha / n_tests)
    else:
        results_df['significant'] = results_df['pval'] < alpha
    
    return results_df

mds_labels = pd.read_csv('/Users/elijah/Desktop/thesis/tests_2/PDs_labels_mds.csv', header=0)
tsne_labels =  pd.read_csv('/Users/elijah/Desktop/thesis/tests_2/PDs_labels_tsne.csv', header=0)
for df in (mds_labels, tsne_labels):
    df['subject_id'] = df['subject_id'].str.replace(r'^sub-', '', regex=True)
metadata =  pd.read_csv('/Users/elijah/Desktop/thesis/AllSubjectsMeta - Sheet1.csv', header=0)

metadata = metadata.rename(columns={'Subject Code':'subject_id'})

output_df = mds_labels.merge(tsne_labels, how = 'inner', on = 'subject_id')
output_df = output_df.merge(metadata, how = 'inner', on = 'subject_id')

def label_language_family(df):
    semitic = ['Hebrew', 'Arabic', 'Amahric']
    indo_european = [
    'Russian', 'English', 'French', 'Portugese', 'Polish',
    'Bulgarian', 'Hungarian', 'Romanian', 'Spanish'
    ]

    # build a lookup
    fam_map = {lang: 'Semitic' for lang in semitic}
    fam_map.update({lang: 'IndoEuropean' for lang in indo_european})

    # map & add new column; unrecognized 'Other'
    df['Native Language Family'] = df['Native Language'].map(fam_map)
    df['Native Language Family'] = df['Native Language Family'].fillna('Other')
    return df

def count_languages(df):

    df['No Languages'] = (
        df['AdditionalLanguages']
        .fillna('')   # turn NaN → empty string
        .apply(lambda s: 
            len([lang for lang in s.split(',') if lang.strip()])  # count non‐empty pieces
        ) + 1 # add 1 for native lang
    )

    return df

output_df = label_language_family(output_df)
output_df = count_languages(output_df)

#output_df.to_csv('/Users/elijah/Desktop/thesis/tests_2/all_labels.csv')


output_dir = '/Users/elijah/Desktop/thesis/tests_2/hypergeom'
os.makedirs(output_dir, exist_ok=True)

# hypergeom test on MDS
for mds_label in list(mds_labels.columns)[1:]:
    for meta_label in list(metadata.columns)[1:]:
        meta_clusters_df = output_df[f'{meta_label}']
        mds_clusters_df = output_df[f'{mds_label}']
        hypergeom_res = hypergeometric_test(mds_clusters_df, meta_clusters_df)
        if hypergeom_res['significant'].any(): # save test results if there are any significant overlaps 
            out_csv = os.path.join(output_dir, f"mds_{mds_label}_vs_{meta_label}.csv")
            hypergeom_res.to_csv(out_csv, index=False)

# hypergeom test on TSNE
for tsne_label in list(tsne_labels.columns)[1:]:
    for meta_label in list(metadata.columns)[1:]:
        meta_clusters_df = output_df[f'{meta_label}']
        tsne_clusters_df = output_df[f'{tsne_label}']
        hypergeom_res = hypergeometric_test(tsne_clusters_df, meta_clusters_df)
        if hypergeom_res['significant'].any(): # save test results if there are any significant overlaps 
            out_csv = os.path.join(output_dir, f"tsne_{tsne_label}_vs_{meta_label}.csv")
            hypergeom_res.to_csv(out_csv, index=False)

# chi^2 nest on all clusterings vs all metadata columns
from scipy.stats import chi2_contingency
import pandas as pd

for mds_label in list(mds_labels.columns)[1:]:
    for meta_label in list(metadata.columns)[1:]:
        mds_clusters = output_df[mds_label]
        meta_clusters = output_df[meta_label]

        # build contingency table
        ct = pd.crosstab(mds_clusters, meta_clusters)
        
        # skip if empty or not at least 2×2
        if ct.empty or ct.shape[0] < 2 or ct.shape[1] < 2:
            #print(f"Skipping {mds_label} × {meta_label}: table too small ({ct.shape})")
            continue

        # now safe to compute chi2
        chi2, p, dof, expected = chi2_contingency(ct)
        if p < .05:
            print(f"MDS-based {mds_label}×{meta_label}: χ²={chi2:.2f}, p={p:.3f}")

for tsne_label in list(tsne_labels.columns)[1:]:
    for meta_label in list(metadata.columns)[1:]:
        tsne_clusters = output_df[tsne_label]
        meta_clusters = output_df[meta_label]

        # build contingency table
        ct = pd.crosstab(tsne_clusters, meta_clusters)
        
        # skip if empty or not at least 2×2
        if ct.empty or ct.shape[0] < 2 or ct.shape[1] < 2:
            #print(f"Skipping {tsne_clusters} × {meta_label}: table too small ({ct.shape})")
            continue

        # now safe to compute chi2
        chi2, p, dof, expected = chi2_contingency(ct)
        if p < .05:
            print(f"t-SNE-based {tsne_label}×{meta_label}: χ²={chi2:.2f}, p={p:.3f}")

### MANOVA on health-related scales ###

from statsmodels.multivariate.manova import MANOVA

labels_df = pd.read_csv('/Users/elijah/Desktop/thesis/tests_2/all_labels.csv')
#labels_df = output_df
labels_df = labels_df.rename(columns={'PD_cluster(k=8;bic=4228.460)':'PD_cluster_k8'})
labels_df = labels_df.rename(columns={'PD_cluster(k=6;bic=4375.533)':'PD_cluster_k6'})
labels_df = labels_df.rename(columns={'PD_cluster(k=7;bic=4245.672)':'PD_cluster_k7'})

scale_columns = ['SevereHealthConditions', 'MajorHealthConditions', 'MinorHealthCondition', 'BrainHealth']

# Create the MANOVA formula
formula = ' + '.join(scale_columns) + ' ~ PD_cluster_k8'

# Run MANOVA
maov = MANOVA.from_formula(formula, data=labels_df)
results = maov.mv_test()
print(results)


### Multinominal Logistic Regression ###

import pandas as pd
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
