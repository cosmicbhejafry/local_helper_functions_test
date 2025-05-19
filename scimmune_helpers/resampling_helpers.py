import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from matplotlib import pyplot as plt

def resample_group_many(df, group_key,group_to_sample, columns_to_balance, bins=10,len_sample=None,num_sample=10):
    
    temp_df = resample_group_one(df, group_key, group_to_sample, columns_to_balance, bins=bins,len_sample=len_sample,iter_num=0)
    
    for i in range(1,num_sample):
        temp_df = resample_group_one(temp_df, group_key, group_to_sample, columns_to_balance, bins=bins,len_sample=len_sample,iter_num=i)
    
    return temp_df

def resample_group_one(df, group_key, group_to_sample, columns_to_balance, bins=10,len_sample=None,iter_num=0):
    """
    Perform importance sampling on `group_to_sample` to match the distribution of other group based on multiple columns.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - group_key (str): Column name dividing the data into two groups.
    - columns_to_balance (list): List of column names to balance the groups over.
    - bins (int or list): Number of bins for histograms (default is 10, or specify a list for each dimension).

    Returns:
    - pd.DataFrame: Resampled group_1 DataFrame that matches the distribution of group_2.
    """
    # Separate the groups
    assert df[group_key].nunique()==2
    
    group_1 = df[df[group_key] == group_to_sample]
    group_2 = df[df[group_key] != group_to_sample]

    # Extract the columns to balance
    group_1_data = group_1[columns_to_balance].values
    group_2_data = group_2[columns_to_balance].values
    
    group_all_data = df[columns_to_balance].values
    
    # Compute multidimensional histograms
    hist_all, edges_all = np.histogramdd(group_all_data,bins=bins,density=True)
    
    hist_1, edges = np.histogramdd(group_1_data, bins=edges_all, density=True)
    hist_2, _ = np.histogramdd(group_2_data, bins=edges_all, density=True)
    
    # Add small epsilon to avoid division by zero
#     hist_all += 1e-20   
    hist_1 += 1e-20
    hist_2 += 1e-20

    # Compute importance weights for group_1 using inverse density of group_2
    group_1_weights = np.ones(len(group_1))
    
    bin_wts = np.nan_to_num(hist_2/hist_1)
#     print(bin_wts)
    
    # Helper function to compute bin indices for data points
    def compute_bin_indices(data, edges):
        indices = []
        for i, edge in enumerate(edges):
            indices.append(np.digitize(data[:, i], edge) - 1)
        return np.array(indices).T

    # Get bin indices for group_1 data
    group_1_indices = compute_bin_indices(group_1_data, edges_all)
#     print(group_1_indices)
    # Map indices to histogram values for group_2
    for i, indices in enumerate(group_1_indices):
#         print(indices)
#         print(i)
#         print(hist_2.shape)
#         print(tuple(indices.clip(0, hist_2.shape[i] - 1)))        
        bin_value = bin_wts[tuple(indices.clip(0, hist_2.shape[1] - 1))]
        
#         print(f'bin value {bin_value}')
        group_1_weights[i] *= bin_value

    # Normalize weights for resampling
    group_1_weights /= np.sum(group_1_weights)

    if len_sample ==None:
        len_sample1= len(group_2)
    else:
        len_sample1 = len_sample
    # Resample group_1 based on computed weights
    sampled_group_1 = group_1.sample(n=len_sample1, weights=group_1_weights, replace=False)

    df[f'sample_{iter_num}_{len_sample1}'] = df.index.map(lambda v : True if v in sampled_group_1.index else False)
    
    return df

## ADD IN HELPER LIBRARY
# https://towardsdatascience.com/hands-on-inverse-propensity-weighting-in-python-with-causallib-14505ebdc109
# https://kosukeimai.github.io/MatchIt/articles/assessing-balance.html

def find_smd_all_covs(input_df,group_key,covs_lst):
    '''function to find the standardized mean difference between groups 
    for a given list of covariates or features, treats groups pairwise
    
    output: dataframe with covariates on index, each column name corresponds to a pairwise difference, 
    value is corresponding smd
    
    reference- 
    https://statisticaloddsandends.wordpress.com/2021/10/31/standardized-mean-difference-smd-in-causal-inference/    
    '''
    
    mean_df = input_df.groupby(group_key)[covs_lst].mean().T
    std_df = input_df.groupby(group_key)[covs_lst].std().T

    count_df = input_df.groupby(group_key)[group_key].count()

    sorted_list = sorted(list(mean_df.columns))

    pairwise_tuples = []
    smd_results = {}

    for i in range(len(sorted_list)):
        for j in range(i + 1, len(sorted_list)):

            grp1 = sorted_list[i]
            grp2 = sorted_list[j]
            
            n1 = count_df[grp1]
            n2 = count_df[grp2]
            
            mean_diff = np.abs(mean_df[grp1] - mean_df[grp2])
            
            # check this!!! pooling in cohen's d is a bit different -- scale by group sizes basically
            
            pooled_std = np.sqrt(
                ((n1-1)*std_df[grp1]**2 + (n2-1)*std_df[grp2]**2) / (n1 + n2 - 2)
            )

            smd_results[f'{grp1}_vs_{grp2}'] = mean_diff / pooled_std
            
    return pd.concat(smd_results,axis=1).T


def find_pval_per_sample(df_with_sample1,downsample_lens_lst1,grp_key,grp_sample_to,col_lst,num_sample,test_type,apply_corr=True):

    dict_lens_pvals_s_ns = {}

    for len_tot in downsample_lens_lst1:

        pval_dict= {v : [] for v in col_lst}

        for i in range(num_sample):

            v = f'sample_{i}_{len_tot}'
            
            for kv in pval_dict.keys():        
                col_val = kv
                data1 = df_with_sample1[df_with_sample1[v]==True][col_val]
                data2 = df_with_sample1[df_with_sample1[grp_key]==grp_sample_to][col_val]

                if test_type == 'KS':
                    statval = stats.ks_2samp(data1, data2, alternative='two-sided')

                else:
                    statval = stats.mannwhitneyu(data1,data2,alternative='two-sided')    

                pv = statval[1]                
                if apply_corr==True:
                    pv = np.clip(0,1,pv*num_sample)

                pval_dict[kv].append(pv)

        dict_lens_pvals_s_ns[len_tot] = pval_dict
        
    return dict_lens_pvals_s_ns

def summarize_pval_per_sample(dict_lens_pvals_s_ns):

    s_ns_summary_pval_stats = {}

    # TODO: NEEDS FIXING

    for len_tot in dict_lens_pvals_s_ns.keys():
        pval_stats_dict = {}
        pval_stats_dict['mean'] = np.mean(list(dict_lens_pvals_s_ns[len_tot].values()))
        pval_stats_dict['min'] = np.min(list(dict_lens_pvals_s_ns[len_tot].values()))
        pval_stats_dict['max'] = np.max(list(dict_lens_pvals_s_ns[len_tot].values()))

        for quant_val in [0.25,0.5,0.75]:
            pval_stats_dict[f'{quant_val}_quantile'] = np.quantile(list(dict_lens_pvals_s_ns[len_tot].values()),quant_val)

        s_ns_summary_pval_stats[len_tot] =  pval_stats_dict
    
    return s_ns_summary_pval_stats

def plot_diff_hist(df_with_sample,edges_all,grp_key,num_feat,col_lst,num_iter,len_sample,grp_sample,grp_sample_to):

    # Set the Seaborn theme
    sns.set_theme(style="whitegrid")

    # Create the plot
    plt.figure(figsize=(10, 6))
    
    balance_feature=col_lst[num_feat]
    sample_key = f'sample_{num_iter}_{len_sample}'
    
    sns.histplot(df_with_sample[df_with_sample[sample_key] == True][balance_feature], 
                 bins=edges_all[num_feat], kde=False, color='blue', label='Sampled', alpha=0.25,stat='density',element='step')

    sns.histplot(df_with_sample[df_with_sample[grp_key] == grp_sample][balance_feature], 
                 bins=edges_all[num_feat], kde=False, color='orange', label=f'Group {grp_sample}', alpha=0.25,stat='density',element='step')

    sns.histplot(df_with_sample[df_with_sample[grp_key] == grp_sample_to][balance_feature], 
                 bins=edges_all[num_feat], kde=False, color='green', label=f'Group {grp_sample_to}', alpha=0.25,stat='density',element='step')

    # Add labels and title
    plt.xlabel(balance_feature)
    plt.ylabel('frequency')
    plt.title(f'{balance_feature} distribution by group and sample, {len_sample} samples')
    plt.legend()  # Add a legend to distinguish the groups

    # Show the plot
    plt.show()

def smd_across_samples_df(df_with_sample_ns_none,downsample_lens_lst,num_sample,group_key,grp_sample_to,col_lst):
    smds_across_all = []

    for len_sample in downsample_lens_lst:

        lst_smds = []

        for num_iter in range(num_sample):

            sample_key = f'sample_{num_iter}_{len_sample}'

            selected_bc = list(df_with_sample_ns_none[df_with_sample_ns_none[group_key]==grp_sample_to].index)

            selected_bc += list(df_with_sample_ns_none[df_with_sample_ns_none[sample_key]==True].index)

            temp_sample_df = df_with_sample_ns_none[df_with_sample_ns_none.index.isin(selected_bc)]

            temp_find_smd = find_smd_all_covs(temp_sample_df,group_key,col_lst).dropna()

            temp_find_smd.index = [f'{v}_{sample_key}' for v in temp_find_smd.index]

            lst_smds.append(temp_find_smd)

        tmp_smds = pd.concat(lst_smds)

        summary_smd = pd.concat([tmp_smds.quantile(0.25), tmp_smds.quantile(0.5), tmp_smds.quantile(0.75)],axis=1).T

        summary_smd.index = [f'{len_sample}_{v}' for v in summary_smd.index]

        smds_across_all.append(summary_smd)

    return pd.concat(smds_across_all)