import pandas as pd

def get_pos_cell_passing_dict(coverage_df,group_key,cell_key,is_list=False,depth_thr=10,start_pos=1,end_pos=16569):
    
    'expect a coverage df or a list of coverage dfs with keys group and cell'
    
    if is_list:    
        concat_dict = {}
        
        sp1 = start_pos
        ep1 = end_pos
        is_list1 = False
        
        for val in coverage_df:
            # modify sp1 and ep1 according to dataframe
            concat_dict.update(get_pos_cell_passing_dict(val,group_key,cell_key,is_list1,depth_thr,sp1,ep1))
            
        return concat_dict
    
    else:
        cov_cols = list(coverage_df.columns)
        
        assert group_key in cov_cols
        assert cell_key in cov_cols
        
        keep_cols = [group_key,cell_key]
        keep_cols.extend([str(v) for v in range(start_pos,end_pos+1)])
        
        tmp_cov = coverage_df[keep_cols].set_index([group_key,cell_key])
        
        tmp_cov_dict = tmp_cov[tmp_cov>=depth_thr].groupby(group_key).count().to_dict()
        
        return {f'{batch_ct}_{pos}' : val for pos, batch_val in tmp_cov_dict.items() for batch_ct, val in batch_val.items()}
        
def add_grouped_prevalence(variant_df,group_pos_dict,group_key,ct_key,donor_key='donor_id',pos_key='POS',cell_key='cell_id',depth_thr=10):
    
#     added_col_names = []

    variant_df[f'{group_key}_POS'] =  variant_df[group_key].astype(str)+'_'+variant_df[pos_key].astype(str)
    # added_col_names.append(f'{group_key}_POS')

    variant_df[f'cells_possible_{depth_thr}_{group_key}'] = variant_df[f'{group_key}_POS'].map(group_pos_dict)
    # added_col_names.append(f'cells_possible_{depth_thr}_{group_key}')

    variant_df[f'{group_key}_mut'] = variant_df[group_key].astype(str) + '_'+variant_df['MUT']
    # added_col_names.append(f'{group_key}_mut')

    num_cells_with_mut = variant_df.groupby(f'{group_key}_mut').count()[cell_key]
    num_cells_possible = variant_df.groupby(f'{group_key}_mut').mean(numeric_only=True)[f'cells_possible_{depth_thr}_{group_key}']
    
    cell_prop = dict(num_cells_with_mut/num_cells_possible)
    
    # num_cells_with_mut[num_cells_with_mut==1].index
    
    variant_df[f'prevalence_{group_key}']=variant_df[f'{group_key}_mut'].map(cell_prop)
    
#     added_col_names.append(f'prevalence_{group_key}')
                    
    # variant_df = variant_df.drop(['group_mut','group_POS'],axis=1)
    
    return variant_df

def add_col_from_dict(input_df,dict_val,new_col_name,mapping_key,is_dict_path=True,fill_na_key='NA'):

    if is_dict_path == False:
        df_dict = dict_val
    
    if is_dict_path == True:
        df_dict = pd.read_pickle(dict_val)

    input_df[new_col_name] = input_df[mapping_key].map(df_dict).fillna(fill_na_key)
    # return input_df, [new_col_name]
    return input_df

def add_col_from_df(input_df,df_val,new_col_name,mapping_key,col_to_map,is_df_path=True,fill_na_key='NA'):
    
    if is_df_path == False:
        df_tmp = df_val
    
    if is_df_path == True:
        df_tmp = pd.read_parquet(df_val)
    
    df_dict = df_tmp[col_to_map].to_dict()
    
    input_df[new_col_name] = input_df[mapping_key].map(df_dict).fillna(fill_na_key)
    
    # return input_df, [new_col_name]
    return input_df

def pcr_filtering_snv(variant_df,donor_key,mut_key,hf_key,pcr_hf_thresh=0.3,only_cryptic=False):

    # 

    # filter for SNVs first:     
    variants_snv = variant_df[variant_df.apply(lambda row: len(row['REF']) == 1 and len(row['ALT']) == 1, axis=1)]
    variants_snv['donor_MUT'] = variants_snv[donor_key] + '_' + variants_snv[mut_key]

    if only_cryptic == True:
        # PCR filtering - only filter cryptics below 30% 
        group_filter_key='mutant_type'
        group_filter_val='Cryptic'
        variants_snv = variants_snv[~((variants_snv[group_filter_key]==group_filter_val) & (variants_snv['HF']<pcr_hf_thresh))]
    
    else:
        # DEFAULT: filter by max
        # for each donor,MUT pair --> filter mut if the max heteroplasmy is less than 30%
        unique_muts = variants_snv.groupby([donor_key,mut_key,'donor_MUT']).max(numeric_only=True).reset_index()
        filtered_muts_pcr = unique_muts[unique_muts[hf_key]>=pcr_hf_thresh]
        variants_snv = variants_snv[variants_snv['donor_MUT'].isin(filtered_muts_pcr['donor_MUT'])]

    return variants_snv

    # PCR filtering - OUTDATED: using the median h0f 
    # -- Assumes cells with same heteroplasmy being sampled --> basically wrong!!
    # unique_muts = variants_snv.groupby([donor_key,mut_key,'donor_MUT']).median(numeric_only=True).reset_index()
    # filtered_muts_pcr = unique_muts[unique_muts[hf_key]>=pcr_hf_thresh]
    # variants_snv = variants_snv[variants_snv['donor_MUT'].isin(filtered_muts_pcr['donor_MUT'])]

def add_hhlp_col(variant_df,group_mut_key,prev_key,hf_key='HF',hf_thresh=0.3,prev_thresh=0.01,cell_bc_key='cell_id'):
    
    ## modify 

    hhlp_type = f'hhlp_{hf_thresh}_{prev_thresh}'
    hhlp_true_false = f'hhlp_{hf_thresh}_{prev_thresh}_bool'

    # map cells with LP mutation to the max heteroplasmy of LP mut in that cell
    cell_highest_hf_map = variant_df[variant_df[prev_key]<=prev_thresh].groupby(cell_bc_key)[hf_key].apply(max)
    cell_highest_hf_map = cell_highest_hf_map.reset_index()

    # cell has hhlp if max heteroplasmy is more than 30%
    cell_highest_hf_map['LP_type'] = cell_highest_hf_map[hf_key].map(lambda v : 'HH-LP' if v>hf_thresh else 'LH-LP')

    # map cell barcodes to HH-LP or LH-LP
    cell_lp_dct = cell_highest_hf_map.set_index(cell_bc_key)['LP_type'].to_dict()

    # save whether cell has HHLP, LHLP or None
    variant_df[hhlp_type] = variant_df[cell_bc_key].map(cell_lp_dct).fillna('None')

    # True if cell has HHLP, False if not
    variant_df[hhlp_true_false] = variant_df[hhlp_type].map(lambda v : True if v in ['HH-LP'] else False)

    return variant_df

    # outdated:
    # variant_df[added_col] = variant_df[group_mut_key].isin(unique_muts_hhlp.index)
    # unique_muts = variant_df.groupby(group_mut_key).median(numeric_only=True)    
    # unique_muts_hhlp = unique_muts[(unique_muts[hf_key]>=hf_thresh) & (unique_muts[prev_key]<=prev_thresh)].copy()
