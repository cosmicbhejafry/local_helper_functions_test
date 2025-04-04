import pandas as pd
import numpy as np
import logging
from datetime import datetime
import os
import scanpy as sc
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        
def add_grouped_prevalence(variant_df,group_pos_dict,group_key,ct_key,cell_key='cell_id',pos_key='POS',depth_thr=10):
    
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
    
    # added_col_names.append(f'prevalence_{group_key}')
                    
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
    variants_snv = variant_df[variant_df.apply(lambda row: len(row['REF']) == 1 and len(row['ALT']) == 1, axis=1)].copy()
    
    variants_snv['donor_MUT'] = variants_snv[donor_key] + '_' + variants_snv[mut_key]
    #     variants_snv.loc[:, 'donor_MUT'] = variants_snv['sample_id'] + '_' + variants_snv['MUT']

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

def add_hhlp_col_var(variant_df,prev_key,hf_key='HF',hf_thresh=0.3,prev_thresh=0.01,cell_bc_key='cell_id'):
    
    ## modify 
    # If using pandas groupby().apply(max), switch to groupby().agg(max) for better performance.

    hhlp_type = f'hhlp_{hf_thresh}_{prev_thresh}'
    hhlp_true_false = f'hhlp_{hf_thresh}_{prev_thresh}_bool'

    # map cells with LP mutation to the max heteroplasmy of LP mut in that cell
    cell_highest_hf_map = variant_df[variant_df[prev_key]<=prev_thresh].groupby(cell_bc_key)[hf_key].apply(np.maximum.reduce)    
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

def dataframe_to_dict(input_df,dict_key_col,dict_val_col):
    # would be good to add tests - TODO
    return input_df[[dict_key_col,dict_val_col]].set_index(dict_key_col).to_dict()[dict_val_col]

def add_true_cryptic_col_var(variant_df,group_mut_key,prev_key,hf_key='HF',hf_thresh=0.3,prev_thresh=0.01,cell_bc_key='cell_id'):
    
    ## TODO 
    
    hhlp_type = f'tc_{hf_thresh}_{prev_thresh}'
    hhlp_true_false = f'tc_{hf_thresh}_{prev_thresh}_bool'

    # list of high heteroplasmy cryptic mutations:
    lp_var_df = variant_df[variant_df[prev_key]<=prev_thresh].copy()

    lp_var_df['is_cryptic'] = lp_var_df[group_mut_key].map(lp_var_df.groupby(group_mut_key)[cell_bc_key].count()==1)

    cryptic_lp_var_df = lp_var_df[lp_var_df['is_cryptic']].copy()

    cryptic_lp_var_df['cryptic_type'] = cryptic_lp_var_df[hf_key].map(lambda v : 'HH-TC' if v>hf_thresh else 'LH-TC')
       
    # for each cell: if the cell has a high heteroplasmy cryptic mutation then 'HH-TC'

    # save whether cell has HHLP, LHLP or None
    variant_df[hhlp_type] = variant_df[cell_bc_key].map(cryptic_lp_var_df.set_index(cell_bc_key)['cryptic_type'].to_dict()).fillna('None')

    # True if cell has HHLP, False if not
    variant_df[hhlp_true_false] = variant_df[hhlp_type].map(lambda v : True if v in ['HH-TC'] else False)

    return variant_df

#### ------------ USEFUL FUNCTIONS ---------------

# can go into general helpers
def load_data(batch_base_dir,file_nm):
    """
    Load the filtered h5ad object.
    """
    try:
        batch_filtered_outs = os.path.join(batch_base_dir, file_nm)            
        return sc.read_h5ad(batch_filtered_outs)
    except FileNotFoundError:
        logger.error(f"File not found for batch {file_nm}")
        raise
    except Exception as e:
        logger.error(f"Error loading data for {file_nm}: {e}")
        raise

class GeoGuessPipelineError(Exception):
    pass
        
def filter_gex_by_coverage(filtered_h5, cell_id_key_h5, cell_id_key_cov,current_date):
    """
    Filter gene expression data based on cell barcodes that have coverage.
    """
    try:
        # Check if necessary keys are present in the input
        if cell_id_key_h5 not in filtered_h5.obs:
            raise KeyError(f"{cell_id_key_h5} not found in filtered_h5.obs")
        if 'coverage0' not in filtered_h5.uns: 
            raise KeyError(f'coverage0 not in filtered_h5.uns')
        if cell_id_key_cov not in filtered_h5.uns['coverage0']:
            raise KeyError(f"Coverage data with key {cell_id_key_cov} not found in filtered_h5.uns['coverage0']")
                    
        # Extract gene expression and coverage barcodes
        gex_bcs = set(filtered_h5.obs[cell_id_key_h5])
        cov_bcs = set(filtered_h5.uns['coverage0'][cell_id_key_cov])

        # Check if coverage is a subset of gene expression
        if not cov_bcs.issubset(gex_bcs):
            raise GeoGuessPipelineError(f'coverage has Not been filtered for Gex - please filter before running this function')
        
        # Find common barcodes
        common_bcs = gex_bcs.intersection(cov_bcs)

        # Filter gene expression data to keep only common barcodes
        filtered_h5 = filtered_h5[filtered_h5.obs[cell_id_key_h5].isin(common_bcs)].copy()

        # Log the results: how many barcodes were filtered
        logger.info(f"Filtered {len(gex_bcs) - len(common_bcs)} barcodes from {len(gex_bcs)} gene expression barcodes")
        logger.info(f"{len(common_bcs)} common barcodes retained")
        
        tmp_str1 = f'filter gex to only include barcodes with sufficient coverage'
        filtered_h5.uns['filteringForGexByCov'] = f'{tmp_str1}, modified {current_date}' 

        return filtered_h5

    except KeyError as e:
        logger.error(f"KeyError: {e}")
        raise
    except Exception as e:
        logger.error(f"An error occurred while filtering gene expression by coverage: {e}")
        raise

def concat_coverage(filtered_h5):
    """
    Concatenate coverage data.
    """
    try:
        # Ensure all coverage keys exist
        cov_keys = [f'coverage{i}' for i in range(6)]
        missing_keys = [key for key in cov_keys if key not in filtered_h5.uns]

        if len(missing_keys) > 0:
            raise ValueError(f"Missing coverage keys in `filtered_h5.uns`: {missing_keys}")

        if len(list(filtered_h5.uns['coverage0'].index.names))!=1:
            raise GeoGuessPipelineError('split coverage object has multi-index: unsupported saving type')

        cov_inx_lst = [v for v in filtered_h5.uns['coverage0'].columns if not v.isdigit()]

        if cov_inx_lst is None:    
            cov_concat = pd.concat([filtered_h5.uns[f'coverage{i}'] for i in range(6)], axis=1)

        else: 
            cov_concat = pd.concat([filtered_h5.uns[f'coverage{i}'].set_index(cov_inx_lst) for i in range(6)], axis=1)
    
    except Exception as e:
        logger.error(f"Error: {e}")
        raise
        
    logger.info(f"coverage concatenated")    
    return cov_concat

def cov_concat_reset_index_drop(cov_concat,cols_to_drop=None):
    """
    Reset index of `cov_concat` and optionally drop specified columns.

    Parameters:
    - cov_concat (pd.DataFrame): The concatenated coverage DataFrame.
    - cols_to_drop (list, optional): List of column names to drop after resetting index.

    Returns:
    - pd.DataFrame: DataFrame with the index reset and selected columns dropped.
    """    
    if cols_to_drop is None:
        logger.info(f"coverage index reset")        
        return cov_concat.reset_index()
    
    else:                
        valid_cols = [col for col in cols_to_drop if col in cov_concat.index.names]       
        logger.info(f"coverage index reset, drop {valid_cols}")        
        return cov_concat.reset_index().drop(valid_cols, axis=1)
    
def filter_by_min_per_celltype(filtered_h5,cov_concat, vars_df, donor_key, cid_h5,cid_cov, ct_str,min_cells_per_ct = 100):
    """
    Filter by minimum number of cells per cell type to calculate prevalence.
    """
    ## potential issues here: .isin() performance optimization, separation of donor and ct
    
    donor_ct_key = 'donor_ct'
    cov_concat[donor_ct_key] = cov_concat[donor_key] + '_' + cov_concat[ct_str]
    
#     logger.info(cov_concat[donor_ct_key])
    
    # Count number of cells per donor-cell type pair
    num_ct_per_donor_df = cov_concat.groupby(donor_ct_key)[cid_cov].count().reset_index()
    num_cells_per_ct = num_ct_per_donor_df[cid_cov].astype(int)
    num_ct_per_donor_df_passing = num_ct_per_donor_df[num_cells_per_ct >= min_cells_per_ct].copy()
    num_ct_per_donor_df_passing.rename(columns={cid_cov: 'num_ct_per_donor'}, inplace=True)

    # Split donor_ct into donor_id and cell type
    for i, temp_col in enumerate([donor_key, ct_str]):
        num_ct_per_donor_df_passing[temp_col] = num_ct_per_donor_df_passing[donor_ct_key].apply(lambda v: v.split('_')[i])
#         .str.split("_", expand=True)

    cov_passing = cov_concat[cov_concat[donor_ct_key].isin(num_ct_per_donor_df_passing[donor_ct_key])].copy()
    vars_df_passing = vars_df[vars_df[cid_cov].isin(cov_passing[cid_cov])].copy()

    # Filter `filtered_h5` based on `cov_passing`
    filtered_h5 = filtered_h5[filtered_h5.obs[cid_h5].isin(cov_passing[cid_cov])].copy()

    # Add filtering metadata
    current_date = datetime.now().strftime("%Y-%m-%d")
    tmp_str1 = f"Only keep (donor, cell type) pairs with at least {min_cells_per_ct} cells per donor."
    tmp_str2 = f"i.e., if donor1 has 99 T cells, exclude T cells from donor1."
    filtered_h5.uns["filteringForPrevalenceCalc"] = f"{tmp_str1} {tmp_str2} Modified {current_date}."
    
    logger.info(f"Filtered by cell type with at least {min_cells_per_ct} cells per donor")
    
    return filtered_h5,cov_passing, vars_df_passing


def find_tot_site_stats(cov_passing,cell_key,depth_thresh=10,sites_thresh=100):
    """
    Compute total depth, total sites passing a depth threshold, and average depth per site.
    
    Parameters:
        cov_passing (pd.DataFrame): Coverage data per cell.
        cell_key (str): Column name identifying cells.
        depth_thresh (int): Minimum depth threshold for a site
        sites_thresh (int): Minimum number of sites with passing depth
    
    Returns:
        dict: {'tot_depth': dict, 'tot_sites': dict, 'avg_depth_per_site': dict}
    """    
    try:
        
        cov_inx_lst = [v for v in cov_passing if not v.isdigit()]        
        # total depth
        tot_depth = cov_passing.set_index(cov_inx_lst)[cov_passing.set_index(cov_inx_lst)>=depth_thresh].sum(axis=1)

        # total sites passing
        tot_sites = (cov_passing.set_index(cov_inx_lst)>=depth_thresh).sum(axis=1)        
        if tot_sites.shape[0] != tot_sites[tot_sites>sites_thresh].shape[0]:
            raise GeoGuessPipelineError(f'coverage has cells that do not pass total sites threshold')

        avg_depth_per_site = tot_depth/tot_sites

        cell_avg_depth_dict = dataframe_to_dict(avg_depth_per_site.reset_index(),cell_key,0)        
        td_dict = dataframe_to_dict(tot_depth.reset_index(),cell_key,0)    
        ts_dict = dataframe_to_dict(tot_sites.reset_index(),cell_key,0)

        return {'tot_depth' : td_dict, 'tot_sites': ts_dict, 'avg_depth_per_site' : cell_avg_depth_dict}

    except KeyError as e:
        logger.error(f"Missing key in DataFrame: {e}")
    except ValueError as e:
        logger.error(f"Value error occurred: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        
    return None 


def map_tot_site_stats(filtered_h5,cov_passing,cid_h5, cid_cov,map_h5_index=False):
    
    try:
        depth_thresh = int(filtered_h5.uns['variantThresholds'].loc['minDF'].iloc[0])
        sites_thresh = int(filtered_h5.uns['variantThresholds'].loc['minSites'].iloc[0])

        tot_summary_dict = find_tot_site_stats(cov_passing,cid_cov,depth_thresh=depth_thresh,sites_thresh=sites_thresh)

        for kv in tot_summary_dict.keys():
            if map_h5_index==False:
                filtered_h5.obs[kv] = filtered_h5.obs[cid_h5].map(tot_summary_dict[kv])
            else:
                filtered_h5.obs[kv] = filtered_h5.index.map(tot_summary_dict[kv])            
                
        logger.info(f'added columns {list(tot_summary_dict.keys())} to filtered_h5')
        return filtered_h5
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        
    return None


## function to add helper columns ??

def add_hhlp_cols_h5(filtered_h5,variants_snv,cid_h5,cid_cov,patho_key='MutPred_Labels',s_ns_key='S_NS',prev_key='prevalence_donor_ct',hhlp_key='hhlp_0.3_0.01'):
    
    """
    add hhlp col
    """
    
    try:
        prev_thresh = float(filtered_h5.uns['hhlpThresholds'].loc['prevThresh'].iloc[0])
        hf_thresh = float(filtered_h5.uns['hhlpThresholds'].loc['HFThresh'].iloc[0])

        vars_lp = variants_snv[variants_snv[prev_key]<=prev_thresh].copy()

        lp_vals = ['HH-LP','LH-LP']
        hhlp_str = 'HH-LP'
        lhlp_str = 'LH-LP'    

        # if a cell is HH-LP --> then see which high heteroplasmy LP mutations it has
        vars_type_hhlp = vars_lp[(vars_lp[f'hhlp_{hf_thresh}_{prev_thresh}']==hhlp_str) & (vars_lp[f'HF']>hf_thresh)].copy()

        # for each cell -- set of HHLP types within it
        cell_hhlp_set = vars_type_hhlp.groupby(cid_cov)[s_ns_key].apply(set)

        other_muts_key = 'Many Types'

        cell_hhlp_labels = cell_hhlp_set.map(lambda v : str(min(v)) if len(v)==1 else other_muts_key)
        cell_hhlp_labels = cell_hhlp_labels.reset_index().copy()        

        # for each cell -- set of pathogenicities within it
        cell_patho_set = vars_type_hhlp.groupby(cid_cov)[patho_key].apply(set).copy()
        # cell_patho_set.value_counts()
        patho_mix_key = 'Patho Mix'
        cell_patho_map = cell_patho_set.map(lambda v : str(min(v-{'NA'})) if len(v-{'NA'})==1 else (patho_mix_key if len(v-{'NA'})==2 else 'NA'))

        cell_hhlp_labels['hhlp_patho_type'] = cell_hhlp_labels[cid_cov].map(cell_patho_map)

        # num hhlp in cell
        cell_hhlp_labels['num_hhlp_in_cell'] = cell_hhlp_labels[cid_cov].map(vars_type_hhlp.groupby(cid_cov)['donor_ct_mut'].count().astype(str))        

        cell_hhlp_labels = cell_hhlp_labels.rename(columns = {s_ns_key : 'hhlp_mut_type'}).copy()

        tmp_dict_cell_hhlp_type = dataframe_to_dict(variants_snv[[hhlp_key,cid_cov]].drop_duplicates(),cid_cov,hhlp_key)
        filtered_h5.obs[hhlp_key] = filtered_h5.obs[cid_h5].map(tmp_dict_cell_hhlp_type).fillna('None')        

        for colval in ['hhlp_mut_type', 'hhlp_patho_type', 'num_hhlp_in_cell']:
            filtered_h5.obs[colval] = filtered_h5.obs[cid_h5].map(dataframe_to_dict(cell_hhlp_labels,cid_cov,colval)).fillna('NA')    
            
            non_hhlp_keys = ['None', 'LH-LP']
            if filtered_h5.obs[filtered_h5.obs['hhlp_0.3_0.01'].isin(non_hhlp_keys)][colval].nunique() != 1:
                raise GeoGuessPipelineError(f'cells with no HHLP has non na value')
            
        logger.info('added hhlp cols to h5')
        return filtered_h5
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        
    return None

def save_cov_in_h5(cov_passing,filtered_h5_save):
    cov_inx_lst = [v for v in cov_passing if not v.isdigit()]
    
    cov = cov_passing.set_index(cov_inx_lst).copy()
    cov.columns = cov.columns.map(str)
    cols = cov.columns
    colSplits = np.array_split(cols, 6)
        
    for i, cin in enumerate(colSplits):
        filtered_h5_save.uns[f'coverage{i}'] = cov[cin].reset_index()
    
    return filtered_h5_save

def save_dict_in_h5_uns(df_dict,filtered_h5_save):
    
    for kv in df_dict.keys():
        filtered_h5_save.uns[kv] = df_dict[kv]
    
    return filtered_h5_save