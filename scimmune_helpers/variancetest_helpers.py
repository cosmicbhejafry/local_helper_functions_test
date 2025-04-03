import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import mannwhitneyu
from scipy.stats import nbinom
from statsmodels.base.model import GenericLikelihoodModel

# ------ parametrizing variance --------
def get_p_var(mu,sig):
    return mu/sig

def get_success_var(mu,sig):
    return (mu*mu)/(sig - mu)

def get_mu(p,success):
    return success*(1.0 - p)/p

def get_sig(p,success):
    return get_mu(p,success)/p

def _ll_nb2_var(y, mu, sig):
    prob = get_p_var(mu,sig)
    success = get_success_var(mu,sig)
    ll = nbinom.logpmf(y, success, prob)
    return ll

class NBin_var(GenericLikelihoodModel):
    def __init__(self, endog, exog = None, **kwds):
        super().__init__(endog, exog, **kwds)

    def nloglikeobs(self, params):
        sig = params[-1]
        mu = params[:-1]
        ll = _ll_nb2_var(self.endog, mu, sig)
        return -ll

    def fit(self, start_params=None, maxiter=1000, maxfun=1000, **kwds):
        # we have one additional parameter and we need to add it for summary
        if start_params == None:
            # Reasonable starting values
            start_params = [0.5,1.1]            
#             # intercept
#             start_params[-2] = np.log(self.endog.mean())
        return super(NBin_var, self).fit(start_params=start_params,
                                     maxiter=maxiter, maxfun=maxfun,
                                     **kwds)


# ------ parametrizing dispersion --------
def get_p_disp(mu,disp):
    return 1.0/(1.0 + disp*mu)
    
def get_success_disp(mu,disp):
    return 1.0/disp

def get_mu(p,success):
    return success*(1.0 - p)/p

def get_disp(p,success):
    return 1/success

def _ll_nb2_disp(y, mu, disp):
    prob = get_p_disp(mu,disp)
    success = get_success_disp(mu,disp)
    ll = nbinom.logpmf(y, success, prob)
    return ll

class NBin_disp(GenericLikelihoodModel):
    def __init__(self, endog, exog = None, **kwds):
        super().__init__(endog, exog, **kwds)

    def nloglikeobs(self, params):
        sig = params[-1]
        mu = params[:-1]
        ll = _ll_nb2_var(self.endog, mu, sig)
        return -ll

    def fit(self, start_params=None, maxiter=1000, maxfun=1000, **kwds):
        # we have one additional parameter and we need to add it for summary
        if start_params == None:
            # Reasonable starting values
            start_params = [0.5,1.1]            
#             # intercept
#             start_params[-2] = np.log(self.endog.mean())
        return super(NBin_disp, self).fit(start_params=start_params,
                                     maxiter=maxiter, maxfun=maxfun,
                                     **kwds)

# ------ parametrizing dispersion --------

def varianceParams_oneGroup_oneGene(h5_obj,obs_mask,gene_name,bootstrap=True,parametrization='variance'):    
        
    count_data = h5_obj[obs_mask,h5_obj.var_names==gene_name].layers['counts']

    num_nz = count_data.nnz
    
    count_data = count_data.toarray().flatten().copy()

    if parametrization=='variance':
        mod1 = NBin_var(count_data)
    
    if parametrization=='dispersion':
        mod1 = NBin_disp(count_data)
               
    res1 = mod1.fit()

    num_params = len(res1.params)

    res_dict = {'num_nz' : num_nz,'parametrization' : parametrization}

    res_dict.update({f'param{i}_val' : res1.params[i] for i in range(num_params)})

    res_dict.update({f'param{i}_bse' : res1.bse[i] for i in range(num_params)})    

    if bootstrap:
        boot_outs = res1.bootstrap()
        
        for i in range(num_params):
            
            for qv in [0.25,0.5,0.75]:
                res_dict[f'bootstrap_param{i}_{qv}_quantile'] = np.quantile(boot_outs[2][:,0],qv)                

            res_dict[f'bootstrap_param{i}_mean'] = boot_outs[0][i]
            res_dict[f'bootstrap_param{i}_bse'] = boot_outs[1][i]
        
            res_dict[f'bootstrap_param{i}_vals'] = boot_outs[2][:,i]            
            res_dict[f'bootstrap_param{i}_nans'] = np.isnan(boot_outs[2][:,i]).sum()        
        
    return res_dict

def varianceParams_manyGroup_manyGene(h5_obj,h5_group_key,gene_lst,bootstrap=True,parametrization='variance'):

    results_dict = {}
    
    # get groups
    group_keys = list(h5_obj.obs[h5_group_key].unique())
    
    for i,gk in enumerate(group_keys):
        # get boolean masks        
        gk_mask = h5_obj.obs[h5_group_key]==gk
        
        results_dict[gk] = {}
        
        # loop over gene names
        for gene_name in gene_lst:
            results_dict[gk][gene_name] = varianceParams_oneGroup_oneGene(h5_obj,gk_mask,gene_name,bootstrap=bootstrap,parametrization=parametrization)
            
            
    exclude_keys = ['bootstrap_param0_vals','bootstrap_param1_vals']
    
    df = pd.DataFrame([
        {"group": group, "name": gene, **{kv : results[kv] for kv in results.keys() if kv not in exclude_keys}}
        for group, gene in results_dict.items()
        for gene, results in gene.items()
    ])
        
    return results_dict, df

def distributionTest_bootstrapParams_twoGroup(boot_params_dict,grp0_key,grp1_key,gene_lst,num_params=2):
    
    #check grp0 and grp1 key in bootparams
    pval_results = {}
    
    for gene_name in gene_lst:
        
        pval_results[gene_name] = {}
        
        for i in range(num_params):

            val0 = boot_params_dict[grp0_key][gene_name][f'bootstrap_param{i}_vals']
            val1 = boot_params_dict[grp1_key][gene_name][f'bootstrap_param{i}_vals']        

            testprefix = 'mwu-'

            res_v, pv_v = mannwhitneyu(val0,val1)
            pval_results[gene_name][f'{testprefix}u_param{i}'] = res_v
            pval_results[gene_name][f'{testprefix}pval_param{i}'] = pv_v

            tmp_us = 2*res_v/(len(val0)*len(val1))
            pval_results[gene_name][f'{testprefix}rankbiserial_param{i}'] = tmp_us-1
            
    return pd.DataFrame(pval_results).T

def add_diff_to_pval_df(pval_df,df,group_key,grp1,grp2):

    df_diff = df.pivot(index='name', columns=group_key)

    diff_func = lambda colval : df_diff[(colval,grp1)] - df_diff[(colval,grp2)]
    boot_col = lambda i,qv : f'bootstrap_param{i}_{qv}_quantile'
    interval_overlap = lambda low_vals,high_vals : df_diff[boot_col(0,0.75)].min(axis=1) - df_diff[boot_col(0,0.25)].max(axis=1)

    iqr_olp = interval_overlap(df_diff[boot_col(0,0.25)],df_diff[boot_col(0,0.75)])

    pval_df['param0_diff'] = pval_df.index.map(diff_func('param0_val'))
    pval_df['param1_diff'] = pval_df.index.map(diff_func('param1_val'))
    pval_df['iqr_overlap'] = pval_df.index.map(iqr_olp)    
    
    return pval_df