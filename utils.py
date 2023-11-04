import numpy as np
import scanpy as sc
import pandas as pd
import anndata as ad
import pertpy as pt
import matplotlib.pyplot as plt

def plt_legend(x=None, y=None, **kwargs):
    if not x:
        x = 1.01
    if not y:
        y = 1.03
    plt.legend(bbox_to_anchor=(x, y), **kwargs)

def annotate(adata, params):
    """Adds metadata annotations and a 'perturbation' column to SplatteR simulated data."""
    adata.obs['group'] = [x.split('Path')[1] for x in adata.obs.Group.values]
    for col in params.columns:
        adata.obs[col] = adata.obs.group.map(dict(zip(params.index.astype(str), params[col])))

    adata.obs['log(DEProb)'] = np.log10(adata.obs.DEProb.values)
    adata.obs['perturbation'] = adata.obs.Group.replace({'Path1': 'control'})

def scanpy_setup(adata):
    """In-place."""
    if 'counts' in adata.layers:
        adata.X = adata.layers['counts'].copy()
    else:
        adata.layers['counts'] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e6, exclude_highly_expressed=True)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    sc.pp.pca(adata, use_highly_variable=True)
    adata.obs_names_make_unique()
    adata.var_names_make_unique()

def ctrl_categories_setup(adata, resolution):
    """In-place."""
    ctrl_adata = adata[adata.obs.perturbation == 'control']
    sc.pp.neighbors(ctrl_adata)
    sc.tl.leiden(ctrl_adata, resolution=resolution)
    
    print(ctrl_adata.obs.leiden.value_counts())
    
    # set values in original adata
    adata.obs['leiden'] = np.nan
    adata.obs['leiden'][ctrl_adata.obs.index] = ctrl_adata.obs['leiden'].values

### Sampling and evaluation-specific utlities ###
def sample_and_merge_control(adata, control_key, n=5):
    """
    Merge `n` control groups determined using leiden into the original adata,
    labeled under `'perturbation'`.

    Parameters
    ----------
    adata : AnnData
    n : int, optional
        Number of control categories (leiden clusters) to merge (default is 5).
    """
    control = adata[adata.obs.perturbation == control_key]
    control.obs['perturbation'] = control.obs['perturbation'].astype(str)
    for cat in range(n):
        idx = control[control.obs.leiden == str(cat)].obs.index
        control.obs['perturbation'][idx] = control_key + str(cat)
        
    return ad.concat([adata[adata.obs.perturbation != control_key], control], join='outer')

def sample_and_merge_control_random(adata, control_key_or_indices, n=5):
    """
    Randomly sample control data and merge it with the original dataset.

    Parameters
    ----------
    adata : anndata.AnnData
        Anndata dataset with `'perturbation'` in .obs.
    n : int (default: 5)
        Number of control categories to split and merge.

    Returns
    -------
    anndata.AnnData
        A new Anndata dataset with the sampled control data merged into the original dataset.
        The `perturbation` column is updated to mark which are the sampled cells.
    """
    if type(control_key_or_indices) is str:
        control = adata[adata.obs.perturbation == control_key_or_indices]
    else:
        control = adata[control_key_or_indices]
    indices = list(control.obs.index)
    np.random.shuffle(indices)

    # floor division and relabel
    n_per_control = control.shape[0] // n
    new = control[indices[:n_per_control*n]]  # equivalent to shuffling the control cells
    new_label = []
    for i in range(n):
        new_label += [f'control{i}']*n_per_control
    new.obs['perturbation'] = new_label

    no_ctrl = adata[~adata.obs_names.isin(control.obs_names)]
    return ad.concat([no_ctrl, new], join="outer")

def remove_groups(adata, min_cells):
    """
    Remove categories with fewer than `min_cells` cells. If there are more than 100 categories remaining, randomly select 100.

    Parameters
    ----------
    adata : anndata.AnnData
        Anndata dataset with `'perturbation'` in .obs.
    min_cells : int
        The minimum number of cells (inclusive) required for a category to be kept.

    Returns
    -------
    anndata.AnnData
        A new Anndata dataset with perturbations that have at least `min_cells` cells.
    """
    group_counts = adata.obs["perturbation"].value_counts()
    selected_groups = group_counts[group_counts >= min_cells].index
    selected_groups_c = [x for x in selected_groups if 'control' in x]
    selected_groups_p = [x for x in selected_groups if 'control' not in x]
    print("number of perturbations above min count:", len(selected_groups_p), flush=True)

    # remove until 100 left
    if len(selected_groups_p) > 100:
        selected_groups_p = np.random.choice(selected_groups_p, size=100, replace=False)
    selected_groups = set(selected_groups_p) | set(selected_groups_c)

    return adata[adata.obs["perturbation"].isin(selected_groups)]

def subsample(adata, n_cells, groupby='perturbation'):
    """
    Subsample all perturbations to contain at most `n_cells`.

    Parameters
    ----------
    adata : anndata.AnnData
        The input Anndata dataset with `'perturbation'` in .obs.
    n_cells : int
        The maximum number of cells allowed for a perturbation after subsampling.

    Returns
    -------
    anndata.AnnData
        A new Anndata dataset with perturbations subsampled to contain at most the specified number of cells.
    """
    groups = adata.obs.groupby(groupby).apply(lambda x: x.sample(n=n_cells, random_state=0, replace=False))
    cells = [i for _, i in groups.index]
    new = adata[adata.obs_names.isin(cells)]
    return new

def generate(cond, train, min_cells=500):
    """
    Filter perturbations with fewer than 500 cells and subsample each perturbation to ncell cells.

    Parameters
    ----------
    cond : int
        Integer representing number of cells to subsample.
    train : anndata.AnnData
        The input Anndata dataset containing all conditions + `control`. Must have a column in `.obs`
        named `'perturbation'`

    Returns
    -------
    list
        A list of Anndata datasets, each representing a different experimental scenario.
    """
    # Filter out perturbations with not enough cells
    perturbation_counts = train.obs['perturbation'].value_counts()
    filter_condition = perturbation_counts >= min_cells
    filtered_train = train[train.obs['perturbation'].map(filter_condition)]
    
    if filtered_train.shape[0] == 0:
        raise ValueError(f'All conditions had fewer than {min_cells} cells.')

    # Subsample ncell cells for each perturbation
    groups = filtered_train.obs.groupby("perturbation").apply(lambda x: x.sample(n=cond, random_state=1, replace=False))
    selected_names = [name for _, name in groups.index]
    new_train = filtered_train[groups.index.get_level_values(1)]

    return new_train

def inplace_check(metrics, results, res, recompute=False):
    if res.res_string in results and not recompute:
        res = results[res.res_string]

    res.compute_pwdf(metrics, recompute=recompute)
    results[res.res_string] = res

## Calculating pairwise dfs
def get_pwdf_per_condition(target_adata, metrics, controls, cond_label, rep='pca'):
    """
    Computes a pwdf dict where keys are a description of the settings.
    Always computes on ALL features of the `target_adata` that is passed in.

    Parameters:
    -----------
    target_adata : AnnData
    metrics : list
        A list of distance metrics to compute pairwise distances betewen conditions.
    cond_label : str
        A label to identify the condition in the resulting DataFrames.
        Does not contain the metric name.
    rep : str (default, pca)
        The data representation to use ('pca', 'lognorm', or 'counts').
    """
    if type(controls) is not list:
        raise ValueError(f"Got {controls} for controls. Did you pass your arguments in in the right order?")

    calc_ndegs = False
    if 'n_genes' in target_adata.uns.keys():
        calc_ndegs = True
        print('Warning: calculating n_degs. Please make sure this is desired behavior.')

    def df_from_onesided(distance, adata, controls, **kwargs):
        dists = []
        for group in controls:
            dist = distance.onesided_distances(adata, 'perturbation', selected_group=group, show_progressbar=False, n_jobs=-1)
            dists.append(dist)
        return pd.concat(dists, axis=1)

    dfs = {}
    for metric in metrics:
        if rep == 'pca':
            sc.pp.pca(target_adata, use_highly_variable=False)  # rerun PCA in case the number of features has changed
            distance = pt.tools.Distance(metric=metric)
        elif rep == 'lognorm':
            try:  # sparse
                target_adata.layers['lognorm'] = target_adata.X.A
            except AttributeError:
                target_adata.layers['lognorm'] = np.asarray(target_adata.X)
            distance = pt.tools.Distance(metric=metric, layer_key='lognorm')
        elif rep == 'counts':
            distance = pt.tools.Distance(metric=metric, layer_key='counts')
        else:
            raise ValueError('`rep` must be one of pca, lognorm, or counts.')

        # do something completely different when evaluating only on DEGs
        if calc_ndegs:
            pwdf = calc_DEG_pwdf(distance, target_adata, controls)
        else:
            pwdf = df_from_onesided(distance, target_adata, controls)
        pwdf.columns = controls
        dfs[metric + '-' + str(cond_label)] = pwdf.T

    return dfs

def calc_DEG_pwdf(distance, target_adata, controls):
    """calculates a distance per perturbation on the respective DEGs. Uses .uns['rank_genes_groups'] which must be assigned manually"""
    ndegs = target_adata.uns['n_genes']

    dfs = []
    for c in controls:
        res = {}
        for p in target_adata.obs.perturbation.unique():
            if p == c:
                res[p] = 0
                continue

            top_genes = sc.get.rank_genes_groups_df(target_adata, group=p).names.values[:ndegs]
            subset = target_adata[:, top_genes]
            # sc.pp.pca(subset, use_highly_variable=False)  # rerun PCA in case it's being used # can't use like this too slow
            d = distance.onesided_distances(subset, 'perturbation', selected_group=c, groups=[p], show_progressbar=False, n_jobs=-1)
            res[p] = d[p]
        dfs.append(pd.DataFrame.from_dict(res, orient='index'))
    collat = pd.concat(dfs, axis=1)
    collat.index.name = 'perturbation'
    return collat

## Plotting
def get_flat_df(pwdfs, controls=None, label='condi'):
    """
    Transform a dictionary of pairwise distance dataframes into a flat dataframe,
    averaged per control condition (versus across - see get_melted_df_per_perturbation).

    Parameters:
    -----------
    pwdfs : dict
        A dictionary where keys are metric names + exp. conditions, and values are pairwise distance dataframes.
    controls : list
        List of control conditions for averaging distances.
    label : str, optional
        Label for the experimental condition. Default is 'condi'.

    Returns:
    --------
    melted_df : pandas.DataFrame
        A melted dataframe with metric name, distance, and condition label.
    """
    res_dict = {"avg_dist": [], "metric": [], label: []}

    for metric_condi, pwdf in pwdfs.items():
#        if controls is None:
#            controls = [x for x in pwdf.columns if 'control' in x]

        # average distance per control = source of variation
        ctrl_stim = pwdf.loc[controls, :]
        ctrl_stim = ctrl_stim.drop(controls, axis=1)
        avg_dists = ctrl_stim.mean(1).values

        res_dict["avg_dist"].append(avg_dists)
        res_dict["metric"].append(metric_condi.split('-')[0])
        try:
            res_dict[label].append(int(metric_condi.split('-')[1]))
        except ValueError:  # not an integer
            res_dict[label].append(metric_condi.split('-')[1])

    df = pd.DataFrame.from_dict(res_dict)

    # Create a flat structure for the data
    flat_data = []
    for avg_dist, metric, condi in zip(res_dict['avg_dist'], res_dict['metric'], res_dict[label]):
        for value in avg_dist:
            flat_data.append({'avg_dist': value, 'metric': metric, label: condi})

    melted_df = pd.DataFrame(flat_data)
    
    return melted_df

def normalize_per_metric(melted_df, label='avg_dist'):
    """Given a dataframe with distances and a column with metrics names,
    scales metrics so they are plotted on the same relative value scale.
    """
    # normalize per metric (correct? maybe I should just set the means to be equal? idk)
    df = melted_df.copy()

    for m in df.metric.unique():
        avg_dists = df[df.metric == m][label].values
        max_val = np.max(avg_dists)
        min_val = np.min(avg_dists)
        normalized_arr = (avg_dists - min_val) / (max_val - min_val)
        df[label][df.metric == m] = normalized_arr
    
    return df

def get_distance_per_perturbation(pwdfs, metrics, controls, label='condi'):
    """
    Return the distances per perturbation, averaged over controls.

    Parameters
    ----------
    pwdfs : dict
        A dictionary containing pairwise distance DataFrames for different metrics.

    Returns
    -------
    list of pandas.DataFrame, dict
        A list of DataFrames, each containing average distances for perturbations with a 'metric' column.
        The second return value is a dictionary with control-to-control average distances for each metric.
    """
    dfs = {}
    for key, pwdf in pwdfs.items():
        metric = key.split('-')[0]
        cond_name = key.split('-')[1]

        # Get average distance across controls per perturbation
        ctrl_stim = pwdf.loc[controls, :].drop(controls, axis=1)
        distances = pd.DataFrame(ctrl_stim.mean(0))

        # Get average distance of control to control (exclude diagonal) and add to dataframe
        ctrl_ctrl = pwdf.loc[controls, controls]
        ctrl_mean = ctrl_ctrl.replace(0, np.NaN).mean()
        distances = pd.concat([distances, pd.DataFrame(ctrl_mean)])

        # add overall control mean to dataframe
        distances.loc['control'] = [ctrl_mean.mean()]
#        print(f"avg ctrl dist for {metric}-{cond_name}:", ctrl_dist)

        distances['metric'] = metric
        distances[label] = cond_name
        dfs[key] = distances
        
    return dfs

def add_rank_col(df, single_metric_df, sort_per_control=True):
    """In-place. Rank the perturbations using the first distance's dataframe.
    Dataframes may be the same, but `df` must still be formatted properly.

    Control rankings are assigned without counting other controls as conditions.
    """
    rank_df = single_metric_df.sort_values(by=0).reset_index().reset_index()
    rank_dict = dict(zip(rank_df['perturbation'], rank_df['index']))
    df['rank'] = df['perturbation'].map(rank_dict)
    
    # add an is_control column
    df['is_control'] = ['control' if 'control' in x else 'perturbation' for x in df['perturbation'].values]

    if sort_per_control:
        n_controls = df.is_control.value_counts()['control']
        df['rank'][df.is_control == 'perturbation'] -= n_controls
        ctrl_subset = df[df.is_control == 'control'].sort_values(by='rank')
        ctrl_subset['rank'] -= np.array(range(n_controls))
        df['rank'].loc[ctrl_subset.index] = ctrl_subset['rank']

    return df

def get_melted_df_per_perturbation(pwdfs, metrics, controls, label, ndegs=None, reference=None, adata=None):
    """
    reference : str
        Rank reference label (the condition to evaluate on).
    """
    dfs = get_distance_per_perturbation(pwdfs, metrics, controls, label=label)

    df = pd.concat(dfs.values()).reset_index()
    df.columns = ['perturbation', 'distance', 'metric', label]

    # add metadata labels
    if ndegs:
        df['n_degs'] = df.perturbation.map(ndegs)
    if reference:
        add_rank_col(df, dfs[reference], sort_per_control=False)
    else:
        add_rank_col(df, list(dfs.values())[0], sort_per_control=False)
    if adata:
        df['n_cells'] = df.perturbation.map(adata.obs.perturbation.value_counts().to_dict()).astype(float)
        df['log(n_cells)'] = np.log(df['n_cells'])
    
    return df

def generate_mix_control_into_perturbed(adata, n_mix=100):
    """
    Mix control cells into the perturbed conditions in a single-cell RNA-seq dataset.

    Parameters
    ----------
    adata : AnnData
        Contains a `'perturbation'` and `'mixin'` column in .obs.
    n_mix : int (default: 100)
        Number of control cells to mix into each perturbed condition.

    Raises
    ------
    ValueError
        If there are not enough control cells left for a robust 5-way split after mixing.
    """
    # calculate how many mixins are needed
    n_perts = len(adata.obs.perturbation.unique()) - 1
    cells_needed = n_perts*n_mix

    if adata.obs.perturbation.value_counts()['control'] < cells_needed:
        raise ValueError("Not enough control cells left for mixing.")

    # make a new adata so we can still use the perturbation column
    mixin = adata.copy()
    mixin.obs['perturbation'] = mixin.obs['mixin'].values
    mix_idxs = np.random.choice(mixin[mixin.obs.perturbation == 'mixin'].obs_names, size=cells_needed, replace=False)

    perturbations = set(mixin.obs.perturbation.unique()) - set(['control'])
    mixes = sample_and_merge_control_random(mixin, mix_idxs, n=len(perturbations))

    controls = set(mixes.obs.perturbation.unique()) - set(mixin.obs.perturbation.unique())
    mixes.obs['perturbation'] = mixes.obs.perturbation.replace(dict(zip(controls, perturbations)))
    return mixes

def get_ranked_df_per_perturbation(pwdfs, metrics, controls, label='condi'):
    """
    Rank all perturbations for each metric-condition dataframe.

    Parameters:
    -----------
    metrics : list
        A list of metrics to calculate rankings for.
    controls : list
        A list of control conditions.
    label : str, optional
        The label for the control condition in the DataFrame, defaults to 'condi'.
    """
    dfs = get_distance_per_perturbation(pwdfs, metrics, controls, label=label)
    df = pd.concat(dfs.values()).reset_index()
    df.columns = ['perturbation', 'distance', 'metric', label]

    ## add rank column to each dataframe individually while removing 'control'
    ## added in get_distance_per_perturbation
    for key, single_metric_df in dfs.items():
        full_df = single_metric_df.drop('control').reset_index()
        full_df.columns = ['perturbation', 'distance', 'metric', label]
        dfs[key] = add_rank_col(full_df, single_metric_df)

    return pd.concat(dfs.values())

def calc_rank_percentile(ind_ranked, target='control'):
    """
    Calculate rank percentiles for a specific perturbation condition.

    Parameters:
    -----------
    ind_ranked : pd.DataFrame
        A Pandas DataFrame from `get_ranked_df_per_perturbation`.
    target : str, optional
        The specific condition for which rank percentiles are calculated.
        Defaults to 'control'.
    """
    if type(target) is str:
        target_ranks = ind_ranked[ind_ranked.perturbation == target]
    else:
        target_ranks = ind_ranked[ind_ranked.perturbation.isin(target)]
    target_ranks['rank'] = target_ranks['rank']/len(ind_ranked.perturbation.unique())
    target_ranks = target_ranks.reset_index().drop(columns=['index'])
    return target_ranks

def generate_DEG_adatas(adata, filtered, included_perturbations, n):
    """
    Generate AnnData subsets with the top differentially expressed
    genes (DEGs) for each perturbation condition.

    Parameters:
    ----------
    adata : AnnData
        The original AnnData object containing gene expression data.
    filtered : AnnData
        The AnnData object after any filtering or preprocessing steps.
    included_perturbations : list
        List of perturbation conditions for which DEGs will be computed and subset.
    n : int
        Number of top DEGs to select for each perturbation condition.
    """
    subset_adatas = []

    for p in included_perturbations:
        # Subset the original AnnData object for a specific perturbation condition
        subset_adata = filtered[filtered.obs['perturbation'] == p].copy()

        # get the top 50 DEGs from the original adata, versus control
        top_genes = sc.get.rank_genes_groups_df(adata, group=p).names.values[:n]

        # Subset the AnnData for this set of top genes
        subset_adata = subset_adata[:, top_genes]
        subset_adata.var_names = list(range(n))  # reset var_names to allow concat

        subset_adatas.append(subset_adata)

    return ad.concat(subset_adatas)

def generate_sparsity(adata, obs, percentage):
    """Decreases a percentage of counts by 1 in the original adata,
    and then filters by the same filtering used to generate `obs`.
    """
    mtx = adata.layers['counts'].copy()

    # Calculate the number of values to select
    total_values = len(mtx.data)
    values_to_select = int(percentage / 100 * total_values)

    # Get a random sample of indices to select using numpy
    selected_indices = np.random.choice(total_values, values_to_select, replace=False)

    # Subtract 1 from the values at selected indices
    mtx.data[selected_indices] -= 1
    
    # Set all values below 0 to 0
    mtx.data[mtx.data < 0] = 0
    
    new_adata = adata.copy()
    new_adata.X = mtx
    new_adata.layers['counts'] = mtx
    
    sc.pp.normalize_total(new_adata, target_sum=1e6, exclude_highly_expressed=True)
    sc.pp.log1p(new_adata)
    
    new_adata = new_adata[obs.index]
    new_adata.obs = obs
    # return the original subset
    return new_adata

