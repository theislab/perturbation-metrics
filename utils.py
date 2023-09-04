import numpy as np
import scanpy as sc
import pandas as pd
import anndata as ad

def annotate(adata, params):
    """Adds metadata annotations and a 'perturbation' column to SplatteR simulated data."""
    adata.obs['group'] = [x.split('Path')[1] for x in adata.obs.Group.values]
    for col in params.columns:
        adata.obs[col] = adata.obs.group.map(dict(zip(params.index.astype(str), params[col])))

    adata.obs['log(DEProb)'] = np.log10(adata.obs.DEProb.values)
    adata.obs['perturbation'] = adata.obs.Group.replace({'Path1': 'control'})


def scanpy_setup(adata):
    """In-place."""
    adata.layers['counts'] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e6, exclude_highly_expressed=True)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=5000)
    sc.pp.pca(adata, use_highly_variable=True)

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
def sample_and_merge_control(adata, control, n=5):
    """
    Merge `n` control groups determined using leiden into the original adata,
    labeled under `'perturbation'`.

    Parameters
    ----------
    adata : AnnData
    control : AnnData
        Control dataset in AnnData format with 'leiden' clustering labels.
    n : int, optional
        Number of control categories (leiden clusters) to merge (default is 5).
    """
    control.obs['perturbation'] = control.obs['perturbation'].astype(str)
    for cat in range(n):
        idx = control[control.obs.leiden == str(cat)].obs.index
        control.obs['perturbation'][idx] = 'control' + str(cat)
        
    return ad.concat([adata, control], join='outer')

def remove_groups(adata, min_cells):
    """
    Remove categories with fewer than `min_cells` cells.

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
    return adata[adata.obs["perturbation"].isin(selected_groups)]

def subsample(adata, n_cells):
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
    groups = adata.obs.groupby("perturbation").apply(lambda x: x.sample(n=n_cells, random_state=0, replace=False))
    cells = [i for _, i in groups.index]
    new = adata[adata.obs_names.isin(cells)]
    return new

def generate(cond, train, min_cells=500):
    """
    Filter perturbations with fewer than 500 cells and subsample each perturbation to ncell cells.

    Parameters
    ----------
    cond : list
        A list of integers representing different experimental scenarios (number of cells to subsample).
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

def get_flat_df(pwdfs, controls, label='condi'):
    """
    Transform a dictionary of pairwise distance dataframes into a flat dataframe.

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

        # average distance per control = source of variation
        ctrl_stim = pwdf.loc[controls, :]
        ctrl_stim = ctrl_stim.drop(controls, axis=1)
        avg_dists = ctrl_stim.mean(1).values

        res_dict["avg_dist"].append(avg_dists)
        res_dict["metric"].append(metric_condi.split('_')[0])
        try:
            res_dict[label].append(int(metric_condi.split('_')[1]))
        except ValueError:  # not an integer
            res_dict[label].append(metric_condi.split('_')[1])

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

def get_distance_per_perturbation(pwdfs, cond_name, metrics, controls):
    """
    Return the distances per perturbation, averaged over controls.

    Parameters
    ----------
    pwdfs : dict
        A dictionary containing pairwise distance DataFrames for different metrics.
    cond_name : str
        Name of the experimental condition or dataset to subset to.

    Returns
    -------
    list of pandas.DataFrame, dict
        A list of DataFrames, each containing average distances for perturbations with a 'metric' column.
        The second return value is a dictionary with control-to-control average distances for each metric.
    """
    dfs = []
    ctrls = {}
    for metric in metrics:
        try:
            pwdf = pwdfs[f'{metric}_{cond_name}']
        except KeyError:  # maybe some didn't run through
            continue

        # Get average distance of control to control (exclude diagonal)
        ctrl_ctrl = pwdf.loc[controls, controls]
        ctrl_dist = ctrl_ctrl.sum().sum() / (25 - 5) 
        ctrls[metric] = ctrl_dist

        print("control:", ctrl_dist)

        # Get average distance across controls per perturbation
        ctrl_stim = pwdf.loc[controls, :].drop(controls, axis=1)
        distances = pd.DataFrame(ctrl_stim.mean(0))

        distances['metric'] = metric
        dfs.append(distances)
        
    return dfs, ctrls

def add_rank_col(df, single_metric_df):
    """In-place."""
    # rank the perturbations using one of the distances (edist in this case), and then plot all distances by that ranking
    rank_df = single_metric_df.sort_values(by=0).reset_index().reset_index()
    rank_dict = dict(zip(rank_df['perturbation'], rank_df['index']))
    df['rank'] = df['perturbation'].map(rank_dict)
    
    # add an is_control column
    df['is_control'] = ['control' if 'control' in x else 'perturbation' for x in df['perturbation'].values]

### Real data specific functions ###
def compile_from_pwdfs(pwdfs, metrics, controls, ndegs):
    """For real data."""
    dfs, _ = get_distance_per_perturbation(pwdfs, 'dixit', metrics, controls)

    df = pd.concat(dfs).reset_index()
    df.columns = ['perturbation', 'distance', 'metric']

    # add metadata labels
    df['n_degs'] = df.perturbation.map(ndegs)
    add_rank_col(df, dfs[0])
    df['n_cells'] = df.perturbation.map(filtered.obs.perturbation.value_counts().to_dict())
    df['log(n_cells)'] = np.log(df['n_cells'])
    
    return df
