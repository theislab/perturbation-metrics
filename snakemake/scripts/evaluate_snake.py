import pickle
import numpy as np
import anndata as ad
import scanpy as sc

from distance_result import DistanceResult
from ..utils.utils import subsample, generate_sparsity
from ..utils.utils import inplace_check

np.random.seed(0)

### Note that the default mode, without any flags, runs hvgs, ncells, and
### libsize across lognorm, counts, and pca representations.

test_mode = snakemake.params.test_mode
eval_mode = snakemake.params.eval_mode
with_DEGs = snakemake.params.with_DEGs
dataset = snakemake.input[0]
save_file = snakemake.output[0]
if eval_mode: save_file = save_file.replace('.pkl', '_sub.pkl')

controls = ['control0', 'control1', 'control2', 'control3', 'control4']
metrics = [snakemake.wildcards.metric]
rep = snakemake.wildcards.rep

print(f"running with test mode {test_mode}, dataset {dataset}, saving to {save_file}", flush=True)

# get prepared data object
adata = sc.read(dataset)
n_min_cells = 390 if dataset == 'norman' else 270

# here would be the space were you compute the evaluation for the metric and datasets specified
# I do not fully understand your code down there

results = {}

### n_HVGs ###
print('running hvgs', flush=True)
experiment_condi = [10, 50, 100, 500, 1000, 2000, 5000]
if eval_mode: experiment_condi = [1000, 2000]

## separately compute top n HVGs
feature_subsets = {}
for n in experiment_condi:
    sc.pp.highly_variable_genes(adata, n_top_genes=n, flavor='seurat')
    feature_subsets[n] = list(adata.var_names[adata.var['highly_variable']])
# reset highly_variable genes for the remaining experiments
sc.pp.highly_variable_genes(adata, n_top_genes=2000)

for n, features in feature_subsets.items():
    subset = subsample(adata, n_min_cells)[:, features]
    inplace_check(metrics, results, DistanceResult(subset, controls, str(n), rep, 'n_genes'))

### n_cells ###
print('running n_cells', flush=True)
experiment_condi = list(range(100, n_min_cells+10, 50)) + [n_min_cells]

for ncell in experiment_condi:
    subset = subsample(adata, ncell)[:, adata.var['highly_variable']]
    inplace_check(metrics, results, DistanceResult(subset, controls, str(ncell), rep, 'n_cells'))


### libsize ###
print('running libsize', flush=True)
experiment_condi = list(range(10, 91, 10))

for perc in experiment_condi:
    subset = generate_sparsity(
        adata[:, adata.var['highly_variable']],
        subsample(adata, n_min_cells).obs,
        perc
    )
    count_mean = subset.layers['counts'].mean()
    inplace_check(metrics, results, DistanceResult(subset, controls, "{:.3f}".format(count_mean), rep, 'count_mean'))


### n_DEGs ###
if with_DEGs:
    print('running ndegs', flush=True)
    if rep == 'pca':
        raise ValueError('Cannot run n_degs eval with pca right now due to runtime constraints.')
    experiment_condi = [10, 20, 30, 40, 50, 100, 150, 200, 300, 400]

    # create new adata to calculate DEGs and has run-specific info
    filt_wctrl = ad.concat([adata, adata[adata.obs.perturbation == 'control']])
    sc.tl.rank_genes_groups(
        filt_wctrl,
        groupby='perturbation',
        reference='control',
        rankby_abs=True
    )
    subset = subsample(adata, n_min_cells)
    subset.uns = filt_wctrl.uns # using a new adata which does not have the 'control' condition

    for n in experiment_condi:
        subset.uns['n_genes'] = n
        inplace_check(metrics, results, DistanceResult(subset, controls, str(n), rep, 'n_DEGs'))
        
print('finished with', results.keys(), flush=True)
with open(save_file, 'wb') as file:
    pickle.dump(results, file)
