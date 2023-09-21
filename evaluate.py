import pickle
import argparse
import numpy as np
import scanpy as sc
import pertpy as pt

from distance_result import DistanceResult
from utils import scanpy_setup
from utils import sample_and_merge_control_random, remove_groups, subsample, generate_sparsity
from utils import inplace_check

parser = argparse.ArgumentParser()

parser.add_argument("--save_file", type=str, default='test.pkl', required=True)
parser.add_argument("--dataset", type=str, default='', required=True)
parser.add_argument("--test_mode", dest='test_mode', default=False, action='store_true')

args = parser.parse_args()
test_mode = args.test_mode
dss_path = '/dss/dssfs02/lwp-dss-0001/pn36po/pn36po-dss-0001/di93zoj'
save_file = args.save_file

controls = ['control0', 'control1', 'control2', 'control3', 'control4']
metrics = ['euclidean', 'pearson_distance', 'mean_absolute_error', 'r2_distance', 'spearman_distance']
metrics += ['mmd', 'kl_divergence', 't_test', 'wasserstein']  # the experimental ones
metrics += ['edistance', 'mse', 'cosine_distance']  # newly added

print(f"running with test mode {test_mode}, dataset {args.dataset}, saving to {save_file}", flush=True)

### dataset-specific filtering ###
if args.dataset == 'sciplex_K562':
    adata = pt.data.srivatsan_2020_sciplex3()

    # sciplex3 is huge so we subset to the smallest cell type, which only contains the 24hr dose
    adata = adata[adata.obs.cell_line == 'K562']
    sc.pp.filter_genes(adata, min_cells=100)
    sc.pp.filter_cells(adata, min_genes=200)

    # separate out the perturbations by dose
    adata.obs['perturbation_name'] = adata.obs.perturbation.values
    adata.obs['perturbation'] = adata.obs['perturbation_name'].astype(str) + '_' + adata.obs.dose_value.astype(str)
    adata.obs['perturbation'] = adata.obs['perturbation'].replace({'control_0.0':'control'})

    n_min_cells = 270
elif args.dataset == 'sciplex_MCF7':
    adata = pt.data.srivatsan_2020_sciplex3()

    # sciplex3 is huge so we subset to the smallest cell type, which only contains the 24hr dose
    adata = adata[adata.obs.cell_line == 'MCF7']
    sc.pp.filter_genes(adata, min_cells=100)
    sc.pp.filter_cells(adata, min_genes=200)

    # separate out the perturbations by dose
    adata.obs['perturbation_name'] = adata.obs.perturbation.values
    adata.obs['perturbation'] = adata.obs['perturbation_name'].astype(str) + '_' + adata.obs.dose_value.astype(str)
    adata.obs['perturbation'] = adata.obs['perturbation'].replace({'control_0.0':'control'})

    n_min_cells = 270
elif args.dataset == 'norman':
    adata = pt.data.norman_2019()
    adata.obs['perturbation'] = adata.obs.perturbation_name
    
    n_min_cells = 390
else:
    raise ValueError('must pass available dataset')

if test_mode:
    print("Test mode: subsampling", flush=True)
    sc.pp.subsample(adata, .1)
    save_file = 'test.pkl'
    n_min_cells = int(n_min_cells/8)


### metric runs ###
scanpy_setup(adata)

# set filtered adata used for all runs
merged = sample_and_merge_control_random(adata, 'control', n=5)
filtered = remove_groups(merged, min_cells=n_min_cells)

print(adata, flush=True)
print("average number of counts per cell:", filtered.obs.ncounts.mean(), flush=True)
print("number of perturbations remaining:", len(filtered.obs.perturbation.unique()), flush=True)

# load previous file if it exists
try:
    with open(f'{dss_path}/{save_file}', 'rb') as file:
        results = pickle.load(file)
    print('starting with', results.keys(), flush=True)
except:
    results = {}

if test_mode:
    results = {}

current_keys = set(results.keys())

for rep in ['lognorm', 'counts', 'pca']:
    
    ### n_cells ###
    print('running n_cells', flush=True)
    experiment_condi = list(range(100, n_min_cells+10, 50)) + [n_min_cells]

    for ncell in experiment_condi:
        subset = subsample(filtered, ncell)[:, adata.var['highly_variable']]
        inplace_check(metrics, results, DistanceResult(subset, str(ncell), rep, 'n_cells'))


    ### n_HVGs ###
    print('running hvgs', flush=True)
    experiment_condi = [10, 50, 100, 500, 1000, 2000, 5000]

    ## separately compute top n HVGs
    feature_subsets = {}
    for n in experiment_condi:
        sc.pp.highly_variable_genes(adata, n_top_genes=n, flavor='seurat')
        feature_subsets[n] = list(adata.var_names[adata.var['highly_variable']])
    # reset highly_variable genes for the remaining experiments
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)

    for n, features in feature_subsets.items():
        subset = subsample(filtered, n_min_cells)[:, features]
        inplace_check(metrics, results, DistanceResult(subset, str(n), rep, 'n_genes'))
    
    
    ### libsize ###
    print('running libsize', flush=True)
    experiment_condi = list(range(10, 91, 10))

    for perc in experiment_condi:
        subset = generate_sparsity(
            adata[:, adata.var['highly_variable']],
            subsample(filtered, n_min_cells).obs,
            perc
        )
        count_mean = subset.layers['counts'].mean()
        inplace_check(metrics, results, DistanceResult(subset, "{:.3f}".format(count_mean), rep, 'count_mean'))
        
    break  # temporary for runtime
        
print('finished with', results.keys(), flush=True)
print('added keys:', set(results.keys())-current_keys, flush=True)
with open(f'{dss_path}/{save_file}', 'wb') as file:
    pickle.dump(results, file)
