import pickle
import argparse
import numpy as np
import anndata as ad
import scanpy as sc
import pertpy as pt

from distance_result import DistanceResult
from utils import scanpy_setup
from utils import sample_and_merge_control_random, remove_groups, subsample, generate_sparsity
from utils import inplace_check

np.random.seed(0)

parser = argparse.ArgumentParser()

parser.add_argument("--save_file", type=str, default='test.pkl', required=True)
parser.add_argument("--dataset", type=str, default='', required=True)
parser.add_argument("--test_mode", dest='test_mode', default=False, action='store_true')  # evaluate on subset
parser.add_argument("--eval_mode", dest='eval_mode', default=False, action='store_true')  # run bare min needed for table
parser.add_argument("--with_DEGs", dest='with_DEGs', default=False, action='store_true')  # add DEGs on lognorm to run

### Note that the default mode, without any flags, runs hvgs, ncells, and
### libsize across lognorm, counts, and pca representations.
### n_min_cells decides the perturbations which are bretained

args = parser.parse_args()
test_mode = args.test_mode
eval_mode = args.eval_mode
with_DEGs = args.with_DEGs
dss_path = '/dss/dssfs02/lwp-dss-0001/pn36po/pn36po-dss-0001/di93zoj'
save_file = args.save_file
if eval_mode: save_file += '_sub'

controls = ['control0', 'control1', 'control2', 'control3', 'control4']
metrics = ['euclidean', 'spearman_distance', 'mean_absolute_error']  # representative
metrics += ['r2_distance', 'pearson_distance', 'mse', 'cosine_distance']  # fast
metrics += ['edistance', 'jeffreys', 'mmd', 'ks_test', 't_test', 'wasserstein'] # slow
metrics += ['classifier_proba', 'classifier_cp', 'kendalltau_distance']  # newly added

print(f"running with test mode {test_mode}, dataset {args.dataset}, saving to {save_file}", flush=True)

### dataset-specific filtering ###
if args.dataset in ['sciplex_K562', 'sciplex_A549', 'sciplex_MCF7']:
    cell_line = args.dataset.split('_')[1]
    adata = pt.data.srivatsan_2020_sciplex3()

    if cell_line == 'A549':  # two doses in here, we only want the 24hr
        adata = adata[adata.obs.time == 24]
        
    adata = adata[adata.obs.cell_line == cell_line]
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
elif args.dataset == 'mcfarland':
    adata = pt.data.mcfarland_2020()

    # subset to common timepoints and most frequently occurring cell line - no better options
    adata = adata[adata.obs.time.isin(['6', '24']) & (adata.obs.cell_line == 'COLO680N')]
    adata.obs['perturbation'] = adata.obs['perturbation'].astype(str) + '_' + adata.obs.time.astype(str)
    adata.obs['perturbation'] = adata.obs.perturbation.replace({'control_24':'control', 'control_6':'control'})

    n_min_cells = 100
elif args.dataset == 'schiebinger':
    adata = pt.data.schiebinger_2019_18day()
    # take only the Dox and control conditions, representing full, "normal" reprogramming
    adata = adata[adata.obs.perturbation.isin(['control', 'Dox']) & (~adata.obs.age.isin(['iPSC', 'D0', 'D0.5']))]
    adata.obs['perturbation_old'] = adata.obs.perturbation
    adata.obs['perturbation'] = adata.obs.age.replace({'D1':'control'})

    n_min_cells = 400
elif args.dataset == 'garcia':
    adata = sc.read('./data/garcia2022.h5ad')

    adata = adata[adata.obs.cell_type == 'Ovarian interstitial cells']
    adata.obs['perturbation'] = adata.obs.age.replace({8.6:'control'}).astype(str)

    n_min_cells = 300
elif args.dataset == 'satinha':
    adata = sc.read('./data/SantinhaPlatt2023_GSE236519_pooled_screen_CBh_temp.h5ad')

    sc.pp.filter_genes(adata, min_cells=100)
    adata = adata[~adata.obs.per_gene.isnull()]
    included_cts = adata.obs.cell_types.value_counts()[adata.obs.cell_types.value_counts() > 1000].index
    adata = adata[adata.obs.cell_types.isin(included_cts)]
    adata.obs['perturbation'] = adata.obs.per_gene.replace({'Safe_H':'control'})

    n_min_cells = 480  # just enough to split control 5 ways
else:
    raise ValueError('must pass available dataset')

if 'control' not in adata.obs.perturbation.unique():
    raise ValueError('control must be a condition in `.obs.perturbation`')

if test_mode:
    print("Test mode: subsampling", flush=True)
    sc.pp.subsample(adata, .1)
    save_file = 'test.pkl'
    n_min_cells = int(n_min_cells/8)


### metric runs ###
scanpy_setup(adata)
adata.obs['ncounts'] = adata.X.A.sum(axis=1)

# set filtered adata used for all runs
merged = sample_and_merge_control_random(adata, 'control', n=5)
filtered = remove_groups(merged, min_cells=n_min_cells)

print(filtered, flush=True)
print("average number of counts per cell:", filtered.obs.ncounts.mean(), flush=True)
print("number of categories evaluated:", len(filtered.obs.perturbation.unique()), flush=True)

# load previous file if it exists
try:
    with open(f'{dss_path}/{save_file}.pkl', 'rb') as file:
        results = pickle.load(file)
    print('starting with', results.keys(), flush=True)
except:
    results = {}

if test_mode:
    results = {}

current_keys = set(results.keys())

for rep in ['lognorm', 'counts', 'pca']:

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
        subset = subsample(filtered, n_min_cells)[:, features]
        inplace_check(metrics, results, DistanceResult(subset, controls, str(n), rep, 'n_genes'))

    if eval_mode:  # we only need one of the metric runs to evaluate on a new dataset
        break

    ### n_cells ###
    print('running n_cells', flush=True)
    max_n_cells = n_min_cells if n_min_cells < 400 else 400
    experiment_condi = list(range(100, max_n_cells+10, 50)) + [max_n_cells]

    for ncell in experiment_condi:
        subset = subsample(filtered, ncell)[:, adata.var['highly_variable']]
        inplace_check(metrics, results, DistanceResult(subset, controls, str(ncell), rep, 'n_cells'))


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
        inplace_check(metrics, results, DistanceResult(subset, controls, "{:.3f}".format(count_mean), rep, 'count_mean'))


    ### n_DEGs ###
    if with_DEGs:
        print('running ndegs', flush=True)
        if rep == 'pca':
            raise ValueError('Cannot run n_degs eval with pca right now due to runtime constraints.')
        experiment_condi = [10, 20, 30, 40, 50, 100, 150, 200, 300, 400]

        # create new adata to calculate DEGs and has run-specific info
        filt_wctrl = ad.concat([filtered, adata[adata.obs.perturbation == 'control']])
        sc.tl.rank_genes_groups(
            filt_wctrl,
            groupby='perturbation',
            reference='control',
            rankby_abs=True
        )
        subset = subsample(filtered, n_min_cells)
        subset.uns = filt_wctrl.uns # using a new adata which does not have the 'control' condition

        for n in experiment_condi:
            subset.uns['n_genes'] = n
            inplace_check(metrics, results, DistanceResult(subset, controls, str(n), rep, 'n_DEGs'))

        break  # temporary for runtime, run only lognorm for everything
        
print('finished with', results.keys(), flush=True)
print('added keys:', set(results.keys())-current_keys, flush=True)
with open(f'{dss_path}/{save_file}.pkl', 'wb') as file:
    pickle.dump(results, file)
