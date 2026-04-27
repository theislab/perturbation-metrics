import sys
import re
import pickle
import argparse
import ast
import numpy as np
import anndata as ad
import scanpy as sc
import pertpy as pt

from distance_result import DistanceResult
from utils import scanpy_setup
from utils import sample_and_merge_control_random, remove_groups, subsample, generate_sparsity
from utils import inplace_check

parser = argparse.ArgumentParser()

parser.add_argument("--save_file", type=str, default='test.pkl', required=True)
parser.add_argument("--dataset", type=str, default='', required=True)
parser.add_argument("--dir", type=str, default='', required=False)
parser.add_argument("--test_mode", dest='test_mode', default=False, action='store_true')  # evaluate on subset
parser.add_argument("--eval_mode", dest='eval_mode', default=False, action='store_true')  # run bare min needed for table
parser.add_argument("--with_DEGs", dest='with_DEGs', default=False, action='store_true')  # add DEGs on lognorm to run
parser.add_argument("--signal_injection", dest='signal_injection', default=False, action='store_true')  # add signal injection to run
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--n_random_states", type=int, default=5, help="Number of times to generate population subsamplings.")

### Note that the default mode, without any flags, runs hvgs, ncells, and
### libsize across lognorm, counts, and pca representations.
### n_min_cells decides the perturbations which are retained

args = parser.parse_args()
test_mode = args.test_mode
eval_mode = args.eval_mode
with_DEGs = args.with_DEGs
signal_injection = args.signal_injection
save_file = args.save_file
if eval_mode: save_file += '_sub'

dir = args.dir
dss_path = f'/lustre/scratch/users/yuge.ji/metrics_revisions/{dir}'

metrics = ['euclidean', 'spearman_distance', 'mean_absolute_error']  # representative
metrics += ['r2_distance', 'pearson_distance', 'mse', 'cosine_distance']  # fast
metrics += ['edistance', 'sym_kldiv', 'mmd', 'mmd_rbf', 'ks_test', 't_test', 'wasserstein'] # slow
metrics += ['classifier_proba', 'classifier_cp', 'kendalltau_distance']  # newly added

print(f"running with test mode {test_mode}, dataset {args.dataset}, saving to {save_file} at {dss_path}", flush=True)

### dataset-specific filtering ###
if args.dataset in [
    'sciplex_K562',
    'sciplex_A549',
    'sciplex_MCF7',
    'sciplex_MCF7-batch-replicate-rep1',
    'sciplex_MCF7-batch-replicate-rep2',
    'sciplex_A549-batch-replicate-rep1',
    'sciplex_A549-batch-replicate-rep2',
    'sciplex_K562-batch-replicate-rep1',
    'sciplex_K562-batch-replicate-rep2'
]:
    cell_line = re.split(r'[_-]', args.dataset)[1]
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
elif args.dataset == 'simulated':
    adata = sc.read('./data/splatter_sim.h5ad')
    
    n_min_cells = 600
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

    n_min_cells = 100  # just enough to split control 5 ways (541 cells)
elif args.dataset in ['schiebinger', 'schiebinger-batch-replicate-1', 'schiebinger-batch-replicate-2']:
    adata = pt.data.schiebinger_2019_18day()
    adata.obs['replicate'] = adata.obs['replicate'].astype(str)

    # take only the Dox and control conditions, representing full, "normal" reprogramming
    adata = adata[adata.obs.perturbation.isin(['control', 'Dox']) & (~adata.obs.age.isin(['iPSC', 'D0', 'D0.5']))]
    adata.obs['perturbation_old'] = adata.obs.perturbation
    adata.obs['perturbation'] = adata.obs.age.replace({'D1':'control'})

    n_min_cells = 400
elif args.dataset == 'garcia':
    adata = sc.read('./data/garcia2022.h5ad')

    adata = adata[adata.obs.cell_type == 'Ovarian interstitial cells']
    adata.obs['perturbation'] = adata.obs.age.replace({8.6:'control'}).astype(str)

    n_min_cells = 300  # just enough to split control 5 ways (1784 cells)
elif args.dataset == 'satinha':
    adata = sc.read('./data/SantinhaPlatt2023_GSE236519_pooled_screen_CBh_temp.h5ad')

    sc.pp.filter_genes(adata, min_cells=100)
    adata = adata[~adata.obs.per_gene.isnull()]
    included_cts = adata.obs.cell_types.value_counts()[adata.obs.cell_types.value_counts() > 1000].index
    adata = adata[adata.obs.cell_types.isin(included_cts)]
    adata.obs['perturbation'] = adata.obs.per_gene.replace({'Safe_H':'control'})

    n_min_cells = 480  # just enough to split control 5 ways
elif args.dataset in [
    'tahoe_A549',
    'tahoe_HT29',
    'tahoe_BT474',
    # using the most frequently occurring plates here
    'tahoe_A549-batch-plate-plate_12',
    'tahoe_A549-batch-plate-plate_2',
    'tahoe_HT29-batch-plate-plate_12',
    'tahoe_HT29-batch-plate-plate_2',
    'tahoe_BT474-batch-plate-plate_12',
    'tahoe_BT474-batch-plate-plate_6'
]:
    cell_line = re.split(r'[_-]', args.dataset)[1]
    adata = sc.read(f'./data/Tahoe_{cell_line}_only_full_genes.h5ad')
    adata.X = adata.layers['counts']
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # separate out the perturbations by dose
    adata.obs['dose (uM)'] = adata.obs['drugname_drugconc'].apply(lambda x: ast.literal_eval(x)[0][1])
    adata.obs['perturbation'] = adata.obs.drug.astype(str) + '_' + adata.obs['dose (uM)'].astype(str)
    adata.obs['perturbation'] = adata.obs.perturbation.replace({'DMSO_TF_0.0':'control'})

    n_min_cells = 300
elif args.dataset == 'saunders':
    # added for revisions
    adata = sc.read('./data/saunders_spatial_2025.h5ad')
    adata.obs['perturbation'] = adata.obs.singlet_gene

    n_min_cells = 400

    # individually run because it only has 209 genes so the normal pipeline doesn't work
    n_controls = 5
    merged = sample_and_merge_control_random(adata, 'control', n=n_controls)
    controls = [f'control{i}' for i in range(n_controls)]
    filtered = remove_groups(merged, min_cells=n_min_cells)
    random_states = [args.seed + i for i in range(args.n_random_states)]
        
    results = {}
    subset_list = [subsample(filtered, n_min_cells, random_state=rs) for rs in random_states]
    inplace_check(metrics, results, DistanceResult(controls, str(filtered.shape[1]), 'lognorm', 'n_genes'), adata_list=subset_list)
    print('finished with', results.keys(), flush=True)
    with open(f'{dss_path}/{save_file}.pkl', 'wb') as file:
        pickle.dump(results, file)

    sys.exit(0)
else:
    raise ValueError('must pass available dataset')

if 'batch' in args.dataset:
    groupkey = args.dataset.split('-')[-2]
    groupname = args.dataset.split('-')[-1]
    adata = adata[adata.obs[groupkey] == groupname]
    
    if 'tahoe' in args.dataset or 'schiebinger' in args.dataset:  # not as many cells as sciplex but I still want to make it work
        n_min_cells = int(adata[adata.obs.perturbation == 'control'].shape[0]/5 -1)
    print(f'Assessing batch effects by running {groupname} in {groupkey}, mincells reduced to {n_min_cells}', flush=True)
    if n_min_cells < 100:
        raise ValueError('Not enough control cells left for batch effect assessment')

if 'control' not in adata.obs.perturbation.unique():
    raise ValueError('control must be a condition in `.obs.perturbation`')

if test_mode:
    print("Test mode: subsampling", flush=True)
    sc.pp.subsample(adata, .1)
    save_file = 'test.pkl'
    n_min_cells = int(n_min_cells/8)

### metric runs ###
scanpy_setup(adata)
try:
    adata.obs['ncounts'] = adata.X.A.sum(axis=1)
except AttributeError:  # handle non-matrices
    adata.obs['ncounts'] = adata.X.sum(axis=1)

# set filtered adata used for all runs
# note this adjusting the number of control subsamples is hardcoded because
# adjusting you would also need to adjust min_cells per dataset above.
n_controls = 5
merged = sample_and_merge_control_random(adata, 'control', n=n_controls)
controls = [f'control{i}' for i in range(n_controls)]
filtered = remove_groups(merged, min_cells=n_min_cells)

print(filtered, flush=True)
print("average number of counts per cell:", filtered.obs.ncounts.mean(), flush=True)
print("number of categories evaluated:", len(filtered.obs.perturbation.unique()), flush=True)

random_states = [args.seed + i for i in range(args.n_random_states)]
print(f"Using {args.n_random_states} random state(s) for subsampling: {random_states}", flush=True)

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

    feature_subsets = {}
    for n in experiment_condi:
        sc.pp.highly_variable_genes(adata, n_top_genes=n, flavor='seurat')
        feature_subsets[n] = list(adata.var_names[adata.var['highly_variable']])
    # reset highly_variable genes for the remaining experiments
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)

    for n, features in feature_subsets.items():
        subset_list = [subsample(filtered, n_min_cells, random_state=rs)[:, features] for rs in random_states]
        inplace_check(metrics, results, DistanceResult(controls, str(n), rep, 'n_genes'), adata_list=subset_list)

    ### n_DEGs ###
    if with_DEGs:
        print('running ndegs', flush=True)
        if rep == 'pca':
            print('Warning: Skipping n_degs eval with pca due to runtime constraints.', flush=True)
            continue
        experiment_condi = [10, 20, 30, 40, 50, 100, 150, 200, 300, 400]

        # create new adata to calculate DEGs and has run-specific info
        filt_wctrl = ad.concat([filtered, adata[adata.obs.perturbation == 'control']])
        sc.tl.rank_genes_groups(
            filt_wctrl,
            groupby='perturbation',
            reference='control',
            rankby_abs=True
        )
        for n in experiment_condi:
            subset_list = []
            for rs in random_states:
                subset = subsample(filtered, n_min_cells, random_state=rs)
                subset.uns = filt_wctrl.uns  # using a new adata which does not have the 'control' condition
                subset.uns['n_genes'] = n
                subset_list.append(subset)
            inplace_check(metrics, results, DistanceResult(controls, str(n), rep, 'n_DEGs'), adata_list=subset_list)

    ### signal injection ###
    # Warning: only usable with datasets where there is an excess of control cells!
    if signal_injection and rep != 'pca':
        print('running signal injection', flush=True)
        experiment_condi = [50, 100, 500, 1000]

        # how many perturbations to spike in, limited for runtime
        n_perts = 10
        pert_options = [p for p in filtered.obs.perturbation.unique() if p not in controls and p != "control"]
        selected_perts = np.random.choice(pert_options, size=n_perts, replace=False)

        # "spike in" differentially expressed genes into the control condition by taking it from the
        # perturbed condition. The control condition to be spiked into is chosen from among the cells
        # already used for control.
        merged = sample_and_merge_control_random(adata, 'control', n=n_controls+n_perts)
        used_ctrl_cells = merged.obs_names[merged.obs.perturbation.isin(controls)]
        unused_control_adata = adata[
            (adata.obs.perturbation == "control") & ~adata.obs_names.isin(used_ctrl_cells)
        ]
        if unused_control_adata.shape[0] < n_perts*n_min_cells:
            raise ValueError(f"Not enough control cells left for spike-in, needed:", n_perts*n_min_cells, "had:", unused_control_adata.shape[0])

        subset_list = []
        filtered_list = []  # we also need to randomly sample the spiked-in cells
        for rs in random_states:
            # split control n_perts ways and grab them
            spikein_ctrls = sample_and_merge_control_random(unused_control_adata, 'control', n=n_perts)  # automatically shuffles per run
            spikein_ctrls = spikein_ctrls[spikein_ctrls.obs.perturbation.isin([f'control{n}' for n in range(n_perts)])]
            # rename the perturbations so downstream code doesn't get confused
            spikein_ctrls.obs['perturbation'] = spikein_ctrls.obs['perturbation'].map({f'control{i}': f'spikedctrl{i}' for i in range(n_perts)})
            # sample both real controls and spike in controls down to n_min_cells and concat the real control cells back in
            real_controls = filtered[filtered.obs.perturbation.isin(controls)]
            real_controls = subsample(real_controls, n_min_cells, random_state=rs)
            subset_list.append(ad.concat([
                real_controls,
                subsample(spikein_ctrls, n_min_cells, random_state=rs)
                ]))
            filtered_list.append(subsample(filtered, n_min_cells, random_state=rs))
        for n_genes in experiment_condi:
            print(f'running signal injection with experiment_condi:{n_genes} genes', flush=True)
            spike_in_genes = np.random.choice(list(adata.var_names[adata.var["highly_variable"]]), size=n_genes, replace=False)


            # inject the perturbations into the control conditions
            for j, a in enumerate(subset_list):
                    a[a.obs.perturbation == f'spikedctrl{i}'][:, spike_in_genes].X = filtered_list[j][filtered_list[j].obs.perturbation == p][:, spike_in_genes].X.copy()

            inplace_check(metrics, results, DistanceResult(controls, str(n_genes), rep, 'n_signal'), adata_list=subset_list)
        break  # never run anything except signal injection when it's true because it redefines `merged` and also takes too long

    if eval_mode:  # we only need a minimalist HVG run for the main evaluation
        break

    ### n_cells ###
    max_n_cells = n_min_cells if n_min_cells < 600 else 600
    experiment_condi = list(range(100, max_n_cells+10, 50)) + [max_n_cells]
    ## uncomment for a version with tiny cell counts only
    # experiment_condi = [10, 20, 30, 50, 70, 100]
    print('running n_cells with', experiment_condi, flush=True)

    for ncell in experiment_condi:
        subset_list = [subsample(filtered, ncell, random_state=rs)[:, adata.var['highly_variable']] for rs in random_states]
        inplace_check(metrics, results, DistanceResult(controls, str(ncell), rep, 'n_cells'), adata_list=subset_list)


    ### libsize ###
    print('running libsize', flush=True)
    experiment_condi = list(range(10, 91, 10))

    for perc in experiment_condi:
        subset_list = [
            generate_sparsity(
                adata[:, adata.var['highly_variable']],
                subsample(filtered, n_min_cells, random_state=rs).obs,
                perc
            )
            for rs in random_states
        ]
        count_mean = subset_list[0].layers['counts'].mean()
        inplace_check(metrics, results, DistanceResult(controls, "{:.3f}".format(count_mean), rep, 'count_mean'), adata_list=subset_list)
        
print('finished with', results.keys(), flush=True)
print('added keys:', set(results.keys())-current_keys, flush=True)
with open(f'{dss_path}/{save_file}.pkl', 'wb') as file:
    pickle.dump(results, file)
