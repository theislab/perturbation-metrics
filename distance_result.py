from utils import get_pwdf_per_condition
import pandas as pd


class DistanceResult():
    """DistanceResult object. Stores pairwise distance dataframes for various metrics.
    Pwdfs are concatenations of ctrl×perturbation dataframes from multiple subsets (along axis=0).
    Does not store adata to save memory; adata_list is passed at compute time."""
    
    def __init__(self, reference_groups, descriptor, rep_used, task) -> None:
        self.reference_groups = reference_groups
        self.descr = descriptor  # a value of n_cell, n_genes, n_DEGs, mislabel, libsize
        self.repr = rep_used
        if self.repr not in ['counts', 'lognorm', 'pca']:
            raise ValueError("Representations are currently limited to 'counts', 'lognorm', 'pca'")
        self.task = task  # 'n_cell', 'n_genes', 'n_DEGs',  'mislabel', or 'libsize'
        self.res_string = f"{self.descr}-{self.repr}-{self.task}"
        self.pwdfs = None
        self.metrics = None  # set
            
    def compute_pwdf(self, metrics, adata_list, recompute=False):
        """Compute pwdfs from a list of adatas, concatenating ctrl×perturbation dataframes along axis=0."""
        if adata_list is None and (self.pwdfs is None or recompute):
            raise ValueError("adata_list is required to compute pwdfs")
        if adata_list is not None and len(adata_list) == 0:
            raise ValueError("adata_list must not be empty")
        if self.pwdfs is None or recompute:
            pwdfs_per_subset = [
                get_pwdf_per_condition(adata, metrics, self.reference_groups, self.descr, self.repr)
                for adata in adata_list
            ]
            self.pwdfs = {}
            for key in pwdfs_per_subset[0]:
                self.pwdfs[key] = pd.concat([p[key] for p in pwdfs_per_subset], axis=0)
            self.metrics = metrics

        else: # only run metrics which have not already been computed
            metrics_for_recompute = list(set(metrics) - set(self.metrics))
            if metrics_for_recompute and adata_list is None:
                raise ValueError("adata_list is required to compute additional metrics")
            if metrics_for_recompute:
                pwdfs_per_subset = [
                    get_pwdf_per_condition(adata, metrics_for_recompute, self.reference_groups, self.descr, self.repr)
                    for adata in adata_list
                ]
                for key in pwdfs_per_subset[0]:
                    self.pwdfs[key] = pd.concat([p[key] for p in pwdfs_per_subset], axis=0)
                self.metrics = list(set(metrics) | set(self.metrics))
