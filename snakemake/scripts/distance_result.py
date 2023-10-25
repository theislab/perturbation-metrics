import sys
sys.path.append('../utils')
from utils import get_pwdf_per_condition
class DistanceResult():
    """DistanceResult object. Stores information about the adata used and
    the pairwise distance dataframes for various metrics."""
    
    def __init__(self, adata, reference_groups, descriptor, rep_used, task) -> None:
        self.adata = adata
        self.reference_groups = reference_groups
        self.descr = descriptor  # a value of n_cell, n_genes, n_DEGs, mislabel, libsize
        self.repr = rep_used
        if self.repr not in ['counts', 'lognorm', 'pca']:
            raise ValueError("Representations are currently limited to 'counts', 'lognorm', 'pca'")
        self.task = task # 'n_cell', 'n_genes', 'n_DEGs',  'mislabel', or 'libsize'
        
        self.res_string = f"{self.descr}-{self.repr}-{self.task}"
        self.pwdfs = None
        self.metrics = None  # set
            
    def compute_pwdf(self, metrics, recompute=False):
        if self.pwdfs is None or recompute:
            self.pwdfs = get_pwdf_per_condition(self.adata, metrics, self.reference_groups, self.descr, self.repr)
            self.metrics = metrics

        else: # only run metrics which have not already been computed
            metrics_for_recompute = list(set(metrics) - set(self.metrics))
            self.pwdfs.update(get_pwdf_per_condition(self.adata, metrics_for_recompute, self.reference_groups, self.descr, self.repr))
            self.metrics = list(set(metrics) | set(self.metrics))
