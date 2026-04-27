import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import seaborn as sns
from scipy.stats import spearmanr
from utils import get_ranked_df_per_perturbation, calc_rank_percentile, get_melted_df_per_perturbation
from utils import get_flat_df, normalize_per_metric, plt_legend


class _RankHueNorm(Normalize):
    """Map distinct hue values to equal steps in [0, 1] by sort order (default continuous colormap)."""

    def __init__(self, values):
        s = pd.Series(values).dropna()
        u = np.sort(s.unique()) if pd.api.types.is_numeric_dtype(s) else sorted(s.unique(), key=str)
        self._map = {v: i / max(len(u) - 1, 1) for i, v in enumerate(u)}
        super().__init__(vmin=0, vmax=1, clip=False)

    def __call__(self, value, clip=None):
        arr = np.asarray(value)
        out = pd.Series(arr.ravel()).map(self._map).astype(float).values.reshape(arr.shape)
        return np.ma.masked_invalid(out)


def plot(results, tags, plot=True):
    metrics = list(results.values())[0].metrics
    controls = ['control0', 'control1', 'control2', 'control3', 'control4']
    label_tag = tags[0]
    
    pwdfs = {}
    for k, res in results.items():
        if all(t in k for t in tags):
            pwdfs.update(res.pwdfs)
    
    if len(pwdfs) < 2:
        raise ValueError(f'Conditions {tags} were not run.')

    individually_ranked = get_ranked_df_per_perturbation(pwdfs, metrics, controls, label_tag)
    ctrl_ranks = calc_rank_percentile(individually_ranked, controls)

    if plot:
        melted_df = get_flat_df(pwdfs, controls, label=label_tag)
        plt.figure(figsize=(10, 5))
        sns.lineplot(data=normalize_per_metric(melted_df), x=label_tag, y='avg_dist', hue='metric')
        plt.ylabel('relative avg dist')
        if 'n_genes' in label_tag:
            plt.xscale('log')
        plt_legend()
        plt.title(f'distance behavior w.r.t {label_tag} in {tags[1]} space')
        plt.show()

        ctrl_ranks[label_tag] = ctrl_ranks[label_tag].astype(float)
        plt.figure(figsize=(10, 5))
        sns.lineplot(data=ctrl_ranks, x=label_tag, y='rank', hue='metric')
        plt.ylabel('control rank percentile')
        plt.ylim(-.05, 1)
        if 'n_genes' in label_tag:
            plt.xscale('log')
        plt_legend()
        plt.title(f'sensitivity w.r.t. {label_tag} in {tags[1]} space')
        plt.show()
    
    return pwdfs, ctrl_ranks

def evaluate_on_dataset(
    results,
    annotate_fn,
    ground_truth_label,
    optimal_distance='mean_absolute_error',
    rep='lognorm',
    exp='n_genes',
    numeric_value='2000',
    plot_performance=True
    ):
    """
    Computes evaluation metrics for a given dataset and experimental settings.

    Note that rep, exp and numeric_value must exist in the key string of the results dictionary.

    Parameters
    ----------
    results : dict
        Dictionary of DistanceResult objects.
    annotate_fn : function
        Function to annotate the dataframe with additional information.
    ground_truth_label : str
        Label of the ground truth metric.
    optimal_distance : str
        Name of a distance to use as reference for plotting the lineplot. 
        Choose the best performing distance for clearest visuals.
    rep : str
        Representation of the data, one of 'lognorm', 'counts', 'pca'.
    exp : str
        Experimental setting which was varied, one of 'n_genes', 'n_cells', 'libsize'.
    numeric_value : str
        Numeric value of the experimental setting. 
    """
    metrics = list(results.values())[0].metrics
    controls = ['control0', 'control1', 'control2', 'control3', 'control4']

    pwdfs, ctrl_ranks = plot(results, [exp, rep, numeric_value], plot=False)
    df = get_melted_df_per_perturbation(pwdfs, metrics, controls, exp, reference=f'{optimal_distance}-{numeric_value}')
    annotate_fn(df)
    df = df[df.is_control == 'perturbation']
    
    sr = {}
    for m in metrics:
        sub = df[df.metric == m]
        sr[m] = spearmanr(sub['distance'].values, sub[ground_truth_label].values)[0]
    corr_wreal = pd.DataFrame.from_dict(sr, orient='index').sort_values(0)
#    corr_wreal[0] = 1 - corr_wreal[0]  # flip so smaller is better
    corr_wreal.columns = [f'corr_{ground_truth_label}']

    # add in rank dataframe (must use same exp/numeric_value for sensitivity/robustness)
    avg_rank, var_rank = perf_df(results, rep=rep, exp=exp, numeric_value=numeric_value)
    avg_rank = 1-avg_rank
    var_rank = 1-var_rank
    results = pd.concat([avg_rank, var_rank, corr_wreal], axis=1).sort_values(by=f'corr_{ground_truth_label}')

    if plot_performance:
        # dataframe plot (not customizeable)
        plt.figure(figsize=(5, 5))
        sns.heatmap(results, annot=True, cmap='gist_heat', fmt=".3f", linewidths=.5, cbar_kws={'label': 'relative values'})
        plt.grid(None)
        plt.show()

        # lineplot (customizeable)
        normed_df = normalize_per_metric(df, label='distance')
        plt.figure(figsize=(10, 5))
        sns.lineplot(data=normed_df, x='rank', y='distance', hue='metric', alpha=.5)
        sns.scatterplot(
            data=normed_df,
            x='rank',
            y='distance',
            hue=ground_truth_label,
            style='is_control',
            hue_norm=_RankHueNorm(normed_df[ground_truth_label]),
        )
        plt.legend(bbox_to_anchor=(1.01, 1.05))

    return results

def perf_df(results, rep='lognorm', exp='n_genes', numeric_value='2000'):
    pwdfs, ctrl_ranks = plot(results, [exp, rep, str(numeric_value)], plot=False)
    best_case = ctrl_ranks[ctrl_ranks[exp].astype(str) == str(numeric_value)]

    avg_rank = best_case[['rank', 'metric']].groupby('metric').mean().sort_values('rank')
    avg_rank.columns = ['sensitivity']

    var_rank = best_case[['rank', 'metric']].groupby('metric').var().sort_values('rank')
    var_rank.columns = ['robustness']
    
    return avg_rank, var_rank

def identify_pmax_by_marginal_gain(
    df,
    perf_col="rank",
    metric_col="metric",
    n_col="n_cells",
    marginal_frac=0.05,
    require_consecutive=2,
    threshold_mode="relative",   # "relative" or "absolute"
    perf_range=1.0,              # used when threshold_mode="absolute"
):
    """
    Identify the n at which the evaluation measure plateaus. Assumes lower scores are better.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns dataset, n_cells or whatever varies, metric, and evaluation score.
    perf_col : str
        Column name of the evaluation score.
    metric_col : str
    n_col : str
        Column name of the variable that varies.
    marginal_frac : float
        Fraction of the total observed gain which can be tolerated to consider the performace as plateued.
        If threshold_mode is "relative", this is the fraction of the total observed gain.
        If threshold_mode is "absolute", this is the fraction of the perf_range.
    require_consecutive : int
        Number of consecutive points required to identify a plateau.
    threshold_mode : str
        See marginal_frac.
    perf_range : float
        Range of the evaluation measure, if threshold_mode="absolute"
    """
    df = df.copy()
    df[n_col] = pd.to_numeric(df[n_col], errors="coerce")
    df[perf_col] = pd.to_numeric(df[perf_col], errors="coerce")
    df = df.dropna(subset=[metric_col, n_col, perf_col])

    curve_df = (
        df.groupby([metric_col, n_col], as_index=False)[perf_col]
          .mean()
          .sort_values([metric_col, n_col])
          .reset_index(drop=True)
    )

    all_curves = []
    summaries = []

    for metric, g in curve_df.groupby(metric_col, sort=False):
        g = g.sort_values(n_col).copy()

        # lower is better
        g["delta_perf"] = g[perf_col].shift(1) - g[perf_col]
        g["delta_n"] = g[n_col] - g[n_col].shift(1)
    
        total_observed_gain = g[perf_col].iloc[0] - g[perf_col].iloc[-1]

        if threshold_mode == "relative":
            marginal_gain_threshold = marginal_frac * total_observed_gain
        elif threshold_mode == "absolute":
            marginal_gain_threshold = marginal_frac * perf_range
        else:
            raise ValueError("threshold_mode must be 'relative' or 'absolute'")

        overall_gets_worse = total_observed_gain <= 0

        if overall_gets_worse:
            g["is_marginal"] = False
            plateau_found = False
            plateau_start_n = np.nan
            pmax = np.nan
            plateau_n_values = []
        else:
            g["is_marginal"] = g["delta_perf"] <= marginal_gain_threshold

            plateau_start_idx = None
            vals = g["is_marginal"].fillna(False).to_numpy()

            for i in range(1, len(g)):
                end = i + require_consecutive
                if end <= len(g) and vals[i:end].all():
                    plateau_start_idx = i
                    break

            if plateau_start_idx is None:
                plateau_found = False
                plateau_start_n = np.nan
                pmax = np.nan
                plateau_n_values = []
            else:
                plateau_found = True
                plateau_start_n = g.iloc[plateau_start_idx][n_col]
                plateau_region = g.loc[g[n_col] >= plateau_start_n, perf_col]
                pmax = plateau_region.mean()
                plateau_n_values = g.loc[g[n_col] >= plateau_start_n, n_col].tolist()

        g["plateau_found"] = plateau_found
        g["plateau_start_n"] = plateau_start_n
        g["pmax"] = pmax
        g["overall_gets_worse"] = overall_gets_worse
        g["threshold_mode"] = threshold_mode
        g["marginal_gain_threshold"] = marginal_gain_threshold

        all_curves.append(g)

        summaries.append({
            metric_col: metric,
            "plateau_found": plateau_found,
            "overall_gets_worse": overall_gets_worse,
            "plateau_start_n": plateau_start_n,
            "pmax": pmax,
            "total_observed_gain": total_observed_gain,
            "threshold_mode": threshold_mode,
            "marginal_gain_threshold": marginal_gain_threshold,
            "plateau_n_values": plateau_n_values,
        })

    summary_df = pd.DataFrame(summaries)
    curve_df = pd.concat(all_curves, ignore_index=True)
    return summary_df, curve_df