import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from utils import get_ranked_df_per_perturbation, calc_rank_percentile, get_melted_df_per_perturbation
from utils import get_flat_df, normalize_per_metric, plt_legend

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
        plt.ylabel('relative avg dist');
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

def evaluate_on_dataset(results, annotate_fn, ground_truth_label, optimal_distance='mean_absolute_error', rep='lognorm'):
    exp = 'n_genes'
    metrics = list(results.values())[0].metrics
    controls = ['control0', 'control1', 'control2', 'control3', 'control4']

    pwdfs, ctrl_ranks = plot(results, [exp, rep, '2000'], plot=False)
    df = get_melted_df_per_perturbation(pwdfs, metrics, controls, exp, reference=f'{optimal_distance}-2000')
    annotate_fn(df)
    df = df[df.is_control == 'perturbation']
    
    sr = {}
    for m in metrics:
        sub = df[df.metric == m]
        sr[m] = spearmanr(sub['distance'].values, sub[ground_truth_label].values)[0]
    corr_wreal = pd.DataFrame.from_dict(sr, orient='index').sort_values(0)
#    corr_wreal[0] = 1 - corr_wreal[0]  # flip so smaller is better
    corr_wreal.columns = [f'corr_{ground_truth_label}']

    # add in rank dataframe
    avg_rank, var_rank = perf_df(results, rep=rep)
    avg_rank = 1-avg_rank
    var_rank = 1-var_rank
    results = pd.concat([avg_rank, var_rank, corr_wreal], axis=1).sort_values(by=f'1-corr_{ground_truth_label}')

    # dataframe plot (not customizeable)
    plt.figure(figsize=(5, 5))
    sns.heatmap(results, annot=True, cmap='gist_heat', fmt=".3f", linewidths=.5, cbar_kws={'label': 'relative values'})
    plt.grid(None)
    plt.show()

    # lineplot (customizeable)
    normed_df = normalize_per_metric(df, label='distance')
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=normed_df, x='rank', y='distance', hue='metric', alpha=.5)
    sns.scatterplot(data=normed_df, x='rank', y='distance', hue=ground_truth_label, style='is_control')
    plt.legend(bbox_to_anchor=(1.01, 1.05))

    return results

def perf_df(results, rep='lognorm'):
    exp = 'n_genes'
    pwdfs, ctrl_ranks = plot(results, [exp, rep], plot=False)
    best_case = ctrl_ranks[ctrl_ranks[exp] == '2000']

    avg_rank = best_case[['rank', 'metric']].groupby('metric').mean().sort_values('rank')
    avg_rank.columns = ['sensitivity']

    var_rank = best_case[['rank', 'metric']].groupby('metric').var().sort_values('rank')
    var_rank.columns = ['robustness']
    
    return avg_rank, var_rank
