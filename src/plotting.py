import os
import pickle
import pandas as pd
import seaborn as sns
import re
import matplotlib.pyplot as plt


def plot_histories_cnn(PATH, plots_dir, file_best_exec):

    history_file = file_best_exec + "_history.pickle"

    with open(os.path.join(PATH, history_file), 'rb') as f:
        history = pickle.load(f)

    hist_df = pd.DataFrame(history)
    hist_df["Execution"] = range(len(hist_df))

    hist_df_melt = pd.melt(hist_df, id_vars=['Execution'], value_vars=['loss', 'val_loss'])
    hist_df_melt = hist_df_melt.rename(columns={"variable": "Metric", "value": "Loss"})

    sns.set_style("whitegrid")
    ax4 = sns.lineplot(x="Execution", y="Loss", hue="Metric", data=hist_df_melt, palette="Blues_d")
    ax4.legend(loc=1)
    ax4.figure.savefig(plots_dir + "best_cnn_epochs_loss.pdf", dpi=300)

    all_history_files = [each for each in os.listdir(PATH) if each.endswith('history.pickle')]

    history_files_dfs = []

    for history_file in all_history_files:
        with open(os.path.join(PATH,history_file), 'rb') as f:
            history = pickle.load(f)
            hist_df = pd.DataFrame(history) 
            hist_df["Execution"] = range(len(hist_df))
            hist_df["Model"] = re.search('[0-9]_+(.+?)_history.pickle', history_file).group(1).split("_")[0]
            history_files_dfs.append(hist_df)


    dfs_histories = pd.concat((history_files_dfs))
    dfs_histories_melt = pd.melt(dfs_histories, id_vars=['Execution', 'Model'], value_vars=['loss', 'val_loss'])
    dfs_histories_melt = dfs_histories_melt.rename(columns={"variable": "Metric", "value": "Loss"})

#     display(dfs_histories_melt)

    plt.clf()
    plt.figure(figsize=(9,6))
    sns.set_style("whitegrid")
    ax5 = sns.lineplot(x="Execution", y="Loss", hue="Model", data=dfs_histories_melt, palette="Blues_d")
    ax5.set(xlabel='Epoch', ylabel='Loss')
    ax5.legend(loc=1)
    ax5.figure.savefig(plots_dir + "cnns_epochs_loss_comparison.pdf", dpi=300, bbox_inches='tight',pad_inches=0.25)
    
    
def generate_bar_plot(data, x, y, hue, path, file_name, xlabel=None, ylabel=None):

    sns.set_style("whitegrid")
    ax2 = sns.barplot(x=x, y=y, hue=hue, data=data, palette="Blues_d")
    ax2.set(xlabel=xlabel, ylabel=ylabel)
    ax2.legend(loc=4)
    ax2.figure.savefig(os.path.join(path, file_name), dpi=300)

    plt.clf()