import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import os
import seaborn as sns
from sklearn.decomposition import PCA
import math
from sklearn.preprocessing import StandardScaler
from scipy.sparse import issparse

def correlation_barplot(all_corr, models, ood_primaries, data, save_folder, xlabel, ylabel, log_scale=False):
    evaluation_metric = save_folder.split('/')[-1]
    setting_name = save_folder.split('/')[-2]
    # Number of OOD primaries and models
    n_ood = len(ood_primaries)
    n_models = len(models)
    total = n_ood * n_models

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(15, 6))

    # X positions for each group of bars (one group per ood_primary)
    x = np.arange(n_ood)

    # Width of the bars
    bar_width = 0.12

    # Offset for each model within a group
    offsets = np.arange(-(n_models - 1) / 2, (n_models + 1) / 2) * bar_width

    # Plotting each model's correlations
    for i, model in enumerate(models):
        # model_corr = all_corr[i * n_ood:(i + 1) * n_ood]  # Extract the correlation values for this model
        indices = np.arange(i,total,n_models)
        model_corr = [all_corr[i] for i in indices]
        ax.bar(x + offsets[i], model_corr, bar_width, label=model)
        

    # Setting labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    # ax.set_title(f'Spearman Correlation for {data} - setting {setting_name}')
    ax.set_title(f"({data} - {evaluation_metric})- setting {setting_name}")
    ax.set_xticks(x)
    ax.set_xticklabels(ood_primaries)
    ax.legend(title='Models')
    if log_scale:
        ax.set_yscale('log')

    # Save the figure
    plot_save_path = os.path.join(save_folder, f'{data}_{evaluation_metric}.png')
    plt.savefig(plot_save_path, dpi=300)

    # Show the plot
    plt.show()

    # Close the plot
    plt.close(fig)
    
# def true_pos_lineplot(results, ood_primary, deg_type, data, save_folder, xlabel, ylabel, legend_title='Method', y_min=None, y_max=None):
#     setting_name = save_folder.split('/')[-2]
#     models = results.keys()

#     plt.figure(figsize=(10, 6))
#     model_colors = plt.cm.get_cmap('tab10', len(models)).colors

#     min_global = y_min
#     max_global = y_max

#     for j, model in enumerate(models):
#         if j>=(len(models)/2):
#             style = 'dashed'
#         else:
#             style = 'solid'
#         plt.plot(results[model]['n'], results[model]['true_positives'], 
#                  label=model, color=model_colors[j], linewidth=1.5, linestyle=style)
#         if y_min is not None and results[model]['true_positives'].min() < min_global:
#             min_global = results[model]['true_positives'].min()
#         if y_max is not None and results[model]['true_positives'].max() > max_global:
#             max_global = results[model]['true_positives'].max()


#     plt.title(f"{ood_primary} ({data} - {deg_type})- setting {setting_name}")
#     plt.xlabel(xlabel)
#     plt.ylabel(ylabel)
#     plt.legend(title=legend_title)
#     # plt.grid(False)
#     plt.grid(alpha=0.3)
#     if y_min is not None and y_max is not None:
#         plt.ylim(min_global, max_global)

#     # Save the plot
#     plot_filename = f"{data}_{ood_primary}_{deg_type.split('_')[0]}.png"
#     plt.savefig(os.path.join(save_folder, plot_filename), dpi=300)
#     plt.close()

def true_pos_lineplot(results, ood_primary, deg_type, data, save_folder, xlabel, ylabel, legend_title='Method', y_min=None, y_max=None, title=None):
    # Prepare data for seaborn
    df_list = []
    for model, vals in results.items():
        temp_df = pd.DataFrame({
            xlabel: vals['n'],
            ylabel: vals['true_positives'],
            'Model': model
        })
        df_list.append(temp_df)
    df = pd.concat(df_list, ignore_index=True)

    # --- Aesthetics ---
    sns.set_theme(style="whitegrid", context="talk")
    palette = sns.color_palette("tab10", len(results))

    plt.figure(figsize=(6, 6))

    # --- Main Plot ---
    sns.lineplot(
        data=df,
        x=xlabel,
        y=ylabel,
        hue='Model',
        # style='Model',
        palette=palette,
        linewidth=3,
        alpha=0.9,
        # markers=True,
        # dashes=False
    )

    # --- Formatting ---
    if title is None:
        plt.title(f"{ood_primary} ({data} - {deg_type})\nSetting: {save_folder.split('/')[-2]}",
        # plt.title(f"{ood_primary} ({data} - {deg_type})",
                fontsize=18, fontweight='semibold', pad=15)
    else:
        # plt.title(f"{ood_primary} ({data} - {deg_type})\n{title}", fontsize=14, pad=15)
        plt.title(title, fontsize=18, pad=15)
    plt.xlabel(xlabel, fontsize=18)
    plt.ylabel(ylabel, fontsize=18)

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # Remove top/right spines for cleaner appearance
    sns.despine()

    # Light grid on both axes
    plt.grid(axis='both', alpha=0.3, linestyle='--')

    # Legend styling
    plt.legend(
        title=legend_title,
        title_fontsize=12,
        fontsize=11,
        frameon=True,
        edgecolor='gray',
        fancybox=True,
        loc='best'
    )

    # Apply y-limits if provided
    if y_min is not None and y_max is not None:
        plt.ylim(y_min, y_max)

    # Tight layout and save
    plt.tight_layout()
    plot_filename = f"{data}_{ood_primary}_{deg_type.split('_')[0]}.png"
    plt.savefig(os.path.join(save_folder, plot_filename), dpi=600, bbox_inches='tight')
    plot_filename = f"{data}_{ood_primary}_{deg_type.split('_')[0]}.pdf"
    plt.savefig(os.path.join(save_folder, plot_filename), dpi=600, bbox_inches='tight')
    plt.close()

def draw_boxplot(data_dict, dataset, save_folder, file_name, title=None, yscale=False):
    setting_name = save_folder.split('/')[-2]
    labels, data = data_dict.keys(), data_dict.values()

    # colors = plt.cm.tab10(np.linspace(0, 1, len(labels)))
    pastel_colors = sns.color_palette("pastel", len(labels))

    fig, ax = plt.subplots(figsize=(10, 6))

    box = ax.boxplot(data, patch_artist=True, labels=labels)
    for patch, color in zip(box['boxes'], pastel_colors):
        patch.set_facecolor(color)
    
    plt.xticks(range(1, len(labels) + 1), labels, rotation=-15, fontsize=7)
    if yscale:
        plt.yscale('log')
    if title is None:
        ax.set_title(f"{file_name} - setting {setting_name}")
    else:
        ax.set_title(f"{file_name} - {title}")

    plot_save_path = os.path.join(save_folder, f'{dataset}_{file_name}.png')
    plt.savefig(plot_save_path, dpi=300)
    plt.close()


def draw_boxplot2(data_dict, dataset, save_folder, file_name, title=None):
    setting_name = save_folder.split('/')[-2]
    labels, data = data_dict.keys(), data_dict.values()

    # colors = plt.cm.tab10(np.linspace(0, 1, len(labels)))
    pastel_colors = sns.color_palette("pastel", len(labels))

    fig, ax = plt.subplots(figsize=(8, 6))

    box = ax.boxplot(data, patch_artist=True, labels=labels)
    for patch, color in zip(box['boxes'], pastel_colors):
        patch.set_facecolor(color)
    
    plt.xticks(range(1, len(labels) + 1), labels, rotation=-15, fontsize=15)
    ax.set_ylabel(f'{file_name}', fontsize=18)
    if title is None:
        ax.set_title(f"{dataset} - {setting_name.upper()} - {file_name}", fontsize=20)
    else:
        ax.set_title(f"{dataset} - {setting_name.upper()} - {title}", fontsize=20)

    plot_save_path = os.path.join(save_folder, f'{dataset}_{file_name}.png')
    plt.savefig(plot_save_path, dpi=300)
    plt.close()


def draw_boxplot_multiple(data_dict_list, dataset, save_folder, file_name, subtitles, title=None, x_label=None, y_label=None):
    setting_name = save_folder.split('/')[-2]

    rows = len(data_dict_list)
    fig, axes = plt.subplots(rows, 1, figsize=(8, 4* rows))
    for i, data_dict in enumerate(data_dict_list):
        ax = axes[i]
        labels, data = data_dict.keys(), data_dict.values()
        pastel_colors = sns.color_palette("pastel", len(labels))

        box = ax.boxplot(data, patch_artist=True, labels=labels)
        for patch, color in zip(box['boxes'], pastel_colors):
            patch.set_facecolor(color)
        
        ax.set_xticks(range(1, len(labels) + 1), labels, rotation=-15, fontsize=7)
        ax.set_title(subtitles[i])
        if x_label is not None:
            ax.set_xlabel(x_label)
        if y_label is not None:
            ax.set_ylabel(y_label)
        
    if title is None:
        plt.suptitle(f"{file_name} - setting {setting_name}")
    else:
        plt.suptitle(f"{file_name} - {title}")

    plot_save_path = os.path.join(save_folder, f'{dataset}_{file_name}.png')
    plt.savefig(plot_save_path, dpi=300)
    plt.close()

def top_k_plot(results_gsea, models, ood_primary, data, save_folder):
    setting_name = save_folder.split('/')[-2]

    plt.figure(figsize=(10, 6))
    model_colors = plt.cm.get_cmap('tab10', len(models)).colors

    for j, model in enumerate(models):
        if j>=(len(models)/2):
            style = 'dashed'
        else:
            style = 'solid'
        plt.plot(results_gsea[model]['n'], results_gsea[model]['top_n'], 
                 label=model, color=model_colors[j], linewidth=1.5, linestyle=style)

    plt.title(f"{ood_primary} ({data})- setting {setting_name}")
    plt.xlabel("Number of Top predicted correlation")
    plt.ylabel(f"Number of True {ood_primary} from ood")
    plt.legend(title='Method')
    plt.grid(False)

    # Save the plot
    plot_filename = f"{data}_{ood_primary}.png"
    plt.savefig(os.path.join(save_folder, plot_filename), dpi=300)
    plt.close()


def pca_plot(ctrl_adata, true_adata, pca_dict, modality_variable, ood_primary, data, path_to_save):
    models = pca_dict.keys()
    num = len(models)
    cols = math.ceil(math.sqrt(num))  
    rows = math.ceil(num / cols)     

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows)) 
    if num>1:
        axes = axes.flatten()
    else:
        axes = [axes]

    for idx, (key, value) in enumerate(pca_dict.items()):

        pca_process(ctrl_adata, true_adata, value, modality_variable, key, axes[idx]) 
        # axes[idx].set_title(key, fontsize=12)
        axes[idx].set_xlabel('Principal Component 1')
        axes[idx].set_ylabel('Principal Component 2')
        axes[idx].legend(title=modality_variable)
        axes[idx].grid()
    fig.suptitle(f"cell type: {ood_primary}", fontsize=20)

    for idx in range(num, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.close()

    plot_filename = f"{data}_{ood_primary}.png"
    fig.savefig(os.path.join(path_to_save, plot_filename), dpi=300)


def pca_process(ctrl_adata, true_adata, pred_adata, modality_variable, key, ax):
    # labels = []
    # labels.extend(ctrl_adata.obs[modality_variable].values)
    # labels.extend(true_adata.obs[modality_variable].values)
    # labels.extend(pred_adata.obs[modality_variable].values)

    # df_list = [ctrl_adata.to_df(), true_adata.to_df(), pred_adata.to_df()]

    # expression_df = pd.concat(df_list, axis=0)
    # # Standardize the data
    # # scaler = StandardScaler()
    # # expression_df = scaler.fit_transform(expression_df)

    # pca = PCA(n_components=2)
    # principal_components = pca.fit_transform(expression_df)

    # pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
    # pca_df[modality_variable] = labels
        
    # for label in pca_df[modality_variable].unique():
    #     subset = pca_df[pca_df[modality_variable] == label]
    #     ax.scatter(subset['PC1'], subset['PC2'], label=label, alpha=0.6)
    #     ax.set_title(f"{key}\nxpln_var_ratio 1 & 2: [{pca.explained_variance_ratio_[0]:.3f}, {pca.explained_variance_ratio_[1]:.3f}]", fontsize=12)
    
    samples1 = ctrl_adata.to_df().to_numpy()
    samples2 = true_adata.to_df().to_numpy()
    predicted = pred_adata.to_df().to_numpy()
    
    reference_data = np.vstack([samples1, samples2])
    
    pca = PCA(n_components=2)
    pca.fit(reference_data)

    samples1_pca = pca.transform(samples1)
    samples2_pca = pca.transform(samples2)
    predicted_pca = pca.transform(predicted)

    ax.scatter(samples1_pca[:, 0], samples1_pca[:, 1], label=ctrl_adata.obs[modality_variable].values.unique()[0], alpha=0.6)
    ax.scatter(samples2_pca[:, 0], samples2_pca[:, 1], label=true_adata.obs[modality_variable].values.unique()[0], alpha=0.6)
    ax.scatter(predicted_pca[:, 0], predicted_pca[:, 1], label=pred_adata.obs[modality_variable].values.unique()[0], alpha=0.6)
    ax.set_title(f"{key}\nxpln_var_ratio 1 & 2: [{pca.explained_variance_ratio_[0]:.3f}, {pca.explained_variance_ratio_[1]:.3f}]", fontsize=12)




def plot_pca_conditional(samples, labels, generated_samples, generated_labels, epoch, modalities_list, label_encodings, path_to_save, plot_all_labels=True, title=""):
    pca = PCA(n_components=2)
    pca.fit(samples)
    
    real_pca = pca.transform(samples)
    generated_pca = pca.transform(generated_samples)

    plt.figure(figsize=(8, 6))
    plt.scatter(real_pca[:, 0], real_pca[:, 1], c='blue', alpha=0.5, s=10, label="Real Data")  # Blue for real data
    plt.scatter(generated_pca[:, 0], generated_pca[:, 1], c='red', alpha=0.5, marker='x', s=20, label="Generated Data")  # Red for generated data
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend()
    plt.title(f"PCA Projection of Real and Generated Samples- Epoch {epoch}\n{title}")
    # plt.show()
    plot_filename = f"real_vs_generated_epoch{epoch}.png"
    plt.savefig(os.path.join(path_to_save, plot_filename), dpi=300)

    plt.close()

    # plt.figure(figsize=(8, 6))
    
    x_min, x_max = min(generated_pca[:, 0].min(), real_pca[:, 0].min()) - 1, \
                    max(generated_pca[:, 0].max(), real_pca[:, 0].max()) + 1
    
    y_min, y_max = min(generated_pca[:, 1].min(), real_pca[:, 1].min()) - 1, \
                    max(generated_pca[:, 1].max(), real_pca[:, 1].max()) + 1
    
    if plot_all_labels:
        num_modalities = len(modalities_list)
        fig, axs = plt.subplots(nrows=num_modalities, ncols=2, figsize=(12, 5 * num_modalities))

        for modality_idx, modality_name in enumerate(modalities_list):
            # Get the labels for this modality only
            real_labels = labels[:, modality_idx]
            gen_labels = generated_labels[:, modality_idx]
    
            # Unique classes in this modality
            unique_classes = np.unique(real_labels)
            colors = plt.get_cmap("tab10", len(unique_classes))
    
            ax_real = axs[modality_idx, 0] if num_modalities > 1 else axs[0]
            ax_gen = axs[modality_idx, 1] if num_modalities > 1 else axs[1]    
            
            for class_idx in unique_classes:
                class_name = label_encodings[modality_idx][class_idx]
    
                # Real samples
                real_mask = real_labels == class_idx
                ax_real.scatter(real_pca[real_mask, 0], real_pca[real_mask, 1],
                            c=[colors(class_idx)], alpha=0.5, s=10, label=f"Real {class_name}")
            ax_real.set_title(f"Real - modality {modality_idx}")
            ax_real.set_xlim(x_min, x_max)
            ax_real.set_ylim(y_min, y_max)
            ax_real.set_xlabel("PCA Component 1")
            ax_real.set_ylabel("PCA Component 2")
            ax_real.legend(fontsize=7)
            
            for class_idx in unique_classes:
                class_name = label_encodings[modality_idx][class_idx]
    
                # Generated samples
                gen_mask = gen_labels == class_idx
                ax_gen.scatter(generated_pca[gen_mask, 0], generated_pca[gen_mask, 1],
                           c=[colors(class_idx)], alpha=0.5, s=20, marker='x', label=f"Generated {class_name}")
            ax_gen.set_title(f"Generated - {modality_idx}")
            ax_gen.set_xlim(x_min, x_max)
            ax_gen.set_ylim(y_min, y_max)
            ax_gen.set_xlabel("PCA Component 1")
            ax_gen.set_ylabel("PCA Component 2")
            ax_gen.legend(fontsize=7)
        plt.suptitle(f"PCA of Real and Generated Samples by Modality - Epoch {epoch}\n{title}", fontsize=14)
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])  # Make room for suptitle
        plot_filename = f"label_colored_epoch{epoch}.png"
        plt.savefig(os.path.join(path_to_save, plot_filename), dpi=300)
        plt.close()



def lineplot_with_errorbar(results, ood_primary, data, save_folder, xlabel, ylabel, filename, title, 
    legend_title='Method', y_min=None, y_max=None, x_log_scale=True, figsize=(10, 6), color_indices=None, num_colors=None, model_style=None):

    # setting_name = save_folder.split('/')[-2]
    # styles = {}
    # models = results.keys()
    # for j, model in enumerate(models):
    #     if j>=(len(models)/2):
    #         styles[model] = 'dashed'
    #     else:
    #         styles[model] = 'solid'

    # plt.figure(figsize=figsize)
    # for method, results_per_gene in results.items():
    #     x = []
    #     y_mean = []
    #     y_std = []

    #     for num_genes in sorted(results_per_gene.keys()):
    #         scores = results_per_gene[num_genes]
    #         x.append(num_genes)
    #         y_mean.append(np.mean(scores))
    #         y_std.append(np.std(scores))
        
    #     plt.errorbar(x, y_mean, yerr=y_std, label=method, capsize=5, marker='o', linestyle=styles[method])

    # plt.xlabel(xlabel)
    # plt.ylabel(ylabel)
    # if x_log_scale:
    #     plt.xscale('log')
    # plt.title(title)
    # plt.legend(title=legend_title, fontsize=7)
    # plt.grid(alpha=0.3)
    # if y_min is not None and y_max is not None:
    #     plt.ylim(y_min, y_max)
    # plt.tight_layout()
    # plt.savefig(os.path.join(save_folder, filename), dpi=300)
    # plt.close()

    # --- Prepare data for Seaborn ---
    styles = {}
    if model_style is not None:
        for j, model in enumerate(results.keys()):
            styles[model] = model_style[j]
    else:
        for model in results.keys():
            styles[model] = 'solid'

    # --- Style setup ---
    sns.set_theme(style="whitegrid", context="talk", font_scale=1.1)
    if not num_colors:
        palette = sns.color_palette("tab10", n_colors=len(results))
    else:
        palette = sns.color_palette("tab10", n_colors=num_colors)
    if color_indices:
        palette = [palette[col] for col in color_indices]

    handles = []
    for method, color in zip(results.keys(), palette):
        line_style = styles[method]
        handles.append(
            Line2D(
                [0], [0],
                color=color,
                linestyle=line_style,
                linewidth=4,
                label=method
            )
        )

    plot_data = []
    for method, results_per_gene in results.items():
        for num_genes, scores in results_per_gene.items():
            for s in scores:
                plot_data.append({
                    'Method': method,
                    'NumGenes': num_genes,
                    'Score': s,
                    'style':styles[method]
                })
    
    df = pd.DataFrame(plot_data)

 
    # --- Plot ---
    plt.figure(figsize=figsize)
    ax = sns.lineplot(
        data=df,
        x="NumGenes",
        y="Score",
        hue="Method",
        palette=palette,
        errorbar='sd',
        # marker="o",
        linewidth=3,        ## 2, 3
        markersize=5,
        style='style',
        err_style="band"      # smooth shaded error bands, 
    )
    ax.tick_params(axis='both', which='major', labelsize=12)

    # --- Axis & labels ---
    if x_log_scale:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.set_title(title, fontsize=16, pad=15)
    
    # --- Legend & limits ---
    # ax.legend(title=legend_title, fontsize=6, title_fontsize=7, loc="best", frameon=True)
    # ax.legend(handles=handles, title=legend_title, fontsize=8, title_fontsize=9, loc="best", frameon=True)
    ax.legend(handles=handles, title=legend_title, fontsize=12, title_fontsize=12, loc="best", frameon=True)

    if y_min is not None and y_max is not None:
        ax.set_ylim(y_min, y_max)

    # --- Styling tweaks ---
    sns.despine()
    ax.grid(alpha=0.3)
    plt.tight_layout()

    # --- Save plot ---
    os.makedirs(save_folder, exist_ok=True)
    plt.savefig(f"{os.path.join(save_folder, filename)}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{os.path.join(save_folder, filename)}.pdf", dpi=300, bbox_inches="tight")
    plt.close()


def plot_gene_expression_boxplots_nonzero(
    adata, ood_primary, gene_list, predictions=None, model="", ood_modality='stimulated', ctrl_key='ctrl', primary_variable='cell_type', 
    modality_variable='condition', max_cols=5, output_file='gene_boxplots.png'):

    def extract_nonzero(X):
        return X.A1 if issparse(X) else np.ravel(X)

    adata_subset = adata[adata.obs[primary_variable] == ood_primary].copy()

    n_genes = len(gene_list)
    n_rows = int(np.ceil(n_genes / max_cols))

    missing_genes = [g for g in gene_list if g not in adata_subset.var_names]
    if missing_genes:
        raise ValueError(f"Genes not found in dataset: {missing_genes}")

    fig, axs = plt.subplots(n_rows, max_cols, figsize=(max_cols*5, n_rows*5), squeeze=False)

    group_labels = [ctrl_key, ood_modality, 'predicted']
    colors = ['lightgreen', 'skyblue', 'salmon']

    for idx, gene in enumerate(gene_list):
        row, col = divmod(idx, max_cols)
        ax = axs[row][col]

        ctrl_vals = extract_nonzero(adata_subset[adata_subset.obs[modality_variable] == ctrl_key][:, gene].X)
        stim_vals = extract_nonzero(adata_subset[adata_subset.obs[modality_variable] == ood_modality][:, gene].X)
        if predictions is None:
            pred_vals = np.array([0])
        else:
            pred_vals = extract_nonzero(predictions[:, gene].X) 

        # Keep only non-zero values
        ctrl_nonzero = ctrl_vals[ctrl_vals > 0]
        if len(ctrl_nonzero) == 0:
            ctrl_nonzero = [0]
        stim_nonzero = stim_vals[stim_vals > 0]
        if len(stim_nonzero) == 0:
            stim_nonzero = [0]
        pred_nonzero = pred_vals[pred_vals > 0]
        if len(pred_nonzero) == 0:
            pred_nonzero = [0]

        box_data = [ctrl_nonzero, stim_nonzero, pred_nonzero]
        box = ax.boxplot(box_data, showfliers=False, patch_artist=True)
        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)
        ax.set_title(gene)
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(group_labels, rotation=45)

        zero_counts = [np.sum(ctrl_vals == 0), np.sum(stim_vals == 0), np.sum(pred_vals == 0)]
        cell_counts = [len(ctrl_vals), len(stim_vals), len(pred_vals)]

        for i, count in enumerate(zero_counts):
            ax.text(i+1, ax.get_ylim()[0], f'Zeros: {count}/{cell_counts[i]}', ha='center', va='bottom', fontsize=9, color='gray')

        for j in range(n_genes, n_rows * max_cols):
            axs[j // max_cols][j % max_cols].set_visible(False)

    plt.suptitle(f"Expression Comparison for {ood_primary} and model {model}", fontsize=30)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()


def plot_mean_gene_expression_distribution(
    adata, ood_primary, gene_list=None, ood_modality='stimulated', ctrl_key='ctrl', primary_variable='cell_type', 
    modality_variable='condition', output_file='mean_distribution.png'):

    # if issparse(adata.X):
    #     adata.X = adata.X.toarray()
    if gene_list is not None:    
        missing_genes = [g for g in gene_list if g not in adata.var_names]
        if missing_genes:
            raise ValueError(f"Genes not found in dataset: {missing_genes}")
    ctrl_adata = adata[(adata.obs[primary_variable] == ood_primary) & (adata.obs[modality_variable] == ctrl_key)].copy()
    stim_adata = adata[(adata.obs[primary_variable] == ood_primary) & (adata.obs[modality_variable] == ood_modality)].copy()

    ctrl_df = ctrl_adata.to_df()
    stim_df = stim_adata.to_df()

    ctrl_withzero_means = np.mean(ctrl_df.drop(columns=gene_list), axis=0).values
    stim_withzero_means = np.mean(stim_df.drop(columns=gene_list), axis=0).values
    ctrl_nonzero_means = np.nan_to_num(np.mean(ctrl_df.drop(columns=gene_list).replace(0, np.NaN), axis=0).values)
    stim_nonzero_means = np.nan_to_num(np.mean(stim_df.drop(columns=gene_list).replace(0, np.NaN), axis=0).values)

    data_all = [[ctrl_withzero_means, stim_withzero_means], [ctrl_nonzero_means, stim_nonzero_means]]
    positions = [0, 1]
    subtitles = ['with-zero mean expression', 'non-zero mean expression']
    colors = sns.color_palette("pastel", len(gene_list))
    scale = 0.07

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    # ax.set_facecolor('#fddbb0')  # Match background

    for mod, data in enumerate(data_all):
        ax = axes[mod]
        for pos, values in zip(positions, data):
            jitter = np.random.normal(loc=0, scale=scale, size=len(values))
            x_vals = np.full_like(values, fill_value=pos) + jitter
            ax.scatter(x_vals, values, color='gray', alpha=0.7, s=10)
            # Add mean lines
            # mean_val = np.mean(values)
            # ax.hlines(mean_val, positions[mod] - 0.2, positions[mod] + 0.2, color='red', linewidth=5)

        for i, gene in enumerate(gene_list):
            if mod==0:
                ctrl_mean = ctrl_adata[:, gene].X.mean() 
                stim_mean = stim_adata[:, gene].X.mean()
            else:
                ctrl_mean = np.nan_to_num(np.mean(ctrl_df.replace(0, np.NaN).loc[:,gene]))
                stim_mean = np.nan_to_num(np.mean(stim_df.replace(0, np.NaN).loc[:,gene]))
            ctrl_val = positions[0] + np.random.normal(loc=0, scale=scale, size=1)
            stim_val = positions[1] + np.random.normal(loc=0, scale=scale, size=1)
            ax.scatter(ctrl_val, ctrl_mean, color=colors[i], zorder=5, s=10, marker="D")
            ax.scatter(stim_val, stim_mean, color=colors[i], zorder=5, s=10, marker="D")
            ax.text(ctrl_val+0.02, ctrl_mean+np.random.choice((-2,1))*0.05, gene, color=colors[i], fontsize=5)
            ax.text(stim_val+0.02, stim_mean+np.random.choice((-2,1))*0.05, gene, color=colors[i], fontsize=5)
        ax.set_xticks(positions)
        ax.set_xticklabels(['Control', 'Stimulated'])
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(0, None)
        ax.set_title(subtitles[mod])
    plt.suptitle(f"Mean Gene Expression Comparison for {ood_primary} in two conditions", fontsize=10)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()





