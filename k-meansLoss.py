import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from prince import MCA
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from matplotlib.gridspec import GridSpec
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score





def load_and_preprocess_data(filepath):
    data = pd.read_csv(filepath)
    data = data.dropna()
    print("Data shape after dropping NA values:", data.shape)

    numerical_cols = ['glucose', 'heartRate', 'BMI', 'diaBP', 'sysBP', 'totChol', 'cigsPerDay', 'age']

    categorical_cols = [col for col in data.columns if col not in numerical_cols]

    print(f"\nNumerical columns: {numerical_cols}")
    print(f"Categorical columns: {categorical_cols}")

    existing_num_cols = [col for col in numerical_cols if col in data.columns]
    if len(existing_num_cols) != len(numerical_cols):
        missing = set(numerical_cols) - set(existing_num_cols)
        print(f"Warning: Some numerical columns are missing from the dataset: {missing}")
        numerical_cols = existing_num_cols

    existing_cat_cols = [col for col in categorical_cols if col in data.columns]
    if len(existing_cat_cols) != len(categorical_cols):
        missing = set(categorical_cols) - set(existing_cat_cols)
        print(f"Warning: Some categorical columns are missing from the dataset: {missing}")
        categorical_cols = existing_cat_cols

    return data, numerical_cols, categorical_cols


def apply_dimension_reduction(data, numerical_cols, categorical_cols, n_components=5):

    if len(categorical_cols) > 0:
        X_cat = data[categorical_cols].copy()

        for col in categorical_cols:
            X_cat[col] = X_cat[col].astype(str)

        total_categories = X_cat.nunique().sum()
        max_mca_components = min(total_categories - len(categorical_cols), len(categorical_cols))
        n_cat_components = min(n_components, max_mca_components)

        if n_cat_components > 0:
            try:
                mca = MCA(n_components=n_cat_components)
                X_cat_reduced = mca.fit_transform(X_cat).values
                # Get eigenvalues if available, otherwise skip
                if hasattr(mca, 'eigenvalues_'):
                    print(f"MCA eigenvalues: {mca.eigenvalues_[:n_cat_components]}")
                print(f"MCA components retained: {X_cat_reduced.shape[1]}")
            except Exception as e:
                print(f"Error in MCA: {e}")
                # Fallback to one-hot encoding if MCA fails
                print("Falling back to one-hot encoding for categorical data")
                X_cat_encoded = pd.get_dummies(X_cat)

                # Apply PCA to the one-hot encoded data
                pca_cat = PCA(n_components=min(n_components, X_cat_encoded.shape[1]))
                X_cat_reduced = pca_cat.fit_transform(X_cat_encoded)
                print(
                    f"PCA on one-hot encoded data explained variance: {np.sum(pca_cat.explained_variance_ratio_):.2f}")
                print(f"One-hot PCA components retained: {X_cat_reduced.shape[1]}")
        else:
            X_cat_reduced = np.empty((data.shape[0], 0))
            print("No MCA components retained (insufficient categories).")

    if len(numerical_cols) > 0:
        X_num = data[numerical_cols]
        X_num_scaled = X_num
        # scaler = StandardScaler()
        # X_num_scaled = scaler.fit_transform(X_num)

        pca_num = PCA(n_components=min(n_components, len(numerical_cols)))
        X_num_reduced = pca_num.fit_transform(X_num)
        print(f"\nPCA on numerical data explained variance: {np.sum(pca_num.explained_variance_ratio_):.2f}")
        print(f"PCA components retained: {X_num_reduced.shape[1]}")

    X_combined = np.hstack((X_num_reduced, X_cat_reduced))
    pca_num = PCA(n_components=n_components)
    X_reduced = pca_num.fit_transform(X_combined)
    print(f"Combined data shape after dimension reduction: {X_reduced.shape}")


    return X_combined


def apply_tsne(X, n_components=2, perplexity=30, random_state=42):
    perplexity = min(perplexity, X.shape[0] - 1)
    print(f"\nApplying t-SNE with perplexity={perplexity}")

    # Add check and use appropriate method based on n_components
    if n_components > 3:
        # For n_components > 3, use 'exact' method instead of default 'barnes_hut'
        tsne = TSNE(n_components=n_components, perplexity=perplexity,
                    random_state=random_state, method='exact')
    else:
        tsne = TSNE(n_components=n_components, perplexity=perplexity,
                    random_state=random_state)

    X_tsne = tsne.fit_transform(X)
    print(f"t-SNE output shape: {X_tsne.shape}")
    return X_tsne



def plot_loss_vs_num_clusters(X, max_clusters=15):
    losses = []
    cluster_range = range(1, max_clusters + 1)

    for k in cluster_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        losses.append(kmeans.inertia_)

    initial_loss = losses[0]
    optimal_clusters = next(k for k, loss in enumerate(losses, start=1) if loss <= initial_loss * 0.5)

    plt.figure(figsize=(8, 5))
    plt.plot(cluster_range, losses, marker='o', linestyle='-', color='b', label='K-means loss')
    plt.axvline(x=optimal_clusters, color='k', linestyle='--', label=f'Optimal Clusters ({optimal_clusters})')
    plt.xlabel('Number of Clusters', fontsize=18)
    plt.ylabel('K-means Loss', fontsize=18)
    plt.title('Elbow Method for Optimal Clusters', fontsize=18)
    plt.legend()
    plt.grid()
    plt.show()


def loss_heatmap(data, numerical_cols, categorical_cols, random_state=42):
    cluster_range = range(2, 11)
    dimension_range = range(2, 11)
    loss_matrix = np.zeros((len(dimension_range), len(cluster_range)))

    for i, dim in enumerate(dimension_range):
        data_c = data.copy()
        # First apply dimension reduction
        X_combined = apply_dimension_reduction(data_c, numerical_cols, categorical_cols, n_components=dim)

        for j, k in enumerate(cluster_range):
            # Then apply K-Means on the reduced data
            kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
            kmeans.fit(X_combined)  # Use the reduced data here
            loss_matrix[i, j] = kmeans.inertia_  # Store the loss (inertia)

    plt.figure(figsize=(10, 6))
    sns.heatmap(loss_matrix, xticklabels=list(cluster_range), yticklabels=list(dimension_range), annot=True,
                fmt='.0f',
                cmap='coolwarm')
    plt.xlabel('Number of Clusters', fontsize=18)
    plt.ylabel('Dimension for Reduction', fontsize=18)
    plt.title('K-Means Loss Heatmap (after Dimension Reduction)', fontsize=18)
    plt.show()


def create_clustering_heatmaps(data, numerical_cols, categorical_cols):
    cluster_range = range(2, 11)
    dimension_range = range(2, 11)
    # dimension_range = [10,50,100,200,300]

    n_clusters_range = list(cluster_range)
    n_dims_range = list(dimension_range)

    kmeans_scores = np.zeros((len(n_dims_range), len(n_clusters_range)))
    hierarchical_scores = np.zeros((len(n_dims_range), len(n_clusters_range)))

    for i, n_dims in enumerate(n_dims_range):
        # Use your custom dimension reduction functions
        X_combined = apply_dimension_reduction(data.copy(), numerical_cols, categorical_cols, n_components=n_dims)
        # X_tsne = apply_tsne(X_combined, n_components=2, perplexity=30, random_state=42)
        reduced_data = X_combined  # This now contains the t-SNE reduced data

        for j, n_clusters in enumerate(n_clusters_range):
            try:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                kmeans_labels = kmeans.fit_predict(reduced_data)
                kmeans_scores[i, j] = silhouette_score(reduced_data, kmeans_labels) if len(
                    np.unique(kmeans_labels)) > 1 else -1
            except Exception:
                kmeans_scores[i, j] = -1

            try:
                hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
                hierarchical_labels = hierarchical.fit_predict(reduced_data)
                hierarchical_scores[i, j] = silhouette_score(reduced_data, hierarchical_labels) if len(
                    np.unique(hierarchical_labels)) > 1 else -1
            except Exception:
                hierarchical_scores[i, j] = -1

    # Create first figure for K-means
    fig1 = plt.figure(figsize=(7, 6))
    sns.heatmap(kmeans_scores, annot=True, fmt=".2f", cmap="viridis",
                xticklabels=n_clusters_range, yticklabels=n_dims_range,
                cbar_kws={'label': 'Silhouette Score'})
    plt.title('K-means Silhouette Score', fontsize=16)
    plt.xlabel('Number of Clusters',fontsize=18)
    plt.ylabel('Number of Dimensions (Dimension Reduction)',fontsize=18)
    plt.tight_layout()

    # Create second figure for Hierarchical Clustering
    fig2 = plt.figure(figsize=(7, 6))
    sns.heatmap(hierarchical_scores, annot=True, fmt=".2f", cmap="viridis",
                xticklabels=n_clusters_range, yticklabels=n_dims_range,
                cbar_kws={'label': 'Silhouette Score'})
    plt.title('Hierarchical Clustering Silhouette Score', fontsize=16)
    plt.xlabel('Number of Clusters', fontsize=18)
    plt.ylabel('Number of Dimensions (Dimension Reduction)', fontsize=18)
    plt.tight_layout()

    # Return both figures as a tuple
    return fig1, fig2





def validate_clustering_solution(data, numerical_cols, categorical_cols, n_clusters=4, n_dims=8):

    # X_combined = apply_dimension_reduction(data.copy(), numerical_cols, categorical_cols, n_components=4)
    # X_tsne = apply_tsne(X_combined, n_components=2, perplexity=30, random_state=42)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(data)

    X_combined = apply_dimension_reduction(data.copy(), numerical_cols, categorical_cols, n_components=n_dims)
    X_tsne = apply_tsne(X_combined, n_components=2, perplexity=30, random_state=42)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111)
    scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1],
                         c=cluster_labels, cmap='viridis', s=5, alpha=0.7)
    fig.colorbar(scatter, label='Cluster')
    ax.set_xlabel("t-SNE Dimension 1", fontsize=18)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=18)
    plt.show()

    data_with_clusters = data.copy()
    data_with_clusters['cluster'] = cluster_labels

    results = {}

    results['silhouette_score'] = silhouette_score(X_combined, cluster_labels)
    results['davies_bouldin_score'] = davies_bouldin_score(X_combined, cluster_labels)
    results['calinski_harabasz_score'] = calinski_harabasz_score(X_combined, cluster_labels)

    print(f"Cluster Validation Metrics:")
    print(f"---------------------------")
    print(f"Silhouette Score: {results['silhouette_score']:.4f} (higher is better, range: -1 to 1)")
    print(f"Davies-Bouldin Score: {results['davies_bouldin_score']:.4f} (lower is better)")
    print(f"Calinski-Harabasz Score: {results['calinski_harabasz_score']:.4f} (higher is better)")

    all_columns = [col for col in data_with_clusters.columns if col != 'cluster']

    anova_results = {}
    print("\nANOVA Tests for ALL Features:")
    print("---------------------------")

    for col in all_columns:
        try:
            if data[col].dtype in ['object', 'category', 'bool']:
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                temp_col = le.fit_transform(data_with_clusters[col].fillna('missing'))

                groups = [temp_col[data_with_clusters['cluster'] == i] for i in range(n_clusters)]
            else:
                groups = [data_with_clusters[data_with_clusters['cluster'] == i][col].values for i in range(n_clusters)]

            groups = [group[~np.isnan(group)] for group in groups]

            if all(len(group) > 1 for group in groups):
                f_val, p_val = stats.f_oneway(*groups)
                anova_results[col] = {'F-value': f_val, 'p-value': p_val, 'significant': p_val < 0.05}

                print(f"{col}: F={f_val:.4f}, p={p_val:.4f} {'(significant)' if p_val < 0.05 else ''}")
            else:
                print(f"{col}: Insufficient data for ANOVA test")
                anova_results[col] = {'F-value': None, 'p-value': None, 'significant': False}
        except Exception as e:
            print(f"{col}: Error in ANOVA test - {str(e)}")
            anova_results[col] = {'F-value': None, 'p-value': None, 'significant': False, 'error': str(e)}

    results['anova'] = anova_results

    kw_results = {}
    print("\nKruskal-Wallis Tests for ALL Features (non-parametric alternative to ANOVA):")
    print("-----------------------------------------------------------------------")

    for col in all_columns:
        try:
            if data[col].dtype in ['object', 'category', 'bool']:
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                temp_col = le.fit_transform(data_with_clusters[col].fillna('missing'))

                groups = [temp_col[data_with_clusters['cluster'] == i] for i in range(n_clusters)]
            else:
                groups = [data_with_clusters[data_with_clusters['cluster'] == i][col].values for i in range(n_clusters)]

            groups = [group[~np.isnan(group)] for group in groups]

            if all(len(group) > 1 for group in groups):
                kw_stat, p_val = stats.kruskal(*groups)
                kw_results[col] = {'H-statistic': kw_stat, 'p-value': p_val, 'significant': p_val < 0.05}

                print(f"{col}: H={kw_stat:.4f}, p={p_val:.4f} {'(significant)' if p_val < 0.05 else ''}")
            else:
                print(f"{col}: Insufficient data for Kruskal-Wallis test")
                kw_results[col] = {'H-statistic': None, 'p-value': None, 'significant': False}
        except Exception as e:
            print(f"{col}: Error in Kruskal-Wallis test - {str(e)}")
            kw_results[col] = {'H-statistic': None, 'p-value': None, 'significant': False, 'error': str(e)}

    results['kruskal_wallis'] = kw_results

    plt.figure(figsize=(10, 6))
    cluster_sizes = pd.Series(cluster_labels).value_counts().sort_index()
    ax = cluster_sizes.plot(kind='bar', color='skyblue')
    ax.set_xlabel('Cluster', fontsize=18)
    ax.set_ylabel('Number of Samples', fontsize=18)
    ax.set_title(f'Cluster Sizes (n_clusters={n_clusters}, n_dims={n_dims})', fontsize=18)

    for i, v in enumerate(cluster_sizes):
        ax.text(i, v + 5, str(v), ha='center')

    plt.tight_layout()
    plt.show()

    significant_features = [col for col in all_columns if col in anova_results and anova_results[col]['significant']]

    if len(significant_features) > 10:
        significant_features = sorted(
            significant_features,
            key=lambda col: anova_results[col]['p-value'] if anova_results[col]['p-value'] is not None else 1.0
        )[:10]

    if significant_features:
        max_features_per_plot = 3
        num_plots = (len(significant_features) + max_features_per_plot - 1) // max_features_per_plot

        for plot_idx in range(num_plots):
            start_idx = plot_idx * max_features_per_plot
            end_idx = min((plot_idx + 1) * max_features_per_plot, len(significant_features))
            features_to_plot = significant_features[start_idx:end_idx]

            fig, axes = plt.subplots(1, len(features_to_plot), figsize=(5 * len(features_to_plot), 5))
            if len(features_to_plot) == 1:
                axes = [axes]

            for i, feature in enumerate(features_to_plot):
                try:
                    if data[feature].dtype in ['object', 'category', 'bool']:
                        cat_counts = []
                        for cluster in range(n_clusters):
                            cluster_data = data_with_clusters[data_with_clusters['cluster'] == cluster][
                                feature].value_counts(normalize=True)
                            cat_counts.append(cluster_data)

                        all_cats = set()
                        for counts in cat_counts:
                            all_cats.update(counts.index)

                        if len(all_cats) > 5:
                            all_cats = set()
                            for counts in cat_counts:
                                all_cats.update(counts.nlargest(5).index)

                        plot_data = pd.DataFrame(index=list(all_cats))
                        for c, counts in enumerate(cat_counts):
                            plot_data[f'Cluster {c}'] = counts.reindex(plot_data.index, fill_value=0)

                        plot_data.plot(kind='bar', ax=axes[i])
                        axes[i].set_title(f'{feature} Distribution by Cluster', fontsize=18)
                        axes[i].set_xlabel(feature, fontsize=18)
                        axes[i].legend(title='Cluster')
                        axes[i].tick_params(axis='x', rotation=45)
                    else:
                        for cluster in range(n_clusters):
                            cluster_data = data_with_clusters[data_with_clusters['cluster'] == cluster][feature]
                            if len(cluster_data) > 0:
                                sns.kdeplot(cluster_data, ax=axes[i], label=f'Cluster {cluster}')

                        axes[i].set_title(f'{feature} Distribution', fontsize=18)
                        axes[i].set_xlabel(feature, fontsize=18)
                        axes[i].legend()
                except Exception as e:
                    axes[i].text(0.5, 0.5, f"Error plotting {feature}:\n{str(e)}",
                                 ha='center', va='center', transform=axes[i].transAxes)

            plt.tight_layout()
            plt.show()

    alt_clusters = [2, 3, 4, 5, 6]
    silhouette_scores = []
    db_scores = []
    ch_scores = []

    for n_clust in alt_clusters:
        kmeans = KMeans(n_clusters=n_clust, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_combined)

        silhouette_scores.append(silhouette_score(X_combined, labels))
        db_scores.append(davies_bouldin_score(X_combined, labels))
        ch_scores.append(calinski_harabasz_score(X_combined, labels))

    # Plot Silhouette Score
    plt.figure(figsize=(8, 6))
    plt.plot(alt_clusters, silhouette_scores, 'o-', color='blue', linewidth=2, markersize=10)
    plt.axvline(x=n_clusters, color='red', linestyle='--', linewidth=2)
    plt.title('Silhouette Score (higher is better)', fontsize=18)
    plt.xlabel('Number of Clusters', fontsize=16)
    plt.ylabel('Score', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(alt_clusters)
    plt.tight_layout()
    plt.show()

    # Plot Davies-Bouldin Score
    plt.figure(figsize=(8, 6))
    plt.plot(alt_clusters, db_scores, 'o-', color='green', linewidth=2, markersize=10)
    plt.axvline(x=n_clusters, color='red', linestyle='--', linewidth=2)
    plt.title('Davies-Bouldin Score (lower is better)', fontsize=18)
    plt.xlabel('Number of Clusters', fontsize=16)
    plt.ylabel('Score', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(alt_clusters)
    plt.tight_layout()
    plt.show()

    # Plot Calinski-Harabasz Score
    plt.figure(figsize=(8, 6))
    plt.plot(alt_clusters, ch_scores, 'o-', color='purple', linewidth=2, markersize=10)
    plt.axvline(x=n_clusters, color='red', linestyle='--', linewidth=2)
    plt.title('Calinski-Harabasz Score (higher is better)', fontsize=18)
    plt.xlabel('Number of Clusters', fontsize=16)
    plt.ylabel('Score', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(alt_clusters)
    plt.tight_layout()
    plt.show()

    num_significant_anova = sum(1 for col in anova_results if anova_results[col].get('significant', False))
    num_significant_kw = sum(1 for col in kw_results if kw_results[col].get('significant', False))

    silhouette_quality = "Good" if results['silhouette_score'] > 0.5 else \
        "Fair" if results['silhouette_score'] > 0.25 else "Poor"

    optimal_cluster_idx = silhouette_scores.index(max(silhouette_scores))
    optimal_clusters = alt_clusters[optimal_cluster_idx]

    print("\nSummary Evaluation:")
    print("------------------")
    print(f"Current configuration: {n_clusters} clusters with {n_dims} dimensions")
    print(
        f"Number of significant features (ANOVA): {num_significant_anova}/{len(anova_results)} ({num_significant_anova / len(anova_results) * 100:.1f}%)")
    print(
        f"Number of significant features (Kruskal-Wallis): {num_significant_kw}/{len(kw_results)} ({num_significant_kw / len(kw_results) * 100:.1f}%)")
    print(f"Silhouette score quality: {silhouette_quality}")

    if optimal_clusters == n_clusters:
        print(f"The current choice of {n_clusters} clusters appears optimal based on silhouette score.")
    else:
        print(f"A different number of clusters ({optimal_clusters}) might be more optimal based on silhouette score.")

    print("\nRecommendation:")
    significance_threshold = 0.4

    if silhouette_quality == "Good" and optimal_clusters == n_clusters and \
            (num_significant_anova / len(anova_results) >= significance_threshold):
        print(f"The chosen configuration of {n_clusters} clusters and {n_dims} dimensions appears to be a good choice.")
        print(
            f"The clusters show significant differences in {num_significant_anova} out of {len(anova_results)} features.")
    elif silhouette_quality == "Fair" and \
            (optimal_clusters == n_clusters or abs(optimal_clusters - n_clusters) <= 1) and \
            (num_significant_anova / len(anova_results) >= significance_threshold / 2):
        print(f"The chosen configuration is acceptable, but could potentially be improved.")
        # print(f"Consider using {optimal_clusters} clusters instead of {n_clusters}.")

    return results


if __name__ == "__main__":
    filepath = '/Users/arielshamis/Downloads/framingham_heart_study.csv'

    data, numerical_cols, categorical_cols = load_and_preprocess_data(filepath)

    X_combined = apply_dimension_reduction(data, numerical_cols, categorical_cols)

    # X_tsne = apply_tsne(X_combined)

    plot_loss_vs_num_clusters(X_combined)
    loss_heatmap(data, numerical_cols, categorical_cols)

    fig = create_clustering_heatmaps(data, numerical_cols, categorical_cols)
    plt.show()

    result = validate_clustering_solution(data, numerical_cols, categorical_cols)
    print(result)

