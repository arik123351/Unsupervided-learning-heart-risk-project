import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import OneClassSVM
from sklearn.neighbors import NearestNeighbors
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor



def load_data(csv_path, remove_outliers=True):

    try:
        data = pd.read_csv(csv_path)

        if remove_outliers and not data.empty:
            # Health metrics columns to check for outliers
            health_columns = ['glucose', 'heartRate', 'BMI', 'diaBP', 'sysBP',
                              'totChol', 'cigsPerDay', 'age']

            # Verify which columns exist in the dataset
            existing_columns = [col for col in health_columns if col in data.columns]

            if existing_columns:
                initial_rows = len(data)
                preserve_mask = data['TenYearCHD'] == 1 if 'TenYearCHD' in data.columns else pd.Series(False,
                                                                                                       index=data.index)
                rows_to_keep = pd.Series(True, index=data.index)
                for column in existing_columns:
                    lower_bound = data[column].quantile(0.01)
                    upper_bound = data[column].quantile(0.99)
                    column_mask = (data[column] >= lower_bound) & (data[column] <= upper_bound)
                    rows_to_keep = rows_to_keep & (column_mask | preserve_mask)
                data = data[rows_to_keep]
                removed_rows = initial_rows - len(data)
                preserved_cases = sum(preserve_mask)

        # data['age'] = data['age'] // 2

        return data

    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None




def preprocess_data(data):
    # Replace NaN values with the median of each column
    data = data.dropna()

    # Exclude TenYearCHD from features used for anomaly detection
    feature_cols = [col for col in data.select_dtypes(include=['float64', 'int64']).columns
                   if col != 'TenYearCHD']
    X = data[feature_cols]

    return X.values, X.columns, data


def detect_anomalies_isolation_forest(X, contamination=0.1, threshold_percentile=None, random_state=42):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = X

    model = IsolationForest(contamination=contamination, random_state=random_state)
    model.fit(X_scaled)

    # Anomaly scores: negative values mean more abnormal
    anomaly_scores = -model.decision_function(X_scaled)

    if threshold_percentile is not None:
        threshold = np.percentile(anomaly_scores, threshold_percentile)
    else:
        threshold = np.sort(anomaly_scores)[-int(len(anomaly_scores) * contamination)]

    anomalies = anomaly_scores > threshold

    return {
        'anomalies': anomalies,
        'scores': anomaly_scores,
        'threshold': threshold,
        'model': model,
        'scaler': scaler
    }

def detect_anomalies_lof(X, n_neighbors=20, contamination=0.1, threshold_percentile=None):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = X

    model = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination, novelty=False)
    model.fit(X_scaled)

    # LOF gives negative_outlier_factor_ (the lower, the more outlier)
    anomaly_scores = -model.negative_outlier_factor_

    if threshold_percentile is not None:
        threshold = np.percentile(anomaly_scores, threshold_percentile)
    else:
        threshold = np.sort(anomaly_scores)[-int(len(anomaly_scores) * contamination)]

    anomalies = anomaly_scores > threshold

    return {
        'anomalies': anomalies,
        'scores': anomaly_scores,
        'threshold': threshold,
        'model': model,
        'scaler': scaler,
        'n_neighbors': n_neighbors
    }


def detect_anomalies_kmeans(X, n_clusters=4, threshold_percentile=95):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(X)

    distances = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        distances[i] = np.linalg.norm(X[i] - kmeans.cluster_centers_[cluster_labels[i]])

    threshold = np.percentile(distances, threshold_percentile)

    anomalies = distances > threshold

    return {
        'cluster_labels': cluster_labels,
        'distances': distances,
        'anomalies': anomalies,
        'threshold': threshold,
        'kmeans_model': kmeans
    }


def detect_anomalies_ocsvm(X, nu=0.05, kernel='rbf', gamma='scale'):
    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = X

    # Apply One-Class SVM
    ocsvm = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)
    ocsvm.fit(X_scaled)

    # Get decision function values (distance from the hyperplane)
    decision_scores = ocsvm.decision_function(X_scaled)

    # In OneClassSVM, predictions are 1 for inliers and -1 for outliers
    # We convert to boolean where True means anomaly
    predictions = ocsvm.predict(X_scaled)
    anomalies = predictions == -1

    return {
        'anomalies': anomalies,
        'decision_scores': decision_scores,
        'ocsvm_model': ocsvm,
        'scaler': scaler
    }


def detect_anomalies_dbscan(X, eps=None, min_samples=None, auto_eps=True, eps_percentile=95):
    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = X

    # Automatically determine eps if not provided
    if eps is None and auto_eps:
        # Calculate distances to the k-th nearest neighbor
        k = min_samples if min_samples else max(5, int(0.01 * X.shape[0]))
        neigh = NearestNeighbors(n_neighbors=k)
        neigh.fit(X_scaled)
        distances, _ = neigh.kneighbors(X_scaled)
        # Sort and get the k-th distances
        k_distances = np.sort(distances[:, -1])

        # Find the "elbow" in the k-distance graph or use a percentile
        eps = np.percentile(k_distances, eps_percentile)

        print(f"Auto-selected eps = {eps:.4f} based on {eps_percentile}th percentile of k-distances")

    # Use default values if parameters are not provided
    if min_samples is None:
        min_samples = max(5, int(0.01 * X.shape[0]))  # Default to 1% of data size, at least 5
    if eps is None:
        eps = 0.5  # Default value

    # Apply DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X_scaled)

    # In DBSCAN, label -1 represents noise points (outliers)
    anomalies = labels == -1

    # Calculate distances to nearest core point for visualization
    # First, identify core points
    core_samples_mask = np.zeros_like(labels, dtype=bool)
    core_samples_mask[dbscan.core_sample_indices_] = True

    # Calculate distances from each point to nearest core point
    distances = np.ones(X_scaled.shape[0]) * np.inf

    if np.any(core_samples_mask):  # Check if any core points exist
        for i in range(X_scaled.shape[0]):
            if not core_samples_mask[i]:  # Only calculate for non-core points
                # Find the nearest core point
                core_points = X_scaled[core_samples_mask]
                if core_points.size > 0:  # Ensure there are core points
                    distances[i] = np.min(np.linalg.norm(X_scaled[i] - core_points, axis=1))
            else:
                distances[i] = 0  # Core points have zero distance to themselves

    return {
        'anomalies': anomalies,
        'labels': labels,
        'dbscan_model': dbscan,
        'distances': distances,
        'scaler': scaler,
        'eps': eps,
        'min_samples': min_samples
    }


def detect_anomalies_knn(X, n_neighbors=5, threshold_percentile=95, contamination=0.1):

    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = X

    # Create and fit KNN model
    knn = NearestNeighbors(n_neighbors=n_neighbors)
    knn.fit(X_scaled)

    # Calculate distances to k-nearest neighbors
    distances, _ = knn.kneighbors(X_scaled)

    # Use the mean distance to k nearest neighbors as anomaly score
    # (alternatively, could use max distance or other metrics)
    anomaly_scores = np.mean(distances, axis=1)

    # Determine threshold
    if threshold_percentile is not None:
        # Using percentile-based threshold
        threshold = np.percentile(anomaly_scores, threshold_percentile)
    else:
        # Using contamination-based threshold
        threshold = np.sort(anomaly_scores)[-int(len(anomaly_scores) * contamination)]

    # Identify anomalies
    anomalies = anomaly_scores > threshold

    return {
        'anomalies': anomalies,
        'distances': anomaly_scores,
        'threshold': threshold,
        'knn_model': knn,
        'scaler': scaler,
        'n_neighbors': n_neighbors
    }


def visualize_results(X, results, original_data, title="Anomaly Detection Results", method="kmeans", n_clusters=4):
    if 'TenYearCHD' not in original_data.columns:
        print("TenYearCHD column not found in the dataset")
        return

    # Perform K-means clustering upfront for all methods
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X)

    # Create figure
    plt.figure(figsize=(14, 8))

    # Get basic masks for all methods
    non_anomalies = ~results['anomalies']
    normal_with_chd = non_anomalies & (original_data['TenYearCHD'] == 1)
    anomaly_with_chd = results['anomalies'] & (original_data['TenYearCHD'] == 1)

    # Set up data for plotting
    cluster_labels_text = [f'Cluster {i}' for i in range(n_clusters)]
    bar_width = 0.35
    positions = np.arange(len(cluster_labels_text))

    # Calculate percentages for each cluster
    normal_chd_percentages = []
    anomaly_chd_percentages = []

    for cluster in range(n_clusters):
        # Calculate normal points with CHD percentage
        normal_points_in_cluster = np.sum((cluster_labels == cluster) & non_anomalies)
        normal_chd_points = np.sum((cluster_labels == cluster) & normal_with_chd)
        normal_chd_percent = (normal_chd_points / normal_points_in_cluster * 100) if normal_points_in_cluster > 0 else 0
        normal_chd_percentages.append(normal_chd_percent)

        # Calculate anomalous points with CHD percentage
        anomaly_points_in_cluster = np.sum((cluster_labels == cluster) & results['anomalies'])
        anomaly_chd_points = np.sum((cluster_labels == cluster) & anomaly_with_chd)
        anomaly_chd_percent = (
                    anomaly_chd_points / anomaly_points_in_cluster * 100) if anomaly_points_in_cluster > 0 else 0
        anomaly_chd_percentages.append(anomaly_chd_percent)

    # Create bar chart
    plt.bar(positions - bar_width / 2, normal_chd_percentages, bar_width,
            label='Normal with CHD', color='blue', alpha=0.7)
    plt.bar(positions + bar_width / 2, anomaly_chd_percentages, bar_width,
            label='Anomaly with CHD', color='red', alpha=0.7)

    # Add percentage labels
    for i in range(len(positions)):
        plt.text(positions[i] - bar_width / 2, normal_chd_percentages[i] + 0.5,
                 f"{normal_chd_percentages[i]:.1f}%", ha='center', va='bottom', fontsize=13)
        plt.text(positions[i] + bar_width / 2, anomaly_chd_percentages[i] + 0.5,
                 f"{anomaly_chd_percentages[i]:.1f}%", ha='center', va='bottom', fontsize=13)

    # Add additional information based on the method
    if method == "ocsvm":
        # Add decision score information
        decision_scores = results['decision_scores']
        avg_scores = [np.mean(decision_scores[cluster_labels == i]) for i in range(n_clusters)]

        for i, avg in enumerate(avg_scores):
            plt.annotate(f"Avg Score: {avg:.2f}",
                         xy=(positions[i], -5), ha='center', fontsize=13)

        plt.subplots_adjust(bottom=0.15)

    elif method == "knn":
        # Add distance information
        distances = results['distances']
        avg_distances = [np.mean(distances[cluster_labels == i]) for i in range(n_clusters)]

        for i, avg in enumerate(avg_distances):
            plt.annotate(f"Avg Dist: {avg:.2f}",
                         xy=(positions[i], -5), ha='center', fontsize=13)

        plt.subplots_adjust(bottom=0.15)

    elif method == "dbscan":
        # Add DBSCAN label distribution info
        dbscan_labels = results['labels']

        for i in range(n_clusters):
            dbscan_label_counts = {}
            cluster_indices = np.where(cluster_labels == i)[0]
            for label in set(dbscan_labels[cluster_indices]):
                count = np.sum(dbscan_labels[cluster_indices] == label)
                dbscan_label_counts[label] = count

            # Find the dominant DBSCAN label
            dominant_label = max(dbscan_label_counts, key=dbscan_label_counts.get)
            dominant_pct = dbscan_label_counts[dominant_label] / len(cluster_indices) * 100

            label_text = "Noise" if dominant_label == -1 else f"DBSCAN {dominant_label}"
            plt.annotate(f"{label_text}: {dominant_pct:.1f}%",
                         xy=(positions[i], -5), ha='center', fontsize=13)

        plt.subplots_adjust(bottom=0.15)

    # Set common labels and title
    plt.xlabel('K-means Clusters', fontsize=18)
    plt.ylabel('CHD Percentage (%)', fontsize=18)
    plt.title(f'CHD Percentages for {method.upper()} Algorithm (Using K-means Clusters)', fontsize=18)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Return kmeans model and labels for potential further use
    return {'kmeans_model': kmeans, 'kmeans_labels': cluster_labels}


# Add this function to create a comprehensive visualization for all methods
def visualize_all_methods_chd(X, kmeans_results, ocsvm_results, dbscan_results, knn_results, original_data):

    if 'TenYearCHD' not in original_data.columns:
        print("TenYearCHD column not found in the dataset")
        return

    # Call individual visualization functions
    visualize_results(X, kmeans_results, original_data,
                      title="K-means CHD Analysis", method="kmeans")

    visualize_results(X, ocsvm_results, original_data,
                      title="One-Class SVM CHD Analysis", method="ocsvm")

    visualize_results(X, dbscan_results, original_data,
                      title="DBSCAN CHD Analysis", method="dbscan")

    visualize_results(X, knn_results, original_data,
                      title="KNN CHD Analysis", method="knn")


def analyze_anomalies(data, results, feature_names):
    feature_names = [f for f in feature_names if f != 'TenYearCHD']
    anomalous_data = data[results['anomalies']]
    normal_data = data[~results['anomalies']]

    stats = []
    for feature in feature_names:
        anomaly_mean = anomalous_data[feature].mean()
        normal_mean = normal_data[feature].mean()
        percent_diff = ((anomaly_mean - normal_mean) / normal_mean * 100) if normal_mean != 0 else float('inf')

        stats.append({
            'feature': feature,
            'normal_mean': normal_mean,
            'anomaly_mean': anomaly_mean,
            'percent_diff': percent_diff
        })

    stats_sorted = sorted(stats, key=lambda x: abs(x['percent_diff']), reverse=True)

    return stats_sorted


def analyze_clusters_dbscan(X, results, feature_names):
    labels = results['labels']
    unique_labels = np.unique(labels)

    # Convert to DataFrame for easier analysis
    data_df = pd.DataFrame(X, columns=feature_names)
    data_df['cluster'] = labels

    # Create cluster summary
    cluster_summary = []

    for label in unique_labels:
        cluster_data = data_df[data_df['cluster'] == label]
        cluster_size = len(cluster_data)

        if cluster_size == 0:
            continue

        # Calculate basic statistics for each feature in this cluster
        feature_stats = {}
        for feature in feature_names:
            feature_stats[f"{feature}_mean"] = cluster_data[feature].mean()
            feature_stats[f"{feature}_std"] = cluster_data[feature].std()

        cluster_summary.append({
            'cluster': label,
            'size': cluster_size,
            'percentage': (cluster_size / len(data_df) * 100),
            **feature_stats
        })

    # Convert to DataFrame
    summary_df = pd.DataFrame(cluster_summary)

    # Add a descriptive label
    summary_df['description'] = summary_df['cluster'].apply(
        lambda x: 'Anomalies (Noise)' if x == -1 else f'Cluster {x}')

    return summary_df


def compare_methods(X, kmeans_results, ocsvm_results, dbscan_results=None, knn_results=None):
    # Get anomaly indicators from methods
    kmeans_anomalies = kmeans_results['anomalies']
    ocsvm_anomalies = ocsvm_results['anomalies']

    methods = ["KMeans", "OneClassSVM"]
    anomalies_list = [kmeans_anomalies, ocsvm_anomalies]

    if dbscan_results is not None:
        dbscan_anomalies = dbscan_results['anomalies']
        methods.append("DBSCAN")
        anomalies_list.append(dbscan_anomalies)

    if knn_results is not None:
        knn_anomalies = knn_results['anomalies']
        methods.append("KNN")
        anomalies_list.append(knn_anomalies)

    total_points = X.shape[0]

    print("\nMethod Comparison:")
    print(f"Total data points: {total_points}")

    # Print individual method statistics
    for method, anomalies in zip(methods, anomalies_list):
        anomaly_count = np.sum(anomalies)
        print(f"{method} detected anomalies: {anomaly_count} ({anomaly_count / total_points * 100:.2f}%)")

    # Create a matrix to compare agreement between each pair of methods
    n_methods = len(methods)

    print("\nAgreement Matrix (%):")
    header = "Method".ljust(12) + " | " + " | ".join(method.ljust(12) for method in methods)
    print(header)
    print("-" * len(header))

    for i, method1 in enumerate(methods):
        row = method1.ljust(12) + " | "
        for j, method2 in enumerate(methods):
            if i == j:
                agreement = 100.0
            else:
                agreement_count = np.sum(anomalies_list[i] == anomalies_list[j])
                agreement = (agreement_count / total_points) * 100
            row += f"{agreement:6.2f}%".ljust(12) + " | "
        print(row)

    # Analyze points flagged by multiple methods
    print("\nAnomaly Detection Overlap:")

    # Points flagged by all methods
    all_methods_mask = np.ones(total_points, dtype=bool)
    for anomalies in anomalies_list:
        all_methods_mask = all_methods_mask & anomalies

    all_methods_anomalies = np.sum(all_methods_mask)
    print(f"Points flagged as anomalous by all {n_methods} methods: {all_methods_anomalies}")

    # Pairwise overlaps
    print("\nPairwise Overlap:")
    for i in range(n_methods):
        for j in range(i + 1, n_methods):
            overlap = np.sum(anomalies_list[i] & anomalies_list[j])
            print(f"Points flagged by both {methods[i]} and {methods[j]}: {overlap}")

    # Points flagged by only one method
    print("\nExclusive Detections:")
    for i in range(n_methods):
        exclusive_mask = anomalies_list[i].copy()
        for j in range(n_methods):
            if i != j:
                exclusive_mask = exclusive_mask & ~anomalies_list[j]
        exclusive_count = np.sum(exclusive_mask)
        print(f"Points flagged only by {methods[i]}: {exclusive_count}")


def analyze_chd_anomalies(original_data, results):
    if 'TenYearCHD' not in original_data.columns:
        print("TenYearCHD column not found in the dataset")
        return None

    # Get anomaly indicators
    anomalies = results['anomalies']

    # Calculate CHD rates
    total_points = len(original_data)
    total_chd = original_data['TenYearCHD'].sum()
    total_chd_rate = total_chd / total_points * 100

    # Anomalous points
    anomaly_points = np.sum(anomalies)
    anomaly_chd = np.sum(original_data.loc[anomalies, 'TenYearCHD'])
    anomaly_chd_rate = anomaly_chd / anomaly_points * 100 if anomaly_points > 0 else 0

    # Non-anomalous points
    normal_points = total_points - anomaly_points
    normal_chd = total_chd - anomaly_chd
    normal_chd_rate = normal_chd / normal_points * 100 if normal_points > 0 else 0

    # Calculate difference
    chd_rate_difference = anomaly_chd_rate - normal_chd_rate
    relative_difference = (anomaly_chd_rate / normal_chd_rate - 1) * 100 if normal_chd_rate > 0 else float('inf')

    # Print results
    print("\nCHD Analysis:")
    print(f"Overall CHD rate: {total_chd_rate:.2f}% ({total_chd}/{total_points})")
    print(f"CHD rate in anomalous points: {anomaly_chd_rate:.2f}% ({anomaly_chd}/{anomaly_points})")
    print(f"CHD rate in normal points: {normal_chd_rate:.2f}% ({normal_chd}/{normal_points})")
    print(f"Absolute difference in CHD rate: {chd_rate_difference:.2f}%")
    print(f"Relative difference in CHD rate: {relative_difference:.2f}%")

    # Calculate statistical significance using chi-square test
    from scipy.stats import chi2_contingency

    contingency_table = np.array([
        [anomaly_chd, anomaly_points - anomaly_chd],
        [normal_chd, normal_points - normal_chd]
    ])

    chi2, p_value, _, _ = chi2_contingency(contingency_table)
    print(
        f"Chi-square p-value: {p_value:.4f} ({'statistically significant' if p_value < 0.05 else 'not statistically significant'})")

    return {
        'total_chd_rate': total_chd_rate,
        'anomaly_chd_rate': anomaly_chd_rate,
        'normal_chd_rate': normal_chd_rate,
        'chd_rate_difference': chd_rate_difference,
        'relative_difference': relative_difference,
        'p_value': p_value
    }


def visualize_feature_importance(anomaly_stats, method_name, top_n=10):
    filtered_stats = [stat for stat in anomaly_stats if stat['feature'] != 'TenYearCHD']

    for stat in filtered_stats:
        if not np.isfinite(stat['percent_diff']):
            stat['percent_diff'] = 0.0

    sorted_stats = sorted(filtered_stats, key=lambda x: abs(x['percent_diff']), reverse=True)
    top_stats = sorted_stats[:top_n]

    features = [stat['feature'] for stat in top_stats]
    percent_diffs = [stat['percent_diff'] for stat in top_stats]

    colors = ['#ff9999' if pd < 0 else '#66b3ff' for pd in percent_diffs]

    plt.figure(figsize=(12, 8))
    bars = plt.barh(features, [abs(pd) for pd in percent_diffs], color=colors)

    for i, bar in enumerate(bars):
        width = bar.get_width()
        if width > 0:
            plt.text(width + 1, bar.get_y() + bar.get_height() / 2,
                     f"{abs(percent_diffs[i]):.1f}%",
                     va='center', ha='left', fontsize=13)

            indicator = "↑" if percent_diffs[i] > 0 else "↓"
            plt.text(0, bar.get_y() + bar.get_height() / 2,
                     indicator,
                     va='center', ha='right', fontsize=13,
                     color='white', fontweight='bold')

    plt.xlabel('Absolute Percentage Difference from Normal', fontsize=18)
    plt.ylabel('Features', fontsize=18)
    plt.xticks(size=12)
    plt.yticks(size=14)
    plt.title(f'Top {top_n} Features Characterizing {method_name} Anomalies', fontsize=18)

    higher_patch = plt.Rectangle((0, 0), 1, 1, fc='#66b3ff')
    lower_patch = plt.Rectangle((0, 0), 1, 1, fc='#ff9999')
    plt.legend([higher_patch, lower_patch],
               ['Higher in Anomalies', 'Lower in Anomalies'],
               loc='upper right')

    plt.tight_layout()
    plt.show()

    return plt


def compare_feature_importance_across_methods(kmeans_stats, ocsvm_stats, dbscan_stats, knn_stats, top_n=5):
    filtered_kmeans = [stat for stat in kmeans_stats]
    filtered_ocsvm = [stat for stat in ocsvm_stats]
    filtered_dbscan = [stat for stat in dbscan_stats]
    filtered_knn = [stat for stat in knn_stats]

    for stats_list in [filtered_kmeans, filtered_ocsvm, filtered_dbscan, filtered_knn]:
        for stat in stats_list:
            if not np.isfinite(stat['percent_diff']):
                stat['percent_diff'] = 0.0

    method_stats = {
        "K-means": sorted(filtered_kmeans, key=lambda x: abs(x['percent_diff']), reverse=True)[:top_n],
        "One-Class SVM": sorted(filtered_ocsvm, key=lambda x: abs(x['percent_diff']), reverse=True)[:top_n],
        "DBSCAN": sorted(filtered_dbscan, key=lambda x: abs(x['percent_diff']), reverse=True)[:top_n],
        "KNN": sorted(filtered_knn, key=lambda x: abs(x['percent_diff']), reverse=True)[:top_n]
    }

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for i, (method_name, stats) in enumerate(method_stats.items()):
        ax = axes[i]

        features = [stat['feature'] for stat in stats]
        percent_diffs = [stat['percent_diff'] for stat in stats]

        colors = ['#ff9999' if pd < 0 else '#66b3ff' for pd in percent_diffs]

        bars = ax.barh(features, [abs(pd) for pd in percent_diffs], color=colors)

        for j, bar in enumerate(bars):
            width = bar.get_width()
            if width > 0:
                ax.text(width + 1, bar.get_y() + bar.get_height() / 2,
                        f"{abs(percent_diffs[j]):.1f}%",
                        va='center', ha='left', fontsize=13)

                indicator = "↑" if percent_diffs[j] > 0 else "↓"
                ax.text(0, bar.get_y() + bar.get_height() / 2,
                        indicator,
                        va='center', ha='right', fontsize=13,
                        color='white', fontweight='bold')

        # ax.set_xlabel('Absolute Percentage Difference')
        ax.set_ylabel('Features', fontsize=18)
        ax.set_title(f'{method_name} Top Features', fontsize=18)

        if i == 0:
            higher_patch = plt.Rectangle((0, 0), 1, 1, fc='#66b3ff')
            lower_patch = plt.Rectangle((0, 0), 1, 1, fc='#ff9999')
            ax.legend([higher_patch, lower_patch],
                      ['Higher in Anomalies', 'Lower in Anomalies'],
                      loc='lower right', fontsize=8)

    fig.suptitle('Feature Importance Comparison Across Anomaly Detection Methods', fontsize=16)

    plt.tight_layout()
    plt.subplots_adjust(top=0.9, bottom=0.1)
    plt.show()

    return fig


def create_variable_importance_heatmap(kmeans_stats, ocsvm_stats, dbscan_stats, knn_stats):
    methods = ["K-means", "One-Class SVM", "DBSCAN", "KNN"]

    filtered_kmeans = [stat for stat in kmeans_stats if stat['feature'] != 'TenYearCHD']
    filtered_ocsvm = [stat for stat in ocsvm_stats if stat['feature'] != 'TenYearCHD']
    filtered_dbscan = [stat for stat in dbscan_stats if stat['feature'] != 'TenYearCHD']
    filtered_knn = [stat for stat in knn_stats if stat['feature'] != 'TenYearCHD']

    for stats_list in [filtered_kmeans, filtered_ocsvm, filtered_dbscan, filtered_knn]:
        for stat in stats_list:
            if not np.isfinite(stat['percent_diff']):
                stat['percent_diff'] = 0.0

    all_stats = [filtered_kmeans, filtered_ocsvm, filtered_dbscan, filtered_knn]

    all_features = set()
    for stats in all_stats:
        all_features.update([stat['feature'] for stat in stats])

    all_features = list(all_features)

    data_dict = {}
    for feature in all_features:
        data_dict[feature] = []
        for method_stats in all_stats:
            feature_val = 0.0
            for stat in method_stats:
                if stat['feature'] == feature:
                    feature_val = stat['percent_diff']
                    break
            data_dict[feature].append(feature_val)

    df = pd.DataFrame(data_dict, index=methods)

    feature_avg_importance = df.abs().mean().sort_values(ascending=False)
    sorted_features = feature_avg_importance.index.tolist()

    top_n = min(15, len(sorted_features))
    top_features = sorted_features[:top_n]
    df_top = df[top_features]

    plt.figure(figsize=(14, 10))

    masked_data = df_top.copy()
    mask = np.zeros_like(masked_data.T, dtype=bool)

    cmap = sns.diverging_palette(240, 10, as_cmap=True)

    sns.heatmap(masked_data.T, annot=True, cmap=cmap, center=0, fmt=".1f",
                linewidths=.5, cbar_kws={'label': 'Percent Difference from Normal'})

    plt.title('Variable Importance Across Anomaly Detection Methods', fontsize=16)
    plt.ylabel('Features')
    plt.xlabel('Methods')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    return plt


def test_feature_importance(original_data, anomaly_results, output_csv_path='feature_importance_tests.csv'):
    import pandas as pd
    import numpy as np
    from scipy.stats import mannwhitneyu, ttest_ind

    results = []
    anomalies = anomaly_results['anomalies']

    # Group data by anomaly status
    anomalous_data = original_data[anomalies]
    normal_data = original_data[~anomalies]

    for feature in original_data.columns:
        # Skip non-numeric columns or target variable
        if feature == 'TenYearCHD' or not pd.api.types.is_numeric_dtype(original_data[feature]):
            continue

        # Get feature values for normal and anomalous points
        normal_values = normal_data[feature].dropna()
        anomaly_values = anomalous_data[feature].dropna()

        # Skip if too few samples
        if len(normal_values) < 2 or len(anomaly_values) < 2:
            results.append({
                'feature': feature,
                'ttest_pvalue': np.nan,
                'mannwhitney_pvalue': np.nan
            })
            continue

        # Perform t-test (parametric)
        try:
            _, ttest_pvalue = ttest_ind(normal_values, anomaly_values, equal_var=False)
        except:
            ttest_pvalue = np.nan

        # Perform Mann-Whitney U test (non-parametric)
        try:
            _, mannwhitney_pvalue = mannwhitneyu(normal_values, anomaly_values)
        except:
            mannwhitney_pvalue = np.nan

        results.append({
            'feature': feature,
            'ttest_pvalue': ttest_pvalue,
            'mannwhitney_pvalue': mannwhitney_pvalue
        })

    # Create DataFrame and sort by p-values
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by='mannwhitney_pvalue')

    # Save to CSV
    results_df.to_csv(output_csv_path, index=False)

    return results_df


if __name__ == "__main__":
    csv_path = '/Users/arielshamis/Downloads/framingham_heart_study.csv'

    data = load_data(csv_path)
    if data is not None:
        print(f"Dataset shape: {data.shape}")
        print("\nFeature summary:")
        print(data.describe().T[['count', 'mean', 'std', 'min', 'max']])

        X, feature_names, original_data = preprocess_data(data)

        # K-means anomaly detection
        print("\n=== K-means Anomaly Detection ===")
        threshold_percentile = 90
        kmeans_results = detect_anomalies_kmeans(X, n_clusters=4, threshold_percentile=threshold_percentile)

        anomaly_count = np.sum(kmeans_results['anomalies'])
        print(f"\nDetected {anomaly_count} anomalies out of {X.shape[0]} data points "
              f"({anomaly_count / X.shape[0] * 100:.2f}%)")

        for cluster in range(kmeans_results['kmeans_model'].n_clusters):
            cluster_points = np.sum(kmeans_results['cluster_labels'] == cluster)
            cluster_anomalies = np.sum((kmeans_results['cluster_labels'] == cluster) & kmeans_results['anomalies'])
            print(f"Cluster {cluster}: {cluster_points} points, {cluster_anomalies} anomalies "
                  f"({cluster_anomalies / cluster_points * 100:.2f}%)")

        # One-Class SVM anomaly detection
        print("\n=== One-Class SVM Anomaly Detection ===")
        ocsvm_results = detect_anomalies_ocsvm(X, nu=0.1)  # nu is the expected proportion of outliers

        anomaly_count_svm = np.sum(ocsvm_results['anomalies'])
        print(f"\nDetected {anomaly_count_svm} anomalies out of {X.shape[0]} data points "
              f"({anomaly_count_svm / X.shape[0] * 100:.2f}%)")

        # DBSCAN anomaly detection
        print("\n=== DBSCAN Anomaly Detection ===")
        # Auto-determine parameters based on dataset
        dbscan_results = detect_anomalies_dbscan(X, auto_eps=True, eps_percentile=70)

        anomaly_count_dbscan = np.sum(dbscan_results['anomalies'])
        print(f"\nDetected {anomaly_count_dbscan} anomalies out of {X.shape[0]} data points "
              f"({anomaly_count_dbscan / X.shape[0] * 100:.2f}%)")

        # KNN anomaly detection
        print("\n=== KNN Anomaly Detection ===")
        knn_results = detect_anomalies_knn(X, n_neighbors=5, threshold_percentile=90)

        anomaly_count_knn = np.sum(knn_results['anomalies'])
        print(f"\nDetected {anomaly_count_knn} anomalies out of {X.shape[0]} data points "
              f"({anomaly_count_knn / X.shape[0] * 100:.2f}%)")

        isolation_forest_results = detect_anomalies_isolation_forest(X, threshold_percentile=90)
        lof_results = detect_anomalies_lof(X, n_neighbors=5, threshold_percentile=90)

        # Print cluster information for DBSCAN
        n_clusters = len(set(dbscan_results['labels'])) - (1 if -1 in dbscan_results['labels'] else 0)
        print(f"Number of clusters found by DBSCAN: {n_clusters}")

        # Analyze DBSCAN clusters
        cluster_summary = analyze_clusters_dbscan(X, dbscan_results, feature_names)
        print("\nDBSCAN Cluster Summary:")
        print(cluster_summary[['description', 'size', 'percentage']].to_string(index=False))

        # Compare all methods
        compare_methods(X, kmeans_results, ocsvm_results, dbscan_results, knn_results)

        # Visualize all methods
        visualize_results(X, kmeans_results, original_data,
                          title=f"K-means Anomaly Detection (n_clusters={4})",
                          method="kmeans")

        visualize_results(X, ocsvm_results, original_data,
                          title="One-Class SVM Anomaly Detection",
                          method="ocsvm")

        visualize_results(X, dbscan_results, original_data,
                          title=f"DBSCAN Anomaly Detection (eps={dbscan_results['eps']:.3f}, min_samples={dbscan_results['min_samples']})",
                          method="dbscan")

        visualize_results(X, knn_results, original_data,
                          title=f"KNN Anomaly Detection (n_neighbors={knn_results['n_neighbors']})",
                          method="knn")

        visualize_results(X, isolation_forest_results, original_data,
                          title=f"Isolation Forest Anomaly Detection",
                          method="Isolation Forest")

        visualize_results(X, lof_results, original_data,
                          title=f"LOF Anomaly Detection (n_neighbors={lof_results['n_neighbors']})",
                          method="LOF")

        # Analyze anomalies for all methods
        print("\n=== K-means Anomaly Analysis ===")
        kmeans_anomaly_stats = analyze_anomalies(pd.DataFrame(X, columns=feature_names),
                                                 kmeans_results, feature_names)

        print("\nTop features characterizing K-means anomalies:")
        for i, stat in enumerate(kmeans_anomaly_stats[:10]):
            print(f"{i + 1}. {stat['feature']}: Normal mean = {stat['normal_mean']:.3f}, "
                  f"Anomaly mean = {stat['anomaly_mean']:.3f}, "
                  f"Percent difference = {stat['percent_diff']:.2f}%")

        print("\n=== One-Class SVM Anomaly Analysis ===")
        ocsvm_anomaly_stats = analyze_anomalies(pd.DataFrame(X, columns=feature_names),
                                                ocsvm_results, feature_names)

        print("\nTop features characterizing One-Class SVM anomalies:")
        for i, stat in enumerate(ocsvm_anomaly_stats[:10]):
            print(f"{i + 1}. {stat['feature']}: Normal mean = {stat['normal_mean']:.3f}, "
                  f"Anomaly mean = {stat['anomaly_mean']:.3f}, "
                  f"Percent difference = {stat['percent_diff']:.2f}%")

        print("\n=== DBSCAN Anomaly Analysis ===")
        dbscan_anomaly_stats = analyze_anomalies(pd.DataFrame(X, columns=feature_names),
                                                 dbscan_results, feature_names)

        print("\nTop features characterizing DBSCAN anomalies:")
        for i, stat in enumerate(dbscan_anomaly_stats[:10]):
            print(f"{i + 1}. {stat['feature']}: Normal mean = {stat['normal_mean']:.3f}, "
                  f"Anomaly mean = {stat['anomaly_mean']:.3f}, "
                  f"Percent difference = {stat['percent_diff']:.2f}%")

        print("\n=== KNN Anomaly Analysis ===")
        knn_anomaly_stats = analyze_anomalies(pd.DataFrame(X, columns=feature_names),
                                              knn_results, feature_names)
        isolation_forest_anomaly_stats = analyze_anomalies(pd.DataFrame(X, columns=feature_names),
                                              isolation_forest_results, feature_names)
        lof_anomaly_stats = analyze_anomalies(pd.DataFrame(X, columns=feature_names),
                                              lof_results, feature_names)

        print("\nTop features characterizing KNN anomalies:")
        for i, stat in enumerate(knn_anomaly_stats[:10]):
            print(f"{i + 1}. {stat['feature']}: Normal mean = {stat['normal_mean']:.3f}, "
                  f"Anomaly mean = {stat['anomaly_mean']:.3f}, "
                  f"Percent difference = {stat['percent_diff']:.2f}%")
    else:
        print("Failed to load dataset. Please check the file path.")

    if 'TenYearCHD' in original_data.columns:
        print("\n=== CHD Analysis for K-means Anomalies ===")
        kmeans_chd_stats = analyze_chd_anomalies(original_data, kmeans_results)

        print("\n=== CHD Analysis for One-Class SVM Anomalies ===")
        ocsvm_chd_stats = analyze_chd_anomalies(original_data, ocsvm_results)

        print("\n=== CHD Analysis for DBSCAN Anomalies ===")
        dbscan_chd_stats = analyze_chd_anomalies(original_data, dbscan_results)

        print("\n=== CHD Analysis for KNN Anomalies ===")
        knn_chd_stats = analyze_chd_anomalies(original_data, knn_results)
        isolation_forest_chd_stats = analyze_chd_anomalies(original_data, isolation_forest_results)
        lof_chd_stats = analyze_chd_anomalies(original_data, lof_results)

    visualize_feature_importance(kmeans_anomaly_stats, "K-means")
    visualize_feature_importance(ocsvm_anomaly_stats, "One-Class SVM")
    visualize_feature_importance(dbscan_anomaly_stats, "DBSCAN")
    visualize_feature_importance(knn_anomaly_stats, "KNN")
    visualize_feature_importance(isolation_forest_anomaly_stats, "Isolation Forest")
    visualize_feature_importance(lof_anomaly_stats, "LOF")


    compare_feature_importance_across_methods(kmeans_anomaly_stats, ocsvm_anomaly_stats,
                                              dbscan_anomaly_stats, knn_anomaly_stats)

    create_variable_importance_heatmap(kmeans_anomaly_stats, ocsvm_anomaly_stats,
                                       dbscan_anomaly_stats, knn_anomaly_stats)

    # After running your anomaly detection methods:
    kmeans_test_results = test_feature_importance(
        pd.DataFrame(X, columns=feature_names),
        kmeans_results,
        'kmeans_feature_importance.csv'
    )

    ocsvm_test_results = test_feature_importance(
        pd.DataFrame(X, columns=feature_names),
        ocsvm_results,
        'ocsvm_feature_importance.csv'
    )

    dbscan_test_results = test_feature_importance(
        pd.DataFrame(X, columns=feature_names),
        dbscan_results,
        'dbscan_feature_importance.csv'
    )

    knn_test_results = test_feature_importance(
        pd.DataFrame(X, columns=feature_names),
        knn_results,
        'knn_feature_importance.csv'
    )

    isolation_forest_test_results = test_feature_importance(
        pd.DataFrame(X, columns=feature_names),
        isolation_forest_results,
        'isolation_forest_feature_importance.csv'
    )

    lof_test_results = test_feature_importance(
        pd.DataFrame(X, columns=feature_names),
        lof_results,
        'lof_importance.csv'
    )

    # Print the tables
    print("K-means Feature Significance:")
    print(kmeans_test_results)
    print(kmeans_test_results['ttest_pvalue'].median())

    print("\nOne-Class SVM Feature Significance:")
    print(ocsvm_test_results)

    print("\nDBSCAN Feature Significance:")
    print(dbscan_test_results)

    print("\nKNN Feature Significance:")
    print(knn_test_results)

    print("\nIsolation Forest Feature Significance:")
    print(isolation_forest_test_results)

    print("\nLOF Feature Significance:")
    print(lof_test_results)