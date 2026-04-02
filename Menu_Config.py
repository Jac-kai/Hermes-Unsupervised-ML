# -------------------- Common clustering options --------------------
COMMON_CLUSTER_PARAM_CONFIG = {
    "cat_encoder": {
        "label": "🧩 Categorical Encoder",
        "options": {
            1: "ohe",
            2: "ordinal",
        },
        "default": 1,
    },
    "scaler_type": {
        "label": "📏 Scaler Type",
        "options": {
            1: "standard",
            2: "minmax",
            3: "robust",
            4: None,
        },
        "default": 1,
    },
}

# -------------------- Model-specific clustering parameter options --------------------
MODEL_CLUSTER_PARAM_CONFIG = {
    "KMeans": [
        {
            "name": "n_clusters",
            "label": "✒️ Number of Clusters",
            "options": {
                1: 2,
                2: 3,
                3: 4,
                4: 5,
                5: 6,
                6: 8,
                7: 10,
            },
            "default": 2,
        },
        {
            "name": "init",
            "label": "🧪 Init Method",
            "options": {
                1: "k-means++",
                2: "random",
            },
            "default": 1,
        },
        {
            "name": "n_init",
            "label": "♻️ N Init",
            "options": {
                1: 10,
                2: 20,
                3: 50,
                4: "auto",
            },
            "default": 4,
        },
        {
            "name": "random_state",
            "label": "🎲 Model Random State",
            "options": {
                1: 42,
                2: 0,
                3: 7,
                4: 123,
                5: None,
            },
            "default": 1,
        },
    ],
    "DBSCAN": [
        {
            "name": "eps",
            "label": "📍 Epsilon",
            "options": {
                1: 0.3,
                2: 0.5,
                3: 0.7,
                4: 1.0,
                5: 1.5,
            },
            "default": 2,
        },
        {
            "name": "min_samples",
            "label": "🧺 Min Samples",
            "options": {
                1: 3,
                2: 5,
                3: 10,
                4: 15,
            },
            "default": 2,
        },
        {
            "name": "metric",
            "label": "📐 Distance Metric",
            "options": {
                1: "euclidean",
                2: "manhattan",
                3: "cosine",
            },
            "default": 1,
        },
    ],
    "AgglomerativeClustering": [
        {
            "name": "n_clusters",
            "label": "✒️ Number of Clusters",
            "options": {
                1: 2,
                2: 3,
                3: 4,
                4: 5,
                5: 6,
                6: 8,
            },
            "default": 2,
        },
        {
            "name": "linkage",
            "label": "🧷 Linkage",
            "options": {
                1: "ward",
                2: "complete",
                3: "average",
                4: "single",
            },
            "default": 1,
        },
        {
            "name": "metric",
            "label": "📐 Distance Metric",
            "options": {
                1: "euclidean",
                2: "manhattan",
                3: "cosine",
            },
            "default": 1,
            "depends_on": ("linkage", ("complete", "average", "single")),
        },
    ],
}

# -------------------- Evaluation menu options --------------------
EVALUATION_MENU_OPTIONS = {
    1: "🪶 Show Current Model Summary",
    2: "🪶 Show Cluster Evaluation",
    3: "🪶 Show Cluster Label Preview",
    4: "🪶Show Clustered Data Preview",
    5: "🪶Show Cluster Size Summary",
    6: "🖌️ Plot Cluster Size Barplot",
    7: "🖌️ Plot Cluster Scatter",
    8: "🖌️ Plot Cluster PCA",
    9: "🖌️ Plot Silhouette",
    10: "🖌️ Plot Cluster Profile Heatmap",
    11: "📈 KMeans Elbow Plot",
    12: "📉 DBSCAN K-Distance Plot",
    13: "📊 Agglomerative Dendrogram Plot",
}

# =================================================
