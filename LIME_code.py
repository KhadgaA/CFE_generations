from functools import partial
from matplotlib import pyplot as plt
import sklearn
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def generate_neighbours(data, test_points, num_samples, categorical_features=None):
    transforms = sklearn.preprocessing.StandardScaler()
    X_train_ = transforms.fit(data)
    std = transforms.scale_
    mean = transforms.mean_
    mean[categorical_features] = 0
    std[categorical_features] = 1
    num_features = data.shape[1]
    num_samples = num_samples + 1
    perturbations = np.random.normal(scale=1, loc=0, size=(num_samples, num_features))
    # print(perturbations.shape, std.shape, data.shape, mean.shape)
    neighbours = perturbations * std + mean
    if categorical_features is not None:
        for feature in categorical_features:
            # print(feature)
            categorical_features = np.random.choice(
                [1, 0], num_samples, p=[mean[feature], 1 - mean[feature]], replace=True
            )
            neighbours[:, feature] = (
                categorical_features == test_points[feature]
            ) * categorical_features
    neighbours[0] = test_points
    return neighbours, mean, std


def generate_predictions(neighbours, predictor):
    predictions_neighbours = predictor(neighbours)
    return predictions_neighbours


def calculate_distances(neighbours, mean, std):
    scaled_neighbours = (neighbours - mean) / std
    # scaled_test_point = (test_points - mean) / std
    scaled_test_point = scaled_neighbours[0]
    distances = pairwise_distances(
        scaled_neighbours, scaled_test_point.reshape(1, -1), metric="euclidean"
    ).ravel()
    return distances


# categorical_features = ["blue", "dual_sim", "four_g", "three_g", "touch_screen", "wifi"]
# categorical_features_idx = [
#     idx for idx, feature in enumerate(data.columns) if feature in categorical_features
# ]

# data["price_range"] = data["price_range"].map({0: 0, 1: 0, 2: 1, 3: 1})



data = pd.read_csv("diabetes.csv")
data = data.dropna()

categorical_features_idx = []
target_column = "Outcome"
X_train, X_test, y_train, y_test = train_test_split(
    data.drop(columns=[target_column]).values, data[target_column], random_state=1
)

transforms = sklearn.preprocessing.StandardScaler()
X_train_ = transforms.fit_transform(X_train)
X_test = transforms.transform(X_test)

rf = RandomForestClassifier(n_estimators=100, random_state=1)
rf.fit(X_train, y_train)
rf.predict(X_test)


num_features = X_train.shape[1]
test_points = X_test[0]
num_samples = 5000
k_features = 8


neighbours, mean, std = generate_neighbours(
    X_train, test_points, num_samples, categorical_features_idx
)
predictions_neighbours = generate_predictions(neighbours, rf.predict_proba)
distances = calculate_distances(neighbours, mean, std)

class_names = ["0", "1"]
top_labels = 8

local_exp = {}
intercept = {}
score = {}
local_pred = {}
top_lablels_list = []
local_features = {}
# if top_labels is None:
labels = np.argsort(predictions_neighbours[0])[-top_labels:]
top_labels_list = list(labels)
top_labels_list.reverse()

kernel_width = np.sqrt(num_features) * 0.75
kernel_width = float(kernel_width)


def kernel(d, kernel_width):
    return np.sqrt(np.exp(-(d**2) / kernel_width**2))


kernel_fn = partial(kernel, kernel_width=kernel_width)


def feature_selection(data, labels, weights, k_features):
    clf = Ridge(alpha=0.01, fit_intercept=True, random_state=0)
    clf.fit(data, labels, sample_weight=weights)
    coef = clf.coef_
    weighted_data = coef * data[0]
    feature_weights = sorted(
        zip(range(data.shape[1]), weighted_data),
        key=lambda x: np.abs(x[1]),
        reverse=True,
    )
    return np.array([x[0] for x in feature_weights[:k_features]])


def interpret_instance(
    neighbours, predictions_neighbours, distances, label, k_features, feature_selection
):

    weights = kernel_fn(distances)
    labels_column = predictions_neighbours[:, label]
    used_features = feature_selection(neighbours, labels_column, weights, k_features)
    # print(used_features)

    model_regressor = Ridge(alpha=1, fit_intercept=True, random_state=0)
    linear_model = model_regressor
    linear_model.fit(neighbours[:, used_features], labels_column, sample_weight=weights)
    prediction_score = linear_model.score(
        neighbours[:, used_features], labels_column, sample_weight=weights
    )

    local_pred = linear_model.predict(neighbours[0, used_features].reshape(1, -1))

    if True:
        print("Intercept", linear_model.intercept_)
        print(
            "Prediction_local",
            local_pred,
        )
        print("Right:", predictions_neighbours[0, label])
    return (
        linear_model.intercept_,
        sorted(
            zip(used_features, linear_model.coef_),
            key=lambda x: np.abs(x[1]),
            reverse=True,
        ),
        prediction_score,
        local_pred,
    )


for label in top_labels_list:
    print("Class: ",label)

    intercept[label], local_exp[label], score[label], local_pred[label] = (
        interpret_instance(
            neighbours,
            predictions_neighbours,
            distances,
            label,
            k_features,
            feature_selection,
        )
    )


def plot_local_exp(local_exp):
    fig, axs = plt.subplots(len(local_exp), 1, figsize=(10, 3 * len(local_exp)))

    for i, (class_label, explanations) in enumerate(local_exp.items()):
        features, weights = zip(*explanations)
        feature_names = [data.columns[i] for i in features]

        axs[i].barh(
            feature_names, weights, color=["red" if w < 0 else "blue" for w in weights]
        )
        axs[i].set_title(f"Class {class_label}")
        axs[i].set_xlabel("Weights")
        axs[i].set_ylabel("Features")

    plt.tight_layout()
    plt.show()


plot_local_exp(local_exp)

# import lime
# import lime.lime_tabular

# explainer = lime.lime_tabular.LimeTabularExplainer(
#     X_train,
#     feature_names=data.columns[:-1],
#     class_names=class_names,
#     discretize_continuous=False,
# )
# exp = explainer.explain_instance(
#     test_points, rf.predict_proba, num_features=5, top_labels=top_labels
# )
# exp.show_in_notebook(show_table=True, show_all=False)
