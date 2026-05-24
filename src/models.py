"""
Machine Learning models for the "To bee or not to bee" project.

Provides:
- Two supervised non-DL, non-ensemble methods (SVM, KNN)
- One supervised ensemble method (Random Forest)
- Two clustering methods (KMeans, DBSCAN/Agglomerative)
- An optional simple CNN (Keras) — only used if tensorflow is installed
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (StratifiedKFold, cross_val_score,
                                     cross_val_predict, train_test_split,
                                     GridSearchCV)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score,
                             adjusted_rand_score, normalized_mutual_info_score,
                             silhouette_score)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def make_pipeline(estimator):
    """Standard scaler + estimator."""
    return Pipeline([('scaler', StandardScaler()),
                     ('clf', estimator)])


def evaluate_supervised(model, X, y, cv=5, name='model'):
    """
    Run stratified k-fold CV and print a classification report based on
    out-of-fold predictions. Returns a dict of metrics.
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=0)
    y_pred = cross_val_predict(model, X, y, cv=skf)
    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred, average='weighted')

    print(f'\n=== {name} ===')
    print(f'CV accuracy: {acc:.3f}   weighted F1: {f1:.3f}')
    print(classification_report(y, y_pred))

    return {
        'model': name,
        'accuracy': acc,
        'f1_weighted': f1,
        'y_pred': y_pred,
    }


def plot_confusion(y_true, y_pred, labels=None, title='Confusion matrix',
                   ax=None, save_path=None):
    """Confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))
    else:
        fig = ax.figure
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


# -----------------------------------------------------------------------------
# Supervised models
# -----------------------------------------------------------------------------

def build_svm():
    """SVM with RBF kernel — non-DL, non-ensemble."""
    return make_pipeline(SVC(kernel='rbf', C=1.0, gamma='scale',
                             class_weight='balanced', random_state=0))


def build_knn(n_neighbors=5):
    """K-Nearest Neighbors — non-DL, non-ensemble."""
    return make_pipeline(KNeighborsClassifier(n_neighbors=n_neighbors,
                                              weights='distance'))


def build_logreg():
    """Logistic Regression — non-DL, non-ensemble (alternative)."""
    return make_pipeline(LogisticRegression(max_iter=1000,
                                            class_weight='balanced',
                                            random_state=0))


def build_random_forest(n_estimators=300):
    """Random Forest — supervised ensemble method."""
    return make_pipeline(RandomForestClassifier(n_estimators=n_estimators,
                                                class_weight='balanced',
                                                random_state=0,
                                                n_jobs=-1))


def build_gradient_boosting():
    """Gradient Boosting — alternative ensemble method."""
    return make_pipeline(GradientBoostingClassifier(random_state=0))


def tune_svm(X, y, cv=None):
    """Grid search SVM, avec validation croisée stratifiée ET mélangée."""
    if cv is None:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    grid = {
        'clf__C': [0.1, 1, 10, 100],
        'clf__gamma': ['scale', 0.01, 0.1, 1],
    }
    gs = GridSearchCV(build_svm(), grid, cv=cv,
                      scoring='f1_weighted', n_jobs=-1)
    gs.fit(X, y)
    return gs


def tune_random_forest(X, y, cv=None):
    """Grid search Random Forest, avec validation croisée stratifiée ET mélangée."""
    if cv is None:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    grid = {
        'clf__n_estimators': [200, 300, 500],
        'clf__max_depth': [None, 10, 20],
        'clf__min_samples_split': [2, 5],
    }
    gs = GridSearchCV(build_random_forest(), grid, cv=cv,
                      scoring='f1_weighted', n_jobs=-1)
    gs.fit(X, y)
    return gs


# -----------------------------------------------------------------------------
# Clustering
# -----------------------------------------------------------------------------

def cluster_kmeans(X, n_clusters=3, random_state=0):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
    return km.fit_predict(Xs), km, Xs


def cluster_dbscan(X, eps=1.0, min_samples=5):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    db = DBSCAN(eps=eps, min_samples=min_samples)
    return db.fit_predict(Xs), db, Xs


def cluster_agglomerative(X, n_clusters=3, linkage='ward'):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    ag = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    return ag.fit_predict(Xs), ag, Xs


def evaluate_clustering(labels_true, labels_pred, X=None, name='clustering'):
    """Print common clustering quality indexes."""
    print(f'\n=== {name} ===')
    n_clusters = len(set(labels_pred)) - (1 if -1 in labels_pred else 0)
    print(f'Number of clusters found: {n_clusters}')

    ari = adjusted_rand_score(labels_true, labels_pred)
    nmi = normalized_mutual_info_score(labels_true, labels_pred)
    print(f'Adjusted Rand Index (vs ground truth): {ari:.3f}')
    print(f'Normalized Mutual Information:         {nmi:.3f}')

    if X is not None:
        mask = labels_pred != -1
        if mask.sum() > 1 and len(set(labels_pred[mask])) > 1:
            sil = silhouette_score(X[mask], labels_pred[mask])
            print(f'Silhouette score:                      {sil:.3f}')
            return {'name': name, 'ari': ari, 'nmi': nmi, 'silhouette': sil}
    return {'name': name, 'ari': ari, 'nmi': nmi}


# -----------------------------------------------------------------------------
# Optional Deep Learning model (only used if tensorflow is installed)
# -----------------------------------------------------------------------------

def build_simple_cnn(input_shape=(128, 128, 3), n_classes=3):
    """
    A small CNN to classify cropped insect images.
    Optional bonus: do NOT use to produce the final CSV unless explicitly trained
    for that purpose.
    """
    try:
        from tensorflow.keras import layers, models
    except ImportError as e:
        raise ImportError('tensorflow is required for the CNN model') from e

    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Rescaling(1.0 / 255),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dense(n_classes, activation='softmax'),
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# -----------------------------------------------------------------------------
# Submission helper
# -----------------------------------------------------------------------------

def make_submission(model, X_test, ids, output_path, label_encoder=None):
    """
    Predict on the test set and save the required CSV with columns ID, bug type.
    """
    preds = model.predict(X_test)
    if label_encoder is not None:
        preds = label_encoder.inverse_transform(preds)
    sub = pd.DataFrame({'ID': ids, 'bug type': preds})
    sub.to_csv(output_path, index=False)
    print(f'Submission saved to {output_path}')
    return sub
