import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import tensorflow as tf

def load_and_preprocess_excel(data.csv):
    df = pd.read_excel(file_path)
    X = df.select_dtypes(include=[np.number]).iloc[:, :-1]
    y = df.select_dtypes(include=[np.number]).iloc[:, -1]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=min(20, X_scaled.shape[1]))
    X_pca = pca.fit_transform(X_scaled)
    svm = SVC(kernel='linear')
    svm.fit(X_pca, y)
    feature_importance = np.abs(svm.coef_).flatten()
    top_features_indices = feature_importance.argsort()[-10:]
    X_selected = X_pca[:, top_features_indices]
    return X_selected, y

excel_files = ["dataset1.xlsx", "dataset2.xlsx", "dataset3.xlsx"]
results = []

for i, file in enumerate(excel_files):
    X, y = load_and_preprocess_excel(file)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes=2)
    y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes=2)
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train_cat, epochs=5, batch_size=32, verbose=0)
    y_pred = model.predict(X_test)
    y_pred_labels = np.argmax(y_pred, axis=1)
    acc = accuracy_score(y_test, y_pred_labels)
    report = classification_report(y_test, y_pred_labels, output_dict=True)
    results.append({
        'dataset': f'Dataset {i+1}',
        'accuracy': acc,
        'precision': report['weighted avg']['precision'],
        'recall': report['weighted avg']['recall'],
        'f1-score': report['weighted avg']['f1-score']
    })

results_df = pd.DataFrame(results)
print(results_df)
