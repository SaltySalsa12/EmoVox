from sklearn.metrics import classification_report
import joblib
import tensorflow as tf

from utils.data_loader import load_dataset

def cross_dataset_test():
    model = tf.keras.models.load_model('models/trained_model.h5')
    selector = joblib.load('models/feature_selector.pkl')
    svm = joblib.load('models/svm_classifier.pkl')
    
    datasets = ['ravdess', 'emodb']
    
    for dataset in datasets:
        X, y = load_dataset(f'data/{dataset}', dataset)
        X = selector.transform(X)
        
        # Deep Learning Evaluation
        dl_pred = model.predict(X).argmax(axis=1)
        print(f"DL Results for {dataset}:")
        print(classification_report(y, dl_pred))
        
        # Hybrid Evaluation
        feature_extractor = model(inputs=model.input, outputs=model.layers[-2].output)
        deep_features = feature_extractor.predict(X)
        svm_pred = svm.predict(deep_features)
        print(f"Hybrid Results for {dataset}:")
        print(classification_report(y, svm_pred))