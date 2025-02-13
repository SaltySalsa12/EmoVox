import joblib
from tensorflow.keras import layers, Model # type: ignore
from sklearn.svm import SVC
from utils import *
from utils.data_loader import load_dataset
from utils.features import FeatureSelector

# Model Architecture
def build_dscnn_ssa(input_shape=(42,), num_classes=4):
    inputs = layers.Input(shape=input_shape)
    
    # Strided DSCNN
    x = layers.Reshape((input_shape[0], 1))(inputs)
    x = layers.DepthwiseConv1D(3, strides=2, padding='same')(x)
    x = layers.Conv1D(32, 1, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    
    # SSA-like Attention
    x = layers.MultiHeadAttention(num_heads=2, key_dim=32)(x, x)
    
    # Temporal Pooling
    x = layers.GlobalAveragePooling1D()(x)
    
    # Output
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    return Model(inputs, outputs)

# Training Pipeline
def main():
    # Load data
    X_train, y_train = load_dataset('data/iemocap', 'iemocap')

    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)
        
    # Verify non-empty data
    assert X_train.shape[0] > 0, "No valid training samples found!"
    
    
    # Feature selection
    selector = FeatureSelector(k=50)
    selector.fit(X_train, y_train)
    X_train = selector.transform(X_train)
    
    # Save selector
    joblib.dump(selector, 'models/feature_selector.pkl')
    
    # Build and train model
    model = build_dscnn_ssa()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=50, validation_split=0.2)
    model.save('models/trained_model.h5')
    
    # Train SVM on deep features
    feature_extractor = Model(inputs=model.input, outputs=model.layers[-2].output)
    deep_features = feature_extractor.predict(X_train)
    svm = SVC()
    svm.fit(deep_features, y_train)
    joblib.dump(svm, 'models/svm_classifier.pkl')

if __name__ == '__main__':
    main()