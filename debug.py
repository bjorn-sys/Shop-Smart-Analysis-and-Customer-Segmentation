# check_model_issues.py
import joblib
import pickle
import numpy as np
import pandas as pd

def analyze_model_issues():
    print("🔍 Analyzing Model Issues...")
    
    # Load models
    kmeans = joblib.load('customer_segmentation_model.pkl')
    scaler = joblib.load('scaler.pkl')
    pca = joblib.load('pca_model.pkl')
    
    with open('model_metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    
    print("📊 Model Analysis:")
    print(f"KMeans cluster centers shape: {kmeans.cluster_centers_.shape}")
    print(f"PCA components shape: {pca.components_.shape}")
    print(f"PCA explained variance: {pca.explained_variance_ratio_}")
    
    # Check the scaler statistics
    print(f"\n🔧 Scaler Statistics:")
    print(f"Scaler mean: {scaler.mean_}")
    print(f"Scaler scale: {scaler.scale_}")
    
    # Check what the training data looked like
    print(f"\n📈 Expected Input Ranges (from scaler):")
    features = metadata['feature_names']
    for i, feature in enumerate(features):
        mean = scaler.mean_[i]
        std = scaler.scale_[i]
        print(f"{feature}: mean={mean:.2f}, std={std:.2f}")
    
    # Test with extreme values that should trigger different clusters
    print(f"\n🧪 Testing Extreme Values:")
    
    extreme_cases = [
        # Should be High Value
        {'location': 243, 'order_count': 20, 'quantity': 50, 'order_amount': 20000, 'order_duration': 100, 'status': 1},
        # Should be Mid Value  
        {'location': 120, 'order_count': 10, 'quantity': 25, 'order_amount': 5000, 'order_duration': 5000, 'status': 1},
        # Should be Low Value
        {'location': 1, 'order_count': 1, 'quantity': 1, 'order_amount': 50, 'order_duration': 100000, 'status': 0},
    ]
    
    for i, test_case in enumerate(extreme_cases):
        print(f"\nTest Case {i+1}: {test_case}")
        
        input_array = np.array([[test_case[feature] for feature in features]])
        scaled_data = scaler.transform(input_array)
        pca_data = pca.transform(scaled_data)
        
        print(f"Scaled: {[f'{x:.2f}' for x in scaled_data[0]]}")
        print(f"PCA: {[f'{x:.2f}' for x in pca_data[0]]}")
        
        segment = kmeans.predict(pca_data)[0]
        distances = np.linalg.norm(pca_data - kmeans.cluster_centers_, axis=1)
        
        print(f"Predicted: {segment} -> {metadata['segment_names'][segment]}")
        print(f"Distances: {[f'{d:.2f}' for d in distances]}")

if __name__ == "__main__":
    analyze_model_issues()