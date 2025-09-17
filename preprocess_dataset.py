import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
import argparse

def preprocess_dataset(input_csv, output_file, k_features=30, test_size=0.2, random_state=42):
    df = pd.read_csv(input_csv)
    # Xóa các cột không cần thiết

    df = df.drop(columns=['Flow ID'])
    df = df.drop(columns=['Src IP'])
    df = df.drop(columns=['Dst IP'])

    # Mã hóa các cột object (trừ Label)
    object_cols = df.select_dtypes(include=['object']).columns.tolist()
    if 'Label' in object_cols:
        object_cols.remove('Label')
    for col in object_cols:
        df[col] = df[col].astype('category').cat.codes

    # Sử dụng LabelEncoder để chuyển đổi cột 'Label' thành nhãn số (0-6)
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['Label'])
    print("\n[Label Encoding] Các lớp và mã hóa tương ứng:")
    for i, class_name in enumerate(le.classes_):
        print(f"  - Lớp '{class_name}': {i}")

    # Chuẩn hóa và chọn feature
    # Sử dụng cột 'label' mới làm mục tiêu (y)
    X = df.drop(columns=['label']).values
    y = df['label'].values
    X = np.nan_to_num(X)
    X = np.clip(X, -1e10, 1e10)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    selector = SelectKBest(f_classif, k=min(k_features, X.shape[1]))
    X_selected = selector.fit_transform(X_scaled, y)
    feature_names = df.drop(columns=['label']).columns
    selected_features = feature_names[selector.get_support(indices=True)]
    print('\n[Feature Selection] Các feature được chọn:')
    for i, feat in enumerate(selected_features):
        print(f'  {i+1}. {feat}')

    # Lưu ra file
    processed_df = pd.DataFrame(X_selected, columns=selected_features)
    processed_df['label'] = df['label'] # Lưu cột label đa lớp
    processed_df.to_csv(output_file, index=False)
    print(f"\nProcessed data saved to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess IoTDIAD dataset.")
    parser.add_argument('--input', type=str, default='IoTDIAD_sum.csv', help='Input CSV file')
    parser.add_argument('--out', type=str, default='IoTDIAD_processed.csv', help='Output processed CSV')
    parser.add_argument('--k_features', type=int, default=30, help='Number of features to select')
    parser.add_argument('--random_state', type=int, default=42, help='Random state')
    args = parser.parse_args()
    preprocess_dataset(args.input, args.out, args.k_features, args.random_state)