import pandas as pd
from itertools import product
from sklearn.model_selection import StratifiedKFold, train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, roc_auc_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# 1. 데이터 로드
data = pd.read_csv('../../../tabular_text_image_data.csv')  # 실제 경로로 변경

# 2. 결측치 제거
data = data.dropna()

# 3. 데이터 타입 변환
data = data.astype({'is_find': 'bool'})  # 'is_find' 컬럼을 bool 타입으로 변환

# 4. 특징 데이터(X)와 라벨(y) 분리
X = data.drop(columns='is_fraud')  # 'is_fraud'는 타겟 컬럼명, 실제 데이터셋에 맞게 변경
y = data['is_fraud']

# 5. 라벨 인코딩 (필요한 경우)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# 6. Stratified train-test split (test_size = 0.3)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, stratify=y_encoded, random_state=42)

# 세 가지 경우의 입력 데이터 정의
feature_sets = {
    'tabular_text': X.filter(regex='^(?!image_embedding_).*').values,
    'tabular_image': X.filter(regex='^(?!text_embedding_).*').values,
    'tabular_text_image': X.values
}

# 결과 저장을 위한 변수 초기화
best_results = {}
all_results = []

# 7. 하이퍼파라미터 조합 설정 (수동으로 조합을 설정)
param_grid = {
    'n_estimators': [500, 1000],
    'max_depth': [5, 12],
    'learning_rate': [0.01, 0.05],
    'min_child_weight': [5, 10],
    'colsample_bytree': [0.7, 0.85],
    'scale_pos_weight': [10, 30],
    'gamma': [0.1, 0.3]
}

# 8. 모든 하이퍼파라미터 조합 생성
combinations = list(product(
    param_grid['n_estimators'],
    param_grid['max_depth'],
    param_grid['learning_rate'],
    param_grid['colsample_bytree'],
    param_grid['scale_pos_weight'],
    param_grid['min_child_weight'],
    param_grid['gamma']
    )
)

# 각 경우별 평가 및 성능 저장
for feature_type, X_features in tqdm(feature_sets.items()):
    best_f1 = -1
    best_params = None
    best_metrics = {}

    # 9. 하이퍼파라미터 조합별 학습 및 평가
    for params in tqdm(combinations, desc=f"Evaluating {feature_type} combinations"):
        n_estimators, max_depth, learning_rate, colsample_bytree, min_child_weight, scale_pos_weight, gamma = params

        # 파이프라인 정의 (StandardScaler + PCA + XGBoost 모델)
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                colsample_bytree=colsample_bytree,
                min_child_weight=min_child_weight,
                scale_pos_weight=scale_pos_weight,
                gamma=gamma,
                random_state=42
            ))
        ])

        # SMOTE 적용 (훈련 데이터에만)
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

        # 모델 학습
        pipeline.fit(X_train_resampled, y_train_resampled)

        # 테스트 데이터셋으로 예측
        y_pred = pipeline.predict(X_test)

        # f1 score 및 다른 성능 지표 계산
        f1 = f1_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)

        # 최고 f1 score를 기록
        if f1 > best_f1:
            best_f1 = f1
            best_params = params
            best_metrics = {
                'f1_score': f1,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'roc_auc': roc_auc,
                'confusion_matrix': conf_matrix
            }

        print(f"Params: {params} | F1 Score: {f1:.4f} | Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | ROC-AUC: {roc_auc:.4f}")
        print(f"Confusion Matrix:\n{conf_matrix}")

        # 각 조합의 성능 저장
        all_results.append({
            'feature_type': feature_type,
            'params': params,
            'f1_score': f1,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'roc_auc': roc_auc,
            'confusion_matrix': conf_matrix
        })

    # Best result 저장
    best_results[feature_type] = {'params': best_params, 'metrics': best_metrics}

# 결과 저장
results_df = pd.DataFrame(all_results)
results_df.to_csv('model_performance_results.csv', index=False)

# 성능 시각화
for feature_type, result in best_results.items():
    best_metrics = result['metrics']
    y_pred_best = pipeline.predict(X_features[len(y_train):])
    fpr, tpr, _ = roc_curve(y_test, y_pred_best)
    roc_auc = auc(fpr, tpr)

    # Confusion matrix
    plt.figure()
    plt.matshow(best_metrics['confusion_matrix'], cmap='coolwarm')
    plt.title(f"Confusion Matrix - {feature_type}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.colorbar()
    plt.show()

    # ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='blue', label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    plt.title(f"ROC Curve - {feature_type}")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.show()