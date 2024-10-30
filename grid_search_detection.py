import pandas as pd
from itertools import product
from sklearn.model_selection import StratifiedKFold, train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
import numpy as np
from tqdm import tqdm

# 1. 데이터 로드
data = pd.read_csv('../tabular_text_image_data.csv')  # 실제 경로로 변경

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

# 7. 하이퍼파라미터 조합 설정 (수동으로 조합을 설정)
param_grid = {
    'n_estimators': [100, 200, 500],
    'max_depth': [5, 10, 20],
    'learning_rate': [0.01, 0.03, 0.1],
    'colsample_bytree': [0.7, 0.85, 1.0]
}

# 8. 모든 하이퍼파라미터 조합 생성
combinations = list(product(
    param_grid['n_estimators'],
    param_grid['max_depth'],
    param_grid['learning_rate'],
    param_grid['colsample_bytree']
    )
)

# 최고 성능 저장을 위한 변수 초기화
best_f1 = -1
best_params = None
best_metrics = {}

# 9. 하이퍼파라미터 조합별 학습 및 평가
for params in tqdm(combinations, desc="Evaluating hyperparameter combinations"):
    n_estimators, max_depth, learning_rate, colsample_bytree = params

    # 파이프라인 정의 (StandardScaler + PCA + XGBoost 모델)
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            colsample_bytree=colsample_bytree,
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

# 최고 성능 출력
print("\nBest Parameters and Performance:")
print(f"Best Params: {best_params}")
print(f"Best F1 Score: {best_metrics['f1_score']:.4f}")
print(f"Best Accuracy: {best_metrics['accuracy']:.4f}")
print(f"Best Precision: {best_metrics['precision']:.4f}")
print(f"Best Recall: {best_metrics['recall']:.4f}")
print(f"Best ROC-AUC: {best_metrics['roc_auc']:.4f}")
print(f"Best Confusion Matrix:\n{best_metrics['confusion_matrix']}")