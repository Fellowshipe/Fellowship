import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from imblearn.combine import SMOTEENN  # 수정: SMOTE → SMOTEENN
import numpy as np
import pandas as pd
import optuna

# Load the uploaded CSV files to inspect their structure and contents
text_image_data = pd.read_csv('../../data/text_image.csv')
class_data = pd.read_csv('../../data/class_data.csv')

# Merge the two datasets on the 'id' column to create a unified dataset for training
merged_data = pd.merge(text_image_data, class_data, on="id")

print(f"결측치 제거 전: {merged_data.shape}")

# 결측치 제거
merged_data = merged_data.dropna()
print(f"결측치 제거 후: {merged_data.shape}")

# 타겟 라벨을 숫자로 변환 (True/False -> 1/0)
le = LabelEncoder()
merged_data['is_fraud'] = le.fit_transform(merged_data['is_fraud'])

# 특성과 타겟 분리
X = merged_data.drop(columns=['id', 'is_fraud']).values
y = merged_data['is_fraud'].values

# 데이터 표준화
scaler = StandardScaler()
X = scaler.fit_transform(X)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# FCNN 모델 클래스 정의
class FCNN(nn.Module):
    def __init__(self, input_dim, num_layers, hidden_size, dropout_rate):
        super(FCNN, self).__init__()
        layers = []
        in_features = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(nn.LeakyReLU(negative_slope=0.01))
            layers.append(nn.Dropout(dropout_rate))
            in_features = hidden_size
        layers.append(nn.Linear(hidden_size, 1))
        layers.append(nn.Sigmoid())
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# 결과 저장용 리스트
results = []

# Objective 함수 정의
def objective(trial):
    # 하이퍼파라미터 설정
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    dropout_rate = trial.suggest_uniform('dropout_rate', 0.2, 0.5)
    num_layers = trial.suggest_int('num_layers', 3, 10)
    hidden_size = trial.suggest_int('hidden_size', 256, 512)
    num_epochs = trial.suggest_int('num_epochs', 40, 100)
    smote_ratio = trial.suggest_uniform('smote_ratio', 0.1, 1.0)  # SMOTE 비율 파라미터

    # 교차 검증 설정 (k=4, test_size=0.25)
    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    f1_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"Trial {trial.number}, Fold {fold + 1}, Parameters: lr={learning_rate:.5f}, dropout={dropout_rate}, layers={num_layers}, hidden={hidden_size}, smote_ratio={smote_ratio:.2f}")
        
        # 학습/검증 데이터 분할
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # 수정: SMOTE → SMOTEENN
        smote_enn = SMOTEENN(sampling_strategy=smote_ratio, random_state=42)
        X_train_resampled, y_train_resampled = smote_enn.fit_resample(X_train, y_train)

        # Tensor 변환
        X_train_tensor = torch.tensor(X_train_resampled, dtype=torch.float32).to(device)
        y_train_tensor = torch.tensor(y_train_resampled, dtype=torch.float32).to(device)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)

        # DataLoader 설정
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        # 모델, 손실 함수, 최적화기 설정
        input_dim = X_train_tensor.shape[1]
        model = FCNN(input_dim, num_layers, hidden_size, dropout_rate).to(device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # 학습 진행
        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = model(X_batch).squeeze()
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            avg_loss = epoch_loss / len(train_loader)
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

        # 검증 데이터 평가
        model.eval()
        y_val_pred = model(X_val_tensor).squeeze().cpu().detach().numpy()
        y_val_pred_binary = (y_val_pred > 0.5).astype(int)

        # 각 Trial, Fold의 성능 지표 계산
        precision = precision_score(y_val, y_val_pred_binary)
        recall = recall_score(y_val, y_val_pred_binary)
        f1 = f1_score(y_val, y_val_pred_binary)
        roc_auc = roc_auc_score(y_val, y_val_pred)
        cm = confusion_matrix(y_val, y_val_pred_binary)

        # 성능 지표 출력
        print(f"Performance Metrics for Trial {trial.number}, Fold {fold + 1}:")
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1)
        print("ROC AUC Score:", roc_auc)
        print("Confusion Matrix:\n", cm)
        print("="*40)

        # 결과 저장
        results.append({
            "trial": trial.number,
            "fold": fold + 1,
            "learning_rate": learning_rate,
            "dropout_rate": dropout_rate,
            "num_layers": num_layers,
            "hidden_size": hidden_size,
            "num_epochs": num_epochs,
            "smote_ratio": smote_ratio,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": roc_auc,
            "confusion_matrix": cm.tolist()  # Confusion matrix를 리스트로 변환하여 저장
        })

        # F1 Score 저장
        f1_scores.append(f1)

    # 평균 검증 F1 스코어 반환 (최적화 목표)
    return np.mean(f1_scores)

# Optuna 최적화 수행
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=10)

# 최적의 하이퍼파라미터로 테스트 성능 평가
print("Best hyperparameters:", study.best_params)

# 결과를 DataFrame으로 저장하고 CSV 파일로 출력
results_df = pd.DataFrame(results)
results_df.to_csv("optuna_trial_results.csv", index=False)

# 저장된 결과 확인
print("\nResults saved to 'optuna_trial_results.csv'.")