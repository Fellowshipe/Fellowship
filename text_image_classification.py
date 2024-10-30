import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE

# Load the uploaded CSV files to inspect their structure and contents
text_image_data = pd.read_csv('processed_data.csv')
class_data = pd.read_csv('data/class_data.csv')


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

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# SMOTE 적용을 통한 학습 데이터 오버샘플링 (클래스 불균형 해결)
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Tensor로 변환 (PyTorch 학습을 위한 준비)
X_train_tensor = torch.tensor(X_train_resampled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_resampled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# GPU 사용 가능 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 데이터를 GPU로 이동 (GPU 사용 가능할 때만)
X_train_tensor = X_train_tensor.to(device)
y_train_tensor = y_train_tensor.to(device)
X_test_tensor = X_test_tensor.to(device)
y_test_tensor = y_test_tensor.to(device)

# DataLoader 설정 (배치 단위 학습을 위한 데이터로더 생성)
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# FCNN 모델 정의
class FCNN(nn.Module):
    def __init__(self, input_dim):
        super(FCNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.relu(self.fc4(x))
        x = torch.sigmoid(self.fc5(x))
        return x

# 모델, 손실 함수, 최적화기 설정
input_dim = X_train_tensor.shape[1]
model = FCNN(input_dim).to(device)  # 모델을 GPU로 이동
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 모델 학습 (학습 진행 상황 출력)
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0  # 각 epoch의 손실 값 초기화
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)  # 배치를 GPU로 이동
        optimizer.zero_grad()
        outputs = model(X_batch).squeeze()
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()  # 배치 손실 값을 누적

    # Epoch 마다 평균 손실 출력
    avg_loss = epoch_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

# 모델 평가
model.eval()
y_pred = []
y_pred_proba = []
with torch.no_grad():
    for X_batch, _ in test_loader:
        X_batch = X_batch.to(device)  # 배치를 GPU로 이동
        outputs = model(X_batch).squeeze()
        y_pred_proba.extend(outputs.cpu().numpy())  # GPU에서 CPU로 이동 후 Numpy 배열로 변환
        y_pred.extend((outputs > 0.5).cpu().numpy())  # 동일한 과정으로 이진 분류 결과

# 성능 지표 계산
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

# 결과 출력
print("Final Evaluation Metrics:")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("ROC AUC Score:", roc_auc)