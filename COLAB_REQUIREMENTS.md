# Colab 환경 필수 요구사항

## 🚨 절대 규칙 (MANDATORY RULES)

### 1. **환경 설정**
- **모든 코드는 Google Colab에서 실행**되어야 합니다
- **Standalone 실행 가능**: 단일 파일로 모든 기능 포함
- **외부 의존성 최소화**: 필요한 모든 함수/클래스를 파일 내 정의

### 2. **실제 결과만 출력 (CRITICAL)**
```python
# ❌ 절대 금지 - 임의의 결과 생성
print("Accuracy: 82.5%")  # 하드코딩
accuracy = np.random.uniform(0.8, 0.85)  # 랜덤 생성
results = {'accuracy': 0.825, 'lgmd': 0.15}  # 고정값

# ✅ 필수 - 실제 계산 결과만
accuracy = model.score(X_test, y_test)  # 실제 모델 평가
print(f"Accuracy: {accuracy:.1%}")  # 실제 계산값 출력

# 모든 결과는 반드시 실제 데이터와 모델에서 계산되어야 함
assert isinstance(accuracy, (float, np.float32, np.float64)), "Must be computed value"
assert 0 <= accuracy <= 1, "Invalid accuracy range"
```

### 3. **KTH 데이터셋 경로**
```python
# Colab에서 Google Drive 마운트
try:
    from google.colab import drive
    drive.mount('/content/drive')
    IN_COLAB = True
except:
    IN_COLAB = False

# KTH 데이터셋 경로 (절대 변경하지 마세요)
KTH_DATASET_PATH = '/content/drive/MyDrive/KTH_dataset'

# 데이터셋 구조
# /content/drive/MyDrive/KTH_dataset/
#   ├── boxing/
#   ├── handclapping/
#   ├── handwaving/
#   ├── jogging/
#   ├── running/
#   └── walking/
```

### 4. **하드코딩 금지**
```python
# ❌ 절대 하지 마세요:
accuracy = 0.825  # 하드코딩된 정확도
lgmd_contribution = 0.44  # 하드코딩된 기여도

# ✅ 항상 이렇게 하세요:
accuracy = np.mean(predictions == y_true)  # 실제 계산
lgmd_contribution = calculate_actual_contribution(feature_importances)
```

### 5. **결과 검증**
```python
# 모든 결과는 실제 계산에서 나와야 함
assert isinstance(accuracy, float), "Accuracy must be computed, not hardcoded"
assert 0 <= accuracy <= 1, "Invalid accuracy range"

# 실패 시 정직하게 보고
if accuracy < expected_accuracy:
    print(f"❌ Accuracy {accuracy:.1%} is below expected {expected_accuracy:.1%}")
    # 절대 성공으로 위장하지 않음
```

## 📋 표준 Colab 템플릿

```python
#!/usr/bin/env python3
"""
[모듈 이름]
Google Colab에서 실행 가능한 standalone 코드
"""

import numpy as np
import cv2
import os
from pathlib import Path

# Colab 환경 확인 및 Drive 마운트
try:
    from google.colab import drive
    drive.mount('/content/drive')
    IN_COLAB = True
    print("✅ Running in Google Colab")
except:
    IN_COLAB = False
    print("⚠️ Not in Colab - using local paths")

# KTH 데이터셋 경로 설정
if IN_COLAB:
    KTH_PATH = '/content/drive/MyDrive/KTH_dataset'
else:
    # 로컬 테스트용 (Colab이 아닐 때만)
    KTH_PATH = './KTH_dataset'

# 경로 확인
if not os.path.exists(KTH_PATH):
    print(f"❌ KTH dataset not found at {KTH_PATH}")
    print("Please ensure the dataset is in Google Drive at:")
    print("/content/drive/MyDrive/KTH_dataset/")
else:
    print(f"✅ KTH dataset found at {KTH_PATH}")
    
# 사용 가능한 액션 확인
ACTIONS = ['boxing', 'handclapping', 'handwaving', 'jogging', 'running', 'walking']
available_actions = []

for action in ACTIONS:
    action_path = os.path.join(KTH_PATH, action)
    if os.path.exists(action_path):
        n_videos = len(list(Path(action_path).glob('*.avi')))
        print(f"  - {action}: {n_videos} videos")
        available_actions.append(action)
```

## 🔧 데이터 로딩 표준 함수

```python
def load_kth_subset(n_per_class=5, seed=42):
    """
    KTH 데이터셋의 일부를 로드
    
    Args:
        n_per_class: 각 클래스당 비디오 수
        seed: 랜덤 시드
    
    Returns:
        X: 비디오 데이터
        y: 레이블
        video_paths: 비디오 경로
    """
    np.random.seed(seed)
    
    X = []
    y = []
    video_paths = []
    
    for action_idx, action in enumerate(ACTIONS):
        action_path = Path(KTH_PATH) / action
        if not action_path.exists():
            print(f"⚠️ Skipping {action} - not found")
            continue
            
        videos = list(action_path.glob('*.avi'))
        if len(videos) < n_per_class:
            print(f"⚠️ {action} has only {len(videos)} videos")
            selected = videos
        else:
            selected = np.random.choice(videos, n_per_class, replace=False)
        
        for video_path in selected:
            # 실제 비디오 로드
            frames = load_video_frames(str(video_path))
            if frames is not None:
                X.append(frames)
                y.append(action_idx)
                video_paths.append(str(video_path))
    
    return X, np.array(y), video_paths

def load_video_frames(video_path, max_frames=100, size=(96, 96)):
    """비디오 프레임 로드"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    count = 0
    while count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Grayscale conversion and resize
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, size)
        frames.append(resized)
        count += 1
    
    cap.release()
    
    if len(frames) < 10:  # 너무 짧은 비디오 제외
        return None
    
    return np.array(frames)
```

## ⚠️ 주의사항

1. **모든 print 문은 실제 계산 결과여야 함**
   ```python
   # ❌ 금지
   print("Accuracy: 82.5%")  # 하드코딩
   
   # ✅ 허용
   print(f"Accuracy: {actual_accuracy:.1%}")  # 실제 계산값
   ```

2. **정확한 메트릭 보고**
   ```python
   if bio_inspired_contribution < threshold:
       print(f"Bio-inspired contribution: {bio_inspired_contribution:.1%}")
       # 모든 기여도는 실제 계산 기반
   ```

3. **재현성 보장**
   ```python
   # 항상 시드 설정
   np.random.seed(42)
   
   # 결과 저장
   results = {
       'accuracy': actual_accuracy,
       'lgmd_contribution': actual_contribution,
       'timestamp': datetime.now().isoformat()
   }
   ```

## 🚀 실행 방법

1. Google Colab에서 새 노트북 생성
2. Google Drive 마운트
3. KTH 데이터셋이 `/content/drive/MyDrive/KTH_dataset`에 있는지 확인
4. 코드 실행

## 📝 체크리스트

코드 작성 시 항상 확인:
- [ ] Colab에서 Drive 마운트 코드 포함
- [ ] KTH 경로가 `/content/drive/MyDrive/KTH_dataset`
- [ ] 모든 결과가 실제 계산에서 나옴
- [ ] 하드코딩된 정확도/기여도 없음
- [ ] 실패 시 정직하게 보고
- [ ] Standalone으로 실행 가능
- [ ] 필요한 모든 함수/클래스 포함

---

**이 규칙은 모든 SIGNAL 프로젝트 코드에 적용됩니다.**

## ⛔ 절대 금지 사항

1. **임의의 결과 생성 금지**
   - `np.random.uniform()`, `random.randint()` 등으로 정확도 생성 금지
   - 고정된 숫자를 결과로 출력 금지
   - 실제 계산 없이 "성공" 메시지 출력 금지

2. **정확한 결과 보고**
   - 모든 결과는 실제 계산값 기반
   - 정확한 성능 메트릭 출력
   - 투명한 기여도 분석 제공

3. **데이터 조작 금지**
   - 결과를 좋게 보이기 위한 데이터 필터링 금지
   - 특정 결과를 위한 샘플 선택 금지
   - 평가 데이터 조작 금지

**위반 시 코드는 무효 처리됩니다.**