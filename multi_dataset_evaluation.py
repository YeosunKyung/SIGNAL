import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import cv2
import os
import glob
import time
import warnings
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json

warnings.filterwarnings('ignore')

class MultiDatasetEvaluator:
    """
    다중 데이터셋 평가를 통한 일반화 능력 검증
    - UCF101, HMDB51, KTH 등 다양한 행동 인식 데이터셋
    - Collision prediction 등 추가 task 평가
    """
    
    def __init__(self, lgmd_encoder, base_classifier):
        self.lgmd_encoder = lgmd_encoder
        self.base_classifier = base_classifier
        self.results = {}
        
    def load_ucf101_dataset(self, drive_path="/content/drive/MyDrive/UCF101"):
        """
        UCF101 데이터셋 로드
        """
        print("📹 Loading UCF101 dataset...")
        
        if not os.path.exists(drive_path):
            print(f"❌ UCF101 dataset not found at {drive_path}")
            print("Creating synthetic UCF101-like dataset...")
            return self.create_synthetic_ucf101()
        
        video_data = []
        labels = []
        label_mapping = {}
        
        # Get all action classes
        action_classes = sorted([d for d in os.listdir(drive_path) 
                               if os.path.isdir(os.path.join(drive_path, d))])
        
        for class_idx, action in enumerate(action_classes):
            label_mapping[action] = class_idx
            action_path = os.path.join(drive_path, action)
            
            # Get video files (limit for computational efficiency)
            video_files = glob.glob(os.path.join(action_path, "*.avi")) + \
                         glob.glob(os.path.join(action_path, "*.mp4"))
            
            print(f"Found {len(video_files)} videos for {action}")
            
            # Limit videos per class for faster processing
            max_videos_per_class = 50
            video_files = video_files[:max_videos_per_class]
            
            for video_file in tqdm(video_files, desc=f"Loading {action}"):
                try:
                    frames = self.load_video_frames(video_file, max_frames=30)
                    if len(frames) > 0:
                        video_data.append(frames)
                        labels.append(class_idx)
                except Exception as e:
                    print(f"Error loading {video_file}: {e}")
                    continue
        
        if len(video_data) == 0:
            print("❌ No videos loaded successfully!")
            return self.create_synthetic_ucf101()
        
        print(f"✅ UCF101 loaded: {len(video_data)} videos, {len(set(labels))} classes")
        print(f"📊 Class distribution: {Counter(labels)}")
        
        return video_data, np.array(labels), label_mapping
    
    def load_hmdb51_dataset(self, drive_path="/content/drive/MyDrive/HMDB51"):
        """
        HMDB51 데이터셋 로드
        """
        print("📹 Loading HMDB51 dataset...")
        
        if not os.path.exists(drive_path):
            print(f"❌ HMDB51 dataset not found at {drive_path}")
            print("Creating synthetic HMDB51-like dataset...")
            return self.create_synthetic_hmdb51()
        
        video_data = []
        labels = []
        label_mapping = {}
        
        # HMDB51 has 51 action classes
        action_classes = sorted([d for d in os.listdir(drive_path) 
                               if os.path.isdir(os.path.join(drive_path, d))])
        
        for class_idx, action in enumerate(action_classes):
            label_mapping[action] = class_idx
            action_path = os.path.join(drive_path, action)
            
            video_files = glob.glob(os.path.join(action_path, "*.avi")) + \
                         glob.glob(os.path.join(action_path, "*.mp4"))
            
            print(f"Found {len(video_files)} videos for {action}")
            
            # Limit videos per class
            max_videos_per_class = 30
            video_files = video_files[:max_videos_per_class]
            
            for video_file in tqdm(video_files, desc=f"Loading {action}"):
                try:
                    frames = self.load_video_frames(video_file, max_frames=25)
                    if len(frames) > 0:
                        video_data.append(frames)
                        labels.append(class_idx)
                except Exception as e:
                    print(f"Error loading {video_file}: {e}")
                    continue
        
        if len(video_data) == 0:
            print("❌ No videos loaded successfully!")
            return self.create_synthetic_hmdb51()
        
        print(f"✅ HMDB51 loaded: {len(video_data)} videos, {len(set(labels))} classes")
        print(f"📊 Class distribution: {Counter(labels)}")
        
        return video_data, np.array(labels), label_mapping
    
    def create_synthetic_ucf101(self, num_videos=1000):
        """
        UCF101과 유사한 합성 데이터셋 생성
        """
        print("🎬 Creating synthetic UCF101-like dataset...")
        
        # UCF101 주요 액션 클래스들
        action_classes = [
            'ApplyEyeMakeup', 'ApplyLipstick', 'Archery', 'BabyCrawling', 'BalanceBeam',
            'BandMarching', 'BaseballPitch', 'Basketball', 'BasketballDunk', 'BenchPress',
            'Biking', 'Billiards', 'BlowDryHair', 'BlowingCandles', 'BodyWeightSquats',
            'Bowling', 'BoxingPunchingBag', 'BoxingSpeedBag', 'BreastStroke', 'BrushingTeeth'
        ]
        
        video_data = []
        labels = []
        
        for class_idx, action in enumerate(action_classes):
            videos_per_class = num_videos // len(action_classes)
            
            for _ in range(videos_per_class):
                frames = self.generate_synthetic_action_frames(action, num_frames=30)
                video_data.append(frames)
                labels.append(class_idx)
        
        label_mapping = {action: idx for idx, action in enumerate(action_classes)}
        
        print(f"✅ Synthetic UCF101 created: {len(video_data)} videos, {len(set(labels))} classes")
        return video_data, np.array(labels), label_mapping
    
    def create_synthetic_hmdb51(self, num_videos=800):
        """
        HMDB51과 유사한 합성 데이터셋 생성
        """
        print("🎬 Creating synthetic HMDB51-like dataset...")
        
        # HMDB51 주요 액션 클래스들
        action_classes = [
            'brush_hair', 'catch', 'clap', 'climb_stairs', 'golf', 'jump', 'kick_ball',
            'pick', 'pour', 'pullup', 'push', 'run', 'shoot_ball', 'shoot_bow', 'shoot_gun',
            'sit', 'stand', 'swing_baseball', 'throw', 'walk', 'wave'
        ]
        
        video_data = []
        labels = []
        
        for class_idx, action in enumerate(action_classes):
            videos_per_class = num_videos // len(action_classes)
            
            for _ in range(videos_per_class):
                frames = self.generate_synthetic_action_frames(action, num_frames=25)
                video_data.append(frames)
                labels.append(class_idx)
        
        label_mapping = {action: idx for idx, action in enumerate(action_classes)}
        
        print(f"✅ Synthetic HMDB51 created: {len(video_data)} videos, {len(set(labels))} classes")
        return video_data, np.array(labels), label_mapping
    
    def generate_synthetic_action_frames(self, action, num_frames=30):
        """
        액션별 합성 프레임 생성
        """
        frame_height, frame_width = 120, 160
        
        frames = []
        base_frame = np.random.rand(frame_height, frame_width) * 0.3
        
        # 액션별 모션 패턴 생성
        if 'walk' in action.lower() or 'run' in action.lower():
            # 수평 이동
            motion_speed = 0.8 if 'run' in action.lower() else 0.4
            for frame_idx in range(num_frames):
                frame = base_frame.copy()
                motion = np.sin(frame_idx * motion_speed) * 20
                frame[:, int(frame_width//2 + motion):int(frame_width//2 + motion + 15)] += 0.7
                frames.append(frame)
                
        elif 'jump' in action.lower() or 'climb' in action.lower():
            # 수직 이동
            for frame_idx in range(num_frames):
                frame = base_frame.copy()
                motion = np.sin(frame_idx * 0.5) * 15
                frame[int(frame_height//2 + motion):int(frame_height//2 + motion + 10), :] += 0.6
                frames.append(frame)
                
        elif 'throw' in action.lower() or 'catch' in action.lower():
            # 원형 모션
            for frame_idx in range(num_frames):
                frame = base_frame.copy()
                angle = frame_idx * 0.3
                center_x, center_y = frame_width//2, frame_height//2
                x = int(center_x + np.cos(angle) * 25)
                y = int(center_y + np.sin(angle) * 25)
                frame[max(0, y-8):min(frame_height, y+8), max(0, x-8):min(frame_width, x+8)] += 0.8
                frames.append(frame)
                
        else:
            # 기본 모션
            for frame_idx in range(num_frames):
                frame = base_frame.copy()
                motion = np.sin(frame_idx * 0.4) * 0.3
                frame += motion
                frames.append(frame)
        
        return np.array(frames)
    
    def load_video_frames(self, video_path, max_frames=30):
        """
        비디오 파일에서 프레임 로드
        """
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        while len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 그레이스케일 변환 및 리사이즈
            if len(frame.shape) == 3:
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray_frame = frame
            
            gray_frame = cv2.resize(gray_frame, (160, 120))
            gray_frame = gray_frame.astype(np.float32) / 255.0
            
            frames.append(gray_frame)
        
        cap.release()
        return np.array(frames)
    
    def evaluate_dataset(self, dataset_name, video_data, labels, label_mapping):
        """
        특정 데이터셋에서 모델 평가
        """
        print(f"\n🏆 Evaluating on {dataset_name} dataset")
        print("=" * 60)
        
        start_time = time.time()
        
        # 1. LGMD 특징 추출
        print("🔬 Extracting LGMD features...")
        features = []
        valid_labels = []
        
        for i, (video, label) in enumerate(tqdm(zip(video_data, labels), total=len(video_data))):
            try:
                feature = self.lgmd_encoder.encode(video)
                if len(feature) > 0 and not np.any(np.isnan(feature)):
                    features.append(feature)
                    valid_labels.append(label)
            except Exception as e:
                continue
        
        features = np.array(features)
        valid_labels = np.array(valid_labels)
        
        print(f"✅ Features extracted: {features.shape}")
        
        # 2. 특징 정규화
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # 3. 교차 검증
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        accuracies = []
        
        for fold, (train_idx, test_idx) in enumerate(skf.split(features_scaled, valid_labels), 1):
            print(f"\n🔄 Fold {fold}/3")
            
            X_train = features_scaled[train_idx]
            y_train = valid_labels[train_idx]
            X_test = features_scaled[test_idx]
            y_test = valid_labels[test_idx]
            
            # 분류기 학습 및 예측
            self.base_classifier.fit(X_train, y_train)
            y_pred = self.base_classifier.predict(X_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            accuracies.append(accuracy)
            
            print(f"  📊 Accuracy: {accuracy:.4f}")
        
        # 4. 결과 저장
        mean_accuracy = np.mean(accuracies)
        std_accuracy = np.std(accuracies)
        
        evaluation_time = time.time() - start_time
        
        results = {
            'dataset_name': dataset_name,
            'num_videos': len(video_data),
            'num_classes': len(set(labels)),
            'feature_shape': features.shape,
            'accuracies': accuracies,
            'mean_accuracy': mean_accuracy,
            'std_accuracy': std_accuracy,
            'evaluation_time': evaluation_time,
            'label_mapping': label_mapping
        }
        
        print(f"\n📊 {dataset_name} Results:")
        print(f"  🎯 Mean Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
        print(f"  ⏱️ Evaluation Time: {evaluation_time:.1f}s")
        print(f"  📈 Individual Accuracies: {[f'{acc:.3f}' for acc in accuracies]}")
        
        self.results[dataset_name] = results
        return results
    
    def collision_prediction_task(self, drive_path="/content/drive/MyDrive/collision_dataset"):
        """
        Collision prediction task 평가
        """
        print("\n🚗 Collision Prediction Task Evaluation")
        print("=" * 60)
        
        # 합성 충돌 예측 데이터셋 생성
        print("🎬 Creating collision prediction dataset...")
        
        collision_data = []
        collision_labels = []
        
        # 충돌 시나리오 생성
        for scenario in range(200):
            frames = []
            
            # 초기 상태 (안전)
            for frame_idx in range(20):
                frame = np.random.rand(120, 160) * 0.3
                
                # 차량 위치 (점진적 접근)
                vehicle_x = 80 + frame_idx * 2  # 점진적 이동
                vehicle_y = 60 + np.sin(frame_idx * 0.3) * 10
                
                # 차량 그리기
                frame[int(vehicle_y-5):int(vehicle_y+5), int(vehicle_x-8):int(vehicle_x+8)] += 0.8
                
                # 장애물 (고정 위치)
                obstacle_x, obstacle_y = 150, 60
                frame[int(obstacle_y-3):int(obstacle_y+3), int(obstacle_x-5):int(obstacle_x+5)] += 0.6
                
                frames.append(frame)
            
            collision_data.append(np.array(frames))
            
            # 충돌 여부 판단 (거리 기반)
            final_distance = np.sqrt((vehicle_x - obstacle_x)**2 + (vehicle_y - obstacle_y)**2)
            collision_labels.append(1 if final_distance < 15 else 0)
        
        # 비충돌 시나리오 생성
        for scenario in range(200):
            frames = []
            
            for frame_idx in range(20):
                frame = np.random.rand(120, 160) * 0.3
                
                # 차량이 장애물을 피해 이동
                vehicle_x = 80 + frame_idx * 2
                vehicle_y = 30 + np.sin(frame_idx * 0.5) * 20  # 더 큰 진폭으로 회피
                
                frame[int(vehicle_y-5):int(vehicle_y+5), int(vehicle_x-8):int(vehicle_x+8)] += 0.8
                
                # 장애물
                obstacle_x, obstacle_y = 150, 60
                frame[int(obstacle_y-3):int(obstacle_y+3), int(obstacle_x-5):int(obstacle_x+5)] += 0.6
                
                frames.append(frame)
            
            collision_data.append(np.array(frames))
            collision_labels.append(0)  # 비충돌
        
        collision_data = np.array(collision_data)
        collision_labels = np.array(collision_labels)
        
        print(f"✅ Collision dataset created: {len(collision_data)} scenarios")
        print(f"📊 Collision rate: {np.mean(collision_labels):.2f}")
        
        # 충돌 예측 평가
        collision_results = self.evaluate_dataset("Collision_Prediction", 
                                                collision_data, collision_labels, 
                                                {'no_collision': 0, 'collision': 1})
        
        return collision_results
    
    def cross_dataset_generalization(self):
        """
        크로스 데이터셋 일반화 능력 평가
        """
        print("\n🌐 Cross-Dataset Generalization Analysis")
        print("=" * 60)
        
        # KTH에서 학습, UCF101에서 테스트
        print("🔄 Training on KTH, Testing on UCF101...")
        
        # KTH 데이터 로드 (학습용)
        kth_data, kth_labels, _ = self.load_kth_dataset()
        
        # UCF101 데이터 로드 (테스트용)
        ucf_data, ucf_labels, _ = self.load_ucf101_dataset()
        
        # KTH에서 특징 추출 및 학습
        print("🔬 Extracting features from KTH...")
        kth_features = []
        kth_valid_labels = []
        
        for video, label in tqdm(zip(kth_data, kth_labels), total=len(kth_data)):
            try:
                feature = self.lgmd_encoder.encode(video)
                if len(feature) > 0:
                    kth_features.append(feature)
                    kth_valid_labels.append(label)
            except:
                continue
        
        kth_features = np.array(kth_features)
        kth_valid_labels = np.array(kth_valid_labels)
        
        # UCF101에서 특징 추출
        print("🔬 Extracting features from UCF101...")
        ucf_features = []
        ucf_valid_labels = []
        
        for video, label in tqdm(zip(ucf_data, ucf_labels), total=len(ucf_data)):
            try:
                feature = self.lgmd_encoder.encode(video)
                if len(feature) > 0:
                    ucf_features.append(feature)
                    ucf_valid_labels.append(label)
            except:
                continue
        
        ucf_features = np.array(ucf_features)
        ucf_valid_labels = np.array(ucf_valid_labels)
        
        # 특징 정규화
        scaler = StandardScaler()
        kth_features_scaled = scaler.fit_transform(kth_features)
        ucf_features_scaled = scaler.transform(ucf_features)
        
        # KTH에서 학습, UCF101에서 테스트
        self.base_classifier.fit(kth_features_scaled, kth_valid_labels)
        ucf_predictions = self.base_classifier.predict(ucf_features_scaled)
        
        # 결과 계산
        cross_dataset_accuracy = accuracy_score(ucf_valid_labels, ucf_predictions)
        
        cross_results = {
            'source_dataset': 'KTH',
            'target_dataset': 'UCF101',
            'cross_dataset_accuracy': cross_dataset_accuracy,
            'source_samples': len(kth_features),
            'target_samples': len(ucf_features)
        }
        
        print(f"📊 Cross-dataset accuracy (KTH→UCF101): {cross_dataset_accuracy:.4f}")
        
        self.results['cross_dataset'] = cross_results
        return cross_results
    
    def generate_generalization_plots(self):
        """
        일반화 능력 시각화
        """
        if not self.results:
            print("❌ No results available. Run evaluation first.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Dataset performance comparison
        dataset_names = []
        accuracies = []
        stds = []
        
        for dataset_name, result in self.results.items():
            if 'mean_accuracy' in result:
                dataset_names.append(dataset_name)
                accuracies.append(result['mean_accuracy'])
                stds.append(result['std_accuracy'])
        
        if dataset_names:
            axes[0, 0].bar(dataset_names, accuracies, yerr=stds, capsize=5, alpha=0.8)
            axes[0, 0].set_title('Performance Across Datasets')
            axes[0, 0].set_ylabel('Accuracy')
            axes[0, 0].tick_params(axis='x', rotation=45)
            axes[0, 0].set_ylim(0, 1)
        
        # 2. Cross-dataset generalization
        if 'cross_dataset' in self.results:
            cross_result = self.results['cross_dataset']
            axes[0, 1].bar(['Cross-Dataset'], [cross_result['cross_dataset_accuracy']], alpha=0.8)
            axes[0, 1].set_title('Cross-Dataset Generalization')
            axes[0, 1].set_ylabel('Accuracy')
            axes[0, 1].set_ylim(0, 1)
        
        # 3. Feature distribution comparison
        if len(self.results) > 1:
            feature_shapes = []
            dataset_names_plot = []
            
            for dataset_name, result in self.results.items():
                if 'feature_shape' in result:
                    feature_shapes.append(result['feature_shape'][1])
                    dataset_names_plot.append(dataset_name)
            
            if feature_shapes:
                axes[1, 0].bar(dataset_names_plot, feature_shapes, alpha=0.8)
                axes[1, 0].set_title('Feature Dimensions Across Datasets')
                axes[1, 0].set_ylabel('Feature Dimension')
                axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. Evaluation time comparison
        eval_times = []
        dataset_names_time = []
        
        for dataset_name, result in self.results.items():
            if 'evaluation_time' in result:
                eval_times.append(result['evaluation_time'])
                dataset_names_time.append(dataset_name)
        
        if eval_times:
            axes[1, 1].bar(dataset_names_time, eval_times, alpha=0.8)
            axes[1, 1].set_title('Evaluation Time Across Datasets')
            axes[1, 1].set_ylabel('Time (seconds)')
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('/content/drive/MyDrive/generalization_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def save_generalization_results(self, save_path="/content/drive/MyDrive/generalization_results.json"):
        """
        일반화 평가 결과 저장
        """
        try:
            # Convert numpy types to Python types for JSON serialization
            results_serializable = {}
            for key, value in self.results.items():
                if isinstance(value, dict):
                    results_serializable[key] = {}
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, np.ndarray):
                            results_serializable[key][sub_key] = sub_value.tolist()
                        elif isinstance(sub_value, np.integer):
                            results_serializable[key][sub_key] = int(sub_value)
                        elif isinstance(sub_value, np.floating):
                            results_serializable[key][sub_key] = float(sub_value)
                        else:
                            results_serializable[key][sub_key] = sub_value
                else:
                    results_serializable[key] = value
            
            with open(save_path, 'w') as f:
                json.dump(results_serializable, f, indent=2)
            
            print(f"✅ Generalization results saved to: {save_path}")
            return True
            
        except Exception as e:
            print(f"❌ Error saving results: {e}")
            return False

# Usage example
def run_multi_dataset_evaluation(lgmd_encoder, base_classifier):
    """
    전체 다중 데이터셋 평가 실행
    """
    evaluator = MultiDatasetEvaluator(lgmd_encoder, base_classifier)
    
    # 1. UCF101 평가
    ucf_data, ucf_labels, ucf_mapping = evaluator.load_ucf101_dataset()
    ucf_results = evaluator.evaluate_dataset("UCF101", ucf_data, ucf_labels, ucf_mapping)
    
    # 2. HMDB51 평가
    hmdb_data, hmdb_labels, hmdb_mapping = evaluator.load_hmdb51_dataset()
    hmdb_results = evaluator.evaluate_dataset("HMDB51", hmdb_data, hmdb_labels, hmdb_mapping)
    
    # 3. Collision prediction task
    collision_results = evaluator.collision_prediction_task()
    
    # 4. Cross-dataset generalization
    cross_results = evaluator.cross_dataset_generalization()
    
    # 5. 결과 시각화 및 저장
    evaluator.generate_generalization_plots()
    evaluator.save_generalization_results()
    
    return evaluator.results 