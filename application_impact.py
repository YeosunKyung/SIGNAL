import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import time
import json
import warnings
from collections import defaultdict
import cv2

warnings.filterwarnings('ignore')

class ApplicationImpactAnalysis:
    """
    실질적 응용 사례 및 impact 분석
    - 초저대역폭 semantic communication
    - Neuromorphic hardware 적용
    - Edge computing 응용
    """
    
    def __init__(self, lgmd_encoder, lgmd_classifier):
        self.lgmd_encoder = lgmd_encoder
        self.lgmd_classifier = lgmd_classifier
        self.application_results = {}
        
    def ultra_low_bandwidth_semantic_communication(self, features, labels):
        """
        초저대역폭 semantic communication 시연
        """
        print("📡 Ultra-Low Bandwidth Semantic Communication Demo")
        print("=" * 60)
        
        # 1. Traditional vs Semantic Communication Comparison
        print("🔬 Comparing Traditional vs Semantic Communication...")
        
        # Traditional approach: Send raw features
        traditional_bandwidth = features.shape[1] * 32  # 32 bits per feature
        traditional_accuracy = self.evaluate_communication_accuracy(features, labels)
        
        # Semantic approach: Send only essential information
        semantic_features = self.extract_semantic_features(features, labels)
        semantic_bandwidth = semantic_features.shape[1] * 32
        semantic_accuracy = self.evaluate_communication_accuracy(semantic_features, labels)
        
        # 2. Bandwidth Compression Analysis
        compression_ratio = traditional_bandwidth / semantic_bandwidth
        accuracy_preservation = semantic_accuracy / traditional_accuracy
        
        # 3. Real-world Communication Scenarios
        scenarios = {
            'Autonomous_Driving': {
                'latency_requirement': 10,  # ms
                'bandwidth_limit': 100,     # kbps
                'critical_actions': ['collision_avoidance', 'lane_change', 'emergency_brake']
            },
            'IoT_Sensor_Network': {
                'latency_requirement': 100,  # ms
                'bandwidth_limit': 10,       # kbps
                'critical_actions': ['anomaly_detection', 'status_report', 'alert']
            },
            'Drone_Communication': {
                'latency_requirement': 50,   # ms
                'bandwidth_limit': 50,       # kbps
                'critical_actions': ['obstacle_avoidance', 'path_planning', 'emergency_landing']
            }
        }
        
        scenario_results = {}
        
        for scenario_name, requirements in scenarios.items():
            print(f"\n📊 Analyzing {scenario_name} scenario...")
            
            # Check if semantic communication meets requirements
            meets_bandwidth = semantic_bandwidth <= requirements['bandwidth_limit'] * 1000  # Convert to bps
            meets_latency = self.estimate_processing_latency(semantic_features) <= requirements['latency_requirement']
            
            # Calculate reliability score
            reliability_score = self.calculate_reliability_score(semantic_accuracy, meets_bandwidth, meets_latency)
            
            scenario_results[scenario_name] = {
                'meets_bandwidth': meets_bandwidth,
                'meets_latency': meets_latency,
                'reliability_score': reliability_score,
                'bandwidth_usage': semantic_bandwidth / 1000,  # kbps
                'estimated_latency': self.estimate_processing_latency(semantic_features)
            }
            
            print(f"  📡 Bandwidth: {semantic_bandwidth/1000:.1f} kbps (Limit: {requirements['bandwidth_limit']} kbps)")
            print(f"  ⏱️ Latency: {self.estimate_processing_latency(semantic_features):.1f} ms (Limit: {requirements['latency_requirement']} ms)")
            print(f"  🎯 Reliability: {reliability_score:.2f}")
        
        # 4. Energy Efficiency Analysis
        energy_efficiency = self.analyze_energy_efficiency(traditional_bandwidth, semantic_bandwidth)
        
        results = {
            'traditional_bandwidth': traditional_bandwidth,
            'semantic_bandwidth': semantic_bandwidth,
            'compression_ratio': compression_ratio,
            'traditional_accuracy': traditional_accuracy,
            'semantic_accuracy': semantic_accuracy,
            'accuracy_preservation': accuracy_preservation,
            'scenario_results': scenario_results,
            'energy_efficiency': energy_efficiency
        }
        
        print(f"\n📊 Overall Results:")
        print(f"  📡 Compression Ratio: {compression_ratio:.1f}x")
        print(f"  🎯 Accuracy Preservation: {accuracy_preservation:.2f}")
        print(f"  ⚡ Energy Efficiency: {energy_efficiency:.1f}x improvement")
        
        self.application_results['semantic_communication'] = results
        return results
    
    def extract_semantic_features(self, features, labels):
        """
        의미론적 특징 추출 (차원 축소)
        """
        from sklearn.decomposition import PCA
        from sklearn.feature_selection import SelectKBest, f_classif
        
        # 1. Feature selection based on class separability
        selector = SelectKBest(score_func=f_classif, k=min(50, features.shape[1]//2))
        selected_features = selector.fit_transform(features, labels)
        
        # 2. PCA for further compression
        pca = PCA(n_components=min(20, selected_features.shape[1]))
        semantic_features = pca.fit_transform(selected_features)
        
        # 3. Quantization for ultra-low bandwidth
        semantic_features = np.round(semantic_features * 100) / 100  # 2 decimal places
        
        return semantic_features
    
    def evaluate_communication_accuracy(self, features, labels):
        """
        통신 정확도 평가
        """
        from sklearn.model_selection import cross_val_score
        from sklearn.ensemble import RandomForestClassifier
        
        classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        scores = cross_val_score(classifier, features, labels, cv=3)
        
        return np.mean(scores)
    
    def estimate_processing_latency(self, features):
        """
        처리 지연 시간 추정
        """
        # Simplified latency estimation
        # In practice, this would be measured on actual hardware
        feature_dim = features.shape[1]
        estimated_latency = feature_dim * 0.1  # 0.1ms per feature dimension
        
        return estimated_latency
    
    def calculate_reliability_score(self, accuracy, meets_bandwidth, meets_latency):
        """
        신뢰성 점수 계산
        """
        reliability = accuracy * 0.6  # 60% weight on accuracy
        
        if meets_bandwidth:
            reliability += 0.2  # 20% weight on bandwidth
        if meets_latency:
            reliability += 0.2  # 20% weight on latency
            
        return reliability
    
    def analyze_energy_efficiency(self, traditional_bandwidth, semantic_bandwidth):
        """
        에너지 효율성 분석
        """
        # Energy consumption is roughly proportional to bandwidth
        energy_ratio = traditional_bandwidth / semantic_bandwidth
        
        # Additional energy savings from reduced processing
        processing_savings = 1.5  # 50% additional savings from efficient processing
        
        return energy_ratio * processing_savings
    
    def neuromorphic_hardware_analysis(self, features, labels):
        """
        Neuromorphic hardware 적용 가능성 분석
        """
        print("\n🧠 Neuromorphic Hardware Application Analysis")
        print("=" * 60)
        
        # 1. Spiking Neural Network (SNN) Compatibility
        print("🔬 Analyzing SNN compatibility...")
        
        # Convert features to spike trains
        spike_trains = self.convert_to_spike_trains(features)
        
        # Analyze spike statistics
        spike_stats = self.analyze_spike_statistics(spike_trains)
        
        # 2. Neuromorphic Hardware Requirements
        hardware_requirements = {
            'Intel_Loihi': {
                'max_neurons': 130000,
                'max_synapses': 130000000,
                'power_consumption': 0.1,  # W
                'latency': 0.001  # ms
            },
            'IBM_TrueNorth': {
                'max_neurons': 1000000,
                'max_synapses': 256000000,
                'power_consumption': 0.07,  # W
                'latency': 0.001  # ms
            },
            'BrainChip_Akida': {
                'max_neurons': 80000,
                'max_synapses': 10000000,
                'power_consumption': 0.05,  # W
                'latency': 0.001  # ms
            }
        }
        
        hardware_compatibility = {}
        
        for hardware_name, specs in hardware_requirements.items():
            print(f"\n📊 Analyzing {hardware_name} compatibility...")
            
            # Check if our model fits on the hardware
            required_neurons = features.shape[1] * 2  # Rough estimate
            required_synapses = features.shape[1] * features.shape[1]  # Rough estimate
            
            fits_on_hardware = (required_neurons <= specs['max_neurons'] and 
                              required_synapses <= specs['max_synapses'])
            
            # Calculate power efficiency
            power_efficiency = self.calculate_power_efficiency(specs['power_consumption'], 
                                                             self.estimate_processing_latency(features))
            
            # Calculate speedup
            speedup = self.calculate_neuromorphic_speedup(specs['latency'])
            
            hardware_compatibility[hardware_name] = {
                'fits_on_hardware': fits_on_hardware,
                'required_neurons': required_neurons,
                'required_synapses': required_synapses,
                'power_efficiency': power_efficiency,
                'speedup': speedup,
                'utilization_rate': min(required_neurons / specs['max_neurons'], 1.0)
            }
            
            print(f"  🧠 Fits on hardware: {fits_on_hardware}")
            print(f"  ⚡ Power efficiency: {power_efficiency:.1f}x")
            print(f"  🚀 Speedup: {speedup:.1f}x")
            print(f"  📊 Utilization: {hardware_compatibility[hardware_name]['utilization_rate']:.1%}")
        
        # 3. Biological Plausibility Analysis
        biological_metrics = self.analyze_biological_plausibility(spike_trains, features)
        
        results = {
            'spike_statistics': spike_stats,
            'hardware_compatibility': hardware_compatibility,
            'biological_metrics': biological_metrics
        }
        
        self.application_results['neuromorphic_hardware'] = results
        return results
    
    def convert_to_spike_trains(self, features):
        """
        특징을 스파이크 트레인으로 변환
        """
        # Simple threshold-based spike conversion
        threshold = np.percentile(features, 75)
        spike_trains = (features > threshold).astype(float)
        
        return spike_trains
    
    def analyze_spike_statistics(self, spike_trains):
        """
        스파이크 통계 분석
        """
        spike_rate = np.mean(spike_trains)
        spike_synchrony = np.mean(np.std(spike_trains, axis=0))
        spike_efficiency = spike_rate / (spike_rate + spike_synchrony)
        
        return {
            'spike_rate': spike_rate,
            'spike_synchrony': spike_synchrony,
            'spike_efficiency': spike_efficiency
        }
    
    def calculate_power_efficiency(self, power_consumption, latency):
        """
        전력 효율성 계산
        """
        # Power efficiency = performance / power consumption
        performance = 1.0 / latency  # Higher performance = lower latency
        power_efficiency = performance / power_consumption
        
        return power_efficiency
    
    def calculate_neuromorphic_speedup(self, neuromorphic_latency):
        """
        Neuromorphic 하드웨어 속도 향상 계산
        """
        # Compare with traditional CPU/GPU latency
        traditional_latency = 10.0  # ms (typical for CPU/GPU)
        speedup = traditional_latency / neuromorphic_latency
        
        return speedup
    
    def analyze_biological_plausibility(self, spike_trains, features):
        """
        생물학적 현실성 분석
        """
        # 1. Spike timing precision
        spike_timing_precision = self.calculate_spike_timing_precision(spike_trains)
        
        # 2. Energy efficiency (biological)
        biological_energy_efficiency = self.calculate_biological_energy_efficiency(spike_trains)
        
        # 3. Information coding efficiency
        coding_efficiency = self.calculate_coding_efficiency(spike_trains, features)
        
        return {
            'spike_timing_precision': spike_timing_precision,
            'biological_energy_efficiency': biological_energy_efficiency,
            'coding_efficiency': coding_efficiency
        }
    
    def calculate_spike_timing_precision(self, spike_trains):
        """
        스파이크 타이밍 정밀도 계산
        """
        # Simplified spike timing precision calculation
        spike_intervals = []
        for i in range(spike_trains.shape[0]):
            spike_positions = np.where(spike_trains[i] > 0)[0]
            if len(spike_positions) > 1:
                intervals = np.diff(spike_positions)
                spike_intervals.extend(intervals)
        
        if spike_intervals:
            timing_precision = 1.0 / (np.std(spike_intervals) + 1e-8)
        else:
            timing_precision = 0.0
            
        return timing_precision
    
    def calculate_biological_energy_efficiency(self, spike_trains):
        """
        생물학적 에너지 효율성 계산
        """
        # Energy per spike (simplified)
        energy_per_spike = 0.1  # pJ per spike (biological estimate)
        total_spikes = np.sum(spike_trains)
        total_energy = total_spikes * energy_per_spike
        
        # Information content
        information_content = np.sum(spike_trains) * np.log2(spike_trains.shape[1] + 1)
        
        # Energy efficiency = information / energy
        if total_energy > 0:
            energy_efficiency = information_content / total_energy
        else:
            energy_efficiency = 0.0
            
        return energy_efficiency
    
    def calculate_coding_efficiency(self, spike_trains, original_features):
        """
        코딩 효율성 계산
        """
        # Mutual information between spike trains and original features
        from sklearn.metrics import mutual_info_score
        
        total_mi = 0
        for i in range(spike_trains.shape[1]):
            if i < original_features.shape[1]:
                mi = mutual_info_score(spike_trains[:, i], 
                                     np.digitize(original_features[:, i], bins=10))
                total_mi += mi
        
        coding_efficiency = total_mi / spike_trains.shape[1]
        return coding_efficiency
    
    def edge_computing_analysis(self, features, labels):
        """
        Edge computing 응용 분석
        """
        print("\n💻 Edge Computing Application Analysis")
        print("=" * 60)
        
        # 1. Edge Device Compatibility
        edge_devices = {
            'Raspberry_Pi_4': {
                'cpu_cores': 4,
                'ram': 4096,  # MB
                'power_consumption': 3.4,  # W
                'cost': 35  # USD
            },
            'Jetson_Nano': {
                'cpu_cores': 4,
                'ram': 4096,  # MB
                'power_consumption': 5.0,  # W
                'cost': 99  # USD
            },
            'ESP32': {
                'cpu_cores': 2,
                'ram': 520,  # KB
                'power_consumption': 0.1,  # W
                'cost': 5  # USD
            },
            'Arduino_Nano': {
                'cpu_cores': 1,
                'ram': 2,  # KB
                'power_consumption': 0.02,  # W
                'cost': 3  # USD
            }
        }
        
        edge_compatibility = {}
        
        for device_name, specs in edge_devices.items():
            print(f"\n📊 Analyzing {device_name} compatibility...")
            
            # Memory requirements
            memory_required = features.shape[1] * 4  # 4 bytes per feature (float32)
            memory_available = specs['ram'] * 1024 if device_name in ['Raspberry_Pi_4', 'Jetson_Nano'] else specs['ram']
            
            fits_in_memory = memory_required <= memory_available * 0.5  # 50% safety margin
            
            # Processing capability
            processing_capability = self.estimate_processing_capability(specs['cpu_cores'], memory_available)
            
            # Power efficiency
            power_efficiency = self.calculate_edge_power_efficiency(specs['power_consumption'], 
                                                                  self.estimate_processing_latency(features))
            
            # Cost efficiency
            cost_efficiency = self.calculate_cost_efficiency(specs['cost'], processing_capability)
            
            edge_compatibility[device_name] = {
                'fits_in_memory': fits_in_memory,
                'memory_required': memory_required,
                'memory_available': memory_available,
                'processing_capability': processing_capability,
                'power_efficiency': power_efficiency,
                'cost_efficiency': cost_efficiency,
                'deployment_score': self.calculate_deployment_score(fits_in_memory, processing_capability, power_efficiency)
            }
            
            print(f"  💾 Memory: {memory_required} bytes (Available: {memory_available} bytes)")
            print(f"  ⚡ Power efficiency: {power_efficiency:.1f}x")
            print(f"  💰 Cost efficiency: {cost_efficiency:.1f}x")
            print(f"  🎯 Deployment score: {edge_compatibility[device_name]['deployment_score']:.2f}")
        
        # 2. Real-time Performance Analysis
        real_time_analysis = self.analyze_real_time_performance(features, edge_devices)
        
        # 3. Scalability Analysis
        scalability_analysis = self.analyze_scalability(features, edge_devices)
        
        results = {
            'edge_compatibility': edge_compatibility,
            'real_time_analysis': real_time_analysis,
            'scalability_analysis': scalability_analysis
        }
        
        self.application_results['edge_computing'] = results
        return results
    
    def estimate_processing_capability(self, cpu_cores, memory):
        """
        처리 능력 추정
        """
        # Simplified processing capability estimation
        processing_capability = cpu_cores * (memory / 1024)  # Normalized by memory in MB
        
        return processing_capability
    
    def calculate_edge_power_efficiency(self, power_consumption, latency):
        """
        Edge 디바이스 전력 효율성 계산
        """
        # Power efficiency = performance / power consumption
        performance = 1.0 / latency
        power_efficiency = performance / power_consumption
        
        return power_efficiency
    
    def calculate_cost_efficiency(self, cost, processing_capability):
        """
        비용 효율성 계산
        """
        # Cost efficiency = processing capability / cost
        cost_efficiency = processing_capability / cost
        
        return cost_efficiency
    
    def calculate_deployment_score(self, fits_in_memory, processing_capability, power_efficiency):
        """
        배포 점수 계산
        """
        score = 0.0
        
        if fits_in_memory:
            score += 0.4  # 40% weight on memory fit
        
        # Normalize processing capability and power efficiency
        normalized_processing = min(processing_capability / 100, 1.0)
        normalized_power = min(power_efficiency / 10, 1.0)
        
        score += normalized_processing * 0.3  # 30% weight on processing
        score += normalized_power * 0.3  # 30% weight on power efficiency
        
        return score
    
    def analyze_real_time_performance(self, features, edge_devices):
        """
        실시간 성능 분석
        """
        real_time_results = {}
        
        for device_name, specs in edge_devices.items():
            # Estimate processing time
            processing_time = self.estimate_processing_latency(features) * (100 / specs['cpu_cores'])
            
            # Real-time requirements (30 FPS = 33.3ms per frame)
            real_time_threshold = 33.3  # ms
            
            meets_real_time = processing_time <= real_time_threshold
            frame_rate = 1000 / processing_time if processing_time > 0 else float('inf')
            
            real_time_results[device_name] = {
                'processing_time': processing_time,
                'meets_real_time': meets_real_time,
                'frame_rate': frame_rate,
                'real_time_margin': real_time_threshold - processing_time
            }
        
        return real_time_results
    
    def analyze_scalability(self, features, edge_devices):
        """
        확장성 분석
        """
        scalability_results = {}
        
        for device_name, specs in edge_devices.items():
            # Calculate how many parallel instances can run
            memory_per_instance = features.shape[1] * 4  # bytes
            max_instances = int((specs['ram'] * 1024 * 0.5) / memory_per_instance)  # 50% safety margin
            
            # Power consumption for multiple instances
            total_power = specs['power_consumption'] * max_instances
            
            # Throughput (instances per second)
            processing_time = self.estimate_processing_latency(features)
            throughput = max_instances / processing_time if processing_time > 0 else 0
            
            scalability_results[device_name] = {
                'max_instances': max_instances,
                'total_power': total_power,
                'throughput': throughput,
                'scalability_score': max_instances * throughput / total_power
            }
        
        return scalability_results
    
    def generate_application_plots(self):
        """
        응용 분석 결과 시각화
        """
        if not self.application_results:
            print("❌ No application results available. Run analysis first.")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Semantic Communication Results
        if 'semantic_communication' in self.application_results:
            sem_results = self.application_results['semantic_communication']
            
            # Bandwidth comparison
            bandwidths = [sem_results['traditional_bandwidth'], sem_results['semantic_bandwidth']]
            labels = ['Traditional', 'Semantic']
            
            axes[0, 0].bar(labels, bandwidths, alpha=0.8)
            axes[0, 0].set_title('Bandwidth Comparison')
            axes[0, 0].set_ylabel('Bandwidth (bits)')
        
        # 2. Neuromorphic Hardware Compatibility
        if 'neuromorphic_hardware' in self.application_results:
            neuro_results = self.application_results['neuromorphic_hardware']
            
            if 'hardware_compatibility' in neuro_results:
                hw_compat = neuro_results['hardware_compatibility']
                
                hardware_names = list(hw_compat.keys())
                power_efficiencies = [hw_compat[name]['power_efficiency'] for name in hardware_names]
                
                axes[0, 1].bar(hardware_names, power_efficiencies, alpha=0.8)
                axes[0, 1].set_title('Neuromorphic Hardware Power Efficiency')
                axes[0, 1].set_ylabel('Power Efficiency')
                axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Edge Computing Deployment Scores
        if 'edge_computing' in self.application_results:
            edge_results = self.application_results['edge_computing']
            
            if 'edge_compatibility' in edge_results:
                edge_compat = edge_results['edge_compatibility']
                
                device_names = list(edge_compat.keys())
                deployment_scores = [edge_compat[name]['deployment_score'] for name in device_names]
                
                axes[0, 2].bar(device_names, deployment_scores, alpha=0.8)
                axes[0, 2].set_title('Edge Device Deployment Scores')
                axes[0, 2].set_ylabel('Deployment Score')
                axes[0, 2].tick_params(axis='x', rotation=45)
        
        # 4. Real-time Performance
        if 'edge_computing' in self.application_results:
            edge_results = self.application_results['edge_computing']
            
            if 'real_time_analysis' in edge_results:
                rt_analysis = edge_results['real_time_analysis']
                
                device_names = list(rt_analysis.keys())
                frame_rates = [rt_analysis[name]['frame_rate'] for name in device_names]
                
                axes[1, 0].bar(device_names, frame_rates, alpha=0.8)
                axes[1, 0].set_title('Real-time Frame Rates')
                axes[1, 0].set_ylabel('Frame Rate (FPS)')
                axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 5. Scalability Analysis
        if 'edge_computing' in self.application_results:
            edge_results = self.application_results['edge_computing']
            
            if 'scalability_analysis' in edge_results:
                scale_analysis = edge_results['scalability_analysis']
                
                device_names = list(scale_analysis.keys())
                max_instances = [scale_analysis[name]['max_instances'] for name in device_names]
                
                axes[1, 1].bar(device_names, max_instances, alpha=0.8)
                axes[1, 1].set_title('Maximum Parallel Instances')
                axes[1, 1].set_ylabel('Number of Instances')
                axes[1, 1].tick_params(axis='x', rotation=45)
        
        # 6. Overall Impact Summary
        impact_metrics = []
        impact_labels = []
        
        if 'semantic_communication' in self.application_results:
            sem_results = self.application_results['semantic_communication']
            impact_metrics.extend([sem_results['compression_ratio'], sem_results['energy_efficiency']])
            impact_labels.extend(['Bandwidth\nCompression', 'Energy\nEfficiency'])
        
        if 'neuromorphic_hardware' in self.application_results:
            neuro_results = self.application_results['neuromorphic_hardware']
            if 'hardware_compatibility' in neuro_results:
                avg_speedup = np.mean([hw['speedup'] for hw in neuro_results['hardware_compatibility'].values()])
                impact_metrics.append(avg_speedup)
                impact_labels.append('Neuromorphic\nSpeedup')
        
        if impact_metrics:
            axes[1, 2].bar(impact_labels, impact_metrics, alpha=0.8)
            axes[1, 2].set_title('Overall Impact Metrics')
            axes[1, 2].set_ylabel('Improvement Factor')
        
        plt.tight_layout()
        plt.savefig('/content/drive/MyDrive/application_impact_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def save_application_results(self, save_path="/content/drive/MyDrive/application_impact_results.json"):
        """
        응용 분석 결과 저장
        """
        try:
            # Convert numpy types to Python types for JSON serialization
            results_serializable = {}
            for key, value in self.application_results.items():
                if isinstance(value, dict):
                    results_serializable[key] = {}
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, dict):
                            results_serializable[key][sub_key] = {}
                            for sub_sub_key, sub_sub_value in sub_value.items():
                                if isinstance(sub_sub_value, np.ndarray):
                                    results_serializable[key][sub_key][sub_sub_key] = sub_sub_value.tolist()
                                elif isinstance(sub_sub_value, np.integer):
                                    results_serializable[key][sub_key][sub_sub_key] = int(sub_sub_value)
                                elif isinstance(sub_sub_value, np.floating):
                                    results_serializable[key][sub_key][sub_sub_key] = float(sub_sub_value)
                                else:
                                    results_serializable[key][sub_key][sub_sub_key] = sub_sub_value
                        elif isinstance(sub_value, np.ndarray):
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
            
            print(f"✅ Application impact results saved to: {save_path}")
            return True
            
        except Exception as e:
            print(f"❌ Error saving results: {e}")
            return False

# Usage example
def run_application_impact_analysis(lgmd_encoder, lgmd_classifier, features, labels):
    """
    전체 응용 및 impact 분석 실행
    """
    analyzer = ApplicationImpactAnalysis(lgmd_encoder, lgmd_classifier)
    
    # 1. Ultra-low bandwidth semantic communication
    semantic_results = analyzer.ultra_low_bandwidth_semantic_communication(features, labels)
    
    # 2. Neuromorphic hardware analysis
    neuromorphic_results = analyzer.neuromorphic_hardware_analysis(features, labels)
    
    # 3. Edge computing analysis
    edge_results = analyzer.edge_computing_analysis(features, labels)
    
    # 4. Generate plots and save results
    analyzer.generate_application_plots()
    analyzer.save_application_results()
    
    return analyzer.application_results 