# 실행 가이드 #

본 프로젝트는 VLA 기반 지능 모델과 물리적 강건성 제어를 위한 PPO 복원 루프를 결합한 하이브리드 시스템이다. 이 문서는 연구 환경 구성부터 모델 추론, 학습, 평가 및 정밀 진단까지의 전 과정을 기술한다

---
## 1. 실험 환경 및 인프라 구성 ##
본 프로젝트는 고성능 GPU 인프라와 정밀 물리 시뮬레이션 환경의 동기화를 위해 엄격한 시스템 구성이 요구된다.
모든 설치는 로컬 환경의 충돌을 방지하기 위해 전용 가상환경(```venv```) 내에서 수행한다

### 1.1 하드웨어 및 운영체제 요구사항 ###
|항목|사양|비고|
|---|---|---|
|OS|Ubuntu 22.04 LTS|커널 6.8 버전 이상 권장|
|GPU|NVIDIA RTX 4090 24GB|VRAM 24GB 이상 필수 (GROOT 모델 탑재용)|
|Driver|NVIDIA Driver 550.x+|CUDA 12.x 호환성 확인 필수|
|Framework|NVIDIA Isaac Lab|Isaac Sim 4.x 기반 특수 빌드|

### 1.2 시스템 설치 및 환경 구축 ###
**(1) GROOT 인지 엔진 환경** (```groot_project```)
```bash
# 리포지토리 클론 및 폴더 이동
git clone <GROOT_REPO_URL> groot_project
cd groot_project

# 전용 가상환경 생성 및 활성화
conda create -n gr00t_project python=3.10 -y
conda activate gr00t_project

# 의존성 설치
pip install --upgrade pip
pip install -r requirements.txt
```
- 주의: 본 환경은 CUDA 12.x 버전을 기준으로 구성되었습니다. 시스템의 NVIDIA 드라이버가 CUDA 12.x와 호환되는지 확인하십시오

**(2) Isaac Lab 물리 시뮬레이션 환경** (```IsaacLab```)
```bash
# 리포지토리 클론 및 폴더 이동
git clone <ISAACLAB_REPO_URL> IsaacLab
cd IsaacLab

# 전용 가상환경 생성 및 활성화
conda create -n isaac_env python=3.10 -y
conda activate isaac_env

# Isaac Lab 설치 및 빌드
./isaaclab.sh --install
```

### 1.3 시뮬레이션 환경 (Custom Isaac Lab) ###

본 실험은 표준 Isaac Lab 환경에 물리 타격 프로토콜이 임베딩된 Isaac-Lift-Cube-Franka-v0 커스텀 빌드를 사용한다
- 빌드 검증: 물리 엔진 내 커스텀 환경이 정상 로드되는지 확인한다

```bash
# IsaacLab 폴더 내에서 실행
conda activate isaac_env
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Lift-Cube-Franka-v0 \
    --num_envs 1 --device cuda
```
- 위 명령 실행 시, 로봇이 정상 스폰되고 외란 인가 모듈(Perturbation Module)이 활성화된 화면이 나타나면 실험 환경 구축이 완료된 것이다

### 1.4 PPO 기반 복원 제어 전략 ###
본 프로젝트는 PPO(Proximal Policy Optimization)를 활용하여, 외란 발생 시 로봇의 자세를 즉각적으로 재보정하는 복원 루프를 구축한다

- 선택 근거: 정책 업데이트 시 급격한 성능 저하를 방지하는 클리핑 메커니즘을 통해 불규칙한 외란 환경에서도 안정적인 토크 제어를 보장한다
- 복원 로직: 외부 외란 유입 시 정책 네트워크가 실시간으로 관절 토크 변위($\Delta \tau$)를 계산하여, 물체를 파지한 상태의 자세 안정성을 유지한다
 
|파일 경로|주요 역할|
|---|---|
|```isaac_project/groot_project/gr00t/rl_agent/ppo_agent.py```|Actor-Critic 모델 기반의 PPO 정책 업데이트 및 최적화 루프 담당|
|```isaac_project/groot_project/gr00t/rl_agent/ppo_buffer.py```|로봇의 상태, 행동, 보상 기록을 저장하고 텐서 형태로 변환하여 학습 데이터 관리|

## 2. 정교한 외란 프로토콜 고도화 (Config) ##
실험의 핵심은 ```force_protocol.yaml```에 정의된 물리적 인가 프로토콜에 있으며, 이는 로봇이 물체를 잡고 난 후의 '취약 구간'을 정밀하게 타격하여 시스템의 강건성을 검증하도록 설계되었다

```YAML
# Force Injection Protocol (Tuned for Robustness)
protocol_metadata:
  target_link: "panda_hand"
  noise_amplitude: 150.0  # 정량적 한계 임계값 (150N)
  temporal_sampling: 2.0  # 타격 인가 주기 (2.0s)
  injection_mode: "non-linear-pulse" 
  recovery_threshold: 0.05
```
**실험 설계의 근거**
- 타격 시점(2.0s): 로봇의 평균 이송 완료 직후, 안착 단계의 가장 불안정한 시점에 외력을 주입하여 제어 성능을 평가한다
- 노이즈 강도 및 주파수(150N/2.0s): 실제 환경의 공기역학적 난류를 모사하고 모델의 복원 한계 임계값을 측정하기 위해 튜닝된 최적값이다
- 시스템 제어 변수: 물리 엔진 샘플링 주기와 동기화하여, 특정 외란 패턴에 대한 모델의 과적합을 방지하고 반사 복원력을 극대화한다


## 3. 파이프라인 실행 (가상 시나리오) ##
본 시스템은 GROOT 모델이 추론하는 상위 수준의 인지와 PPO가 제어하는 하위 수준의 복원 루프가 독립적으로 연동된다

### 3.1 GROOT 백본 인지 서비스 실행 ###
상위 지각 엔진을 먼저 가동하여 실시간 시각적 상태를 토큰화한다

```bash
# 외부 제어 루프와의 비동기 통신을 위해 서비스화된 추론 환경 가동
CUDA_VISIBLE_DEVICES=1 python3 scripts/inference_service.py \
    --server \
    --model_path ~/Isaac-GR00T-n1.5-table/weights \
    --embodiment-tag new_embodiment \
    --data-config so100_dualcam \
    --denoising-steps 4
  ```

- 주의: `denoising-steps`는 실시간성 확보를 위해 4로 고정하며, 이 값을 높일 경우 물리 복원 레이어와 시각 데이터 간의 주파수 불일치가 발생할 수 있다

### 3.2 하부 제어 루프 연동 (Isaac Lab) ###
GROOT의 추론 값을 입력으로 받아 로봇 관절의 토크를 매핑하는 하부 RL 루프를 가동한다

```bash
# 외부 외란 인가 모듈이 활성화된 특수 환경 로드
CUDA_VISIBLE_DEVICES=1 ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Lift-Cube-Franka-v0 \
    --use_pretrained_checkpoint \
    --livestream 2 \ # 원격 시각화를 위한 WebRTC 모드 활성화
    --num_envs 1 \
    --enable_cameras \
    --device cuda
```

- 중요: `num_envs 1` 설정은 단일 환경 내 정밀한 물리 강건성 계측을 위해 강제된다. 병렬 환경 적용 시 외란 프로토콜의 인가 주기가 물리 엔진 스텝과 비동기화될 수 있다

### 3.3 원격 시각화 및 WebRTC 제어 ###
실제 환경에서는 서버의 GUI를 직접 보지 않고, 로컬 PC에서 **WebRTC Streaming Client**를 통해 시뮬레이션 화면을 스트리밍받아 실시간으로 제어 상황을 모니터링한다

- 스트리밍 접속: 서버 가동 후 로컬 웹 브라우저에서 ```http://<서버-IP>:8208```에 접속하여 실시간 렌더링 화면을 확인한다.
- WebRTC 설정: ```--livestream 2``` 옵션은 WebRTC 프로토콜을 사용한 저지연 스트리밍을 활성화하며, 화면 해상도나 프레임 레이트 조절이 필요한 경우 ```IsaacLab``` 폴더 내의 ```user_settings.yaml```에서 ```livestream``` 설정을 변경한다.
- 주의: 네트워크 대역폭 부족으로 인한 화면 멈춤 발생 시, 스트리밍 설정에서 ```bitrate```를 하향 조정하여 제어 루프와의 동기화를 유지할 것을 권장한다.

### 3.4 시스템 핵심 구조 및 파라미터 제어 ###
파이프라인을 수정하거나 실험 조건을 변경하기 위해 접근해야 할 파일 경로와 제어 로직은 다음과 같다

**(1) 외란 프로토콜 및 타격 설정** (```configs/disturbance/force_protocol.yaml```)
```YAML
perturbation:
  target_link: "panda_hand"      # 타격 대상
  trigger_time_s: 2.0           # 외란 유입 기점
  interval_range_s: [2.0, 2.0]  # 2초 주기 펄스 루프
  force_range: [-150.0, 150.0]  # 강건성 실증을 위한 충격 범위
```

**(2) 제어 루프 및 샘플링 주기 제약**

제어 주기는 환경 설정 파일 내에서 강제된다
- 파일 경로: env/franka_task.py
- 설정 파라미터:
  - sim.dt: 0.01s (100Hz 물리 엔진 구동)
  - decimation: 2 (50Hz 제어 정책 송출 명령)
  - max_episode_length: 10.0s (타임아웃 윈도우 확장값)

## 4. 제어 안정성 확보 및 병목 해결 전략 ##
실험 과정에서 발생하는 기술적 병목은 시스템의 물리적 제약과 인지-제어 간의 데이터 동기화 문제에서 기인한다

|항목|발생 원인|진단 방법|제어 및 최적화 전략|
|---|---|---|---|
|통신 동기화 (Sync)|인지/물리 엔진 간 주파수 불일치|터미널 로그 내 latency_offset 수치 확인|프로세스 우선순위 조정 및 ```latency_offset``` 20ms 이내 유지|
|물리 제어 붕괴 (Collapse)|학습 초기 관절 토크 과출력|Isaac Lab 뷰어의 관절 거동 불안정성 모니터링|PPO ```clip_range``` 0.1 이하 설정으로 토크 과출력 방지|
|메모리 효율 (Memory)|VRAM 부족으로 인한 병목|```watch -n 1 nvidia-smi```로 VRAM 점유율 실시간 감시|배치 크기(Batch Size) 2로 고정하여 VRAM 점유 최적화|

