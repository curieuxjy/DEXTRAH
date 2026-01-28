# DextrAH 환경 설정 및 학습 가이드

이 문서는 IsaacSim 5.1.0과 IsaacLab (main branch)을 사용하여 DextrAH를 실행하는 방법을 설명합니다.

## 환경 구성

| 구성 요소 | 버전 | 경로 |
|----------|------|------|
| IsaacSim | 5.1.0 (소스 빌드) | `/home/avery/isaacsim` |
| IsaacLab | main branch (v2.3.1+) | `/home/avery/Documents/IsaacLab` |
| FABRICS | 0.1.0 | `/home/avery/Documents/FABRICS` |
| DextrAH | 0.1.0 | `/home/avery/Documents/DEXTRAH` |
| Python | 3.11.13 | IsaacSim 내장 |

## 버전 호환성

IsaacLab과 IsaacSim 버전 호환성 표:

| IsaacLab 버전 | IsaacSim 버전 |
|--------------|---------------|
| v2.3.X | 4.5 / 5.0 / **5.1** |
| v2.2.X | 4.5 / 5.0 |
| v2.1.X | 4.5 |
| v2.0.X | 4.5 |

**참고**: DextrAH는 원래 IsaacLab v2.2.1용으로 작성되었으나, 최신 main 브랜치(v2.3.1+)에서 실행하기 위해 일부 코드 수정이 필요했습니다.

## 설치 과정

### 1. IsaacSim 5.1.0 빌드

```bash
cd /home/avery/isaacsim
./build.sh
```

빌드 결과물: `/home/avery/isaacsim/_build/linux-x86_64/release/`

### 2. IsaacLab 설치 (main branch)

```bash
cd /home/avery/Documents
git clone https://github.com/curieuxjy/IsaacLab.git
cd IsaacLab
# main 브랜치 사용 (v2.3.1 이후 최신 개발 버전)

# IsaacSim 심볼릭 링크 생성
ln -s /home/avery/isaacsim/_build/linux-x86_64/release _isaac_sim

# IsaacLab 설치
./_isaac_sim/python.sh -m pip install -e source/isaaclab
./_isaac_sim/python.sh -m pip install -e source/isaaclab_tasks
./_isaac_sim/python.sh -m pip install -e source/isaaclab_assets
./_isaac_sim/python.sh -m pip install -e source/isaaclab_rl
./_isaac_sim/python.sh -m pip install rl_games
```

### 3. FABRICS 설치

```bash
cd /home/avery/Documents
git clone https://github.com/NVlabs/FABRICS.git
cd /home/avery/Documents/IsaacLab
./_isaac_sim/python.sh -m pip install -e /home/avery/Documents/FABRICS
```

### 4. DextrAH 설치

```bash
cd /home/avery/Documents/IsaacLab
./_isaac_sim/python.sh -m pip install -e /home/avery/Documents/DEXTRAH
```

### 5. 텍스처 에셋 다운로드

```bash
cd /home/avery/Documents/DEXTRAH/dextrah_lab/assets
wget -O textures.zip "https://huggingface.co/datasets/nvidia/dextrah_textures/resolve/main/textures.zip"
unzip textures.zip
rm textures.zip
```

## 코드 수정 사항

### IsaacLab 호환성 수정

`dump_pickle` 함수가 IsaacLab v2.3.x에서 제거되어 다음 파일들에 직접 추가:

- `dextrah_lab/rl_games/train.py`
- `dextrah_lab/distillation/run_distillation.py`
- `dextrah_lab/distillation/run_distillation_transformer.py`

수정 내용:
```python
from isaaclab.utils.io import dump_yaml
import pickle

def dump_pickle(filename: str, data: object):
    """Dump data to a pickle file."""
    with open(filename, "wb") as f:
        pickle.dump(data, f)
```

### NumPy 2.0 호환성 수정

urdfpy 라이브러리에서 `np.float` 사용으로 인한 에러 수정:

```bash
sed -i 's/np\.float\b/float/g' /home/avery/Documents/IsaacLab/_isaac_sim/kit/python/lib/python3.11/site-packages/urdfpy/urdf.py
```

### networkx 버전 업그레이드

Python 3.11 호환을 위해 networkx 업그레이드:

```bash
./_isaac_sim/python.sh -m pip install "networkx>=3.0"
```

## 학습 실행

### 기본 실행 명령어

```bash
cd /home/avery/Documents/IsaacLab

./_isaac_sim/python.sh /home/avery/Documents/DEXTRAH/dextrah_lab/rl_games/train.py \
    --task=Dextrah-Kuka-Allegro \
    --num_envs 1024 \
    --headless \
    --max_iterations 100 \
    env.objects_dir=visdex_objects \
    env.max_pose_angle=45.0 \
    agent.wandb_activate=False \
    agent.params.config.minibatch_size=8192 \
    agent.params.config.central_value_config.minibatch_size=8192
```

### 필수 파라미터

| 파라미터 | 설명 | 예시 값 |
|---------|------|---------|
| `--task` | 태스크 이름 | `Dextrah-Kuka-Allegro` |
| `--num_envs` | 병렬 환경 수 | `1024` |
| `--headless` | GUI 없이 실행 | - |
| `--max_iterations` | 최대 학습 에폭 수 | `100` |
| `env.objects_dir` | 오브젝트 디렉토리 | `visdex_objects` |
| `env.max_pose_angle` | 최대 포즈 각도 | `45.0` |
| `agent.wandb_activate` | W&B 로깅 활성화 | `False` |

### GPU 메모리 고려사항

RTX 4070 Laptop GPU (8GB VRAM) 기준:
- `--num_envs 4096`: OOM 에러 발생
- `--num_envs 1024`: 정상 작동
- minibatch_size 조정 필요: 기본값 16384 → 8192

배치 크기 계산:
```
batch_size = num_envs × horizon_length
           = 1024 × 16 = 16384

조건: batch_size % minibatch_size == 0
```

## 학습 결과

### 테스트 학습 결과 (100 epochs)

```
Epochs: 100/100
Total Frames: 1,622,016
Best Reward: 5.917524
Average FPS: ~6,750
```

### 체크포인트 저장 위치

```
/home/avery/Documents/IsaacLab/logs/rl_games/dextrah_lstm/<timestamp>/nn/
├── dextrah_lstm.pth                    # Best checkpoint
└── last_dextrah_lstm_ep_XXX_rew_XXX.pth # Last checkpoint
```

## 학습된 모델 시각화 (GUI)

```bash
cd /home/avery/Documents/IsaacLab

./_isaac_sim/python.sh /home/avery/Documents/DEXTRAH/dextrah_lab/rl_games/play.py \
    --task=Dextrah-Kuka-Allegro \
    --num_envs 4 \
    --checkpoint /home/avery/Documents/IsaacLab/logs/rl_games/dextrah_lstm/<timestamp>/nn/dextrah_lstm.pth
```

## 학습 모니터링

### 실시간 로그 확인

```bash
# 백그라운드 실행 시
tail -f /tmp/claude/-home-avery-Documents-DEXTRAH/tasks/<task_id>.output

# 또는 로그 디렉토리에서
tail -f /home/avery/Documents/IsaacLab/logs/rl_games/dextrah_lstm/<timestamp>/log.txt
```

### 학습 지표

출력 예시:
```
fps step: 15716 fps step and policy inference: 15042 fps total: 6777 epoch: 98/100 frames: 1589248
```

- `fps step`: 시뮬레이션 스텝 FPS
- `fps total`: 전체 학습 FPS (학습 + 시뮬레이션)
- `epoch`: 현재 에폭 / 총 에폭
- `frames`: 총 처리된 프레임 수

## 문제 해결

### 1. CUDA Out of Memory

**증상**: `torch.OutOfMemoryError: CUDA out of memory`

**해결**:
- `--num_envs` 감소
- `minibatch_size` 감소

### 2. Batch Size Assertion Error

**증상**: `AssertionError: self.batch_size % self.minibatch_size == 0`

**해결**:
- `num_envs × horizon_length`가 `minibatch_size`로 나누어 떨어지도록 조정
- 예: num_envs=1024, horizon_length=16 → batch_size=16384
- minibatch_size를 8192로 설정하면 16384 % 8192 = 0

### 3. Max Pose Angle Error

**증상**: `ValueError: Max pose angle must be positive`

**해결**:
- `env.max_pose_angle=45.0` 파라미터 추가

### 4. Valid Objects Directory Error

**증상**: `ValueError: Need to specify valid directory of objects for training`

**해결**:
- `env.objects_dir=visdex_objects` 파라미터 추가
- 텍스처 에셋이 다운로드되어 있는지 확인

## 디렉토리 구조

```
/home/avery/
├── isaacsim/                          # IsaacSim 5.1.0 소스
│   └── _build/linux-x86_64/release/   # 빌드 결과물
└── Documents/
    ├── IsaacLab/                      # IsaacLab main branch (curieuxjy fork)
    │   ├── _isaac_sim -> /home/avery/isaacsim/_build/linux-x86_64/release
    │   └── logs/rl_games/             # 학습 로그 및 체크포인트
    ├── DEXTRAH/                       # DextrAH 프로젝트
    │   ├── dextrah_lab/
    │   │   ├── assets/
    │   │   │   └── visdex_objects/    # 오브젝트 에셋
    │   │   ├── rl_games/
    │   │   │   └── train.py           # 학습 스크립트
    │   │   └── distillation/          # 증류 학습 스크립트
    │   └── docs/                      # 문서
    └── FABRICS/                       # FABRICS 라이브러리
```

## 참고 자료

- [IsaacLab Documentation](https://isaac-sim.github.io/IsaacLab)
- [DextrAH README](../README.md)
- [FABRICS Repository](https://github.com/NVlabs/FABRICS)
