# DFGFlow

시계열 전처리, autoencoder/flow matching 학습, 생성만 포함한 독립 패키지입니다. 기존 DirectDiff-TS 폴더는 변경하지 않았습니다.

## 포함 범위

- CSV(Energy·ETTh1 등), Accel XYZ NPY, MIMIC-IV 전처리 코드
- 기존 모델 구조, 학습 loss 및 학습 진행 로그
- 정규화된 complex STFT의 실수·허수 표현과 시계열 복원
- 시계열 생성 및 채널별 1D plot

데이터 파일, 기존 체크포인트, 실험 결과, metric 평가, ablation 실행기, GT 비교 그림 및 spectrogram 시각화는 포함하지 않습니다.

## 생성 결과

- `.npy`: 원래 단위의 시계열. 형상은 **[샘플 수, 시계열 길이, 채널 수]**입니다.
- `<출력 이름>_plots/sample_XXXX.png`: 샘플마다 채널별 1D subplot을 배치한 그림입니다. 기본적으로 처음 4개 샘플을 그리며 `max_plots=0`이면 NPY만 저장합니다.

새 체크포인트에는 샘플을 제외한 정규화 통계와 STFT 설정을 저장하므로, 생성 시 원본 학습 데이터가 필요하지 않습니다. 기존 체크포인트를 사용할 경우에는 복원 메타데이터가 담긴 학습 NPZ를 별도로 지정해야 합니다. Waveform 생성은 phase가 보존된 complex 표현을 사용합니다.

## 구성

- `src/dfgflow/config.py`, `model.py`: 모델과 설정
- `src/dfgflow/data.py`, `lazy_stft.py`: 학습 입력과 on-demand STFT
- `src/dfgflow/losses.py`: 학습 loss와 STFT 복원
- `src/dfgflow/metadata.py`: 복원 메타데이터
- `src/dfgflow/train.py`: AE/Flow 학습
- `src/dfgflow/generate.py`: 시계열 NPY 및 1D plot 생성
- `scripts/`: 전처리 도구만 포함

패키지 이름은 `dfgflow`입니다. 모델 클래스 이름과 checkpoint state_dict 키, 체크포인트 파일명은 기존 호환성을 위해 유지했습니다. 모델 구조와 학습 계산은 원본과 동일합니다.

일반 CSV/Accel 전처리는 제공한 입력에서 정규화 통계를 계산하므로 학습용 데이터만 입력해야 합니다. MIMIC packer는 train 통계를 val/test에 재사용합니다. 학습용 인접 쌍은 동일 record 내에 인접 window가 존재해야 합니다.
