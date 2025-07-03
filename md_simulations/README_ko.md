# 분자 역학 시뮬레이션 프레임워크

이 프레임워크는 CHARMM36 힘 필드 및 SPC/E 물 모델과 함께 GROMACS 5.1.2를 사용하여 분자 역학 시뮬레이션을 실행하기 위한 Python 인터페이스를 제공합니다. 참조 프로토콜에 설명된 정확한 방법론을 구현합니다.

## 전제 조건

- GROMACS 5.1.2
- Python 3.7 이상
- 필수 Python 패키지 (`pip install -r requirements.txt`를 통해 설치)

## 설치

1. GROMACS 5.1.2가 설치되어 있고 PATH에서 액세스할 수 있는지 확인합니다.
2. 이 리포지토리를 복제합니다.
3. Python 종속성을 설치합니다.
   ```bash
   pip install -r requirements.txt
   ```

## 사용법

```python
from src.md_simulation import MDSimulation

# 입력 구조로 시뮬레이션 초기화
sim = MDSimulation(
    structure_path="path/to/protein.pdb",
    output_dir="simulation_output"
)

# 구조 준비 (기본적으로 pH 7.0)
sim.prepare_structure()

# 시뮬레이션 상자 설정
sim.setup_box(distance=1.2)  # 단백질에서 상자 가장자리까지 1.2nm

# 용매 및 이온 추가
sim.solvate_and_ions()

# 에너지 최소화 수행
sim.energy_minimization(nsteps=2000)

# 다른 온도에서 생산 시뮬레이션 실행
temperatures = [343, 353, 363, 400]
for temp in temperatures:
    sim.run_simulation(
        temperature=temp,
        duration_ns=100,  # 100ns 시뮬레이션
        dt=0.002  # 2fs 시간 단계
    )

    # 궤적 분석
    results = sim.analyze_trajectory(temperature=temp)

# 유연한 영역 식별
flexible_regions = sim.identify_flexible_regions(
    temperatures=[343, 353, 363]
)
```

## 기능

- pH 7.0에서 적절한 양성자화 상태로 구조 준비
- 삼사정계 상자 및 SPC/E 물 모델을 사용한 시스템 설정
- 최급강하법을 사용한 에너지 최소화
- 다음을 사용한 생산 MD:
  - 장거리 정전기학을 위한 입자-메시 Ewald (컷오프 0.8nm)
  - 트윈 범위 전위 (0.8/1.4nm)를 사용한 반 데르 발스 상호 작용
  - 수소 결합을 위한 LINCS 알고리즘
  - Nose-Hoover 온도 조절기
  - Parrinello-Rahman 압력 조절기
  - 다중 온도 지원 (343K, 353K, 363K, 400K)

## 분석 기능

- RMSD 계산
- RMSF 분석
- 회전 반경
- 용매 접근 가능 표면적
- 수소 결합
- 유연한 영역 식별

## 디렉터리 구조

```
md_simulations/
├── src/
│   └── md_simulation.py
├── structures/
├── config/
├── requirements.txt
└── README.md
```

## 참고

- 구현은 참조에 지정된 정확한 프로토콜을 따릅니다.
- 모든 시뮬레이션은 기본적으로 100ns 동안 실행됩니다.
- 유연한 영역은 여러 온도에서 RMSF 분석을 사용하여 식별됩니다.
- N-말단 잔기는 유연성 분석에서 제외됩니다.
- GROMACS의 분석 도구는 궤적 분석에 사용됩니다.

## 참조

구현은 다음을 사용하여 참조 프로토콜에 설명된 방법론을 따릅니다.
- GROMACS 5.1.2
- CHARMM36 힘 필드
- SPC/E 물 모델
- 양성자화 상태를 위한 H++ 서버
