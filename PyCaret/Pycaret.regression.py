# **PyCaret 환경 오류 해결 Flow 정리**

# **1. 초기 문제**

- ModuleNotFoundError: No module named 'pycaret.regression'
- 원인:
    - PyCaret이 user-site (~/.local/lib/python3.8/site-packages) 에 깨진 상태로 설치되어 있었음
    - 따라서 conda 환경(py09) 내부 패키지가 아닌 잘못된 경로가 먼저 import 됨

# **2. 문제 원인 파악**

- sys.executable 확인 → Python은 py09 환경을 가리키고 있었음.
- 하지만 pycaret @ /home/alpaco/.local/... 로 찍히는 걸로 봐서 user-site 버전이 로드됨

## **3. 조치 단계**

1)User-site 패키지 제거

`rm -rf ~/.local/lib/python3.8/site-packages/pycaret*`

2)pth 파일 내 PyCaret 경로 삽입 여부도 점검/삭제

`python -m pip install --no-user --no-cache-dir "pycaret[full]==3.2.0"
python -m pip install "joblib==1.2.0" "scikit-learn==1.2.2" "imbalanced-learn==0.10.1" \
"threadpoolctl>=2.2.0,<4" "matplotlib>=3.5" \
"scipy>=1.10,<1.13" "pandas>=1.3,<2.2"`

3)user-site 강제 차단 후 테스트

`PYTHONNOUSERSITE=1 python -c "import pycaret; print(pycaret.**file**)"`

 → 경로가 py09 내부로 찍히면 정상.

1. 부족한 의존성 보완
- threadpoolctl이 user-site에만 있었기 때문에 다시 설치:

`conda install -n py09 threadpoolctl=3.5`

- 이후 numpy, scipy 등도 버전 호환 맞춤

`‘’‘ 터미널에서 시행`

`PYTHONNOUSERSITE=1 python - <<'PY'
from pycaret.regression import setup, compare_models, pull
import pandas as pd
df = pd.DataFrame({"x":[1,2,3,4,5,6,7,8], "y":[1.2,1.9,3.1,4.1,5.2,5.9,7.1,7.9]})
_ = setup(df, target="y", session_id=42, silent=True, verbose=False)
best = compare_models(n_select=1)
print("BEST:", best)
print(pull().head())
PY`

- 정상 실행 → PyCaret 완전 동작 확인

**5. 정리**

- 항상 python -m pip 사용 → 현재 환경 안에 정확히 설치됨.
- user-site (~/.local) 는 가급적 비활성화 (PYTHONNOUSERSITE=1)
- conda 환경마다 필요한 패키지는 환경 안에만 설치.
- 의존성 버전은 PyCaret 문서에서 권장 조합대로 맞추는 게 안정적.
