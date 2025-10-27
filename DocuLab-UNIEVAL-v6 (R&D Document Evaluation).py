!pip install html2text
!pip install sentence-transformers transformers torch python-docx PyPDF2 scikit-learn pandas tqdm

import torch
print("CUDA available:", torch.cuda.is_available())
# ===========================================================
# Folder-based Evaluation + Visualization Report (Colab)
# ===========================================================
import os, re, sys, subprocess, time
import pandas as pd, numpy as np
from pathlib import Path
from tqdm import tqdm

# --- Optional: lightweight install guards (Colab-friendly)
def _maybe_pip_install(pkg):
    try:
        __import__(pkg)
    except Exception:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=False)

for _pkg in ["python-docx", "PyPDF2", "sentence_transformers", "transformers", "plotly", "html2text"]:
    # import name != pip name for python-docx/sentence-transformers handled above
    pass

# Core imports (after optional install)
from docx import Document
from PyPDF2 import PdfReader
import plotly.graph_objects as go

import torch
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline

# -------------------------------
# 1️⃣ Google Drive mount
# -------------------------------
try:
    from google.colab import drive, files
    drive.mount('/content/drive')
    IN_COLAB = True
except Exception:
    IN_COLAB = False
    print("[WARN] Not running in Colab. Drive mount skipped.")
SAVE_DIR = "/content/drive/MyDrive/unieval_results" if IN_COLAB else "./unieval_results"
os.makedirs(SAVE_DIR, exist_ok=True)

# -------------------------------
# 2️⃣ Core model setup
# -------------------------------
EMBEDDING_MODEL = "intfloat/e5-large"
NLI_MODEL = "joeddav/xlm-roberta-large-xnli"
WEIGHTS = {"accuracy":0.35,"relevance":0.15,"coherence":0.15,
           "fluency":0.10,"consistency":0.15,"redundancy":0.10}
REDUNDANCY_TH = 0.90

_embedder, _nli = None, None

def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _embedder

def get_nli():
    """
    Robust NLI pipeline:
    - return_all_scores=True → 예측 분포 확보
    - truncation=True → 긴 문장 안전 처리
    - device 자동 선택
    """
    global _nli
    if _nli is None:
        tok = AutoTokenizer.from_pretrained(NLI_MODEL)
        mdl = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL)
        device = 0 if torch.cuda.is_available() else -1
        _nli = TextClassificationPipeline(
            model=mdl,
            tokenizer=tok,
            return_all_scores=True,
            truncation=True,
            device=device
        )
    return _nli

# -------------------------------
# 3️⃣ Text loading utilities
# -------------------------------
def load_text(path):
    suf = Path(path).suffix.lower()
    try:
        if suf==".docx":
            return "\n".join([p.text for p in Document(path).paragraphs])
        elif suf==".pdf":
            return "\n".join([(p.extract_text() or "") for p in PdfReader(path).pages])
        elif suf==".txt":
            return Path(path).read_text(encoding="utf-8", errors="ignore")
        else:
            print(f"[INFO] Skip unsupported file: {path}")
            return ""
    except Exception as e:
        print(f"[ERROR] load_text failed for {path}: {e}")
        return ""

def sent_split(text):
    # 한국어 종결(다/요/까/함) + 영문 문장부호 혼용
    # 너무 짧은 토큰(2자 이하) 제거
    chunks = re.split(r'(?<=[\.!\?]|다|요|까|함)\s+', text)
    return [s.strip() for s in chunks if len(s.strip()) > 2]

# -------------------------------
# 4️⃣ Metric functions
# -------------------------------
def cosine_redundancy(sents, emb):
    """1(좋음) ~ 0(나쁨): 문장 간 과도한 중복이 많을수록 점수↓"""
    n = len(sents)
    if n < 2:
        return 1.0
    embs = emb.encode(sents, normalize_embeddings=True)
    sim = util.cos_sim(embs, embs).cpu().numpy()
    # 상삼각만 사용하여 중복 카운트 (대칭/자기자신 제외)
    triu_idx = np.triu_indices(n, k=1)
    pair_sims = sim[triu_idx]
    redundant_ratio = np.mean(pair_sims > REDUNDANCY_TH) if pair_sims.size > 0 else 0.0
    return float(1 - redundant_ratio)

def coherence_score(sents, emb):
    """인접 문장간 의미 유사도 평균 (높을수록 응집/coherence↑)"""
    if len(sents) < 2:
        return 0.0
    embs = emb.encode(sents, normalize_embeddings=True)
    sims = [util.cos_sim(embs[i], embs[i+1]).item() for i in range(len(embs)-1)]
    return float(np.mean(sims)) if sims else 0.0

def fluency_score(sents):
    """문장 길이 분산 기반 유창성 근사치 (길이 표준편차/평균 비율 역수)"""
    tokens = [len(s.split()) for s in sents]
    if not tokens:
        return 0.0
    mean_len = np.mean(tokens)
    std_len = np.std(tokens)
    ratio = max(0.0, 1 - std_len / (mean_len + 1e-6))
    return float(round(ratio, 3))

def relevance_score(title, text, emb):
    """제목(query) vs 본문(text)의 의미 유사도"""
    if not text.strip():
        return 0.0
    embs = emb.encode([f"query: {title}", text], normalize_embeddings=True)
    return float(util.cos_sim(embs[0], embs[1]))

def _parse_nli_result(res):
    """HF pipeline 결과를 안전 파싱하여 (label, score) 중 스코어 최대 항목 반환"""
    # 가능한 형태: dict / [dict] / [[dict,...]]
    first = res
    if isinstance(first, list) and first:
        first = first[0]
    if isinstance(first, list) and first:
        best = max(first, key=lambda x: x.get("score", 0.0))
    elif isinstance(first, dict):
        best = first
    else:
        # 예상치 못한 구조 → 보수적 처리
        return ("NEUTRAL", 0.0)
    return (best.get("label", "NEUTRAL").upper(), best.get("score", 0.0))

def consistency_score(sents):
    """
    문장 간 논리 일관성: 인접 문장 간 NLI로 CONTRADICTION 비율을 패널티화
    점수 = 1 - contra / max(entail+contra, 1)
    """
    if len(sents) < 2:
        return 1.0
    nli = get_nli()
    pairs = [(sents[i], sents[i+1]) for i in range(len(sents)-1)]
    entail = contra = unknown = 0

    for a, b in pairs:
        try:
            res = nli({"text": a, "text_pair": b})
            lab, _ = _parse_nli_result(res)
            if "ENTAIL" in lab:
                entail += 1
            elif "CONTRAD" in lab:
                contra += 1
            else:
                unknown += 1
        except Exception as e:
            # 파이프라인/토큰화 오류 등은 unknown 처리
            unknown += 1

    denom = max(entail + contra, 1)
    score = 1.0 - (contra / denom)
    return float(max(0.0, min(1.0, score)))

def accuracy_score(claims, evid):
    """
    Accuracy(정합성): claim 문장(숫자/목표 포함) vs evidence(기타 문장) 간
    최대 유사도의 평균. 0~1 (높을수록 claim이 내부 근거로 잘 뒷받침됨)
    """
    if not claims or not evid:
        return 0.0
    emb = get_embedder()
    c_emb = emb.encode(claims, normalize_embeddings=True)
    e_emb = emb.encode(evid, normalize_embeddings=True)
    sims = util.cos_sim(c_emb, e_emb).cpu().numpy()
    return float(np.mean(np.max(sims, axis=1))) if sims.size > 0 else 0.0

# -------------------------------
# 5️⃣ Evaluation functions
# -------------------------------
def evaluate_section(title, text):
    emb = get_embedder()
    sents = sent_split(text)

    # 간단한 claim 추출 휴리스틱(숫자/목표/성과/달성 키워드)
    claims = [s for s in sents if re.search(r'\d|목표|성과|달성', s)]
    evid   = [s for s in sents if s not in claims]

    metrics = {
        "accuracy":    accuracy_score(claims, evid),
        "relevance":   relevance_score(title, text, emb),
        "coherence":   coherence_score(sents, emb),
        "fluency":     fluency_score(sents),
        "consistency": consistency_score(sents),
        "redundancy":  cosine_redundancy(sents, emb),
    }
    metrics["final"] = float(sum(metrics[k] * WEIGHTS[k] for k in WEIGHTS))
    return metrics

def evaluate_folder(target_folder):
    target = Path(target_folder)
    files_to_eval = [f for f in target.glob("*") if f.suffix.lower() in [".docx",".pdf",".txt"]]

    if not files_to_eval:
        print(f"[WARN] No evaluable files in: {target_folder}")
        return pd.DataFrame([])

    all_results = []
    for f in tqdm(files_to_eval, desc="Evaluate"):
        try:
            text = load_text(f)
            title = f.stem
            result = evaluate_section(title, text)
            result["file"] = f.name
            all_results.append(result)
            print(f"✅ {f.name} → 평가완료 (final={result['final']:.3f})")
        except Exception as e:
            print(f"❌ {f.name} → 평가 실패: {e}")

    if not all_results:
        print("[WARN] No results produced.")
        return pd.DataFrame([])

    df = pd.DataFrame(all_results)
    csv_path = os.path.join(SAVE_DIR, "results_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n📊 CSV 저장 완료: {csv_path}")

    generate_report_html(df, SAVE_DIR)
    return df

# -------------------------------
# 6️⃣ Visualization Report
# -------------------------------
def generate_report_html(df, save_dir):
    if df is None or df.empty:
        print("[WARN] Empty DataFrame. Skip report generation.")
        return

    metrics = ["accuracy","relevance","coherence","fluency","consistency","redundancy"]
    # 결측 방지
    for m in metrics + ["final"]:
        if m not in df.columns:
            df[m] = np.nan
    df = df.fillna(0.0)

    mean_scores = df[metrics].mean(numeric_only=True)
    values = [float(mean_scores.get(m, 0.0)) for m in metrics]

    # Radar Chart
    radar = go.Figure()
    radar.add_trace(go.Scatterpolar(
        r=values + [values[0]],
        theta=metrics + [metrics[0]],
        fill='toself',
        name='평균 점수'
    ))
    radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0,1])),
        title="DocuLab UNIEVAL R&D Radar Chart"
    )

    # Bar Chart
    bar = go.Figure([go.Bar(x=df["file"], y=df["final"])])
    bar.update_layout(title="문서별 Final Score", xaxis_title="문서", yaxis_title="점수(0~1)")

    # Combine HTML
    html_path = os.path.join(save_dir, "evaluation_report.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write("<h2>DocuLab UNIEVAL R&D Evaluation Report</h2>")
        f.write("<p>자동 문서 품질평가 결과입니다.</p>")
        f.write(radar.to_html(full_html=False, include_plotlyjs='cdn'))
        f.write(bar.to_html(full_html=False, include_plotlyjs=False))
        f.write(df.to_html(index=False))
    print(f"📈 시각화 리포트 저장 완료: {html_path}")

# -------------------------------
# 7️⃣ Example Run
# -------------------------------
target_folder = "/content/drive/MyDrive/1027" if IN_COLAB else "./samples"
df = evaluate_folder(target_folder)   # → results_summary.csv + evaluation_report.html 생성

# ===========================================================
# 8️⃣ HTML → Markdown 변환
# ===========================================================
html_path = os.path.join(SAVE_DIR, "evaluation_report.html")
outpath = "/content/evaluation_report.md" if IN_COLAB else "./evaluation_report.md"

if os.path.exists(html_path):
    try:
        import html2text
    except Exception:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "html2text"], check=False)
        import html2text

    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    md_content = html2text.html2text(html_content)

    with open(outpath, "w", encoding="utf-8") as f:
        f.write(md_content)

    print("\n✅ 보고서 생성 완료:", outpath)
    if IN_COLAB:
        from google.colab import files
        time.sleep(1)
        files.download(outpath)
else:
    print("⚠️ 결과 파일이 생성되지 않았습니다. 오류 로그를 확인해주세요.")
