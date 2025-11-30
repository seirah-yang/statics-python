## 1. 사전 준비 (Safety Set, TEAE 필터)
import pandas as pd

adae = pd.read_csv("mock_ADAE.csv")   # 이전에 만든 ADAE 가정
adsl = pd.read_csv("mock_ADSL.csv")   # 없으면 DM을 ADSL처럼 사용해도 됨

# Safety Set: 여기서는 "적어도 1회 EX 기록 존재" or 단순 DM all treated 가정
# adsl에 SAFFL 있고, ARM 정보 포함 가정
saffl = adsl[adsl["SAFFL"] == "Y"][["USUBJID","ARM"]].copy()

# ADAE + Safety Set merge
adae = adae.merge(saffl, on="USUBJID", how="inner")  # Safety set만

# TEAE만 선택
adae_teae = adae[adae["TRTEMFL"] == "Y"].copy()


## 2. Treatment Group별 N (분모)
# 분모: 각 ARM 내 Safety Set N
denom = saffi = saffl.groupby("ARM")["USUBJID"].nunique()
# denom는 Series: ARM → N


## 3. “Any TEAE / Any Serious TEAE” 요약
# Any TEAE
any_teae = (adae_teae.groupby(["ARM","USUBJID"])
            .size()
            .reset_index(name="HAS_TEAE"))
any_teae["HAS_TEAE"] = 1
any_teae_summary = (any_teae.groupby("ARM")["HAS_TEAE"]
                    .sum()
                    .reindex(denom.index)
                    )

# Any Serious TEAE
ser_teae = (adae_teae[adae_teae["AESER"] == "Y"]
            .groupby(["ARM","USUBJID"])
            .size()
            .reset_index(name="HAS_SAE"))
ser_teae["HAS_SAE"] = 1
ser_teae_summary = (ser_teae.groupby("ARM")["HAS_SAE"]
                    .sum()
                    .reindex(denom.index)
                    )


## 4. PT별 요약 (AEDECOD 기준)
# 피험자 단위 집계: arm, decod, subject
pt_subj = (adae_teae
           .groupby(["ARM","AEDECOD","USUBJID"])
           .size()
           .reset_index(name="CNT"))
pt_subj["SUBJ_FLAG"] = 1

pt_summary = (pt_subj
              .groupby(["ARM","AEDECOD"])["SUBJ_FLAG"]
              .sum()
              .reset_index()
              )

# 비율 계산
pt_summary = pt_summary.merge(
    denom.rename("DENOM").reset_index(),
    on="ARM",
    how="left"
)
pt_summary["PCT"] = 100 * pt_summary["SUBJ_FLAG"] / pt_summary["DENOM"]

# 예: ARM별로 가로 pivot
pt_table = (pt_summary
            .pivot_table(index="AEDECOD",
                         columns="ARM",
                         values=["SUBJ_FLAG","PCT"],
                         aggfunc="first")
           )
