import pandas as pd

def create_adlb(lb: pd.DataFrame, dm: pd.DataFrame) -> pd.DataFrame:
    """
    SDTM 스타일 LB + DM → ADLB 생성 템플릿
    필수 컬럼 가정:
      LB: STUDYID, USUBJID, LBTEST, LBORRES, LBORRESU, LBDTC, LBNRIND
      DM: USUBJID, RFSTDTC
    """
    lb = lb.copy()
    dm = dm.copy()

    # 날짜 처리
    to_date = lambda x: pd.to_datetime(x, errors="coerce")
    lb["LBDT"] = to_date(lb["LBDTC"])
    dm["RFSTDTC"] = to_date(dm["RFSTDTC"])

    # DM merge (기준일)
    adlb = lb.merge(dm[["USUBJID", "RFSTDTC"]], on="USUBJID", how="left")

    # PARAM/PARAMCD
    # 간단하게 LBTEST를 PARAM, 대문자/언더스코어를 PARAMCD로 설정
    adlb["PARAM"] = adlb["LBTEST"]
    adlb["PARAMCD"] = adlb["LBTEST"].str.upper().str.replace(" ", "_")

    # 분석 날짜(ADT), Study Day(ADY)
    adlb["ADT"] = adlb["LBDT"]
    adlb["ADY"] = (adlb["ADT"] - adlb["RFSTDTC"]).dt.days + 1

    # AVISIT (예: visit 번호를 VSDAY나 LBSEQ/패턴에서 파생 가능)
    # 여기서는 간단히 LBSEQ 기준 visit label 부여 (실제는 일정표에 맞춰 mapping)
    if "LBSEQ" in adlb.columns:
        adlb["AVISITN"] = adlb["LBSEQ"] // 10  # mock 규칙 예시
        adlb["AVISIT"] = "Visit " + adlb["AVISITN"].astype(str)
    else:
        adlb["AVISIT"] = "Visit 1"
        adlb["AVISITN"] = 1

    # 분석 값
    adlb["AVAL"] = adlb["LBORRES"]
    adlb["AVALU"] = adlb["LBORRESU"]

    # Baseline 정의: AVISITN == 1 (혹은 ADY <= 1 등) 예시
    adlb["ABLFL"] = ""
    baseline_mask = adlb["AVISITN"] == 1
    adlb.loc[baseline_mask, "ABLFL"] = "Y"

    # BASE: 피험자+PARAMCD 기준 baseline 레코드의 AVAL
    base = (adlb[adlb["ABLFL"] == "Y"]
            .groupby(["USUBJID", "PARAMCD"])["AVAL"]
            .first()
            .reset_index()
            .rename(columns={"AVAL": "BASE"})
            )

    adlb = adlb.merge(base, on=["USUBJID", "PARAMCD"], how="left")

    # CHG = AVAL − BASE
    adlb["CHG"] = adlb["AVAL"] - adlb["BASE"]

    # BNRIND: LBNRIND 그대로 사용 (LOW/HIGH/NORMAL)
    adlb["BNRIND"] = adlb["LBNRIND"]

    # ASEQ: 분석 순번
    adlb = adlb.sort_values(["USUBJID", "PARAMCD", "ADT", "AVISITN"])
    adlb["ASEQ"] = adlb.groupby(["USUBJID", "PARAMCD"]).cumcount() + 1

    # 최소 컬럼셋 예시
    adlb_final = adlb[[
        "STUDYID","USUBJID","ASEQ",
        "PARAMCD","PARAM","AVISITN","AVISIT",
        "ADT","ADY",
        "AVAL","AVALU","BASE","CHG",
        "BNRIND","ABLFL"
    ]]

    return adlb_final
