def create_advs(vs: pd.DataFrame, dm: pd.DataFrame) -> pd.DataFrame:
    """
    SDTM 스타일 VS + DM → ADVS 생성 템플릿
    필수 컬럼:
      VS: STUDYID, USUBJID, VSTEST, VSORRES, VSORRESU, VSDTC
      DM: USUBJID, RFSTDTC
    """
    vs = vs.copy()
    dm = dm.copy()

    to_date = lambda x: pd.to_datetime(x, errors="coerce")
    vs["VSDT"] = to_date(vs["VSDTC"])
    dm["RFSTDTC"] = to_date(dm["RFSTDTC"])

    advs = vs.merge(dm[["USUBJID","RFSTDTC"]], on="USUBJID", how="left")

    # PARAM/PARAMCD
    advs["PARAM"] = advs["VSTEST"]
    advs["PARAMCD"] = advs["VSTEST"].str.upper().str.replace(" ", "_")

    # ADT, ADY
    advs["ADT"] = advs["VSDT"]
    advs["ADY"] = (advs["ADT"] - advs["RFSTDTC"]).dt.days + 1

    # AVISIT/AVISITN 예시 (VSSEQ 기준)
    if "VSSEQ" in advs.columns:
        advs["AVISITN"] = advs["VSSEQ"] // 10
        advs["AVISIT"] = "Visit " + advs["AVISITN"].astype(str)
    else:
        advs["AVISITN"] = 1
        advs["AVISIT"] = "Visit 1"

    # 분석 값
    advs["AVAL"] = advs["VSORRES"]
    advs["AVALU"] = advs["VSORRESU"]

    # Baseline: AVISITN == 1
    advs["ABLFL"] = ""
    base_mask = advs["AVISITN"] == 1
    advs.loc[base_mask, "ABLFL"] = "Y"

    base = (advs[advs["ABLFL"] == "Y"]
            .groupby(["USUBJID","PARAMCD"])["AVAL"]
            .first()
            .reset_index()
            .rename(columns={"AVAL": "BASE"})
            )

    advs = advs.merge(base, on=["USUBJID","PARAMCD"], how="left")

    advs["CHG"] = advs["AVAL"] - advs["BASE"]

    # ASEQ
    advs = advs.sort_values(["USUBJID","PARAMCD","ADT","AVISITN"])
    advs["ASEQ"] = advs.groupby(["USUBJID","PARAMCD"]).cumcount() + 1

    advs_final = advs[[
        "STUDYID","USUBJID","ASEQ",
        "PARAMCD","PARAM","AVISITN","AVISIT",
        "ADT","ADY",
        "AVAL","AVALU","BASE","CHG",
        "ABLFL"
    ]]

    return advs_final
