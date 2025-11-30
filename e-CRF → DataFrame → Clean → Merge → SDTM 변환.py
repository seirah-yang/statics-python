import pandas as pd
from datetime import datetime

# 1. Load Raw eCRF Data
df_dm = pd.read_excel("RAW_DM.xlsx")
df_ae = pd.read_excel("RAW_AE.xlsx")

# 2. Standardize columns
df_dm.columns = df_dm.columns.str.upper()
df_ae.columns = df_ae.columns.str.upper()

# 3. Clean data
df_dm['BRTHDAT'] = pd.to_datetime(df_dm['BRTHDAT'], errors='coerce')
df_dm['AGE'] = pd.to_numeric(df_dm['AGE'], errors='coerce')
df_ae['AESTDAT'] = pd.to_datetime(df_ae['AESTDAT'], errors='coerce')

# 4. Create USUBJID
df_dm['USUBJID'] = df_dm['SITEID'].astype(str) + "-" + df_dm['SUBJID'].astype(str)
df_ae['USUBJID'] = df_ae['SITEID'].astype(str) + "-" + df_ae['SUBJID'].astype(str)

# 5. Merge DM + AE
df_profile = df_ae.merge(df_dm[['USUBJID','AGE','SEX']], on='USUBJID', how='left')

# 6. Build SDTM DM Domain
df_sdtm_dm = pd.DataFrame({
    'STUDYID': 'ABC123',
    'DOMAIN': 'DM',
    'USUBJID': df_dm['USUBJID'],
    'SUBJID': df_dm['SUBJID'],
    'BRTHDTC': df_dm['BRTHDAT'].dt.strftime('%Y-%m-%d'),
    'SEX': df_dm['SEX'],
    'AGE': df_dm['AGE']
})

# 7. SDTM column order
dm_cols = ['STUDYID','DOMAIN','USUBJID','SUBJID','BRTHDTC','SEX','AGE']
df_sdtm_dm = df_sdtm_dm[dm_cols]

print(df_sdtm_dm.head())
