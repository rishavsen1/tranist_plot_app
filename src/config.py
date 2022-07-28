# Datasets config
MTA_PARQUET_PREPROCESS = "cleaned-wego-daily.apc.parquet"
CARTA_PARQUET_PREPROCESS = "carta-apc.parquet"

MTA_PARQUET = "processed_parquet_JP_all"
CARTA_PARQUET = "carta_apc_out.parquet"

# Prediction config
TARGET_MAX = 100
CLASS_BINS = [0, 33, 66, 100]
TARGET_COLUMN_REGRESSION = 'y_reg'
TARGET_COLUMN_CLASSIFICATION = 'y_class'