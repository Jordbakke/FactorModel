import pandas as pd
import numpy as np
import sys
import os
import gc
import itertools
import random
import tqdm
from functools import reduce
# Add the parent directory to sys.path
sys.path.append(r"C:\repos\Deep-learning-trj")
from data.code import data_utils
from data.code.llm_utils import OpenaiModel, HuggingFaceModel
from sentence_transformers import SentenceTransformer, models
from dotenv import load_dotenv

load_dotenv(r"C:\repos\Deep-learning-trj\.env")

SYSTEM_MESSAGES = {"business_model_generator": {"role": "system", "content": """You are an expert business analyst. When provided with a company name and a specific year, your job is to generate a description of the provided company's business model as of the end of the specified year.
Describe only what was known up to that time, with absolutely no information or insights from beyond that year.
Style: Use the minimum number of tokens necessary to convey the details. The language must be simple, clear, and straightforward, focusing only on concrete information. Avoid stylistic or descriptive phrases. The description cannot exceed 300 words
Do not write a title but use subtitles to clearly separate different aspects of the business model. Use the ticker rather than full company name to save tokens. E.g if the provided ticker is "AAPL-US" use "AAPL" in the description. (s) means that it could be singular or plural.
It is extremely important that you do not hallucinate or make up things. Be strictly factual.
"""
},

}
USER_MESSAGES = {

"business_model_generator": {"role": "user", "content": """Describe the business model of {}, ticker {}, at the end of {}. The business model description shall include the following aspects:
                               
Sector and Supply Chain Position:  In {}, which sector did the company operate in, and what role did the company have in the supply chain?
Business Area: What were the company's business area(s) in {}? List the business area(s) as bullet points. If it's just one, write it as a single bullet point. Ignore insignificant, non-core business areas.
Business Sub-Segment: In {}, which sub-segment(s) did the company's product(s) serve?. Go deeper into the business area(s) to reach the granular sub-segment(s). Do not write about customer types. Do not use company specific product names. 
Product Offering: Granular description of the company's product(s) or service(s) and what purpose they served in {}. If the company offered assets, such as vessels, real estate or rigs, describe their technological specialization, their target segment, and the age composition of the capital assets at the time. Do not use company specific product names.
Revenue Model: Description of how the company generated revenue and what its primary revenue sources were in {}. Do not use company specific product names.
Customer Segments: Description of the company's customer segment(s) in {}. Go deep into the customer base and identify what kind of customers the company served. List the customer segment(s) as bullet point(s).
Market Position: Short description of the how the company's market position was compared to competitors in {}.
Cost Structure: What were the company's 3 most important cost drivers in {}? Write about the services, components and materials the company used to provide its own product(s) or service(s) rather than broad cost categories, such as "manufacturing costs". When applicable, name the specific components and commodities. List the cost drivers as as bullet points.
Geographic Regions:  In {}, what were the key geographic region(s) where the company generated significant revenue? List the region(s) as bullet point(s). If it was just one region, write it as a single bullet point.
Risk: Description of what the 2 key risk factors were in {}. Don't mention competition. List them as bullet points.
Market Growth Drivers: In {}, what were the key drivers of market growth for the company's product(s) or service(s)? List them as bullet points.
Strengths: In {}, what were the key strengths of the business? List them as bullet points. 
Weaknesses: In {}, what were the key weaknesses of the business? List them as bullet points.
"""
}
}

def custom_format(user_message, val1, val2, val3):
    # Split the template string into parts using '{}' as the delimiter
    val1 = str(val1)
    val2 = str(val2)
    val3 = str(val3)

    parts = user_message.split('{}')
    
    # Insert the first and second values into the first two placeholders
    formatted = f"{parts[0]}{val1}{parts[1]}{val2}"
    
    # For any remaining parts, join them with val3
    if len(parts) > 2:
        formatted += val3.join(parts[2:])
    
    return formatted

def process_fundamentals(fundamentals_raw_df: pd.DataFrame,
                              max_null_percentage=0.95, sequence_length=20, columns_to_drop=(
    "currency", "fsym_id", "ff_upd_type","ff_fp_ind_code",
    "ff_report_frequency", "ff_fyr", "ff_fy_length_days", "ff_fpnc",
    "adjdate", "ff_fiscal_date", "ff_eps_rpt_date", "ff_source_is_date", "ff_source_bs_date",
    "ff_source_cf_date", "ff_dps_ddate", "ff_dps_exdate",
    "ff_source_doc", "ff_fp_ind_code", "ff_actg_standard", "date", "ff_report_freq_code", "middle_date",
    "fsym_regional_id"
    )) -> pd.DataFrame:

    #"ff_com_shs_out_eps_basic", "ff_com_shs_out_eps_dil","ff_com_shs_out",
    #"ff_com_shs_out_eps", "ff_shs_repurch_auth_shs"

    fundamentals_df = fundamentals_raw_df[
    (
        (fundamentals_raw_df["ff_fy_length_days"] < 100) &
        (fundamentals_raw_df["ff_fy_length_days"] > 80)
    ) |
    (fundamentals_raw_df["ff_report_freq_code"] == 1)
    ] #only include quarterly fiscal periods

    fundamentals_df = fundamentals_df[fundamentals_df["currency"] == "USD"] # only companies reporting in USD
    #insert 91 days when ff_fy_length_days is null
    fundamentals_df["ff_fy_length_days"] = fundamentals_df["ff_fy_length_days"].fillna(91)
    fundamentals_df = fundamentals_df.dropna(subset=["date"])

    if not pd.api.types.is_datetime64_any_dtype(fundamentals_df["date"]):
        fundamentals_df["date"] = pd.to_datetime(fundamentals_df["date"])

    fundamentals_df["middle_date"] = fundamentals_df["date"] - pd.to_timedelta(fundamentals_df["ff_fy_length_days"] / 2, unit="D")
    fundamentals_df["calendar_quarter"] = fundamentals_df["middle_date"].dt.to_period("Q")

    # fundamentals_df = fundamentals_df[~((fundamentals_df['calendar_quarter'] == pd.Period("1994-08", freq="Q")) & (fundamentals_df['ticker_region'] == "AAPL-US"))]

    fundamentals_df["release_date"] = fundamentals_df["ff_source_is_date"].combine_first(fundamentals_df["ff_source_bs_date"]).combine_first(fundamentals_df["ff_source_cf_date"])

    if not pd.api.types.is_datetime64_any_dtype(fundamentals_df["release_date"]):
        fundamentals_df["release_date"] = pd.to_datetime(fundamentals_df["release_date"])

    fundamentals_df["date_release_diff"] = fundamentals_df["release_date"] - fundamentals_df["date"]
    fundamentals_df["release_date"] = fundamentals_df[["date", "release_date", "date_release_diff"]].apply(
        lambda row: row["date"] + pd.DateOffset(days=90)
        if row["date_release_diff"] > pd.Timedelta(days=120)
        else row["release_date"],
        axis=1
    ) #When the release date is more than 120 days after the date, set the release date to 90 days after the date because oftentimes the
    # financials were released earlier, but minor adjustments were announced later.

    fundamentals_df = fundamentals_df.drop(columns=list(columns_to_drop), errors="ignore")
    fundamentals_df = data_utils.add_year_month_column(fundamentals_df, "release_date", "release_month", drop_date_column=True)
    fundamentals_df = fundamentals_df.drop_duplicates(subset=["calendar_quarter", "ticker_region"])
    fundamentals_df = data_utils.remove_columns_with_nulls(fundamentals_df, max_null_percentage=max_null_percentage)

    non_numeric_cols = ["ticker_region", "calendar_quarter", "release_month"]
    numeric_cols = [col for col in fundamentals_df.columns if col not in non_numeric_cols]
    fundamentals_df[numeric_cols] = fundamentals_df[numeric_cols].apply(pd.to_numeric, errors='coerce').astype(float) #convert columns which should be numeric

    #Get missing features mask before imputing values
    missing_features_masks = fundamentals_df[numeric_cols].isna().astype(float).to_numpy().tolist()
    missing_features_masks = pd.Series(missing_features_masks, name="missing_features_mask", index=fundamentals_df.index)
    fundamentals_df = pd.concat([fundamentals_df, missing_features_masks], axis=1)

    # impute missing features with 0. most of nan values are actually 0 (extraordinary costs etc)
    fundamentals_df[numeric_cols] = fundamentals_df[numeric_cols].fillna(0)
    fundamentals_df = data_utils.standardize_time_windows(df=fundamentals_df, period_col="calendar_quarter", period_freq="Q",
                                                          num_periods=4, cols_to_transform=numeric_cols, include_current_period=True,
                                                          min_periods=4) #standardize the values for the last 4 quarters
    fundamentals_df[numeric_cols] = fundamentals_df[numeric_cols].fillna(0) # NaN occurs when the standard deviation is 0, so the standardized values are undefined.
    
    fundamentals_df = fundamentals_df.groupby("ticker_region", group_keys=False)[fundamentals_df.columns].apply(
    lambda group: data_utils.add_missing_timestamps(
        df=group, period_column="calendar_quarter", freq="Q", entity_columns=["ticker_region"], include_padding_mask=True, padding_mask_col="padding_mask",
                        pad_value=0, numeric_cols=numeric_cols, include_missing_features_mask=True, missing_features_mask_col="missing_features_mask"
    ), include_groups=True
    ).reset_index(drop=True)
    #add release_month to padded timestamps
    if fundamentals_df["padding_mask"].any():
        fundamentals_df.loc[fundamentals_df["padding_mask"] == True, "release_month"] =  fundamentals_df.loc[fundamentals_df["padding_mask"] == True, "calendar_quarter"].apply(
            lambda x: x.end_time.to_period("M"))
    
    fundamentals_df["fundamentals_list"] = fundamentals_df.select_dtypes(include=['number']).to_numpy().tolist()
    fundamentals_df = fundamentals_df.groupby("ticker_region", group_keys=False)[fundamentals_df.columns].apply(lambda group: data_utils.convert_df_to_sequences( #group_keys=False to  have the group col as column and not index
                                        df=group, value_columns=["fundamentals_list"],
                                        date_column="calendar_quarter",
                                        new_column_name="fundamentals_sequence", padding_mask_column="padding_mask",
                                        missing_features_mask_column="missing_features_mask",
                                        sequence_length=sequence_length, include_current_row=True,
                                        drop_empty_sequences=True, direction="previous", drop_value_columns=False)
                                        ).reset_index(drop=True) #include_groups=True, the passed in group will also have the group column

    fundamentals_df = fundamentals_df[["calendar_quarter", "release_month", "ticker_region", "fundamentals_sequence", "padding_mask" ,"missing_features_mask"]]

    return fundamentals_df

def process_prices(prices_raw_df: pd.DataFrame, sequence_length = 60, index_ticker="SPY-US") -> pd.DataFrame:

    """
    Function to process price data which will be used for company embeddings
    """

    assert prices_raw_df["adj_price"].isna().sum() == 0, "There are missing values in the adj_price column" #check for missing features. Should not be any

    prices_df = data_utils.add_year_month_column(prices_raw_df, "price_date", "calendar_month", drop_date_column=True)
    prices_df = prices_df[prices_df["calendar_month"] >= pd.Period("1999-12", freq="M")]
    prices_df = prices_df.groupby("ticker_region", group_keys=False)[prices_df.columns].apply(
        lambda group: data_utils.add_missing_timestamps(
            df=group, period_column="calendar_month", freq="M", entity_columns=["ticker_region"], include_padding_mask=True, padding_mask_col="padding_mask",
            pad_value=np.nan
        )
    ).reset_index(drop=True)

    prices_df = prices_df.ffill()
    
    prices_df = prices_df.sort_values(by="calendar_month")
    prices_df = prices_df.groupby("ticker_region", group_keys=False)[prices_df.columns].apply(
        lambda group: data_utils.log_ratio_transform(
            df=group, cols_to_transform=["adj_price"], new_column_names=["log_adj_price_ratio"], inplace=True
        )
    ).reset_index(drop=True)

    prices_df["release_month"] = prices_df["calendar_month"]

    # Filter for index prices (e.g., SPY)
    index_prices = prices_df[prices_df["ticker_region"] == index_ticker][["calendar_month", "log_adj_price_ratio"]]
    index_prices = index_prices.rename(columns={"log_adj_price_ratio": "index_log_adj_price_ratio"})

    prices_df = prices_df[prices_df["ticker_region"] != index_ticker]
    merged_df = prices_df.merge(index_prices, on=["calendar_month"], how="left")
    merged_df["stock_index_log_adj_price_ratios"] = merged_df[['log_adj_price_ratio', 'index_log_adj_price_ratio']].values.tolist()
    merged_df.dropna(inplace=True) #remove first row as log_adj_price_ratio is NaN for the first row
    merged_df = merged_df.drop_duplicates(subset=["calendar_month", "ticker_region"])
    merged_df = merged_df.groupby("ticker_region")[merged_df.columns].apply(
    lambda group: data_utils.convert_df_to_sequences(
        df=group, value_columns=["stock_index_log_adj_price_ratios"],
        date_column="calendar_month", new_column_name="stock_index_log_adj_price_ratios_sequence",
        padding_mask_column="padding_mask", sequence_length=sequence_length, include_current_row=True,
        drop_empty_sequences=True, direction="previous", drop_value_columns=False
    )).reset_index(drop=True)

    return merged_df[["calendar_month", "release_month", "ticker_region", "stock_index_log_adj_price_ratios_sequence"]]

def process_company_features(company_features_df: pd.DataFrame, columns_to_drop: tuple = ("iso_country", "iso_country_cor"),
                             categorical_columns: tuple = ("iso_country_cor_georev", "industry_code")) -> pd.DataFrame:

    if columns_to_drop is not None:
        company_features_df = company_features_df.drop(columns=columns_to_drop, errors="ignore")

    company_features_df = data_utils.one_hot_encode_categorical_columns(company_features_df, categorical_columns)
    company_features_df = data_utils.standardize_columns(company_features_df, cols_to_transform=["year_founded"])
    company_features_df["company_features_array"] = company_features_df.select_dtypes(include=['number']).to_numpy().tolist()
    company_features_df = company_features_df.drop_duplicates(subset=["ticker_region"])

    return company_features_df

def process_macro_data(macro_directory=r"C:\repos\Deep-learning-trj\data\raw_data_storage\macro_data",
                       monthly_release_date_offset: int = 1, quarterly_release_date_offset=2, sequence_length = 60,
                       release_month_column_name: str = "release_month"):

    def process_helper(date_col: str, numeric_cols: list, df: pd.DataFrame = None, file_path: str = None, pd_read_func=pd.read_csv,
                        rename_cols: dict = None, is_quarterly_data=False, is_daily_data=False,
                        cutoff_month: str="1990-01"
                        ) -> pd.DataFrame:

        """
        Helper function that creates monthly data and renames columns
        """

        if file_path:
            df = pd_read_func(file_path)
        if df is None:
            raise ValueError("No dataframe or file path provided")
        
        df.columns = [col.strip().lower() for col in df.columns]
        
        rename_cols = {key.lower(): value for key, value in rename_cols.items()} if rename_cols else None
        if rename_cols:
            df = df.rename(columns=rename_cols)

        df[date_col] = pd.to_datetime(df[date_col])
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce').astype(float)

        if is_quarterly_data:
            df = data_utils.add_year_quarter_column(df, date_col, "calendar_quarter", drop_date_column=True)
            df["calendar_month"] = df["calendar_quarter"].apply(lambda x: pd.Period(x.end_time, freq="M"))
            df[release_month_column_name] = df["calendar_month"] + quarterly_release_date_offset
            df = df.drop(columns="calendar_quarter")

        elif is_daily_data:
            df = data_utils.add_year_month_column(df, date_col, "calendar_month", drop_date_column=False)
            df = (df.sort_values(date_col).groupby("calendar_month").last().reset_index()).drop(date_col, axis=1)
            df[release_month_column_name] = df["calendar_month"]

        else:
            df = data_utils.add_year_month_column(df, date_col, "calendar_month", drop_date_column=True)
            df[release_month_column_name] = df["calendar_month"] + monthly_release_date_offset
        
        df = df[df["calendar_month"] >= pd.Period(cutoff_month, freq="M")]

        return df

    files_to_process = {
    "10y_2y_treasury_spread": {
        "file_path": os.path.join(macro_directory, "10y_2y_treasury_spread.csv"),
        "date_col": "date",
        "pd_read_func": pd.read_csv,
        "rename_cols": {"T10Y2Y": "10y_2y_spread"},
        "numeric_cols": ["10y_2y_spread"],
        "is_daily_data": True
    },
    "10y_treasury_yield": {
        "file_path": os.path.join(macro_directory, "10y_treasury_yield.csv"),
        "date_col": "date",
        "pd_read_func": pd.read_csv,
        "rename_cols": {"DGS10": "10y_yield"},
        "numeric_cols": ["10y_yield"],
        "is_daily_data": True
    },
    "US_core_inflation": {
        "file_path": os.path.join(macro_directory, "US_core_inflation.csv"),
        "date_col": "date",
        "pd_read_func": pd.read_csv,
        "rename_cols": {"CORESTICKM159SFRBATL": "core_inflation"},
        "numeric_cols": ["core_inflation"]
    },
    "US_corporate_debt": {
        "file_path": os.path.join(macro_directory, "US_corporate_debt.csv"),
        "date_col": "date",
        "pd_read_func": pd.read_csv,
        "rename_cols": {"BCNSDODNS": "corporate_debt", "observation_date": "date"},
        "numeric_cols": ["corporate_debt"],
        "is_quarterly_data": True
    },
    "EBIT_US_companies": {
        "file_path": os.path.join(macro_directory, "EBIT_US_companies.csv"),
        "date_col": "date",
        "pd_read_func": pd.read_csv,
        "rename_cols": {"BOGZ1FA106110115Q": "ebit"},
        "numeric_cols": ["ebit"],
        "is_quarterly_data": True

    },
    "high_yield_spread": {
        "file_path": os.path.join(macro_directory, "high_yield_spread.csv"),
        "date_col": "date",
        "pd_read_func": pd.read_csv,
        "rename_cols": {"BAMLH0A0HYM2": "high_yield_spread"},
        "numeric_cols": ["high_yield_spread"],
        "is_daily_data": True
    },
    "US_unemployment_rate": {
        "file_path": os.path.join(macro_directory, "US_unemployment_rate.csv"),
        "date_col": "date",
        "pd_read_func": pd.read_csv,
        "rename_cols": {"UNRATE": "unemployment_rate"},
        "numeric_cols": ["unemployment_rate"],
    },
    "US_real_gdp": {
        "file_path": os.path.join(macro_directory, "US_real_gdp.csv"),
        "date_col": "date",
        "pd_read_func": pd.read_csv,
        "rename_cols": {"GDPC1": "real_gdp"},
        "numeric_cols": ["real_gdp"],
        "is_quarterly_data": True
    },
    "US_consumer_sentiment": {
        "file_path": os.path.join(macro_directory, "US_consumer_sentiment.csv"),
        "date_col": "date",
        "pd_read_func": pd.read_csv,
        "rename_cols": {"UMCSENT": "consumer_sentiment"},
        "numeric_cols": ["consumer_sentiment"]
    }
    }

    all_macro_dfs = []

    for file in files_to_process:
        file_dict = files_to_process[file]
        df= process_helper(**file_dict)
        all_macro_dfs.append(df)

    #S&P data - From Robert Schiller
    sp_data = pd.read_excel(os.path.join(r"C:\repos\Deep-learning-trj\data\raw_data_storage\macro_data", "robert_schiller_data.xlsx"))
    sp_data["Date"] = sp_data["Date"].astype(str)
    sp_data["Date"] = sp_data["Date"].apply(lambda x: x + "0" if x.endswith(".1") else x)
    sp_data = process_helper(date_col="date", df=sp_data, numeric_cols=["price", "earnings"])
    sp_data = sp_data.sort_values(by="calendar_month")
    sp_data["earnings"] = sp_data.apply(lambda row: row["earnings"] if row["calendar_month"].month in [3, 6, 9, 12] else np.nan, axis=1)
    sp_data["earnings"] = sp_data["earnings"].ffill(limit=3)

    sp_data["pe_3m_lagged_earnings"] = sp_data["price"] / sp_data["earnings"].shift(3) #rolling sum of 12 months
    sp_data = sp_data[["calendar_month", "price", "pe_3m_lagged_earnings"]]
    sp_data[["price", "pe_3m_lagged_earnings"]] = sp_data[["price", "pe_3m_lagged_earnings"]].apply(pd.to_numeric, errors='coerce').astype(float)
    sp_data[release_month_column_name] = sp_data["calendar_month"]
    all_macro_dfs.append(sp_data)

    # Real interest rates
    treasury_yield = process_helper(date_col="date", file_path=os.path.join(macro_directory, "10y_treasury_yield.csv"),
                                    rename_cols={"DGS10": "10y_yield"}, numeric_cols=["10y_yield"], is_daily_data=True).drop(columns=["release_month"])
    core_inflation = process_helper(date_col="date", file_path=os.path.join(macro_directory, "US_core_inflation.csv"),
                                    rename_cols={"CORESTICKM159SFRBATL": "core_inflation"}, numeric_cols=["core_inflation"]).drop(columns=["release_month"])
    real_interest_rates = treasury_yield.merge(core_inflation, on="calendar_month", how="inner")
    real_interest_rates[["10y_yield", "core_inflation"]] = real_interest_rates[["10y_yield", "core_inflation"]].apply(pd.to_numeric, errors='coerce').astype(float)
    real_interest_rates["real_interest_rate"] = (real_interest_rates["10y_yield"] + 1) / (real_interest_rates["core_inflation"] + 1) - 1
    real_interest_rates[release_month_column_name] = real_interest_rates["calendar_month"] + monthly_release_date_offset
    real_interest_rates = real_interest_rates[["calendar_month", "real_interest_rate", release_month_column_name]]
    all_macro_dfs.append(real_interest_rates)

    macro_df = reduce(lambda left_df, right_df: pd.merge(left_df, right_df, on=["calendar_month", "release_month"], how="outer"), all_macro_dfs)
    macro_df = data_utils.release_date_aware_processing_and_sequence_creation(df=macro_df, calendar_period_col="calendar_month",
                                                                         release_period_col="release_month",
                                                                         new_column_name="macro_sequence", sequence_length=sequence_length,
                                                                         cols_to_standardize=["10y_2y_spread", "10y_yield", "core_inflation", "high_yield_spread",
                                                                                                                        "unemployment_rate", "consumer_sentiment", "pe_3m_lagged_earnings", "real_interest_rate"],
                                                                         log_transform_cols=["corporate_debt", "ebit", "real_gdp", "price"],
                                                                         log_transform_shift_value=1
                                                                         
                                                                         )
    return macro_df

def process_commodity_data(macro_directory=r"C:\repos\Deep-learning-trj\data\raw_data_storage\macro_data",
                       cpi_file_path = r"C:\repos\Deep-learning-trj\data\raw_data_storage\macro_data\US_CPI.csv", sequence_length = 60,
                       release_month_column_name: str = "release_month"):
    
    us_cpi = pd.read_csv(cpi_file_path).rename(columns={"CPIAUCSL": "us_cpi", "DATE": "date"})
    us_cpi = data_utils.add_year_month_column(us_cpi, "date", "calendar_month", drop_date_column=True)
    us_cpi = data_utils.add_missing_timestamps(us_cpi, "calendar_month", "M")
    us_cpi = us_cpi.ffill(limit=3)
    us_cpi = us_cpi.sort_values(by="calendar_month")
    us_cpi["us_cpi"] = us_cpi["us_cpi"] / us_cpi["us_cpi"].iloc[0]

    def process_helper(date_col: str, file_path: str = None, pd_read_func=pd.read_csv,
                        rename_cols: dict = None, numeric_cols: list = None
                        ) -> pd.DataFrame:

        df = pd_read_func(file_path)
        if rename_cols:
            df = df.rename(columns=rename_cols)
        df = data_utils.add_year_month_column(df, date_col, "calendar_month", drop_date_column=True)
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce').astype(float)
        df[release_month_column_name] = df["calendar_month"]

        return df

    files_to_process = {
    "oil_prices": {
        "file_path": os.path.join(macro_directory, "oil_prices.xlsx"),
        "date_col": "date",
        "pd_read_func": pd.read_excel,
        "numeric_cols": ["brent", "wti"]
    }
    }

    all_commodity_dfs = []

    for file in files_to_process:
        file_dict = files_to_process[file]
        df= process_helper(**file_dict)
        all_commodity_dfs.append(df)
    
    commodity_df = reduce(lambda left_df, right_df: pd.merge(left_df, right_df, on=["calendar_month"], how="outer"), all_commodity_dfs)
    commodity_df = commodity_df.merge(us_cpi, on="calendar_month", how="left")

    numeric_cols = commodity_df.select_dtypes(include='number').columns.tolist()
    commodity_df[numeric_cols] = commodity_df[numeric_cols].ffill(limit=3)
    numeric_cols = [col for col in numeric_cols if col != "us_cpi"]
    data_utils.open_in_excel(commodity_df)
    commodity_df[numeric_cols] = commodity_df[numeric_cols].div(commodity_df['us_cpi'], axis=0)
    data_utils.open_in_excel(commodity_df)
    commodity_df.drop(columns=["us_cpi"], inplace=True)

    if commodity_df.isna().sum().sum() > 0:
        raise ValueError("There are still missing values in the commodity data")

    commodity_df = data_utils.standardize_columns(commodity_df, cols_to_transform=numeric_cols)
    commodity_df["commodity_values"] = commodity_df[numeric_cols].apply(lambda row: np.nan if row.isna().any() else row.tolist(), axis=1)
    commodity_df = data_utils.convert_df_to_sequences(df=commodity_df, value_columns=["commodity_values"], date_column="calendar_month",
                                                        new_column_name="commodity_sequence", sequence_length=sequence_length, include_current_row=True,
                                                        drop_empty_sequences=True, direction="previous", drop_value_columns=False)

    return commodity_df

def get_stock_returns(prices: pd.Series, periods: int) -> pd.Series:
    return prices.pct_change(periods=periods, fill_method=None)

def append_stock_returns(prices_df: pd.DataFrame, period_column: str, price_col: str, periods: int = 3, entity_columns: list = None) -> pd.DataFrame:

    """
    Function to append rolling stock returns to the prices DataFrame
    """
    if not entity_columns:
        entity_columns = []

    # Sort by ticker_region and date in ascending order for correct forward-looking calculation
    prices_df = prices_df.sort_values(by=[period_column] + entity_columns).drop_duplicates(
        subset=entity_columns + [period_column], keep='first'
    )

    # Calculate rolling returns with ascending order within each group
    prices_df[f"{periods}m_return"] = prices_df.groupby("ticker_region")[price_col].apply(
        lambda x: get_stock_returns(x, periods)
    ).reset_index(level=0, drop=True)

    return prices_df

def append_similarity_metrics(prices_df: pd.DataFrame, sequence_column: str, period_column: str = "calendar_month") -> pd.DataFrame:
    # Extract sequences into a matrix
    sequences = np.vstack(prices_df[sequence_column].values)
    ticker_regions = prices_df['ticker_region'].values
    periods = prices_df[period_column].values

    # Calculate pairwise correlations
    correlation_matrix = np.corrcoef(sequences)

    # Get indices for the upper triangle of the matrix (excluding the diagonal)
    row_idx, col_idx = np.triu_indices_from(correlation_matrix, k=1) #k=1 to exclude diagonal

    # Create a DataFrame from the pairwise results
    similarity_df = pd.DataFrame({
        "ticker_region1": ticker_regions[row_idx],
        "ticker_region2": ticker_regions[col_idx],
        "correlation": correlation_matrix[row_idx, col_idx],
        period_column: periods[row_idx]  # Assume periods are the same for corresponding rows
    })

    return similarity_df

def append_similarity_metrics_parallel(prices_df: pd.DataFrame, sequence_column: str, period_column: str = "calendar_month", chunk_size=5000) -> pd.DataFrame:
    ticker_regions = prices_df['ticker_region'].values
    periods = prices_df[period_column].values
    sequences = np.vstack(prices_df[sequence_column].values)

    n = len(sequences)
    results = []

    # Process in chunks to avoid memory overload
    for start in tqdm(range(0, n, chunk_size)):
        end = min(start + chunk_size, n)

        # Copy the chunk to GPU
        chunk = cp.asarray(sequences[start:end])
        chunk_ticker_regions = ticker_regions[start:end]
        chunk_periods = periods[start:end]

        # Compute pairwise correlations for the chunk against all rows
        correlations = cp.corrcoef(chunk, cp.asarray(sequences))

        # Convert to upper triangle indices (excluding diagonal)
        row_idx, col_idx = cp.triu_indices(correlations.shape[0], k=1)
        row_idx, col_idx = cp.asnumpy(row_idx), cp.asnumpy(col_idx)

        # Store results
        results.extend([
            {
                "ticker_region1": chunk_ticker_regions[i],
                "ticker_region2": ticker_regions[j],
                "correlation": float(correlations[i, j].get()),  # Retrieve GPU result
                period_column: chunk_periods[i]
            }
            for i, j in zip(row_idx, col_idx)
        ])

    # Combine results into a DataFrame
    similarity_df = pd.DataFrame(results)
    return similarity_df

def process_target_data(prices_df: pd.DataFrame, periods=3, sequence_length: int = 4) -> pd.DataFrame:

    if len(prices_df["ticker_region"].unique()) < 2:
        raise ValueError("At least two companies are required to calculate similarity metrics")

    target_df = data_utils.add_year_month_column(prices_df, "price_date", "calendar_month", drop_date_column=False)
    target_df = data_utils.add_missing_timestamps(target_df, "calendar_month", "M", entity_columns=["ticker_region"])
    target_df = append_stock_returns(prices_df = target_df, period_column="calendar_month", price_col="adj_price",
                                     periods=periods, entity_columns=["ticker_region"])

    for i in range(sequence_length):
        target_df[f"return_t{periods*i}_t{periods*(i+1)}"] = target_df.groupby("ticker_region")[f"{periods}m_return"].shift(-periods*(i+1))

    target_df["return_sequence"] = target_df[[f"return_t{periods*i}_t{periods*(i+1)}" for i in range(sequence_length)]].values.tolist()
    target_df["return_sequence"] = target_df["return_sequence"].apply(lambda x: x if not any(np.isnan(x)) else None)
    target_df["month_int"] = target_df["calendar_month"].apply(lambda date: date.month)
    target_df = target_df[target_df["month_int"].isin([i for i in range(1, 13) if i % periods == 0])]
    target_df = target_df.drop(columns=["month_int"])
    target_df = target_df.dropna(subset=["return_sequence", "ticker_region", "calendar_month"])
    
    target_df = target_df.groupby("calendar_month", group_keys=False)[target_df.columns].apply(lambda group: append_similarity_metrics(group, "return_sequence"))

    target_df = data_utils.standardize_columns(target_df, cols_to_transform=["correlation"])
    target_df = target_df[['calendar_month', 'ticker_region1','ticker_region2', 'correlation']]
    
    return target_df, None

def process_arima_data(target_df: pd.DataFrame,sequence_length: int,storage_path: str = None,
                       random_state: int = 42,arima_time_series: int = 3,) -> pd.DataFrame:
    # Create a unique identifier for grouping
    target_df["ticker_region1_ticker_region2"] = (
        target_df["ticker_region1"] + "_" + target_df["ticker_region2"]
    )

    # Get unique combinations and sample
    unique_combinations = target_df["ticker_region1_ticker_region2"].unique()
    np.random.seed(random_state)
    random_sample = np.random.choice(
        unique_combinations,
        size=min(arima_time_series, len(unique_combinations)),
        replace=False,
    )

    # Filter the data based on the sample
    arima_data = target_df[target_df["ticker_region1_ticker_region2"].isin(random_sample)]
    
    # Select relevant columns
    arima_data = arima_data[
        ["calendar_month", "ticker_region1", "ticker_region2", "correlation"]
    ].copy()
    
    # Standardize selected columns
    arima_data = data_utils.standardize_grouped_columns(
        arima_data,
        group_columns="ticker_region1_ticker_region2",
        cols_to_transform=["correlation"],
    )
    
    # Add row numbers within each group and filter by sequence length
    arima_data["row_number"] = arima_data.groupby("ticker_region1_ticker_region2").cumcount()
    arima_data = arima_data[arima_data["row_number"] % sequence_length == 0].drop(columns="row_number") #to avoid model looking into future returns. 

    # Drop the temporary column from the original DataFrame
    target_df = target_df.drop(columns=["ticker_region1_ticker_region2"])

    # Save to Parquet if a storage path is provided
    if storage_path: 
        if not arima_data.empty:
            arima_data.to_parquet(storage_path)
        else:
            raise ValueError("No data to save. The resulting DataFrame is empty.")

    return arima_data

def create_company_description_request_units(openai_model: OpenaiModel, company_description_units: list, user_message_key: str, system_message_key: str) -> list:

    request_units= []
    custom_ids = set()
    for company_description_unit in company_description_units:
        company_name = company_description_unit["company_name"]
        ticker_region = company_description_unit["ticker_region"]
        calendar_year = company_description_unit["calendar_year"]

        user_message = USER_MESSAGES[user_message_key].copy()
        user_message["content"] = custom_format(USER_MESSAGES["business_model_generator"]["content"], company_name, ticker_region, calendar_year)
        system_message = SYSTEM_MESSAGES[system_message_key]

        ticker_region_calendar_year = f"{ticker_region}_{calendar_year}"
        if ticker_region_calendar_year in custom_ids:
            continue

        request_unit = openai_model.create_request_unit(custom_id=ticker_region_calendar_year, system_message=system_message, user_message=user_message,
                                                        is_chat_completion_request=True)

        request_units.append(request_unit)
        custom_ids.add(ticker_region_calendar_year)

    return request_units

def create_company_descriptions_batch_job(openai_model: OpenaiModel, prices_raw: pd.DataFrame, company_names_df: pd.DataFrame,
                                          company_name_col:str, user_message_key: str = "business_model_generator", system_message_key: str = "business_model_generator",
                                        batch_job_ids_file_path = r"C:\repos\Deep-learning-trj\stock_correlation_project\openai_api_json_file\batch_id_list_company_descriptions.jsonl",
                                        overwrite_preexisting_batch_jobs=True,
                                        max_batch_size = 50000,
                                        request_units_file_name = r"C:\repos\Deep-learning-trj\stock_correlation_project\openai_api_json_file\batch_api_file.jsonl",
                                        pre_existing_company_descriptions_file = r"C:\repos\Deep-learning-trj\data\processed_data_storage\company_descriptions\company_descriptions.parquet"):

    company_names_df = company_names_df[[company_name_col, "ticker_region"]]
    company_names_df = company_names_df.rename(columns={company_name_col: "company_name"})
    prices_df = data_utils.add_year_column(prices_raw, "price_date", "calendar_year", drop_date_column=True)
    company_description_df = prices_df[["ticker_region", "calendar_year"]].drop_duplicates()
    company_description_df["calendar_year"] = company_description_df["calendar_year"] - 1 #Since end of last year's description is being used 
   
    company_description_df = company_description_df.merge(company_names_df, on="ticker_region", how="left")
    company_description_df = company_description_df.dropna(subset=["company_name"])

    if "company_name" not in company_description_df.columns or "ticker_region" not in company_description_df.columns or "calendar_year" not in company_description_df.columns:
        raise ValueError("The DataFrame must have columns 'company_name', 'ticker_region', and 'calendar_year'.")

    company_description_units = company_description_df.to_dict(orient="records") 

    company_description_request_units = create_company_description_request_units(openai_model=openai_model, company_description_units=company_description_units, user_message_key=user_message_key, system_message_key=system_message_key)
    if pre_existing_company_descriptions_file:
        company_description_request_units = openai_model.filter_request_units(company_description_request_units, pre_existing_company_descriptions_file, custom_id_col="ticker_region_calendar_year")

    openai_model.post_batch_job(request_units=company_description_request_units, batch_job_ids_file_path=batch_job_ids_file_path,
                            is_chat_completion_request=True, request_units_file_name=request_units_file_name,
                            overwrite_preexisting_batch_jobs=overwrite_preexisting_batch_jobs,
                            max_batch_size=max_batch_size)

def fetch_and_store_company_descriptions(openai_model: OpenaiModel, batch_ids_file_path: str,
                                        pre_existing_company_descriptions_file: str = r"C:\repos\Deep-learning-trj\data\processed_data_storage\company_descriptions\company_descriptions.parquet",
                                        storage_file_path: str = r"C:\repos\Deep-learning-trj\data\processed_data_storage\company_descriptions\company_descriptions.parquet",
                                        batch_result_file_path=r"C:\repos\Deep-learning-trj\stock_correlation_project\openai_api_json_file\batch_job_results.jsonl",
                                        overwrite=False
                                        ):

    batch_results = openai_model.get_batch_results(batch_ids_file_path=batch_ids_file_path, batch_result_file_path=batch_result_file_path)

    if not batch_results:
        print("No new data to store")
        return None

    new_data_df = openai_model.store_batch_results(batch_results=batch_results, is_chat_completion=True, custom_id_col="ticker_region_calendar_year",
                                                content_col="company_description", preexisting_file_path=pre_existing_company_descriptions_file, storage_file=storage_file_path, overwrite=overwrite)

    new_data_df[['ticker_region', 'calendar_year']] = new_data_df['ticker_region_calendar_year'].str.extract((r'(.*?-[^_]+)_(.*)'))
    new_data_df["calendar_year"] = new_data_df["calendar_year"].astype("period[Y]")
    new_data_df["release_month"] = new_data_df["calendar_year"].apply(lambda year: year.end_time.to_period("M"))
    new_data_df.to_parquet(storage_file_path)

def create_processed_data(example_companies = ["AAL-US", "MSFT-US", "AAPL-US", "HIMS-US"]):

    prices_raw = pd.read_csv(r"C:\repos\Deep-learning-trj\data\raw_data_storage\company_data\largest_5000_US_tickers_by_year_shares_only\prices\prices.csv")
    prices_raw = prices_raw[prices_raw["ticker_region"].isin(example_companies + ["SPY-US"])]
    prices = process_prices(prices_raw)
    print("Prices processed")
    target_data, _ = process_target_data(prices_raw, periods=3, sequence_length=4)
    print("Target data processed")
    fundamentals = pd.read_parquet(r"C:\repos\Deep-learning-trj\data\raw_data_storage\company_data\largest_5000_US_tickers_by_year_shares_only\fundamentals\fundamentals.parquet")
    fundamentals = fundamentals[fundamentals["ticker_region"].isin(example_companies)]
    fundamentals = process_fundamentals(fundamentals[fundamentals["ticker_region"].isin(example_companies)])
    print("Fundamentals processed")
    company_features_raw = pd.read_csv(r"C:\repos\Deep-learning-trj\data\raw_data_storage\company_data\largest_5000_US_tickers_by_year_shares_only\company_features\company_features.csv")
    company_features = process_company_features(company_features_raw[company_features_raw["ticker_region"].isin(example_companies)])
    print("Company features processed")
    macro = process_macro_data()
    print("Macro data processed")
    return macro, prices, fundamentals, company_features, target_data

def process_data_create_tensors_and_mapping_df(example_companies=["RIG-US", "MSFT-US", "AAPL-US", "VAL-US"], cut_off_date="1999-12"):

    # all_companies = pd.read_csv(r"C:\repos\Deep-learning-trj\data\raw_data_storage\company_data\largest_5000_US_tickers_by_year_shares_only\largest_5000_US_tickers_by_year_shares_only.csv")["ticker_region"].unique()

    if cut_off_date:
        cut_off_date = pd.Period(cut_off_date, freq="M")

    #macro
    processed_macro_df = process_macro_data()

    #prices
    prices = pd.read_csv(r"C:\repos\Deep-learning-trj\data\raw_data_storage\company_data\largest_5000_US_tickers_by_year_shares_only\prices\prices.csv")
    prices =  prices[prices["ticker_region"].isin(example_companies + ["SPY-US"])]

    processed_prices = process_prices(prices)
    processed_prices = processed_prices[processed_prices["release_month"] >= cut_off_date]
    
    #target data
    target_df, _ = process_target_data(prices, periods=3, sequence_length=4)
    target_df = target_df[target_df["calendar_month"] >= cut_off_date]

    #fundamentals
    fundamentals = pd.read_parquet(r"C:\repos\Deep-learning-trj\data\raw_data_storage\company_data\largest_5000_US_tickers_by_year_shares_only\fundamentals\fundamentals.parquet")
    fundamentals = fundamentals[fundamentals["ticker_region"].isin(example_companies)]

    processed_fundamentals = process_fundamentals(fundamentals[fundamentals["ticker_region"].isin(example_companies)])
    processed_fundamentals = processed_fundamentals[processed_fundamentals["release_month"] >= cut_off_date]

    #company features
    company_features_raw = pd.read_csv(r"C:\repos\Deep-learning-trj\data\raw_data_storage\company_data\largest_5000_US_tickers_by_year_shares_only\company_features\company_features.csv")
    processed_company_features = process_company_features(company_features_raw[company_features_raw["ticker_region"].isin(example_companies)])

    #company descriptions
    company_descriptions = pd.read_parquet(r"C:\repos\Deep-learning-trj\data\processed_data_storage\company_descriptions\company_descriptions.parquet")
    company_descriptions = company_descriptions[company_descriptions["ticker_region"].isin(example_companies)]

    mapping_df = target_df[["calendar_month", "ticker_region1", "ticker_region2"]]

    for col in ["ticker_region1", "ticker_region2"]:

        mapping_df = data_utils.create_tensor_mapping(mapping_df=mapping_df, tensor_df = processed_prices, tensor_df_reference_name="prices_" + col,
                                                      tensor_column = "stock_index_log_adj_price_ratios_sequence", left_on=["calendar_month", col],
                                                      right_on=["release_month", "ticker_region"], how="left")

        mapping_df = data_utils.create_tensor_mapping(mapping_df=mapping_df, tensor_df = processed_fundamentals, tensor_df_reference_name="fundamentals_" + col,
                                                      tensor_column = "fundamentals_sequence", left_on=["calendar_month", col],
                                                      right_on=["release_month", "ticker_region"], how="outer", apply_add_missing_timestamps=True,
                                                      period_col="release_month", freq="M", entity_columns=["ticker_region"], ffill_limit=3
                                                      )

        mapping_df = data_utils.create_tensor_mapping(mapping_df=mapping_df, tensor_df = processed_company_features, tensor_df_reference_name="company_features_" + col,
                                                      tensor_column = "company_features_array", left_on=[col],
                                                      right_on=["ticker_region"], how="left")
        
        mapping_df = data_utils.create_tensor_mapping(mapping_df=mapping_df, tensor_df = company_descriptions, tensor_df_reference_name="company_description_" + col,
                                                      tensor_column = "company_description", left_on=["calendar_month", col],
                                                      right_on=["release_month", "ticker_region"], how="outer", apply_add_missing_timestamps=True,
                                                      period_col="release_month", freq="M", entity_columns=["ticker_region"], ffill_limit=12)

    mapping_df = data_utils.create_tensor_mapping(mapping_df=mapping_df, tensor_df = target_df, tensor_df_reference_name="target_data",
                                                    tensor_column = "correlation", left_on=["calendar_month", "ticker_region1", "ticker_region2"],
                                                    right_on=["calendar_month", "ticker_region1", "ticker_region2"], how="left")

    mapping_df = data_utils.create_tensor_mapping(mapping_df=mapping_df, tensor_df = processed_macro_df, tensor_df_reference_name="macro",
                                                tensor_column = "macro_sequence", left_on=["calendar_month"],
                                                right_on=["release_month"], how="left")

    prices_tensor, price_key_padding_mask = data_utils.convert_iterable_to_tensor(processed_prices["stock_index_log_adj_price_ratios_sequence"], target_sequence_length=60,
                                                                       sequence_pad_value=0, pre_pad=True)

    data_utils.store_tensor(prices_tensor, r"C:\repos\Deep-learning-trj\stock_correlation_project\test_train_data\price_tensor")
    if price_key_padding_mask is not None:
        data_utils.store_tensor(price_key_padding_mask, r"C:\repos\Deep-learning-trj\stock_correlation_project\test_train_data\price_key_padding_mask")

    fundamentals_tensor, fundamentals_key_padding_mask = data_utils.convert_iterable_to_tensor(processed_fundamentals["fundamentals_sequence"], target_sequence_length=20,
                                                                                     sequence_pad_value=0, pre_pad=True)
    data_utils.store_tensor(fundamentals_tensor, r"C:\repos\Deep-learning-trj\stock_correlation_project\test_train_data\fundamentals_tensor")
    if fundamentals_key_padding_mask is not None:
        data_utils.store_tensor(fundamentals_key_padding_mask, r"C:\repos\Deep-learning-trj\stock_correlation_project\test_train_data\fundamentals_key_padding_mask")

    missing_features_fundamentals_mask, missing_features_key_padding_mask = data_utils.convert_iterable_to_tensor(processed_fundamentals["missing_feature_mask_sequence"], target_sequence_length=20,
                                                                                    sequence_pad_value=1, pre_pad=True)

    data_utils.store_tensor(missing_features_fundamentals_mask, r"C:\repos\Deep-learning-trj\stock_correlation_project\test_train_data\missing_features_fundamentals_mask")
    if missing_features_key_padding_mask is not None:
        data_utils.store_tensor(missing_features_key_padding_mask, r"C:\repos\Deep-learning-trj\stock_correlation_project\test_train_data\missing_features_fundamentals_key_padding_mask")

    company_features_tensor, _ = data_utils.convert_iterable_to_tensor(processed_company_features["company_features_array"], target_sequence_length=None)
    data_utils.store_tensor(company_features_tensor, r"C:\repos\Deep-learning-trj\stock_correlation_project\test_train_data\company_features_tensor")

    macro_tensor, macro_key_padding_mask = data_utils.convert_iterable_to_tensor(processed_macro_df["macro_sequence"], target_sequence_length=60,
                                                                                 sequence_pad_value=0, pre_pad=True)
    data_utils.store_tensor(macro_tensor, r"C:\repos\Deep-learning-trj\stock_correlation_project\test_train_data\macro_tensor")
    if macro_key_padding_mask is not None:
        data_utils.store_tensor(macro_key_padding_mask, r"C:\repos\Deep-learning-trj\stock_correlation_project\test_train_data\macro_key_padding_mask")

    company_description_model = SentenceTransformer(modules=[models.Transformer("bert-large-uncased")])
    huggingface_model = HuggingFaceModel(tokenizer=company_description_model.tokenizer)
    company_descriptions_input_tensor, company_descriptions_key_padding_mask = huggingface_model.batch_encode_texts(list(company_descriptions["company_description"]),
                                                                                                              padding="max_length", max_length=512)

    data_utils.store_tensor(company_descriptions_input_tensor, r"C:\repos\Deep-learning-trj\stock_correlation_project\test_train_data\company_descriptions_input_ids_tensor", dtype='<i8')
    data_utils.store_tensor(company_descriptions_key_padding_mask, r"C:\repos\Deep-learning-trj\stock_correlation_project\test_train_data\company_descriptions_key_padding_mask", dtype='<i8')

    correlation_tensor, _ = data_utils.convert_iterable_to_tensor(target_df["correlation"])
    data_utils.store_tensor(correlation_tensor, r"C:\repos\Deep-learning-trj\stock_correlation_project\test_train_data\correlation_tensor")
    mapping_df.to_parquet(r"C:\repos\Deep-learning-trj\stock_correlation_project\test_train_data\mapping_df.parquet")

    return mapping_df

def count_target_data():

    prices_raw = pd.read_csv(r"C:\repos\Deep-learning-trj\data\raw_data_storage\company_data\largest_5000_US_tickers_by_year_shares_only\prices\prices.csv")
    prices_raw = prices_raw[["ticker_region", "price_date"]]
    prices_raw = data_utils.add_year_quarter_column(prices_raw, "price_date", "calendar_quarter", drop_date_column=True)
    prices_raw = prices_raw.drop_duplicates()
    df = prices_raw.groupby("calendar_quarter")["ticker_region"].nunique().reset_index(drop=False)
    df["samples"] = df["ticker_region"].apply(lambda x: x*(x-1)/2)
    return df["samples"].sum()

def count_company_descriptions():
    prices_raw = pd.read_csv(r"C:\repos\Deep-learning-trj\data\raw_data_storage\company_data\largest_5000_US_tickers_by_year_shares_only\prices\prices.csv")
    prices_raw = prices_raw[["ticker_region", "price_date"]]
    prices_raw = data_utils.add_year_quarter_column(prices_raw, "price_date", "calendar_year", drop_date_column=True)
    prices_raw["calendar_year"] = prices_raw["calendar_year"].apply(lambda x: x.year)
    prices_raw = prices_raw.drop_duplicates()
    df = prices_raw.groupby("calendar_year")["ticker_region"].nunique().reset_index(drop=False)
    
    return df["ticker_region"].sum(), df["ticker_region"].mean()


if __name__ == "__main__":

    example_companies = ["MSFT-US", "AAPL-US", "VAL-US", "RIG-US"]

    # prices_raw = pd.read_csv(r"C:\repos\Deep-learning-trj\data\raw_data_storage\company_data\largest_5000_US_tickers_by_year_shares_only\prices\prices.csv")
    # prices_raw = prices_raw[prices_raw["ticker_region"].isin(example_companies + ["SPY-US"])]
    # prices = process_prices(prices_raw)

    # target_data, _ = process_target_data(prices_raw, periods=3, sequence_length=4)
    # data_utils.open_in_excel(target_data)

    # create company descriptions batch
    load_dotenv(r"C:\repos\Deep-learning-trj\.env")
    openai_model = OpenaiModel(api_key=os.getenv("OPENAI_API_KEY"))

    # company_names_df = pd.read_csv(r"C:\repos\Deep-learning-trj\data\raw_data_storage\company_data\largest_5000_US_tickers_by_year_shares_only\company_features\company_features.csv")
    # create_company_descriptions_batch_job(openai_model=openai_model, target_df=target_data, company_names_df=company_names_df, company_name_col="entity_proper_name",
    #                                       pre_existing_company_descriptions_file=r"C:\repos\Deep-learning-trj\data\processed_data_storage\company_descriptions\company_descriptions.parquet")
    # company_description_df = pd.read_parquet(r"C:\repos\Deep-learning-trj\data\processed_data_storage\company_descriptions\company_descriptions.parquet")
    # data_utils.open_in_excel(company_description_df)
    # print(company_description_df["ticker_region"].unique()) 

    #fetch and store batch results
    # response = openai_model.retrieve_batch("batch_677b9050b13081908c6fd1cfe7da1557")
    # print(response)
    # fetch_and_store_company_descriptions(openai_model=openai_model, batch_ids_file_path=r"C:\repos\Deep-learning-trj\stock_correlation_project\openai_api_json_file\batch_id_list_company_descriptions.jsonl",
    #                                     pre_existing_company_descriptions_file = r"C:\repos\Deep-learning-trj\data\processed_data_storage\company_descriptions\company_descriptions.parquet",
    #                                     storage_file_path = r"C:\repos\Deep-learning-trj\data\processed_data_storage\company_descriptions\company_descriptions.parquet",
    #                                     batch_result_file_path=r"C:\repos\Deep-learning-trj\stock_correlation_project\openai_api_json_file\batch_job_results.jsonl", overwrite=True)

    # print(company_description_df["ticker_region"].unique())
    huggingface_model = HuggingFaceModel()
    # company_description_df["token_count"] = company_description_df["company_description"].apply(lambda x: huggingface_model.count_tokens(x))
    # data_utils.open_in_excel(company_description_df)

    user_message = USER_MESSAGES["business_model_generator"].copy()
    user_message["content"] = custom_format(USER_MESSAGES["business_model_generator"]["content"], "Tesla", "TSLA-US", "2022")
    system_message = SYSTEM_MESSAGES["business_model_generator"]
    output = openai_model.generate_text(user_message=user_message, system_message=system_message)
    print(output)
    # openai_tokens = openai_model.count_tokens(output)
    # huggingface_tokens = huggingface_model.count_tokens(output)
    # print(openai_tokens, huggingface_tokens)

    # mapping_df = process_data_create_tensors_and_mapping_df(example_companies=example_companies, cut_off_date="1999-12")
    # data_utils.open_in_excel(pd.read_parquet(r"C:\repos\Deep-learning-trj\stock_correlation_project\test_train_data\mapping_df.parquet"))

    # prices_tensor = data_utils.load_tensor(r"C:\repos\Deep-learning-trj\stock_correlation_project\test_train_data\price_tensor")
    # prices_key_padding_mask = data_utils.load_tensor(r"C:\repos\Deep-learning-trj\stock_correlation_project\test_train_data\price_key_padding_mask")
    # fundamentals_tensor = data_utils.load_tensor(r"C:\repos\Deep-learning-trj\stock_correlation_project\test_train_data\fundamentals_tensor")
    # fundamentals_key_padding_mask = data_utils.load_tensor(r"C:\repos\Deep-learning-trj\stock_correlation_project\test_train_data\fundamentals_key_padding_mask")
    # fundamentals_missing_features_mask = data_utils.load_tensor(r"C:\repos\Deep-learning-trj\stock_correlation_project\test_train_data\missing_features_fundamentals_mask")
    # fundamentals_missing_features_key_padding_mask = data_utils.load_tensor(r"C:\repos\Deep-learning-trj\stock_correlation_project\test_train_data\missing_features_fundamentals_key_padding_mask")
    # company_features_tensor = data_utils.load_tensor(r"C:\repos\Deep-learning-trj\stock_correlation_project\test_train_data\company_features_tensor")
    # macro_tensor = data_utils.load_tensor(r"C:\repos\Deep-learning-trj\stock_correlation_project\test_train_data\macro_tensor")
    # macro_key_padding_mask = data_utils.load_tensor(r"C:\repos\Deep-learning-trj\stock_correlation_project\test_train_data\macro_key_padding_mask")
    # company_description_input_ids_tensor = data_utils.load_tensor(r"C:\repos\Deep-learning-trj\stock_correlation_project\test_train_data\company_descriptions_input_ids_tensor")
    # company_description_key_padding_mask = data_utils.load_tensor(r"C:\repos\Deep-learning-trj\stock_correlation_project\test_train_data\company_descriptions_key_padding_mask")
    # target_tensor = data_utils.load_tensor(r"C:\repos\Deep-learning-trj\stock_correlation_project\test_train_data\correlation_tensor")

    # mapping_df = pd.read_parquet(r"C:\repos\Deep-learning-trj\stock_correlation_project\test_train_data\mapping_df.parquet")

    # macro, prices, fundamentals, company_features, target_data = create_processed_data(example_companies=example_companies)
    # data_utils.open_in_excel(macro, "macro")
    # data_utils.open_in_excel(prices, "prices")
    # data_utils.open_in_excel(fundamentals, "fundamentals")
    # data_utils.open_in_excel(company_features, "company_features")
    # data_utils.open_in_excel(target_data, "target_data")
    # data_utils.open_in_excel(pd.read_parquet(r"C:\repos\Deep-learning-trj\data\processed_data_storage\company_descriptions\company_descriptions.parquet"))

    # rand_int = random.randint(0, len(mapping_df))
    # selected_target_data = mapping_df.iloc[rand_int]
    # print(selected_target_data)
    
    # for col in ["ticker_region1", "ticker_region2"]:

    #     price_index = int(selected_target_data["prices_" + col])
    #     price_tensor_sample = prices_tensor[price_index, :, :]
    #     price_key_padding_mask_sample = prices_key_padding_mask[price_index]
    #     price_df = pd.DataFrame(price_tensor_sample.numpy())
    #     price_mask = pd.DataFrame(price_key_padding_mask_sample.numpy())
    #     data_utils.open_in_excel(price_df, "prices_" + col)
    #     data_utils.open_in_excel(price_mask, "prices_mask_" + col)
    
    #     fundamentals_index = int(selected_target_data["fundamentals_" + col])
    #     fundamentals_tensor_sample = fundamentals_tensor[fundamentals_index, :, :]
    #     fundamentals_key_padding_mask_sample = fundamentals_key_padding_mask[fundamentals_index, :]
    #     fundamentals_missing_features_mask_sample = fundamentals_missing_features_mask[fundamentals_index, :, :]
    #     fundamentals_missing_features_key_padding_mask_sample = fundamentals_missing_features_key_padding_mask[fundamentals_index, :]
    #     fundamentals_df = pd.DataFrame(fundamentals_tensor_sample.numpy())
    #     fundamentals_mask = pd.DataFrame(fundamentals_key_padding_mask_sample.numpy())
    #     missing_features_mask = pd.DataFrame(fundamentals_missing_features_mask_sample.numpy())
    #     missing_features_key_padding_mask = pd.DataFrame(fundamentals_missing_features_key_padding_mask_sample.numpy())
    #     data_utils.open_in_excel(fundamentals_df, "fundamentals_" + col)
    #     data_utils.open_in_excel(fundamentals_mask, "fundamentals_mask_" + col)
    #     data_utils.open_in_excel(missing_features_mask, "missing_features_mask_" + col)
    #     data_utils.open_in_excel(missing_features_key_padding_mask, "missing_features_mask_key_padding_" + col)

    #     company_features_index = int(selected_target_data["company_features_" + col])
    #     company_features_tensor_sample = company_features_tensor[company_features_index, :, :]
    #     company_features_df = pd.DataFrame(company_features_tensor_sample.numpy())
    #     data_utils.open_in_excel(company_features_df, "company_features_" + col)

    #     company_description_index = int(selected_target_data["company_description_" + col])
    #     company_description_input_ids_sample = company_description_input_ids_tensor[company_description_index]
    #     company_description_key_padding_mask_sample = company_description_key_padding_mask[company_description_index]
    #     decoded_text = huggingface_model.decode_tokens(company_description_input_ids_sample, company_description_key_padding_mask_sample)
    #     print(decoded_text)

    # macro_index = int(selected_target_data["macro"])
    # macro_tensor = macro_tensor[macro_index, :, :]
    # macro_key_padding_mask = macro_key_padding_mask[macro_index, :]
    # macro_df = pd.DataFrame(macro_tensor.numpy())
    # macro_mask = pd.DataFrame(macro_key_padding_mask.numpy())
    # data_utils.open_in_excel(macro_df, "macro1")
    # data_utils.open_in_excel(macro_mask, "macro1_mask")

    # target_index = int(selected_target_data["target_data"])
    # target_tensor = target_tensor[target_index]
    # target_df = pd.DataFrame(target_tensor.numpy())
    # data_utils.open_in_excel(target_df, "target_data1")
    
    # macro_data = process_macro_data()
    # data_utils.open_in_excel(macro_data)
    # commodity_data = process_commodity_data()
    # data_utils.open_in_excel(commodity_data)

 

    
 