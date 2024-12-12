import pandas as pd
import numpy as np
import sys
import os
# Add the parent directory to sys.path
sys.path.append(r"C:\repos\Deep-learning-trj\data\code")
import data_utils
import itertools
from llm_utils import OpenaiModel, HuggingFaceModel
from openai import OpenAI
from sentence_transformers import SentenceTransformer, models
from dotenv import load_dotenv

load_dotenv(r"C:\repos\Deep-learning-trj\data\code\.env")

SYSTEM_MESSAGES = {"business_model_generator": {"role": "system", "content": """You are an expert stock portfolio risk analyst. When provided with a company name and a specific year and quarter, your job is to generate a business model description for the provided company as of the end of the specified year and quarter.
Describe only what was known up to that time, with absolutely no information or insights from beyond that date. The business model descriptions will be used for portfolio risk analysis to identify similar and dissimilar companies.
Style: Write in present tense, as if you went back in time to the end of the given year and quarter and wrote the business descriptio at that time. The business model description should be extremely concise, objective, and to the point. Avoid any unneccessary words. Make the answers no longer than 300 words.
Ignore the company's insignificant products, divisions, services etc throughout the description."""

},
"short_term_view": {"role": "system", "content": """You are a stock portfolio manager focusing on short-term stock price development. When provided with a company name and a specific year and quarter, describe whether
                    we have seen a bullish or bearish development in demand and supply expectations over the past 6 months leading up to the given year and quarter. 
                    Do not describe the development in financial statements. 
                    
                    Describe only what was known up to that time, with absolutely no information or insights from beyond that date.

                    Style: Write in present tense, as if you went back in time to the end of the given year and quarter and wrote it at that time. The description should be concise, strictly objective and to the point."""

}
}
USER_MESSAGES = {
    
    "intra_sector_differentiator": {"role": "user", "content": """Write a description of how the business model of {} with ticker {} differs from other companies and competitors in the same sector/industry as of {} {}."""},
    "business_model_generator": {"role": "user", "content": """Write a description of the business model of {} with ticker {} as of {} {}. The business model description shall include the following aspects: 

Value Proposition: What problem does the company solve and how does it solve it? 
Product Offering: Description of the company's product(s) or service(s)                     
Revenue Model: Description of how the company generates revenue and its primary revenue sources, such as product sales or recurring revenue.
Revenue Visibility: Description of the company's revenue visibility - how much of future revenue is already contracted?
Product Lifetime Cycle: What stage of the lifetime cycle is the company's product(s) in: introduction, growth, maturity, or decline?
Sector and Supply Chain Position: Which sector does the company operate in, and what role does the company have in the supply chain?
Cost Structure: What are the company's 3 most important cost drivers? Be granular - down to the raw material level when suitable. Write hierarchically, starting with the most significant cost driver.
Customer Segments: Granular decription of the primary customer segments and target market.
Key Geographic Regions: The main geographic regions where the company operates or generates significant revenue. Just answer with the two most significant regions.
Risk Factors: Description of the key risk factors, both company-specific and industry-specific.
Industry Value-Drivers: Describe the top 3 industry-wide value-driving factors.
Short Investment Case arguments: As of the end of the given year and quarter, what are bearish investors' arguments for shorting the stock?
Buy Investment Case Arguments: As of the end of the given year and quarter, what are bullish investor's arguments for buying the stock?

Ignore the company's insignificant products, divisions, services etc throughout the description.
                                 
Style: Write in present tense, as if you went back in time to the end of the given year and quarter and wrote it at that time. The description should be concise, strictly objective and to the point.
       Describe only what was known up to that time, with absolutely no information or insights from beyond that date."""
}
}

def process_fundamentals_data(fundamentals_raw_df: pd.DataFrame,
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

    #convert numeric cols to float64
    non_numeric_cols = ["ticker_region", "calendar_quarter", "release_month"]
    numeric_cols = [col for col in fundamentals_df.columns if col not in non_numeric_cols]
    fundamentals_df[numeric_cols] = fundamentals_df[numeric_cols].apply(pd.to_numeric, errors='coerce').astype(float) #convert columns which should be numeric

    fundamentals_df = fundamentals_df.groupby("ticker_region", group_keys=False)[fundamentals_df.columns].apply(
        lambda group: data_utils.add_missing_periods(
            df=group, period_column="calendar_quarter", freq="Q", entity_columns=["ticker_region"]
        ), include_groups=True
    ).reset_index(drop=True)

    #Get missing features mask before imputing values
    numeric_cols = fundamentals_df.select_dtypes(include='number').columns.tolist()
    missing_feature_masks = fundamentals_df[numeric_cols].isna().astype(float).values.tolist()
    fundamentals_df["missing_feature_mask"] = missing_feature_masks
    # impute missing values with 0
    fundamentals_df[numeric_cols] = fundamentals_df[numeric_cols].replace(np.nan, 0)

    fundamentals_df = data_utils.standardize_time_windows(df=fundamentals_df, period_col="calendar_quarter", period_freq="Q",
                                                          num_periods=12, cols_to_standardize=numeric_cols, include_current_period=False,
                                                          min_periods=4) #12 quarters to get 3 years. min_periods=4 to get at least 1 year.
    
    fundamentals_df[numeric_cols] = fundamentals_df[numeric_cols].replace(np.nan, 0) # NaN occurs when the standard deviation is 0, so the standardized values are undefined.
    fundamentals_df["fundamentals_list"] = fundamentals_df.select_dtypes(include=['number']).values.tolist()
    fundamentals_df = fundamentals_df.groupby("ticker_region", group_keys=False)[fundamentals_df.columns].apply(lambda group: data_utils.collect_previous_values( #group_keys=False to  have the group col as column and not index
                                        df=group, value_column="fundamentals_list",
                                        date_column="calendar_quarter",
                                        new_column_name="fundamentals_sequence", include_current_row=True,
                                        drop_empty_sequences=True,
                                        sequence_length=sequence_length)
                                        ).reset_index(drop=True) #include_groups=True, the passed in group will also have the group column
    

    fundamentals_df = fundamentals_df.groupby("ticker_region", group_keys=False)[fundamentals_df.columns].apply(lambda group: data_utils.collect_previous_values( #group_keys=False to  have the group col as column and not index
                                        df=group, value_column="missing_feature_mask",
                                        date_column="calendar_quarter",
                                        new_column_name="missing_feature_mask_sequence", include_current_row=True,
                                        drop_empty_sequences=True,
                                        sequence_length=sequence_length)
                                        ).reset_index(drop=True)
    
    return fundamentals_df[["calendar_quarter", "release_month", "ticker_region", "fundamentals_sequence", "missing_feature_mask_sequence"]]

def process_prices(prices_raw_df: pd.DataFrame, sequence_length = 60, index_ticker="SPY-US") -> pd.DataFrame:

    """
    Function to process price data which will be used for company embeddings
    """
    
    prices_df = data_utils.add_year_month_column(prices_raw_df, "price_date", "calendar_month", drop_date_column=True)
    prices_df = prices_df[prices_df["calendar_month"] >= pd.Period("1999-12-12", freq="M")]
    prices_df = prices_df.groupby("ticker_region", group_keys=False)[prices_df.columns].apply(
        lambda group: data_utils.add_missing_periods(
            df=group, period_column="calendar_month", freq="M", entity_columns=["ticker_region"]
        )
    ).reset_index(drop=True)

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
    # Merge stock and index prices on 'price_date'
    merged_df = prices_df.merge(index_prices, on=["calendar_month"], how="left")
    # Create the new column with stock and index prices

    merged_df["stock_index_log_adj_price_ratios"] = merged_df.apply(
        lambda row: [row["log_adj_price_ratio"], row["index_log_adj_price_ratio"] if (not pd.isna(row["index_log_adj_price_ratio"]) and not pd.isna(row["log_adj_price_ratio"])) else np.nan],
        axis=1
    )

    merged_df = merged_df.groupby("ticker_region", group_keys=False)[merged_df.columns].apply(lambda group: group.sort_values(by="calendar_month").iloc[1:, :]).reset_index(drop=True) #remove the first row for each group as this will always have nan values when log transforming.

    merged_df = merged_df.drop_duplicates(subset=["calendar_month", "ticker_region"])
    merged_df = merged_df.groupby("ticker_region")[merged_df.columns].apply(
    lambda group: data_utils.collect_previous_values(
        df=group,
        value_column="stock_index_log_adj_price_ratios",
        date_column="calendar_month",
        new_column_name=f"stock_index_log_adj_price_ratios_sequence",
        include_current_row=True,
        sequence_length=sequence_length,
        drop_empty_sequences=True,
    )).reset_index(drop=True)


    return merged_df[["calendar_month", "release_month", "ticker_region", "stock_index_log_adj_price_ratios_sequence"]]

def process_company_features(company_features_df: pd.DataFrame, columns_to_drop: tuple = ("iso_country", "iso_country_cor"),
                             categorical_columns: tuple = ("iso_country_cor_georev", "industry_code")) -> pd.DataFrame:
    
    if columns_to_drop is not None:
        company_features_df = company_features_df.drop(columns=columns_to_drop, errors="ignore")
    
    company_features_df = data_utils.one_hot_encode_categorical_columns(company_features_df, categorical_columns)
    company_features_df = data_utils.standardize_columns(company_features_df, cols_to_standardize=["year_founded"])
    company_features_df["company_features_array"] = list(company_features_df.select_dtypes(include=['number']).to_numpy())
    company_features_df = company_features_df.drop_duplicates(subset=["ticker_region"])
    return company_features_df

def process_macro_data(macro_directory=r"C:\repos\Deep-learning-trj\data\raw_data_storage\macro_data",
                       monthly_release_date_offset: int = 1, quarterly_release_date_offset=2, sequence_length = 60, ffill_limit=1,
                       release_month_column_name: str = "release_month"):
    
    def process_helper(file_path: str, date_col: str, cols_to_standardize: list = None, pd_read_func=pd.read_csv,
                        rename_cols: dict = None, agg_func=None, is_quarterly_data=False, is_daily_data=False,
                        growth_to_ratios_cols: list = None, agg_func_groupby_col: str = None,
                        ratio_log_transform_cols: list = None, ffill=False,
                        ffill_limit=1, destination_path: str = None) -> pd.DataFrame:
        
        # Read file and prepare columns
        df = pd_read_func(file_path)
        df.columns = [col.strip().lower() for col in df.columns]

        rename_cols = {key.lower(): value for key, value in rename_cols.items()} if rename_cols else None
        if rename_cols:
            df = df.rename(columns=rename_cols)

        df[date_col] = pd.to_datetime(df[date_col])

        if is_quarterly_data:
            df = data_utils.add_year_quarter_column(df, date_col, "calendar_quarter", drop_date_column=True)
            df[release_month_column_name] = df["calendar_quarter"].apply(lambda x: pd.Period(x.end_time, freq="M") + quarterly_release_date_offset)
            df = data_utils.expand_time_series(df=df, period_col="calendar_quarter", from_period_freq="Q", to_period_freq="M", new_period_col_name="calendar_month")
            df = df.drop(columns="calendar_quarter")
            
        elif is_daily_data:
            df = data_utils.add_year_month_column(df, date_col, "calendar_month", drop_date_column=False)
            df = (df.sort_values(date_col).groupby("calendar_month").last().reset_index()).drop(date_col,axis=1)
            df[release_month_column_name] = df["calendar_month"]

        else:
            df = data_utils.add_year_month_column(df, date_col, "calendar_month", drop_date_column=True)
            df[release_month_column_name] = df["calendar_month"].apply(lambda x: pd.Period(x.end_time, freq="M") + monthly_release_date_offset)

        df = data_utils.add_missing_periods(df, "calendar_month", "M")

        if ffill and ffill_limit:
            df = df.sort_values(by="calendar_month").ffill(limit=ffill_limit)

        if cols_to_standardize:
            df[cols_to_standardize] = df[cols_to_standardize].apply(pd.to_numeric, errors='coerce').astype(float)
            df = data_utils.standardize_columns(df, cols_to_standardize)

        if agg_func and agg_func_groupby_col:
            release_month_col = df[['release_month', agg_func_groupby_col]].drop_duplicates()
            df_aggregated = df.groupby(agg_func_groupby_col, group_keys=False).agg(agg_func).reset_index()
            df = df_aggregated.merge(release_month_col, on=agg_func_groupby_col, how='left')

        if growth_to_ratios_cols:
            df[growth_to_ratios_cols] = df[growth_to_ratios_cols].apply(pd.to_numeric, errors='coerce').astype(float)
            df[growth_to_ratios_cols] = df[growth_to_ratios_cols].apply(lambda x: 1 + x)
            
        if ratio_log_transform_cols:
            df = data_utils.log_ratio_transform(df, ratio_log_transform_cols, inplace=True)
            ratio_log_transform_cols = [col + "_log_ratio" for col in ratio_log_transform_cols]
            df = df[["calendar_month", release_month_column_name] + ratio_log_transform_cols]

        # Save to destination if specified
        if destination_path:
            df.to_parquet(destination_path)

        return df

    files_to_process = {
    "10y_2y_treasury_spread": {
        "file_path": os.path.join(macro_directory, "10y_2y_treasury_spread.csv"),
        "date_col": "date",
        "cols_to_standardize": ["10y_2y_spread"],
        "pd_read_func": pd.read_csv,
        "rename_cols": {"T10Y2Y": "10y_2y_spread"},
        "is_daily_data": True,
        "ffill": True,
        "destination_path": None #r"C:\repos\Deep-learning-trj\data\processed_data_storage\macro_data\10y_2y_treasury_spread.parquet"
    },
    "10y_treasury_yield": {
        "file_path": os.path.join(macro_directory, "10y_treasury_yield.csv"),
        "date_col": "date",
        "cols_to_standardize": ["10y_yield"],
        "pd_read_func": pd.read_csv,
        "rename_cols": {"DGS10": "10y_yield"},
        "is_daily_data": True,
        "ffill": True,
        "destination_path": None #r"C:\repos\Deep-learning-trj\data\processed_data_storage\macro_data\10y_treasury_yield.parquet"
    },
    "US_core_inflation": {
        "file_path": os.path.join(macro_directory, "US_core_inflation.csv"),
        "date_col": "date",
        "cols_to_standardize": ["core_inflation"],
        "pd_read_func": pd.read_csv,
        "rename_cols": {"CORESTICKM159SFRBATL": "core_inflation"},
        "ffill": True,
        "destination_path": None #r"C:\repos\Deep-learning-trj\data\processed_data_storage\macro_data\US_core_inflation.parquet"
    },
    "US_corporate_debt_growth": {
        "file_path": os.path.join(macro_directory, "US_corporate_debt_growth.csv"),
        "date_col": "date",
        "pd_read_func": pd.read_csv,
        "rename_cols": {"BOGZ1FG104104005Q": "corporate_debt_growth"},
        "is_quarterly_data": True,
        "ffill": True,
        "growth_to_ratios_cols": ["corporate_debt_growth"],
        "ratio_log_transform_cols": ["corporate_debt_growth"],
        "destination_path": None #r"C:\repos\Deep-learning-trj\data\processed_data_storage\macro_data\US_corporate_debt_growth.parquet"
    },
    "EBIT_US_companies": {
        "file_path": os.path.join(macro_directory, "EBIT_US_companies.csv"),
        "date_col": "date",
        "pd_read_func": pd.read_csv,
        "rename_cols": {"BOGZ1FA106110115Q": "ebit"},
        "is_quarterly_data": True,
        "ffill": True,
        "ratio_log_transform_cols": ["ebit"],
        "destination_path": None #r"C:\repos\Deep-learning-trj\data\processed_data_storage\macro_data\EBIT_US_companies.parquet"
    },
    "high_yield_spread": {
        "file_path": os.path.join(macro_directory, "high_yield_spread.csv"),
        "date_col": "date",
        "cols_to_standardize": ["high_yield_spread"],
        "pd_read_func": pd.read_csv,
        "rename_cols": {"BAMLH0A0HYM2": "high_yield_spread"},
        "is_daily_data": True,
        "ffill": True,
        "destination_path": None #r"C:\repos\Deep-learning-trj\data\processed_data_storage\macro_data\high_yield_spread.parquet"
    },
    "US_unemployment_rate": {
        "file_path": os.path.join(macro_directory, "US_unemployment_rate.csv"),
        "date_col": "date",
        "cols_to_standardize": ["unemployment_rate"],
        "pd_read_func": pd.read_csv,
        "rename_cols": {"UNRATE": "unemployment_rate"},
        "ffill": True,
        "destination_path": None #r"C:\repos\Deep-learning-trj\data\processed_data_storage\macro_data\US_unemployment_rate.parquet"
    },
    "US_real_gdp": {
        "file_path": os.path.join(macro_directory, "US_real_gdp.csv"),
        "date_col": "date",
        "pd_read_func": pd.read_csv,
        "rename_cols": {"GDPC1": "real_gdp"},
        "is_quarterly_data": True,
        "ffill": True,
        "ratio_log_transform_cols": ["real_gdp"],  # No aggregation, pct_change computed later
        "destination_path": None #r"C:\repos\Deep-learning-trj\data\processed_data_storage\macro_data\US_real_gdp.parquet"
    },
    "US_consumer_sentiment": {
        "file_path": os.path.join(macro_directory, "US_consumer_sentiment.csv"),
        "date_col": "date",
        "pd_read_func": pd.read_csv,
        "rename_cols": {"UMCSENT": "consumer_sentiment"},
        "ffill": True,
        "cols_to_standardize": ["consumer_sentiment"],  # No aggregation, pct_change computed later
        "destination_path": None #r"C:\repos\Deep-learning-trj\data\processed_data_storage\macro_data\US_real_gdp.parquet"
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
    sp_data = data_utils.add_year_month_column(sp_data, "Date", "calendar_month", drop_date_column=True)
    sp_data = sp_data[sp_data["calendar_month"] >= pd.Period("1990-01", freq="M")]
    sp_data = data_utils.add_missing_periods(sp_data, "calendar_month", "M")
    sp_data = sp_data.rename(columns={"Price": "price", "Earnings": "earnings"})
    sp_data = sp_data.sort_values(by="calendar_month")
    sp_data["earnings"] = sp_data.apply(lambda row: row["earnings"] if row["calendar_month"].month in [3, 6, 9, 12] else np.nan, axis=1)
    sp_data["earnings"] = sp_data["earnings"].ffill(limit=3)
    
    sp_data["pe_3m_lagged_earnings"] = sp_data["price"] / sp_data["earnings"].shift(3) #rolling sum of 12 months
    sp_data = sp_data[["calendar_month", "price", "pe_3m_lagged_earnings", "earnings"]]
    sp_data = sp_data.dropna()
    sp_data = sp_data.ffill(limit=ffill_limit)
    sp_data[release_month_column_name] = sp_data["calendar_month"]
    sp_data = data_utils.standardize_columns(sp_data, cols_to_standardize=["pe_3m_lagged_earnings"])
    sp_data = data_utils.log_ratio_transform(sp_data, cols_to_transform=["price"])
    sp_data = sp_data[["calendar_month", "release_month", "pe_3m_lagged_earnings", "price_log_ratio"]]
    all_macro_dfs.append(sp_data)

    # Real interest rates
    treasury_yield = pd.read_csv(os.path.join(macro_directory, "10y_treasury_yield.csv")).rename(columns={"DGS10": "10y_yield", "DATE": "date"})
    core_inflation = pd.read_csv(os.path.join(macro_directory, "US_core_inflation.csv")).rename(columns={"CORESTICKM159SFRBATL": "core_inflation", "DATE": "date"})
    real_interest_rates = treasury_yield.merge(core_inflation, on="date", how="inner")
    real_interest_rates = data_utils.add_year_month_column(real_interest_rates, "date", "calendar_month", drop_date_column=True)
    real_interest_rates["10y_yield"] = pd.to_numeric(real_interest_rates["10y_yield"], errors="coerce").astype(float).ffill(limit=1)
    real_interest_rates["core_inflation"] = pd.to_numeric(core_inflation["core_inflation"], errors="coerce").astype(float).ffill(limit=1)
    real_interest_rates["real_interest_rate"] = real_interest_rates["10y_yield"] - real_interest_rates["core_inflation"]
    real_interest_rates[release_month_column_name] = real_interest_rates["calendar_month"].apply(lambda x: pd.Period(x.end_time, freq="M") + monthly_release_date_offset)
    real_interest_rates = real_interest_rates[["calendar_month", "real_interest_rate", release_month_column_name]]
    real_interest_rates = data_utils.standardize_columns(real_interest_rates, cols_to_standardize=["real_interest_rate"])
    all_macro_dfs.append(real_interest_rates)
    
    # COMMODITIES 

    #Oil price - inflation adjusted
    oil_prices = pd.read_excel(os.path.join(macro_directory, "oil_prices.xlsx"))
    oil_prices = data_utils.add_year_month_column(oil_prices, "date", "calendar_month", drop_date_column=True)
    us_cpi = pd.read_csv(os.path.join(macro_directory, "US_CPI.csv")).rename(columns={"CPIAUCSL": "us_cpi", "DATE": "date"})
    us_cpi = us_cpi.sort_values(by="date")
    us_cpi["us_cpi"] = us_cpi["us_cpi"] / us_cpi["us_cpi"].iloc[0]
    us_cpi = data_utils.add_year_month_column(us_cpi, "date", "calendar_month", drop_date_column=True)
    oil_prices = oil_prices.merge(us_cpi, on="calendar_month", how="left")
    oil_prices["brent_inflation_adjusted"] = oil_prices["brent"] / oil_prices["us_cpi"]
    oil_prices["wti_inflation_adjusted"] = oil_prices["wti"] / oil_prices["us_cpi"]
    oil_prices[release_month_column_name] = oil_prices["calendar_month"]
    oil_prices = oil_prices[["calendar_month", "brent_inflation_adjusted", "wti_inflation_adjusted" ,release_month_column_name]]
    all_macro_dfs.append(oil_prices)
    macro_df = data_utils.release_date_aware_merge_and_sequence_creation(dfs=all_macro_dfs, calendar_period_col="calendar_month", release_period_col=release_month_column_name,
                                                                         sequence_column_name="macro_sequence", sequence_length=sequence_length, ffill_limit=3)
    return macro_df

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

def append_similarity_metrics(prices_df: pd.DataFrame, sequence_column: str, period_column:str = "calendar_month") -> pd.DataFrame:
    # Create an empty list to store the results
    results = []

    # Calculate cosine similarity and dot product for each pair
    for (_, row1), (_, row2) in itertools.combinations(prices_df.iterrows(), 2):
        # Cosine similarity

        dot_product = np.dot(row1[sequence_column], row2[sequence_column])
        correlation = np.corrcoef(row1[sequence_column], row2[sequence_column])[0, 1]
        results.append({
            "ticker_region1": row1['ticker_region'],
            "ticker_region2": row2['ticker_region'],
            "dot_product": dot_product,
            "correlation": correlation,
            period_column: row1[period_column]
        })

    # Convert results to DataFrame
    similarity_df = pd.DataFrame(results)

    return similarity_df

def process_target_data(prices_df: pd.DataFrame, periods=3, sequence_length: int = 4, step_size=3) -> pd.DataFrame:
    
    def exclude_rows(group, step_size=3):
        
        group["return_sequence"] = [
            return_sequence if i % step_size == 0 and not any(pd.isna(val) for val in return_sequence) else np.nan
            for i, return_sequence in enumerate(group["return_sequence"])
        ]

        return group

    if len(prices_df["ticker_region"].unique()) < 2:
        raise ValueError("At least two companies are required to calculate similarity metrics")
    
    target_df = data_utils.add_year_month_column(prices_df, "price_date", "calendar_month", drop_date_column=False)
    target_df = data_utils.add_missing_periods(target_df, "calendar_month", "M", entity_columns=["ticker_region"])
    target_df = append_stock_returns(prices_df = target_df, period_column="calendar_month", price_col="adj_price",
                                     periods=periods, entity_columns=["ticker_region"])
    
    for i in range(sequence_length):
        target_df[f"return_t{periods*i}_t{periods*(i+1)}"] = target_df.groupby("ticker_region")[f"{periods}m_return"].shift(-periods*(i+1))
    
    target_df["return_sequence"] = target_df[[f"return_t{periods*i}_t{periods*(i+1)}" for i in range(sequence_length)]].values.tolist()
    target_df = target_df.groupby("ticker_region")[target_df.columns].apply(exclude_rows, step_size=step_size)
    target_df = target_df.dropna(subset=["return_sequence", "ticker_region", "calendar_month"])
    target_df = target_df.groupby("calendar_month", group_keys=False)[target_df.columns].apply(lambda group: append_similarity_metrics(group, "return_sequence"))
    target_df = data_utils.standardize_columns(target_df, cols_to_standardize=["dot_product" , "correlation"])
    target_df = target_df[['calendar_month', 'ticker_region1','ticker_region2', 'dot_product', 'correlation']]
    
    return target_df

def create_company_description_request_units(openai_model: OpenaiModel, company_description_units: list, user_message_key: str, system_message_key: str) -> list:
    
    request_units= []

    for company_description_unit in company_description_units:
        company_name = company_description_unit["company_name"]
        ticker_region = company_description_unit["ticker_region"]
        calendar_quarter = company_description_unit["calendar_quarter"]
        year = str(calendar_quarter.year)
        quarter = "Q" + str(calendar_quarter.quarter)

        user_message = USER_MESSAGES[user_message_key].copy()
        user_message["content"] = user_message["content"].format(company_name, ticker_region, quarter, year)
        system_message = SYSTEM_MESSAGES[system_message_key]

        ticker_region_calendar_quarter = f"{ticker_region}_{calendar_quarter}"
        request_unit = openai_model.create_request_unit(custom_id=ticker_region_calendar_quarter, system_message=system_message, user_message=user_message,
                                                        is_chat_completion_request=True)
        
        request_units.append(request_unit)

    return request_units

def create_company_descriptions_batch_job(openai_model: OpenaiModel, target_df: pd.DataFrame, company_names_df: pd.DataFrame,
                                          company_name_col:str, user_message_key: str = "business_model_generator", system_message_key: str = "business_model_generator",
                                        batch_job_ids_file_path = r"C:\repos\Deep-learning-trj\data\code\stock_correlations_project\openai_api_json_file\batch_id_list_company_descriptions.jsonl",
                                        overwrite_preexisting_batch_jobs=True,
                                        max_batch_size = 50000,
                                        file_name = r"C:\repos\Deep-learning-trj\data\code\stock_correlations_project\openai_api_json_file\batch_api_file.jsonl",
                                        pre_existing_company_descriptions_file = r"C:\repos\Deep-learning-trj\data\processed_data_storage\company_descriptions\company_descriptions.parquet"):

    company_names_df = company_names_df[[company_name_col, "ticker_region"]]
    company_names_df = company_names_df.rename(columns={company_name_col: "company_name"})
    
    company_description_df = (
    pd.concat([
        target_df[["calendar_month", "ticker_region1"]].rename(columns={"ticker_region1": "ticker_region"}),
        target_df[["calendar_month", "ticker_region2"]].rename(columns={"ticker_region2": "ticker_region"})
    ])
    .drop_duplicates()
    )

    company_description_df["calendar_quarter"] = company_description_df["calendar_month"].dt.to_timestamp().dt.to_period("Q")
    company_description_df = company_description_df.drop(columns="calendar_month")
    company_description_df = company_description_df.merge(company_names_df, on="ticker_region", how="left")

    if "company_name" not in company_description_df.columns or "ticker_region" not in company_description_df.columns or "calendar_quarter" not in company_description_df.columns:
        
        raise ValueError("The DataFrame must have columns 'company_name', 'ticker_region', and 'calendar_quarter'.")
    
    company_description_units = company_description_df.to_dict(orient="records")
    company_description_request_units = create_company_description_request_units(openai_model=openai_model, company_description_units=company_description_units, user_message_key=user_message_key, system_message_key=system_message_key)
    company_description_request_units = openai_model.filter_request_units(company_description_request_units, pre_existing_company_descriptions_file, custom_id_col="ticker_region_calendar_quarter")
    openai_model.post_batch_job(request_units=company_description_request_units, batch_job_ids_file_path=batch_job_ids_file_path,
                            is_chat_completion_request=True, file_name=file_name,
                            overwrite_preexisting_batch_jobs=overwrite_preexisting_batch_jobs,
                            max_batch_size=max_batch_size)

def fetch_and_store_company_descriptions(openai_model: OpenaiModel, batch_ids_file_path: str, company_descriptions_file_path: str,
                                        batch_result_file_path=r"C:\repos\Deep-learning-trj\data\code\stock_correlations_project\openai_api_json_file\batch_job_results.jsonl", 
                                        ):

    batch_result = openai_model.get_batch_results(batch_ids_file_path=batch_ids_file_path, batch_result_file_path=batch_result_file_path)
    new_data_df = openai_model.store_batch_results(batch_results=batch_result, is_chat_completion=True, custom_id_col="ticker_region_calendar_quarter", 
                                                content_col="company_description")
    new_data_df[['ticker_region', 'calendar_quarter']] = new_data_df['ticker_region_calendar_quarter'].str.extract(r'(.*?-[^_]+)_(.*)')
    new_data_df = new_data_df.drop(columns="ticker_region_calendar_quarter")
    new_data_df["calendar_quarter"] = new_data_df["calendar_quarter"].astype("period[Q]")
    new_data_df["release_month"] = new_data_df["calendar_quarter"] + 1
    new_data_df["release_month"] = new_data_df["release_month"].asfreq('M', how='start')
    new_data_df.to_parquet(company_descriptions_file_path)

def create_processed_data(example_companies = ["AAL-US", "MSFT-US", "AAPL-US", "SPY-US"]):

    macro_df = process_macro_data()  
    macro_df.to_parquet(r"C:\repos\Deep-learning-trj\data\processed_data_storage\macro_data\macro_df.parquet")
    print("Successfully processed macro data")
    prices_raw = pd.read_csv(r"C:\repos\Deep-learning-trj\data\raw_data_storage\stock_correlations_project\top_US_USD_companies_shares_only\prices\prices.csv")
    prices_raw = prices_raw[prices_raw["ticker_region"].isin(example_companies)]
    prices = process_prices(prices_raw)
    prices.to_parquet(r"C:\repos\Deep-learning-trj\data\processed_data_storage\stock_correlations_project\example_companies\prices.parquet")
    print("Successfully processed prices data")
    target_data = process_target_data(prices_raw, periods=3, sequence_length=4, step_size=3)
    target_data.to_parquet(r"C:\repos\Deep-learning-trj\data\processed_data_storage\stock_correlations_project\example_companies\target_data.parquet")
    print("Successfully processed target data")
    fundamentals = pd.read_parquet(r"C:\repos\Deep-learning-trj\data\raw_data_storage\stock_correlations_project\top_US_USD_companies_shares_only\fundamentals\fundamentals.parquet")
    fundamentals = fundamentals[fundamentals["ticker_region"].isin(example_companies)]
    fundamentals = process_fundamentals_data(fundamentals[fundamentals["ticker_region"].isin(example_companies)])
    fundamentals.to_parquet(r"C:\repos\Deep-learning-trj\data\processed_data_storage\stock_correlations_project\example_companies\fundamentals.parquet")
    print("Successfully processed fundamentals data")
    company_features_raw = pd.read_csv(r"C:\repos\Deep-learning-trj\data\raw_data_storage\stock_correlations_project\top_US_USD_companies_shares_only\company_features\company_features.csv")
    company_features = process_company_features(company_features_raw[company_features_raw["ticker_region"].isin(example_companies)])
    company_features.to_parquet(r"C:\repos\Deep-learning-trj\data\processed_data_storage\stock_correlations_project\example_companies\company_features.parquet")
    print("Successfully processed company features data")

def process_data_create_tensors_and_mapping_df(example_companies=["AAL-US", "MSFT-US", "AAPL-US", "SPY-US"], cut_off_date="1999-12"):

    if cut_off_date:
        cut_off_date = pd.Period(cut_off_date, freq="M")

    #macro
    processed_macro_df = process_macro_data()  

    #prices
    prices = pd.read_csv(r"C:\repos\Deep-learning-trj\data\raw_data_storage\stock_correlations_project\top_US_USD_companies_shares_only\prices\prices.csv")
    prices =  prices[prices["ticker_region"].isin(example_companies)]
    processed_prices = process_prices(prices)
    processed_prices = processed_prices[processed_prices["release_month"] >= cut_off_date]

    #target data
    target_df = process_target_data(prices, periods=3, sequence_length=4, step_size=3)
    target_df = target_df[target_df["calendar_month"] >= cut_off_date]

    #fundamentals
    fundamentals = pd.read_parquet(r"C:\repos\Deep-learning-trj\data\raw_data_storage\stock_correlations_project\top_US_USD_companies_shares_only\fundamentals\fundamentals.parquet")
    fundamentals = fundamentals[fundamentals["ticker_region"].isin(example_companies)]
    processed_fundamentals = process_fundamentals_data(fundamentals[fundamentals["ticker_region"].isin(example_companies)])
    processed_fundamentals = processed_fundamentals[processed_fundamentals["release_month"] >= cut_off_date]

    #company features
    company_features_raw = pd.read_csv(r"C:\repos\Deep-learning-trj\data\raw_data_storage\stock_correlations_project\top_US_USD_companies_shares_only\company_features\company_features.csv")
    processed_company_features = process_company_features(company_features_raw[company_features_raw["ticker_region"].isin(example_companies)])

    #company descriptions
    company_descriptions = pd.read_parquet(r"C:\repos\Deep-learning-trj\data\processed_data_storage\company_descriptions\company_descriptions.parquet")
    company_descriptions = company_descriptions[company_descriptions["ticker_region"].isin(example_companies)]

    mapping_df = target_df

    for col in ["ticker_region1", "ticker_region2"]:

        mapping_df = data_utils.create_tensor_mapping(mapping_df=mapping_df, tensor_df = processed_prices, tensor_df_reference_name="prices_" + col,
                                                      tensor_column = "stock_index_log_adj_price_ratios_sequence", left_on=["calendar_month", col],
                                                      right_on=["release_month", "ticker_region"], how="left")
        
        mapping_df = data_utils.create_tensor_mapping(mapping_df=mapping_df, tensor_df = processed_fundamentals, tensor_df_reference_name="fundamentals_" + col,
                                                      tensor_column = "fundamentals_sequence", left_on=["calendar_month", col],
                                                      right_on=["release_month", "ticker_region"], how="left", apply_add_missing_periods=True,
                                                      period_col="release_month", freq="M", entity_columns=["ticker_region"], ffill_limit=3)
        
        mapping_df = data_utils.create_tensor_mapping(mapping_df=mapping_df, tensor_df = processed_company_features, tensor_df_reference_name="company_features_" + col,
                                                      tensor_column = "company_features_array", left_on=[col],
                                                      right_on=["ticker_region"], how="left")
        
        mapping_df = data_utils.create_tensor_mapping(mapping_df=mapping_df, tensor_df = company_descriptions, tensor_df_reference_name="company_descriptions_" + col,
                                                      tensor_column = "company_description", left_on=["calendar_month", col],
                                                      right_on=["release_month", "ticker_region"], how="left", apply_add_missing_periods=True,
                                                      period_col="release_month", freq="M", entity_columns=["ticker_region"], ffill_limit=3)

    mapping_df = data_utils.create_tensor_mapping(mapping_df=mapping_df, tensor_df = processed_macro_df, tensor_df_reference_name="macro_",
                                                tensor_column = "macro_sequence", left_on=["calendar_month"],
                                                right_on=["release_month"], how="left")

    prices_tensor, price_key_padding_mask = data_utils.convert_iterable_to_tensor(processed_prices["stock_index_log_adj_price_ratios_sequence"], target_sequence_length=60,
                                                                       sequence_pad_value=0, pre_pad=True)

    data_utils.store_tensor(prices_tensor, r"C:\repos\Deep-learning-trj\data\processed_data_storage\stock_correlations_project\example_companies\prices_tensor")
    if price_key_padding_mask is not None:
        data_utils.store_tensor(price_key_padding_mask, r"C:\repos\Deep-learning-trj\data\processed_data_storage\stock_correlations_project\example_companies\prices_tensor_mask")
    
    fundamentals_tensor, fundamentals_key_padding_mask = data_utils.convert_iterable_to_tensor(processed_fundamentals["fundamentals_sequence"], target_sequence_length=20,
                                                                                     sequence_pad_value=0, pre_pad=True)
    data_utils.store_tensor(fundamentals_tensor, r"C:\repos\Deep-learning-trj\data\processed_data_storage\stock_correlations_project\example_companies\fundamentals_tensor")
    if fundamentals_key_padding_mask is not None:
        data_utils.store_tensor(fundamentals_key_padding_mask, r"C:\repos\Deep-learning-trj\data\processed_data_storage\stock_correlations_project\example_companies\fundamentals_tensor_mask")
    
    missing_features_fundamentals_mask, missing_features_padding_mask = data_utils.convert_iterable_to_tensor(processed_fundamentals["missing_feature_mask_sequence"], target_sequence_length=20,
                                                                                    sequence_pad_value=1, pre_pad=True)
    
    data_utils.store_tensor(missing_features_fundamentals_mask, r"C:\repos\Deep-learning-trj\data\processed_data_storage\stock_correlations_project\example_companies\missing_features_fundamentals_mask")
    if missing_features_padding_mask is not None:
        data_utils.store_tensor(missing_features_padding_mask, r"C:\repos\Deep-learning-trj\data\processed_data_storage\stock_correlations_project\example_companies\missing_features_fundamentals_key_padding_mask")

    company_features_tensor, company_features_key_padding_mask = data_utils.convert_iterable_to_tensor(processed_company_features["company_features_array"], target_sequence_length=None)
    data_utils.store_tensor(company_features_tensor, r"C:\repos\Deep-learning-trj\data\processed_data_storage\stock_correlations_project\example_companies\company_features_tensor")
    if company_features_key_padding_mask is not None:
        data_utils.store_tensor(company_features_key_padding_mask, r"C:\repos\Deep-learning-trj\data\processed_data_storage\stock_correlations_project\example_companies\company_features_tensor_mask")
    
    macro_tensor, macro_key_padding_mask = data_utils.convert_iterable_to_tensor(processed_macro_df["macro_sequence"], target_sequence_length=60,
                                                                                 sequence_pad_value=0, pre_pad=True)
    data_utils.store_tensor(macro_tensor, r"C:\repos\Deep-learning-trj\data\processed_data_storage\macro_data\macro_tensor")
    if macro_key_padding_mask is not None:
        data_utils.store_tensor(macro_key_padding_mask, r"C:\repos\Deep-learning-trj\data\processed_data_storage\macro_data\macro_tensor_mask")

    company_description_model = SentenceTransformer(modules=[models.Transformer("bert-large-uncased")])
    tokenizer = company_description_model.tokenizer
    company_descriptions_input_tensor, company_descriptions_key_padding_mask = HuggingFaceModel.batch_encode_texts(company_descriptions["company_description"],
                                                                                                             tokenizer, padding="max_length", max_length=512)
    
    data_utils.store_tensor(company_descriptions_input_tensor, r"C:\repos\Deep-learning-trj\data\processed_data_storage\stock_correlations_project\example_companies\company_descriptions_input_tensor")
    data_utils.store_tensor(company_descriptions_key_padding_mask, r"C:\repos\Deep-learning-trj\data\processed_data_storage\stock_correlations_project\example_companies\company_descriptions_key_padding_mask")

    mapping_df.to_parquet(r"C:\repos\Deep-learning-trj\data\processed_data_storage\stock_correlations_project\example_companies\target_data_tensor_mapping.parquet")

    return mapping_df
    
if __name__ == "__main__":

    # example_companies = ["AAL-US", "MSFT-US", "AAPL-US", "SPY-US"]
    # print("Successfully created processed data")

    # mapping_df = process_data_create_tensors_and_mapping_df()
    # print("Successfully created tensor mapping")
    
    #create company descriptions batch
    # load_dotenv(r"C:\repos\Deep-learning-trj\data\code\.env")
    # openai_model = OpenaiModel(api_key=os.getenv("OPENAI_API_KEY"), max_output_tokens=450)
    # company_names_df = pd.read_csv(r"C:\repos\Deep-learning-trj\data\raw_data_storage\stock_correlations_project\top_US_USD_companies_shares_only\company_features\company_features.csv")
    # create_company_descriptions_batch_job(openai_model = openai_model, target_df=mapping_df, company_names_df=company_names_df, company_name_col="entity_proper_name")
    
    #fetch and store batch results
    # fetch_and_store_company_descriptions(openai_model=openai_model, batch_ids_file_path = r"C:\repos\Deep-learning-trj\data\code\stock_correlations_project\openai_api_json_file\batch_id_list_company_descriptions.jsonl",
    # company_descriptions_file_path=r"C:\repos\Deep-learning-trj\data\processed_data_storage\company_descriptions\company_descriptions.parquet")

    # prices_tensor = data_utils.load_tensor(r"C:\repos\Deep-learning-trj\data\processed_data_storage\stock_correlations_project\example_companies\prices_tensor")
    # prices_tensor_mask = data_utils.load_tensor(r"C:\repos\Deep-learning-trj\data\processed_data_storage\stock_correlations_project\example_companies\prices_tensor_mask")
    # fundamentals_tensor = data_utils.load_tensor(r"C:\repos\Deep-learning-trj\data\processed_data_storage\stock_correlations_project\example_companies\fundamentals_tensor")
    # fundamentals_tensor_mask = data_utils.load_tensor(r"C:\repos\Deep-learning-trj\data\processed_data_storage\stock_correlations_project\example_companies\fundamentals_tensor_mask")
    # company_features_tensor = data_utils.load_tensor(r"C:\repos\Deep-learning-trj\data\processed_data_storage\stock_correlations_project\example_companies\company_features_tensor")
    # macro_tensor = data_utils.load_tensor(r"C:\repos\Deep-learning-trj\data\processed_data_storage\macro_data\macro_tensor")
    # macro_tensor_mask = data_utils.load_tensor(r"C:\repos\Deep-learning-trj\data\processed_data_storage\macro_data\macro_tensor_mask")

    # for i in range(5):

    #     rand_int = random.randint(0, len(mapping_df))
    #     selected_target_data = mapping_df.iloc[rand_int]
    #     rand_tensor = fundamentals_tensor[int(selected_target_data["fundamentals_ticker_region1"]), :, :]
    #     mask = fundamentals_tensor_mask[int(selected_target_data["fundamentals_ticker_region1"]), :]
    #     print("random row: ", selected_target_data)
    #     print(rand_tensor)
    #     print(mask)
    #     print("-----------------")
    
  

    # openai_model = OpenaiModel(api_key=os.getenv("OPENAI_API_KEY"), max_output_tokens=450)
    # HuggingFaceModel = HuggingFaceModel()
    # user_message= USER_MESSAGES["business_model_generator"].copy()
    # user_message["content"]=user_message["content"].format("Microsoft", "MSFT-US", "Q2", "2022")

    # result = openai_model.generate_text(system_message=SYSTEM_MESSAGES["business_model_generator"], user_message=user_message)
    # hugging_face_token_count = HuggingFaceModel.count_tokens(result)
    # openai_token_count = openai_model.count_tokens(result)
    # print(f"Openai token count: {openai_token_count}, Hugging Face token count: {hugging_face_token_count}")
    # print(result)

    # user_message= USER_MESSAGES["business_model_generator"].copy()
    # user_message["content"]=user_message["content"].format("Odfjell Drilling", "ODL-NO", "Q2", "2022")

    # result = openai_model.generate_text(system_message=SYSTEM_MESSAGES["business_model_generator"], user_message=user_message)
    # hugging_face_token_count = HuggingFaceModel.count_tokens(result)
    # openai_token_count = openai_model.count_tokens(result)
    # print(f"Openai token count: {openai_token_count}, Hugging Face token count: {hugging_face_token_count}")
    # print(result)