import pandas as pd
import openai
import json
import os
import tiktoken
import sys
sys.path.append(r"C:\repos\Deep-learning-trj\data\code")
import data_utils
from llm_utils import OpenaiModel
from openai import OpenAI
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModel
from ast import literal_eval
import torch

load_dotenv(r"C:\repos\Deep-learning-trj\data\code\.env")
openai.api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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
    
    "short_term_view": {"role": "user", "content": """Write a description of whether and how value-driving factors for the company {} with ticker {} as of {} {}."""},
    "business_model_generator": {"role": "user", "content": """Write a description of the business model of {} with ticker {} as of {} {}. The business model description shall include the following aspects: 

Value proposition: What problem does the company solve and how does it solve it?                            
Revenue Model: Description of the company's product(s) or service(s) and primary revenue sources, such as product sales, services or subscriptions.
Product Lifetime Cycle: What stage of the lifetime cycle is the company's product(s) in: introduction, growth, maturity, or decline?
Sector and supply chain position: Which sector does the company operate in, and what role does it play within the supply chain?
Contract duration: What is the typical customer contract duration?
Cost Structure: What are the company's 3 most important cost drivers? Be granular - down to the raw material level when suitable. Write hierarchically, starting with the most significant cost driver.
Customer Segments: Granular decription of the primary customer segments and target market.
Key Geographic Regions: The main geographic regions where the company operates or generates significant revenue. Just answer with the two most significant regions.
Risk factors: Description of the key risk factors, both company-specific and industry-specific.
Industry value-drivers: Describe the top 3 industry-wide value-driving factors.
Short investment case arguments: As of the end of the given year and quarter, what are bearish investors' arguments for shorting the stock?
Buy investment case arguments: As of the end of the given year and quarter, what are bullish investor's arguments for buying the stock?

Ignore the company's insignificant products, divisions, services etc throughout the description.
                                 
Style: Write in present tense, as if you went back in time to the end of the given year and quarter and wrote it at that time. The description should be concise, strictly objective and to the point.
       Describe only what was known up to that time, with absolutely no information or insights from beyond that date."""
}
}


def create_company_description_request_units(openai_model: OpenaiModel, company_description_units: list, user_message_key: str, system_message_key: str) -> list:
    
    request_units= []

    for company_description_unit in company_description_units:
        company_name = company_description_unit["company_name"]
        ticker_region = company_description_unit["ticker_region"]
        year_quarter = company_description_unit["year_quarter"]
        year = str(year_quarter.year)
        quarter = "Q" + str(year_quarter.quarter)

        user_message = USER_MESSAGES[user_message_key].copy()
        user_message["content"] = user_message["content"].format(company_name, ticker_region, quarter, year)
        
        messages = [
            SYSTEM_MESSAGES[system_message_key],
            user_message
        ]

        ticker_year_quarter = f"{ticker_region}_{year_quarter}"
        request_body =  {"model": openai_model.openai_model, "messages": messages,"max_tokens": openai_model.max_output_tokens,
                            "temperature": openai_model.temperature, "top_p": openai_model.top_p, "frequency_penalty": openai_model.frequency_penalty,
                            "presence_penalty": openai_model.presence_penalty}
        request_unit = {"custom_id": ticker_year_quarter, "method": "POST", "url": "/v1/chat/completions", "body": request_body}
        request_units.append(request_unit)

    return request_units

def create_company_descriptions_batch_job(openai_model: OpenaiModel, df: pd.DataFrame, user_message_key: str = "business_model_generator", system_message_key: str = "business_model_generator", batch_job_ids_file_path = r"C:\repos\Deep-learning-trj\data\code\company_descriptions\openai_api_json_file\batch_id_list_company_descriptions.jsonl",
                                                    overwrite_preexisting_batch_jobs=True,
                                                    max_batch_size = 50000,
                                                    file_name = r"C:\repos\Deep-learning-trj\data\code\company_descriptions\openai_api_json_file\batch_api_file.jsonl",
                                                    pre_existing_company_descriptions_file = r"C:\repos\Deep-learning-trj\data\code\company_descriptions\descriptions_embeddings\company_descriptions.parquet"):
    
    df = df.drop_duplicates(subset=["company_name", "ticker_region", "year_quarter"])

    if "company_name" not in df.columns or "ticker_region" not in df.columns or "year_quarter" not in df.columns:
        raise ValueError("The DataFrame must have columns 'company_name', 'ticker_region', and 'year_quarter'.")
    
    company_description_units = df.to_dict(orient="records")
    company_description_request_units = openai_model.create_company_description_request_units(company_description_units=company_description_units, user_message_key=user_message_key, system_message_key=system_message_key)
    company_description_request_units = openai_model.filter_request_units(company_description_request_units, pre_existing_company_descriptions_file)
    openai_model.post_batch_job(request_units=company_description_request_units, batch_job_ids_file_path=batch_job_ids_file_path,
                            is_chat_completion_request=True, file_name=file_name,
                            overwrite_preexisting_batch_jobs=overwrite_preexisting_batch_jobs,
                            max_batch_size=max_batch_size)

def create_embeddings_batch_job(openai_model: OpenaiModel, company_descriptions: pd.DataFrame,
                                batch_job_ids_file_path=r"C:\repos\Deep-learning-trj\data\code\company_descriptions\openai_api_json_file\batch_id_list_embeddings.jsonl",
                                file_name=r"C:\repos\Deep-learning-trj\data\code\company_descriptions\openai_api_json_file\batch_api_file.jsonl",
                                overwrite_preexisting_batch_jobs=True, max_batch_size=50000, pre_existing_embedding_file = r"C:\repos\Deep-learning-trj\data\code\company_descriptions\descriptions_embeddings\company_description_embeddings.csv"):
    
    
    if "ticker_year_quarter" not in company_descriptions.columns or "company_description" not in company_descriptions.columns:
        raise ValueError("The DataFrame must have columns 'ticker_year_quarter' and 'company_description'.")
    
    request_units = list(zip(company_descriptions["ticker_year_quarter"], company_descriptions["company_description"]))
    embedding_request_units = []
    for ticker_year_quarter, company_description in request_units:
    
        request_body = {"input": company_description, "model": openai_model.openai_embedding_model, "dimensions": openai_model.embedding_dim}
        request = {"custom_id": ticker_year_quarter, "method": "POST", "url": "/v1/embeddings", "body": request_body}
        embedding_request_units.append(request)
    
    embedding_request_units = openai_model.filter_request_units(embedding_request_units, pre_existing_embedding_file)
    openai_model.post_batch_job(request_units=embedding_request_units, batch_job_ids_file_path=batch_job_ids_file_path,
                            is_chat_completion_request=False, file_name=file_name,
                            overwrite_preexisting_batch_jobs=overwrite_preexisting_batch_jobs,
                            max_batch_size=max_batch_size) 
    

if __name__== "__main__":
    
    writer = OpenaiModel(model="gpt-4o-mini", top_p = 1, temperature=0,
                frequency_penalty = 0, presence_penalty=0,
                input_token_limit = 80000, max_output_tokens=1000,
                openai_embedding_model="text-embedding-3-small", embedding_dim=1536, max_tokens_embedding_model=8191)

    user_message = USER_MESSAGES["business_model_generator"].copy()
    user_message["content"] = user_message["content"].format("Microsoft", "MSFT-US", "Q1", "2020")
    system_message = SYSTEM_MESSAGES["business_model_generator"]
    output = writer.generate_text(user_message, system_message)
    print(output)
    # ticker_year_quarter_df = pd.read_parquet(r"C:\repos\Deep-learning-trj\data\processed_data_storage\stock_correlations_project\example_companies\fundamentals.parquet")
    # ticker_year_quarter_df = ticker_year_quarter_df[["ticker_region", "year_quarter"]]

    # company_name_df = pd.read_parquet(r"C:\repos\Deep-learning-trj\data\raw_data_storage\stock_correlations_project\russell_3000_companies\company_features.parquet")
    # company_name_df = company_name_df[["entity_proper_name", "ticker_region"]]
    # company_name_df = company_name_df.rename(columns={"entity_proper_name": "company_name"})

    # company_description_units = pd.merge(ticker_year_quarter_df, company_name_df, on="ticker_region", how="inner")
    # company_description_units = company_description_units[company_description_units["year_quarter"] >= pd.Period("2020Q1")]
    
    # writer.create_company_descriptions_batch_job(company_description_units, user_message_key="business_model_generator", system_message_key="business_model_generator")

    # batch_results = writer.get_batch_results(r"C:\repos\Deep-learning-trj\data\code\company_descriptions\openai_api_json_file\batch_id_list_company_descriptions.jsonl")

    # if batch_results:
    #     writer.store_batch_results(batch_results, is_company_description=True, storage_file=r"C:\repos\Deep-learning-trj\data\processed_data_storage\descriptions_embeddings\company_descriptions.parquet")

    ########## Create embeddings batch job ##########
    # company_descriptions = pd.read_parquet(r"C:\repos\Deep-learning-trj\data\processed_data_storage\descriptions_embeddings\company_descriptions.parquet")
    # writer.create_embeddings_batch_job(company_descriptions)
    
    # df = pd.read_parquet(r"C:\repos\Deep-learning-trj\data\processed_data_storage\descriptions_embeddings\company_description_embeddings.parquet")

