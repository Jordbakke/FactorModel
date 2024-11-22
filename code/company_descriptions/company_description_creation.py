import pandas as pd
import openai
import json
import os
import tiktoken
import sys
sys.path.append(r"C:\repos\Deep-learning-trj\data\code")
import data_utils
from openai import OpenAI
from dotenv import load_dotenv
from ast import literal_eval

load_dotenv(r"C:\repos\Deep-learning-trj\data\code\.env")
openai.api_key = os.getenv("OPENAI_API_KEY") or "your_api_key_here"
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) or "your_api_key_here"


SYSTEM_MESSAGES = {"business_model_generator": {"role": "system", "content": """As a stock portfolio manager, generate a business model description for a company as of the end of the specified year and quarter. Describe only what was known up to that time, with absolutely no information or insights from beyond that date. The business model descriptions will be used for portfolio risk analysis to identify similar and dissimilar companies. The business model description shall include the following aspects:

Revenue Model: Detailed description of the company's product or service and how the company generates revenue. Include information on the primary sources of revenue, such as product sales, services, and subscriptions, and how much visibility the company has into its future revenue streams.
Cost Structure: The top 3 cost components involved in the company's operations. Write hierarchically with the most significant cost component first.
Customer Segments: Detailed decription of the primary customer segments and customer subcategories. Write hierarchically with the most significant customer segment first.
Key Geographic Regions: The main geographic regions where the company operates or generates significant revenue.
Growth: How fast is the market expected to grow over the next years and is it expected to grow faster than the rest of the economy?
Risk Factors: The key risks associated with the company's business model
Market Development: Have value-driving factors changed over the past 6 months leading up to the given year and quarter?

Style: Write in present tense as if you went back in time to the end of the given year and quarter and wrote the business descriptio at that time. The business model description should be concise, objective without subjective opinions, and to the point."""

}
}
USER_MESSAGES = {
    
    "summary_writer":{"role": "user", "content":  "Company Description:\n'{}'\n\nCompany description summary:"},
    
    "rewriter":{"role": "user", "content":  "Company Description Summary:\n'{}'\n\nRewritten company description summary:"},

    "dated_company_describer": {"role": "user", "content": "Write a company description of {}, ticker {}, as of {} {}."},

    "business_model_generator": {"role": "user", "content": "Write a description of the business model of {} with ticker {} as of {} {}."}
}

def pop_and_return(dictionary, key):
    dictionary.pop(key, None)
    return dictionary

def count_tokens(text, model="gpt-4o-mini"):
    # Initialize the encoder for the specified model
    encoding = tiktoken.encoding_for_model(model)
    # Encode the text and get the list of tokens
    tokens = encoding.encode(text)
    # Return the number of tokens
    return len(tokens)

class CompanyDescriptionWriter:

    def __init__(self, openai_model="gpt-4o-mini", top_p = 1, temperature=0,
                rewrite_temperature = 1.2, rewrite_top_p =0.7,
                input_token_limit = 80000, max_output_tokens=2000,
                openai_embedding_model="text-embedding-3-small", embedding_dim=1536, max_tokens_embedding_model=8191):
        
        self.input_token_limit = input_token_limit
        self.openai_model = openai_model
        self.max_output_tokens = max_output_tokens
        self.tokenizer = tiktoken.encoding_for_model(self.openai_model)
        self.top_p = top_p
        self.temperature = temperature
        self.rewrite_temperature = rewrite_temperature
        self.rewrite_top_p = rewrite_top_p
        self.openai_embedding_model = openai_embedding_model
        self.embedding_dim = embedding_dim
        self.max_tokens_embedding_model = max_tokens_embedding_model
    
    def get_company_description_example(self, company_name, ticker_region, year, quarter):
        
        user_message = USER_MESSAGES["business_model_generator"].copy()
        user_message["content"] = user_message["content"].format(company_name, ticker_region, year, quarter)
        try:
            # Make the API call to the OpenAI chat completion endpoint
            response = openai.chat.completions.create(
                model=self.openai_model,
                messages=[SYSTEM_MESSAGES["business_model_generator"], user_message],
                max_tokens=self.max_output_tokens,
                temperature=self.temperature,
                top_p=self.top_p
            )
            
            # Extract and return the response text
            return response.choices[0].message.content
        except Exception as e:
            return f"An error occurred: {str(e)}"
        
    def post_batch_job(self, request_units: list, batch_job_ids_file_path: str, is_chat_completion_request: bool,
                        file_name: str = r"C:\repos\Deep-learning-trj\data\code\company_descriptions\openai_api_json_file\batch_api_file.jsonl",
                        overwrite_preexisting_batch_jobs: bool=False, max_batch_size: int = 50000):
        
        """
        Function that sends request to OpenAI to complete a batch job
        """
        # remove duplicate request_units
        request_units = data_utils.remove_duplicate_dicts(request_units)

        batch_job_ids = []
        # Check if there are existing batch jobs
        if os.path.exists(batch_job_ids_file_path) and not overwrite_preexisting_batch_jobs:
            with open(batch_job_ids_file_path, "r") as file:
                try:
                    data = json.load(file)
                    batch_job_ids = data.get("batch_job_ids", [])
                except (json.JSONDecodeError, KeyError):
                    print("Warning: Batch ID file is corrupted or incorrectly formatted. Starting fresh.")
                    batch_job_ids = []
    
        # Split the request_units into sublists of size max_batch_size
        batch_jobs = [request_units[i:i + max_batch_size] for i in range(0, len(request_units), max_batch_size)]
    
        # Create a file for each sublist and make a batch API call for each file
        for batch_job in batch_jobs:
            with open(file_name, "w") as file:
                for request_unit in batch_job:
                    file.write(json.dumps(request_unit) + "\n")
            
            with open(file_name, "rb") as file:
                batch_file = client.files.create(
                    file=file,
                    purpose="batch"
                )
    
            batch_response = client.batches.create(
                input_file_id=batch_file.id,
                endpoint='/v1/chat/completions' if is_chat_completion_request else '/v1/embeddings',
                completion_window="24h"
            )
            
            batch_job_ids.append(batch_response.id)
    
        # Save the batch IDs to a file for later retrieval
        with open(batch_job_ids_file_path, "w") as file:
            json.dump({"batch_job_ids": batch_job_ids}, file)

        print("Successfully created {} batch job(s)".format(len(batch_job_ids)))

    def get_batch_results(self, batch_ids_file_path:str ,
                               batch_result_file_path=r"C:\repos\Deep-learning-trj\data\batch_job_results.jsonl",
                               ):
        """
        Function that retrieves the results of a batch job
        """
        with open(batch_ids_file_path, "r") as file:
            json_line = file.readline()
            data = json.loads(json_line)

        batch_id_list = data.get("batch_job_ids", [])

        all_batch_results = []
        
        for batch_id in batch_id_list:
            batch_job = client.batches.retrieve(batch_id)

            if batch_job.status == "failed":
                print("Batch job with ID {} has failed. Skipping.".format(batch_id))
                print(batch_job.errors)
                print()
                continue

            if batch_job.status != "completed":
                print("Batch job with ID {} is not completed yet. Batch job status : {} Skipping.".format(batch_id, batch_job.status))
                continue
            else:
                result_file_id = batch_job.output_file_id
                result = client.files.content(result_file_id).content.decode("UTF-8")
                
                with open(batch_result_file_path, "w") as file:  # Open in text mode
                    file.write(result)
                
                results = []

                with open(batch_result_file_path, "r") as file:
                    for line in file:
                        json_object = json.loads(line.strip())
                        results.append(json_object)
                all_batch_results.extend(results)
        
        return all_batch_results
    
    def store_batch_results(self, batch_results: list, is_company_description: bool, storage_file:str):
        
        """
        Function that stores the results of a batch job in a file
        """
        if storage_file is None or not batch_results:
            return
        try:
            preexisting_df = pd.read_parquet(storage_file)
        except:
            preexisting_df = None
            
        new_data_list = []
        for i, request_result in enumerate(batch_results):
            custom_id = request_result["custom_id"]
            if is_company_description:
                content = request_result["response"]["body"]["choices"][0]["message"]["content"]
                content_type = "company_description"

            else:
                content = request_result["response"]["body"]["data"][0]["embedding"]
                content_type = "embedding"
            
            new_data_list.append({"ticker_year_quarter": custom_id, content_type: content})
        
        new_data_df = pd.DataFrame(new_data_list)
        new_data_df["date_created"] = pd.to_datetime('today').strftime('%Y-%m-%d')
        
        if preexisting_df is not None:
            # Perform the merge with suffixes for date columns
            new_data_df = pd.merge(preexisting_df, new_data_df, on=["ticker_year_quarter"], how='outer', suffixes=('_left', '_right'))

            # Prioritize the 'date_created_left' column but use 'date_created_right' where 'date_created_left' is NaN
            new_data_df["date_created"] = new_data_df["date_created_left"].combine_first(new_data_df["date_created_right"])
            # Drop the redundant columns
            new_data_df.drop(columns=["date_created_left", "date_created_right"], inplace=True)

        new_data_df[["ticker", "year_quarter"]] = new_data_df["ticker_year_quarter"].str.split("_", n=1, expand=True)
        new_data_df['year_quarter'] = new_data_df['year_quarter'].apply(lambda x: pd.Period(x, freq='Q'))
        new_data_df["year_month"] = new_data_df["year_quarter"].apply(lambda year_quarter: year_quarter.asfreq('M', how='end'))
        
        if storage_file is not None:
            new_data_df.to_parquet(storage_file)

    def filter_request_units(self, request_units, pre_existing_df_path):

        """
        Function that checks if data which the user wants to request from the API already exists in a pre-existing DataFrame.
        """
        if os.path.exists(pre_existing_df_path) and os.path.getsize(pre_existing_df_path) > 0:
                    # File exists and is not empty, proceed with reading the file
            pre_existing_df = pd.read_parquet(pre_existing_df_path)
        else:
            # File doesn't exist or is empty, handle the case accordingly (e.g., set to None or an empty DataFrame)
            pre_existing_df = None
            print(f"File '{pre_existing_df_path}' does not exist or is empty.")

        if pre_existing_df is not None:
            filtered_request_units = [request_unit for request_unit in request_units if request_unit["custom_id"] not in pre_existing_df["ticker_year_quarter"].values]
        else:
            # If None, assume no pre-existing descriptions, so all requests are valid
            filtered_request_units = request_units

        return filtered_request_units

    def create_company_description_request_units(self, company_description_units: list, user_message_key: str, system_message_key: str) -> list:
        
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
            request_body =  {"model": self.openai_model, "messages": messages,"max_tokens": self.max_output_tokens, "temperature": self.temperature, "top_p": self.top_p}
            request_unit = {"custom_id": ticker_year_quarter, "method": "POST", "url": "/v1/chat/completions", "body": request_body}
            request_units.append(request_unit)

        # request_units = remove_duplicate_dicts(request_units)

        return request_units
    
    def create_company_descriptions_batch_job(self, df: pd.DataFrame, user_message_key: str = "business_model_generator", system_message_key: str = "business_model_generator", batch_job_ids_file_path = r"C:\repos\Deep-learning-trj\data\code\company_descriptions\openai_api_json_file\batch_id_list_company_descriptions.jsonl",
                                                        overwrite_preexisting_batch_jobs=True,
                                                        max_batch_size = 50000,
                                                        file_name = r"C:\repos\Deep-learning-trj\data\code\company_descriptions\openai_api_json_file\batch_api_file.jsonl",
                                                        pre_existing_company_descriptions_file = r"C:\repos\Deep-learning-trj\data\code\company_descriptions\descriptions_embeddings\company_descriptions.parquet"):
        
        df = df.drop_duplicates(subset=["company_name", "ticker_region", "year_quarter"])

        if "company_name" not in df.columns or "ticker_region" not in df.columns or "year_quarter" not in df.columns:
            raise ValueError("The DataFrame must have columns 'company_name', 'ticker_region', and 'year_quarter'.")
        
        company_description_units = df.to_dict(orient="records")
        company_description_request_units = self.create_company_description_request_units(company_description_units=company_description_units, user_message_key=user_message_key, system_message_key=system_message_key)
        company_description_request_units = self.filter_request_units(company_description_request_units, pre_existing_company_descriptions_file)
        self.post_batch_job(request_units=company_description_request_units, batch_job_ids_file_path=batch_job_ids_file_path,
                             is_chat_completion_request=True, file_name=file_name,
                             overwrite_preexisting_batch_jobs=overwrite_preexisting_batch_jobs,
                             max_batch_size=max_batch_size)

    def create_embeddings_batch_job(self, company_descriptions: pd.DataFrame,
                                    batch_job_ids_file_path=r"C:\repos\Deep-learning-trj\data\code\company_descriptions\openai_api_json_file\batch_id_list_embeddings.jsonl",
                                    file_name=r"C:\repos\Deep-learning-trj\data\code\company_descriptions\openai_api_json_file\batch_api_file.jsonl",
                                    overwrite_preexisting_batch_jobs=True, max_batch_size=50000, pre_existing_embedding_file = r"C:\repos\Deep-learning-trj\data\code\company_descriptions\descriptions_embeddings\company_description_embeddings.csv"):
        
        
        if "ticker_year_quarter" not in company_descriptions.columns or "company_description" not in company_descriptions.columns:
            raise ValueError("The DataFrame must have columns 'ticker_year_quarter' and 'company_description'.")
        
        request_units = list(zip(company_descriptions["ticker_year_quarter"], company_descriptions["company_description"]))
        embedding_request_units = []
        for ticker_year_quarter, company_description in request_units:
        
            request_body = {"input": company_description, "model": self.openai_embedding_model, "dimensions": self.embedding_dim}
            request = {"custom_id": ticker_year_quarter, "method": "POST", "url": "/v1/embeddings", "body": request_body}
            embedding_request_units.append(request)
        
        embedding_request_units = self.filter_request_units(embedding_request_units, pre_existing_embedding_file)
        self.post_batch_job(request_units=embedding_request_units, batch_job_ids_file_path=batch_job_ids_file_path,
                             is_chat_completion_request=False, file_name=file_name,
                             overwrite_preexisting_batch_jobs=overwrite_preexisting_batch_jobs,
                             max_batch_size=max_batch_size) 

if __name__== "__main__":
    
    writer = CompanyDescriptionWriter()

    print(writer.get_company_description_example("Wells Fargo.", "WFC-US", "2019", "Q4"))
    print(writer.get_company_description_example("Goldman Sachs.", "GS-US", "2019", "Q4"))
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
    #     writer.store_batch_results(batch_results, is_company_description=True, storage_file=r"C:\repos\Deep-learning-trj\data\code\company_descriptions\descriptions_embeddings\company_descriptions.parquet")

    ########## Create embeddings batch job ##########
    # company_descriptions = pd.read_parquet(r"C:\repos\Deep-learning-trj\data\code\company_descriptions\descriptions_embeddings\company_descriptions.parquet")
    # writer.create_embeddings_batch_job(company_descriptions)

    # df = pd.read_parquet(r"C:\repos\Deep-learning-trj\data\code\company_descriptions\descriptions_embeddings\company_description_embeddings.parquet")

    # df['embedding'] = df['embedding'].apply(lambda x: ast.literal_eval(x))
    # df_dict = dict(zip(df['ticker_year_quarter'], df['embedding']))
    # desired_keys = ['MOWI-NO_1Q2024', 'MOWI-NO_2Q2024', 'NSIS.B-DK_1Q2024',
    #                 'NSIS.B-DK_2Q2024', 'SALM-NO_1Q2024', 'SALM-NO_2Q2024',
    #                 'SALM-NO_3Q2024', 'SCHB-NO_1Q2024', 'SCHB-NO_2Q2024', 'XXL-NO_1Q2024']

    # filtered_dict = {key: df_dict[key] for key in desired_keys if key in df_dict}

