import pandas as pd
import openai
import tiktoken
import json
import os
import sys
import sec
import ast
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from pyspark.sql import SparkSession
from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv
from ast import literal_eval

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY") or "your_api_key_here"
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) or "your_api_key_here"

SYSTEM_MESSAGES = {"summary_writer": {"role": "system", "content":
                       """
You are an expert financial analyst specialized in summarizing company descriptions. Investors will use your summaries to decide whether a given company fits their desired investment exposure. When provided with a company description, your task is to generate an objective, concise and informative summary from an investor's perspective. Your summary should highlight the following key aspects:
    - Market: The primary market(s) the company operates in and how they are evolving.
    - Geography: Where is the company based and which are the important geographic regions the company serves.
    - Sector: The industry sector to which the company belongs.
    - Products: The main products or services offered by the company.
    - Revenue Model: How the company generates its revenue.
    - Competition: How the market competition is.
    - Customer Base: The primary customers or customer segments the company serves.
    
Ensure that the summary is brief yet comprehensive, capturing the essential information an investor would need.

The following is an example of a good company description summary - use the same format when writing your summaries:
                       
'**Tesla, Inc.**

**Market:** Tesla operates primarily in the electric vehicle (EV) and renewable energy markets, both of which are rapidly evolving with increasing global demand for sustainable solutions.

**Geography:** Headquartered in Palo Alto, California, Tesla serves a global market with significant operations in the U.S., China, and Europe. The company has manufacturing facilities in the U.S., China, Germany, and is expanding to Mexico.

**Sector:** Tesla belongs to the automotive and energy sectors, focusing on electric vehicles and renewable energy solutions.

**Products:** Tesla's product portfolio includes high-performance electric vehicles (Model 3, Y, S, X, Cybertruck, and Tesla Semi), energy storage solutions (Powerwall and Megapack), and solar energy products (Solar Roof and retrofit solar systems).

**Revenue Model:** Tesla generates revenue through direct sales and leasing of electric vehicles, sales of automotive regulatory credits, energy storage products, and solar 
energy systems. Additional revenue streams include vehicle services, Supercharging, insurance, and software upgrades.

**Competition:** The automotive market is highly competitive, with increasing entries from both established and new manufacturers in the EV space. Tesla also faces competition in the energy storage and solar energy markets from traditional utilities and other renewable energy companies.

**Customer Base:** Tesla's primary customers include individual consumers, commercial entities, industrial clients, and utilities seeking sustainable energy solutions. The company also targets residential customers for its solar and energy storage products.

Tesla's vertically integrated business model, focus on innovation, and commitment to sustainability position it uniquely in the market, appealing to investors interested in 
the growing sectors of electric vehicles and renewable energy.'"""},

"rewriter": {"role": "system", "content":"""You are a textual data augmentation expert. Your job is to rewrite a given company description summary to augment the data. The augmented data will be used to further train a neural network. Rewrite the company description summary to introduce
variation and diversity but the meaning must be maintained. Remember to maintain clarity and coherence in your rewritten summary."""},

"company_description_writer": {"role": "system", "content": """You are an expert at writing company descriptions from a business perspective. When provided with a ticker, write a company description from a business perspective. Be objective and don't try to hype the company. The following is an example of a good company description for TSLA-US - use the same format when writing your descriptions: 
             
'**Tesla, Inc.**

**Market:** Tesla operates primarily in the electric vehicle (EV) and renewable energy markets, both of which are rapidly evolving with increasing global demand for sustainable solutions.

**Geography:** Headquartered in Palo Alto, California, Tesla serves a global market with significant operations in the U.S., China, and Europe. The company has manufacturing facilities in the U.S., China, Germany, and is expanding to Mexico.

**Sector:** Tesla belongs to the automotive and energy sectors, focusing on electric vehicles and renewable energy solutions.

**Products:** Tesla's product portfolio includes high-performance electric vehicles (Model 3, Y, S, X, Cybertruck, and Tesla Semi), energy storage solutions (Powerwall and Megapack), and solar energy products (Solar Roof and retrofit solar systems).

**Revenue Model:** Tesla generates revenue through direct sales and leasing of electric vehicles, sales of automotive regulatory credits, energy storage products, and solar 
energy systems. Additional revenue streams include vehicle services, Supercharging, insurance, and software upgrades.

**Competition:** The automotive market is highly competitive, with increasing entries from both established and new manufacturers in the EV space. Tesla also faces competition in the energy storage and solar energy markets from traditional utilities and other renewable energy companies.

**Customer Base:** Tesla's primary customers include individual consumers, commercial entities, industrial clients, and utilities seeking sustainable energy solutions. The company also targets residential customers for its solar and energy storage products.

Tesla's vertically integrated business model, focus on innovation, and commitment to sustainability position it uniquely in the market, appealing to investors interested in 
the growing sectors of electric vehicles and renewable energy.'"""},

"dated_company_describer": {"role": "system", "content": """
Instruction:

You are an expert in writing business descriptions from a professional and objective perspective. When provided with a company name, its ticker symbol, a specific year, and a quarter, your task is to write a concise description of the company as it operated during that time period. Provide concise, clear, and focused responses. Avoid unnecessary elaboration or repetition

Requirements:
Business Operations and Products: Describe the company's primary operations, key products or services, and any relevant business activities as of the specified year and quarter.

Market Conditions: Reflect the broader market conditions, including supply and demand, industry trends, competition, and economic factors relevant to the company during that period.

Geographical Focus: Mention the company's geographical presence, highlighting key regions where it operates or has significant influence.

Sector and Industry: Clearly identify the sector(s) and industry(ies) in which the company operates.

Revenue Model: Outline the company’s revenue streams, specifying how it generates income through its products, services, or other business activities.

Customer Base: Identify the primary customers or client segments the company serves.

Writing Style:
Tense: Write in the present tense, as if describing the company in the context of the specified year and quarter. Write on point, avoid babbling.
Tone: Maintain an objective and neutral tone; avoid any promotional language or subjective opinions.
Format: Use a structured format similar to the example provided, ensuring that each aspect of the company is covered comprehensively.
Example:
Below is an example of a well-written company description for Tesla, Inc. (Ticker: TSLA) as of Q1 2023. Follow this format when writing your descriptions:

Tesla, Inc. (Ticker: TSLA) - 1Q 2023
                            
Business Operations and Products:
Tesla, Inc. operates primarily as an electric vehicle (EV) manufacturer, with its product lineup including the Model S, Model 3, Model X, and Model Y, which are all-electric cars catering to different market segments. The company is also deeply involved in the energy sector through its SolarCity subsidiary, providing solar energy solutions such as solar panels, solar roofs, and energy storage products like the Powerwall, Powerpack, and Megapack. During 1Q 2023, Tesla continues to advance its autonomous driving technology, offering features such as Full Self-Driving (FSD) software, which remains in beta testing.
Market Conditions:
In 1Q 2023, Tesla operates within a rapidly growing and competitive electric vehicle market, driven by increased global demand for sustainable transportation and renewable energy solutions. The EV market is experiencing strong growth due to regulatory support for reducing carbon emissions and advancements in battery technology. However, Tesla faces heightened competition from both established automotive manufacturers expanding their EV offerings and new entrants. Despite a maturing EV market, Tesla continues to benefit from its first-mover advantage and strong brand recognition. Supply chain challenges, particularly in semiconductor availability, continue to impact the broader automotive industry, including Tesla.
Geographical Focus:
Tesla maintains a significant global presence, with major manufacturing facilities in the United States (Fremont, California, and Gigafactory Texas), China (Gigafactory Shanghai), and Germany (Gigafactory Berlin). The company has a strong sales and service network across North America, Europe, and Asia, with China and Europe being key growth markets. Tesla's global supply chain and manufacturing footprint allow it to efficiently serve a diverse customer base while managing production costs and logistics.
Sector and Industry:
Tesla operates within the automotive sector, specifically focusing on the electric vehicle industry. Additionally, the company is a key player in the renewable energy sector, particularly in the solar energy and energy storage industries.
Revenue Model:
Tesla generates revenue primarily through the sale of electric vehicles, which accounts for the majority of its income. Additionally, the company derives revenue from its energy generation and storage segment, which includes the sale and installation of solar energy systems and energy storage products. Tesla also earns recurring income from its software offerings, including the Full Self-Driving (FSD) subscription service and over-the-air software updates, as well as from regulatory credits sold to other automakers.
Customer Base:
Tesla's primary customer base consists of individual consumers and businesses seeking electric vehicles, ranging from mid-tier to luxury segments. The company's energy products attract residential, commercial, and industrial customers interested in renewable energy solutions and energy independence. Tesla also appeals to environmentally conscious consumers and early adopters of technology who prioritize sustainability and innovation in their purchasing decisions.
Note: Ensure your description reflects the specific time frame (year and quarter) provided, accounting for any relevant changes in the company's operations, market conditions, or product offerings.
"""
}
}
USER_MESSAGES = {
    
    "summary_writer":{"role": "user", "content":  "Company Description:\n'{}'\n\nCompany description summary:"},
    
    "rewriter":{"role": "user", "content":  "Company Description Summary:\n'{}'\n\nRewritten company description summary:"},

    "dated_company_describer": {"role": "user", "content": "Write a company description of {}, ticker {}, as of {} {}."}
}

def pop_and_return(dictionary, key):
    dictionary.pop(key, None)
    return dictionary

def convert_to_tuple(item):
    """
    Recursively converts dictionaries to tuples, handling nested dictionaries as well.
    """
    if isinstance(item, dict):
        return tuple((key, convert_to_tuple(value)) for key, value in sorted(item.items()))
    elif isinstance(item, list):
        return tuple(convert_to_tuple(elem) for elem in item)
    else:
        return item

def plot_tsne_2d(vectors_dict):
    """
    This function takes a dictionary where keys are labels and values are high-dimensional vectors,
    applies t-SNE to reduce the dimensionality to 2D, and plots the 2D representation.

    :param vectors_dict: Dictionary where keys are labels (strings) and values are vectors (arrays).
    """
    # Extract labels and vectors from the dictionary
    labels = list(vectors_dict.keys())
    vectors = np.array(list(vectors_dict.values()))

    # Use t-SNE to reduce the dimensionality to 2D, set perplexity to a lower value
    tsne = TSNE(n_components=2, perplexity=2, random_state=42)
    vectors_2d = tsne.fit_transform(vectors)

    # Plot the 2D representation of the vectors
    plt.figure(figsize=(8, 6))
    plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], marker='o')

    # Annotate the points with their labels
    for i, label in enumerate(labels):
        plt.annotate(label, (vectors_2d[i, 0], vectors_2d[i, 1]))

    plt.title('2D visualization of high-dimensional vectors using t-SNE')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.show()

def remove_duplicate_dicts(dict_list):
    seen = set()  # Set to track seen dictionaries
    unique_list = []  # List to hold unique dictionaries
    
    for d in dict_list:
        # Convert dictionary and nested structures to a tuple
        d_tuple = convert_to_tuple(d)
        
        if d_tuple not in seen:
            seen.add(d_tuple)  # Add to seen set
            unique_list.append(d)  # Add to result list

    return unique_list
class CompanyDescriptionWriter:

    def __init__(self, openai_model="gpt-3.5-turbo-0125", top_p = 0.2, temperature=1.0,
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
    
    def get_company_description_example(self, company_name, ticker, quarter, year):
            
        user_message = USER_MESSAGES["dated_company_describer"].copy()
        user_message["content"] = user_message["content"].format(company_name, ticker, quarter, year)
        try:
            # Make the API call to the OpenAI chat completion endpoint
            response = openai.chat.completions.create(
                model=self.openai_model,
                messages=[SYSTEM_MESSAGES["dated_company_describer"], user_message],
                max_tokens=self.max_output_tokens,
                temperature=self.temperature,
                top_p=self.top_p
            )
            
            # Extract and return the response text
            return response.choices[0].message.content
        except Exception as e:
            return f"An error occurred: {str(e)}"
        
    def make_batch_job_helper(self, requests: list, batch_job_ids_file_path, is_chat_completion_request,
                        file_name = r"C:\repos\Deep-learning-trj\data\batch_api_file.jsonl",
                        overwrite_preexisting_batch_jobs=False, max_batch_size = 50000):
        
        """
        Helper function
        """
        # remove duplicate requests
        requests = remove_duplicate_dicts(requests)

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
    
        # Split the requests into sublists of size max_batch_size
        sublists = [requests[i:i + max_batch_size] for i in range(0, len(requests), max_batch_size)]
    
        # Create a file for each sublist and make a batch API call for each file
        for sublist in sublists:
            with open(file_name, "w") as file:
                for request in sublist:
                    file.write(json.dumps(request) + "\n")
            
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

    def create_company_description_requests(self, transaction_units: list):
        
        requests = []

        for transaction_unit in transaction_units:
        
            company_name = transaction_unit["proper_name"]
            ticker_quarter_year = transaction_unit["ticker_quarter_year"]
            ticker, quarter_year = ticker_quarter_year.split("_")
            quarter, year = quarter_year.split("Q")
            quarter = quarter + "Q"
            
            user_message = USER_MESSAGES["dated_company_describer"].copy()
            user_message["content"] = user_message["content"].format(company_name, ticker, quarter, year)
            
            messages = [
                SYSTEM_MESSAGES["dated_company_describer"],
                user_message
            ]
        
            request_body =  {"model": self.openai_model, "messages": messages,"max_tokens": self.max_output_tokens, "temperature": self.temperature, "top_p": self.top_p}
            request = {"custom_id": ticker_quarter_year, "method": "POST", "url": "/v1/chat/completions", "body": request_body}
            requests.append(request)
        
        return requests
    
    def create_company_descriptions_batch_job(self, transactions: pd.DataFrame, 
                                                        batch_job_ids_file_path = r"C:\repos\Deep-learning-trj\data\batch_id_list_company_descriptions.jsonl",
                                                        overwrite_preexisting_batch_jobs=False,
                                                        max_batch_size = 50000,
                                                        file_name = r"C:\repos\Deep-learning-trj\data\batch_api_file.jsonl",
                                                        pre_existing_company_descriptions_file = r"C:\repos\Deep-learning-trj\data\company_descriptions\company_descriptions.csv"
                                                        ):
        
        """ 
        group by investor, find last transaction and use its previous_transaction_list
        Must use all current portfolios. 
        """
        # Get last row of previous transaction for each investor
        previous_transactions = transactions.sort_values('report_date').groupby('factset_fund_id').apply(lambda x: json.loads(x.iloc[-1]['previous_transactions']))

        # Flatten the list of lists into a single list
        prev_transactions = [pop_and_return(transaction, "holding_change") for investor in previous_transactions for transaction in investor]
   
        transactions['portfolio_last_reporting_date'] = transactions['portfolio_last_reporting_date'].apply(lambda x: json.loads(x) if not isinstance(x, float) else [])
        
        prev_portfolios = [stock for portfolio in transactions['portfolio_last_reporting_date'] for stock in portfolio]

        transaction_units = prev_transactions + prev_portfolios
        
        company_description_requests = self.create_company_description_requests(transaction_units)
        
        # Check if we already have a company description for the given ticker_quarter_year
        pre_existing_company_descriptions = pd.read_csv(pre_existing_company_descriptions_file)

        company_description_requests = [request for request in company_description_requests if request["custom_id"] not in pre_existing_company_descriptions["ticker_quarter_year"].values]

        self.make_batch_job_helper(requests=company_description_requests, batch_job_ids_file_path=batch_job_ids_file_path,
                             is_chat_completion_request=True, file_name=file_name,
                             overwrite_preexisting_batch_jobs=overwrite_preexisting_batch_jobs,
                             max_batch_size=max_batch_size)
        
    def create_embeddings_batch_job(self, company_descriptions: pd.DataFrame,
                                    batch_job_ids_file_path=r"C:\repos\Deep-learning-trj\data\batch_id_list_embeddings.jsonl",
                                    file_name=r"C:\repos\Deep-learning-trj\data\batch_api_file.jsonl",
                                    overwrite_preexisting_batch_jobs=True, max_batch_size=50000, pre_existing_embedding_file = r"C:\repos\Deep-learning-trj\data\company_descriptions\company_description_embeddings.csv"):
        
        request_units = list(zip(company_descriptions["ticker_quarter_year"], company_descriptions["company_description"]))
        embeddings_requests = []
        for ticker_year_quarter, company_description in request_units:
        
            request_body = {"input": company_description, "model": self.openai_embedding_model, "dimensions": self.embedding_dim}
            request = {"custom_id": ticker_year_quarter, "method": "POST", "url": "/v1/embeddings", "body": request_body}
            embeddings_requests.append(request)
        
        # Check if we already have an embedding for the given ticker_quarter_year
        pre_existing_embeddings = pd.read_csv(pre_existing_embedding_file)

        embeddings_requests = [request for request in embeddings_requests if request["custom_id"] not in pre_existing_embeddings["ticker_quarter_year"].values]

        self.make_batch_job_helper(requests=embeddings_requests, batch_job_ids_file_path=batch_job_ids_file_path,
                             is_chat_completion_request=False, file_name=file_name,
                             overwrite_preexisting_batch_jobs=overwrite_preexisting_batch_jobs,
                             max_batch_size=max_batch_size) 
    
    def retrieve_batch_results(self, batch_ids_file_path,
                               batch_result_file_path=r"C:\repos\Deep-learning-trj\data\batch_job_results.jsonl",
                               ):

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
    
    def store_batch_results(self, batch_results, is_company_description, storage_file):

        if storage_file is None or not batch_results:
            return
        
        try:
            preexisting_df = pd.read_csv(storage_file)
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
            
            new_data_list.append({"ticker_quarter_year": custom_id, content_type: content})
        
        new_data_df = pd.DataFrame(new_data_list)
        new_data_df["date_created"] = pd.to_datetime('today').strftime('%Y-%m-%d')
        
        if preexisting_df is not None:
            # Perform the merge with suffixes for date columns
            new_data_df = pd.merge(preexisting_df, new_data_df, on=["ticker_quarter_year"], how='outer', suffixes=('_left', '_right'))

            # Prioritize the 'date_created_left' column but use 'date_created_right' where 'date_created_left' is NaN
            new_data_df["date_created"] = new_data_df["date_created_left"].combine_first(new_data_df["date_created_right"])

            # Drop the redundant columns
            new_data_df.drop(columns=["date_created_left", "date_created_right"], inplace=True)

        if storage_file is not None:
            new_data_df.to_csv(storage_file, sep="|", index=False)
        
if __name__== "__main__":
    
    writer = CompanyDescriptionWriter()
    
    #transactions = pd.read_csv(r"C:\repos\Deep-learning-trj\data\transactions\example_transactions.csv")
    
    #writer.create_company_descriptions_batch_job(transactions)

    # batch_results = writer.retrieve_batch_results(r"C:\repos\Deep-learning-trj\data\batch_id_list_company_descriptions.jsonl")

    # if batch_results:
    #     writer.store_batch_results(batch_results, is_company_description=True, storage_file=r"C:\repos\Deep-learning-trj\data\company_descriptions\company_descriptions.csv")

    ########## Create embeddings batch job ##########
    #company_descriptions = pd.read_csv(r"C:\repos\Deep-learning-trj\data\company_descriptions\company_descriptions.csv")
    #writer.create_embeddings_batch_job(company_descriptions)

    # batch_results = writer.retrieve_batch_results(r"C:\repos\Deep-learning-trj\data\batch_id_list_embeddings.jsonl")
    # writer.store_batch_results(batch_results, is_company_description=False, storage_file=r"C:\repos\Deep-learning-trj\data\company_descriptions\company_description_embeddings.csv")

    # df = pd.read_csv(r"C:\repos\Deep-learning-trj\data\company_descriptions\company_description_embeddings.csv", sep="|")
    # df['embedding'] = df['embedding'].apply(lambda x: ast.literal_eval(x))
    # df_dict = dict(zip(df['ticker_quarter_year'], df['embedding']))
    # desired_keys = [
    # 'ATEA-NO_1Q2024', 'ATEA-NO_2Q2024', 'MOWI-NO_1Q2024', 
    # 'MOWI-NO_2Q2024', 'SALM-NO_1Q2024', 'SALM-NO_2Q2024', 
    # 'BOUV-NO_1Q2024', 'BOUV-NO_2Q2024'
    # ]

    # filtered_dict = {key: df_dict[key] for key in desired_keys if key in df_dict}
    # plot_tsne_2d(filtered_dict)