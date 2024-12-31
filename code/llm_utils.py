import pandas as pd
import openai
import json
import os
import tiktoken
import torch
from openai import OpenAI
from sentence_transformers import SentenceTransformer, models
from transformers import AutoModel, AutoTokenizer


class OpenaiModel:

    def __init__(self, api_key, model="gpt-4o-mini", top_p=1, temperature=0,
                frequency_penalty = 0, presence_penalty=0,
                input_token_limit = 80000, max_output_tokens=1024,
                openai_embedding_model="text-embedding-3-small", embedding_dim=1536, max_tokens_embedding_model=8191,
                ):
        
        self.input_token_limit = input_token_limit
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.tokenizer = tiktoken.encoding_for_model(self.model)
        self.top_p = top_p
        self.temperature = temperature
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.openai_embedding_model = openai_embedding_model
        self.embedding_dim = embedding_dim
        self.max_tokens_embedding_model = max_tokens_embedding_model
        self.client = OpenAI(api_key=api_key)
        self.api_key = api_key
    
    def count_tokens(self, text):
        # Initialize the encoder for the specified model
        encoding = tiktoken.encoding_for_model(self.model)
        # Encode the text and get the list of tokens
        tokens = encoding.encode(text)
        # Return the number of tokens
        return len(tokens)

    def generate_text(self, system_message: dict, user_message: dict):
        
        try:
            # Make the API call to the OpenAI chat completion endpoint
            response = openai.chat.completions.create(
                model=self.model,
                messages=[system_message, user_message],
                max_tokens=self.max_output_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                frequency_penalty=self.frequency_penalty,
                presence_penalty=self.presence_penalty
            )
            
            # Extract and return the response text
            return response.choices[0].message.content
        except Exception as e:
            return f"An error occurred: {str(e)}"
    
    def create_embedding(self, text: str):
        embedding = self.client.embeddings.create(
        input=text, model=self.openai_embedding_model, dimensions=self.embedding_dim).data[0].embedding

        return embedding

    def create_request_unit(self, custom_id: str, system_message : dict, user_message: dict, is_chat_completion_request: bool):

        request_body =  {"model": self.model, "messages": [system_message, user_message], "max_tokens": self.max_output_tokens,
                            "temperature": self.temperature, "top_p": self.top_p, "frequency_penalty": self.frequency_penalty,
                            "presence_penalty": self.presence_penalty}
        
        url = "/v1/chat/completions" if is_chat_completion_request else "/v1/embeddings"

        request_unit = {"custom_id": custom_id, "method": "POST", "url": url, "body": request_body}

        return request_unit
    
    def post_batch_job(self, request_units: list, batch_job_ids_file_path: str, is_chat_completion_request: bool,
                        request_units_file_name: str, overwrite_preexisting_batch_jobs: bool, max_batch_size: int = 50000):
        
        """
        Function that sends request to OpenAI to complete a batch job
        """

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
            with open(request_units_file_name, "w") as file:
                for request_unit in batch_job:
                    file.write(json.dumps(request_unit) + "\n")
            
            with open(request_units_file_name, "rb") as file:
                batch_file = self.client.files.create(
                    file=file,
                    purpose="batch"
                )

            end_point = '/v1/chat/completions' if is_chat_completion_request else '/v1/embeddings'

            batch_response = self.client.batches.create(
                input_file_id=batch_file.id,
                endpoint=end_point,
                completion_window="24h"
            )
            
            batch_job_ids.append(batch_response.id)
    
        # Save the batch IDs to a file for later retrieval
        with open(batch_job_ids_file_path, "w") as file:
            json.dump({"batch_job_ids": batch_job_ids}, file)

        print("Successfully created {} batch job(s)".format(len(batch_job_ids)))

    def get_batch_results(self, batch_ids_file_path:str, batch_result_file_path: str):

        """
        Function that retrieves the results of a batch job
        """
        with open(batch_ids_file_path, "r") as file:
            json_line = file.readline()
            data = json.loads(json_line)

        batch_id_list = data.get("batch_job_ids", [])

        all_batch_results = []
        
        for batch_id in batch_id_list:
            batch_job = self.client.batches.retrieve(batch_id)

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
                result = self.client.files.content(result_file_id).content.decode("UTF-8")
                with open(batch_result_file_path, "w") as file:  # Open in text mode
                    file.write(result)
                
                results = []

                with open(batch_result_file_path, "r") as file:
                    for line in file:
                        json_object = json.loads(line.strip())
                        results.append(json_object)
                all_batch_results.extend(results)
        
        if not all_batch_results:
            print("No batch results were retrieved.")
            return None
        
        return all_batch_results
    
    @staticmethod
    def store_batch_results(batch_results: list, is_chat_completion: bool, custom_id_col: str, content_col: str, preexisting_file_path: str = None, storage_file:str = None, overwrite=False):

        """
        Function that stores the results of a batch job in a file
        """
        if not batch_results:
            print("No storage file provided or no batch results to store.")
            return None
        try:
            preexisting_df = pd.read_parquet(preexisting_file_path)
        except:
            preexisting_df = None
            
        new_data_list = []
        for request_result in batch_results:
            custom_id = request_result["custom_id"]
            if is_chat_completion:
                content = request_result["response"]["body"]["choices"][0]["message"]["content"]
            else:
                content = request_result["response"]["body"]["data"][0]["embedding"]
                
            new_data_list.append({custom_id_col: custom_id, content_col: content})

        new_data_df = pd.DataFrame(new_data_list)
        new_data_df["date_created"] = pd.to_datetime('today').strftime('%Y-%m-%d')
        
        if preexisting_df is not None:
            # Perform the merge with suffixes for date columns
            new_data_df = pd.merge(preexisting_df, new_data_df, on=[custom_id_col], how='outer', suffixes=('_left', '_right'))

            # Prioritize the 'date_created_left' column but use 'date_created_right' where 'date_created_left' is NaN
            if not overwrite:
                new_data_df["date_created"] = new_data_df["date_created_left"].combine_first(new_data_df["date_created_right"])
                new_data_df[content_col] = new_data_df[content_col + "_left"].combine_first(new_data_df[content_col+"_right"])
            else:
                new_data_df["date_created"] = new_data_df["date_created_right"].combine_first(new_data_df["date_created_left"])
                new_data_df[content_col] = new_data_df[content_col+"_right"].combine_first(new_data_df[content_col + "_left"])
            new_data_df.drop(columns=["date_created_left", "date_created_right", content_col +"_left", content_col+"_right"], inplace=True)
        
        if storage_file is not None:
            new_data_df.to_parquet(storage_file)

        return new_data_df

    @staticmethod
    def filter_request_units(request_units: list, pre_existing_df_path, custom_id_col: str):

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
            filtered_request_units = [request_unit for request_unit in request_units if request_unit["custom_id"] not in pre_existing_df[custom_id_col].values]
        else:
            # If None, assume no pre-existing descriptions, so all requests are valid
            filtered_request_units = request_units

        return filtered_request_units
    
    def retrieve_batch(self, batch_id: str):
        response = self.client.batches.retrieve(batch_id)
        return response
    
    def cancel_batch_job(self, batch_id: str):
        # Construct the API endpoint
        response = self.client.batches.cancel(batch_id)
        return response

class HuggingFaceModel:

    def __init__(self, model="bert-base-uncased", tokenizer=None):
        self.model = AutoModel.from_pretrained(model)
        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(model)
        else:
            self.tokenizer = tokenizer
    
    def encode_texts(self, texts: list, truncation=True, padding=True, max_length=None, tokenizer=None):

        """
        Function that encodes company descriptions using a Hugging Face model
        """
        if tokenizer is None:
            tokenizer = self.tokenizer

        encoded_text = tokenizer(texts, return_tensors="pt", truncation=truncation, padding=padding, max_length=max_length)
        input_ids = torch.tensor(encoded_text["input_ids"].tolist())
        attention_mask = torch.tensor(encoded_text["attention_mask"].tolist())
        
        return input_ids, attention_mask
    
    def batch_encode_texts(self, texts: list, batch_size: int = 10000, truncation=True, padding=True, max_length=None,
                           tokenizer=None) -> tuple:
        """
        Function to encode texts in batches using a Hugging Face model tokenizer.
        """
        if tokenizer is None:
            tokenizer = self.tokenizer
        
        input_ids_list = []
        attention_mask_list = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            encoded_text = tokenizer(
                batch, 
                return_tensors="pt", 
                truncation=truncation, 
                padding=padding, 
                max_length=max_length
            )

            input_ids_list.append(encoded_text["input_ids"])
            attention_mask_list.append(encoded_text["attention_mask"])

        input_ids = torch.cat(input_ids_list, dim=0)
        attention_mask = torch.cat(attention_mask_list, dim=0)
        
        return input_ids, attention_mask

    def count_tokens(self, text: str, tokenizer=None):
        """
        Function to count the number of tokens in a given text using the tokenizer.
        """
        if tokenizer is None:
            tokenizer = self.tokenizer
        
        encoded_text = tokenizer(text, return_tensors="pt", truncation=False, padding=False)
        return len(encoded_text["input_ids"][0])

    def decode_tokens(self, input_ids, attention_mask, tokenizer=None):
        """
        Function to decode the tokenized input IDs back to text using the tokenizer.
        """
        if tokenizer is None:
            tokenizer = self.tokenizer
        
        decoded_text = tokenizer.decode(token_ids=input_ids, attention_mask=attention_mask)
        return decoded_text

    def get_model_output(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask)

if __name__== "__main__":
    t1 = "Hello, my name is Per"
    t2 = "The dog is red"

    texts = [t1, t2]

    stransformer = SentenceTransformer(modules=[models.Transformer("bert-large-uncased")])
    hfacemodel = HuggingFaceModel(tokenizer=stransformer.tokenizer)

    i, a = hfacemodel.batch_encode_texts(texts, batch_size=1, padding="max_length", max_length=512)
    print(i.shape)
    decoded_text = hfacemodel.decode_tokens(i[0], a[0])
    print(decoded_text)


