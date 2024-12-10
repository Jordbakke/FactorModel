import pandas as pd
import openai
import json
import os
import tiktoken
import sys
import torch
from transformers import AutoTokenizer, AutoModel

class OpenaiModel:

    def __init__(self, model="gpt-4o-mini", top_p = 1, temperature=0,
                frequency_penalty = 0, presence_penalty=0,
                input_token_limit = 80000, max_output_tokens=512,
                openai_embedding_model="text-embedding-3-small", embedding_dim=1536, max_tokens_embedding_model=8191):
        
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
    
    def count_tokens_openai(self, text):
        # Initialize the encoder for the specified model
        encoding = tiktoken.encoding_for_model(self.model)
        # Encode the text and get the list of tokens
        tokens = encoding.encode(text)
        # Return the number of tokens
        return len(tokens)

    def generate_text(self, system_message, user_message):
        
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
    
    def create_embedding(self, client: openai.OpenAI, text: str):
        embedding = client.embeddings.create(
        input=text, model=self.openai_embedding_model, dimensions=self.embedding_dim).data[0].embedding

        return embedding
    
    def post_batch_job(self, client:openai.OpenAI, request_units: list, batch_job_ids_file_path: str, is_chat_completion_request: bool,
                        file_name: str, overwrite_preexisting_batch_jobs: bool=False, max_batch_size: int = 50000):
        
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

    def get_batch_results(self, client: openai.OpenAI, batch_ids_file_path:str ,
                               batch_result_file_path
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
    
    def store_batch_results(self, batch_results: list, is_chat_completion: bool, custom_id_col: str, content_col: str, storage_file:str):
        
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
            new_data_df["date_created"] = new_data_df["date_created_left"].combine_first(new_data_df["date_created_right"])
            # Drop the redundant columns
            new_data_df.drop(columns=["date_created_left", "date_created_right"], inplace=True)
        
        if storage_file is not None:
            new_data_df.to_parquet(storage_file)

    def filter_request_units(self, request_units, pre_existing_df_path, custom_id_col: str):

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

class HuggingFaceModel:

    def __init__(self, model_name="bert-base-uncased"):
        self.model_name = model_name
        self.model = AutoModel.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def count_tokens(self, text):
        """
        Function that counts the number of tokens in a text
        """
        tokens = self.tokenizer.tokenize(text)
        return len(tokens)
    
    def encode_texts_huggingface(self, texts: list, truncation=True, padding=True, max_length=None):

        """
        Function that encodes company descriptions using a Hugging Face model
        """
        encoded_text = self.tokenizer(texts, return_tensors="pt", truncation=truncation, padding=padding, max_length=max_length)
        input_ids = torch.tensor(encoded_text["input_ids"].tolist())
        attention_masks = torch.tensor(encoded_text["attention_mask"].tolist())
        
        return input_ids, attention_masks
    
    def get_model_output(self, input_ids, attention_masks):
        return self.model(input_ids, attention_masks)



if __name__== "__main__":
    t1 = "Hello, my name is Per"
    t2 = "The dog is red"

    model = HuggingFaceModel()

    input_ids, attention_mask = model.encode_texts_huggingface([t1, t2])

    print(input_ids.shape)
    print(attention_mask.shape)

    output = model.generate_cls_embedding(input_ids, attention_mask)
    print(output)

    output = model.get_last_hidden_state(input_ids, attention_mask)[:, 0, :]
    print(output)