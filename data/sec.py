
import os
from sec_api import ExtractorApi, QueryApi
from dotenv import load_dotenv, find_dotenv


load_dotenv()
SEC_API_KEY = os.getenv("SEC_API_KEY") or "your_api_key_here"

extractor_api = ExtractorApi(SEC_API_KEY)
query_api = QueryApi(SEC_API_KEY)

def get_filing_urls(ticker, start_date, end_date, size=50, form_type="10-K"):
    start = 0
    filing_urls = []

    while True:
        query = {
            "query": {
                "bool": {
                    "must": [
                        {"query_string": {"query": f'(formType:"{form_type}") AND ticker:{ticker}'}},
                        {"range": {"periodOfReport": {"gte": start_date, "lte": end_date}}}
                    ],
                    "must_not": [
                        {"query_string": {"query": 'formType:"10-K/A"'}}
                    ]
                }
            },
            "from": start,
            "size": size,
            "sort": [{"filedAt": {"order": "asc"}}]
        }
        
        query_result = query_api.get_filings(query)
        filings = query_result["filings"]
        if len(query_result["filings"]) < 1:
            break
        start += size
        for filing in filings:
            filing_urls.append(filing["linkToFilingDetails"])
        
    return filing_urls

def get_first_preceding_filing_url(ticker, date, form_type="10-K"):
    date_splitted = date.split("-")
    start_date = str(int(date_splitted[0])-2) + "-" + date_splitted[1] + "-" + date_splitted[2]
    filing_urls = get_filing_urls(ticker, start_date, date, form_type=form_type)
    if len(filing_urls) < 1:
        return None
    return filing_urls[-1]

def get_company_description(ticker, date):
    ticker_code_country = ticker.split("-")
    if len(ticker_code_country) < 2:
        raise ValueError("Invalid ticker format - must include ticker code")

    ticker, country_code = ticker_code_country 
    if country_code == "US":
        pass
    else:
        raise ValueError("Only US companies are supported")
    
    url = get_first_preceding_filing_url(ticker, date, form_type="10-K")
    if url is None:
        return None
    company_description = extractor_api.get_section(url, "1", "text")

    return company_description

#print(get_company_description("AAPL-US", "2024-05-06"))
