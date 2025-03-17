import requests
import json

# Correct API URL with straight quotes
api_url = "https://api.apify.com/v2/acts/curious_coder~facebook-ads-library-scraper/run-sync-get-dataset-items"

# Correct API token with straight quotes
api_token = "apify_api_VcJ2nOpWzMVPp8SgW8fVwoaQfrsR202iusRV"

def fetch_data(api_key=None, params=None):
    """
    Fetch data from the API using the provided API key and parameters
    
    Args:
        api_key (str): Your API key for authentication, defaults to api_token if not provided
        params (dict): Parameters for the API request
        
    Returns:
        dict: The JSON response from the API
    """
    if api_key is None:
        api_key = api_token
        
    if params is None:
        params = {}
        
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    response = requests.post(api_url, headers=headers, json=params)
    
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"API request failed with status code {response.status_code}: {response.text}")

# Example usage
if __name__ == "__main__":
    # Example parameters
    params = {
        "searchTerms": ["example term"],
        "pageType": "SEARCH",
        "adType": "ALL",
        "adStatus": "ACTIVE"
    }
    
    try:
        results = fetch_data(params=params)
        print(f"Retrieved {len(results)} results")
    except Exception as e:
        print(f"Error: {str(e)}") 