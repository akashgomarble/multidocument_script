import os
from dotenv import load_dotenv
from api_client import fetch_data

# Load environment variables
load_dotenv()

def main():
    # You can either use the default API token from api_client.py
    # or provide your own from environment variables or directly
    
    # Option 1: Use default token from api_client.py
    print("Using default API token from api_client.py")
    
    # Example parameters with straight quotes
    params = {
        "searchTerms": ["marketing example"],
        "pageType": "SEARCH",
        "adType": "ALL",
        "adStatus": "ACTIVE"
    }
    
    try:
        results = fetch_data(params=params)
        print(f"Retrieved {len(results)} results")
        
        # Print the first result if available
        if results and len(results) > 0:
            print("\nFirst result:")
            print(results[0])
    except Exception as e:
        print(f"Error: {str(e)}")
    
    # Option 2: Use token from environment variable
    print("\n\nUsing API token from environment variable")
    api_key = os.getenv("APIFY_API_KEY")
    
    if api_key:
        try:
            results = fetch_data(api_key=api_key, params=params)
            print(f"Retrieved {len(results)} results")
        except Exception as e:
            print(f"Error: {str(e)}")
    else:
        print("APIFY_API_KEY not found in environment variables")

if __name__ == "__main__":
    main() 