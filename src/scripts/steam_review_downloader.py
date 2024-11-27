import requests
import json
import time
from datetime import datetime
import urllib.parse
import sys
import pandas as pd

def get_steam_reviews(app_id, cursor="*", total_reviews=None):
    reviews_list = []
    total_possible = None
    
    while True:
        encoded_cursor = urllib.parse.quote(cursor)
        url = f"https://store.steampowered.com/appreviews/{app_id}?json=1&cursor={encoded_cursor}&num=100&language=all&purchase_type=all&filter=all&day_range=365"
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            
            if not data.get('success', 0):
                print("\nFailed to fetch reviews")
                break
                
            query_summary = data.get('query_summary', {})
            reviews_data = data.get('reviews', [])
            
            # Get total reviews count from the first request
            if total_possible is None:
                total_possible = query_summary.get('total_reviews', 0)
                print(f"\nTotal reviews available: {total_possible}")
            
            if not reviews_data:
                print("\nNo more reviews available")
                break
                
            reviews_list.extend(reviews_data)
            
            sys.stdout.write('\033[K')  # Clear line
            print(f"Progress: {len(reviews_list)} / {total_possible} reviews fetched ({(len(reviews_list)/total_possible*100):.1f}%)", end='\r')
            sys.stdout.flush()
            
            cursor = data.get('cursor')
            if not cursor or cursor == '*':
                print("\nReached end of reviews")
                break
                
            if total_reviews and len(reviews_list) >= total_reviews:
                reviews_list = reviews_list[:total_reviews]
                break
            
            time.sleep(1.5)
            
        except requests.exceptions.RequestException as e:
            print(f"\nError fetching reviews: {e}")
            break
        except KeyError as e:
            print(f"\nUnexpected API response structure: {e}")
            print(f"Response: {data}")
            break
        except Exception as e:
            print(f"\nUnexpected error: {e}")
            break
    
    print(f"\nFinished fetching {len(reviews_list)} reviews")
    return reviews_list


def save_reviews_to_csv(reviews, appid):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"data/steam_reviews_{appid}_{timestamp}.csv"
    
    print(f"\nSaving {len(reviews)} reviews to {filename}...")
    
    # Convert reviews to a DataFrame
    df = pd.json_normalize(reviews, sep='_')
    
    # Add app-specific metadata
    df['appid'] = appid
    df['fetch_date'] = timestamp
    df['total_reviews'] = len(reviews)
    
    # Write to CSV
    df.to_csv(filename, index=False, encoding='utf-8')
    print(f"Successfully saved reviews to {filename}")

def main():

    app_id = 220 
    #70 Half Life
    #220 Half Life 2

    print(f"Starting to fetch reviews for app ID: {app_id}")
    reviews = get_steam_reviews(app_id, total_reviews=100)
    
    if reviews:
        save_reviews_to_csv(reviews, app_id)

if __name__ == "__main__":
    main()