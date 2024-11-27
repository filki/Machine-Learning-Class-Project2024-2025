import requests
import os
import time
from datetime import datetime
import urllib.parse
import sys
import logging
import pandas as pd

# Configure logging at the beginning of the script
logging.basicConfig(
    filename='./logs/downloader.log', 
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def get_steam_reviews(app_id, game, cursor="*", total_reviews=None):
    reviews_list = []
    
    while True:
        encoded_cursor = urllib.parse.quote(cursor)
        url = f"https://store.steampowered.com/appreviews/{app_id}?json=1&cursor={encoded_cursor}&num=100&language=all&purchase_type=all&filter=all&day_range=365"
        
        try:
            logging.info(f"Fetching reviews for {game} ({app_id}).")
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            
            if not data.get('success', 0):
                logging.error(f"Failed to fetch reviews for {game} ({app_id})")
                print("\nFailed to fetch reviews")
                break

            reviews_data = data.get('reviews', [])  
            
            if not reviews_data:
                logging.info(f"No more reviews available for {game} ({app_id})")
                print("\nNo more reviews available")
                break
                
            reviews_list.extend(reviews_data)
            
            # Logging progress
            sys.stdout.write('\033[K')  # Clear line
            progress_message = f"Progress: {len(reviews_list)} reviews fetched for: {app_id} ({game})."
            print(progress_message ,end='\r')
            sys.stdout.flush()
            logging.info(progress_message)
            
            cursor = data.get('cursor')
            if not cursor or cursor == '*':
                logging.info(f"Reached end of reviews for {game} ({app_id})")
                print("\nReached end of reviews")
                break
                
            if total_reviews and len(reviews_list) >= total_reviews:
                reviews_list = reviews_list[:total_reviews]
                logging.info(f"Reached target of {total_reviews} reviews for {game} ({app_id})")
                break
            
            time.sleep(1.5)
            
        except requests.exceptions.RequestException as e:
            logging.error(f"Error fetching reviews for {game} ({app_id}): {e}")
            print(f"\nError fetching reviews: {e}")
            break
        except KeyError as e:
            logging.error(f"Unexpected API response structure for {game} ({app_id}): {e}, Response: {data}")
            print(f"\nUnexpected API response structure: {e}")
            print(f"Response: {data}")
            break
        except Exception as e:
            logging.error(f"Unexpected error for {game} ({app_id}): {e}")
            print(f"\nUnexpected error: {e}")
            break
    
    logging.info(f"Finished fetching {len(reviews_list)} reviews for {game} ({app_id})")
    print(f"\nFinished fetching {len(reviews_list)} reviews")
    return reviews_list


def save_reviews_to_csv(reviews, appid):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"data/steam_reviews_{appid}_{timestamp}.csv"
    
    logging.info(f"Saving {len(reviews)} reviews to {filename}...")
    print(f"\nSaving {len(reviews)} reviews to {filename}...")
    
    # Convert reviews to a DataFrame
    df = pd.json_normalize(reviews, sep='_')
    
    # Add app-specific metadata
    df['appid'] = appid
    df['fetch_date'] = timestamp
    df['total_reviews'] = len(reviews)
    
    # Write to CSV
    df.to_csv(filename, index=False, encoding='utf-8')
    logging.info(f"Successfully saved reviews to {filename}")
    print(f"Successfully saved reviews to {filename}")


def main():
    # Read the CSV containing app IDs
    app_ids_df = pd.read_csv(os.path.join(os.path.dirname(__file__), '../../93182_steam_games.csv'))
    
    for _, row in app_ids_df.iterrows():
        app_id = row['AppID']
        game = row['Name']
        logging.info(f"Starting to fetch reviews for app ID: {app_id} ({game})")
        print(f"Starting to fetch reviews for app ID: {app_id} ({game})")
        
        reviews = get_steam_reviews(app_id=app_id, game=game, total_reviews=100)
        
        if reviews:
            save_reviews_to_csv(reviews, app_id)

if __name__ == "__main__":
    main()
