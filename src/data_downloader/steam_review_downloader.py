import os
import sys
import logging
import asyncio
import aiohttp
import pandas as pd
from datetime import datetime
import urllib.parse
from collections import defaultdict

# Configure logging
logging.basicConfig(
    filename='./logs/downloader.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Global progress tracking
progress = defaultdict(lambda: defaultdict(int))
games_completed = 0
total_games = 0

def update_progress_display():
    """Update the console display with current progress for all games"""
    sys.stdout.write('\033[K')  # Clear the line

    # Show overall progress first
    print(f"Overall progress: {games_completed}/{total_games} games completed", end="\n")

    # Show active downloads
    messages = []
    for game, counts in progress.items():
        pos_count = counts['positive']
        neg_count = counts['negative']
        messages.append(f"{game}: +{pos_count} / -{neg_count} reviews")
    
    if messages:
        display = "Active downloads:\n" + "\n".join(messages)
        print(display, end='\r')

    sys.stdout.flush()

async def get_reviews_batch(session, app_id, game, review_type, cursor="*"):
    """Async function to fetch a single batch of reviews"""
    encoded_cursor = urllib.parse.quote(cursor)
    review_filter = 'positive' if review_type == 'positive' else 'negative'
    url = f"https://store.steampowered.com/appreviews/{app_id}?json=1&cursor={encoded_cursor}&num=100&language=english&purchase_type=all&filter={review_filter}&day_range=365"

    try:
        async with session.get(url) as response:
            data = await response.json()

            if not data.get('success', 0):
                logging.error(f"Failed to fetch {review_type} reviews for {game} ({app_id})")
                return None, None

            return data.get('reviews', []), data.get('cursor')

    except Exception as e:
        logging.error(f"Error fetching {review_type} reviews for {game} ({app_id}): {e}")
        return None, None

async def get_steam_reviews(app_id, game, review_type, total_reviews=100):
    """Asynchronously fetch all reviews of a specific type for a game"""
    reviews_list = []
    cursor = "*"

    async with aiohttp.ClientSession() as session:
        while True:
            reviews_batch, new_cursor = await get_reviews_batch(session, app_id, game, review_type, cursor)

            if not reviews_batch:
                break

            reviews_list.extend(reviews_batch)
            progress[game][review_type] = len(reviews_list)
            update_progress_display()

            if not new_cursor or new_cursor == '*':
                break

            if total_reviews and len(reviews_list) >= total_reviews:
                reviews_list = reviews_list[:total_reviews]
                break

            cursor = new_cursor
            await asyncio.sleep(1)

    logging.info(f"Completed {game} {review_type} reviews: {len(reviews_list)} reviews")
    return reviews_list

def save_reviews_to_csv(reviews, appid, game, review_type):
    """Save reviews to CSV with optimized DataFrame handling"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"data/downloader/steam_reviews_{appid}_{review_type}_{timestamp}.csv"

    df = pd.DataFrame.from_records(reviews)
    df['appid'] = appid
    df['fetch_date'] = timestamp
    df['review_type'] = review_type
    df['total_reviews'] = len(reviews)

    df.to_csv(filename, index=False, encoding='utf-8', compression=None)
    logging.info(f"Saved {game} {review_type} reviews to {filename}")

async def process_game(app_id, game):
    """Process both positive and negative reviews for a single game"""
    global games_completed

    try:
        logging.info(f"Starting download for {game} ({app_id})")
        
        # Create tasks for both positive and negative reviews
        positive_task = get_steam_reviews(app_id=app_id, game=game, review_type='positive')
        negative_task = get_steam_reviews(app_id=app_id, game=game, review_type='negative')
        
        # Wait for both tasks to complete
        positive_reviews, negative_reviews = await asyncio.gather(positive_task, negative_task)

        # Save both types of reviews
        if positive_reviews:
            save_reviews_to_csv(positive_reviews, app_id, game, 'positive')
        if negative_reviews:
            save_reviews_to_csv(negative_reviews, app_id, game, 'negative')

        # Update completion count and clean up progress
        games_completed += 1
        del progress[game]
        update_progress_display()

    except Exception as e:
        logging.error(f"Error processing {game}: {e}")
        games_completed += 1  # Count failed games as completed

async def main():
    global total_games

    app_ids_df = pd.read_csv(os.path.join(os.path.dirname(__file__), '../../data/steam_games.csv'))
    total_games = len(app_ids_df)
    print(f"Starting download for {total_games} games...")

    # Create queue of games to process
    games_queue = [(row['AppID'], row['Name']) for _, row in app_ids_df.iterrows()]
    active_tasks = set()
    max_concurrent = 3  # Reduced from 5 since we're now downloading 2 types per game

    while games_queue or active_tasks:
        # Start new tasks if we have capacity and games waiting
        while len(active_tasks) < max_concurrent and games_queue:
            app_id, game = games_queue.pop(0)
            task = asyncio.create_task(process_game(app_id, game))
            active_tasks.add(task)
            task.add_done_callback(active_tasks.discard)

        if active_tasks:
            # Wait for at least one task to complete
            await asyncio.wait(active_tasks, return_when=asyncio.FIRST_COMPLETED)
            update_progress_display()

        # Small delay to prevent hitting rate limits too hard
        await asyncio.sleep(1)

    print("\nDownload complete!")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nScript interrupted by user")
    finally:
        sys.stdout.write('\033[K')
        sys.stdout.flush()