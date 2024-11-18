import requests

def get_steam_reviews(appid):
    url = f"https://store.steampowered.com/appreviews/{appid}?json=1"
    response = requests.get(url)
    if response.status_code == 200:
         reviews = response.json()
         return reviews
    else:
        print(f"Error fetching reviews: {response.status_code}")
        return None
    
appid = 70 
reviews = get_steam_reviews(appid)

print(reviews)