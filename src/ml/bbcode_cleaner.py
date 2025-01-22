import re
from typing import List, Dict, Union
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BBCodeCleaner:
    """A class to clean BBCode tags from text while preserving meaningful content."""
    
    # Common BBCode tags used in Steam reviews
    BBCODE_PATTERNS = {
        'headers': r'\[h[1-6]\](.*?)\[/h[1-6]\]',
        'bold': r'\[b\](.*?)\[/b\]',
        'italic': r'\[i\](.*?)\[/i\]',
        'underline': r'\[u\](.*?)\[/u\]',
        'strike': r'\[s\](.*?)\[/s\]',
        'url_with_text': r'\[url=.*?\](.*?)\[/url\]',
        'url_simple': r'\[url\](.*?)\[/url\]',
        'image': r'\[img\].*?\[/img\]',
        'quote': r'\[quote\](.*?)\[/quote\]',
        'list': r'\[list\](.*?)\[/list\]',
        'list_item': r'\[\*\](.*?)(?=\[\*\]|\[/list\])',
        'size': r'\[size=.*?\](.*?)\[/size\]',
        'color': r'\[color=.*?\](.*?)\[/color\]',
        'spoiler': r'\[spoiler\](.*?)\[/spoiler\]',
        'code': r'\[code\](.*?)\[/code\]',
        'noparse': r'\[noparse\](.*?)\[/noparse\]',
        'table': r'\[table\].*?\[/table\]',
        'emoticon': r'\[emoticon\].*?\[/emoticon\]'
    }
    
    def __init__(self, preserve_content: bool = True):
        """
        Initialize the BBCode cleaner.
        
        Args:
            preserve_content (bool): If True, preserves the content within tags.
                                   If False, removes both tags and content.
        """
        self.preserve_content = preserve_content
        
    def clean_text(self, text: str) -> str:
        """
        Clean BBCode tags from a single text string.
        
        Args:
            text (str): Text containing BBCode tags
            
        Returns:
            str: Cleaned text with BBCode tags removed
        """
        if not isinstance(text, str):
            return ""
            
        cleaned_text = text
        
        # Handle each BBCode pattern
        for tag_name, pattern in self.BBCODE_PATTERNS.items():
            if self.preserve_content:
                # Replace tags with their content
                cleaned_text = re.sub(pattern, r'\1', cleaned_text, 
                                    flags=re.IGNORECASE | re.DOTALL)
            else:
                # Remove tags and their content completely
                cleaned_text = re.sub(pattern, '', cleaned_text, 
                                    flags=re.IGNORECASE | re.DOTALL)
        
        # Clean up extra whitespace
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
        cleaned_text = cleaned_text.strip()
        
        return cleaned_text
    
    def clean_reviews(self, reviews: Union[List[str], List[Dict]]) -> List[str]:
        """
        Clean BBCode tags from a list of reviews.
        
        Args:
            reviews (Union[List[str], List[Dict]]): List of review texts or dictionaries
                                                   containing review text
                
        Returns:
            List[str]: List of cleaned review texts
        """
        cleaned_reviews = []
        
        for review in reviews:
            if isinstance(review, dict):
                # If review is a dictionary, extract the text field
                # Assuming 'review' or 'text' as possible keys
                text = review.get('review', review.get('text', ''))
            else:
                text = review
                
            cleaned_text = self.clean_text(text)
            cleaned_reviews.append(cleaned_text)
            
        logger.info(f"Cleaned {len(cleaned_reviews)} reviews")
        return cleaned_reviews
    
    def analyze_bbcode_usage(self, reviews: List[str]) -> Dict[str, int]:
        """
        Analyze the usage of different BBCode tags in the reviews.
        
        Args:
            reviews (List[str]): List of review texts
            
        Returns:
            Dict[str, int]: Dictionary with tag counts
        """
        tag_counts = {tag: 0 for tag in self.BBCODE_PATTERNS.keys()}
        
        for review in reviews:
            if not isinstance(review, str):
                continue
                
            for tag, pattern in self.BBCODE_PATTERNS.items():
                matches = re.findall(pattern, review, flags=re.IGNORECASE | re.DOTALL)
                tag_counts[tag] += len(matches)
        
        logger.info("BBCode usage analysis completed")
        return tag_counts


# Example usage
if __name__ == "__main__":
    # Example reviews with BBCode
    example_reviews = [
        "[h1]Great Game![/h1] [b]Highly recommended![/b]",
        "[quote]Best game ever[/quote] [url=steam://]Check it out[/url]",
        "[spoiler]Secret ending is amazing[/spoiler]"
    ]
    
    # Initialize cleaner
    cleaner = BBCodeCleaner(preserve_content=True)
    
    # Clean reviews
    cleaned_reviews = cleaner.clean_reviews(example_reviews)
    print("\nCleaned Reviews:")
    for original, cleaned in zip(example_reviews, cleaned_reviews):
        print(f"\nOriginal: {original}")
        print(f"Cleaned:  {cleaned}")
    
    # Analyze BBCode usage
    tag_counts = cleaner.analyze_bbcode_usage(example_reviews)
    print("\nBBCode Tag Usage:")
    for tag, count in tag_counts.items():
        if count > 0:
            print(f"{tag}: {count} occurrences")
