"""
Category normalization mapping for Google Maps business categories.
Maps raw categories to ~25 normalized category families.
"""

# Normalized category families (25 categories)
CATEGORY_FAMILIES = [
    "restaurant",
    "cafe", 
    "bar",
    "dessert",
    "fast_food",
    "pizza",
    "sushi",
    "burger",
    "mexican",
    "italian",
    "chinese",
    "asian",
    "american",
    "seafood",
    "steakhouse",
    "bbq",
    "breakfast",
    "bakery",
    "coffee",
    "ice_cream",
    "pub",
    "brewery",
    "wine_bar",
    "nightclub",
    "other"
]

# Mapping from raw categories (lowercase) to normalized families
CATEGORY_MAP = {
    # Restaurant general
    "restaurant": "restaurant",
    "meal_takeaway": "restaurant",
    "meal_delivery": "restaurant",
    "food": "restaurant",
    "establishment": "restaurant",
    
    # Cafe & Coffee
    "cafe": "cafe",
    "coffee_shop": "coffee",
    "coffee": "coffee",
    "espresso_bar": "coffee",
    
    # Bar & Nightlife
    "bar": "bar",
    "night_club": "nightclub",
    "nightclub": "nightclub",
    "pub": "pub",
    "sports_bar": "bar",
    "cocktail_bar": "bar",
    "wine_bar": "wine_bar",
    "brewery": "brewery",
    "taproom": "brewery",
    
    # Dessert & Sweets
    "bakery": "bakery",
    "ice_cream_shop": "ice_cream",
    "ice_cream": "ice_cream",
    "dessert_shop": "dessert",
    "dessert": "dessert",
    "candy_store": "dessert",
    "chocolate_shop": "dessert",
    "donut_shop": "dessert",
    "frozen_yogurt_shop": "ice_cream",
    
    # Fast Food & Quick Service
    "fast_food_restaurant": "fast_food",
    "fast_food": "fast_food",
    "sandwich_shop": "fast_food",
    "deli": "fast_food",
    
    # Pizza
    "pizza_restaurant": "pizza",
    "pizza": "pizza",
    "pizzeria": "pizza",
    
    # Burger
    "hamburger_restaurant": "burger",
    "burger": "burger",
    "burger_restaurant": "burger",
    
    # Breakfast & Brunch
    "breakfast_restaurant": "breakfast",
    "brunch_restaurant": "breakfast",
    "breakfast": "breakfast",
    "brunch": "breakfast",
    
    # BBQ & Grill
    "barbecue_restaurant": "bbq",
    "bbq": "bbq",
    "grill": "steakhouse",
    "steakhouse": "steakhouse",
    "steak_house": "steakhouse",
    
    # Seafood
    "seafood_restaurant": "seafood",
    "seafood": "seafood",
    "fish_restaurant": "seafood",
    "sushi_restaurant": "sushi",
    "sushi": "sushi",
    "japanese_restaurant": "sushi",
    
    # American
    "american_restaurant": "american",
    "new_american_restaurant": "american",
    "southern_restaurant": "american",
    "soul_food_restaurant": "american",
    
    # Mexican
    "mexican_restaurant": "mexican",
    "mexican": "mexican",
    "tex-mex_restaurant": "mexican",
    "taco_restaurant": "mexican",
    "burrito_restaurant": "mexican",
    
    # Italian
    "italian_restaurant": "italian",
    "italian": "italian",
    "pasta_restaurant": "italian",
    
    # Chinese
    "chinese_restaurant": "chinese",
    "chinese": "chinese",
    "dim_sum_restaurant": "chinese",
    
    # Asian (other)
    "asian_restaurant": "asian",
    "thai_restaurant": "asian",
    "vietnamese_restaurant": "asian",
    "korean_restaurant": "asian",
    "indian_restaurant": "asian",
    "asian_fusion_restaurant": "asian",
    "ramen_restaurant": "asian",
    "pho_restaurant": "asian",
    "noodle_restaurant": "asian",
    
    # Other cuisines
    "mediterranean_restaurant": "restaurant",
    "middle_eastern_restaurant": "restaurant",
    "greek_restaurant": "restaurant",
    "french_restaurant": "restaurant",
    "spanish_restaurant": "restaurant",
    "latin_american_restaurant": "restaurant",
    "caribbean_restaurant": "restaurant",
    "african_restaurant": "restaurant",
    "ethiopian_restaurant": "restaurant",
    
    # Specialty
    "vegan_restaurant": "restaurant",
    "vegetarian_restaurant": "restaurant",
    "gluten-free_restaurant": "restaurant",
    "organic_restaurant": "restaurant",
    "health_food_restaurant": "restaurant",
    "salad_shop": "restaurant",
    
    # Buffet & Catering
    "buffet_restaurant": "restaurant",
    "caterer": "restaurant",
    "meal_delivery": "restaurant",
}


def normalize_category(raw_category: str) -> str:
    """
    Normalize a raw category string to a category family.
    
    Args:
        raw_category: Raw category string from Google Maps
        
    Returns:
        Normalized category family string
    """
    if not raw_category:
        return "other"
    
    # Convert to lowercase and remove extra whitespace
    raw_lower = raw_category.lower().strip().replace(" ", "_")
    
    # Look up in mapping
    return CATEGORY_MAP.get(raw_lower, "other")


def normalize_category_list(categories: list) -> tuple:
    """
    Normalize a list of categories.
    
    Args:
        categories: List of raw category strings
        
    Returns:
        Tuple of (category_main, category_all_normalized)
        - category_main: First normalized category (primary)
        - category_all_normalized: List of all unique normalized categories
    """
    if not categories or len(categories) == 0:
        return ("other", ["other"])
    
    # Normalize all categories
    normalized = [normalize_category(cat) for cat in categories]
    
    # Remove duplicates while preserving order
    seen = set()
    unique_normalized = []
    for cat in normalized:
        if cat not in seen:
            seen.add(cat)
            unique_normalized.append(cat)
    
    # First category is the main one
    category_main = unique_normalized[0] if unique_normalized else "other"
    
    return (category_main, unique_normalized)


def get_primary_food_category(normalized_categories):
    """
    Extract primary food category from normalized category list.
    
    Args:
        normalized_categories: List of normalized category strings (24 families)
        
    Returns:
        Primary food category string, or None if no food categories exist
    """
    if not normalized_categories:
        return None
    
    # Food categories in priority order (most specific → most general)
    # These are the 24 normalized food families we already have
    food_priority = [
        # Specific cuisines/types (most specific first)
        'sushi', 'pizza', 'burger', 'bbq', 'steakhouse', 'seafood',
        'mexican', 'italian', 'chinese', 'asian', 'american',
        # Meal types
        'breakfast', 'fast_food',
        # Desserts & drinks
        'bakery', 'dessert', 'ice_cream', 'coffee', 'cafe',
        # Bars & nightlife
        'wine_bar', 'brewery', 'pub', 'bar', 'nightclub',
        # Generic (last resort)
        'restaurant'
    ]
    
    # Return first food category found (in priority order)
    for priority_cat in food_priority:
        if priority_cat in normalized_categories:
            return priority_cat
    
    # No food category found - this is a non-food business
    return None

