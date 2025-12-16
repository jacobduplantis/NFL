"""
NFL Team Names Reference
Complete list of valid NFL team names for validation
"""

# Official NFL team names (current as of 2024)
NFL_TEAMS = [
    # AFC East
    "Buffalo Bills",
    "Miami Dolphins",
    "New England Patriots",
    "New York Jets",

    # AFC North
    "Baltimore Ravens",
    "Cincinnati Bengals",
    "Cleveland Browns",
    "Pittsburgh Steelers",

    # AFC South
    "Houston Texans",
    "Indianapolis Colts",
    "Jacksonville Jaguars",
    "Tennessee Titans",

    # AFC West
    "Denver Broncos",
    "Kansas City Chiefs",
    "Las Vegas Raiders",
    "Los Angeles Chargers",

    # NFC East
    "Dallas Cowboys",
    "New York Giants",
    "Philadelphia Eagles",
    "Washington Commanders",

    # NFC North
    "Chicago Bears",
    "Detroit Lions",
    "Green Bay Packers",
    "Minnesota Vikings",

    # NFC South
    "Atlanta Falcons",
    "Carolina Panthers",
    "New Orleans Saints",
    "Tampa Bay Buccaneers",

    # NFC West
    "Arizona Cardinals",
    "Los Angeles Rams",
    "San Francisco 49ers",
    "Seattle Seahawks"
]

# Historical team names (for older data)
HISTORICAL_TEAMS = [
    "St. Louis Rams",  # Now Los Angeles Rams
    "San Diego Chargers",  # Now Los Angeles Chargers
    "Oakland Raiders",  # Now Las Vegas Raiders
    "Washington Redskins",  # Now Washington Commanders
    "Washington Football Team",  # Interim name, now Commanders
]

# Common abbreviations and alternate names
TEAM_ALIASES = {
    # AFC East
    "BUF": "Buffalo Bills",
    "Bills": "Buffalo Bills",
    "MIA": "Miami Dolphins",
    "Dolphins": "Miami Dolphins",
    "NE": "New England Patriots",
    "Patriots": "New England Patriots",
    "Pats": "New England Patriots",
    "NYJ": "New York Jets",
    "Jets": "New York Jets",

    # AFC North
    "BAL": "Baltimore Ravens",
    "Ravens": "Baltimore Ravens",
    "CIN": "Cincinnati Bengals",
    "Bengals": "Cincinnati Bengals",
    "CLE": "Cleveland Browns",
    "Browns": "Cleveland Browns",
    "PIT": "Pittsburgh Steelers",
    "Steelers": "Pittsburgh Steelers",

    # AFC South
    "HOU": "Houston Texans",
    "Texans": "Houston Texans",
    "IND": "Indianapolis Colts",
    "Colts": "Indianapolis Colts",
    "JAX": "Jacksonville Jaguars",
    "JAC": "Jacksonville Jaguars",
    "Jaguars": "Jacksonville Jaguars",
    "Jags": "Jacksonville Jaguars",
    "TEN": "Tennessee Titans",
    "Titans": "Tennessee Titans",

    # AFC West
    "DEN": "Denver Broncos",
    "Broncos": "Denver Broncos",
    "KC": "Kansas City Chiefs",
    "Chiefs": "Kansas City Chiefs",
    "LV": "Las Vegas Raiders",
    "Raiders": "Las Vegas Raiders",
    "LAC": "Los Angeles Chargers",
    "Chargers": "Los Angeles Chargers",

    # NFC East
    "DAL": "Dallas Cowboys",
    "Cowboys": "Dallas Cowboys",
    "NYG": "New York Giants",
    "Giants": "New York Giants",
    "PHI": "Philadelphia Eagles",
    "Eagles": "Philadelphia Eagles",
    "WAS": "Washington Commanders",
    "Commanders": "Washington Commanders",
    "Washington": "Washington Commanders",

    # NFC North
    "CHI": "Chicago Bears",
    "Bears": "Chicago Bears",
    "DET": "Detroit Lions",
    "Lions": "Detroit Lions",
    "GB": "Green Bay Packers",
    "Packers": "Green Bay Packers",
    "MIN": "Minnesota Vikings",
    "Vikings": "Minnesota Vikings",
    "Vikes": "Minnesota Vikings",

    # NFC South
    "ATL": "Atlanta Falcons",
    "Falcons": "Atlanta Falcons",
    "CAR": "Carolina Panthers",
    "Panthers": "Carolina Panthers",
    "NO": "New Orleans Saints",
    "Saints": "New Orleans Saints",
    "TB": "Tampa Bay Buccaneers",
    "Buccaneers": "Tampa Bay Buccaneers",
    "Bucs": "Tampa Bay Buccaneers",

    # NFC West
    "ARI": "Arizona Cardinals",
    "Cardinals": "Arizona Cardinals",
    "LAR": "Los Angeles Rams",
    "Rams": "Los Angeles Rams",
    "SF": "San Francisco 49ers",
    "49ers": "San Francisco 49ers",
    "Niners": "San Francisco 49ers",
    "SEA": "Seattle Seahawks",
    "Seahawks": "Seattle Seahawks",

    # Historical
    "STL": "St. Louis Rams",
    "SD": "San Diego Chargers",
    "OAK": "Oakland Raiders",
}

ALL_VALID_TEAMS = NFL_TEAMS + HISTORICAL_TEAMS


def normalize_team_name(team_name):
    """
    Normalize team name to official format

    Args:
        team_name: Input team name (can be abbreviation or partial)

    Returns:
        Official team name or None if not found
    """
    if not team_name:
        return None

    # Exact match
    if team_name in ALL_VALID_TEAMS:
        return team_name

    # Check aliases
    if team_name in TEAM_ALIASES:
        return TEAM_ALIASES[team_name]

    # Case-insensitive search
    team_lower = team_name.lower()

    # Check exact match (case-insensitive)
    for team in ALL_VALID_TEAMS:
        if team.lower() == team_lower:
            return team

    # Check if it's a partial match (contains)
    matches = []
    for team in ALL_VALID_TEAMS:
        if team_lower in team.lower() or team.lower() in team_lower:
            matches.append(team)

    if len(matches) == 1:
        return matches[0]
    elif len(matches) > 1:
        # Multiple matches, try to find best match
        for team in matches:
            if team.lower() == team_lower:
                return team
        # Return first match if no exact match
        return matches[0]

    return None


def get_all_team_names():
    """Get list of all current NFL teams"""
    return sorted(NFL_TEAMS)


def get_team_abbreviation(team_name):
    """
    Get common abbreviation for a team

    Args:
        team_name: Official team name

    Returns:
        Common abbreviation or None
    """
    for abbr, full_name in TEAM_ALIASES.items():
        if full_name == team_name and abbr.isupper() and len(abbr) <= 3:
            return abbr
    return None


def validate_team_name(team_name):
    """
    Validate if team name exists and return normalized version

    Args:
        team_name: Input team name

    Returns:
        Tuple of (is_valid, normalized_name, suggestions)
    """
    normalized = normalize_team_name(team_name)

    if normalized:
        return True, normalized, []

    # Provide suggestions using fuzzy matching
    suggestions = []
    if team_name:
        team_lower = team_name.lower()
        for team in NFL_TEAMS:
            # Simple similarity check
            team_words = team.lower().split()
            if any(word.startswith(team_lower[:3]) for word in team_words if len(team_lower) >= 3):
                suggestions.append(team)

    return False, None, suggestions[:5]


if __name__ == "__main__":
    print("NFL Teams Reference")
    print("=" * 70)
    print(f"\nTotal Current Teams: {len(NFL_TEAMS)}")
    print("\nAll Teams (by division):\n")

    divisions = {
        "AFC East": NFL_TEAMS[0:4],
        "AFC North": NFL_TEAMS[4:8],
        "AFC South": NFL_TEAMS[8:12],
        "AFC West": NFL_TEAMS[12:16],
        "NFC East": NFL_TEAMS[16:20],
        "NFC North": NFL_TEAMS[20:24],
        "NFC South": NFL_TEAMS[24:28],
        "NFC West": NFL_TEAMS[28:32],
    }

    for division, teams in divisions.items():
        print(f"{division}:")
        for team in teams:
            abbr = get_team_abbreviation(team)
            print(f"  - {team} ({abbr})")
        print()

    # Test validation
    print("\nValidation Examples:")
    test_names = ["KC", "Chiefs", "Kansas City", "Patriots", "49ers", "Packers", "Invalid Team"]
    for name in test_names:
        is_valid, normalized, suggestions = validate_team_name(name)
        if is_valid:
            print(f"  '{name}' -> ✓ {normalized}")
        else:
            print(f"  '{name}' -> ✗ Not found. Suggestions: {suggestions}")
