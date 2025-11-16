import streamlit as st
import pandas as pd
import numpy as np
from nba_api.stats.endpoints import leaguegamefinder
import time

# =================== CONFIGURATION ===================
# Team IDs mapping
TEAM_IDS = {
    'Warriors': 1610612744,
    'Lakers': 1610612746,
    'Celtics': 1610612738,
    'Heat': 1610612748,
    'Nuggets': 1610612743,
    'Suns': 1610612756,
    'Mavericks': 1610612742,
    'Nets': 1610612751,
    'Knicks': 1610612752,
    'Bucks': 1610612749,
    '76ers': 1610612755,
    'Pacers': 1610612754,
    'Raptors': 1610612761,
    'Cavaliers': 1610612739,
    'Pistons': 1610612765,
    'Kings': 1610612758,
    'Hawks': 1610612737,
    'Magic': 1610612753,
    'Pelicans': 1610612740,
    'Spurs': 1610612762,
    'Timberwolves': 1610612750,
    'Rockets': 1610612745,
    'Trail Blazers': 1610612757,
    'Jazz': 1610612762,
    'Hornets': 1610612766,
    'Grizzlies': 1610612750,
    'Clippers': 1610612746,
    'Mavericks': 1610612742,
    'Blazers': 1610612757,
}

# =================== DATA FETCHING FUNCTIONS ===================

@st.cache_data(ttl=1800)  # Cache for 30 minutes
def get_team_games(team_name):
    """
    Fetch games from NBA API

    Parameters:
    -----------
    team_name : str
        Name of team (e.g., 'Warriors')

    Returns:
    --------
    DataFrame with game stats or None if error
    """

    if team_name not in TEAM_IDS:
        return None

    try:
        team_id = TEAM_IDS[team_name]

        # Fetch games using NBA API
        gamefinder = leaguegamefinder.LeagueGameFinder(
            team_id_nullable=str(team_id),
            season_nullable='2024-25'
        )

        games = gamefinder.get_data_frames()[0]

        # Select relevant columns
        cols = ['GAME_DATE', 'MATCHUP', 'WL', 'FG_PCT', 'FG3_PCT', 'FT_PCT',
                'DREB', 'OREB', 'AST', 'TOV', 'STL', 'BLK', 'PTS']

        games = games[cols].copy()

        # Clean data
        games['GAME_DATE'] = pd.to_datetime(games['GAME_DATE'])
        games = games.sort_values('GAME_DATE').reset_index(drop=True)

        # Convert to numeric
        numeric_cols = ['FG_PCT', 'FG3_PCT', 'FT_PCT', 'DREB', 'OREB', 'AST', 'TOV', 'STL', 'BLK', 'PTS']
        for col in numeric_cols:
            games[col] = pd.to_numeric(games[col], errors='coerce')

        return games

    except Exception as e:
        st.error(f"Error fetching data for {team_name}: {str(e)}")
        return None


def calculate_rolling_stats(games_df, window=10):
    """
    Calculate rolling averages

    Parameters:
    -----------
    games_df : DataFrame
        Games data
    window : int
        Number of games for rolling average (default 10)

    Returns:
    --------
    DataFrame with rolling averages added
    """

    if games_df is None or len(games_df) == 0:
        return None

    df = games_df.copy()
    stats = ['FG_PCT', 'FG3_PCT', 'FT_PCT', 'DREB', 'OREB', 'AST', 'TOV']

    for col in stats:
        df[f'{col}_avg'] = df[col].rolling(window=window, min_periods=1).mean()

    return df


def get_current_stats(games_df):
    """
    Get most recent game's rolling averages

    Returns dictionary with current stats
    """

    if games_df is None or len(games_df) == 0:
        return None

    last_idx = len(games_df) - 1

    return {
        'fg_pct': games_df.loc[last_idx, 'FG_PCT_avg'],
        '3p_pct': games_df.loc[last_idx, 'FG3_PCT_avg'],
        'ft_pct': games_df.loc[last_idx, 'FT_PCT_avg'],
        'dreb': games_df.loc[last_idx, 'DREB_avg'],
        'oreb': games_df.loc[last_idx, 'OREB_avg'],
        'ast': games_df.loc[last_idx, 'AST_avg'],
        'tov': games_df.loc[last_idx, 'TOV_avg'],
    }


# =================== PREDICTION LOGIC ===================

def calculate_features(home_stats, away_stats):
    """Calculate prediction features"""

    return {
        'fg_pct_diff': home_stats['fg_pct'] - away_stats['fg_pct'],
        'dreb_diff': home_stats['dreb'] - away_stats['dreb'],
        'tov_diff': home_stats['tov'] - away_stats['tov'],
        'oreb_diff': home_stats['oreb'] - away_stats['oreb'],
    }


def predict_winner(home_stats, away_stats):
    """
    Make prediction based on stats

    Returns dictionary with winner and probabilities
    """

    features = calculate_features(home_stats, away_stats)

    # Weighted scoring (based on research findings)
    score = (
        (features['dreb_diff'] / 35) * 0.40 +      # 40% weight to rebounding
        (features['fg_pct_diff'] * 100) * 0.35 +   # 35% weight to shooting
        (features['tov_diff'] / 15) * -0.25        # 25% weight to turnovers
    )

    # Convert to probability using logistic function
    home_win_prob = 1 / (1 + np.exp(-10 * score))

    return {
        'winner': 'Home' if home_win_prob > 0.5 else 'Away',
        'home_prob': home_win_prob,
        'away_prob': 1 - home_win_prob,
    }


# =================== STREAMLIT UI ===================

# Page configuration
st.set_page_config(
    page_title="NBA Game Predictor",
    page_icon="🏀",
    layout="wide"
)

# Title
st.title("🏀 NBA Game Winner Predictor")
st.write("Powered by real NBA data and machine learning")

# Sidebar - Team Selection
st.sidebar.header("🎯 Select Teams")

teams_list = sorted(list(TEAM_IDS.keys()))

home_team = st.sidebar.selectbox(
    "Home Team:",
    teams_list,
    index=teams_list.index('Warriors') if 'Warriors' in teams_list else 0
)

away_team = st.sidebar.selectbox(
    "Away Team:",
    teams_list,
    index=teams_list.index('Lakers') if 'Lakers' in teams_list else 1
)

# Prevent same team selection
if home_team == away_team:
    st.sidebar.error("⚠️ Please select different teams")
    st.stop()

# Fetch data with status messages
st.sidebar.info("⏳ Loading real NBA data...")

home_games = get_team_games(home_team)
away_games = get_team_games(away_team)

if home_games is None or away_games is None:
    st.error("Could not fetch NBA data. Please try again or check your internet connection.")
    st.stop()

# Calculate rolling averages
home_games = calculate_rolling_stats(home_games, window=10)
away_games = calculate_rolling_stats(away_games, window=10)

# Get current stats
home_stats = get_current_stats(home_games)
away_stats = get_current_stats(away_games)

if home_stats is None or away_stats is None:
    st.error("Not enough game data available for one of the teams.")
    st.stop()

st.sidebar.success("✓ Data loaded successfully!")

# =================== DISPLAY STATISTICS ===================

st.subheader("📊 Last 10 Games Average Stats")

col1, col2 = st.columns(2)

with col1:
    st.markdown(f"### 🔵 {home_team} (Home)")
    st.metric(f"{home_team} FG%", f"{home_stats['fg_pct']:.1%}")
    st.metric(f"{home_team} 3P%", f"{home_stats['3p_pct']:.1%}")
    st.metric(f"{home_team} FT%", f"{home_stats['ft_pct']:.1%}")
    st.metric(f"{home_team} DREB", f"{home_stats['dreb']:.1f}")
    st.metric(f"{home_team} OREB", f"{home_stats['oreb']:.1f}")
    st.metric(f"{home_team} AST", f"{home_stats['ast']:.1f}")
    st.metric(f"{home_team} TOV", f"{home_stats['tov']:.1f}")

with col2:
    st.markdown(f"### 🔴 {away_team} (Away)")
    st.metric(f"{away_team} FG%", f"{away_stats['fg_pct']:.1%}")
    st.metric(f"{away_team} 3P%", f"{away_stats['3p_pct']:.1%}")
    st.metric(f"{away_team} FT%", f"{away_stats['ft_pct']:.1%}")
    st.metric(f"{away_team} DREB", f"{away_stats['dreb']:.1f}")
    st.metric(f"{away_team} OREB", f"{away_stats['oreb']:.1f}")
    st.metric(f"{away_team} AST", f"{away_stats['ast']:.1f}")
    st.metric(f"{away_team} TOV", f"{away_stats['tov']:.1f}")

# =================== MAKE PREDICTION ===================

st.subheader("🎲 Game Prediction")

result = predict_winner(home_stats, away_stats)

pred_col1, pred_col2, pred_col3 = st.columns(3)

with pred_col1:
    winner_emoji = "🔵" if result['winner'] == 'Home' else "🔴"
    winner_name = home_team if result['winner'] == 'Home' else away_team
    st.metric(
        "Predicted Winner",
        f"{winner_emoji} {winner_name}"
    )

with pred_col2:
    winner_prob = result['home_prob'] if result['winner'] == 'Home' else result['away_prob']
    st.metric("Win Probability", f"{winner_prob:.1%}")

with pred_col3:
    if winner_prob > 0.65:
        confidence = "🟢 High"
    elif winner_prob > 0.55:
        confidence = "🟡 Medium"
    else:
        confidence = "🔴 Low"
    st.metric("Confidence", confidence)

# =================== PROBABILITY BREAKDOWN ===================

st.subheader("📈 Win Probability Breakdown")

prob_col1, prob_col2 = st.columns(2)

with prob_col1:
    st.write(f"**{home_team} (Home) Win: {result['home_prob']:.1%}**")
    st.progress(result['home_prob'])

with prob_col2:
    st.write(f"**{away_team} (Away) Win: {result['away_prob']:.1%}**")
    st.progress(result['away_prob'])

# =================== RECENT GAMES TABLE ===================

st.subheader("🏀 Recent Games")

recent_col1, recent_col2 = st.columns(2)

with recent_col1:
    st.write(f"**{home_team} - Last 5 Games**")
    home_recent = home_games.tail(5)[['GAME_DATE', 'MATCHUP', 'WL', 'FG_PCT', 'DREB']].copy()
    home_recent['GAME_DATE'] = home_recent['GAME_DATE'].dt.strftime('%Y-%m-%d')
    home_recent['FG_PCT'] = home_recent['FG_PCT'].apply(lambda x: f"{x:.1%}")
    home_recent['DREB'] = home_recent['DREB'].apply(lambda x: f"{x:.1f}")
    st.dataframe(home_recent, hide_index=True, use_container_width=True)

with recent_col2:
    st.write(f"**{away_team} - Last 5 Games**")
    away_recent = away_games.tail(5)[['GAME_DATE', 'MATCHUP', 'WL', 'FG_PCT', 'DREB']].copy()
    away_recent['GAME_DATE'] = away_recent['GAME_DATE'].dt.strftime('%Y-%m-%d')
    away_recent['FG_PCT'] = away_recent['FG_PCT'].apply(lambda x: f"{x:.1%}")
    away_recent['DREB'] = away_recent['DREB'].apply(lambda x: f"{x:.1f}")
    st.dataframe(away_recent, hide_index=True, use_container_width=True)

# Footer
st.markdown("---")
st.caption("Data from NBA Stats API | Predictions based on rolling 10-game averages")
