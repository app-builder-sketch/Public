import streamlit as st
import requests
import pandas as pd
import time

# TheSportsDB Setup
# Free basic key (works for most endpoints, but limited rate). Get your own free/patreon key for better access.
API_KEY = "1"  # Basic free key (public). Replace with your Patreon key for v2 livescores and higher limits.

# League ID for Scottish Premiership
LEAGUE_ID = "4330"

# Function to fetch current standings
def get_standings():
    url = f"https://www.thesportsdb.com/api/v1/json/{API_KEY}/lookuptable.php?l={LEAGUE_ID}&s=2025-2026"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if 'table' in data:
            table = data['table']
            df = pd.DataFrame(table)[['intRank', 'strTeam', 'intPlayed', 'intWin', 'intDraw', 'intLoss', 'intGoalsFor', 'intGoalsAgainst', 'intGoalDifference', 'intPoints']]
            df.columns = ['Rank', 'Team', 'Played', 'Wins', 'Draws', 'Losses', 'GF', 'GA', 'GD', 'Points']
            df = df.set_index('Rank')
            return df
    return pd.DataFrame()

# Function to fetch all teams
def get_teams():
    url = f"https://www.thesportsdb.com/api/v1/json/{API_KEY}/search_all_teams.php?l=Scottish%20Premiership"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if 'teams' in data:
            return sorted(data['teams'], key=lambda x: x['strTeam'])
    return []

# Function to fetch squad for a team
def get_squad(team_id):
    url = f"https://www.thesportsdb.com/api/v1/json/{API_KEY}/lookup_all_players.php?id={team_id}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if 'players' in data:
            players = data['players']
            df = pd.DataFrame(players)
            if not df.empty:
                cols = ['strPlayer', 'strPosition', 'strNumber', 'dateBorn', 'strHeight', 'strWeight', 'strDescriptionEN']
                df = df[[c for c in cols if c in df.columns]]
                df.columns = ['Player', 'Position', 'Number', 'Born', 'Height', 'Weight', 'Description']
                return df
    return pd.DataFrame()

# NEW: Function to fetch live matches (today's ongoing or recent matches in the league)
def get_live_matches():
    # First, get today's events for the league
    today = time.strftime("%Y-%m-%d")  # Current date
    url = f"https://www.thesportsdb.com/api/v1/json/{API_KEY}/eventsday.php?d={today}&l={LEAGUE_ID}"
    response = requests.get(url)
    live_matches = []
    if response.status_code == 200:
        data = response.json()
        if 'events' in data:
            for event in data['events']:
                if event.get('strSport') == 'Soccer':
                    match = {
                        'Home': event.get('strHomeTeam'),
                        'Away': event.get('strAwayTeam'),
                        'Score': f"{event.get('intHomeScore', '-')}-{event.get('intAwayScore', '-')}",
                        'Status': event.get('strStatus', event.get('strTime', 'Scheduled')),
                        'Time': event.get('strTimeLocal', event.get('strTime', '')),
                    }
                    live_matches.append(match)
    return pd.DataFrame(live_matches) if live_matches else pd.DataFrame()

# Main App
st.set_page_config(page_title="Scottish Premiership Live Dashboard", layout="wide", initial_sidebar_state="collapsed")

st.title("⚽ Live Scottish Premiership Dashboard")

# Refresh Button
if st.button("🔄 Refresh All Data"):
    with st.spinner("Fetching latest data..."):
        st.session_state.standings = get_standings()
        st.session_state.teams = get_teams()
        st.session_state.live_matches = get_live_matches()
        if 'squad_cache' in st.session_state:
            st.session_state.squad_cache.clear()
    st.success("Data refreshed!")
    st.rerun()

# Initialize session state
if 'standings' not in st.session_state:
    st.session_state.standings = get_standings()
if 'teams' not in st.session_state:
    st.session_state.teams = get_teams()
if 'live_matches' not in st.session_state:
    st.session_state.live_matches = get_live_matches()
if 'squad_cache' not in st.session_state:
    st.session_state.squad_cache = {}

# NEW: Live Matches Section
st.subheader("🔴 Live / Today's Matches")
live_df = st.session_state.live_matches
if not live_df.empty:
    live_df = live_df[['Home', 'Score', 'Away', 'Status', 'Time']]
    st.dataframe(live_df, use_container_width=True, hide_index=True)
else:
    st.info("No matches today or no live data available at the moment.")

# Current Standings
st.subheader("📊 Current League Standings")
if not st.session_state.standings.empty:
    st.dataframe(st.session_state.standings, use_container_width=True)
else:
    st.warning("Unable to fetch standings. Try refreshing.")

# Teams Tabs
st.subheader("👥 Teams & Squads")
if st.session_state.teams:
    team_names = [team['strTeam'] for team in st.session_state.teams]
    tabs = st.tabs(team_names)

    for i, tab in enumerate(tabs):
        with tab:
            team = st.session_state.teams[i]
            team_id = team['idTeam']
            team_name = team['strTeam']

            col1, col2 = st.columns([1, 4])
            with col1:
                if team.get('strTeamBadge'):
                    st.image(team['strTeamBadge'], width=120)
                if team.get('strTeamJersey'):
                    st.image(team['strTeamJersey'], caption="Kit", width=120)

            with col2:
                st.markdown(f"### {team_name}")
                st.write(f"**Founded:** {team.get('intFormedYear', 'N/A')}")
                st.write(f"**Stadium:** {team.get('strStadium', 'N/A')} ({team.get('intStadiumCapacity', 'N/A')} capacity)")
                if team.get('strWebsite'):
                    st.write(f"**Website:** [{team['strWebsite']}](https://{team['strWebsite']})")

            st.write("**Team Description:**")
            desc = team.get('strDescriptionEN', 'No description available.')
            st.write(desc[:800] + ("..." if len(desc) > 800 else ""))

            # Squad
            st.markdown("#### 🏃‍♂️ Current Squad")
            if team_id not in st.session_state.squad_cache:
                with st.spinner(f"Loading squad for {team_name}..."):
                    squad_df = get_squad(team_id)
                    st.session_state.squad_cache[team_id] = squad_df
                    time.sleep(0.5)  # Rate limit safety
            else:
                squad_df = st.session_state.squad_cache[team_id]

            if not squad_df.empty:
                position_order = {'Goalkeeper': 0, 'Defender': 1, 'Midfielder': 2, 'Forward': 3, 'Attacker': 3}
                if 'Position' in squad_df.columns:
                    squad_df['pos_order'] = squad_df['Position'].map(position_order).fillna(4)
                    squad_df = squad_df.sort_values(['pos_order', 'Player']).drop('pos_order', axis=1)
                st.dataframe(squad_df, use_container_width=True, hide_index=True)
            else:
                st.info("No squad data available.")
else:
    st.warning("Unable to fetch teams data.")

# Footer
st.caption("Data from TheSportsDB (free API). Live matches show today's fixtures with scores if ongoing/finished.")
st.caption("For more frequent live updates, get a Patreon key from TheSportsDB. Auto-refresh every 5 minutes.")

# Auto-refresh
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = time.time()

if time.time() - st.session_state.last_refresh > 300:  # 5 minutes
    st.session_state.last_refresh = time.time()
    st.rerun()
