"""
Betfair Lay Agent - Main Entry Point
Natural language CLI for interacting with Betfair via Gemini 2.5 Flash.

Usage:
    python main.py

The agent will greet you and wait for instructions like:
    "Lay Black Caviar for $10"
    "Find markets for Winx"
    "Check my current bets"
    "Show me my P&L for the last 7 days"
    "Cancel all bets in market 1.150038686"
"""

import os
import sys
import json
import logging
from pathlib import Path

# Load .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # Manual .env loading if python-dotenv not installed
    env_path = Path(".env")
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    os.environ[key.strip()] = value.strip().strip('"').strip("'")

logging.basicConfig(
    level=logging.WARNING,  # Suppress verbose logs in CLI mode
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
# Enable INFO for our agent modules
logging.getLogger("gemini-agent").setLevel(logging.INFO)

# ─────────────────────────────────────────────────────────────────────────────
# STARTUP CHECKS
# ─────────────────────────────────────────────────────────────────────────────

def check_prerequisites() -> bool:
    """Check that all required files and env vars are present."""
    ok = True

    # Check GEMINI_API_KEY
    if not os.getenv("GEMINI_API_KEY"):
        print("❌ ERROR: GEMINI_API_KEY not found in .env file")
        print("   Create a .env file with: GEMINI_API_KEY=your_key_here")
        ok = False

    # Check credentials.json
    if not Path("credentials.json").exists():
        print("❌ ERROR: credentials.json not found")
        print("   Create credentials.json with:")
        print('   {"username": "your_betfair_username", "password": "your_password", "app_key": "your_app_key"}')
        ok = False
    else:
        try:
            with open("credentials.json") as f:
                creds = json.load(f)
            required = ["username", "password", "app_key"]
            missing = [k for k in required if k not in creds]
            if missing:
                print(f"❌ ERROR: credentials.json missing keys: {missing}")
                ok = False
            else:
                print(f"✅ credentials.json loaded for user: {creds['username']}")
        except json.JSONDecodeError:
            print("❌ ERROR: credentials.json is not valid JSON")
            ok = False

    return ok


def print_banner():
    """Print the startup banner."""
    print("\n" + "="*60)
    print("  🏇  BETFAIR LAY AGENT  🤖")
    print("  Powered by Gemini 2.5 Flash + Betfair API")
    print("  Australian Horse Racing | Lay & Back Betting")
    print("="*60)
    print()
    print("COMMANDS:")
    print("  Type naturally, e.g.:")
    print('  → "Lay <horse name> for $5"')
    print('  → "Check my balance"')
    print('  → "Show my open bets"')
    print('  → "What is my P&L this week?"')
    print('  → "Cancel all bets"')
    print()
    print("MARKET INTELLIGENCE:")
    print('  → "Analyse market 1.xxxxxxxxx"        - WOM + spread + lay score')
    print('  → "Watch market 1.xxxxxxxxx for steam" - 60s real-time price monitor')
    print('  → "When should I bet? Jump in 8 mins" - timing advice')
    print('  → "Is it a good time to lay X?"       - full pre-bet analysis')
    print()
    print("EXTERNAL MODEL RATINGS (NEW):")
    print('  → "Get Kash ratings for Doomben"       - Betfair\'s free thoroughbred model prices')
    print('  → "Get Iggy ratings for Sandown Park"  - Betfair\'s free greyhound model prices')
    print('  → "Model edge for [horse] in [market]" - compare model price vs live market')
    print()
    print("RACE CONTEXT (NEW):")
    print('  → "Race metadata for 1.xxxxxxxxx"      - jockey, barrier, distance, track condition')
    print('  → "Lay context score for [horse]"      - barrier/distance/track composite rating')
    print()
    print("ORCHESTRATION — MULTI-AGENT (NEW):")
    print('  → "What races are at Doomben today?"   - list all markets at venue')
    print('  → "Make money at Doomben today"        - launch full sub-agent session')
    print('     ↳ VenueAnalyst → RiskManager → Executor (Flumine) → Reporter')
    print('  → "Scan Flemington, what should I lay?" - analyst-only, no bets placed')
    print('  → "Session report"                     - live P&L for running session')
    print()
    print("META:")
    print('  → "memory"        - show what the agent remembers + betting performance')
    print('  → "performance"   - full strategy analytics by price/venue/WOM/timing')
    print('  → "cache"         - show API cache hit rate this session')
    print('  → "sync"          - pull latest settlement outcomes into memory')
    print('  → "reset"         - fresh conversation (keeps memory)')
    print('  → "reset hard"    - fresh conversation AND clears memory')
    print('  → "quit" or "exit" - exit the agent')
    print()
    print("-"*60)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # Check prerequisites
    if not check_prerequisites():
        print("\nPlease fix the above errors and try again.")
        sys.exit(1)

    api_key = os.getenv("GEMINI_API_KEY")

    # Import here to allow prerequisite check to fail gracefully
    try:
        from agent import BetfairGeminiAgent
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Run: pip install -r requirements.txt")
        sys.exit(1)

    print_banner()

    # Initialise the agent
    print("Initialising Gemini agent...")
    try:
        agent = BetfairGeminiAgent(
            api_key=api_key,
            credentials_path="credentials.json",
        )
        print("✅ Agent ready!\n")
    except Exception as e:
        print(f"❌ Failed to initialise agent: {e}")
        sys.exit(1)

    # Auto-login on startup
    print("🔐 Logging into Betfair...")
    try:
        login_result = agent.betfair.login()
        if login_result.get("success"):
            print("✅ Betfair login successful!\n")
            agent._logged_in = True
        else:
            print(f"⚠️  Betfair login issue: {login_result.get('message', login_result.get('error'))}")
            print("   You can still chat - the agent will try to login when needed.\n")
    except Exception as e:
        print(f"⚠️  Could not auto-login: {e}\n")

    print("Ready. What would you like to do?\n")

    # Main conversation loop
    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye! 🏇")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q", "bye"):
            print("Goodbye! 🏇")
            break

        if user_input.lower() == "reset":
            agent.reset_conversation(keep_memory=True)
            print("\U0001f504 Conversation reset. Memory preserved — I still remember previous horses/markets.\n")
            continue

        if user_input.lower() == "reset hard":
            agent.reset_conversation(keep_memory=False)
            print("\U0001f504 Conversation and memory fully cleared.\n")
            continue

        if user_input.lower() in ("memory", "mem"):
            print(f"\n{agent.show_memory()}\n")
            continue

        if user_input.lower() in ("performance", "perf", "stats"):
            print(f"\n{agent.show_performance()}\n")
            continue

        if user_input.lower() in ("cache",):
            cs = agent.cache.stats()
            print(f"\n⚡ Cache: {cs['entries']} entries | "
                  f"{cs['hits']} hits | {cs['misses']} misses | "
                  f"{cs['hit_rate']}% hit rate\n")
            continue

        if user_input.lower() in ("sync",):
            synced = agent.betting_memory.sync_outcomes()
            print(f"\n✅ Synced {synced} new settlement outcomes into BettingMemory.\n")
            continue

        if user_input.lower() in ("help", "?"):
            print_banner()
            continue

        # Process the message
        try:
            response = agent.chat_turn(user_input)
            print(f"\nAgent: {response}\n")
            print("-"*60)
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Try 'reset' to start a fresh conversation.\n")


if __name__ == "__main__":
    main()
