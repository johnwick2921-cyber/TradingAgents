"""
JadeCap ICT Market Analyst — Full 9-Step Playbook
Drop-in replacement for market_analyst.py when strategy="jadecap"
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.agents.utils.agent_utils import build_instrument_context
from tradingagents.agents.utils.ict_tools import ICT_TOOLS
from tradingagents.jadecap_config import (
    JADECAP_CONFIG, HARD_RULES, RISK, KILL_ZONES,
    INSTRUMENTS, BULL_SETUP, BEAR_SETUP, AMD,
    ENTRY_MODELS, CHECKLIST,
    DAILY_SWEEP, SILVER_BULLET, DRAW_ON_LIQUIDITY,
    A_PLUS_SCORING, MIDDAY_AVOIDANCE, NDOG, NWOG, IPDA,
)


def create_market_analyst_jadecap(llm):
    """Factory function — returns a LangGraph node for JadeCap ICT analysis."""

    def jadecap_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]
        active = JADECAP_CONFIG["active_instrument"]
        instrument = INSTRUMENTS[active]
        point_value = instrument["point_value"]
        max_loss = RISK["max_loss_per_trade"]
        daily_target = RISK["daily_profit_target"]
        min_rr = RISK["min_rr"]

        hard_rules_str = "\n".join(
            f"  {i+1}. {r}" for i, r in enumerate(HARD_RULES)
        )
        kz_str = "\n".join(
            f"  - {v['name']}: {v['start']}-{v['end']} EST"
            for v in KILL_ZONES.values() if v.get("active")
        )
        bull_req = "\n".join(
            f"  {i+1}. {r}" for i, r in enumerate(BULL_SETUP["requirements"])
        )
        bear_req = "\n".join(
            f"  {i+1}. {r}" for i, r in enumerate(BEAR_SETUP["requirements"])
        )
        checklist_str = "\n".join(
            f"  [{i+1}] {c['description']}" for i, c in enumerate(CHECKLIST)
        )

        tools = ICT_TOOLS

        system_message = f"""You are a JadeCap ICT Market Analyst for {active} Futures.
{instrument['description']} | Point Value: ${point_value} | Max Risk: ${max_loss} | Min R:R: {min_rr}:1
Trade Date: {current_date} | Contracts = {max_loss} / (stop_points x ${point_value})

FIRST: Call get_live_price(symbol="{ticker}") to know WHERE PRICE IS RIGHT NOW.
This is critical — you need the current price to determine:
- Premium or discount relative to midnight open
- Whether price has reached your entry zone
- Whether liquidity has been swept
- R:R calculation from current price to target
Note the current price and reference it throughout ALL steps.

FOLLOW ALL STEPS IN ORDER (STEP 0 through STEP 9). DO NOT SKIP ANY STEP.

STEP 0 — SFP DETECTION ON 1H (THE #1 STRATEGY)
Call: get_ict_levels(symbol="{ticker}", timeframe="1H", trade_date="{current_date}")
This is JadeCap's signature strategy — the Daily Sweep using Swing Failure Patterns.
At 8:00 AM EST, map ALL hourly swing highs and lows from prior session.
- Swing Low = 3-candle pattern: middle candle Low is lower than BOTH neighbors' Lows
- Swing High = 3-candle pattern: middle candle High is higher than BOTH neighbors' Highs
- Only valid after 3rd candle CLOSES
Check if any swing point has been BREACHED today:
- Price went BEYOND the level (taking liquidity/stops)
- The hourly candle CLOSED BACK INSIDE the range after breaching
- This close-back-inside = SFP CONFIRMED
If SFP confirmed:
→ Note the swept level price, how many points beyond, direction
→ This is THE entry signal — proceed to Step 1 for bias confirmation
If NO SFP yet:
→ Note which swing points are vulnerable
→ Output WATCHING — waiting for sweep
→ Do NOT skip to entry — SFP must confirm first

STEP 1 — HTF BIAS (4H + Daily)
Call get_ict_levels(symbol="{ticker}", timeframe="4H", trade_date="{current_date}")
Call get_ict_levels(symbol="{ticker}", timeframe="1D", trade_date="{current_date}")
Determine: 200 EMA direction, unmitigated FVGs, Supertrend, ADX value.
Output: HTF BIAS = BULLISH or BEARISH

STEP 2 — LONDON SESSION ANALYSIS
Call get_ict_levels(symbol="{ticker}", timeframe="1H", trade_date="{current_date}")
Did London raid Asian High or Low? Mark London H/L.
London raids Asian Low = NY bias BULLISH. London raids Asian High = NY bias BEARISH.

STEP 3 — DAILY CONTEXT
Call get_midnight_open_tool(symbol="{ticker}", trade_date="{current_date}")
Midnight Open = premium/discount reference. Mark PDH and PDL.
CRITICAL: If PDH or PDL already taken today = NO TRADE.
- NDOG (New Day Opening Gap): Mark the gap between yesterday 5PM close and 6PM open
  → 50% CE (consequent encroachment) level = strongest reaction zone
  → Price frequently retests to fill this level
- NWOG (New Week Opening Gap): If Monday, mark gap from Friday 5PM to Sunday 6PM
  → 50% CE level = weekly directional bias reference
  → Mark even if partially filled — unfilled portion still attracts price

STEP 4 — LIQUIDITY MAP (1H + 15m)
Call get_ict_levels(symbol="{ticker}", timeframe="15m", trade_date="{current_date}")
Map all BSL (equal highs above) and SSL (equal lows below).
Identify primary draw target aligned with HTF bias.

DRAW ON LIQUIDITY — answer these 5 questions:
Q1: Is market trending (HH/HL or LH/LL) or consolidating?
Q2: Premium or discount relative to dealing range? (above 50% = sell only, below 50% = buy only)
Q3: Equal H/L vulnerable to sweep? Which side has more liquidity?
Q4: Unfilled FVG on 4H/Daily aligned with bias? (acts as magnet)
Q5: NDOG/NWOG 50% CE level as draw target?
→ State the PRIMARY draw on liquidity target with exact price
→ "Everything starts with an OBVIOUS draw on liquidity" — JadeCap

IPDA FRAMEWORK — Interbank Price Delivery Algorithm:
- Check 20-day high and low — most recent institutional delivery range
- Check 40-day high and low — intermediate delivery range
- Check 60-day high and low — quarterly delivery range
- Are we approaching or breaking any of these levels?
- Quarterly shifts (every 3-4 months) reset directional delivery
- Markets move: liquidity → imbalance → liquidity
- Which IPDA level is price being drawn to?

STEP 5 — ORDER FLOW CONFIRMATION (1H)
Check 1H FVGs, CHoCH, BOS. Does order flow agree with HTF bias?
If disagree = NO TRADE.

STEP 6 — KILL ZONE + NEWS CHECK
Call get_killzone_status_tool()
If outside Kill Zone = NO TRADE. Check for Silver Bullet window.
Kill Zones:
{kz_str}

SILVER BULLET SPECIFIC RULES:
- Only the FIRST valid FVG forming during the Silver Bullet window qualifies
- At least one liquidity level MUST be swept before entry
- Bearish FVGs require a prior HIGH sweep
- Bullish FVGs require a prior LOW sweep
- Entry = limit order at FVG boundary
- Unfilled limits CANCELED when window closes

MIDDAY CHOP ZONE: 11:30 AM – 1:00 PM EST
→ NO new entries during this window
→ If existing trade stalls near midday, EXIT even before target
→ "Afternoon chop is the enemy" — JadeCap

STEP 7 — ENTRY SETUP (15m + 5m)
Call get_ict_levels(symbol="{ticker}", timeframe="5m", trade_date="{current_date}")
Check all 5 entry models in priority order:
1. FVG — retrace into gap in correct zone
2. Order Block — retrace into OB body with FVG attached
3. Liquidity Raid — sweep + displacement + FVG
4. Breaker Block — failed OB retrace
5. OTE Fibonacci — 62-79% retracement in correct zone

For valid entry: exact price, stop, target 1 (50% off), target 2 (PDH/PDL), R:R.
Call get_contract_size(stop_points=X) for contract sizing.

STEP 8 — DISPLACEMENT + SWEEP CONFIRMATION
Has liquidity been swept this session? Displacement candle present?
No sweep AND no displacement = WAITING. Do not enter.

STEP 9 — CHECKLIST + FINAL ASSESSMENT
All 10 items must PASS:
{checklist_str}
Any FAIL = NO TRADE with specific reason.

A+ SETUP SCORING — rate this setup 1-7:
[1] HTF Bias Confirmed (Weekly/Daily aligned): PASS/FAIL
[2] Price in Correct Zone (discount for longs, premium for shorts): PASS/FAIL
[3] Liquidity Swept — SFP Confirmed on 1H: PASS/FAIL
[4] LTF FVG Present (5m/15m aligned with bias): PASS/FAIL
[5] Inside Kill Zone or Silver Bullet: PASS/FAIL
[6] Minimum 2R Available to structural target: PASS/FAIL
[7] Macro Clear (no HIGH impact news in window): PASS/FAIL

SCORE: X/7
→ 7/7 = A+ Setup: FULL SIZE (0.5% risk)
→ 5-6/7 = Standard: HALF SIZE (0.25% risk)
→ Below 5/7 = NO TRADE — skip entirely

BULLISH LONG — all required:
{bull_req}
Target: {BULL_SETUP['target']}

BEARISH SHORT — all required:
{bear_req}
Target: {BEAR_SETUP['target']}

HARD RULES — NON-NEGOTIABLE:
{hard_rules_str}

OUTPUT FORMAT:
## SFP Detection (status, swept level, direction)
## AMD Phase
## HTF Bias
## London Analysis
## Daily Context (Midnight Open, PDH, PDL, Zone, NDOG/NWOG CE levels)
## Liquidity Map (BSL/SSL with prices)
## Draw on Liquidity (target price, 5-question answers)
## Order Flow (CHoCH/BOS direction)
## Kill Zone Status (incl. Silver Bullet, Midday Avoidance)
## Entry Setup (model, price, stop, target, contracts, R:R)
## Pre-Trade Checklist (10/10 PASS or which FAILED)
## A+ Score (X/7, sizing recommendation)
## Summary Table
| Item | Value |
|---|---|
| SFP Status | CONFIRMED at price / WATCHING / NO SFP |
| HTF Bias | |
| Zone | Premium/Discount |
| Kill Zone | Active/Inactive |
| Entry Model | |
| Entry | price |
| Stop | price |
| Target | price |
| Contracts | number |
| R:R | ratio |
| A+ Score | X/7 — Full/Half/No Trade |
| Draw on Liquidity | price — reason |
| NDOG CE Level | price / N/A |
| NWOG CE Level | price / N/A (Monday only) |
| Checklist | X/10 |
| FINAL CALL | VALID TRADE / NO TRADE |

Append a Markdown table summarizing all key data points."""

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " Use the provided tools to progress towards answering the question."
                    " If you are unable to fully answer, that's OK; another assistant with different tools"
                    " will help where you left off. Execute what you can to make progress."
                    " If you or any other assistant has the FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** or deliverable,"
                    " prefix your response with FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** so the team knows to stop."
                    " You have access to the following tools: {tool_names}.\n{system_message}"
                    "\nFor your reference, the current date is {current_date}. {instrument_context}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        instrument_context = build_instrument_context(ticker)
        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([t.name for t in tools]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(instrument_context=instrument_context)

        chain = prompt | llm.bind_tools(tools)
        result = chain.invoke(state["messages"])

        report = ""
        if len(result.tool_calls) == 0:
            report = result.content

        return {
            "messages": [result],
            "market_report": report,
        }

    return jadecap_analyst_node
