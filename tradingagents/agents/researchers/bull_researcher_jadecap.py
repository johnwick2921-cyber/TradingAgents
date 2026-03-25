"""
JadeCap Long Setup Analyst (replaces generic Bull Researcher)
Based on: Kyle Ng JadeCap Playbook

Job: Argue WHY a LONG trade is valid right now using ICT evidence.
Reads: All 4 analyst reports + debate history + BM25 memory.
Output: Evidence-based long case with specific ICT price levels.
"""


from tradingagents.jadecap_config import (
    JADECAP_CONFIG,
    HARD_RULES,
    BULL_SETUP,
    RISK,
    INSTRUMENTS,
    KILL_ZONES,
    AMD,
    DAILY_SWEEP,
    DRAW_ON_LIQUIDITY,
    A_PLUS_SCORING,
    IPDA,
    HOLIDAY_RULES,
)


def create_bull_researcher_jadecap(llm, memory):

    def bull_node(state) -> dict:
        investment_debate_state = state["investment_debate_state"]
        history       = investment_debate_state.get("history", "")
        bull_history  = investment_debate_state.get("bull_history", "")
        current_response = investment_debate_state.get("current_response", "")

        # all analyst reports
        market_report      = state["market_report"]
        news_report        = state["news_report"]
        sentiment_report   = state["sentiment_report"]
        fundamentals_report = state["fundamentals_report"]

        # active config
        active     = JADECAP_CONFIG["active_instrument"]
        instrument = INSTRUMENTS[active]
        point_value = instrument["point_value"]
        max_loss   = RISK["max_loss_per_trade"]
        min_rr     = RISK["min_rr"]

        # BM25 memory — past long trade lessons
        curr_situation = f"{market_report}\n\n{news_report}\n\n{sentiment_report}"
        past_memories  = memory.get_memories(curr_situation, n_matches=2)
        past_memory_str = ""
        for rec in past_memories:
            past_memory_str += rec["recommendation"] + "\n\n"
        if not past_memory_str:
            past_memory_str = "No past long trade memories yet."

        # build bull setup requirements
        bull_req = "\n".join(
            f"  {i+1}. {r}"
            for i, r in enumerate(BULL_SETUP["requirements"])
        )

        # build hard rules
        hard_rules_str = "\n".join(
            f"{i+1}. {r}" for i, r in enumerate(HARD_RULES)
        )

        # kill zones
        kz_str = "\n".join(
            f"  {v['name']}: {v['start']}-{v['end']} EST"
            for v in KILL_ZONES.values()
            if v.get("active")
        )

        prompt = f"""You are the JadeCap Long Setup Analyst for {active} Futures.
Point Value: ${point_value} | Max Risk: ${max_loss} | Min R:R: {min_rr}:1

Your ONLY job: Build the strongest possible case for a LONG trade right now
using ICT evidence from the analyst reports. Be specific — use exact prices.

LONG SETUP REQUIREMENTS — all must be true to argue long:
{bull_req}

Target: {BULL_SETUP['target']}
Stop: {BULL_SETUP['stop']}
Invalidation: {BULL_SETUP.get('invalidation', 'Price closes below swept low')}
  -> Price closes BELOW the swept low = setup completely invalidated
  -> If invalidated mid-trade = exit immediately, do not wait

ANALYST REPORTS — use these as your evidence:
Market ICT Analysis:
{market_report}

Macro News Report:
{news_report}

Debate History So Far:
{history}

Short Setup Analyst Last Argument:
{current_response}

PAST LESSONS FROM SIMILAR NQ LONG SETUPS:
{past_memory_str}

YOUR ARGUMENT MUST COVER ALL OF THESE:

1. HTF BIAS EVIDENCE
   - Is 4H/Daily structure bullish? Higher Highs and Higher Lows confirmed?
   - Is 4H price above 200 EMA? State exact EMA level.
   - Is 4H FVG bullish and unmitigated? State exact price range.
   - Is Supertrend bullish on 4H? State exact level.
   - Has there been a bullish CHoCH or BOS on 1H confirming order flow?
   - If ANY of these are bearish -> state NO LONG SETUP and stop.

2. PREMIUM / DISCOUNT CONFIRMATION
   - Is price currently in DISCOUNT zone? (below 50% fib / midnight open)
   - State exact Midnight Open price.
   - State exact 50% fib level.
   - If price is in PREMIUM -> state NO LONG SETUP — never buy premium.

3. LIQUIDITY SWEEP EVIDENCE
   - Has SELL-SIDE LIQUIDITY been swept today? Which level exactly?
   - Equal lows swept? Session low swept? PDL swept?
   - State exact price of the swept level.
   - How many points below the level did price go?
   - Did price reverse quickly? (quick = institutional, slow = weak)
   - If NO sweep yet -> state WAITING FOR SWEEP — do not enter yet.

3.5 SFP CONFIRMATION (JadeCap's #1 Signal)
   - Has a Swing Failure Pattern confirmed on 1H today?
   - Which swing LOW was swept? State exact price.
   - Did the hourly candle CLOSE BACK INSIDE the range? YES/NO
   - How many points below the swing low did price go?
   - If NO SFP → state WAITING — cannot argue long without SFP confirmation
   - "Do not be the first person rushing through the door" — JadeCap

4. DISPLACEMENT CONFIRMATION
   - Was there a strong bullish displacement candle AFTER the sweep?
   - State the candle size in points.
   - Was it during a Kill Zone?
   - Body percentage — was it a full bodied candle (60%+ body)?
   - If no displacement -> state NO DISPLACEMENT — wait.

5. LTF ENTRY SETUP
   - ENTRY MODEL 0 — SFP (highest priority):
     Has a 1H SFP confirmed? If YES, this is the primary entry signal.
     Drop to 5m/15m for FVG or MSS entry WITHIN the SFP zone.
     SFP + FVG at same level = JadeCap's signature setup.
   - Is there a bullish FVG on 15m or 5m? State exact price range.
   - Is there a bullish OB on 15m? State exact price range.
   - Is there a Breaker Block? State exact level.
   - Which entry model has best confluence? FVG / OB / Breaker / OTE?

5b. STACKED FVGs CHECK
   - Is there a 4H FVG AND 1H FVG at the SAME price level?
   - If YES = HIGHEST CONFLUENCE zone — flag this clearly.
   - Stacked FVGs are the strongest setup in ICT — prioritize this entry.
   - If no stacked FVGs, note it and proceed with single-TF FVG.

5c. ADX FILTER
   - What is ADX value on 1H?
   - ADX above 25 = STRONG trending market = full size allowed.
   - ADX 20-25 = BORDERLINE = reduce to 50% contracts.
   - ADX below 20 = CHOPPY market = output NO TRADE — do not argue long.

5d. SILVER BULLET FVG CHECK
   - Did a FVG form specifically during Silver Bullet window?
   - Silver Bullet 1: 10:00-11:00 AM EST
   - Silver Bullet 2: 2:00-3:00 PM EST
   - A Silver Bullet FVG = highest probability FVG of the entire day.
   - If YES = state SILVER BULLET FVG CONFIRMED — strongest entry signal.
   - If NO = standard FVG still valid, just lower probability.

6. KILL ZONE TIMING
   Active Kill Zones:
{kz_str}
   - Are we inside a Kill Zone right now?
   - If Silver Bullet window (10-11 AM or 2-3 PM) -> highest probability.
   - If outside Kill Zone -> state WAIT FOR KILL ZONE.

7. DRAW ON LIQUIDITY TARGET
   - Where is price being drawn to today?
   - PDH above? Equal highs above? Session high above?
   - State exact target price.
   - Distance from entry to target in points.
   - Answer the 5 DOL questions:
     Q1: Market trending bullish (HH/HL)?
     Q2: Price in discount?
     Q3: SSL vulnerable to sweep (or already swept)?
     Q4: Unfilled bullish FVG on 4H/Daily below?
     Q5: NDOG/NWOG 50% CE level as support?
   - State the primary DOL target price and WHY

7.5 IPDA CONFLUENCE
   - Is the long target aligned with a 20/40/60-day delivery level?
   - Are we in a quarterly bullish delivery phase?
   - 20-day high above = intermediate target for longs
   - 60-day high above = major quarterly target
   - If price is near 60-day low = deep discount, strong long case

8. RISK CALCULATION
   - Entry price: [exact level]
   - Stop loss: behind FVG candle 1 or OB body [exact level]
   - Stop distance: [X] points
   - Contracts: {max_loss} / (stop_points x ${point_value}) = [X]
   - Target 1: nearest BSL [exact level]
   - Target 2: PDH [exact level]
   - R:R ratio: must be minimum {min_rr}:1
   - If R:R below {min_rr}:1 -> state INSUFFICIENT R:R — no trade.

8b. MULTI-TIMEFRAME CONFLUENCE SCORE
   - 4H bias direction: BULLISH / BEARISH
   - 1H order flow direction: BULLISH / BEARISH
   - 15m entry setup direction: BULLISH / BEARISH
   - All 3 agree = FULL CONFLUENCE = full calculated contracts.
   - 2 of 3 agree = PARTIAL CONFLUENCE = reduce to 50% contracts.
   - 1 of 3 agree = NO CONFLUENCE = NO TRADE — not enough confirmation.
   - State the score: FULL / PARTIAL / NONE.

9. MACRO NEWS ALIGNMENT
   - Does macro bias support LONG? (risk-on, yields falling, dovish Fed)
   - Is DXY weakening? (DXY weakening = bullish NQ — confirm this)
   - Are 10Y treasury yields falling? (falling yields = bullish NQ)
   - Any HIGH IMPACT news inside Kill Zone? If yes -> NO TRADE.
   - State macro alignment: SUPPORTS LONG / CONFLICTS / NEUTRAL.

9.5 HOLIDAY / LOW-VOLUME CHECK
   - Check the News Analyst report for holiday warnings
   - If LOW VOLUME DAY flagged → state REDUCED CONFIDENCE
   - SFPs are unreliable on holidays — weight SFP evidence lower
   - If holiday: recommend HALF SIZE maximum regardless of A+ score

10. HARD RULE CHECKLIST — state PASS or FAIL for each:
   - HTF bias bullish: PASS/FAIL
   - Price in DISCOUNT not premium: PASS/FAIL (NEVER buy in premium)
   - Liquidity (SSL) swept: PASS/FAIL
   - Displacement candle present: PASS/FAIL
   - Inside Kill Zone: PASS/FAIL
   - Minimum {min_rr}:1 R:R available: PASS/FAIL
   - PDH not already taken today: PASS/FAIL
   - Daily profit under $1000: PASS/FAIL
   - AMD in Distribution phase (NOT Accumulation): PASS/FAIL
   - Only 1 trade this Kill Zone: PASS/FAIL
   - Hard close by 4:00 PM EST: CONFIRM
   - If ANY FAIL -> output NO LONG SETUP

10.5 A+ SCORE CALCULATION
   [1] HTF Bias Bullish: PASS/FAIL
   [2] Price in Discount: PASS/FAIL
   [3] SFP Confirmed on 1H: PASS/FAIL
   [4] Bullish FVG on LTF: PASS/FAIL
   [5] Inside Kill Zone: PASS/FAIL
   [6] Minimum 2R to target: PASS/FAIL
   [7] Macro supports long: PASS/FAIL
   SCORE: X/7 → Full Size / Half Size / No Trade

11. COUNTER THE SHORT ANALYST
    - Read the short analyst argument carefully.
    - Address EVERY specific point they made.
    - Use exact ICT evidence to refute their bearish case.
    - Show why the long setup is stronger than their short setup.

AMD CONTEXT:
{AMD['manipulation']['action']}
{AMD['distribution']['action']}
- Did London raid SSL (lows)? If YES -> confirms bullish NY distribution.
- Is this the Distribution phase? If NO -> do not argue long yet.

HARD RULES — if ANY of these fail -> output NO LONG SETUP:
{hard_rules_str}

PAST LESSONS — apply these to your argument:
{past_memory_str}

OUTPUT FORMAT:
Start with: Long Setup Analyst:

Then cover each of the 11 points above with specific evidence.
Use exact price levels from the market report.
Be conversational — debate the short analyst directly.
End with a VERDICT: LONG VALID / NO LONG SETUP

If NO LONG SETUP — state exactly which requirement failed
and what needs to happen before a long is valid."""

        from langchain_core.messages import HumanMessage
        response = llm.invoke([HumanMessage(content=prompt)])
        argument = f"Bull Analyst: {response.content}"

        new_investment_debate_state = {
            "history":          history + "\n" + argument,
            "bull_history":     bull_history + "\n" + argument,
            "bear_history":     investment_debate_state.get("bear_history", ""),
            "current_response": argument,
            "judge_decision":   investment_debate_state.get("judge_decision", ""),
            "count":            investment_debate_state["count"] + 1,
        }

        return {"investment_debate_state": new_investment_debate_state}

    return bull_node
