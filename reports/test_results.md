# Test Results: News Research Bot

- **Run on:** 2026-10-18 05:33
- **Chat model:** `gpt-5.4-mini` (temperature 0) · **Embeddings:** `text-embedding-3-small`
- **Index:** 3 articles, 23 chunks (chunk size 1000, overlap 200, top-k 4)
- **Time taken:** 40 s

## Articles used

- [TCS Q2 results: Net profit rises 15% to ₹13,884 crore, revenue jumps 11% YoY; TCV rises to $9.6 billion](https://upstox.com/news/market-news/earnings/tcs-q2-results-net-profit-rises-15-to-13-884-crore-revenue-jumps-11-yo-y-tcv-rises-to-9-6-billion/article-201534/)
- [HCLTech Net Profit Rises 21% To ₹5,132 Crore In Q2 FY27, Revenue Up 11.5%](https://www.freepressjournal.in/business/hcltech-net-profit-rises-21-to-5132-crore-in-q2-fy27-revenue-up-115)
- [Wipro Profit Falls Marginally to Rs 3,200.6 Cr in Q2; Revenue Grows 9% to Rs 24,752.7 Cr](https://hdfcsky.com/news/wipro-q2-fy27-results-profit-revenue)

## Summary

| Check | Passed |
|---|---|
| Factual questions (correct answer and source) | 5/5 ✅ |
| Questions not in the articles (says it doesn't know) | 2/2 ✅ |
| Follow-up question uses chat history | 1/1 ✅ |
| Key insights accurate (sentiment, ticker, figures, 3-5 sentence summary) | 3/3 ✅ |
| Agent picks the right tool | 4/4 ✅ |
| **Total** | **15/15** |

Offline unit tests (splitting, loader fallbacks, index build/save/load): run `python -m pytest -v`.

## 1. Factual questions

Pass = the answer contains the expected figure **and** the right article is cited.

| Question | Answer | Sources | Result |
|---|---|---|---|
| What was TCS's consolidated net profit in Q2 FY27? | TCS’s consolidated net profit in Q2 FY27 was ₹13,884 crore. | TCS (Upstox) | ✅ Pass |
| What was HCLTech's revenue for the September quarter? | HCLTech’s revenue for the September quarter was ₹35,620 crore, up 11.5% YoY. | HCLTech (Free Press Journal) | ✅ Pass |
| By how much did Wipro's net profit fall year on year? | Wipro's net profit fell by 1.40 per cent year on year, to Rs 3,200.6 crore from Rs 3,246.2 crore. | Wipro (HDFC Sky) | ✅ Pass |
| What was TCS's total contract value (TCV) for the quarter? | TCS’s total contract value (TCV) for the quarter was $9.6 billion. | TCS (Upstox) | ✅ Pass |
| What were HCLTech's new deal bookings in Q2? | HCLTech reported new deal bookings of $3,258 million (approximately ₹27,476 crore) in Q2. | HCLTech (Free Press Journal) | ✅ Pass |

## 2. Questions whose answers are not in the articles

Pass = the bot says it couldn't find the answer instead of guessing.

| Question | Answer | Result |
|---|---|---|
| What was Infosys's net profit in Q2 FY27? | I couldn't find that in the loaded articles. | ✅ Pass |
| What target price have analysts set for Reliance Industries shares? | I couldn't find that in the loaded articles. | ✅ Pass |

## 3. Follow-up question (chat history)

Pass = the follow-up is rewritten to name **Wipro** and the answer is correct.

| Step | Text |
|---|---|
| Question 1 | What was Wipro's revenue from operations in Q2 FY27? |
| Answer 1 | Wipro’s revenue from operations in Q2 FY27 was Rs 24,752.7 crore. |
| Follow-up | And how much did its net profit change? |
| Understood as | How much did Wipro’s net profit change? |
| Answer 2 | Wipro’s consolidated net profit fell by 1.40 per cent to Rs 3,200.6 crore in Q2 FY27, down from Rs 3,246.2 crore a year ago. |
| **Result** | ✅ Pass |

## 4. Key insights

Pass = sentiment as expected, the right ticker, every key figure appears in the article text (no invented numbers), and a 3–5 sentence summary.

| Article | Sentiment | Tickers | Key figures grounded | Result |
|---|---|---|---|---|
| TCS (Upstox) | Positive ✅ | TCS ✅ | 8/8 found in article ✅ | ✅ Pass |
| HCLTech (Free Press Journal) | Positive ✅ | HCLTECH ✅ | 8/8 found in article ✅ | ✅ Pass |
| Wipro (HDFC Sky) | Neutral ✅ | WIPRO ✅ | 9/9 found in article ✅ | ✅ Pass |

<details><summary>Full insights</summary>

**TCS (Upstox)**: TCS reported a strong Q2 with consolidated net profit rising 15% YoY to ₹13,884 crore and revenue increasing 11.2% to ₹73,188 crore. The company also announced a second interim dividend of ₹12 per share and said its total contract value rose to $9.6 billion for the quarter ended September 2026. Operating profit increased, but margins narrowed, with EBITDA margin down 162 basis points to 25.70%. Management highlighted broad-based growth, strong AI revenue momentum, and two new strategic deals with Porsche AG and Best Buy.

- Consolidated net profit: ₹13,884 crore (+15% YoY)
- Revenue from core operations: ₹73,188 crore (+11.2% YoY)
- Total contract value (TCV): $9.6 billion (surged in the quarter ended September 2026)
- EBITDA: ₹18,815 crore (+4.6% YoY)
- EBITDA margin: 25.70% (-162 basis points)
- Operating margin: 24%
- Net margin: 19%
- Second interim dividend: ₹12 per share
- *Sentiment reason:* The quarter showed double-digit profit and revenue growth, strong deal wins, and a dividend announcement, even though margins contracted.
- *Takeaway:* Investors may view TCS’s Q2 as a solid growth quarter supported by deal momentum and AI-related revenue, but margin pressure remains a point to watch. The ₹12 per share interim dividend adds a shareholder-return element to the update.

**HCLTech (Free Press Journal)**: HCLTech reported a strong second quarter ended 30 September 2026, with consolidated net income rising 21.2% YoY to ₹5,132 crore and revenue increasing 11.5% YoY to ₹35,620 crore. Sequentially, net income rose 11% and revenue increased 3.0%. EBIT came in at ₹6,465 crore with an 18.2% margin, while diluted EPS was ₹70.22, up 12.2% YoY. The company also reported new deal bookings of $3,258 million and declared a dividend of ₹12 per share.

- Consolidated net income: ₹5,132 crore (+21.2% YoY)
- Revenue: ₹35,620 crore (+11.5% YoY)
- Sequential net income change: ₹4,624 crore (+11% QoQ)
- Sequential revenue change: ₹34,579 crore (+3.0% QoQ)
- EBIT: ₹6,465 crore
- EBIT margin: 18.2%
- Diluted EPS: ₹70.22 (+12.2% YoY)
- New deal bookings: $3,258 million
- *Sentiment reason:* HCLTech delivered double-digit growth in profit, revenue, EPS, and bookings, while also expanding margins and announcing a dividend.
- *Takeaway:* The quarter indicates solid operating momentum, with strong deal wins and margin improvement supporting growth. Investors may view the results as evidence that HCLTech is benefiting from accelerating advanced AI revenue and improved capital efficiency.

**Wipro (HDFC Sky)**: Wipro reported a marginal decline in consolidated net profit for Q2 FY27, even as revenue rose 9.1% year over year. Net profit fell to Rs 3,200.6 crore from Rs 3,246.2 crore a year earlier, while revenue from operations increased to Rs 24,752.7 crore from Rs 22,697.3 crore. Sequentially, profit declined 4.5% and revenue grew 1%. Wipro said its IT services revenue was USD 2,611.5 million and guided Q3 FY27 IT services revenue in the range of USD 2,571 million to USD 2,625 million, implying constant-currency sequential growth of (-)1.5% to (+)0.5%.

- Consolidated net profit: Rs 3,200.6 crore (-1.40% YoY)
- Net profit: Rs 3,246.2 crore (year-ago period)
- Revenue from operations: Rs 24,752.7 crore (+9.1% YoY)
- Revenue from operations: Rs 22,697.3 crore (year-ago period)
- Sequential profit change: -4.5%
- Sequential revenue change: +1%
- IT services segment revenue: USD 2,611.5 million (-0.1% Q-o-Q, +0.3% YoY)
- Q3 FY27 IT services revenue guidance: USD 2,571 million to USD 2,625 million ((-)1.5% to (+)0.5% sequential constant currency)
- *Sentiment reason:* Profit declined slightly, but revenue grew solidly year over year and the company provided modest Q3 guidance.
- *Takeaway:* The quarter shows Wipro is still growing revenue, but profit momentum remains weak. Investors will likely focus on whether the company’s strategic investments and AI-led initiatives can support better earnings growth going forward.

</details>

## 5. Agent tool choice

Pass = the agent calls exactly the expected tool (and no others).

| Question | Expected tool | Tools used | Answer (start) | Result |
|---|---|---|---|---|
| What was HCLTech's EBIT margin in Q2? | search_articles | search_articles | HCLTech’s EBIT margin in Q2 was **18.2%**. | ✅ Pass |
| Give me the key insights for the Wipro article | extract_insights | extract_insights | Key insights for Wipro: - Wipro reported a marginal decline in consolidated net profit for Q2 FY27, even as revenue rose 9.1% year over year. - Net profit fell… | ✅ Pass |
| Compare the net profit growth of TCS, HCLTech and Wipro | compare_articles | compare_articles | Here’s a quick comparison of net profit growth: \| Company \| Net profit growth \| Net profit \| \|---\|---:\|---:\| \| TCS \| **15% YoY rise** \| **₹13,884 cr… | ✅ Pass |
| What is the sentiment of each article? | extract_insights ×3 | extract_insights, extract_insights, extract_insights | Here’s the sentiment of each article: - **TCS**: **Positive** - Double-digit profit and revenue growth, strong deal wins, and a dividend announcement. - Margin… | ✅ Pass |
