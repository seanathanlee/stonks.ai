# stonks.ai — Agentic DevOps Sample

An end-to-end sample that shows how to build, containerize, and deploy a **multi-agent AI system** to **Azure** using GitHub Actions.

The system scrapes NASDAQ stock prices daily, stores them in **Azure Data Explorer (Kusto)**, and runs **17 agents** — 9 LLM-based child agents (each with a distinct investment philosophy) and 8 pure-statistical model agents — through a **parent orchestrator agent** to produce ranked stock picks across four time horizons. All forecasts are persisted back to ADX for querying and analysis. An **evaluation agent** then measures how accurately each agent's 1-month forecasts matched actual stock returns, feeding accuracy scores back into ADX and surfacing them in the chat UI.

---

## Repository layout

```
stonks.ai/
├── agents/
│   ├── agent.py              # Original single AI agent (Azure OpenAI + tool-calling)
│   ├── adx_client.py         # ADX query + ingestion wrapper
│   ├── chat_agent.py         # Conversational chat agent (used by the web UI)
│   ├── evaluation_agent.py   # Evaluates 1-month forecast accuracy vs actual returns
│   ├── scraper.py            # NASDAQ symbol list + price fetcher
│   ├── child_agents.py       # 9 LLM child agent definitions + shared runner
│   ├── stat_agents.py        # 8 pure-statistical agent definitions + shared runner
│   └── parent_agent.py       # Orchestrator: reads ADX, fans out to all 17 agents, writes forecasts
├── api/
│   └── main.py               # FastAPI web application (serves chat UI + REST API)
├── frontend/
│   └── index.html            # Browser chatbot UI (Chart.js, agent roster panel, no build step)
├── infra/
│   ├── acr-only.bicep        # Standalone ACR template (provisioned first by deploy.yml)
│   ├── main.bicep            # Root Bicep template (AI Services, ADX, ACR, Container App, Static Web App)
│   └── main.bicepparam       # Default parameter values
├── .github/
│   └── workflows/
│       ├── deploy.yml                # CI/CD pipeline (ACR → Docker build → infra → frontend deploy)
│       ├── daily-scrape.yml          # Scheduled daily price ingestion (01:00 UTC)
│       ├── snapshot-scrape.yml       # Manual price backfill trigger (configurable date range)
│       ├── snapshot-forecast.yml     # Manual forecast backfill trigger (configurable date range)
│       ├── daily-evaluation.yml      # Scheduled daily forecast evaluation (03:00 UTC)
│       ├── snapshot-evaluation.yml   # Manual evaluation backfill trigger (configurable date range)
│       └── agent-rebalance-trade.yml # Manual top-picks rebalance executor
├── tests/
│   └── e2e/                  # Playwright end-to-end tests (pytest-playwright)
├── Dockerfile
├── requirements.txt
├── requirements-test.txt
├── .env.example
└── README.md
```

---

## Architecture

```
GitHub Actions
    │
    ├─► deploy.yml  (on push to main)
    │       ├─► Job 1: Provision Azure Container Registry (ACR)
    │       ├─► Job 2: Build Docker image → push to ACR
    │       ├─► Job 3: Deploy Azure infrastructure (Bicep)
    │       │       └─► Azure AI Services (Azure OpenAI GPT-4o)
    │       │       └─► Azure Data Explorer cluster (stonksaiadx)
    │       │               ├── database: stonksai
    │       │               │       ├── table:             dailyStockPrice
    │       │               │       ├── materialized-view: dailyStockPriceMV
    │       │               │       ├── table:             agentStockForecast
    │       │               │       ├── materialized-view: agentStockForecastMV
    │       │               │       └── table:             agentStockEvaluation
    │       │       └─► Azure Container Registry (stonksaiacr)
    │       │       └─► Azure Container App (stonksai-api)
    │       │       └─► Azure Static Web App  (skewthis.com)
    │       └─► Job 4: Inject API URL into frontend → deploy to Static Web App
    │
    ├─► daily-scrape.yml  (runs 01:00 UTC every day)
    │       └─► agents/scraper.py --mode daily
    │               └─► Yahoo Finance API  ──► ADX dailyStockPrice
    │
    ├─► snapshot-scrape.yml  (manual trigger, configurable date range)
    │       └─► agents/scraper.py --mode snapshot
    │               └─► Yahoo Finance API  ──► ADX dailyStockPrice
    │
    ├─► snapshot-forecast.yml  (manual trigger, configurable date range)
    │       └─► agents/parent_agent.py --date <date>  (matrix job, one per day)
    │               └─► ADX dailyStockPrice ──► 17 agents ──► ADX agentStockForecast
    │
    ├─► agents/parent_agent.py  (run on-demand or via snapshot-forecast.yml)
    │       ├─► ADX dailyStockPriceMV  ──► stock data (30-day window)
    │       ├─► 9 LLM child agents (concurrent)
    │       │       ├── momentum_trader
    │       │       ├── mean_reversion
    │       │       ├── value_investor
    │       │       ├── quality_investor
    │       │       ├── low_volatility
    │       │       ├── growth_investor
    │       │       ├── size_premium
    │       │       ├── dividend_income
    │       │       └── contrarian_investor
    │       ├─► 8 statistical agents (concurrent)
    │       │       ├── momentum_factor        (composite price momentum)
    │       │       ├── historical_volatility  (drift/volatility Sharpe proxy)
    │       │       ├── ets                    (Holt-Winters exponential smoothing)
    │       │       ├── arima                  (ARIMA(p,1,q) auto-selected by AIC)
    │       │       ├── garch_volatility       (GARCH(1,1) volatility-adjusted drift)
    │       │       ├── monte_carlo            (Geometric Brownian Motion, N=1,000 paths)
    │       │       ├── capm_beta              (CAPM beta vs equal-weighted proxy)
    │       │       └── hmm_regime             (2-state Hidden Markov Model)
    │       └─► ADX agentStockForecast  (5 picks × 4 horizons × 17 agents)
    │
    ├─► daily-evaluation.yml  (runs 03:00 UTC every day)
    │       └─► agents/evaluation_agent.py
    │               ├─► ADX agentStockForecast  ──► 1-month forecasts from 30 days ago
    │               ├─► ADX dailyStockPriceMV   ──► actual prices over same window
    │               └─► ADX agentStockEvaluation ◄── accuracy scores written back
    │
    └─► snapshot-evaluation.yml  (manual trigger, configurable date range)
            └─► agents/evaluation_agent.py --date <date>  (matrix job, one per day)

Browser  ──► api/main.py (FastAPI, rate-limited)
    │               ├── POST /api/chat   ──► agents/chat_agent.py
    │               │       ├── get_latest_forecasts    ──► ADX agentStockForecastMV
    │               │       ├── get_price_history       ──► ADX dailyStockPriceMV
    │               │       ├── compare_agent_forecasts ──► ADX agentStockForecastMV
    │               │       └── get_agent_evaluations   ──► ADX agentStockEvaluation
    │               ├── GET /api/health  ──► health check
    │               └── GET /            ──► frontend/index.html
    └── Chart.js charts + collapsible Agent Roster panel rendered inline in chat
```

---

## Prerequisites

| Tool | Version |
|------|---------|
| Python | 3.12+ |
| Docker | 24+ |
| Azure CLI | 2.60+ |
| Bicep CLI | installed via `az bicep install` |

---

## Local development

1. **Clone the repository and create a virtual environment**

   ```bash
   git clone https://github.com/seanathanlee/stonks.ai.git
   cd stonks.ai
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Configure environment variables**

   ```bash
   cp .env.example .env
   # Edit .env and fill in your Azure OpenAI and ADX values
   ```

3. **Start the web chat interface**

   ```bash
   uvicorn api.main:app --reload
   # Open http://localhost:8000 in your browser
   ```

   The chatbot lets you:
   - Run the full 17-agent analysis pipeline
   - View latest forecasts with inline charts
   - Explore price history for any symbol
   - Compare how different agents rate a specific stock
   - View the collapsible **Agent Roster** panel with descriptions of all 17 agents
   - Check **agent accuracy scores** from the evaluation pipeline

4. **Run the original single agent (CLI)**

   ```bash
   python -m agents.agent "What is the current price of AAPL and its 5-day moving average?"
   ```

5. **Backfill 30 days of price data (first-time setup)**

   ```bash
   python -m agents.scraper --mode snapshot
   ```

6. **Run the multi-agent orchestrator (CLI)**

   ```bash
   python -m agents.parent_agent

   # Optionally pass a historical date to backfill forecasts for that day
   python -m agents.parent_agent --date 2025-01-15
   ```

7. **Run the evaluation agent (CLI)**

   Requires at least 30 days of forecast + price data in ADX.

   ```bash
   python -m agents.evaluation_agent
   ```

8. **Run end-to-end tests**

   ```bash
   pip install -r requirements-test.txt
   playwright install chromium
   python -m pytest tests/e2e/ -v --browser chromium
   ```

---

## Deploy to Azure

### 1 — Create Azure resources (one-time)

```bash
# Log in
az login

# Create a resource group
az group create --name rg-stonksai --location eastus

# Deploy the Bicep template
# Provisions: AI Services, ADX cluster + DB + tables + views,
#             Azure Container Registry, Container App, Static Web App
az deployment group create \
  --resource-group rg-stonksai \
  --template-file infra/main.bicep \
  --parameters infra/main.bicepparam
```

### 2 — Configure GitHub Actions (one-time)

Set up **OIDC federated credentials** so the workflow can authenticate to Azure without long-lived secrets:

```bash
az ad sp create-for-rbac \
  --name sp-stonksai-gh \
  --role contributor \
  --scopes /subscriptions/<SUBSCRIPTION_ID>/resourceGroups/rg-stonksai \
  --json-auth
```

Then add the following **repository secrets** in GitHub → Settings → Secrets:

| Secret | Description |
|--------|-------------|
| `AZURE_CLIENT_ID` | Service principal client ID |
| `AZURE_TENANT_ID` | Azure tenant ID |
| `AZURE_SUBSCRIPTION_ID` | Azure subscription ID |
| `AZURE_RESOURCE_GROUP` | `rg-stonksai` |
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI endpoint, e.g. `https://<resource>.openai.azure.com/` |
| `AZURE_OPENAI_API_KEY` | Azure OpenAI API key |
| `ADX_CLUSTER_URI` | ADX cluster URI, e.g. `https://stonksaiadx.eastus.kusto.windows.net` |
| `ADX_DATABASE` | ADX database name (`stonksai`) |

And optionally add these **repository variables**:

| Variable | Default | Description |
|----------|---------|-------------|
| `BASE_NAME` | `stonksai` | Prefix for all Azure resource names |
| `AZURE_OPENAI_DEPLOYMENT` | `gpt-4o` | Azure OpenAI model deployment name |
| `SWA_LOCATION` | `centralus` | Azure region for the Static Web App |

### 3 — Seed historical data (one-time)

After the infrastructure is deployed, trigger the snapshot workflow to backfill 30 days of price data:

GitHub → Actions → **Snapshot Stock Price Scrape** → **Run workflow**

Then backfill forecasts for the same window:

GitHub → Actions → **Snapshot Agent Forecast** → **Run workflow**

And optionally backfill evaluations (requires ≥30 days of forecast + price data):

GitHub → Actions → **Snapshot Agent Evaluation** → **Run workflow**

### 4 — Automated daily workflows

| Workflow | Schedule | What it does |
|----------|----------|--------------|
| `daily-scrape.yml` | 01:00 UTC | Fetches previous day's NASDAQ closing prices → ADX |
| `daily-evaluation.yml` | 03:00 UTC | Evaluates 1-month forecast accuracy → ADX |

All workflows can also be triggered manually from the GitHub Actions tab. The following workflows are **manual-only** and accept a configurable date range for backfilling:

| Workflow | What it does |
|----------|--------------|
| `snapshot-scrape.yml` | Backfills NASDAQ prices for a date range → ADX |
| `snapshot-forecast.yml` | Runs the 17-agent pipeline for each day in a date range → ADX |
| `snapshot-evaluation.yml` | Evaluates forecast accuracy for each day in a date range → ADX |
| `agent-rebalance-trade.yml` | Runs a top-5 rebalance: liquidate holdings, then buy equal-weight picks |

### 5 — Full CI/CD on push to main

Pushing to `main` triggers `deploy.yml`, which runs four sequential jobs:

1. **Provision ACR** — creates the Azure Container Registry if it doesn't exist yet.
2. **Build and push backend** — builds the Docker image and pushes it to ACR.
3. **Deploy Azure infrastructure** — applies the full Bicep template (idempotent), passing the real image URI so no placeholder image is ever deployed.
4. **Deploy frontend** — injects the Container App URL into `frontend/index.html` and deploys it to the Azure Static Web App (served at skewthis.com).

### 6 — Agentic Trading (Robinhood MCP)

Setting up a Robinhood Agentic Trading account lets you automate trading through a connected third-party AI agent via the Robinhood Trading MCP.

#### What is MCP?
Model Context Protocol (MCP) is an open standard that lets AI agents connect to external apps and services and take actions on your behalf.

#### What your agent can do
Your agent can access account context (portfolio value, buying power, balances, transactions) and help place orders with supported order types in your Agentic account.

Example workflows include:
- Build portfolios from market/news context.
- Automate strategy triggers.
- Rebalance allocations.
- Analyze portfolio risks.
- Analyze market moves and bullish/bearish theses.

> These examples are informational only and are not recommendations.

#### Risks
You are responsible for all trades your AI agent places. If you enable unattended actions, your agent can place trades without per-order confirmation. Monitor activity and review disclosures.

#### What your agent can access
When connected to Robinhood Trading MCP, your agent has read access to:
- All Robinhood accounts (including account numbers)
- Position and balance details
- Transaction and order history

> Your agent can place trades only in your Robinhood Agentic account.

#### Connect your AI agent
Agentic Trading is rolling out and may not yet be available to all users.

Robinhood Trading MCP endpoint:
- `https://agent.robinhood.com/mcp/trading`

Supported platform setup examples:
- **Claude Code**: `claude mcp add robinhood-trading --transport http https://agent.robinhood.com/mcp/trading`, then `/mcp` and authenticate
- **Claude Desktop**: Settings → Connectors → Add custom connector
- **ChatGPT**: Developer Mode → Settings → Apps → Create app
- **Codex**: Settings → MCP servers → Streamable HTTP
- **Codex CLI**: `codex mcp add robinhood-trading --url https://agent.robinhood.com/mcp/trading`, then `/mcp`
- **Cursor**: Settings → Cursor Settings → Tools & MCPs → Connect
- **Other MCP-capable platforms**: use the same MCP endpoint

#### Open an Agentic account
To open a Robinhood Agentic account:
1. Maintain a primary individual investing account in good standing.
2. Complete onboarding after connecting to Robinhood Trading MCP.
3. Authenticate your AI agent and follow on-screen account setup.

> Agentic account opening and authentication must be completed on desktop.

#### Troubleshooting
- Confirm your platform is connected to Robinhood Trading MCP.
- Disconnect/reconnect MCP integration if connection issues occur.
- Follow your AI platform’s debugging docs.
- If the agent reports Robinhood-side errors, contact Robinhood support.

#### Disclosures
Robinhood Agentic Trading is a brokerage product that enables a third-party AI agent to automate investment decisions and order placement in a dedicated account. Trades may execute without direct per-trade input.

Agentic trading involves significant risk, including possible total loss. AI strategies may fail in changing market conditions, act quickly, and be hard to monitor or stop in real time.

AI agents can make errors, misinterpret instructions, use incomplete/outdated information, and behave unexpectedly. Robinhood does not guarantee agent output accuracy or suitability. You remain responsible for monitoring account activity and agent behavior.

You assume risk for AI-executed trades and any use of your data by third-party AI providers. Brokerage services are offered through Robinhood Financial LLC (member SIPC) and clearing through Robinhood Securities, LLC (member SIPC).

---

## ADX Schema Reference

ADX queries use **materialized views** (`dailyStockPriceMV`, `agentStockForecastMV`) to deduplicate rows and serve the most recent data efficiently. The underlying raw tables are still available for audit purposes.

### `dailyStockPrice`

| Column | Type | Description |
|--------|------|-------------|
| `reportTime` | datetime | When the row was written |
| `symbol` | string | Stock ticker (e.g. `AAPL`) |
| `price` | real | Closing price |
| `priceDate` | datetime | Trading date for the price |

**Materialized view:** `dailyStockPriceMV` — deduplicates by `(symbol, priceDate)`, retaining the latest `reportTime`.

### `agentStockForecast`

| Column | Type | Description |
|--------|------|-------------|
| `reportTime` | datetime | When the agent pipeline ran |
| `agentName` | string | Child agent name |
| `symbol` | string | Stock ticker |
| `horizon` | string | `"1m"`, `"3m"`, `"6m"`, or `"1y"` |
| `expectedReturn` | real | Forecasted percentage return |
| `rank` | int | Rank within this agent+horizon (1 = best) |

**Materialized view:** `agentStockForecastMV` — deduplicates by `(agentName, symbol, horizon)`, retaining the latest forecast.

### `agentStockEvaluation`

Written by `evaluation_agent.py` after comparing 1-month forecasts against actual returns.

| Column | Type | Description |
|--------|------|-------------|
| `symbol` | string | Stock ticker |
| `forecastReturn` | real | Forecasted % return from the original prediction |
| `actualReturn` | real | Realized % return over the 30-day window |
| `forecastRank` | int | Rank assigned by the forecasting agent |
| `actualRank` | int | Rank based on actual returns (1 = best performer) |
| `accuracyScore` | real | Combined error: avg of absolute return error and absolute rank error (lower = more accurate) |
| `agentName` | string | Name of the forecasting agent |
| `forecastReportTime` | datetime | Timestamp of the original forecast |
| `reportTime` | datetime | When this evaluation ran |
| `runId` | string | Unique ID for this evaluation run |
| `horizon` | string | Forecast horizon evaluated (always `"1m"`) |

---

## Agent Evaluations

The evaluation pipeline measures how well each agent's **1-month forecasts** matched actual stock returns.

**How it works (runs daily at 03:00 UTC via `daily-evaluation.yml`):**

1. Looks up all `"1m"` horizon forecasts stored exactly **30 days ago** in `agentStockForecast`.
2. Fetches the actual closing prices for those symbols over that 30-day window from `dailyStockPriceMV`.
3. Computes **realized returns** for each symbol and ranks them by actual performance.
4. Calculates a per-prediction **accuracy score** = average of:
   - absolute error between forecasted return and actual return
   - absolute error between forecasted rank and actual rank
5. Writes all rows to `agentStockEvaluation` in ADX.

**Lower accuracy score = more accurate agent.**

You can ask the chat UI "Which agents have been most accurate?" to get an inline chart of per-agent accuracy scores powered by `get_agent_evaluations`.

---

## Security

See [SECURITY.md](SECURITY.md) for security reporting guidance.
