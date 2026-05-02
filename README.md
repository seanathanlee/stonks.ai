# stonks.ai — Agentic DevOps Sample

An end-to-end sample that shows how to build, containerize, and deploy a **multi-agent AI system** to **Azure** using GitHub Actions.

The system scrapes NASDAQ stock prices daily, stores them in **Azure Data Explorer (Kusto)**, and runs **9 specialised child agents** (each with a distinct investment philosophy) through a **parent orchestrator agent** to produce ranked stock picks across four time horizons. All forecasts are persisted back to ADX for querying and analysis.

---

## Repository layout

```
stonks.ai/
├── agents/
│   ├── agent.py           # Original single AI agent (Azure OpenAI + tool-calling)
│   ├── adx_client.py      # ADX query + ingestion wrapper
│   ├── scraper.py         # NASDAQ symbol list + price fetcher
│   ├── child_agents.py    # 9 child agent definitions + shared runner
│   └── parent_agent.py    # Orchestrator: reads ADX, fans out, writes forecasts
├── infra/
│   ├── main.bicep         # Root Bicep template (AI Services + ADX cluster/DB/tables)
│   └── main.bicepparam    # Default parameter values
├── .github/
│   └── workflows/
│       ├── deploy.yml          # CI/CD pipeline (infrastructure deploy)
│       ├── daily-scrape.yml    # Scheduled daily price ingestion (01:00 UTC)
│       └── snapshot-scrape.yml # Manual 30-day backfill trigger
├── Dockerfile
├── requirements.txt
├── .env.example
└── README.md
```

---

## Architecture

```
GitHub Actions
    │
    ├─► deploy.yml
    │       └─► Azure AI Services (Azure OpenAI GPT-4o)
    │       └─► Azure Data Explorer cluster (stonksaiadx)
    │               ├── database: stonksai
    │               │       ├── table: dailyStockPrice
    │               │       └── table: agentStockForecast
    │
    ├─► daily-scrape.yml  (runs 01:00 UTC every day)
    │       └─► agents/scraper.py --mode daily
    │               └─► Yahoo Finance API  ──► ADX dailyStockPrice
    │
    ├─► snapshot-scrape.yml  (manual trigger)
    │       └─► agents/scraper.py --mode snapshot
    │               └─► Yahoo Finance API (30 days)  ──► ADX dailyStockPrice
    │
    └─► agents/parent_agent.py  (run on-demand or on a schedule)
            ├─► ADX dailyStockPrice  ──► stock data (30-day window)
            ├─► 9 child agents (concurrent)
            │       ├── momentum_trader
            │       ├── mean_reversion
            │       ├── value_investor
            │       ├── growth_investor
            │       ├── volatility_hunter
            │       ├── sector_rotation
            │       ├── technical_analyst
            │       ├── contrarian_investor
            │       └── risk_adjusted_optimizer
            └─► ADX agentStockForecast  (5 picks × 4 horizons × 9 agents)
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

3. **Run the original single agent**

   ```bash
   python -m agents.agent "What is the current price of AAPL and its 5-day moving average?"
   ```

4. **Backfill 30 days of price data (first-time setup)**

   ```bash
   python -m agents.scraper --mode snapshot
   ```

5. **Run the multi-agent orchestrator**

   ```bash
   python -m agents.parent_agent
   ```

---

## Deploy to Azure

### 1 — Create Azure resources (one-time)

```bash
# Log in
az login

# Create a resource group
az group create --name rg-stonksai --location eastus

# Deploy the Bicep template (provisions AI Services + ADX cluster + tables)
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
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI endpoint URL |
| `AZURE_OPENAI_API_KEY` | Azure OpenAI API key |
| `ADX_CLUSTER_URI` | ADX cluster URI, e.g. `https://stonksaiadx.eastus.kusto.windows.net` |
| `ADX_DATABASE` | ADX database name (`stonksai`) |

And optionally add these **repository variables**:

| Variable | Default |
|----------|---------|
| `BASE_NAME` | `stonksai` |
| `AZURE_OPENAI_DEPLOYMENT` | `gpt-4o` |

### 3 — Seed historical data (one-time)

After the infrastructure is deployed, trigger the snapshot workflow to backfill 30 days of price data:

GitHub → Actions → **Snapshot Stock Price Scrape** → **Run workflow**

### 4 — Automated daily scraping

The `daily-scrape.yml` workflow runs automatically at **01:00 UTC** every day, fetching the previous day's closing prices for all NASDAQ symbols and ingesting them into ADX.

You can also trigger it manually from the GitHub Actions tab.

---

## ADX Schema Reference

### `dailyStockPrice`

| Column | Type | Description |
|--------|------|-------------|
| `reportTime` | datetime | When the row was written |
| `symbol` | string | Stock ticker (e.g. `AAPL`) |
| `price` | real | Closing price |
| `priceDate` | datetime | Trading date for the price |

### `agentStockForecast`

| Column | Type | Description |
|--------|------|-------------|
| `reportTime` | datetime | When the agent pipeline ran |
| `agentName` | string | Child agent name |
| `symbol` | string | Stock ticker |
| `horizon` | string | `"1m"`, `"3m"`, `"6m"`, or `"1y"` |
| `expectedReturn` | real | Forecasted percentage return |
| `rank` | int | Rank within this agent+horizon (1 = best) |

---

## Security

See [SECURITY.md](SECURITY.md) for security reporting guidance.

