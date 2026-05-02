# stonks.ai — Agentic DevOps Sample

An end-to-end sample that shows how to build, containerize, and deploy an **AI agent** to **Azure Container Apps** using GitHub Actions.

The agent (`agents/agent.py`) uses **Azure OpenAI** (GPT-4o) with function-calling tools to answer stock-market questions. The same pattern can be adapted to any agentic workload.

---

## Repository layout

```
stonks.ai/
├── agents/
│   └── agent.py          # AI agent – Azure OpenAI + tool-calling
├── infra/
│   ├── main.bicep         # Root Bicep template (ACR + Container Apps)
│   ├── main.bicepparam    # Default parameter values
│   └── modules/
│       ├── container-registry.bicep
│       └── container-app.bicep
├── .github/
│   └── workflows/
│       └── deploy.yml     # CI/CD pipeline
├── Dockerfile
├── requirements.txt
├── .env.example
└── README.md
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
   # Edit .env and fill in your Azure OpenAI values
   ```

3. **Run the agent locally**

   ```bash
   source .env  # or use python-dotenv
   python -m agents.agent "What is the current price of AAPL and its 5-day moving average?"
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
az deployment group create \
  --resource-group rg-stonksai \
  --template-file infra/main.bicep \
  --parameters infra/main.bicepparam \
  --parameters azureOpenAiApiKey="<your-key>"
```

### 2 — Configure GitHub Actions (one-time)

Set up **OIDC federated credentials** so the workflow can authenticate to Azure without long-lived secrets:

```bash
# Create a service principal
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
| `ACR_LOGIN_SERVER` | e.g. `stonksaiacr.azurecr.io` |
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI endpoint URL |
| `AZURE_OPENAI_API_KEY` | Azure OpenAI API key |

And optionally add these **repository variables**:

| Variable | Default |
|----------|---------|
| `BASE_NAME` | `stonksai` |
| `AZURE_OPENAI_DEPLOYMENT` | `gpt-4o` |

### 3 — Deploy

Push to `main` (or trigger the workflow manually). The pipeline will:

1. Build the Docker image and push it to ACR.
2. Deploy / update the Bicep-managed infrastructure.
3. Update the Container App with the new image tag.

---

## Architecture

```
GitHub Actions
    │
    ├─► Azure Container Registry (ACR)   ← Docker image
    │
    └─► Azure Container Apps
            └─► stonksai-agent container
                    └─► Azure OpenAI (GPT-4o)
```

---

## Security

See [SECURITY.md](SECURITY.md) for security reporting guidance.

