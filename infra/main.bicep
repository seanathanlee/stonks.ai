// ============================================================
// Main Bicep template – Stonks.ai Azure infrastructure
// Provisions:
//   • Azure AI Services (Cognitive Services) account
//   • Azure Data Explorer (Kusto) cluster + database + tables
//   • Azure Container Registry
//   • Azure Container App (FastAPI backend)
// ============================================================

@description('Base name used to derive all resource names.')
param baseName string = 'stonksai'

@description('Azure region for all resources.')
param location string = resourceGroup().location

@description('Azure region for the Static Web App. Must be a region that supports Azure Static Web Apps.')
@allowed([
  'centralus'
  'eastus2'
  'eastasia'
  'westeurope'
  'westus2'
])
param swaLocation string = 'centralus'

@description('SKU for the Cognitive Services account.')
@allowed(['S0'])
param sku string = 'S0'

@description('SKU name for the Azure Data Explorer cluster.')
@allowed(['Dev(No SLA)_Standard_D11_v2', 'Standard_D11_v2', 'Standard_D12_v2'])
param adxSkuName string = 'Dev(No SLA)_Standard_D11_v2'

@description('SKU tier for the Azure Data Explorer cluster.')
@allowed(['Basic', 'Standard'])
param adxSkuTier string = 'Basic'

@description('Number of ADX cluster instances.')
param adxCapacity int = 1

@description('Azure OpenAI API key passed as a secure parameter so it is stored as a Container App secret.')
@secure()
param azureOpenAIApiKey string = ''

@description('Container image to run in the Container App. The deploy workflow passes the real ACR image after pushing it; the default placeholder is only used for fully-manual Bicep deployments.')
param containerImage string = 'mcr.microsoft.com/azuredocs/containerapps-helloworld:latest'

// ============================================================
// Azure AI Services (Cognitive Services) account
// ============================================================

resource cognitiveServices 'Microsoft.CognitiveServices/accounts@2023-05-01' = {
  name: '${baseName}-ai'
  location: location
  kind: 'AIServices'
  sku: {
    name: sku
  }
  properties: {
    customSubDomainName: '${baseName}-ai'
    publicNetworkAccess: 'Enabled'
  }
}

// ============================================================
// Azure OpenAI model deployment: gpt-4.1
// ============================================================

resource gpt41Deployment 'Microsoft.CognitiveServices/accounts/deployments@2023-05-01' = {
  name: 'gpt-4.1'
  parent: cognitiveServices
  sku: {
    name: 'Standard'
    capacity: 60
  }
  properties: {
    model: {
      format: 'OpenAI'
      name: 'gpt-4.1'
      version: '2025-04-14'
    }
  }
}

// ============================================================
// Azure Data Explorer (Kusto) cluster
// ============================================================

resource adxCluster 'Microsoft.Kusto/clusters@2023-08-15' = {
  name: '${baseName}adx'
  location: location
  sku: {
    name: adxSkuName
    tier: adxSkuTier
    capacity: adxCapacity
  }
  properties: {
    enableStreamingIngest: true
    enablePurge: false
    publicNetworkAccess: 'Enabled'
  }
}

// ============================================================
// ADX database: stonksai
// ============================================================

resource adxDatabase 'Microsoft.Kusto/clusters/databases@2023-08-15' = {
  name: 'stonksai'
  parent: adxCluster
  location: location
  kind: 'ReadWrite'
  properties: {
    softDeletePeriod: 'P365D'
    hotCachePeriod: 'P31D'
  }
}

// ============================================================
// ADX table: dailyStockPrice
//   Columns: reportTime (datetime), symbol (string),
//            price (real), priceDate (datetime)
// ============================================================

resource dailyStockPriceTable 'Microsoft.Kusto/clusters/databases/scripts@2023-08-15' = {
  name: 'create-daily-stock-price-table'
  parent: adxDatabase
  properties: {
    #disable-next-line use-secure-value-for-secure-inputs
    scriptContent: '''
.create-merge table dailyStockPrice (
    reportTime: datetime,
    symbol: string,
    price: real,
    priceDate: datetime
)

.alter-merge table dailyStockPrice policy retention softdelete = 365d

.create-or-alter table dailyStockPrice ingestion json mapping 'dailyStockPriceMapping'
'[{"column":"reportTime","path":"$.reportTime","datatype":"datetime"},{"column":"symbol","path":"$.symbol","datatype":"string"},{"column":"price","path":"$.price","datatype":"real"},{"column":"priceDate","path":"$.priceDate","datatype":"datetime"}]'
'''
    continueOnErrors: false
  }
}

// ============================================================
// ADX table: agentStockForecast
//   Columns: reportTime (datetime), agentName (string),
//            symbol (string), horizon (string),
//            expectedReturn (real), rank (int)
// ============================================================

resource agentStockForecastTable 'Microsoft.Kusto/clusters/databases/scripts@2023-08-15' = {
  name: 'create-agent-stock-forecast-table'
  parent: adxDatabase
  properties: {
    #disable-next-line use-secure-value-for-secure-inputs
    scriptContent: '''
.create-merge table agentStockForecast (
    reportTime: datetime,
    agentName: string,
    symbol: string,
    horizon: string,
    expectedReturn: real,
    rank: int
)

.alter-merge table agentStockForecast policy retention softdelete = 365d

.create-or-alter table agentStockForecast ingestion json mapping 'agentStockForecastMapping'
'[{"column":"reportTime","path":"$.reportTime","datatype":"datetime"},{"column":"agentName","path":"$.agentName","datatype":"string"},{"column":"symbol","path":"$.symbol","datatype":"string"},{"column":"horizon","path":"$.horizon","datatype":"string"},{"column":"expectedReturn","path":"$.expectedReturn","datatype":"real"},{"column":"rank","path":"$.rank","datatype":"int"}]'
'''
    continueOnErrors: false
  }
}

// ============================================================
// ADX materialized view: dailyStockPriceMV
//   Deduplicates dailyStockPrice by (symbol, priceDate),
//   retaining the row with the latest reportTime.
// ============================================================

resource dailyStockPriceMVScript 'Microsoft.Kusto/clusters/databases/scripts@2023-08-15' = {
  name: 'create-daily-stock-price-mv'
  parent: adxDatabase
  dependsOn: [dailyStockPriceTable]
  properties: {
    #disable-next-line use-secure-value-for-secure-inputs
    scriptContent: '''
.create-or-alter materialized-view with (backfill=true) dailyStockPriceMV on table dailyStockPrice
{
    dailyStockPrice
    | summarize arg_max(reportTime, price) by symbol, priceDate
}
'''
    continueOnErrors: false
  }
}

// ============================================================
// ADX table: agentStockEvaluation
//   Columns: symbol (string), forecastReturn (real),
//            actualReturn (real), forecastRank (int),
//            actualRank (int), accuracyScore (real),
//            agentName (string), forecastReportTime (datetime),
//            reportTime (datetime), runId (string),
//            horizon (string)
// ============================================================

resource agentStockEvaluationTable 'Microsoft.Kusto/clusters/databases/scripts@2023-08-15' = {
  name: 'create-agent-stock-evaluation-table'
  parent: adxDatabase
  dependsOn: [agentStockForecastTable]
  properties: {
    #disable-next-line use-secure-value-for-secure-inputs
    scriptContent: '''
.create-merge table agentStockEvaluation (
    symbol: string,
    forecastReturn: real,
    actualReturn: real,
    forecastRank: int,
    actualRank: int,
    accuracyScore: real,
    agentName: string,
    forecastReportTime: datetime,
    reportTime: datetime,
    runId: string,
    horizon: string
)

.alter-merge table agentStockEvaluation policy retention softdelete = 365d

.create-or-alter table agentStockEvaluation ingestion json mapping 'agentStockEvaluationMapping'
'[{"column":"symbol","path":"$.symbol","datatype":"string"},{"column":"forecastReturn","path":"$.forecastReturn","datatype":"real"},{"column":"actualReturn","path":"$.actualReturn","datatype":"real"},{"column":"forecastRank","path":"$.forecastRank","datatype":"int"},{"column":"actualRank","path":"$.actualRank","datatype":"int"},{"column":"accuracyScore","path":"$.accuracyScore","datatype":"real"},{"column":"agentName","path":"$.agentName","datatype":"string"},{"column":"forecastReportTime","path":"$.forecastReportTime","datatype":"datetime"},{"column":"reportTime","path":"$.reportTime","datatype":"datetime"},{"column":"runId","path":"$.runId","datatype":"string"},{"column":"horizon","path":"$.horizon","datatype":"string"}]'
'''
    continueOnErrors: false
  }
}

// ============================================================
// Azure Container Registry
// ============================================================

resource acr 'Microsoft.ContainerRegistry/registries@2023-07-01' = {
  name: '${baseName}acr'
  location: location
  sku: {
    name: 'Basic'
  }
  properties: {
    adminUserEnabled: false
    publicNetworkAccess: 'Enabled'
  }
}

// ============================================================
// ADX materialized view: agentStockForecastMV
//   Deduplicates agentStockForecast by (agentName, symbol,
//   horizon), retaining the row with the latest reportTime.
// ============================================================

resource agentStockForecastMVScript 'Microsoft.Kusto/clusters/databases/scripts@2023-08-15' = {
  name: 'create-agent-stock-forecast-mv'
  parent: adxDatabase
  dependsOn: [agentStockForecastTable]
  properties: {
    #disable-next-line use-secure-value-for-secure-inputs
    scriptContent: '''
.create-or-alter materialized-view with (backfill=true) agentStockForecastMV on table agentStockForecast
{
    agentStockForecast
    | summarize arg_max(reportTime, expectedReturn, rank) by agentName, symbol, horizon
}
'''
    continueOnErrors: false
  }
}

// ============================================================
// Container App Environment (Consumption plan)
// ============================================================

resource containerAppEnv 'Microsoft.App/managedEnvironments@2024-03-01' = {
  name: '${baseName}-env'
  location: location
  properties: {
    zoneRedundant: false
  }
}

// ============================================================
// Container App – FastAPI backend
// On first provision the deploy workflow pre-creates the ACR,
// pushes the real image, then passes it via the containerImage
// parameter — so no placeholder image is ever used.
// ============================================================

resource containerApp 'Microsoft.App/containerApps@2024-03-01' = {
  name: '${baseName}-api'
  location: location
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    environmentId: containerAppEnv.id
    configuration: {
      ingress: {
        external: true
        targetPort: 8000
        transport: 'http'
        allowInsecure: false
      }
      secrets: azureOpenAIApiKey != '' ? [
        {
          name: 'azure-openai-api-key'
          value: azureOpenAIApiKey
        }
      ] : []
      registries: [
        {
          server: acr.properties.loginServer
          identity: 'system'
        }
      ]
    }
    template: {
      containers: [
        {
          name: 'stonksai-api'
          image: containerImage
          resources: {
            cpu: json('0.5')
            memory: '1Gi'
          }
          probes: [
            {
              // Allow up to 100 seconds (10 attempts × 10 s period) for the
              // container to start before liveness/readiness kicks in.
              type: 'Startup'
              httpGet: {
                path: '/api/health'
                port: 8000
              }
              initialDelaySeconds: 5
              periodSeconds: 10
              failureThreshold: 10
              timeoutSeconds: 5
            }
            {
              // Restart the container if it stops responding.
              type: 'Liveness'
              httpGet: {
                path: '/api/health'
                port: 8000
              }
              initialDelaySeconds: 10
              periodSeconds: 30
              failureThreshold: 3
              timeoutSeconds: 5
            }
            {
              // Stop routing traffic to the container until it is ready.
              type: 'Readiness'
              httpGet: {
                path: '/api/health'
                port: 8000
              }
              initialDelaySeconds: 5
              periodSeconds: 10
              failureThreshold: 3
              timeoutSeconds: 5
            }
          ]
          env: concat(
            [
              { name: 'AZURE_OPENAI_ENDPOINT', value: cognitiveServices.properties.endpoint }
              { name: 'AZURE_OPENAI_DEPLOYMENT', value: 'gpt-4.1' }
              { name: 'AZURE_OPENAI_API_VERSION', value: '2025-01-01-preview' }
              { name: 'ADX_CLUSTER_URI', value: adxCluster.properties.uri }
              { name: 'ADX_DATABASE', value: 'stonksai' }
              {
                name: 'CORS_ORIGINS'
                value: 'https://skewthis.com,https://www.skewthis.com,https://${staticWebApp.properties.defaultHostname}'
              }
            ],
            azureOpenAIApiKey != '' ? [
              { name: 'AZURE_OPENAI_API_KEY', secretRef: 'azure-openai-api-key' }
            ] : []
          )
        }
      ]
      scale: {
        minReplicas: 1
        maxReplicas: 3
      }
    }
  }
}

// Allow the Container App's managed identity to pull images from ACR
resource acrPullRoleAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  // AcrPull built-in role: 7f951dda-4ed3-4680-a7ca-43fe172d538d
  name: guid(acr.id, containerApp.id, '7f951dda-4ed3-4680-a7ca-43fe172d538d')
  scope: acr
  properties: {
    roleDefinitionId: subscriptionResourceId(
      'Microsoft.Authorization/roleDefinitions',
      '7f951dda-4ed3-4680-a7ca-43fe172d538d'
    )
    principalId: containerApp.identity.principalId
    principalType: 'ServicePrincipal'
  }
}

// Grant the Container App's managed identity access to the ADX database
resource adxContainerAppPrincipal 'Microsoft.Kusto/clusters/databases/principalAssignments@2023-08-15' = {
  name: 'containerapp-user'
  parent: adxDatabase
  properties: {
    principalId: containerApp.identity.principalId
    principalType: 'App'
    role: 'User'
    tenantId: subscription().tenantId
  }
}

// ============================================================
// Azure Static Web App — skewthis.com frontend
// ============================================================

resource staticWebApp 'Microsoft.Web/staticSites@2023-01-01' = {
  name: '${baseName}-web'
  location: swaLocation
  sku: {
    name: 'Free'
    tier: 'Free'
  }
  properties: {
    buildProperties: {
      skipGithubActionWorkflowGeneration: true
    }
  }
}

// Apex domain: skewthis.com — requires a DNS TXT record for validation.
// Before this resource can reach the "Validated" state you must add a
// TXT record at your DNS provider:
//   Name:  @ (or skewthis.com)
//   Type:  TXT
//   Value: <validationToken shown in the Azure Portal for this domain>
resource customDomainApex 'Microsoft.Web/staticSites/customDomains@2023-01-01' = {
  name: 'skewthis.com'
  parent: staticWebApp
  properties: {
    validationMethod: 'dns-txt-token'
  }
}

// www subdomain: www.skewthis.com — requires a CNAME record for validation.
// Add a DNS CNAME record at your DNS provider:
//   Name:  www
//   Type:  CNAME
//   Value: ${staticWebApp.properties.defaultHostname}
resource customDomainWww 'Microsoft.Web/staticSites/customDomains@2023-01-01' = {
  name: 'www.skewthis.com'
  parent: staticWebApp
  properties: {
    validationMethod: 'cname-delegation'
  }
  dependsOn: [customDomainApex]
}

// ============================================================
// Outputs
// ============================================================

output cognitiveServicesEndpoint string = cognitiveServices.properties.endpoint
output cognitiveServicesId string = cognitiveServices.id
output adxClusterUri string = adxCluster.properties.uri
output adxClusterId string = adxCluster.id
output adxDatabaseName string = adxDatabase.name
output staticWebAppName string = staticWebApp.name
output staticWebAppDefaultHostname string = staticWebApp.properties.defaultHostname
output acrLoginServer string = acr.properties.loginServer
output acrName string = acr.name
output containerAppName string = containerApp.name
output containerAppFqdn string = containerApp.properties.configuration.ingress.fqdn
