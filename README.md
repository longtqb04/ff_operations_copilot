# Fast Food Operations Copilot — MVP

Self-contained hackathon MVP for KFC's F&B track: granular forecasts, residual anomalies, evidence-based investigation, prioritized actions, and what-if estimates.

## Included

- LightGBM forecast at `store × date × daypart × channel`
- Time-based 30-day holdout: MAPE, WAPE, RMSE, bias
- Isolation Forest + residual threshold + minimum VND impact
- Channel/daypart decomposition and regional peer comparison
- Stable Copilot report without an API key
- What-if simulation with an explicit non-causal disclaimer
- FastAPI UI, endpoints, and OpenAPI docs
- Five operational channels: dine-in, takeaway, kiosk, delivery, and app
- Seven item-level product categories with quantity, stock-out, and category driver analysis
- Synthetic ground-truth anomaly labels with precision/recall evaluation
- Simulated near-real-time timestamps with measurable mean and P95 detection lag

Current synthetic validation metrics are available at `GET /api/evaluation`. They are demo evidence, not claims of validation on KFC production data.

## Run (PowerShell)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m scripts.bootstrap
.\.venv\Scripts\python.exe -m uvicorn src.api.main:app --reload
```

Open http://127.0.0.1:8000 or `/docs`.

The home page is an operations control-tower dashboard with live KPI cards,
actual-versus-forecast trends, anomaly feed, store ranking, and an interactive
incident investigation panel. It is responsive and requires no frontend build.

## Four-step demo

1. `/api/overview` — business KPIs and model quality.
2. `/api/anomalies` — anomaly feed; select the planted S005 incident.
3. `/api/investigate/S005?date=2026-07-10` — drivers, peers, summary, actions.
4. `/api/what-if/S005?date=2026-07-10&delivery_recovery=0.5` — recoverable revenue.

Files in `data/processed` can be imported into Power BI. An LLM wording adapter can later consume the investigation JSON without changing the analytical contract.

## Agentic investigation and LLM reasoning

The backend includes an autonomous tool-calling investigation loop with seven governed tools: anomaly event, channel breakdown, daypart breakdown, peer comparison, operational signals, action library, and what-if simulation. Copy `.env.example` to `.env` and set `OPENAI_API_KEY` to enable live Responses API reasoning. The default model is `gpt-5.4-mini` and can be changed through `OPENAI_AGENT_MODEL`.

Without an API key, the same endpoint runs a deterministic evidence-first fallback so the demo remains reliable:

```text
POST /api/agent/investigate/{incident_id}
GET  /api/agent/status
```

The response includes a concise reasoning summary, structured report, execution mode, and full tool trace. Hidden chain-of-thought is never exposed.

## Apify external intelligence

The agent has a governed `get_external_signals` tool for delivery availability, review sentiment, and competitor context. Configure `APIFY_TOKEN`, `APIFY_ACTOR_ID`, and actor-specific `APIFY_ACTOR_INPUT_JSON` in `.env` for live Actor runs. Without credentials, clearly labeled fixture signals keep the demo flow operational.

```text
GET  /api/external/apify/status
GET  /api/external/apify/signals/S005
POST /api/external/apify/refresh
POST /api/external/apify/webhook
```

The webhook requires `X-Apify-Webhook-Secret` and a custom payload containing an `items` array. All Actor output is normalized before it is exposed to the investigation agent.
