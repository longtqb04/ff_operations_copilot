from fastapi.testclient import TestClient
from src.api.main import app

def test_health_and_demo_investigation():
    with TestClient(app) as client:
        assert client.get("/health").json()=={"status":"ok"}
        response=client.get("/api/investigate/S005",params={"date":"2026-07-10"})
        assert response.status_code==200
        body=response.json(); assert body["store_id"]=="S005"; assert body["revenue_gap"]<0; assert body["recommended_actions"]

def test_what_if_has_disclaimer():
    with TestClient(app) as client:
        body=client.get("/api/what-if/S005",params={"date":"2026-07-10","delivery_recovery":.5}).json()
        assert body["estimated_uplift"]>=0; assert "not guaranteed causal" in body["disclaimer"]

def test_dashboard_and_chart_data():
    with TestClient(app) as client:
        page=client.get("/")
        assert page.status_code==200
        assert "Control tower" in page.text
        assert len(client.get("/api/trend").json())==14
        stores=client.get("/api/store-performance").json()
        assert stores and {"store_id","variance_pct","revenue_gap"} <= stores[0].keys()
        assert any(route.path=="/api/refresh" and "POST" in route.methods for route in app.routes)
        signal=client.get("/api/critical-signal").json()
        assert signal["active"] is True
        assert signal["store_id"] and signal["date"] and signal["investigation"]

def test_store_monitoring_page_and_payload():
    with TestClient(app) as client:
        page=client.get("/store-monitoring")
        assert page.status_code==200 and "Daypart × channel" in page.text
        stores=client.get("/api/stores").json()
        assert len(stores)==12
        payload=client.get("/api/store-monitor/S005",params={"days":14}).json()
        assert payload["store"]["status"]=="critical"
        assert len(payload["timeline"])==14
        assert len(payload["channels"])==5
        assert len(payload["dayparts"])==15
        assert len(payload["products"])==7
        assert payload["incidents"]

def test_investigation_workspace_and_evidence_contract():
    with TestClient(app) as client:
        page=client.get("/investigations")
        assert page.status_code==200 and "Ranked hypotheses" in page.text
        assert "ai-spinner" in page.text and "startAgentLoading" in page.text
        queue=client.get("/api/investigations").json()
        assert queue and queue[0]["incident_id"]
        detail=client.get(f"/api/investigations/{queue[0]['incident_id']}").json()
        assert detail["evidence"] and detail["hypotheses"] and detail["actions"]
        assert all("source" in evidence for evidence in detail["evidence"])
        assert all("contradiction" in hypothesis for hypothesis in detail["hypotheses"])

def test_agentic_investigation_fallback_has_tool_trace(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY",raising=False)
    with TestClient(app) as client:
        status=client.get("/api/agent/status").json()
        assert status["agent_enabled"] and not status["llm_configured"]
        incident=client.get("/api/investigations").json()[0]["incident_id"]
        result=client.post(f"/api/agent/investigate/{incident}").json()
        assert result["mode"]=="deterministic_fallback"
        assert len(result["tool_trace"])>=6
        assert result["report"]["reasoning_summary"]
        assert result["report"]["recommended_actions"]
        assert "get_external_signals" in [step["tool"] for step in result["tool_trace"]]

def test_apify_fixture_integration_and_webhook_guard(monkeypatch):
    monkeypatch.delenv("APIFY_TOKEN",raising=False)
    monkeypatch.delenv("APIFY_ACTOR_ID",raising=False)
    monkeypatch.delenv("APIFY_WEBHOOK_SECRET",raising=False)
    with TestClient(app) as client:
        status=client.get("/api/external/apify/status").json()
        assert status["mode"]=="fixture"
        refreshed=client.post("/api/external/apify/refresh").json()
        assert refreshed["mode"]=="fixture" and refreshed["signal_count"]>=3
        context=client.get("/api/external/apify/signals/S005").json()
        assert context["freshness"]=="fixture"
        assert context["summary"]["delivery_unavailable"] is True
        assert client.post("/api/external/apify/webhook",json={"items":[]}).status_code==401

def test_what_if_studio_and_scenario_guardrails():
    with TestClient(app) as client:
        page=client.get("/what-if")
        assert page.status_code==200 and "Sensitivity analysis" in page.text
        incident=client.get("/api/investigations").json()[0]["incident_id"]
        scenario=client.get(f"/api/scenario/{incident}",params={"delivery_recovery":.75,"promotion_uplift":.1,"target_eta":24,"stockout_recovery":1}).json()
        assert scenario["scenario"]["estimated_uplift"]>0
        assert len(scenario["sensitivity"])==5
        assert 0 < scenario["scenario"]["confidence"] <= 1
        assert "not guaranteed causal" in scenario["disclaimer"]

def test_track_success_metrics_and_expanded_dimensions():
    with TestClient(app) as client:
        evaluation=client.get("/api/evaluation").json()
        assert evaluation["forecast"]["daily_store_mape"]<=.10
        assert evaluation["anomaly_detection"]["precision"]>=.80
        assert evaluation["anomaly_detection"]["p95_detection_lag_minutes"]<120
        payload=client.get("/api/store-monitor/S005",params={"days":14}).json()
        assert {x["channel"] for x in payload["channels"]}=={"dine_in","takeaway","kiosk","delivery","app"}
        assert {x["product_category"] for x in payload["products"]}=={"chicken","burger","rice","sides","beverage","dessert","combo"}
