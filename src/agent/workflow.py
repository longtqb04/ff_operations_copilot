import json
import os
from datetime import datetime,timezone
from src.agent.tools import TOOL_DEFINITIONS,execute_tool

SYSTEM_INSTRUCTIONS="""You are an F&B Operations Investigation Agent. Autonomously choose tools to investigate the incident. First establish anomaly impact, then isolate channel/daypart, compare peers, inspect operational signals, obtain governed actions, and run a bounded what-if only when useful. Never invent metrics. Distinguish observed evidence from hypotheses. Stop when evidence is sufficient. Your final response must be JSON with keys: executive_summary, reasoning_summary, scope_assessment, primary_hypothesis, hypotheses, recommended_actions, missing_evidence, confidence, disclaimer. hypotheses is an array of objects with title, confidence, supporting_evidence, contradicting_evidence. Do not reveal hidden chain-of-thought; reasoning_summary is a concise evidence-based rationale. Check get_external_signals before concluding; treat fixture signals as demo evidence and label them clearly."""

def _fallback(incident_id):
    order=[("get_anomaly_event",{}),("get_channel_breakdown",{}),("get_daypart_breakdown",{}),("get_product_breakdown",{}),("get_peer_comparison",{}),("get_operational_signals",{}),("get_external_signals",{}),("get_recommended_actions",{}),("run_what_if",{"delivery_recovery":.75,"promotion_uplift":0})]
    trace=[]; data={}
    for name,args in order:
        result=execute_tool(name,incident_id,args); data[name]=result; trace.append({"step":len(trace)+1,"tool":name,"arguments":args,"result":result})
    e=data["get_anomaly_event"]; peer=data["get_peer_comparison"]; signals=data["get_operational_signals"]; actions=data["get_recommended_actions"]
    eta=signals["delivery_eta"]; tx=signals["transactions"]
    report={"executive_summary":f"{e['store_id']} {e['daypart']} {e['channel']} sales were {abs(e['residual_pct']):.1%} below forecast, creating a {abs(e['revenue_gap']):,.0f} VND gap.","reasoning_summary":f"The loss is isolated to {e['channel']} during {e['daypart']}; regional peers were {peer['peer_median_variance_pct']:+.1%}. ETA increased by {eta['delta']:.1f} minutes and transactions changed {tx['change_pct']:+.1%}, supporting a store-specific delivery disruption.","scope_assessment":peer["assessment"],"primary_hypothesis":"Delivery platform availability or operational throughput disruption","hypotheses":[{"title":"Delivery platform or integration disruption","confidence":.87,"supporting_evidence":["Channel-localized revenue gap","Peers stable","Transaction decline"],"contradicting_evidence":["Direct platform uptime telemetry unavailable"]},{"title":"Kitchen throughput increased delivery ETA","confidence":.82,"supporting_evidence":[f"ETA {eta['observed']} minutes vs {eta['baseline']} baseline"],"contradicting_evidence":["Kitchen telemetry unavailable"]}],"recommended_actions":actions,"missing_evidence":["Delivery platform uptime logs","Category-level product sales"],"confidence":.86,"disclaimer":"AI-generated reasoning grounded in tool evidence; associations are not guaranteed causal effects."}
    return {"mode":"deterministic_fallback","model":None,"report":report,"tool_trace":trace,"iterations":len(trace),"generated_at":datetime.now(timezone.utc).isoformat()}

def run_investigation_agent(incident_id,max_iterations=8):
    if not os.getenv("OPENAI_API_KEY"): return _fallback(incident_id)
    from openai import OpenAI
    client=OpenAI(); model=os.getenv("OPENAI_AGENT_MODEL","gpt-5.4-mini")
    inputs=[{"role":"user","content":f"Investigate incident {incident_id}. Use tools autonomously and return the required JSON report."}]; trace=[]
    for iteration in range(1,max_iterations+1):
        response=client.responses.create(model=model,instructions=SYSTEM_INSTRUCTIONS,input=inputs,tools=TOOL_DEFINITIONS,reasoning={"effort":"medium"})
        calls=[item for item in response.output if item.type=="function_call"]
        if not calls:
            try: report=json.loads(response.output_text)
            except json.JSONDecodeError: report={"executive_summary":response.output_text,"reasoning_summary":"The model returned an unstructured narrative.","confidence":0,"missing_evidence":[],"hypotheses":[],"recommended_actions":[],"disclaimer":"LLM output could not be parsed as structured JSON."}
            return {"mode":"llm_tool_calling","model":model,"report":report,"tool_trace":trace,"iterations":iteration,"response_id":response.id,"generated_at":datetime.now(timezone.utc).isoformat()}
        inputs.extend(response.output)
        for call in calls:
            args=json.loads(call.arguments or "{}"); result=execute_tool(call.name,incident_id,args); trace.append({"step":len(trace)+1,"tool":call.name,"arguments":args,"result":result})
            inputs.append({"type":"function_call_output","call_id":call.call_id,"output":json.dumps(result,ensure_ascii=False)})
    fallback=_fallback(incident_id); fallback["mode"]="fallback_after_iteration_limit"; fallback["tool_trace"]=trace+fallback["tool_trace"]; return fallback
