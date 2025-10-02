
from typing import Any, Dict
from core_agents import CoreAgent
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from starlette.templating import Jinja2Templates
import uvicorn

class UIAgent(CoreAgent):
    """
    Agent responsible for exposing a web-based user interface
    for interacting with the Integrated AI Platform.
    Provides forms for input, displays results and metrics.
    """

    def __init__(self, agent_id: str, name: str, host: str = "0.0.0.0", port: int = 8001):
        super().__init__(agent_id, name)
        self.app = FastAPI(title="Integrated AI Platform UI")
        self.host = host
        self.port = port
        self.templates = Jinja2Templates(directory="templates")
        self._register_routes()

    def initialize(self) -> None:
        self.state["initialized"] = True

    def shutdown(self) -> None:
        self.state.clear()

    def handle(self, input_data: Any) -> Any:
        raise NotImplementedError("Use HTTP endpoints to interact with UIAgent")

    def _register_routes(self):
        @self.app.get("/", response_class=HTMLResponse)
        async def index(request: Request):
            return self.templates.TemplateResponse("index.html", {
                "request": request,
                "title": "Integrated AI Platform",
            })

        @self.app.get("/pipeline_ui", response_class=HTMLResponse)
        async def pipeline_form(request: Request):
            return self.templates.TemplateResponse("pipeline_form.html", {
                "request": request,
                "title": "Run Integrated Pipeline",
            })

        @self.app.post("/pipeline_ui", response_class=HTMLResponse)
        async def pipeline_submit(
            request: Request,
            raw_data: str = Form(...),
            hparams: str = Form(...)
        ):
            import json
            spec = {"data": raw_data, "hparams": json.loads(hparams)}
            from integration.integration_orchestrator import IntegratedOrchestrator
            orchestrator = IntegratedOrchestrator()
            result = await orchestrator.run_pipeline(raw_data, spec)
            return self.templates.TemplateResponse("pipeline_result.html", {
                "request": request,
                "title": "Pipeline Result",
                "result": result
            })

        @self.app.get("/metrics_ui", response_class=HTMLResponse)
        async def metrics_view(request: Request):
            from integration.integration_orchestrator import IntegratedOrchestrator
            orchestrator = IntegratedOrchestrator()
            metrics = orchestrator.get_metrics()
            return self.templates.TemplateResponse("metrics.html", {
                "request": request,
                "title": "Platform Metrics",
                "metrics": metrics
            })

    def run(self):
        uvicorn.run(self.app, host=self.host, port=self.port)

if __name__ == "__main__":
    ui_agent = UIAgent(agent_id="ui1", name="UIAgent", host="0.0.0.0", port=8001)
    ui_agent.initialize()
    ui_agent.run()
