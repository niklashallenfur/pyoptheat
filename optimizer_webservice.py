from fastapi import FastAPI, HTTPException

from heat_optimizer_port import OptimizationService, OptimizationParameters

app = FastAPI()
service = OptimizationService()

# Initialize storage for the latest plan and parameters
app.state.latest_plan = None
app.state.latest_params = None

@app.post("/optimize")
def optimize(params: OptimizationParameters):
    try:
        plan = service.optimize(params)
        # Store the latest plan and parameters
        app.state.latest_plan = plan
        app.state.latest_params = params
        return plan
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/optimize")
def get_latest_plan():
    if app.state.latest_plan is not None:
        return app.state.latest_plan
    else:
        raise HTTPException(status_code=404, detail="No optimization has been performed yet.")

@app.get("/optimize/parameters")
def get_latest_parameters():
    if app.state.latest_params is not None:
        return app.state.latest_params
    else:
        raise HTTPException(status_code=404, detail="No optimization has been performed yet.")
