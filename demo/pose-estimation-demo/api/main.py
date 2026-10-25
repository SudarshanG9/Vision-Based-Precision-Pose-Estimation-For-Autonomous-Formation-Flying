from fastapi import FastAPI
from api.routes.experiments import router as experiments_router

app=FastAPI(title='Pose Estimation Demo API')
app.include_router(experiments_router)
