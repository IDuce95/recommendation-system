
import time
import psutil
from typing import Optional
from functools import wraps
from prometheus_client import (
    Counter, Histogram, Gauge, Info,
    generate_latest, CONTENT_TYPE_LATEST
)
from fastapi import Request, Response
from fastapi.responses import Response as FastAPIResponse


api_requests_total = Counter(
    'api_requests_total',
    'Total number of API requests',
    ['method', 'endpoint', 'status_code']
)

api_request_duration_seconds = Histogram(
    'api_request_duration_seconds',
    'API request duration in seconds',
    ['method', 'endpoint'],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
)

api_requests_in_progress = Gauge(
    'api_requests_in_progress',
    'Number of API requests currently being processed',
    ['method', 'endpoint']
)

recommendations_generated_total = Counter(
    'recommendations_generated_total',
    'Total number of recommendations generated',
    ['user_id', 'recommendation_type']
)

recommendation_generation_duration_seconds = Histogram(
    'recommendation_generation_duration_seconds',
    'Time taken to generate recommendations',
    ['recommendation_type'],
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
)

recommendation_quality_score = Gauge(
    'recommendation_quality_score',
    'Quality score of generated recommendations',
    ['recommendation_type']
)

system_cpu_usage_percent = Gauge(
    'system_cpu_usage_percent',
    'Current CPU usage percentage'
)

system_memory_usage_bytes = Gauge(
    'system_memory_usage_bytes',
    'Current memory usage in bytes'
)

system_memory_usage_percent = Gauge(
    'system_memory_usage_percent',
    'Current memory usage percentage'
)

system_disk_usage_percent = Gauge(
    'system_disk_usage_percent',
    'Current disk usage percentage',
    ['device']
)

db_connections_active = Gauge(
    'db_connections_active',
    'Number of active database connections'
)

db_query_duration_seconds = Histogram(
    'db_query_duration_seconds',
    'Database query duration in seconds',
    ['query_type'],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0]
)

mlflow_experiments_logged_total = Counter(
    'mlflow_experiments_logged_total',
    'Total number of MLflow experiments logged'
)

mlflow_models_saved_total = Counter(
    'mlflow_models_saved_total',
    'Total number of models saved to MLflow'
)

app_info = Info(
    'app_info',
    'Application information'
)



class PrometheusMetrics:
    
    def __init__(self):
        self.app_info.info({
            'version': '1.0.0',
            'service': 'recommendation-api',
            'environment': 'development'
        })
    
    def update_system_metrics(self):
        try:
            cpu_percent = psutil.cpu_percent()
            system_cpu_usage_percent.set(cpu_percent)
            
            memory = psutil.virtual_memory()
            system_memory_usage_bytes.set(memory.used)
            system_memory_usage_percent.set(memory.percent)
            
            for disk in psutil.disk_partitions():
                try:
                    usage = psutil.disk_usage(disk.mountpoint)
                    system_disk_usage_percent.labels(
                        device=disk.device
                    ).set(usage.percent)
                except (PermissionError, OSError):
                    continue
                    
        except Exception as e:
            print(f"Error updating system metrics: {e}")
    
    def record_api_request(
        self, 
        method: str, 
        endpoint: str, 
        status_code: int, 
        duration: float
    ):
        api_requests_total.labels(
            method=method,
            endpoint=endpoint,
            status_code=status_code
        ).inc()
        
        api_request_duration_seconds.labels(
            method=method,
            endpoint=endpoint
        ).observe(duration)
    
    def record_recommendation_generated(
        self, 
        user_id: str, 
        recommendation_type: str,
        duration: float,
        quality_score: Optional[float] = None
    ):
        recommendations_generated_total.labels(
            user_id=user_id,
            recommendation_type=recommendation_type
        ).inc()
        
        recommendation_generation_duration_seconds.labels(
            recommendation_type=recommendation_type
        ).observe(duration)
        
        if quality_score is not None:
            recommendation_quality_score.labels(
                recommendation_type=recommendation_type
            ).set(quality_score)
    
    def record_db_query(self, query_type: str, duration: float):
        db_query_duration_seconds.labels(
            query_type=query_type
        ).observe(duration)
    
    def record_mlflow_activity(self, activity_type: str):
        if activity_type == "experiment_logged":
            mlflow_experiments_logged_total.inc()
        elif activity_type == "model_saved":
            mlflow_models_saved_total.inc()


metrics = PrometheusMetrics()

def metrics_middleware():
    
    async def middleware(request: Request, call_next):
        metrics.update_system_metrics()
        
        method = request.method
        endpoint = str(request.url.path)
        
        api_requests_in_progress.labels(
            method=method,
            endpoint=endpoint
        ).inc()
        
        start_time = time.time()
        
        try:
            response = await call_next(request)
            status_code = response.status_code
            
        except Exception as e:
            status_code = 500
            response = FastAPIResponse(
                content=f"Internal Server Error: {str(e)}", 
                status_code=500
            )
        
        finally:
            duration = time.time() - start_time
            
            metrics.record_api_request(
                method=method,
                endpoint=endpoint,
                status_code=status_code,
                duration=duration
            )
            
            api_requests_in_progress.labels(
                method=method,
                endpoint=endpoint
            ).dec()
        
        return response
    
    return middleware


def monitor_recommendation_generation(recommendation_type: str = "default"):
    
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            user_id = kwargs.get('user_id', 'unknown')
            
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                
                quality_score = None
                if isinstance(result, dict) and 'quality_score' in result:
                    quality_score = result['quality_score']
                
                metrics.record_recommendation_generated(
                    user_id=str(user_id),
                    recommendation_type=recommendation_type,
                    duration=duration,
                    quality_score=quality_score
                )
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                metrics.record_recommendation_generated(
                    user_id=str(user_id),
                    recommendation_type=f"{recommendation_type}_error",
                    duration=duration
                )
                raise e
        
        return wrapper
    return decorator

def monitor_db_query(query_type: str):
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                metrics.record_db_query(query_type, duration)
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                metrics.record_db_query(f"{query_type}_error", duration)
                raise e
        
        return wrapper
    return decorator


def get_metrics():
    metrics.update_system_metrics()
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )
