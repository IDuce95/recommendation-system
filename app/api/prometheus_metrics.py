
import time
import psutil
from functools import wraps
from prometheus_client import (
    Counter, Histogram, Gauge, Info,
    generate_latest, CONTENT_TYPE_LATEST
)
from fastapi import Request, Response


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

system_cpu_usage_percent = Gauge(
    'system_cpu_usage_percent',
    'Current CPU usage percentage'
)

system_memory_usage_percent = Gauge(
    'system_memory_usage_percent',
    'Current memory usage percentage'
)

app_info = Info('app_info', 'Application information')


class PrometheusMetrics:
    
    def __init__(self):
        app_info.info({
            'version': '1.0.0',
            'service': 'recommendation-api',
            'environment': 'development'
        })
    
    def update_system_metrics(self):
        try:
            cpu_percent = psutil.cpu_percent()
            system_cpu_usage_percent.set(cpu_percent)
            
            memory = psutil.virtual_memory()
            system_memory_usage_percent.set(memory.percent)
        except Exception as e:
            print(f"Error updating system metrics: {e}")
    
    def record_api_request(self, method: str, endpoint: str,
                           status_code: int, duration: float):
        api_requests_total.labels(
            method=method,
            endpoint=endpoint,
            status_code=status_code
        ).inc()
        
        api_request_duration_seconds.labels(
            method=method,
            endpoint=endpoint
        ).observe(duration)
    
    def record_recommendation(self, user_id: str, rec_type: str,
                              duration: float):
        recommendations_generated_total.labels(
            user_id=user_id,
            recommendation_type=rec_type
        ).inc()
        
        recommendation_generation_duration_seconds.labels(
            recommendation_type=rec_type
        ).observe(duration)


metrics = PrometheusMetrics()


def monitor_recommendation(rec_type: str = "default"):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            user_id = kwargs.get('user_id', 'unknown')
            
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                metrics.record_recommendation(
                    str(user_id), rec_type, duration
                )
                return result
            except Exception as e:
                duration = time.time() - start_time
                metrics.record_recommendation(
                    str(user_id), f"{rec_type}_error", duration
                )
                raise e
        return wrapper
    return decorator


def get_metrics():
    metrics.update_system_metrics()
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


async def metrics_middleware(request: Request, call_next):
    metrics.update_system_metrics()
    
    method = request.method
    endpoint = str(request.url.path)
    start_time = time.time()
    
    response = await call_next(request)
    duration = time.time() - start_time
    
    metrics.record_api_request(
        method, endpoint, response.status_code, duration
    )
    
    return response
