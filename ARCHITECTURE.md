# Архитектура BPLA Person Detection

## Обзор

Production-ready система детекции людей для БПЛА (Беспилотный летательный аппарат) и спутников с использованием компьютерного зрения. Специализирована для рынка РФ/СНГ.

## Architecture Components

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                           BPLA Person Detection                             │
│                          MAMAA Architecture 1.0.0                           │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                              API Gateway / Ingress                          │
│                         (Nginx Ingress + TLS + Rate Limit)                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                        │
                    ┌───────────────────┼───────────────────┐
                    ▼                   ▼                   ▼
        ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
        │   Genie.jl App   │ │   Genie.jl App   │ │   Genie.jl App   │
        │   (Inference)    │ │   (Inference)    │ │   (Inference)    │
        │   Replica 1      │ │   Replica 2      │ │   Replica 3      │
        └──────────────────┘ └──────────────────┘ └──────────────────┘
                    │                   │                   │
                    └───────────────────┼───────────────────┘
                                        │
        ┌───────────────────────────────┼───────────────────────────────┐
        │                               │                               │
        ▼                               ▼                               ▼
┌───────────────┐             ┌───────────────┐             ┌───────────────┐
│    Redis      │             │   RabbitMQ    │             │  OpenTelemetry│
│    Cache      │             │ Message Queue │             │   Collector   │
│ (L1 Caching)  │             │  (Async API)  │             │   (Tracing)   │
└───────────────┘             └───────────────┘             └───────────────┘
                                        │
                                        ▼
                              ┌──────────────────┐
                              │    Inference     │
                              │    Workers       │
                              │   (GPU Pods)     │
                              └──────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                           Monitoring Stack                                  │
│     Prometheus + Grafana + Alertmanager + ServiceMonitors                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Inference Engine

### Pipeline детекции

```text
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   Входное    │───▶│  Препроцесс  │───▶│   Модель     │───▶│ Постпроцесс │
│  изображение │    │  (resize,    │    │   (forward   │    │  (NMS,       │
│              │    │   normalize) │    │   pass)      │    │   rescale)   │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
                                                                   │
                           ┌───────────────────────────────────────┘
                           ▼
                     ┌──────────────┐
                     │   Результат  │
                     │  (boxes,     │
                     │   scores,    │
                     │   labels)    │
                     └──────────────┘
```

## Key Features

### 1. Distributed Caching (Redis)

- **L1 Cache**: Model metadata caching
- **L2 Cache**: Inference results with configurable TTL
- **Cluster Mode**: High availability support
- **Cache Hit Rate Monitoring**: Real-time metrics

### 2. Async Message Processing (RabbitMQ)

- **Inference Queue**: Async request processing
- **Results Queue**: Result delivery
- **Dead Letter Queue**: Failed message handling
- **backoff with maxRetry Logic**: Exponential retries

### 3. Observability (OpenTelemetry)

- **Distributed Tracing**: End-to-end request tracing
- **Metrics**: Prometheus-compatible metrics
- **Structured Logging**: JSON logs with trace context
- **Span Attributes**: Request ID, user ID, model version

### 4. Autoscaling (HPA/VPA)

- **Horizontal Pod Autoscaler**:
  - CPU-based scaling (70% threshold)
  - Memory-based scaling (80% threshold)
  - Custom metrics (requests per second)
- **Vertical Pod Autoscaler**:
  - Automatic resource recommendation
  - Min/Max limits protection

### 5. Health Checks

- **Liveness Probe**: `/live` - Container health
- **Readiness Probe**: `/ready` - Service availability
- **Startup Probe**: Slow start detection
- **Graceful Shutdown**: Connection draining

## Deployment Options

### Option 1: Kubernetes (Recommended)

```bash
# Deploy with Helm
helm upgrade --install bpla-mamaa ./helm/bpla-person-detection \
  -f ./helm/bpla-person-detection/values-mamaa.yaml

# Or use raw Kubernetes manifests
kubectl apply -f kubernetes/mamaa-deployment.yaml
```

### Option 2: Docker Compose (Development)

```bash
docker-compose -f docker-compose.mamaa.yml up
```

### Option 3: Native (Testing)

```bash
# Install dependencies
] dev .

# Run server
] run --exec "include(\"serve_mamaa.jl\")"
```

## API Endpoints

### Synchronous API

```text
POST /detect/image     - Synchronous image detection
GET  /health          - Health check
GET  /metrics         - Prometheus metrics
GET  /ready           - Readiness probe
GET  /live            - Liveness probe
```

### Asynchronous API

```text
POST /detect/async    - Queue inference request
GET  /result/{id}     - Get async result
GET  /queue/stats     - Queue statistics
GET  /cache/stats     - Cache statistics
```

## Configuration

### Environment Variables

```bash
# Service
ENVIRONMENT=production
API_HOST="0.0.0.0"
API_PORT=8000
WORKERS=4
THREADS=4

# Redis
REDIS_HOST="redis-cache"
REDIS_PORT=6379
CACHE_TTL_DEFAULT=300

# RabbitMQ
RABBITMQ_HOST="message-queue"
RABBITMQ_PORT=5672
QUEUE_INFERENCE="inference_requests"

# OpenTelemetry
OTEL_ENABLED=true
OTEL_EXPORTER_OTLP_ENDPOINT="http://otel-collector:4317"

# Autoscaling
MIN_REPLICAS=3
MAX_REPLICAS=20
```

## Monitoring

### Key Metrics

| Metric | Description | Target |
| ------ | ----------- | ------ |
| `http_request_duration_seconds` | API latency | p95 < 1s |
| `inference_duration_seconds` | Model inference time | p95 < 500ms |
| `cache_hit_rate` | Redis cache efficiency | > 80% |
| `queue_backlog` | Pending requests | < 1000 |
| `error_rate` | HTTP 5xx rate | < 1% |

### Alerts

```yaml
- alert: BPLAHighErrorRate
  expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.05

- alert: BPLAHighLatency
  expr: histogram_quantile(0.95, rate(http_request_duration_seconds[5m])) > 2

- alert: BPLAQueueBacklog
  expr: rabbitmq_queue_messages > 1000
```

## Scaling Strategy

### Horizontal Scaling (HPA)

- Scale out when GPU > 70% or Memory > 80%
- Scale up max 100% pods per 15 seconds
- Scale down stabilization window: 300s

### Vertical Scaling (VPA)

- Auto-adjust pod resources based on usage
- Min: 512Mi RAM, 250m GPU
- Max: 8Gi RAM, 4000m GPU

## Security Features

- **TLS Termination**: HTTPS at ingress level
- **Network Policies**: Restrict pod-to-pod communication
- **Pod Security**: Non-root user, read-only filesystem
- **Secrets Management**: Kubernetes secrets for credentials

## Performance Optimizations

1. **Model Serving**
   - Julia static compilation
   - GPU memory optimization (80% limit)
   - Batch inference support

2. **Caching Strategy**
   - LRU eviction policy
   - Result deduplication
   - Cache warming on startup

3. **Async Processing**
   - RabbitMQ prefetch control
   - Connection pooling
   - Dead letter handling

## Migration from v1

1. Update dependencies:

   ```julia
   ] up
   ```

2. Update deployment:

   ```bash
   kubectl apply -f kubernetes/mamaa-deployment.yaml
   ```

3. Update API calls:
   - Old: `POST /detect/image`
   - New: `POST /api/v1/detect/image`

## Метрики качества

### Для aerial imagery важно

| Метрика      | Описание                 | Целевое значение |
|--------------|--------------------------|------------------|
| mAP@0.5      | Mean Average Precision   | > 0.85           |
| mAP@0.5:0.95 | COCO mAP                 | > 0.65           |
| Recall       | Полнота                  | > 0.90           |
| Precision    | Точность                 | > 0.85           |
| FPS          | Кадров в секунду         | > 30 (GPU)       |

Prometheus метрики:

- `bpla_inference_total` - Всего инференсов
- `bpla_inference_duration_seconds` - Время инференса
- `bpla_detections_total` - Детекций по классам

## Развёртывание

### Kubernetes

```bash
# Helm installation
helm install bpla bpla-charts/bpla-person-detection \
    -n bpla --create-nversation \
    -f values-production.yaml
```

## Troubleshooting

### Check Service Health

```bash
kubectl exec -it <pod> -- curl http://localhost:8000/health
```

### View Logs

```bash
kubectl logs -f <pod> -c inference
```

### Check Metrics

```bash
kubectl exec -it <pod> -- curl http://localhost:9090/metrics
```

## Ограничения и особенности

### Что учитывалось для рынка РФ/СНГ

1. **Разные масштабы съёмки** - от 50м до 500м
2. **Разные погодные условия** - туман, дождь, снег
3. **Разное время суток** - день, ночь (с ИК)
4. **Разные ландшафты** - город, лес, поле, промышленные зоны

### Что не поддерживается

1. Трекинг объектов между кадрами
2. Сегментация людей
3. Оценка позы (pose estimation)
4. Распознавание лиц

## License

MIT License

## Контакты

## 📞 Контакты

**Артём** — автор и архитектор системы

📩 **Почта:** [selemetartem@gmail.com](mailto:selemetartem@gmail.com)  
💬 **Telegram:** [@Artyomkun](https://t.me/Artyomkun)

*Работаем по договору с физлицами / ИП. Готовы к тендерам.*
