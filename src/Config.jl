# MIT License
# Copyright (c) 2026 Person Detection Project
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

module Config

using OrderedCollections
using Logging
using Random
using Printf
using Dates
using JSON

const SERVICE_NAME::String = "bpla-person-detection"
const SERVICE_VERSION::String = "1.0.0"
const SERVICE_UUID::String = "bpla-" * string(uuid4())[1:8]
const BUILD_TIMESTAMP::String = string(now())
const ENVIRONMENT::String = get(ENV, "ENVIRONMENT", "production")
const REGION::String = get(ENV, "REGION", "us-east-1")
const ZONE::String = get(ENV, "ZONE", "us-east-1a")
const CLUSTER_NAME::String = get(ENV, "CLUSTER_NAME", "bpla-prod")
const API_HOST::String = "0.0.0.0"
const API_PORT::Int = parse(Int, get(ENV, "API_PORT", "8000"))
const METRICS_PORT::Int = parse(Int, get(ENV, "METRICS_PORT", "9090"))
const HEALTH_PORT::Int = parse(Int, get(ENV, "HEALTH_PORT", "8081"))
const WORKERS::Int = parse(Int, get(ENV, "WORKERS", string(Sys.CPU_THREADS)))
const WORKER_TYPE::String = get(ENV, "WORKER_TYPE", "thread")
const RATE_LIMIT_ENABLED::Bool = true
const RATE_LIMIT_ALGORITHM::String = "token_bucket"
const RATE_LIMIT_REQUESTS::Int = 1000
const RATE_LIMIT_WINDOW::Int = 60
const RATE_LIMIT_BURST::Int = 50
const RATE_LIMITS = Dict(
    "/api/v1/detect" => (100, 60),
    "/api/v1/batch" => (50, 60), 
    "/api/v1/stream" => (10, 60),
    "/metrics" => (0, 0)
)
const CIRCUIT_BREAKER_ENABLED::Bool = true
const CIRCUIT_BREAKER_FAILURE_LIMIT::Int = 5
const CIRCUIT_BREAKER_RECOVERY_TIMEOUT::Int = 30
const CIRCUIT_BREAKER_SLOW_CALL_THRESHOLD::Int = 5000
const CIRCUIT_BREAKER_SLOW_CALL_RATIO::Float64 = 0.5
const REDIS_MODE::String = get(ENV, "REDIS_MODE", "cluster")
const REDIS_HOST::String = get(ENV, "REDIS_HOST", "redis-cache")
const REDIS_PORT::Int = parse(Int, get(ENV, "REDIS_PORT", "6379"))
const REDIS_PASSWORD::String = get(ENV, "REDIS_PASSWORD", "")
const REDIS_SENTINEL_HOSTS::Vector{String} = ["sentinel-0", "sentinel-1", "sentinel-2"]
const REDIS_CLUSTER_NODES::Vector{Tuple{String, Int}} = [
    ("redis-cluster-0", 6379),
    ("redis-cluster-1", 6379), 
    ("redis-cluster-2", 6379),
    ("redis-cluster-3", 6379),
    ("redis-cluster-4", 6379),
    ("redis-cluster-5", 6379),
]
const CACHE_TTL = Dict(
    :default => 300,
    :model => 3600,
    :result => 600,
    :metadata => 86400,
    :config => 3600,
    :health => 60
)
const CACHE_MAX_MEMORY::String = "2gb"
const CACHE_EVICTION_POLICY::String = "allkeys-lru"
const MQ_TYPE::String = get(ENV, "MQ_TYPE", "rabbitmq")
const MQ_HOST::String = get(ENV, "MQ_HOST", "message-queue")
const MQ_PORT::Int = parse(Int, get(ENV, "MQ_PORT", "5672"))
const RABBITMQ_VHOST::String = get(ENV, "RABBITMQ_VHOST", "/")
const RABBITMQ_USER::String = get(ENV, "RABBITMQ_USER", "bpla")
const RABBITMQ_PASSWORD::String = get(ENV, "RABBITMQ_PASSWORD", "")
const KAFKA_BROKERS::Vector{String} = ["kafka-0:9092", "kafka-1:9092", "kafka-2:9092"]
const KAFKA_CONSUMER_GROUP::String = "bpla-detection-group"
const KAFKA_AUTO_OFFSET_RESET::String = "earliest"
const QUEUES = Dict(
    :inference => "inference_requests",
    :results => "inference_results",
    :batch => "batch_requests",
    :stream => "stream_requests",
    :dead_letter => "inference_dlq",
    :retry => "inference_retry",
    :audit => "audit_log"
)
const PREFETCH_COUNT::Int = 10
const CONSUMER_TIMEOUT::Int = 30000
const HEARTBEAT_INTERVAL::Int = 60
const RETRY_POLICY = Dict(
    :max_attempts => 3,
    :initial_delay => 1,
    :max_delay => 300,
    :backoff_multiplier => 2.0,
    :jitter => 0.1
)
const OTEL_ENABLED::Bool = true
const OTEL_SERVICE_NAME::String = SERVICE_NAME
const OTEL_SERVICE_VERSION::String = SERVICE_VERSION
const OTEL_EXPORTER_OTLP_ENDPOINT::String = get(ENV, "OTEL_EXPORTER_OTLP_ENDPOINT", "http://otel-collector:4317")
const OTEL_SAMPLE_RATE::Float64 = parse(Float64, get(ENV, "OTEL_SAMPLE_RATE", "1.0"))
const OTEL_SAMPLING_STRATEGIES = Dict(
    :always_sample => ["/metrics", "/health"],
    :probabilistic => ["/api/v1/detect"],
    :never_sample => ["/favicon.ico"]
)
const OTEL_SPAN_LIMITS = Dict(
    :max_attributes_per_span => 1000,
    :max_events_per_span => 100,
    :max_links_per_span => 100,
    :max_attribute_length => 4096
)
const METRICS_ENABLED::Bool = true
const METRICS_NAMESPACE::String = "bpla"
const METRICS_SUBSYSTEM::String = "detection"
const LATENCY_BUCKETS::Vector{Float64} = [0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0, 15.0, 30.0, 60.0]
const SIZE_BUCKETS::Vector{Float64} = [1024, 5120, 10240, 51200, 102400, 512000, 1_048_576, 5_242_880, 10_485_760]
const CUSTOM_METRICS = Dict(
    :inference_latency => (:histogram, "Inference latency in seconds", LATENCY_BUCKETS),
    :batch_size => (:histogram, "Batch size distribution", [1, 2, 4, 8, 16, 32, 64]),
    :cache_hit_ratio => (:gauge, "Cache hit ratio", nothing),
    :queue_depth => (:gauge, "Message queue depth", nothing),
    :active_connections => (:gauge, "Active WebSocket connections", nothing),
    :model_load_time => (:histogram, "Model loading time", LATENCY_BUCKETS),
    :gpu_memory_used => (:gauge, "GPU memory usage in MB", nothing),
    :gpu_utilization => (:gauge, "GPU utilization percentage", nothing)
)
const LOG_LEVEL::String = get(ENV, "LOG_LEVEL", "INFO")
const LOG_FORMAT::String = get(ENV, "LOG_FORMAT", "json")
const LOG_OUTPUT::String = get(ENV, "LOG_OUTPUT", "stdout")
const LOG_SAMPLE_RATE::Float64 = parse(Float64, get(ENV, "LOG_SAMPLE_RATE", "1.0"))
const LOG_SLOW_REQUEST_THRESHOLD::Int = 1000
const LOG_REDACT_FIELDS::Vector{String} = ["password", "token", "api_key", "authorization", "cookie"]
const MODEL_PATH::String = get(ENV, "MODEL_PATH", "/models/best_model.bson")
const MODEL_CONFIG_PATH::String = get(ENV, "MODEL_CONFIG_PATH", "/models/config.json")
const MODEL_CACHE_PATH::String = get(ENV, "MODEL_CACHE_PATH", "/cache/models")
const MODEL_LOAD_TIMEOUT::Int = parse(Int, get(ENV, "MODEL_LOAD_TIMEOUT", "300"))
const MODEL_WARMUP_ENABLED::Bool = true
const MODEL_WARMUP_ITERATIONS::Int = 10
const MODEL_WARMUP_BATCH_SIZE::Int = 4
const MODEL_BATCH_SIZE::Int = parse(Int, get(ENV, "BATCH_SIZE", "1"))
const MODEL_MAX_BATCH_SIZE::Int = parse(Int, get(ENV, "MAX_BATCH_SIZE", "32"))
const MODEL_BATCH_TIMEOUT::Int = parse(Int, get(ENV, "BATCH_TIMEOUT", "100"))
const MODEL_QUEUE_SIZE::Int = parse(Int, get(ENV, "QUEUE_SIZE", "1000"))
const USE_GPU::Bool = false
const GPU_MEMORY_LIMIT::Float64 = parse(Float64, get(ENV, "GPU_MEMORY_LIMIT", "0.8"))
const GPU_MEMORY_RESERVE::Float64 = parse(Float64, get(ENV, "GPU_MEMORY_RESERVE", "0.1"))
const SERVING_MODE::String = get(ENV, "SERVING_MODE", "default")
const NR_REPLICAS::Int = parse(Int, get(ENV, "REPLICAS", "3"))
const MIN_REPLICAS::Int = parse(Int, get(ENV, "MIN_REPLICAS", "1"))
const MAX_REPLICAS::Int = parse(Int, get(ENV, "MAX_REPLICAS", "10"))
const HPA_ENABLED::Bool = true
const HPA_GPU_THRESHOLD::Int = parse(Int, get(ENV, "HPA_GPU_THRESHOLD", "70"))
const HPA_MEMORY_THRESHOLD::Int = parse(Int, get(ENV, "HPA_MEMORY_THRESHOLD", "80"))
const HPA_CUSTOM_METRICS::Vector{String} = ["requests_per_second", "queue_length"]
const VPA_ENABLED::Bool = true
const VPA_UPDATE_MODE::String = get(ENV, "VPA_UPDATE_MODE", "auto")
const VPA_MIN_CPU::String = get(ENV, "VPA_MIN_CPU", "100m")
const VPA_MAX_CPU::String = get(ENV, "VPA_MAX_CPU", "4")
const VPA_MIN_MEMORY::String = get(ENV, "VPA_MIN_MEMORY", "512Mi")
const VPA_MAX_MEMORY::String = get(ENV, "VPA_MAX_MEMORY", "8Gi")
const PDB_ENABLED::Bool = true
const PDB_MIN_AVAILABLE::Int = 2
const PDB_MAX_UNAVAILABLE::Int = 1
const LIVENESS_PROBE = Dict(
    :path => "/healthz",
    :initial_delay => 30,
    :period => 10,
    :timeout => 5,
    :failure_threshold => 3,
    :success_threshold => 1
)
const READINESS_PROBE = Dict(
    :path => "/ready",
    :initial_delay => 5,
    :period => 5,
    :timeout => 3,
    :failure_threshold => 2,
    :success_threshold => 1
)
const STARTUP_PROBE = Dict(
    :path => "/startup",
    :initial_delay => 0,
    :period => 2,
    :timeout => 2,
    :failure_threshold => 30
)
const SHUTDOWN_TIMEOUT::Int = parse(Int, get(ENV, "SHUTDOWN_TIMEOUT", "30"))
const SHUTDOWN_GRACE_PERIOD::Int = parse(Int, get(ENV, "SHUTDOWN_GRACE_PERIOD", "5"))
const SHUTDOWN_DRAIN_CONNECTIONS::Bool = true
const SHUTDOWN_WAIT_FOR_REQUESTS::Bool = true
const SHUTDOWN_MAX_PENDING::Int = parse(Int, get(ENV, "SHUTDOWN_MAX_PENDING", "100"))
const TLS_MIN_VERSION::String = "TLSv1.2"
const TLS_MAX_VERSION::String = "TLSv1.3"
const TLS_ENABLED::Bool = parse(Bool, get(ENV, "TLS_ENABLED", "true"))
const TLS_CERT_PATH::String = get(ENV, "TLS_CERT_PATH", "/certs/tls.crt")
const TLS_KEY_PATH::String = get(ENV, "TLS_KEY_PATH", "/certs/tls.key")
const TLS_CA_PATH::String = get(ENV, "TLS_CA_PATH", "/certs/ca.crt")
const TLS_CIPHER_SUITES::Vector{String} = ["TLS_AES_256_GCM_SHA384", "TLS_CHACHA20_POLY1305_SHA256", "TLS_AES_128_GCM_SHA256"]
const AUTH_ENABLED::Bool = parse(Bool, get(ENV, "AUTH_ENABLED", "false"))
const AUTH_TYPE::String = get(ENV, "AUTH_TYPE", "jwt")
const JWT_SECRET::String = get(ENV, "JWT_SECRET", "")
const JWT_EXPIRY::Int = parse(Int, get(ENV, "JWT_EXPIRY", "3600"))
const CORS_ENABLED::Bool = true
const CORS_ALLOW_ORIGINS::Vector{String} = ["*"]
const CORS_ALLOW_METHODS::Vector{String} = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
const CORS_ALLOW_HEADERS::Vector{String} = ["Content-Type", "Authorization", "X-Request-ID"]
const CORS_EXPOSE_HEADERS::Vector{String} = ["X-RateLimit-Limit", "X-RateLimit-Remaining"]
const CORS_MAX_AGE::Int = 3600
const AUDIT_ENABLED::Bool = true
const AUDIT_LOG_PATH::String = "/var/log/bpla/audit.log"
const AUDIT_RETENTION_DAYS::Int = 90
const DATA_SOVEREIGNTY_ENABLED::Bool = true
const DATA_STORAGE_REGIONS::Vector{String} = ["us-east-1", "eu-west-1"]
const DATA_ANONYMIZE_PII::Bool = true

const FEATURE_FLAGS = Dict(
    :batch_inference => true,
    :streaming_results => true,
    :model_ab_testing => false,
    :canary_deployment => true,
    :blue_green_deployment => true,
    :shadow_traffic => false,
    :request_logging => true,
    :debug_endpoints => (ENVIRONMENT == "development"),
    :profile_endpoints => (ENVIRONMENT == "development"),
    :experimental_features => false
)

const AB_TEST_CONFIG = Dict(
    :enabled => false,
    :models => ["model_v1.bson", "model_v2.bson"],
    :traffic_split => [0.9, 0.1],
    :metrics_tracking => true
)

const RESOURCE_LIMITS = Dict(
    :gpu => "4",
    :memory => "8Gi",
    :ephemeral_storage => "10Gi",
    :pids => "1000"
)

const RESOURCE_REQUESTS = Dict(
    :gpu => "500m",
    :memory => "1Gi",
    :ephemeral_storage => "1Gi"
)

Base.@kwdef mutable struct AppConfig
    service_name::String = SERVICE_NAME
    service_version::String = SERVICE_VERSION
    service_uuid::String = SERVICE_UUID
    environment::String = ENVIRONMENT
    region::String = REGION
    zone::String = ZONE
    cluster::String = CLUSTER_NAME
    api_host::String = API_HOST
    api_port::Int = API_PORT
    metrics_port::Int = METRICS_PORT
    health_port::Int = HEALTH_PORT
    workers::Int = WORKERS
    worker_type::String = WORKER_TYPE
    rate_limit_enabled::Bool = RATE_LIMIT_ENABLED
    rate_limit_algorithm::String = RATE_LIMIT_ALGORITHM
    rate_limit_requests::Int = RATE_LIMIT_REQUESTS
    rate_limit_window::Int = RATE_LIMIT_WINDOW
    rate_limit_burst::Int = RATE_LIMIT_BURST
    rate_limits::Dict = RATE_LIMITS
    circuit_breaker_enabled::Bool = CIRCUIT_BREAKER_ENABLED
    circuit_breaker_failure_limit::Int = CIRCUIT_BREAKER_FAILURE_LIMIT
    circuit_breaker_recovery_timeout::Int = CIRCUIT_BREAKER_RECOVERY_TIMEOUT
    redis_mode::String = REDIS_MODE
    redis_host::String = REDIS_HOST
    redis_port::Int = REDIS_PORT
    redis_password::String = REDIS_PASSWORD
    redis_sentinel_hosts::Vector{String} = REDIS_SENTINEL_HOSTS
    redis_cluster_nodes::Vector = REDIS_CLUSTER_NODES
    cache_ttl::Dict = CACHE_TTL
    cache_max_memory::String = CACHE_MAX_MEMORY
    cache_eviction_policy::String = CACHE_EVICTION_POLICY
    mq_type::String = MQ_TYPE
    mq_host::String = MQ_HOST
    mq_port::Int = MQ_PORT
    rabbitmq_vhost::String = RABBITMQ_VHOST
    rabbitmq_user::String = RABBITMQ_USER
    rabbitmq_password::String = RABBITMQ_PASSWORD
    kafka_brokers::Vector{String} = KAFKA_BROKERS
    queues::Dict = QUEUES
    prefetch_count::Int = PREFETCH_COUNT
    retry_policy::Dict = RETRY_POLICY
    otel_enabled::Bool = OTEL_ENABLED
    otel_service_name::String = OTEL_SERVICE_NAME
    otel_exporter_endpoint::String = OTEL_EXPORTER_OTLP_ENDPOINT
    otel_sample_rate::Float64 = OTEL_SAMPLE_RATE
    otel_span_limits::Dict = OTEL_SPAN_LIMITS
    metrics_enabled::Bool = METRICS_ENABLED
    metrics_namespace::String = METRICS_NAMESPACE
    metrics_subsystem::String = METRICS_SUBSYSTEM
    latency_buckets::Vector{Float64} = LATENCY_BUCKETS
    custom_metrics::Dict = CUSTOM_METRICS
    log_level::String = LOG_LEVEL
    log_format::String = LOG_FORMAT
    log_output::String = LOG_OUTPUT
    log_sample_rate::Float64 = LOG_SAMPLE_RATE
    log_redact_fields::Vector{String} = LOG_REDACT_FIELDS
    model_path::String = MODEL_PATH
    model_config_path::String = MODEL_CONFIG_PATH
    model_cache_path::String = MODEL_CACHE_PATH
    model_load_timeout::Int = MODEL_LOAD_TIMEOUT
    model_warmup_enabled::Bool = MODEL_WARMUP_ENABLED
    model_batch_size::Int = MODEL_BATCH_SIZE
    model_max_batch_size::Int = MODEL_MAX_BATCH_SIZE
    model_batch_timeout::Int = MODEL_BATCH_TIMEOUT
    model_queue_size::Int = MODEL_QUEUE_SIZE
    use_gpu::Bool = USE_GPU
    gpu_memory_limit::Float64 = GPU_MEMORY_LIMIT
    serving_mode::String = SERVING_MODE
    nr_replicas::Int = NR_REPLICAS
    min_replicas::Int = MIN_REPLICAS
    max_replicas::Int = MAX_REPLICAS
    hpa_enabled::Bool = HPA_ENABLED
    hpa_gpu_threshold::Int = HPA_GPU_THRESHOLD
    hpa_memory_threshold::Int = HPA_MEMORY_THRESHOLD
    vpa_enabled::Bool = VPA_ENABLED
    pdb_enabled::Bool = PDB_ENABLED
    liveness_probe::Dict = LIVENESS_PROBE
    readiness_probe::Dict = READINESS_PROBE
    startup_probe::Dict = STARTUP_PROBE
    shutdown_timeout::Int = SHUTDOWN_TIMEOUT
    shutdown_grace_period::Int = SHUTDOWN_GRACE_PERIOD
    shutdown_drain_connections::Bool = SHUTDOWN_DRAIN_CONNECTIONS
    tls_enabled::Bool = TLS_ENABLED
    tls_cert_path::String = TLS_CERT_PATH
    tls_key_path::String = TLS_KEY_PATH
    auth_enabled::Bool = AUTH_ENABLED
    auth_type::String = AUTH_TYPE
    cors_enabled::Bool = CORS_ENABLED
    cors_allow_origins::Vector{String} = CORS_ALLOW_ORIGINS
    audit_enabled::Bool = AUDIT_ENABLED
    data_sovereignty_enabled::Bool = DATA_SOVEREIGNTY_ENABLED
    feature_flags::Dict = FEATURE_FLAGS
    ab_test_config::Dict = AB_TEST_CONFIG
    resource_limits::Dict = RESOURCE_LIMITS
    resource_requests::Dict = RESOURCE_REQUESTS
    fps_y_offset::Int = 30
    people_count_y_offset::Int = 60
    text_x_offset::Int = 50
    font_scale::Float64 = 0.8
    font_thickness::Int = 2
    label_font_scale::Float64 = 0.7
    label_text_thickness::Int = 3
    rectangle_thickness::Int = 5
    video_codec::Int = 0x7634706D
    batch_size::Int = 16
    resize_to::Int = 640
    num_epochs::Int = 100
    num_workers::Int = 4
    learning_rate::Float64 = 0.0001
    weight_decay::Float64 = 1e-5
    momentum::Float64 = 0.9
    device::String = "gpu"
    base_dir::String = pwd()
    data_dir::String = "train_project/datasets"
    train_dir::String = "datasets/csmr_multimodal/images/train/visible"
    valid_dir::String = "datasets/csmr_multimodal/images/val/visible"
    test_dir::String  = "datasets/csmr_multimodal/images/val/visible"
    classes::Vector{String} = ["background", "pedestrian"]
    num_classes::Int = 2
    visualize_transformed::Bool = true
    out_dir::String = "outputs"
    data_augmentation::Bool = true
    normalize_images::Bool = true
    early_stopping_patience::Int = 10
    early_stopping_min_delta::Float64 = 0.001
    resize_dim_min::Int = 100
    resize_dim_max::Int = 1024
    resize_dim_default::Int = 640
    threshold_min::Float64 = 0.0
    threshold_max::Float64 = 1.0
    threshold_default::Float64 = 0.5
    version::String = "1.0.0"
end

function generate_colors(n::Int)::Vector{NTuple{3, Int}}
    base_colors = [
        (128, 128, 128),
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
        (128, 0, 128),
        (128, 128, 0),
        (0, 128, 128),
    ]
    
    return [base_colors[mod1(i, length(base_colors))] for i in 1:n]
end

function get_class_color(class_id::Int, config::AppConfig)::NTuple{3, Int}
    colors = generate_colors(config.num_classes)
    return colors[mod1(class_id + 1, length(colors))]
end

function fps_position(width::Int, config::AppConfig)::Tuple{Int, Int}
    text_x = div(width, 2) - 50
    return (text_x, config.fps_y_offset)
end

function people_count_position(width::Int, config::AppConfig)::Tuple{Int, Int}
    text_x = div(width, 2) - 50
    return (text_x, config.people_count_y_offset)
end

function label_position(x::Int, y::Int, config::AppConfig)::Tuple{Int, Int}
    return (x + config.text_x_offset, max(y - 10, 10))
end

function to_dict(config::AppConfig)::Dict{Symbol, Any}
    return Dict{Symbol, Any}(
        :fps_y_offset => config.fps_y_offset,
        :people_count_y_offset => config.people_count_y_offset,
        :text_x_offset => config.text_x_offset,
        :font_scale => config.font_scale,
        :font_thickness => config.font_thickness,
        :label_font_scale => config.label_font_scale,
        :label_text_thickness => config.label_text_thickness,
        :rectangle_thickness => config.rectangle_thickness,
        :batch_size => config.batch_size,
        :resize_to => config.resize_to,
        :num_epochs => config.num_epochs,
        :num_workers => config.num_workers,
        :learning_rate => config.learning_rate,
        :weight_decay => config.weight_decay,
        :momentum => config.momentum,
        :device => config.device,
        :classes => config.classes,
        :num_classes => config.num_classes,
        :version => config.version,
        :data_augmentation => config.data_augmentation,
        :normalize_images => config.normalize_images,
        :early_stopping_patience => config.early_stopping_patience,
        :early_stopping_min_delta => config.early_stopping_min_delta
    )
end

function from_dict(dict::Dict)::AppConfig
    config = AppConfig()
    
    for (key, value) in dict
        if hasproperty(config, key)
            setproperty!(config, key, value)
        end
    end
    
    return config
end

function update_config(config::AppConfig; kwargs...)::AppConfig
    new_config = deepcopy(config)
    
    for (key, value) in kwargs
        if hasproperty(new_config, key)
            setproperty!(new_config, key, value)
        else
            @warn "Unknown configuration key: $key"
        end
    end
    
    return new_config
end

function default_config()::AppConfig
    return AppConfig()
end

function fast_config()::AppConfig
    return AppConfig(
        batch_size=32,
        resize_to=416,
        num_epochs=50,
        learning_rate=0.001,
        data_augmentation=false,
        early_stopping_patience=5
    )
end

function accurate_config()::AppConfig
    return AppConfig(
        batch_size=8,
        resize_to=640,
        num_epochs=200,
        learning_rate=0.0001,
        data_augmentation=true,
        early_stopping_patience=15,
        threshold_default=0.3
    )
end

function production_config()::AppConfig
    return AppConfig(
        batch_size=16,
        resize_to=640,
        device="gpu",
        visualize_transformed=false,
        out_dir="/var/log/detections",
        threshold_default=0.7
    )
end

function as_dict(config::AppConfig)::OrderedDict{Symbol, Any}
    return OrderedDict{Symbol, Any}(
        :service_name => config.service_name,
        :service_version => config.service_version,
        :environment => config.environment,
        :region => config.region,
        :api_host => config.api_host,
        :api_port => config.api_port,
        :metrics_port => config.metrics_port,
        :workers => config.workers,
        :rate_limit_enabled => config.rate_limit_enabled,
        :redis_mode => config.redis_mode,
        :mq_type => config.mq_type,
        :otel_enabled => config.otel_enabled,
        :metrics_enabled => config.metrics_enabled,
        :log_level => config.log_level,
        :model_path => config.model_path,
        :use_gpu => config.use_gpu,
        :nr_replicas => config.nr_replicas,
        :tls_enabled => config.tls_enabled,
        :feature_flags => config.feature_flags,
        :version => SERVICE_VERSION
    )
end

function to_json(config::AppConfig)::String
    return JSON.pretty(as_dict(config))
end

function load_from_env()::AppConfig
    return AppConfig()
end

function validate_config(config::AppConfig)::Bool
    if isempty(config.service_name)
        @error "service_name is required"
        return false
    end
    
    if config.api_port < 1024 || config.api_port > 65535
        @error "api_port must be between 1024 and 65535"
        return false
    end
    
    if config.rate_limit_enabled && config.rate_limit_requests <= 0
        @error "rate_limit_requests must be positive"
        return false
    end
    
    for (key, value) in config.cache_ttl
        if value <= 0
            @error "cache_ttl[$key] must be positive"
            return false
        end
    end
    
    return true
end

function dev_config()::AppConfig
    return AppConfig(
        environment = "development",
        log_level = "DEBUG",
        log_format = "text",
        rate_limit_enabled = false,
        tls_enabled = false,
        auth_enabled = false,
        nr_replicas = 1,
        feature_flags = merge(FEATURE_FLAGS, Dict(
            :debug_endpoints => true,
            :profile_endpoints => true
        ))
    )
end

function staging_config()::AppConfig
    return AppConfig(
        environment = "staging",
        log_level = "DEBUG",
        nr_replicas = 2,
        feature_flags = merge(FEATURE_FLAGS, Dict(
            :canary_deployment => true,
            :shadow_traffic => true
        ))
    )
end

function prod_config()::AppConfig
    return AppConfig(
        environment = "production",
        log_level = "INFO",
        log_format = "json",
        rate_limit_enabled = true,
        tls_enabled = true,
        auth_enabled = true,
        nr_replicas = 5,
        hpa_enabled = true,
        vpa_enabled = true,
        pdb_enabled = true
    )
end

export
    generate_colors,
    get_class_color,
    fps_position,
    people_count_position,
    label_position,
    to_dict,
    from_dict,
    update_config,
    validate_config,
    default_config,
    fast_config,
    accurate_config,
    production_config
    AppConfig,
    as_dict,
    to_json,
    load_from_env,
    dev_config,
    staging_config,
    prod_config,
    SERVICE_NAME,
    SERVICE_VERSION,
    ENVIRONMENT,
    BUILD_TIMESTAMP

end  # module Config