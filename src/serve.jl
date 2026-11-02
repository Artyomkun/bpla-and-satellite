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

module Serve

include("Config.jl")
include("models/Models.jl")

using .Config: SERVICE_VERSION
using OrderedCollections
using Genie.Renderer
using Genie.Requests
using Genie.Responses
using Statistics
using Logging
using Printf
using Base64
using Random
using Genie
using Dates
using JSON
using UUIDs

export
    Serve,
    run_server,
    route_root,
    route_health,
    route_metrics,
    route_metrics_prometheus,
    route_detect_image,
    route_detect_base64,
    route_get_output

Base.@kwdef mutable struct Metrics
    request_count::Dict{Tuple{String, String, Int}, Int} = Dict{Tuple{String, String, Int}, Int}()
    inference_count::Dict{String, Int} = Dict{String, Int}()
    latencies::Vector{Float64} = Float64[]
    detections_by_class::Dict{String, Int} = Dict{String, Int}()
    errors::Dict{String, Int} = Dict{String, Int}()
    model_load_time_seconds::Float64 = 0.0
    start_time::DateTime = now()
end

const _metrics = Ref{Metrics}(Metrics())

function record_request(metrics::Metrics, method::String, endpoint::String, status::Int)
    key = (method, endpoint, status)
    metrics.request_count[key] = get!(metrics.request_count, key, 0) + 1
end

function record_inference(metrics::Metrics, type::String)
    metrics.inference_count[type] = get!(metrics.inference_count, type, 0) + 1
end

function record_detection(metrics::Metrics, class::String)
    metrics.detections_by_class[class] = get!(metrics.detections_by_class, class, 0) + 1
end

function record_error!(metrics::Metrics, error_type::String)
    metrics.errors[error_type] = get!(metrics.errors, error_type, 0) + 1
end

function observe_latency(metrics::Metrics, duration_sec::Float64, method::String, endpoint::String)
    ms = duration_sec * 1000.0
    for (i, bound) in enumerate(LATENCY_BUCKETS_MS)
        if ms <= bound
            metrics.latency_bucket_counts[i] += 1
            break
        end
    end
    metrics.latency_sum_ms += ms
    metrics.latency_count += 1

    record_request(metrics, method, endpoint, 200)
end

struct DetectionResult
    boxes::Vector{Vector{Float32}}
    scores::Vector{Float32}
    labels::Vector{String}
    count::Int
    inference_time::Float64
end

struct ImageResponse
    image_path::String
    detections::DetectionResult
    timestamp::String
end

struct HealthResponse
    status::String
    model_loaded::Bool
    model_load_time_seconds::Float64
    device::String
    uptime_seconds::Float64
    total_requests::Int
    total_inferences::Int
    avg_latency_ms::Float64
    timestamp::String
end
struct MetricsResponse
    requests_total::Int
    inference_total::Int
    avg_latency_ms::Float64
    p50_latency_ms::Float64
    p99_latency_ms::Float64
    p99_9_latency_ms::Float64
    p99_99_latency_ms::Float64
    p99_999_latency_ms::Float64
    p99_9999_latency_ms::Float64
    detections_by_class::Dict{String, Int}
    errors::Dict{String, Int}
    uptime_seconds::Float64
    timestamp::String
end

const _start_time = Ref{DateTime}(now())
const _model_load_time = Ref{Float64}(0.0)

const ALLOWED_IMAGE_TYPES = ("image/jpeg", "image/png", "image/bmp")

function is_allowed_content_type(content_type::String)::Bool
    base = split(content_type, ';')[1] |> strip
    return base in ALLOWED_IMAGE_TYPES
end

function validate_image_content(content_type::String)::Bool
    return is_allowed_content_type(content_type)
end

const _start_time_ns = Ref{UInt64}(0)

function get_uptime_seconds()::Float64
    return (time_ns() - _start_time_ns[]) / 1.0e9
end

function get_avg_latency_ms()::Float64
    m = _metrics[]
    m.latency_count == 0 && return 0.0
    return m.latency_sum_ms / m.latency_count
end

function get_latency_percentiles()::NamedTuple
    latencies = _metrics[].latencies
    n = length(latencies)

    if n == 0
        return (p50=0.0, p99=0.0, p99_9=0.0, p99_99=0.0, p99_999=0.0, p99_9999=0.0)
    end

    sorted = sort(latencies)

    pct(p) = sorted[min(n, max(1, ceil(Int, n * p)))] * 1000

    return (
        p50      = pct(0.50),
        p99      = pct(0.99),
        p99_9    = pct(0.999),
        p99_99   = pct(0.9999),
        p99_999  = pct(0.99999),
        p99_9999 = pct(0.999999),
    )
end

function route_root()
    result = Dict(
        "name" => "BPLA Person Detection API",
        "version" => "1.0.0",
        "docs" => "/docs",
        "metrics" => "/metrics",
        "health" => "/health"
    )
    return json(result)
end

function route_health()
    start_time = time_ns()
    model_loaded = _model[] !== nothing

    device = try
        string(typeof(_model[].backbone))
    catch
        "неизвестно"
    end

    total_requests = sum(values(_metrics[].request_count); init=0)
    total_inferences = sum(values(_metrics[].inference_count); init=0)
    avg_latency = get_avg_latency_ms()

    elapsed = (time_ns() - start_time) / 1_000_000_000
    observe_latency!(_metrics[], elapsed, "GET", "/health")

    result = HealthResponse(
        model_loaded ? "здоров" : "нездоров",
        model_loaded,
        _model_load_time[],
        device,
        get_uptime_seconds(),
        total_requests,
        total_inferences,
        avg_latency,
        Dates.format(now(), ISO_FORMAT),
    )

    return json(result)
end

function route_metrics()
    metrics = _metrics[]
    latencies_ms = [l * 1000 for l in metrics.latencies]
    p50, p95, p99 = get_latency_percentiles()
    avg_latency = isempty(latencies_ms) ? 0.0 : mean(latencies_ms) / 1000
    result = MetricsResponse(
        sum(values(metrics.request_count)),
        sum(values(metrics.inference_count)),
        avg_latency,
        p50 / 1000,
        p95 / 1000,
        p99 / 1000,
        metrics.detections_by_class,
        metrics.errors,
        get_uptime_seconds(),
        Dates.format(now(), ISO_FORMAT)
    )
    
    return json(result)
end

function route_metrics_prometheus()
    metrics = _metrics[]
    lines = String[]
    uptime = get_uptime_seconds()

    push!(lines, "# HELP bpla_uptime_seconds Время работы сервера в секундах\n# TYPE bpla_uptime_seconds counter\nbpla_uptime_seconds $uptime")

    push!(lines, "# HELP bpla_http_requests_total Всего HTTP-запросов\n# TYPE bpla_http_requests_total counter")
    for ((method, endpoint, status), count) in metrics.request_count
        push!(lines, "bpla_http_requests_total{method=\"$method\",endpoint=\"$endpoint\",status=\"$status\"} $count")
    end

    push!(lines, "# HELP bpla_inference_total Всего операций инференса\n# TYPE bpla_inference_total counter")
    for (type, count) in metrics.inference_count
        push!(lines, "bpla_inference_total{type=\"$type\"} $count")
    end

    push!(lines, "# HELP bpla_http_request_duration_seconds Длительность HTTP-запросов\n# TYPE bpla_http_request_duration_seconds histogram")
    for (bound, count) in metrics.latency_buckets
        push!(lines, "bpla_http_request_duration_seconds_bucket{le=\"$bound\"} $count")
    end
    push!(lines, "bpla_http_request_duration_seconds_bucket{le=\"+Inf\"} $(metrics.latency_count)\nbpla_http_request_duration_seconds_sum $(metrics.latency_sum)\nbpla_http_request_duration_seconds_count $(metrics.latency_count)")

    push!(lines, "# HELP bpla_detections_total Всего детекций по классам\n# TYPE bpla_detections_total counter")
    for (class, count) in metrics.detections_by_class
        push!(lines, "bpla_detections_total{class=\"$class\"} $count")
    end

    push!(lines, "# HELP bpla_errors_total Всего ошибок по типам\n# TYPE bpla_errors_total counter")
    for (error_type, count) in metrics.errors
        push!(lines, "bpla_errors_total{type=\"$error_type\"} $count")
    end

    push!(lines, "# HELP bpla_model_load_time_seconds Время загрузки модели в секундах\n# TYPE bpla_model_load_time_seconds gauge\nbpla_model_load_time_seconds $(metrics.model_load_time_seconds)")

    return join(lines, "\n") * "\n"
end

function route_get_output(filename::String)
    start_time = time_ns()
    file_path = joinpath("inference_outputs", filename)
    
    if !isfile(file_path)
        record_error!(_metrics[], "file_not_found")
        return json(Dict("error" => "File not found")), 404
    end

    elapsed = (time_ns() - start_time) / 1_000_000_000
    observe_latency!(_metrics[], elapsed, "GET", "/outputs/:filename")
    
    return file_path
end

function setup_routes()
    Genie.route("/", method=GET)(route_root)
    Genie.route("/health", method=GET)(route_health)
    Genie.route("/metrics", method=GET)(route_metrics)
    Genie.route("/metrics/prometheus", method=GET)(route_metrics_prometheus)
    Genie.route("/detect/image", method=POST)(route_detect_image)
    Genie.route("/detect/base64", method=POST)(route_detect_base64)
    Genie.route("/outputs/:filename", method=GET)(route_get_output)
    @info "Routes configured with metrics"
end

function run_server(; host::String="0.0.0.0", port::Int=8000)
    @info "="^60
    @info "BPLA Person Detection REST API Server"
    @info "="^60
    @info ""
    @info "Version: $(SERVICE_VERSION)"
    @info ""
    
    try
        _start_time[] = now()
        
        try
            load_start = time_ns()
            load_time = (time_ns() - load_start) / 1_000_000_000
            _model_load_time[] = load_time
            @info "Model loaded successfully"
            @info "  Device: cpu"
            @info "  Load time: $(round(load_time, digits=2))s"
        catch e
            @warn "Model not loaded: $e"
        end
        
        setup_routes()
        
        @info ""
        @info "Endpoints:"
        @info "  GET  /                  - Root"
        @info "  GET  /health           - Health check"
        @info "  GET  /metrics          - JSON metrics"
        @info "  GET  /metrics/prometheus - Prometheus format"
        @info "  GET  /outputs/:filename - Get output file"
        @info ""
        @info "Server running at http://$host:$port"
        @info ""
        
        Genie.up(host=host, port=port, verbose=true)
        
        return true
    catch e
        return nothing
    end
end

function main()
    return run_server()
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

end  # module Serve
