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

include("../Config.jl")
include("../models/Models.jl")

using Base: invokelatest
using OrderedCollections
using Genie.Responses
using Genie.Renderer
using Genie.Requests
using Serialization
using RedisCache
using Logging
using ImageIO
using FileIO
using Printf
using Random
using Base64
using Images
using LibPQ
using Dates
using Genie
using JSON
using UUIDs

struct DetectionResult
    boxes::Vector{Vector{Float32}}
    scores::Vector{Float32}
    labels::Vector{String}
    count::Int
    inference_time::Float64
end

struct ImageResponse
    request_id::String
    image_path::Union{String, Nothing}
    detections::DetectionResult
    cached::Bool
    timestamp::String
end

struct HealthResponse
    status::String
    model_loaded::Bool
    redis_connected::Bool
    rabbitmq_connected::Bool
    replicas::Int
    uptime_seconds::Float64
    timestamp::String
end

struct QueueStatsResponse
    inference_queue_size::Int
    results_queue_size::Int
    dlq_size::Int
end

const _start_time = Ref{DateTime}(now())
const _ready = Ref{Bool}(false)
const _shutdown_event = Ref{Bool}(false)

struct Cache
    data::Dict{String, Any}
    hits::Int
    misses::Int
end

Cache() = Cache(Dict{String, Any}(), 0, 0)

const _cache = Ref{Cache}(Cache())

async_get(cache::Cache, key::String) = haskey(cache.data, key) ? (cache.hits += 1; cache.data[key]) : (cache.misses += 1; nothing)
async_set(cache::Cache, key::String, value::Any, ttl::Int=300) = cache.data[key] = value

hit_rate(cache::Cache) = cache.hits + cache.misses > 0 ? cache.hits / (cache.hits + cache.misses) : 0.0

struct MessageQueue
    results_queue::Vector{Dict{String, Any}}
    dlq::Vector{Dict{String, Any}}
end

MessageQueue() = MessageQueue(Vector{Dict{String, Any}}(), Vector{Dict{String, Any}}(), Vector{Dict{String, Any}}())

const _mq = Ref{MessageQueue}(MessageQueue())

struct Metrics
    request_count::Dict{Tuple{String, String, Int}, Int}
    inference_count::Dict{String, Int}
    latencies::Vector{Float64}
end

Metrics() = Metrics(Dict{Tuple{String, String, Int}, Int}(), Dict{String, Int}(), Float64[])

const metrics = Ref{Metrics}(Metrics())

record_request(metrics::Metrics, method::String, endpoint::String, status::Int) = metrics.x.request_count[(method, endpoint, status)] = get(metrics.x.request_count, (method, endpoint, status), 0) + 1

function observe_latency(metrics::Metrics, duration::Float64, method::String, endpoint::String)
    push!(metrics.x.latencies, duration)
    record_request(metrics, method, endpoint, 200)
    return nothing
end

function check_readiness_async()
    result = Channel{Bool}(1)
    
    @async begin
        _ready[] = false
        
        checks = Dict{String, Bool}(
            "model" => false,
            "redis" => true,
            "rabbitmq" => true,
        )
        
        try
            checks["model"] = true
        catch e
            @warn "Model check failed: $e"
        end
        
        _ready[] = all(values(checks))
        put!(result, _ready[])
    end
    
    return result
end

function health_check_loop()
    while !_shutdown_event[]
        try
            check_health()
        catch e
            @error "Health check failed" exception=e
        end
        sleep(Config.HEALTH_CHECK_INTERVAL)
    end
end

is_allowed_content_type(content_type::String) = content_type in ["image/jpeg", "image/png", "image/bmp"]

function image_to_base64(img::Matrix{<:Colorant})::String
    temp_file = tempname() * ".jpg"
    
    try
        save(temp_file, img)
        bytes = read(temp_file)
        base64_str = Base64.base64encode(bytes)
        rm(temp_file)
        return base64_str
    catch e
        if isfile(temp_file)
            rm(temp_file)
        end
        rethrow()
    end
end

function base64_to_image_batch(data_list::Vector{String})::Vector{Matrix{RGB{N0f8}}}
    results = Matrix{RGB{N0f8}}[]
    
    for (data) in enumerate(data_list)
        push!(results, base64_to_image(data))
    end
    
    return results
end

function route_root()
    result = Dict(
        "name" => "BPLA Person Detection API",
        "version" => Config.VERSION,
        "environment" => Config.ENVIRONMENT,
        "docs" => "/docs"
    )
    return json(result)
end

function route_health()
    uptime = (now() - _start_time[]).value / 1000.0
    status = _ready[] ? "healthy" : "degraded"
    result = HealthResponse(
        status,
        true,
        true,
        true,
        Config.NR_REPLICAS,
        uptime,
        Dates.format(now(), ISO_FORMAT)
    )
    
    return json(result)
end

function route_ready()
    if !_ready[]
        return json(Dict("error" => "Service not ready")), 503
    end
    return json(Dict("status" => "ready"))
end

function route_live()
    return json(Dict("status" => "alive"))
end

function route_detect_image()
    request_id = string(UUIDs.uuid4())
    
    try
        files = rawpayload(:files, Dict{String, Any})
        
        if !haskey(files, "file")
            @warn "[$request_id] No file provided"
            return json(Dict(
                "request_id" => request_id,
                "error" => "No file provided",
                "success" => false
            )), 400
        end
        
        file = files["file"]
        content_type = get(file, "content_type", "")
        
        if !is_allowed_content_type(content_type)
            allowed_formats = ["image/jpeg", "image/png", "image/bmp", "image/jpg"]
            @warn "[$request_id] Unsupported format: $content_type"
            return json(Dict(
                "request_id" => request_id,
                "error" => "Unsupported format. Allowed: $(join(allowed_formats, ", "))",
                "success" => false
            )), 400
        end
        
        post_data = rawpayload(:post, Dict{String, Any})
        threshold = haskey(post_data, "threshold") ? parse(Float64, post_data["threshold"]) : 0.5
        resize_dim = haskey(post_data, "resize_dim") ? parse(Int, post_data["resize_dim"]) : 640
        return_image = haskey(post_data, "return_image") ? parse(Bool, post_data["return_image"]) : true
        save_result = haskey(post_data, "save") ? parse(Bool, post_data["save"]) : true
        
        img = if haskey(file, "data")
            img_data = file["data"]
            base64_to_image(img_data)
        elseif haskey(file, "tempfile")
            temp_path = file["tempfile"]
            load(temp_path)
        else
            return json(Dict(
                "request_id" => request_id,
                "error" => "No image data found",
                "success" => false
            )), 400
        end
        
        @info "[$request_id] Image loaded: $(size(img, 1))x$(size(img, 2))"
        
        cache_key = "detect:$(hash(string(img))):t=$(threshold):r=$(resize_dim)"
        cached = get_cache(_cache[], cache_key)
        
        if cached !== nothing
            @info "[$request_id] Cache hit!"
            
            if isdefined(_, :metrics) && _metrics[] !== nothing
                increment_counter!(_metrics[], "cache_hits")
            end
            
            return json(Dict(
                "request_id" => request_id,
                "detections" => cached,
                "cached" => true,
                "timestamp" => Dates.format(now(), ISO_FORMAT),
                "success" => true
            ))
        end
        
        start_time = time_ns()
        result = Inference.predict(img; 
            conf_threshold=threshold, 
            resize_dim=resize_dim
        )
        
        elapsed_ms = (time_ns() - start_time) / 1_000_000
        
        detections = Dict(
            "count" => size(result.boxes, 1),
            "boxes" => [Dict(
                "x1" => box[1],
                "y1" => box[2],
                "x2" => box[3],
                "y2" => box[4],
                "score" => result.scores[i],
                "label" => result.labels[i],
                "class_name" => get(result.label_names, result.labels[i], "unknown")
            ) for (i, box) in enumerate(eachrow(result.boxes))],
            "inference_time_ms" => elapsed_ms,
            "image_size" => [size(img, 1), size(img, 2)],
            "threshold_used" => threshold
        )
        
        set_cache(_cache[], cache_key, detections, ConfigMAMAA.CACHE_TTL_RESULT)
        
        if isdefined(_, :metrics) && _metrics[] !== nothing
            increment_counter(_metrics[], "inference_requests")
            observe_latency(_metrics[], elapsed_ms / 1000, "POST", "/detect/image")
        end
        
        image_path = nothing
        image_base64 = nothing
        
        if save_result || return_image
            output_dir = joinpath("inference_outputs", "images", Dates.format(now(), "YYYYmmdd"))
            mkpath(output_dir)
            img_annotated = Inference.draw_results(img, result)
            output_filename = "detect_$(Dates.format(now(), "HHMMSS"))_$request_id.jpg"
            output_path = joinpath(output_dir, output_filename)
            save(output_path, img_annotated)
            image_path = output_path
            if return_image
                image_base64 = image_to_base64(img_annotated)
            end
        end
        
        response = Dict(
            "request_id" => request_id,
            "success" => true,
            "detections" => detections,
            "cached" => false,
            "image_path" => image_path,
            "image_base64" => image_base64,
            "timestamp" => Dates.format(now(), ISO_FORMAT)
        )
        
        return json(response)
    catch e
        return json(Dict(
            "request_id" => request_id,
            "error" => string(e),
            "error_type" => string(typeof(e)),
            "success" => false
        )), 500
    end
end

function is_allowed_content_type(content_type::String)::Bool
    allowed = [
        "image/jpeg", 
        "image/jpg", 
        "image/png", 
        "image/bmp", 
        "image/tiff", 
        "image/tif",
        "image/webp"
    ]
    return content_type in allowed
end

function get_cache(cache::Union{RedisCache, Nothing}, key::String)::Union{Dict, Nothing}
    if cache === nothing
        return nothing
    end
    
    try
        data = Redis.get(cache, key)
        if data !== nothing
            return JSON.read(data, Dict)
        end
    catch e
        @warn "Cache read error: $e"
    end
    
    return nothing
end

function set_cache(cache::Union{RedisCache, Nothing}, key::String, value::Dict, ttl::Int)::Bool
    if cache === nothing
        return false
    end
    
    try
        Redis.setex(cache, key, ttl, JSON.write(value))
        return true
    catch e
        @warn "Cache write error: $e"
        return false
    end
end

function route_get_result(request_id::String)
    try
        if isempty(request_id) || request_id == "undefined"
            return json(Dict(
                "error" => "Invalid request ID",
                "success" => false
            )), 400
        end
        
        cache_key = "result:$request_id"
        cached = get_redis_cache(_redis[], cache_key)
        
        if cached !== nothing
            return json(Dict(
                "request_id" => request_id,
                "status" => "completed",
                "result" => cached,
                "source" => "cache",
                "success" => true,
                "timestamp" => Dates.format(now(), ISO_FORMAT)
            ))
        end
        
        result_paths = [
            joinpath("inference_outputs", "images", "$request_id.json"),
            joinpath("inference_outputs", "images", "detect_$request_id.json"),
            joinpath("inference_outputs", "videos", "$request_id.json"),
            joinpath("inference_outputs", "batch", "$request_id.json"),
            joinpath("outputs", "results", "$request_id.json")
        ]
        
        for path in result_paths
            if isfile(path)
                @info "Result found in file: $path"
                result_data = JSON.read(read(path, String))
                set_redis_cache(_redis[], cache_key, result_data, 3600)
                
                return json(Dict(
                    "request_id" => request_id,
                    "status" => "completed",
                    "result" => result_data,
                    "result_path" => path,
                    "source" => "file",
                    "success" => true,
                    "timestamp" => Dates.format(now(), ISO_FORMAT)
                ))
            end
        end
        
        queue_status = get_queue_status(request_id)
        
        if queue_status !== nothing
            return json(Dict(
                "request_id" => request_id,
                "status" => queue_status["status"],
                "progress" => get(queue_status, "progress", 0),
                "message" => get(queue_status, "message", ""),
                "estimated_time_remaining" => get(queue_status, "eta", nothing),
                "queue_position" => get(queue_status, "position", nothing),
                "source" => "queue",
                "success" => true,
                "timestamp" => Dates.format(now(), ISO_FORMAT)
            ))
        end
        
        db_result = get_db_result(request_id)
        
        if db_result !== nothing
            return json(Dict(
                "request_id" => request_id,
                "status" => "completed",
                "result" => db_result,
                "source" => "database",
                "success" => true,
                "timestamp" => Dates.format(now(), ISO_FORMAT)
            ))
        end
        
        @warn "Result not found for request_id: $request_id"
        
        return json(Dict(
            "request_id" => request_id,
            "status" => "not_found",
            "message" => "No result found for this request ID",
            "success" => false,
            "timestamp" => Dates.format(now(), ISO_FORMAT)
        )), 404
        
    catch e
        @error "Error getting result for $request_id: $e"
        @error stacktrace(catch_backtrace())
        
        return json(Dict(
            "request_id" => request_id,
            "error" => string(e),
            "error_type" => string(typeof(e)),
            "success" => false,
            "timestamp" => Dates.format(now(), ISO_FORMAT)
        )), 500
    end
end

function get_redis_cache(redis::Union{RedisCache, Nothing}, key::String)::Union{Dict, Nothing}
    if redis === nothing
        return nothing
    end
    
    try
        data = Redis.get(redis, key)
        if data !== nothing && !isempty(data)
            return JSON.read(data, Dict)
        end
    catch e
        @warn "Redis get error: $e"
    end
    
    return nothing
end

function set_redis_cache(redis::Union{RedisCache, Nothing}, key::String, value::Dict, ttl::Int)::Bool
    if redis === nothing
        return false
    end
    
    try
        Redis.setex(redis, key, ttl, JSON.write(value))
        return true
    catch
        return false
    end
end

function get_queue_status(request_id::String)::Union{Dict, Nothing}
    if @isdefined(_redis) && _redis[] !== nothing
        queue_key = "queue:status:$request_id"
        status_data = try
            Redis.get(_redis[], queue_key)
        catch
            nothing
        end
        status_data !== nothing && return JSON.read(status_data, Dict)
    end

    return nothing
end

function get_conn()
    host = ENV["DRONE_DB_HOST"]
    port = ENV["DRONE_DB_PORT"]
    db   = ENV["DRONE_DB_NAME"]
    user = ENV["DRONE_DB_USER"]
    conn_str = """
        host=$host port=$port dbname=$db user=$user
        sslmode=verify-full
        sslrootcert=$(ENV["DRONE_DB_CA"])
    """
    return LibPQ.Connection(conn_str)
end

function get_db_result(request_id::String)::Union{Dict, Nothing}
    conn = nothing
    try
        conn = get_conn()
        result = execute(conn, """
            SELECT request_id, drone_id, ts_utc, status, payload
            FROM requests
            WHERE request_id = \$1
            LIMIT 1;
        """, [request_id])
        
        rows = collect(result)
        
        isempty(rows) && return nothing
        
        row = rows[1]
        return Dict(
            "request_id" => row[1],
            "drone_id"   => row[2],
            "ts_utc"     => row[3],
            "status"     => row[4],
            "payload"    => row[5],
        )
        
    catch
        return nothing
    finally
        if conn !== nothing
            close(conn)
        end
    end
end

function route_get_result_batch(request_ids::Vector{String})
    results = Dict{String, Any}()
    
    for request_id in request_ids
        result = try_get_single_result(request_id)
        results[request_id] = result
    end
    
    return json(Dict(
        "batch_size" => length(request_ids),
        "results" => results,
        "success" => true,
        "timestamp" => Dates.format(now(), ISO_FORMAT)
    ))
end

function try_get_single_result(request_id::String)::Dict
    cache_key = "result:$request_id"
    cached = get_redis_cache(_redis[], cache_key)
    if cached !== nothing
        return Dict("status" => "completed", "result" => cached, "source" => "cache")
    end
    
    for path in [
        joinpath("inference_outputs", "images", "$request_id.json"),
        joinpath("inference_outputs", "images", "detect_$request_id.json"),
        joinpath("inference_outputs", "videos", "$request_id.json"),
        joinpath("inference_outputs", "batch", "$request_id.json")
    ]
        if isfile(path)
            data = JSON.read(read(path, String))
            return Dict("status" => "completed", "result" => data, "source" => "file", "path" => path)
        end
    end
    
    queue_status = get_queue_status(request_id)
    if queue_status !== nothing
        return queue_status
    end
    
    return Dict("status" => "not_found", "source" => "none")
end

function route_queue_stats()
    result = QueueStatsResponse(
        length(_mq[].inference_queue),
        length(_mq[].results_queue),
        length(_mq[].dlq)
    )
    return json(result)
end

function route_cache_stats()
    cache = _cache[]
    result = Dict(
        "hits" => cache.hits,
        "misses" => cache.misses,
        "hit_rate" => hit_rate(cache),
        "size" => length(cache.data)
    )
    return json(result)
end

function route_metrics()
    metrics = _metrics[]
    
    result = Dict{String, Any}()
    result["requests_total"] = sum(values(metrics.request_count))
    result["inference_total"] = sum(values(metrics.inference_count))

    if !isempty(metrics.latencies)
        result["avg_latency_seconds"] = mean(metrics.latencies)
    else
        result["avg_latency_seconds"] = 0.0
    end

    result["cache_hit_rate"] = hit_rate(_cache[])
    
    return json(result)
end

function setup_routes()
    Genie.route("/", method=GET)(route_root)
    Genie.route("/health", method=GET)(route_health)
    Genie.route("/ready", method=GET)(route_ready)
    Genie.route("/live", method=GET)(route_live)
    Genie.route("/metrics", method=GET)(route_metrics)
    Genie.route("/queue/stats", method=GET)(route_queue_stats)
    Genie.route("/cache/stats", method=GET)(route_cache_stats)
    Genie.route("/result/:request_id", method=GET)(route_get_result)
end

function run_server(; host::String=Config.API_HOST, port::Int=Config.API_PORT)
    
    try
        _start_time[] = now()
        setup_routes()
        _health_task[] = Threads.@spawn health_check_loop()
        Genie.up(host=host, port=port, verbose=true)
        return true
    catch
        return nothing
    end
end

function main()
    return run_server()
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

export
    Serve,
    run_server,
    main,
    route_root,
    route_health,
    route_ready,
    route_live,
    route_detect_image,
    route_detect_async,
    route_metrics,
    route_queue_stats,
    route_cache_stats

end  # module Serve
