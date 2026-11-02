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

module App

include("Config.jl")
include("Models.jl")
include("Training.jl")

import GPUArrays: AbstractGPUArray
import Images

using ImageTransformations
using OrderedCollections
using Genie.Responses
using Genie.Renderer
using Genie.Requests
using Statistics
using ImageCore
using ImageDraw
using FilePaths
using Logging
using Printf
using Random
using Genie
using Dates
using JSON

export
    App,
    Detections,
    inference_on_image,
    inference_on_frame,
    run_server,
    main,
    get_model

Genie.config.log_requests = false
mutable struct Detections
    boxes::Matrix{Float32}
    scores::Vector{Float32}
    labels::Vector{Int64}
end

is_empty(det::Detections) = size(det.boxes, 1) == 0
count(det::Detections) = size(det.boxes, 1)

const _model_lock = ReentrantLock()
const _model = Ref{Any}(nothing)

function get_model()
    lock(_model_lock) do
        if _model[] === nothing
            model_path = find_latest_model()
            
            if model_path !== nothing && isfile(model_path)
                @info "✅ Загрузка модели из: $model_path"
                _model[] = invokelatest(Models.load_model, model_path)
            else
                @info "⚠️ Модель не найдена, используем заглушку"
                _model[] = invokelatest(Models.RussianYOLOStub, nothing)
            end
        end
        return _model[]
    end
end

function find_latest_model()::Union{String, Nothing}
    base_dirs = [
        "outputs",
        joinpath("outputs", "checkpoints"),
        joinpath("outputs", "augmented_train")
    ]
    
    all_models = String[]
    
    for base_dir in base_dirs
        if !isdir(base_dir)
            continue
        end
        
        for (root, files) in walkdir(base_dir)
            for file in files
                if endswith(file, ".bson") && (
                    occursin("final_model", file) ||
                    occursin("final_model_ema", file) || 
                    occursin("checkpoint", file)
                )
                    push!(all_models, joinpath(root, file))
                end
            end
        end
    end
    
    if !isempty(all_models)
        sort!(all_models; by=mtime, rev=true)
        return all_models[1]
    end
    
    return nothing
end

mtime(path::String) = stat(path).mtime

to_array(x::AbstractArray) = x
to_array(x::AbstractGPUArray) = Array(x)

function run_inference(
    model::Any,
    image::Matrix{RGB},
    threshold::Float64
)::Tuple{Detections, Float64}
    start_time = time_ns()
    results = Models.predict(model, image)
    elapsed = (time_ns() - start_time) / 1_000_000
    fps = 1.0 / max(elapsed / 1000, 1e-6)
    
    if isempty(results)
        return Detections(
            Matrix{Float32}(undef, 0, 4),
            Float32[],
            Int64[]
        ), fps
    end
    
    result = results[1]
    if !hasfield(typeof(result), :boxes) || size(result.boxes.xyxy, 1) == 0
        return Detections(
            Matrix{Float32}(undef, 0, 4),
            Float32[],
            Int64[]
        ), fps
    end
    
    boxes = result.boxes.xyxy
    scores = result.boxes.conf
    labels = result.boxes.cls
    valid_idx = findall(scores .>= threshold)
    
    return Detections(
        boxes[valid_idx, :],
        scores[valid_idx],
        labels[valid_idx]
    ), fps
end

function rescale_boxes(
    boxes::Matrix{Float32},
    resize_dim::Union{Int, Nothing},
    w_orig::Int,
    h_orig::Int
)::Matrix{Float32}
    if resize_dim === nothing || size(boxes, 1) == 0
        return boxes
    end
    
    scale_x = w_orig / resize_dim
    scale_y = h_orig / resize_dim
    scaled = copy(boxes)
    scaled[:, [1, 3]] .*= scale_x
    scaled[:, [2, 4]] .*= scale_y
    return scaled
end

struct ClassCount
    class_id::Int
    count::Int
end

function count_by_class(detections::Detections)::Vector{ClassCount}
    counts = Dict{Int, Int}()
    for label in detections.labels
        counts[label] = get(counts, label, 0) + 1
    end
    return [ClassCount(k, v) for (k, v) in sort(collect(counts))]
end

function draw_detections(
    image::Matrix{RGB},
    detections::Detections,
    colors::Vector{Tuple{Int, Int, Int}};
    thickness::Int = 2,
)::Matrix{RGB}
    img_out = copy(image)
    h, w = size(img_out)
    rgb_colors = [RGB{Float32}(c[1]/255, c[2]/255, c[3]/255) for c in colors]
    for (box, label) in zip(
        eachrow(detections.boxes),
        detections.labels,
        detections.scores
    )
        x1, y1, x2, y2 = map(Int, box)
        x1 = clamp(x1, 1, w); x2 = clamp(x2, 1, w)
        y1 = clamp(y1, 1, h); y2 = clamp(y2, 1, h)
        color = rgb_colors[mod1(label + 1, length(rgb_colors))]
        for t in 0:(thickness - 1)
            for x in x1:x2
                yy1 = clamp(y1 + t, 1, h)
                yy2 = clamp(y2 - t, 1, h)
                img_out[yy1, x] = color
                img_out[yy2, x] = color
            end
            for y in y1:y2
                xx1 = clamp(x1 + t, 1, w)
                xx2 = clamp(x2 - t, 1, w)
                img_out[y, xx1] = color
                img_out[y, xx2] = color
            end
        end
    end
    return img_out
end

struct LetterboxInfo
    scale::Float32
    pad_left::Int
    pad_top::Int
end

function letterbox(img::Matrix{RGB}, target::Int)::Tuple{Matrix{RGB}, LetterboxInfo}
    h, w = size(img)
    scale = min(target / w, target / h)
    new_w, new_h = round(Int, w * scale), round(Int, h * scale)
    resized = imresize(img, (new_h, new_w))
    pad_top  = (target - new_h) ÷ 2
    pad_left = (target - new_w) ÷ 2
    bg = RGB{Float32}(114/255, 114/255, 114/255)
    padded = fill(bg, target, target)
    padded[pad_top+1:pad_top+new_h, pad_left+1:pad_left+new_w] .= resized

    return padded, LetterboxInfo(Float32(scale), pad_left, pad_top)
end

function rescale_boxes(
    boxes::Matrix{Float32},
    info::LetterboxInfo,
    orig_w::Int,
    orig_h::Int,
)::Matrix{Float32}
    out = copy(boxes)
    s = info.scale
    @views out[:, 1] = (boxes[:, 1] .- info.pad_left) ./ s
    @views out[:, 2] = (boxes[:, 2] .- info.pad_top)  ./ s
    @views out[:, 3] = (boxes[:, 3] .- info.pad_left) ./ s
    @views out[:, 4] = (boxes[:, 4] .- info.pad_top)  ./ s
    @views out[:, 1] .= clamp.(out[:, 1], 1f0, Float32(orig_w))
    @views out[:, 2] .= clamp.(out[:, 2], 1f0, Float32(orig_h))
    @views out[:, 3] .= clamp.(out[:, 3], 1f0, Float32(orig_w))
    @views out[:, 4] .= clamp.(out[:, 4], 1f0, Float32(orig_h))

    return out
end

function inference_on_image(
    image::Matrix{RGB};
    resize_dim::Union{Int, Nothing} = nothing,
    threshold::Float64 = 0.25,
)::Tuple{Matrix{RGB}, Float64}
    model = get_model()
    h_orig, w_orig = size(image)
    if resize_dim !== nothing
        img, lb_info = letterbox(image, resize_dim)
    else
        img = image
        lb_info = LetterboxInfo(1.0f0, 0, 0)
    end
    detections, inference_ms = run_inference(model, img, threshold)
    if !isempty(detections)
        new_boxes = rescale_boxes(detections.boxes, lb_info, w_orig, h_orig)
        detections = Detections(new_boxes, detections.scores, detections.labels)
    end
    colors = Config.get_colors(10)
    class_names = Config.classes()
    result_image = draw_detections(image, detections, inference_ms, class_names, colors)
    return result_image, inference_ms
end

function route_home()
    html = """
    <!DOCTYPE html>
    <html lang="ru">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>BPLA Person Detection</title>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                border-radius: 10px;
                padding: 30px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.3);
            }
            h1 {
                color: #333;
                text-align: center;
                margin-bottom: 30px;
            }
            .tabs {
                display: flex;
                gap: 10px;
                margin-bottom: 20px;
            }
            .tab {
                padding: 12px 24px;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                font-size: 16px;
                transition: all 0.3s;
            }
            .tab.active {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
            }
            .tab.inactive {
                background: #f0f0f0;
                color: #333;
            }
            .form-group {
                margin-bottom: 15px;
            }
            label {
                display: block;
                margin-bottom: 5px;
                font-weight: bold;
            }
            input[type="file"] {
                width: 100%;
                padding: 10px;
                border: 2px dashed #667eea;
                border-radius: 5px;
            }
            input[type="range"] {
                width: 100%;
            }
            button[type="submit"] {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 15px 30px;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                font-size: 16px;
                width: 100%;
                margin-top: 10px;
            }
            button[type="submit"]:hover {
                opacity: 0.9;
            }
            .result {
                margin-top: 20px;
                text-align: center;
            }
            .result img {
                max-width: 100%;
                border-radius: 5px;
                box-shadow: 0 5px 15px rgba(0,0,0,0.2);
            }
            .stats {
                display: flex;
                justify-content: center;
                gap: 30px;
                margin-top: 15px;
                font-size: 18px;
            }
            .stat {
                padding: 10px 20px;
                background: #f0f0f0;
                border-radius: 5px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🛸 BPLA Person Detection</h1>
            
            <div class="tabs">
                <button class="tab active" onclick="showTab('image')">📷 Изображение</button>
                <button class="tab inactive" onclick="showTab('video')">🎬 Видео</button>
            </div>
            
            <div id="image-tab">
                <form action="/infer/image" method="post" enctype="multipart/form-data">
                    <div class="form-group">
                        <label for="image">Загрузите изображение:</label>
                        <input type="file" id="image" name="image" accept="image/*" required>
                    </div>
                    <div class="form-group">
                        <label for="resize_dim">Размер ресайза: <span id="resize_val">640</span></label>
                        <input type="range" id="resize_dim" name="resize_dim" min="100" max="1024" value="640" 
                                oninput="document.getElementById('resize_val').textContent = this.value">
                    </div>
                    <div class="form-group">
                        <label for="threshold">Threshold: <span id="thresh_val">0.25</span></label>
                        <input type="range" id="threshold" name="threshold" min="0" max="1" step="0.01" value="0.25"
                                oninput="document.getElementById('thresh_val').textContent = this.value">
                    </div>
                    <button type="submit">🔍 Детекция</button>
                </form>
            </div>
            
            <div id="video-tab" style="display:none;">
                <form action="/infer/video" method="post" enctype="multipart/form-data">
                    <div class="form-group">
                        <label for="video">Загрузите видео:</label>
                        <input type="file" id="video" name="video" accept="video/*" required>
                    </div>
                    <div class="form-group">
                        <label for="resize_dim_v">Размер ресайза: <span id="resize_val_v">640</span></label>
                        <input type="range" id="resize_dim_v" name="resize_dim" min="100" max="1024" value="640"
                                oninput="document.getElementById('resize_val_v').textContent = this.value">
                    </div>
                    <div class="form-group">
                        <label for="threshold_v">Threshold: <span id="thresh_val_v">0.25</span></label>
                        <input type="range" id="threshold_v" name="threshold" min="0" max="1" step="0.01" value="0.25"
                                oninput="document.getElementById('thresh_val_v').textContent = this.value">
                    </div>
                    <button type="submit">🎬 Обработка видео</button>
                </form>
            </div>
            
            <script>
                function showTab(tab) {
                    document.getElementById('image-tab').style.display = tab === 'image' ? 'block' : 'none';
                    document.getElementById('video-tab').style.display = tab === 'video' ? 'block' : 'none';
                    document.querySelectorAll('.tab')[0].className = 'tab ' + (tab === 'image' ? 'active' : 'inactive');
                    document.querySelectorAll('.tab')[1].className = 'tab ' + (tab === 'video' ? 'active' : 'inactive');
                }
            </script>
        </div>
    </body>
    </html>
    """
    return html
end

"""
API: Детекция на изображении
"""
function route_infer_image()
    @info "Processing image inference request"
    
    try
        files = rawpayload(:files, Dict{String, Any})
        
        if !haskey(files, "image")
            return json(Dict("error" => "No image provided"), status=400)
        end
        
        image_file = files["image"]
        image_path = image_file["tempfile"]
        post_data = rawpayload(:post, Dict{String, Any})
        resize_dim = haskey(post_data, "resize_dim") ? parse(Int, post_data["resize_dim"]) : nothing
        threshold = haskey(post_data, "threshold") ? parse(Float64, post_data["threshold"]) : 0.25
        img = load(image_path)
        if ndims(img) == 2
            img = RGB.(img)
        end

        result_image, fps = inference_on_image(img; resize_dim=resize_dim, threshold=threshold)
        output_dir = joinpath("inference_outputs", "images")
        mkpath(output_dir)
        timestamp = Dates.format(Dates.now(), "YYYYmmdd_HHMMSS")
        output_path = joinpath(output_dir, "result_$timestamp.png")
        save(output_path, result_image)
        
        @info "Result saved to: $output_path, FPS: $fps"

        result = Dict(
            "success" => true,
            "output_path" => output_path,
            "fps" => fps,
            "people_count" => count(Detections(
                Matrix{Float32}(undef, 0, 4),
                Float32[],
                Int64[]
            ))
        )
        
        return json(result)
        
    catch e
        @error "Error processing image: $e"
        return json(Dict("error" => string(e)), status=500)
    end
end

function route_infer_video()
    @info "Processing video inference request"
    
    try
        files = rawpayload(:files, Dict{String, Any})
        
        if !haskey(files, "video")
            return json(Dict("error" => "No video provided"), status=400)
        end
        
        video_file = files["video"]
        video_path = video_file["tempfile"]
        
        post_data = rawpayload(:post, Dict{String, Any})
        resize_dim = haskey(post_data, "resize_dim") ? parse(Int, post_data["resize_dim"]) : 640
        threshold = haskey(post_data, "threshold") ? parse(Float64, post_data["threshold"]) : 0.25
        batch_size = haskey(post_data, "batch_size") ? parse(Int, post_data["batch_size"]) : 4
        save_frames = haskey(post_data, "save_frames") ? parse(Bool, post_data["save_frames"]) : false
        output_format = haskey(post_data, "format") ? post_data["format"] : "mp4"
        output_dir = joinpath("outputs", "videos", "$(Dates.format(now(), "YYYYmmdd_HHMMSS"))")
        mkpath(output_dir)
        
        output_filename = "processed_$(basename(video_path))"
        output_path = joinpath(output_dir, output_filename)
        
        @info "Processing video:"
        @info "   Input: $video_path"
        @info "   Output: $output_path"
        @info "   Resize: $resize_dim"
        @info "   Threshold: $threshold"
        @info "   Batch size: $batch_size"
        
        model = get_model()
        cap = VideoCapture(video_path)

        if !isopen(cap)
            error("Cannot open video file: $video_path")
        end
        
        width = get(cap, CAP_PROP_FRAME_WIDTH)
        height = get(cap, CAP_PROP_FRAME_HEIGHT)
        fps = get(cap, CAP_PROP_FPS)
        total_frames = get(cap, CAP_PROP_FRAME_COUNT)
        
        @info "Video info: $(width)x$height, $fps fps, $total_frames frames"

        config = get_config() 
        fourcc = Int32(config.video_codec)
        writer = VideoWriter(output_path, fourcc, fps, (width, height))
        frame_count = 0
        batch_frames = []
        batch_results = []
        p = Progress(total_frames, 1, "Processing frames...")
        
        while true
            ret, frame = read(cap)
            !ret && break
            frame_count += 1
            img_rgb = RGB.(frame)
            input_tensor = preprocess_image(img_rgb, resize_dim)
            push!(batch_frames, (input_tensor, frame, img_rgb))
            
            if length(batch_frames) >= batch_size || frame_count == total_frames
                batch_input = cat([f[1] for f in batch_frames]..., dims=4)
                
                predictions = model(batch_input)
                
                for (i, (_, orig_frame, img_rgb)) in enumerate(batch_frames)
                    boxes, scores, labels = decode_predictions(
                        predictions[i], 
                        model.anchors,
                        model.stride,
                        threshold
                    )
                    
                    keep = scores .>= threshold
                    boxes = boxes[keep, :]
                    scores = scores[keep]
                    labels = labels[keep]
                    
                    if size(boxes, 1) > 0
                        keep_idx = non_maximum_suppression(boxes, scores, 0.45)
                        boxes = boxes[keep_idx, :]
                        scores = scores[keep_idx]
                        labels = labels[keep_idx]
                    end

                    annotated = draw_detections(orig_frame, boxes, scores, labels)
                    write(writer, annotated)
                    if save_frames
                        frame_path = joinpath(output_dir, "frame_$(frame_count).jpg")
                        save(frame_path, annotated)
                    end
                end
                
                empty!(batch_frames)
                next!(p)
            end
        end
        
        release(cap)
        release(writer)
        report = Dict(
            "timestamp" => string(now()),
            "input_video" => basename(video_path),
            "output_video" => output_filename,
            "video_info" => Dict(
                "width" => width,
                "height" => height,
                "fps" => fps,
                "frames" => total_frames
            ),
            "settings" => Dict(
                "resize_dim" => resize_dim,
                "threshold" => threshold,
                "batch_size" => batch_size,
                "save_frames" => save_frames
            )
        )
        
        report_path = joinpath(output_dir, "report.json")
        open(report_path, "w") do f
            JSON.pretty(f, report)
        end
        
        @info "✅ Video processing complete!"
        @info "   Output: $output_path"
        @info "   Report: $report_path"
        
        return json(Dict(
            "success" => true,
            "output_path" => output_path,
            "output_dir" => output_dir,
            "report_path" => report_path,
            "video_info" => Dict(
                "width" => width,
                "height" => height,
                "fps" => fps,
                "frames" => total_frames
            ),
            "message" => "Video processing complete"
        ))
        
    catch e
        @error "Error processing video: $e"
        @error stacktrace(catch_backtrace())
        return json(Dict("error" => string(e)), status=500)
    end
end

function process_video(
    input_path::String;
    resize_dim::Union{Int, Nothing}=640,
    threshold::Float64=0.25,
    batch_size::Int=4,
    output_path::Union{String, Nothing}=nothing,
    save_frames::Bool=false,
    show_progress::Bool=true
)::String
    @info "Processing video: $input_path"
    output_dir = joinpath("inference_outputs", "videos", "$(Dates.format(now(), "YYYYmmdd_HHMMSS"))")
    mkpath(output_dir)
    
    if output_path === nothing
        base_name = splitext(basename(input_path))[1]
        output_path = joinpath(output_dir, "$(base_name)_processed.mp4")
    end
    @info "Loading model..."
    model = get_model()
    @info "Opening video: $input_path"
    cap = VideoCapture(input_path)
    if !isopen(cap)
        error("Cannot open video file: $input_path")
    end
    width = round(Int, get(cap, CAP_PROP_FRAME_WIDTH))
    height = round(Int, get(cap, CAP_PROP_FRAME_HEIGHT))
    fps = get(cap, CAP_PROP_FPS)
    total_frames = round(Int, get(cap, CAP_PROP_FRAME_COUNT))
    @info "Video info:"
    @info "   Resolution: $(width)x$height"
    @info "   FPS: $fps"
    @info "   Frames: $total_frames"
    config = get_config()
    fourcc = Int32(config.video_codec)
    writer = VideoWriter(output_path, fourcc, fps, (width, height))
    if !isopen(writer)
        error("Cannot create video writer with codec $(config.video_codec)")
    end
    total_detections = 0
    processed_frames = 0
    batch_frames = []
    frame_buffer = []
    inference_times = Float64[]
    if show_progress
        p = Progress(total_frames, 1, "Processing frames...")
    end
    if save_frames
        frames_dir = joinpath(output_dir, "frames")
        mkpath(frames_dir)
    end
    try
        while true
            ret, frame = read(cap)
            !ret && break
            processed_frames += 1
            img_rgb = RGB.(frame)
            input_tensor = preprocess_image(img_rgb, resize_dim)
            push!(frame_buffer, (input_tensor, frame, img_rgb))
            if length(frame_buffer) >= batch_size || processed_frames == total_frames
                batch_input = cat([f[1] for f in frame_buffer]..., dims=4)
                t_start = time()
                predictions = model(batch_input)
                for (i, (_, orig_frame, _)) in enumerate(frame_buffer)
                    boxes, scores, labels = decode_predictions(
                        predictions[i],
                        model.anchors,
                        model.stride,
                        threshold
                    )
                    if !isempty(scores)
                        keep = scores .>= threshold
                        boxes = boxes[keep, :]
                        scores = scores[keep]
                        labels = labels[keep]
                        if size(boxes, 1) > 0
                            keep_idx = non_maximum_suppression(boxes, scores, 0.45)
                            boxes = boxes[keep_idx, :]
                            scores = scores[keep_idx]
                            labels = labels[keep_idx]
                        end
                        total_detections += size(boxes, 1)
                    end
                    annotated = draw_detections(orig_frame, boxes, scores, labels)
                    write(writer, annotated)
                    if save_frames
                        frame_path = joinpath(frames_dir, "frame_$(processed_frames).jpg")
                        save(frame_path, annotated)
                    end
                end
                t_elapsed = time() - t_start
                push!(inference_times, t_elapsed / length(frame_buffer))
                empty!(frame_buffer)
            end
            if show_progress
                next!(p)
            end
        end
    catch e
        @error "Error during video processing: $e"
        rethrow()
    finally
        release(cap)
        release(writer)
    end
    avg_time = isempty(inference_times) ? 0.0 : mean(inference_times) * 1000
    fps_achieved = avg_time > 0 ? 1000 / avg_time : 0
    @info "✅ Video processing complete!"
    @info "   Output: $output_path"
    @info "   Frames processed: $processed_frames"
    @info "   Total detections: $total_detections"
    @info "   Avg inference time: $(round(avg_time, digits=2)) ms/frame"
    @info "   Achieved FPS: $(round(fps_achieved, digits=2))"
    report = Dict(
        "timestamp" => string(now()),
        "input_video" => basename(input_path),
        "output_video" => basename(output_path),
        "video_info" => Dict(
            "width" => width,
            "height" => height,
            "fps" => fps,
            "frames" => total_frames
        ),
        "settings" => Dict(
            "resize_dim" => resize_dim,
            "threshold" => threshold,
            "batch_size" => batch_size,
            "save_frames" => save_frames
        ),
        "results" => Dict(
            "frames_processed" => processed_frames,
            "total_detections" => total_detections,
            "avg_inference_time_ms" => avg_time,
            "achieved_fps" => fps_achieved
        )
    )
    report_path = joinpath(output_dir, "report.json")
    open(report_path, "w") do f
        JSON.pretty(f, report)
    end
    @info "Report saved: $report_path"
    
    return output_path
end

function run_server(; host::String="0.0.0.0", port::Int=7860)
    @info "Starting BPLA Person Detection Web Server"
    @info "Server will be available at: http://$host:$port"
    Genie.route("/", method=GET)(route_home)
    Genie.route("/infer/image", method=POST)(route_infer_image)
    Genie.route("/infer/video", method=POST)(route_infer_video)
    Genie.up(host=host, port=port, verbose=true)
    return nothing
end

function main()
    @info "="^60
    @info "BPLA Person Detection Web Interface"
    @info "="^60
    try
        run_server()
        return true
    catch e
        @error "Critical error: $e"
        @error stacktrace(catch_backtrace())
        return false
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

end  # module App
