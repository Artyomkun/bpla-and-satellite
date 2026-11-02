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

module SatelliteDetection

using ..Models: RussianYOLOModel, load_model, model_to_device, decode_predictions
using Statistics
using Logging
using Dates
using Sockets
using Images
using FileIO
using ImageCore
using Base64

global_logger(SimpleLogger(stderr, Logging.Info))

export
    SatelliteCamera,
    SatelliteDetector,
    connect,
    disconnect,
    capture_image,
    detect_on_satellite!,
    stream_detection!

mutable struct SatelliteDetector
    camera::SatelliteCamera
    model::RussianYOLOModel
    device::Symbol
    conf_threshold::Float64
    iou_threshold::Float64
    anchors::Vector{Tuple{Float32, Float32}}
    strides::Vector{Int}
    detection_history::Vector{Dict}
    
    function SatelliteDetector(
        camera_ip::String;
        camera_port::Int=8080,
        model_path::Union{String,Nothing}=nothing,
        device::Symbol=:auto,
        conf_threshold::Float64=0.5,
        iou_threshold::Float64=0.45
    )
        camera = SatelliteCamera(camera_ip, port=camera_port)
        if model_path !== nothing
            @info "Загрузка модели из $model_path"
            model = load_model(model_path)
        else
            @info "Создание новой модели RussianYOLO"
            config = Models.RussianYOLOConfig()
            model = Models.create_model(config)
        end
        model = Models.model_to_device(model, device)
        anchors = [
            (10f0, 13f0), (16f0, 30f0), (33f0, 23f0),
            (30f0, 61f0), (62f0, 45f0), (59f0, 119f0),
            (116f0, 90f0), (156f0, 198f0), (373f0, 326f0)
        ]
        strides = [8, 16, 32]
        
        return new(
            camera, model, device, conf_threshold, iou_threshold,
            anchors, strides, Dict[]
        )
    end
end

function detect_on_satellite(detector::SatelliteDetector)::Dict
    @info "="^60
    @info "Спутниковая детекция - кадр #$(detector.camera.frame_count + 1)"
    @info "="^60
    
    image = capture_image(detector.camera)
    img_tensor = prepare_image(image, detector.model.config.input_size)
    
    if detector.device == :gpu
        img_tensor = img_tensor |> gpu
    end
    
    @info "Запуск RussianYOLO модели..."
    start_time = time_ns()
    
    predictions = detector.model(img_tensor)
    all_boxes = Matrix{Float32}(undef, 0, 4)
    all_scores = Float32[]
    all_labels = Int[]
    
    for (i, pred) in enumerate(predictions)
        stride = detector.strides[i]
        scale_anchors = detector.anchors[(3*(i-1)+1):(3*i)]
        
        boxes, scores, labels = decode_predictions(
            pred,
            scale_anchors,
            stride,
            detector.conf_threshold
        )
        
        if size(boxes, 1) > 0
            all_boxes = vcat(all_boxes, boxes)
            append!(all_scores, scores)
            append!(all_labels, labels)
        end
    end
    
    if size(all_boxes, 1) > 0
        keep_idx = non_maximum_suppression(all_boxes, all_scores, detector.iou_threshold)
        all_boxes = all_boxes[keep_idx, :]
        all_scores = all_scores[keep_idx]
        all_labels = all_labels[keep_idx]
    end
    
    elapsed_ms = (time_ns() - start_time) / 1_000_000
    detection_result = Dict(
        "timestamp" => now(),
        "frame" => detector.camera.frame_count,
        "num_people" => size(all_boxes, 1),
        "boxes" => all_boxes,
        "scores" => all_scores,
        "labels" => all_labels,
        "inference_time_ms" => elapsed_ms,
        "image_size" => size(image)
    )
    
    push!(detector.detection_history, detection_result)
    
    @info "✅ Детекция завершена:"
    @info "   - Найдено людей: $(size(all_boxes, 1))"
    @info "   - Время инференса: $(round(elapsed_ms, digits=1)) мс"
    @info "   - Размер снимка: $(size(image, 2))x$(size(image, 1))"
    
    return detection_result
end

"""
    prepare_image(image::Matrix{UInt8}, target_size::Int) - подготовка изображения для модели
"""
function prepare_image(image::Matrix{UInt8}, target_size::Int)
    h, w = size(image)[1:2]
    scale = min(target_size / h, target_size / w)
    new_h = round(Int, h * scale)
    new_w = round(Int, w * scale)
    img_resized = imresize(image, (new_h, new_w))
    pad_h = target_size - new_h
    pad_w = target_size - new_w
    pad_top = pad_h ÷ 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w ÷ 2
    pad_right = pad_w - pad_left
    img_padded = zeros(UInt8, target_size, target_size, 3)
    img_padded[pad_top+1:pad_top+new_h, pad_left+1:pad_left+new_w, :] = img_resized
    img_float = Float32.(img_padded) / 255.0
    mean = [0.485f0, 0.456f0, 0.406f0]
    std = [0.229f0, 0.224f0, 0.225f0]
    
    for c in 1:3
        img_float[:, :, c] = (img_float[:, :, c] .- mean[c]) ./ std[c]
    end
    
    img_tensor = permutedims(img_float, (3, 2, 1))
    img_batch = reshape(img_tensor, 3, target_size, target_size, 1)
    
    return img_batch
end

function non_maximum_suppression(boxes::Matrix{Float32}, scores::Vector{Float32}, iou_threshold::Float64)
    if size(boxes, 1) == 0
        return Int[]
    end
    
    order = sortperm(scores, rev=true)
    keep = Int[]
    
    while !isempty(order)
        i = order[1]
        push!(keep, i)
        
        if length(order) == 1
            break
        end
        
        rest = order[2:end]
        ious = Float64[]
        
        for j in rest
            x1 = max(boxes[i, 1], boxes[j, 1])
            y1 = max(boxes[i, 2], boxes[j, 2])
            x2 = min(boxes[i, 3], boxes[j, 3])
            y2 = min(boxes[i, 4], boxes[j, 4])
            inter = max(0, x2 - x1) * max(0, y2 - y1)
            area_i = (boxes[i, 3] - boxes[i, 1]) * (boxes[i, 4] - boxes[i, 2])
            area_j = (boxes[j, 3] - boxes[j, 1]) * (boxes[j, 4] - boxes[j, 2])
            union = area_i + area_j - inter
            iou = union > 0 ? inter / union : 0.0
            push!(ious, iou)
        end
        
        order = rest[ious .<= iou_threshold]
    end
    
    return keep
end

function stream_detection!(detector::SatelliteDetector; interval::Int=5, max_frames::Int=0)
    @info "="^60
    @info "Запуск непрерывного мониторинга со спутниковой камеры $(detector.camera.ip)"
    @info "Интервал: $interval сек"
    @info "Максимум кадров: $(max_frames > 0 ? max_frames : "бесконечно")"
    @info "="^60
    
    frame_count = 0
    total_people = 0
    
    try
        while true
            frame_count += 1
            
            if max_frames > 0 && frame_count > max_frames
                @info "Достигнут лимит кадров ($max_frames)"
                break
            end
            
            @info "\n" * "="^60
            @info "Кадр #$frame_count"
            @info "="^60
            
            result = detect_on_satellite!(detector)
            total_people += result["num_people"]
            
            @info "Текущий кадр: $(result["num_people"]) человек"
            @info "Всего людей за сессию: $total_people"
            
            if interval > 0 && frame_count < (max_frames > 0 ? max_frames : typemax(Int))
                @info "Ожидание $(interval) сек до следующего кадра..."
                sleep(interval)
            end
        end
    catch e
        @error "Ошибка в потоке детекции: $e"
    finally
        disconnect(detector)
    end
    
    @info "="^60
    @info "Мониторинг завершен"
    @info "Всего кадров: $frame_count"
    @info "Всего обнаружено людей: $total_people"
    @info "="^60
    
    return frame_count, total_people
end

function get_statistics(detector::SatelliteDetector)
    history = detector.detection_history
    
    if isempty(history)
        return Dict("message" => "Нет данных детекции")
    end
    
    people_per_frame = [h["num_people"] for h in history]
    inference_times = [h["inference_time_ms"] for h in history]
    
    stats = Dict(
        "total_frames" => length(history),
        "total_people" => sum(people_per_frame),
        "avg_people_per_frame" => mean(people_per_frame),
        "max_people_per_frame" => maximum(people_per_frame),
        "min_people_per_frame" => minimum(people_per_frame),
        "avg_inference_time_ms" => mean(inference_times),
        "total_time_seconds" => sum(inference_times) / 1000,
        "camera_ip" => detector.camera.ip,
        "camera_frames" => detector.camera.frame_count
    )
    
    return stats
end

end # module SatelliteDetection