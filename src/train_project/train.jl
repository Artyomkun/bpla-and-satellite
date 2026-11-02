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

module Training

using TensorBoardLogger: TBLogger, log_value, log_histogram, log_text
using Flux: onecold, onehotbatch, logitcrossentropy
using Statistics: mean, std
using ImageTransformations
using Logging: with_logger
using OrderedCollections
using Printf: @sprintf
using ProgressMeter
using Distributed
using Statistics
using Dates: now
using ImageCore
using ImageIO
using Logging
using Images
using FileIO
using Random
using Printf
using Dates
using JSON
using Flux
using BSON
using MPI

global_logger(ConsoleLogger(stderr, Logging.Info))

const CONFIG = OrderedDict{String, Any}(
    "num_epochs" => 100,
    "batch_size" => 8,
    "img_size" => 640,
    "num_workers" => 4,
    "learning_rate" => 0.001,
    "weight_decay" => 0.0005,
    "early_stopping_patience" => 10,
    "early_stopping_min_delta" => 0.001,
    "random_seed" => 42,
    "augment" => true,
    "profile" => true,
    "tensorboard" => true,
    "distributed" => false,
    "num_gpus" => parse(Int, get(ENV, "NUM_GPUS", "0")),
    "log_dir" => "logs/train_$(Dates.format(now(), "YYYYmmdd_HHMMSS"))",
    "profile_output" => "profiling_results.json"
)

const DEVICE = detect()

const CURRENT_DIR = @__DIR__
const DATA_DIR = joinpath(CURRENT_DIR, "datasets", "csmr_multimodal")
const TRAIN_IMAGES_DIR = joinpath(DATA_DIR, "images", "train", "visible")
const VAL_IMAGES_DIR   = joinpath(DATA_DIR, "images", "val",   "visible")
const TRAIN_LABELS_DIR = joinpath(DATA_DIR, "labels", "train")
const VAL_LABELS_DIR   = joinpath(DATA_DIR, "labels", "val")

struct Profiler
    enabled::Bool
    timings::Dict{String, Vector{Float64}}
    memory::Dict{String, Vector{Int}}
    profile_file::String
end

function Profiler(; enabled::Bool=true, profile_file::String="profiling_results.json")
    return Profiler(
        enabled,
        Dict{String, Vector{Float64}}(),
        Dict{String, Vector{Int}}(),
        profile_file
    )
end

function start_timer(profiler::Profiler, name::String)
    profiler.enabled || return nothing
    if !haskey(profiler.timings, name)
        profiler.timings[name] = Float64[]
    end
    push!(profiler.timings[name], time())
end

function stop_timer(profiler::Profiler, name::String)
    profiler.enabled || return nothing
    if haskey(profiler.timings, name) && !isempty(profiler.timings[name])
        start_time = pop!(profiler.timings[name])
        elapsed = time() - start_time
        if !haskey(profiler.timings, "$name" * "_elapsed")
            profiler.timings["$name" * "_elapsed"] = Float64[]
        end
        push!(profiler.timings["$name" * "_elapsed"], elapsed)
    end
end

function log_memory(profiler::Profiler, name::String)
    profiler.enabled || return nothing
    if !haskey(profiler.memory, name)
        profiler.memory[name] = Int[]
    end
    push!(profiler.memory[name], Sys.total_memory() - Sys.free_memory())
end

function save_profile(profiler::Profiler)
    profiler.enabled || return nothing
    
    results = Dict(
        "timings" => Dict(
            name => Dict(
                "mean" => mean(vals),
                "std" => std(vals),
                "min" => minimum(vals),
                "max" => maximum(vals),
                "count" => length(vals)
            ) for (name, vals) in profiler.timings if endswith(name, "_elapsed")
        ),
        "memory" => Dict(
            name => Dict(
                "mean" => mean(vals) / 1024 / 1024,
                "peak" => maximum(vals) / 1024 / 1024,
                "final" => last(vals) / 1024 / 1024
            ) for (name, vals) in profiler.memory
        ),
        "timestamp" => string(now())
    )
    
    open(profiler.profile_file, "w") do f
        JSON.pretty(f, results)
    end
    
    @info "Профиль сохранён в $(profiler.profile_file)"
end

struct TensorBoardLogger
    enabled::Bool
    logger::Union{TBLogger, Nothing}
    log_dir::String
    step::Int
end

function TensorBoardLogger(; enabled::Bool=true, log_dir::String="logs")
    if enabled
        mkpath(log_dir)
        logger = TBLogger(log_dir)
    else
        logger = nothing
    end
    return TensorBoardLogger(enabled, logger, log_dir, 0)
end

function log_scalar(tb::TensorBoardLogger, name::String, value::Real; step::Union{Int,Nothing}=nothing)
    tb.enabled || return nothing
    s = step === nothing ? tb.step : step
    log_value(tb.logger, name, value, s)
end

function log_histogram(tb::TensorBoardLogger, name::String, data::Vector{<:Real}; step::Union{Int,Nothing}=nothing)
    tb.enabled || return nothing
    s = step === nothing ? tb.step : step
    log_histogram(tb.logger, name, data, s)
end

function log_text(tb::TensorBoardLogger, name::String, text::String; step::Union{Int,Nothing}=nothing)
    tb.enabled || return nothing
    s = step === nothing ? tb.step : step
    log_text(tb.logger, name, text, s)
end

function log_config(tb::TensorBoardLogger, config::OrderedDict{String,Any})
    tb.enabled || return nothing
    log_text!(tb, "config", JSON.write(config), step=0)
end

function increment_step(tb::TensorBoardLogger)
    tb.step += 1
end

function (model::RussianYOLOModel)(x)
    features = model.backbone(x)
    features = model.head(features)
    return model.pred(features)
end

struct Advanced
    img_size::Int
    hsv_h::Float32
    hsv_s::Float32
    hsv_v::Float32
    flip_lr::Float32
    scale::Float32
    rotation::Float32
    shear::Float32
    mixup::Bool
end

function Advanced(; img_size=640)
    return Advanced(
        img_size,
        0.015f0,
        0.7f0,
        0.4f0,
        0.5f0,
        0.3f0,
        10.0f0,
        0.1f0,
        false
    )
end

function apply(pipeline::AdvancedPipeline, img, boxes, labels)
    img_aug = deepcopy(img)
    boxes_aug = copy(boxes)
    labels_aug = copy(labels)
    if pipeline.hsv_h > 0 || pipeline.hsv_s > 0 || pipeline.hsv_v > 0
        img_aug, boxes_aug, labels_aug = hsv(img_aug, boxes_aug, labels_aug, pipeline)
    end

    if rand() < pipeline.flip_lr
        img_aug, boxes_aug, labels_aug = fliplr(img_aug, boxes_aug, labels_aug, pipeline)
    end
    
    if pipeline.scale > 0 && rand() < 0.5
        img_aug, boxes_aug, labels_aug = scale(img_aug, boxes_aug, labels_aug, pipeline)
    end
    
    if pipeline.rotation > 0 && rand() < 0.3
        img_aug, boxes_aug, labels_aug = rotation(img_aug, boxes_aug, labels_aug, pipeline)
    end
    
    if pipeline.shear > 0 && rand() < 0.3
        img_aug, boxes_aug, labels_aug = shear(img_aug, boxes_aug, labels_aug, pipeline)
    end
    
    return img_aug, boxes_aug, labels_aug
end

function augment(img, boxes, labels, pipeline)
    img_hsv = HSV.(img)
    h, s, v = channelview(img_hsv)
    h .= mod.(h .+ (rand() - 0.5) * pipeline.hsv_h * 360, 360)
    s .= clamp.(s .* (1 + (rand() - 0.5) * pipeline.hsv_s), 0, 1)
    v .= clamp.(v .* (1 + (rand() - 0.5) * pipeline.hsv_v), 0, 1)
    img_aug = RGB.(HSV.(h, s, v))
    
    return img_aug, boxes, labels
end

function fliplr(img, boxes, labels, pipeline)
    img_aug = reverse(img; dims=2)
    boxes_aug = copy(boxes)
    
    if !isempty(boxes_aug)
        w = size(img, 2)
        boxes_aug[:, [1, 3]] = w .- boxes_aug[:, [3, 1]]
    end
    
    return img_aug, boxes_aug, labels
end

function scale(img, boxes, labels, pipeline)
    h, w = size(img)[1:2]
    scale_factor = 1 + (rand() - 0.5) * pipeline.scale
    new_h = round(Int, h * scale_factor)
    new_w = round(Int, w * scale_factor)
    img_aug = imresize(img, (new_h, new_w))
    boxes_aug = copy(boxes)

    if !isempty(boxes_aug)
        boxes_aug[:, [1, 3]] .*= scale_factor
        boxes_aug[:, [2, 4]] .*= scale_factor
    end
    
    if scale_factor > 1
        x_offset = rand(0:new_w - w)
        y_offset = rand(0:new_h - h)
        img_aug = img_aug[y_offset+1:y_offset+h, x_offset+1:x_offset+w, :]
        
        if !isempty(boxes_aug)
            boxes_aug[:, [1, 3]] .-= x_offset
            boxes_aug[:, [2, 4]] .-= y_offset
        end
    else
        pad_x = w - new_w
        pad_y = h - new_h
        pad_left = rand(0:pad_x)
        pad_right = pad_x - pad_left
        pad_top = rand(0:pad_y)
        pad_bottom = pad_y - pad_top
        
        img_aug = padarray(img_aug, (pad_top, pad_bottom), (pad_left, pad_right))
        
        if !isempty(boxes_aug)
            boxes_aug[:, [1, 3]] .+= pad_left
            boxes_aug[:, [2, 4]] .+= pad_top
        end
    end
    
    if !isempty(boxes_aug)
        boxes_aug[:, 1] = clamp.(boxes_aug[:, 1], 1, w)
        boxes_aug[:, 2] = clamp.(boxes_aug[:, 2], 1, h)
        boxes_aug[:, 3] = clamp.(boxes_aug[:, 3], 1, w)
        boxes_aug[:, 4] = clamp.(boxes_aug[:, 4], 1, h)
        valid = (boxes_aug[:, 3] .- boxes_aug[:, 1]) .> 5 .&& (boxes_aug[:, 4] .- boxes_aug[:, 2]) .> 5
        boxes_aug = boxes_aug[valid, :]
        labels_aug = labels[valid]
    end
    
    return img_aug, boxes_aug, labels_aug
end

function rotation(img, boxes, labels, pipeline)
    angle = (rand() - 0.5) * 2 * pipeline.rotation
    img_aug = imrotate(img, angle * π / 180)
    h, w = size(img)[1:2]
    h_new, w_new = size(img_aug)[1:2]
    cx, cy = w/2, h/2
    cx_new, cy_new = w_new/2, h_new/2
    boxes_aug = copy(boxes)

    if !isempty(boxes_aug)
        for i in axes(boxes_aug, 1)
            x1, y1, x2, y2 = boxes_aug[i, :]
            corners = [
                x1 y1
                x2 y1
                x2 y2
                x1 y2
            ]
            corners_rot = similar(corners)
            for j in 1:4
                x, y = corners[j, :]
                xr = (x - cx) * cos(angle) - (y - cy) * sin(angle) + cx_new
                yr = (x - cx) * sin(angle) + (y - cy) * cos(angle) + cy_new
                corners_rot[j, :] = [xr, yr]
            end
            boxes_aug[i, 1] = minimum(corners_rot[:, 1])
            boxes_aug[i, 2] = minimum(corners_rot[:, 2])
            boxes_aug[i, 3] = maximum(corners_rot[:, 1])
            boxes_aug[i, 4] = maximum(corners_rot[:, 2])
        end
        boxes_aug[:, 1] = clamp.(boxes_aug[:, 1], 1, w_new)
        boxes_aug[:, 2] = clamp.(boxes_aug[:, 2], 1, h_new)
        boxes_aug[:, 3] = clamp.(boxes_aug[:, 3], 1, w_new)
        boxes_aug[:, 4] = clamp.(boxes_aug[:, 4], 1, h_new)
        valid = (boxes_aug[:, 3] .> boxes_aug[:, 1]) .&& (boxes_aug[:, 4] .> boxes_aug[:, 2])
        boxes_aug = boxes_aug[valid, :]
        labels_aug = labels[valid]
    else
        labels_aug = labels
    end
    
    return img_aug, boxes_aug, labels_aug
end

function shear(img, boxes, labels, pipeline)
    shear_x = (rand() - 0.5) * pipeline.shear
    shear_y = (rand() - 0.5) * pipeline.shear
    transform = [1 shear_x 0; shear_y 1 0]
    img_aug = warp(img, transform)
    h, w = size(img)[1:2]
    h_new, w_new = size(img_aug)[1:2]
    boxes_aug = copy(boxes)

    if !isempty(boxes_aug)
        for i in axes(boxes_aug, 1)
            x1, y1, x2, y2 = boxes_aug[i, :]
            corners = [
                x1 y1 1
                x2 y1 1
                x2 y2 1
                x1 y2 1
            ]
            corners_trans = corners * [1 shear_y; shear_x 1; 0 0]'
            boxes_aug[i, 1] = minimum(corners_trans[:, 1])
            boxes_aug[i, 2] = minimum(corners_trans[:, 2])
            boxes_aug[i, 3] = maximum(corners_trans[:, 1])
            boxes_aug[i, 4] = maximum(corners_trans[:, 2])
        end

        boxes_aug[:, 1] = clamp.(boxes_aug[:, 1], 1, w_new)
        boxes_aug[:, 2] = clamp.(boxes_aug[:, 2], 1, h_new)
        boxes_aug[:, 3] = clamp.(boxes_aug[:, 3], 1, w_new)
        boxes_aug[:, 4] = clamp.(boxes_aug[:, 4], 1, h_new)
        valid = (boxes_aug[:, 3] .> boxes_aug[:, 1]) .&& (boxes_aug[:, 4] .> boxes_aug[:, 2])
        boxes_aug = boxes_aug[valid, :]
        labels_aug = labels[valid]
    else
        labels_aug = labels
    end
    
    return img_aug, boxes_aug, labels_aug
end

function padarray(img, (top, bottom), (left, right))
    h, w = size(img)[1:2]
    new_h = h + top + bottom
    new_w = w + left + right
    
    if ndims(img) == 3
        padded = zeros(eltype(img), new_h, new_w, size(img, 3))
        padded[top+1:top+h, left+1:left+w, :] = img
    else
        padded = zeros(eltype(img), new_h, new_w)
        padded[top+1:top+h, left+1:left+w] = img
    end
    return padded
end

function mixup(img1, boxes1, labels1, img2, boxes2, labels2; alpha=0.2f0)
    lambda = rand() * alpha + (1 - alpha)/2
    img_mix = lambda .* img1 .+ (1 - lambda) .* img2
    
    if !isempty(boxes1) && !isempty(boxes2)
        boxes_mix = vcat(boxes1, boxes2)
        labels_mix = vcat(labels1, labels2)
    elseif !isempty(boxes1)
        boxes_mix = boxes1
        labels_mix = labels1
    elseif !isempty(boxes2)
        boxes_mix = boxes2
        labels_mix = labels2
    else
        boxes_mix = Matrix{Float32}(undef, 0, 4)
        labels_mix = Int[]
    end
    
    return img_mix, boxes_mix, labels_mix
end

struct PersonDataset
    image_dir::String
    label_dir::String
    img_size::Int
    augment::Bool
    images::Vector{String}
    transform::Union{Nothing, Any}
end

function PersonDataset(
    image_dir::String,
    label_dir::String;
    img_size::Int=640,
    augment::Bool=true
)
    images = filter(f -> occursin(r"\.(jpg|jpeg|png)$"i, f), readdir(image_dir, join=false))
    sort!(images)
    
    transform = nothing
    if augment
        transform = get_pipeline(img_size)
    else
        transform = get_validation_pipeline(img_size)
    end
    
    return PersonDataset(
        image_dir,
        label_dir,
        img_size,
        augment,
        images,
        transform
    )
end

function get_pipeline(img_size::Int)
    return x -> begin
        x = adjust_brightness(x, 0.6 + 0.8 * rand())
        x = adjust_contrast(x, 0.6 + 0.8 * rand())
        
        if rand() < 0.7
            x = rotr90(x, 2)
        end
        
        angle = (rand() - 0.5) * 2 * 30 * π / 180
        x = recenter(imrotate(x, angle), axes(x))
        x = imresize(x, (img_size, img_size))
        x = float32.(x)
        x = (x .- 0.45f0) ./ 0.27f0
        
        return permutedims(x, (3, 2, 1))
    end
end

function get_validation_pipeline(img_size::Int)
    return x -> begin
        x = imresize(x, (img_size, img_size))
        x = float32.(x)
        x = (x .- 0.45f0) ./ 0.27f0
        return permutedims(x, (3, 2, 1))
    end
end

function adjust_brightness(img::Array{RGB{Float32}, 2}, factor::Real)
    return RGB{Float32}.(clamp01.(float.(img) .* factor))
end

function adjust_contrast(img::Array{RGB{Float32}, 2}, factor::Real)
    mean_gray = mean(float.(img))
    return RGB{Float32}.(clamp01.(mean_gray .+ factor .* (float.(img) .- mean_gray)))
end

function clamp01(x::RGB{Float32})
    return RGB{Float32}(clamp01(x.r), clamp01(x.g), clamp01(x.b))
end

clamp01(x::Real) = clamp(x, 0, 1)

function Base.getindex(dataset::PersonDataset, idx::Int)
    img_path = joinpath(dataset.image_dir, dataset.images[idx])
    img = load(img_path)
    if ndims(img) == 2
        img = RGB.(img)
    elseif ndims(img) == 3 && size(img, 3) == 1
        img = RGB.(img[:, :, 1])
    end

    img = Float32.(img)
    
    label_path = joinpath(dataset.label_dir, splitext(dataset.images[idx])[1] * ".txt")
    boxes = Float32[]
    labels = Int[]
    
    if isfile(label_path)
        open(label_path, "r") do f
            for line in eachline(f)
                parts = split(strip(line))
                length(parts) < 5 && continue
                class_id = parse(Int, parts[1])
                x_center = parse(Float32, parts[2])
                y_center = parse(Float32, parts[3])
                width = parse(Float32, parts[4])
                height = parse(Float32, parts[5])
                x1 = (x_center - width/2) * dataset.img_size
                y1 = (y_center - height/2) * dataset.img_size
                x2 = (x_center + width/2) * dataset.img_size
                y2 = (y_center + height/2) * dataset.img_size
                x1 = clamp(x1, 0.0f0, dataset.img_size - 1)
                y1 = clamp(y1, 0.0f0, dataset.img_size - 1)
                x2 = clamp(x2, 0.0f0, dataset.img_size - 1)
                y2 = clamp(y2, 0.0f0, dataset.img_size - 1)
                
                if x2 > x1 && y2 > y1
                    push!(boxes, x1, y1, x2, y2)
                    push!(labels, class_id)
                end
            end
        end
    end
    
    if !isempty(boxes)
        boxes = reshape(boxes, 4, :)'
    else
        boxes = Matrix{Float32}(undef, 0, 4)
    end
    
    img_tensor = dataset.transform(img)
    
    target = (boxes=boxes, labels=labels)
    return img_tensor, target
end

Base.length(dataset::PersonDataset) = length(dataset.images)

struct DataLoader
    dataset::PersonDataset
    batch_size::Int
    shuffle::Bool
    num_workers::Int
    prefetch_factor::Int
    buffer::Channel
end

function DataLoader(dataset::PersonDataset; batch_size=8, shuffle=true, num_workers=4, prefetch_factor=2)
    buffer = Channel(batch_size * prefetch_factor)
    
    loader = DataLoader(dataset, batch_size, shuffle, num_workers, prefetch_factor, buffer)
    
    if num_workers > 0
        @async prefetch_loop(loader)
    end
    
    return loader
end

function Base.iterate(dl::DataLoader, state=nothing)
    indices = if state === nothing
        idxs = 1:length(dl.dataset)
        dl.shuffle ? shuffle(idxs) : idxs
    else
        state
    end
    isempty(indices) && return nothing
    batch_indices = indices[1:min(dl.batch_size, end)]
    images = []
    targets = []
    for i in batch_indices
        img, t = dl.dataset[i]
        push!(images, img)
        push!(targets, t)
    end
    imgs_batch = cat(images..., dims=4)
    return (imgs_batch, targets), indices[dl.batch_size+1:end]
end

Base.length(dataloader::DataLoader) = ceil(Int, length(dataloader.dataset) / dataloader.batch_size)

struct EarlyStopping
    patience::Int
    min_delta::Float64
    best_loss::Float64
    counter::Int
end

EarlyStopping(; patience::Int=10, min_delta::Float64=0.001) = EarlyStopping(patience, min_delta, Inf, 0)

function check(early_stopping::EarlyStopping, current_loss::Float64)
    if current_loss < early_stopping.best_loss - early_stopping.min_delta
        early_stopping.best_loss = current_loss
        early_stopping.counter = 0
        return true
    else
        early_stopping.counter += 1
        if early_stopping.counter >= early_stopping.patience
            return :stop
        end
        return false
    end
end

function detect()::Bool
    if isdir("/sys/class/drm")
        try
            cards = filter(startswith("card"), readdir("/sys/class/drm"))
            if !isempty(cards)
                return true
            end
        catch
        end
    end
    
    if isfile("/usr/sbin/lspci") || isfile("/usr/bin/lspci")
        try
            out = read(`lspci`, String)
            if occursin(r"VGA compatible|3D controller|Display controller", out)
                return true
            end
        catch
        end
    end
    
    if isdir("/dev/dri")
        try
            nodes = readdir("/dev/dri")
            if any(startswith("card"), nodes) || any(startswith("renderD"), nodes)
                return true
            end
        catch
        end
    end
    
    return false
end

function allreduce_gradients(model, distributed::Bool)
    distributed || return nothing
    
    if MPI.Initialized()
        for p in Flux.params(model)
            if p isa AbstractArray
                MPI.Allreduce!(p, +, MPI.COMM_WORLD)
                p ./= MPI.Comm_size(MPI.COMM_WORLD)
            end
        end
    end
end

function async_parameters(model, distributed::Bool)
    distributed || return nothing
    MPI.Initialized() || return nothing
    
    requests = MPI.Request[]
    for p in Flux.trainables(model)
        p isa AbstractArray || continue
        req = MPI.Ibcast!(p, 0, MPI.COMM_WORLD)
        push!(requests, req)
    end
    
    return requests
end

struct Trainer
    model::RussianYOLOModel
    optimizer::Any
    scheduler::Ref{Float64}
    device::Function
    config::OrderedDict{String, Any}
    profiler::Profiler
    tensorboard::TensorBoardLogger
    distributed::Bool
end

function Trainer(; config::OrderedDict{String, Any}=CONFIG)
    model = RussianYOLOModel(
        num_classes=get(config, "num_classes", 2),
        num_anchors=get(config, "num_anchors", 3),
        img_size=config["img_size"]
    )
    
    optimizer = Flux.AdamW(
        config["learning_rate"],
        (0.9, 0.999),
        config["weight_decay"]
    )
    
    scheduler = Ref(1.0)
    profiler = Profiler(enabled=get(config, "profile", true), profile_file=get(config, "profile_output", "profiling.json"))
    
    tensorboard = TensorBoardLogger(
        enabled=get(config, "tensorboard", true),
        log_dir=get(config, "log_dir", "logs")
    )
    
    distributed = setup_distributed()
    
    if distributed
        async_parameters(model, true)
    end
    
    return Trainer(model, optimizer, scheduler, to_device, config, profiler, tensorboard, distributed)
end

function (trainer::Trainer)(x)
    return trainer.device(x)
end

function update_learning_rate(trainer::Trainer, factor::Float64)
    trainer.scheduler[] = factor
    trainer.optimizer.eta = trainer.config["learning_rate"] * factor
end


function train_epoch(trainer::Trainer, train_loader::DataLoader, epoch::Int)
    trainer.model.train()
    total_loss = 0.0f0
    num_batches = 0
    
    trainer.profiler.enabled && start_timer(trainer.profiler, "epoch_$epoch")
    
    @showprogress for (batch_idx, (images, targets)) in enumerate(train_loader)
        trainer.profiler.enabled && start_timer(trainer.profiler, "batch_$batch_idx")
        trainer.profiler.enabled && log_memory(trainer.profiler, "batch_start")
        images = trainer.device(images)
        current_loss = Ref(0.0f0)
        trainer.profiler.enabled && start_timer(trainer.profiler, "forward_backward")
        
        Flux.train!(trainer.optimizer, [trainer.model]) do
            predictions = trainer.model(images)
            loss = compute_loss(predictions, targets, trainer.config)
            current_loss[] = loss
            return loss
        end
        
        trainer.profiler.enabled && stop_timer(trainer.profiler, "forward_backward")
        
        if trainer.distributed
            allreduce_gradients(trainer.model, true)
        end
        
        batch_loss = current_loss[]
        total_loss += batch_loss
        num_batches += 1
        
        if trainer.tensorboard.enabled
            global_step = (epoch - 1) * length(train_loader) + batch_idx
            log_scalar(trainer.tensorboard, "train/batch_loss", batch_loss, step=global_step)
            
            if batch_idx % 10 == 0
                log_scalar(trainer.tensorboard, "train/learning_rate", trainer.optimizer.eta * trainer.scheduler[], step=global_step)
            end
        end
        
        trainer.profiler.enabled && stop_timer(trainer.profiler, "batch_$batch_idx")
        
        if batch_idx % 10 == 0
            @info "Epoch $epoch, Batch $batch_idx, Loss: $(round(batch_loss, digits=4))"
        end
    end
    
    trainer.profiler.enabled && stop_timer(trainer.profiler, "epoch_$epoch")
    
    epoch_loss = total_loss / max(num_batches, 1)
    
    if trainer.tensorboard.enabled
        log_scalar(trainer.tensorboard, "train/epoch_loss", epoch_loss, step=epoch)
        log_scalar(trainer.tensorboard, "train/lr", trainer.optimizer.eta * trainer.scheduler[], step=epoch)
    end
    
    return epoch_loss
end

function decode_boxes(pred_boxes, anchors, grid_x, grid_y, stride)
    tx = sigmoid.(pred_boxes[:, :, 1, :, :])
    ty = sigmoid.(pred_boxes[:, :, 2, :, :])
    tw = exp.(pred_boxes[:, :, 3, :, :])
    th = exp.(pred_boxes[:, :, 4, :, :])
    bx = (tx .+ grid_x') .* stride
    by = (ty .+ grid_y) .* stride
    bw = tw .* anchors[:, 1]' .* stride
    bh = th .* anchors[:, 2]' .* stride
    x1 = bx .- bw./2
    y1 = by .- bh./2
    x2 = bx .+ bw./2
    y2 = by .+ bh./2
    return x1, y1, x2, y2
end

function yolo_loss(predictions, target_list, anchors, stride, img_size, num_classes, num_anchors, coord_scale, noobj_scale, class_scale)
    batch_size = size(predictions, 4)
    H, W = size(predictions, 3), size(predictions, 2)
    pred = reshape(predictions, batch_size, num_anchors, 5 + num_classes, H, W)
    pred_boxes = pred[:, :, 1:4, :, :]
    pred_obj   = sigmoid.(pred[:, :, 5:5, :, :])
    pred_cls   = pred[:, :, 6:end, :, :]
    grid_x = reshape(collect(0:W-1) .* stride .+ stride/2, 1, 1, 1, 1, W)
    grid_y = reshape(collect(0:H-1) .* stride .+ stride/2, 1, 1, 1, H, 1)
    tx = sigmoid.(pred_boxes[:, :, 1:1, :, :])
    ty = sigmoid.(pred_boxes[:, :, 2:2, :, :])
    tw = exp.(pred_boxes[:, :, 3:3, :, :])
    th = exp.(pred_boxes[:, :, 4:4, :, :])
    bx = (tx .+ grid_x) * stride
    by = (ty .+ grid_y) * stride
    bw = tw .* reshape(anchors[:,1], 1, num_anchors, 1, 1, 1) * stride
    bh = th .* reshape(anchors[:,2], 1, num_anchors, 1, 1, 1) * stride
    x1 = bx - bw/2
    y1 = by - bh/2
    x2 = bx + bw/2
    y2 = by + bh/2
    target_obj   = zeros(Float32, batch_size, num_anchors, H, W)
    target_cls   = zeros(Float32, batch_size, num_anchors, num_classes, H, W)
    target_boxes = zeros(Float32, batch_size, num_anchors, 4, H, W)

    for b in 1:batch_size
        boxes_t = target_list[b][1]
        labels_t = target_list[b][2]
        isempty(boxes_t) && continue

        for i in axes(boxes_t, 1)
            box = boxes_t[i,:]
            cx = (box[1] + box[3]) / 2
            cy = (box[2] + box[4]) / 2
            w = box[3] - box[1]
            h = box[4] - box[2]

            col = floor(Int, cx / stride) + 1
            row = floor(Int, cy / stride) + 1
            if col < 1 || col > W || row < 1 || row > H
                continue
            end

            anchor_ious = Float32[]
            for a in 1:num_anchors
                aw = anchors[a,1] * stride
                ah = anchors[a,2] * stride
                iou = intersection_over_union([cx, cy, w, h], [cx, cy, aw, ah])
                push!(anchor_ious, iou)
            end
            best_anchor = argmax(anchor_ious)

            target_obj[b, best_anchor, row, col] = 1.0f0
            target_cls[b, best_anchor, labels_t[i]+1, row, col] = 1.0f0
            tx_target = (cx / stride - (col-1) - 0.5f0)
            ty_target = (cy / stride - (row-1) - 0.5f0)
            tw_target = log(w / (anchors[best_anchor,1] * stride))
            th_target = log(h / (anchors[best_anchor,2] * stride))
            target_boxes[b, best_anchor, 1, row, col] = tx_target
            target_boxes[b, best_anchor, 2, row, col] = ty_target
            target_boxes[b, best_anchor, 3, row, col] = tw_target
            target_boxes[b, best_anchor, 4, row, col] = th_target
        end
    end

    target_obj   = target_obj   |> device(predictions)
    target_cls   = target_cls   |> device(predictions)
    target_boxes = target_boxes |> device(predictions)
    box_mask = target_obj .> 0.5f0

    if any(box_mask)
        box_loss = mean(((pred_boxes .- target_boxes).^2)[box_mask])
    else
        box_loss = zero(Float32)
    end

    obj_loss = Flux.binarycrossentropy(pred_obj, target_obj)
    if any(box_mask)
        pos_pred_cls = pred_cls[box_mask, :]
        pos_target_cls = target_cls[box_mask, :] 
        cls_loss = mean(Flux.logitcrossentropy(pos_pred_cls, pos_target_cls))
    else
        cls_loss = zero(Float32)
    end

    total_loss = coord_scale * box_loss + noobj_scale * obj_loss + class_scale * cls_loss
    return total_loss
end

function intersection_over_union(box1, box2)
    x1 = max(box1[1] - box1[3]/2, box2[1] - box2[3]/2)
    y1 = max(box1[2] - box1[4]/2, box2[2] - box2[4]/2)
    x2 = min(box1[1] + box1[3]/2, box2[1] + box2[3]/2)
    y2 = min(box1[2] + box1[4]/2, box2[2] + box2[4]/2)
    inter = max(0.0f0, x2-x1) * max(0.0f0, y2-y1)
    area1 = box1[3] * box1[4]
    area2 = box2[3] * box2[4]
    union = area1 + area2 - inter
    return union == 0.0f0 ? 0.0f0 : inter / union
end

struct Target
    boxes::Matrix{Float32}    # [N, 4] — x1, y1, x2, y2
    labels::Vector{Int64}     # [N]
end

struct LossConfig
    anchors::Matrix{Float32}
    img_size::Int
    num_classes::Int
    num_anchors::Int
    coord_scale::Float32
    noobj_scale::Float32
    class_scale::Float32
    obj_scale::Float32
end

function compute_loss(
    predictions::AbstractArray{Float32, 4},
    targets::Vector{Target},
    config::LossConfig,
)::Float32
    batch_size = size(predictions, 1)
    H = size(predictions, 2)
    W = size(predictions, 3)
    @assert length(targets) == batch_size "targets ($(length(targets))) != batch ($batch_size)"
    @assert H == W "predictions должны быть квадратными: H=$H, W=$W"
    @assert H > 0 "H должно быть > 0"
    stride = Float32(config.img_size) / Float32(H)
    return yolo_loss(
        predictions,
        targets,
        config.anchors,
        stride,
        config.num_classes,
        config.num_anchors,
        config.coord_scale,
        config.noobj_scale,
        config.class_scale,
    )
end

function validate(trainer::Trainer, val_loader::DataLoader, epoch::Int)
    trainer.model.eval()
    total_loss = 0.0
    num_batches = 0
    all_losses = Float64[]
    
    trainer.profiler.enabled && start_timer(trainer.profiler, "validation_$epoch")
    
    for (images, targets) in val_loader
        images = trainer.device(images)
        predictions = trainer.model(images)
        loss = compute_loss(predictions, targets, trainer.config)
        
        push!(all_losses, loss)
        total_loss += loss
        num_batches += 1
    end
    
    trainer.profiler.enabled && stop_timer(trainer.profiler, "validation_$epoch")
    val_loss = total_loss / max(num_batches, 1)
    
    if trainer.tensorboard.enabled
        log_scalar(trainer.tensorboard, "val/loss", val_loss, step=epoch)
        log_histogram(trainer.tensorboard, "val/loss_distribution", all_losses, step=epoch)
    end
    
    return val_loss
end

function save_model(model::RussianYOLOModel, path::String; format::Symbol=:bson)::Bool
    try
        mkpath(dirname(path))
        if format == :bson
            BSON.bson(path, Dict(
                :model          => model,
                :config         => model.config,
                :version        => model.version,
                :created_at     => model.created_at,
                :save_timestamp => now(),
            ))
        end

        @info "Model saved" path=path format=format
        return true
    catch e
        @error "Failed to save model" path=path exception=(e, catch_backtrace())
        return false
    end
end

function train(trainer::Trainer; num_epochs::Int=100)::String
    @info "Подготовка данных..."
    log_config(trainer.tensorboard, trainer.config)
    
    if !isdir(TRAIN_IMAGES_DIR)
        @error "Директория с данными не найдена: $TRAIN_IMAGES_DIR"
        error("Data directory not found: $TRAIN_IMAGES_DIR")
    end
    
    train_dataset = PersonDataset(
        TRAIN_IMAGES_DIR, TRAIN_LABELS_DIR;
        img_size=trainer.config["img_size"],
        augment=true
    )
    val_dataset = PersonDataset(
        VAL_IMAGES_DIR, VAL_LABELS_DIR;
        img_size=trainer.config["img_size"],
        augment=false
    )
    
    train_loader = DataLoader(train_dataset; 
        batch_size=trainer.config["batch_size"],
        shuffle=true
    )

    val_loader = DataLoader(val_dataset;
        batch_size=trainer.config["batch_size"],
        shuffle=false
    )
    
    @info "Обучающий датасет: $(length(train_dataset)) изображений"
    @info "Валидационный датасет: $(length(val_dataset)) изображений"
    
    timestamp = Dates.format(Dates.now(), "YYYYmmdd_HHMMSS")
    output_dir = joinpath("outputs", "augmented_train_$timestamp")
    mkpath(output_dir)
    
    early_stopping = EarlyStopping(
        patience=trainer.config["early_stopping_patience"],
        min_delta=trainer.config["early_stopping_min_delta"]
    )
    
    best_loss = Inf
    train_losses = Float64[]
    val_losses = Float64[]
    
    @info "Начало обучения..."
    
    for epoch in 1:num_epochs
        @info "="^60
        @info "Epoch $epoch/$num_epochs"
        @info "="^60
        
        trainer.profiler.enabled && log_memory(trainer.profiler, "epoch_start_$epoch")
        
        train_loss = train_epoch(trainer, train_loader, epoch)
        val_loss = validate(trainer, val_loader, epoch)
        
        push!(train_losses, train_loss)
        push!(val_losses, val_loss)
        
        if epoch > 1 && val_losses[end] >= val_losses[end-1]
            trainer.scheduler[] *= 0.5
            @info "Learning rate уменьшен до: $(trainer.optimizer.eta * trainer.scheduler[])"
        end
        
        @info "Epoch $epoch - Train: $(round(train_loss, digits=4)), Val: $(round(val_loss, digits=4))"
        result = check(early_stopping, val_loss)
        if result === true
            best_loss = val_loss
            save_model(trainer.model, joinpath(output_dir, "best_russian_model.bson"))
            @info "Новая лучшая модель! Loss: $(round(val_loss, digits=4))"
            save_checkpoint(trainer, epoch, train_loss, val_loss, output_dir)
        elseif result === :stop
            @info "Ранняя остановка на эпохе $epoch"
            break
        end
        
        trainer.profiler.enabled && log_memory(trainer.profiler, "epoch_end_$epoch")
        
        if trainer.distributed
            async_parameters(trainer.model, true)
        end
    end
    
    save_profile(trainer.profiler)
    save_training_plot(train_losses, val_losses, output_dir)
    save_model(trainer.model, joinpath(output_dir, "final_russian_model.bson"))

    open(joinpath(output_dir, "config.json"), "w") do f
        JSON.pretty(f, trainer.config)
    end
    
    if trainer.tensorboard.enabled
        @info "TensorBoard логи сохранены в $(trainer.tensorboard.log_dir)"
        @info "Запустите: tensorboard --logdir=$(trainer.tensorboard.log_dir)"
    end
    
    @info "Обучение завершено! Результаты в: $output_dir"
    
    return output_dir
end

function save_checkpoint(
    trainer::Trainer,
    epoch::Int,
    train_loss::Float64,
    val_loss::Float64,
    output_dir::String,
)
    isdir(output_dir) || mkpath(output_dir)
    
    checkpoint = Dict(
        "model"      => trainer.model,
        "optimizer"  => trainer.optimizer,
        "epoch"      => epoch,
        "train_loss" => train_loss,
        "val_loss"   => val_loss,
        "config"     => Dict(trainer.config),
        "timestamp"  => string(now()),
    )
    
    checkpoint_path = joinpath(output_dir, "checkpoint_epoch_$(lpad(epoch, 4, '0')).bson")
    
    try
        BSON.bson(checkpoint_path, checkpoint)
        @info "Чекпоинт сохранён: $checkpoint_path"
    catch e
        @error "Не удалось сохранить чекпоинт: $checkpoint_path" exception=(e, catch_backtrace())
        rethrow()
    end
    
    return checkpoint_path
end

function save_training_plot(train_loss::Vector{Float64}, val_loss::Vector{Float64}, output_dir::String)
    try
        open(joinpath(output_dir, "training_log.txt"), "w") do f
            println(f, "Training Loss History")
            println(f, "="^50)
            for (i, (tr, val)) in enumerate(zip(train_loss, val_loss))
                println(f, "Epoch $i: Train=$(round(tr, digits=4)), Val=$(round(val, digits=4))")
            end
        end
        
        using Serialization
        serialize(joinpath(output_dir, "loss_history.jls"), Dict(
            "train" => train_loss,
            "val" => val_loss
        ))
        
        @info "Графики обучения сохранены в $output_dir"
    catch e
        @warn "Не удалось сохранить графики: $e"
    end
end

function prefetch_loop(loader::PrefetchDataLoader)
    indices = 1:length(loader.dataset)
    if loader.shuffle
        indices = shuffle(indices)
    end
    
    for i in 1:loader.batch_size:length(indices)
        batch_indices = indices[
            i:min(
                i+loader.batch_size-1, 
                end
            )
        ]
        
        images = []
        targets = []
        for idx in batch_indices
            img, target = loader.dataset[idx]
            push!(images, img)
            push!(targets, target)
        end
        
        imgs_batch = cat(images..., dims=4)
        
        put!(loader.buffer, (imgs_batch, targets))
    end
    
    close(loader.buffer)
end

function Base.iterate(loader::PrefetchDataLoader, state=nothing)
    if isready(loader.buffer) || !isclosed(loader.buffer)
        return take!(loader.buffer), nothing
    else
        return nothing
    end
end

function compute_map(predictions, targets; iou_threshold=0.5)
    all_detections = []
    all_groundtruths = []
    
    for (pred, target) in zip(predictions, targets)
        push!(all_detections, pred)
        push!(all_groundtruths, target)
    end

    detections_sorted = sort(all_detections; by=x->x.conf, rev=true)
    
    tp = zeros(length(detections_sorted))
    fp = zeros(length(detections_sorted))
    
    for (i, det) in enumerate(detections_sorted)
        best_iou = 0.0
        best_match = nothing
        
        for (j, gt) in enumerate(all_groundtruths)
            iou = compute_iou(det.bbox, gt.bbox)
            if iou > best_iou && gt.matched == false
                best_iou = iou
                best_match = j
            end
        end
        
        if best_iou >= iou_threshold && !isnothing(best_match)
            tp[i] = 1
            all_groundtruths[best_match].matched = true
        else
            fp[i] = 1
        end
    end
    
    tp_cumsum = cumsum(tp)
    fp_cumsum = cumsum(fp)
    precision = tp_cumsum ./ (tp_cumsum + fp_cumsum)
    recall = tp_cumsum / length(all_groundtruths)
    ap = 0.0

    for t in 0:0.1:1.0
        p = maximum(precision[recall .>= t], init=0.0)
        ap += p / 11
    end
    
    return ap
end

function main()
    @info "="^60
    @info "Запуск обучения на аугментированных данных"
    @info "="^60
    
    try
        trainer = Trainer()
        output_dir = train!(trainer; num_epochs=CONFIG["num_epochs"])
        @info "Обучение завершено! Результаты в: $output_dir"
        return true
    catch e
        @error "Критическая ошибка: $e"
        @error stacktrace(catch_backtrace())
        return false
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

end  # module Training
