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

module Models

using Flux
using Flux: Conv, BatchNorm, MaxPool
using BSON
using Random
using Statistics
using Dates
using Printf
using Logging

export RussianYOLOModel, RussianYOLOConfig, ConvBlock, DetectionHead, create_model, save_model, load_model, export_model, analyze_model, count_parameters, initialize_weights, model_to_device, register_device, predict

Base.@kwdef struct RussianYOLOConfig
    num_classes::Int = 2
    input_size::Int = 640
    num_anchors::Int = 3
    backbone_channels::Vector{Int} = [32, 64, 128, 256, 512, 1024]
    backbone_blocks::Vector{Int} = [1, 2, 8, 8, 4]
    neck_channels::Int = 512
    head_channels::Int = 256
    anchors::Vector{Tuple{Float32, Float32}} = [
        (10f0, 13f0), (16f0, 30f0), (33f0, 23f0),
        (30f0, 61f0), (62f0, 45f0), (59f0, 119f0),
        (116f0, 90f0), (156f0, 198f0), (373f0, 326f0)
    ]
    strides::Vector{Int} = [8, 16, 32]
    activation::Symbol = :leakyrelu
    use_batchnorm::Bool = true
    label_names::Vector{String} = ["person", "vehicle"]
end

function validate_config(config::RussianYOLOConfig)
    valid = true
    if config.num_classes < 1
        @error "validate_config: num_classes must be >= 1" value = config.num_classes
        valid = false
    end
    if config.num_anchors < 1
        @error "validate_config: num_anchors must be >= 1" value = config.num_anchors
        valid = false
    end
    if length(config.backbone_channels) != 6
        @error "validate_config: backbone_channels must hold 6 entries" value = length(config.backbone_channels)
        valid = false
    end
    if length(config.backbone_blocks) != 5
        @error "validate_config: backbone_blocks must hold 5 entries" value = length(config.backbone_blocks)
        valid = false
    end
    if isempty(config.strides)
        @error "validate_config: strides must not be empty"
        valid = false
    elseif config.input_size % maximum(config.strides) != 0
        @error "validate_config: input_size must be divisible by the largest stride" input_size = config.input_size stride = maximum(config.strides)
        valid = false
    end
    if length(config.anchors) != config.num_anchors * length(config.strides)
        @error "validate_config: anchors must provide num_anchors * nscales entries" anchors = length(config.anchors) expected = config.num_anchors * length(config.strides)
        valid = false
    end
    if length(config.label_names) != config.num_classes
        @error "validate_config: label_names must provide num_classes entries" labels = length(config.label_names) expected = config.num_classes
        valid = false
    end
    if config.head_channels < 1 || config.neck_channels < 1
        @error "validate_config: head_channels and neck_channels must be >= 1" head = config.head_channels neck = config.neck_channels
        valid = false
    end
    if valid
        return config
    end
    @error "validate_config: invalid configuration detected, falling back to defaults"
    return RussianYOLOConfig()
end

apply_activation(x, ::Val{:identity}) = x
apply_activation(x, ::Val{:leakyrelu}) = max.(x, 0.01f0 .* x)
apply_activation(x, ::Val{:relu}) = max.(x, zero(x))
apply_activation(x, ::Val{:silu}) = x ./ (1f0 .+ exp.(-x))
function apply_activation(x, ::Val{:mish})
    sp = map(y -> y > 0 ? y + log1p(exp(-y)) : log1p(exp(y)), x)
    return x .* tanh.(sp)
end
apply_activation(x, s::Symbol) = apply_activation(x, Val(s))

_children_reflect(::Nothing) = ()
_children_reflect(::Symbol) = ()
_children_reflect(::AbstractString) = ()
_children_reflect(::Function) = ()
_children_reflect(::Type) = ()

function _children_reflect(x::Tuple)
    out = ()
    for child in x
        out = (out..., _children_reflect(child)...)
    end
    return out
end

function _children_reflect(x::NamedTuple)
    out = ()
    for child in values(x)
        out = (out..., _children_reflect(child)...)
    end
    return out
end

function _children_reflect(x)
    isstructtype(typeof(x)) || return ()
    out = ()
    for f in fieldnames(typeof(x))
        out = (out..., _children_reflect(getfield(x, f))...)
    end
    return out
end

children_of(x) = isdefined(Flux, :children) ? Flux.children(x) : _children_reflect(x)

function param_arrays(model)
    isdefined(Flux, :params) && return collect(Flux.params(model))
    return collect(values(Flux.trainable(model)))
end

function avg_downsample(x::AbstractArray, factor::Int=2)
    nd = ndims(x)
    if !(nd == 3 || nd == 4)
        @error "avg_downsample: expected a 3D or 4D feature map, returning input unchanged" ndims = nd
        return x
    end
    if factor < 1
        @error "avg_downsample: factor must be >= 1, returning input unchanged" factor = factor
        return x
    end
    if factor == 1
        return x
    end
    hd = nd - 1
    wd = nd
    h = size(x, hd)
    w = size(x, wd)
    if !(h % factor == 0 && w % factor == 0)
        @error "avg_downsample: spatial size must be divisible by factor, returning input unchanged" height = h width = w factor = factor
        return x
    end
    oh = h ÷ factor
    ow = w ÷ factor
    pre = ntuple(i -> size(x, i), hd - 1)
    blocks = reshape(x, pre..., oh, factor, ow, factor)
    pooled = mean(blocks; dims = (hd + 1, nd + 2))
    return reshape(pooled, pre..., oh, ow)
end

function nearest_upsample(x::AbstractArray, H::Int, W::Int)
    nd = ndims(x)
    if !(nd == 3 || nd == 4)
        @error "nearest_upsample: expected a 3D or 4D feature map, returning input unchanged" ndims = nd
        return x
    end
    if H < 1 || W < 1
        @error "nearest_upsample: target size must be positive, returning input unchanged" height = H width = W
        return x
    end
    hd = nd - 1
    wd = nd
    h = size(x, hd)
    w = size(x, wd)
    if H == h && W == w
        return x
    end
    if H % h == 0 && W % w == 0
        reps = ntuple(i -> i == hd ? H ÷ h : (i == wd ? W ÷ w : 1), nd)
        return repeat(x, reps...)
    end
    row_idx = [clamp(div((i - 1) * h, H) + 1, 1, h) for i in 1:H]
    col_idx = [clamp(div((j - 1) * w, W) + 1, 1, w) for j in 1:W]
    return nd == 4 ? x[:, :, row_idx, col_idx] : x[:, row_idx, col_idx]
end

struct NoBN
    channels::Int
end

Flux.@functor NoBN

(a::NoBN)(x) = x

struct ConvBlock
    conv::Any
    bn::Any
    activation::Symbol
end

Flux.@functor ConvBlock

function ConvBlock(
    in_ch::Integer,
    out_ch::Integer,
    kernel::Tuple=(3, 3);
    stride::Tuple=(1, 1),
    pad::Tuple=(1, 1),
    activation::Symbol=:leakyrelu,
    use_bn::Bool=true
)
    conv = Conv(kernel, Int(in_ch) => Int(out_ch); stride=stride, pad=pad, bias=!use_bn)
    bn = use_bn ? BatchNorm(Int(out_ch)) : NoBN(Int(out_ch))
    return ConvBlock(conv, bn, activation)
end

function (b::ConvBlock)(x)
    return apply_activation(b.bn(b.conv(x)), b.activation)
end

struct ResidualBlock
    conv1::Any
    bn1::Any
    conv2::Any
    bn2::Any
    residual::Bool
    activation::Symbol
end

Flux.@functor ResidualBlock

function ResidualBlock(
    in_ch::Integer,
    hidden_ch::Integer = in_ch;
    activation::Symbol = :leakyrelu,
    use_bn::Bool = true
)
    c1 = Conv((1, 1), Int(in_ch) => Int(hidden_ch); bias = false)
    c2 = Conv((3, 3), Int(hidden_ch) => Int(in_ch); pad = (1, 1), bias = false)
    n1 = use_bn ? BatchNorm(Int(hidden_ch)) : NoBN(Int(hidden_ch))
    n2 = use_bn ? BatchNorm(Int(in_ch)) : NoBN(Int(in_ch))
    return ResidualBlock(c1, n1, c2, n2, in_ch == hidden_ch, activation)
end

function (b::ResidualBlock)(x)
    y = apply_activation(b.bn1(b.conv1(x)), b.activation)
    y = b.bn2(b.conv2(y))
    return b.residual ? apply_activation(y .+ x, b.activation) : apply_activation(y, b.activation)
end

struct CSPBlock
    cv1::Any
    bottlenecks::Any
    cv2::Any
    cv3::Any
    shortcut::Any
    activation::Symbol
end

Flux.@functor CSPBlock

function CSPBlock(
    in_ch::Integer,
    out_ch::Integer,
    num_blocks::Integer;
    activation::Symbol = :leakyrelu,
    use_bn::Bool = true
)
    if num_blocks < 1
        @error "CSPBlock: num_blocks must be >= 1, falling back to 1" value = num_blocks
        num_blocks = 1
    end

    return CSPBlock(
        ConvBlock(in_ch, out_ch, (3, 3); pad = (1, 1), activation = activation, use_bn = use_bn),
        Chain([ResidualBlock(out_ch; activation = activation, use_bn = use_bn) for _ in 1:num_blocks]...),
        ConvBlock(out_ch, out_ch, (1, 1); pad = (0, 0), activation = activation, use_bn = use_bn),
        ConvBlock(2 * out_ch, out_ch, (1, 1); pad = (0, 0), activation = activation, use_bn = use_bn),
        ConvBlock(in_ch, out_ch, (1, 1); pad = (0, 0), activation = activation, use_bn = use_bn),
        activation,
    )
end

function (b::CSPBlock)(x)
    main = b.cv2(b.bottlenecks(b.cv1(x)))
    return b.cv3(cat(main, b.shortcut(x); dims=3))
end

struct SPPFBlock
    conv1::Any
    pool5::Any
    pool9::Any
    pool13::Any
    conv2::Any
    activation::Symbol
end

Flux.@functor SPPFBlock

function SPPFBlock(
    in_ch::Integer,
    mid_ch::Integer,
    out_ch::Integer = in_ch;
    activation::Symbol = :leakyrelu,
    use_bn::Bool = true
)
    return SPPFBlock(
        ConvBlock(in_ch, mid_ch, (1, 1); pad = (0, 0), activation = activation, use_bn = use_bn),
        MaxPool((5, 5); stride = (1, 1), pad = (2, 2)),
        MaxPool((9, 9); stride = (1, 1), pad = (4, 4)),
        MaxPool((13, 13); stride = (1, 1), pad = (6, 6)),
        ConvBlock(4 * mid_ch, out_ch, (1, 1); pad = (0, 0), activation = activation, use_bn = use_bn),
        activation,
    )
end

function (b::SPPFBlock)(x)
    y = b.conv1(x)
    return b.conv2(cat(y, b.pool5(y), b.pool9(y), b.pool13(y); dims=3))
end

struct Backbone
    stem::Union{ConvBlock, Conv}
    down1::MaxPool
    stage1::CSPBlock
    down2::MaxPool
    stage2::CSPBlock
    down3::MaxPool
    stage3::CSPBlock
    down4::MaxPool
    stage4::CSPBlock
    stage5::CSPBlock
    spp::SPPFBlock
    activation::Symbol
end

Flux.@functor Backbone

function Backbone(config::RussianYOLOConfig)
    ch = config.backbone_channels
    nb = config.backbone_blocks
    act = config.activation
    bn = config.use_batchnorm
    return Backbone(
        ConvBlock(3, ch[1], (3, 3); stride = (2, 2), pad = (1, 1), activation = act, use_bn = bn),
        MaxPool((2, 2); stride = (2, 2)),
        CSPBlock(ch[1], ch[2], nb[1]; activation = act, use_bn = bn),
        MaxPool((2, 2); stride = (2, 2)),
        CSPBlock(ch[2], ch[3], nb[2]; activation = act, use_bn = bn),
        MaxPool((2, 2); stride = (2, 2)),
        CSPBlock(ch[3], ch[4], nb[3]; activation = act, use_bn = bn),
        MaxPool((2, 2); stride = (2, 2)),
        CSPBlock(ch[4], ch[5], nb[4]; activation = act, use_bn = bn),
        CSPBlock(ch[5], ch[6], nb[5]; activation = act, use_bn = bn),
        SPPFBlock(ch[6], ch[6], ch[6]; activation = act, use_bn = bn),
        act,
    )
end

function (b::Backbone)(x)
    x = b.stem(x)
    x = b.stage1(b.down1(x))
    p3 = b.stage2(b.down2(x))
    p4 = b.stage3(b.down3(p3))
    p5 = b.stage4(b.down4(p4))
    p5 = b.spp(b.stage5(p5))
    return (p3, p4, p5)
end

struct PANetNeck
    lateral5::ConvBlock
    lateral4::ConvBlock
    lateral3::ConvBlock
    down5::ConvBlock
    down4::ConvBlock
    up4::ConvBlock
    up5::ConvBlock
    out_ch::Int
end

Flux.@functor PANetNeck

function PANetNeck(in_chs::AbstractVector{<:Integer}, out_ch::Integer; use_bn::Bool = true)
    o = Int(out_ch)
    c5, c4, c3 = if length(in_chs) >= 3
        (Int(in_chs[1]), Int(in_chs[2]), Int(in_chs[3]))
    else
        @error "PANetNeck: expected 3 input feature maps (p5, p4, p3), using out_ch for the missing ones" value = length(in_chs)
        (o, o, o)
    end
    return PANetNeck(
        ConvBlock(c5, o, (1, 1); pad = (0, 0), use_bn = use_bn),
        ConvBlock(c4, o, (1, 1); pad = (0, 0), use_bn = use_bn),
        ConvBlock(c3, o, (1, 1); pad = (0, 0), use_bn = use_bn),
        ConvBlock(2 * o, o, (1, 1); pad = (0, 0), use_bn = use_bn),
        ConvBlock(2 * o, o, (1, 1); pad = (0, 0), use_bn = use_bn),
        ConvBlock(2 * o, o, (1, 1); pad = (0, 0), use_bn = use_bn),
        ConvBlock(2 * o, o, (3, 3); pad = (1, 1), use_bn = use_bn),
        o,
    )
end

function (n::PANetNeck)(p3, p4, p5)
    l3 = n.lateral3(p3)
    l4 = n.lateral4(p4)
    l5 = n.lateral5(p5)
    d4 = n.down5(cat(l4, nearest_upsample(l5, size(l4, 3), size(l4, 4)); dims=3))
    d3 = n.down4(cat(l3, nearest_upsample(d4, size(l3, 3), size(l3, 4)); dims=3))
    b4 = n.up4(cat(avg_downsample(d3), d4; dims=3))
    b5 = n.up5(cat(avg_downsample(b4), l5; dims=3))
    return (p3=d3, p4=b4, p5=b5)
end

struct DetectionHead
    blocks::Chain
    out::Conv
    out_channels::Int
    stride::Int
end

Flux.@functor DetectionHead

function DetectionHead(
    in_ch::Integer,
    num_anchors::Integer,
    num_classes::Integer;
    hidden::Integer = in_ch,
    activation::Symbol = :leakyrelu,
    stride::Integer = 1,
    use_bn::Bool = true
)
    out_ch = Int(num_anchors) * (5 + Int(num_classes))
    return DetectionHead(
        Chain(
            ConvBlock(
                Int(in_ch),
                Int(hidden),
                (3, 3);
                pad = (1, 1),
                activation = activation,
                use_bn = use_bn
            ),
            ConvBlock(
                Int(hidden),
                Int(hidden),
                (3, 3);
                pad = (1, 1),
                activation = activation,
                use_bn = use_bn
            )
        ),
        Conv((1, 1), Int(hidden) => out_ch; bias = true),
        out_ch,
        Int(stride),
    )
end

function (h::DetectionHead)(x)
    return h.out(h.blocks(x))
end

struct RussianYOLOModel
    backbone::Backbone
    neck::PANetNeck
    heads::NamedTuple
    config::RussianYOLOConfig
    num_classes::Int
    num_anchors::Int
end

Flux.@functor RussianYOLOModel

function RussianYOLOModel(
    ;
    config::Union{Nothing,RussianYOLOConfig}=nothing,
    img_size::Union{Integer,Nothing}=nothing,
    kwargs...
)
    if img_size !== nothing
        kwargs = (; kwargs..., input_size=Int(img_size))
    end
    cfg = if config === nothing
        RussianYOLOConfig(; kwargs...)
    else
        names = fieldnames(RussianYOLOConfig)
        base = NamedTuple{names}(ntuple(i -> getfield(config, names[i]), length(names)))
        merged = merge(base, (; kwargs...))
        RussianYOLOConfig(; merged...)
    end
    cfg = validate_config(cfg)
    ch = cfg.backbone_channels
    use_bn = cfg.use_batchnorm
    heads = (
        head3 = DetectionHead(
            cfg.neck_channels,
            cfg.num_anchors,
            cfg.num_classes;
            hidden = cfg.head_channels,
            activation = cfg.activation,
            stride = cfg.strides[1],
            use_bn = use_bn
        ),
        head4 = DetectionHead(
            cfg.neck_channels,
            cfg.num_anchors,
            cfg.num_classes;
            hidden = cfg.head_channels,
            activation = cfg.activation,
            stride = cfg.strides[2],
            use_bn = use_bn
        ),
        head5 = DetectionHead(
            cfg.neck_channels,
            cfg.num_anchors,
            cfg.num_classes;
            hidden = cfg.head_channels,
            activation = cfg.activation,
            stride = cfg.strides[3],
            use_bn = use_bn
        ),
    )
    return RussianYOLOModel(
        Backbone(cfg),
        PANetNeck([ch[6], ch[4], ch[3]], cfg.neck_channels; use_bn = use_bn),
        heads,
        cfg,
        cfg.num_classes,
        cfg.num_anchors,
    )
end

function (m::RussianYOLOModel)(x)
    p3, p4, p5 = m.backbone(x)
    n3, n4, n5 = m.neck(p3, p4, p5)
    return (m.heads.head3(n3), m.heads.head4(n4), m.heads.head5(n5))
end

predict(model::RussianYOLOModel, x) = model(x)
heads_of(model::RussianYOLOModel) = (model.heads.head3, model.heads.head4, model.heads.head5)
create_model(; kwargs...) = RussianYOLOModel(; kwargs...)

_raw(x::Array) = x
_raw(x::AbstractArray) = x
_raw(x) = hasproperty(x, :data) ? _raw(getproperty(x, :data)) : x

function _fill(x, v)
    a = _raw(x)
    fill!(a, v)
    return x
end

function _fan(weight::AbstractArray)
    sz = size(weight)
    nd = length(sz)
    return (prod(sz[1:(nd - 2)]), sz[nd - 1], sz[nd])
end

function _init_weight(w::AbstractArray)
    _, fan_in, fan_out = _fan(w)
    limit = sqrt(3f0 / max(fan_in + fan_out, 1))
    @inbounds for i in eachindex(w)
        w[i] = (rand(Float32) * 2f0 - 1f0) * limit
    end
    return w
end

function _init_module(m)
    if m isa Conv
        _init_weight(_raw(m.weight))
        (m.bias === nothing || m.bias === false) || _fill(m.bias, 0f0)
        return m
    elseif m isa BatchNorm
        _fill(m.β, 0f0)
        _fill(m.γ, 1f0)
        _fill(m.μ, 0f0)
        _fill(m.σ², 1f0)
        return m
    end
    for child in children_of(m)
        _init_module(child)
    end
    return m
end

function initialize_weights(model::RussianYOLOModel; obj_prob::Real=0.01, verbose::Bool=false)
    if !(0f0 < obj_prob < 1f0)
        @error "initialize_weights: obj_prob must lie strictly inside (0, 1), falling back to 0.01" value=obj_prob
        obj_prob = 0.01f0
    end

    _init_module(model)
    per = 5 + model.num_classes
    bias = Float32(-log((1 - obj_prob) / obj_prob))
    for head in heads_of(model)
        b = _raw(head.out.bias)
        for a in 1:model.num_anchors
            b[(a - 1) * per + 5] = bias
        end
    end
    return model
end

count_parameters(model) = sum(length, param_arrays(model); init=0)

const DEVICE_ADAPTERS = Dict{Symbol, Any}()

function _dir_has_entries(path)
    isdir(path) || return false
    try
        return !isempty(readdir(path))
    catch
        return false
    end
end

function detect_device()::Symbol
    _dir_has_entries("/dev/dri") && return :gpu
    _dir_has_entries("/dev/accel") && return :npu
    (haskey(ENV, "TPU_NAME") || isdir("/dev/tpu")) && return :tpu
    return :gpu
end

function device_symbol(device::Symbol)::Symbol
    return device
end

function device_symbol(device::AbstractString)::Symbol
    s = lowercase(strip(device))
    return Symbol(s)
end

device_symbol(device)::Symbol = detect_device()

function register_device(name::Symbol, adapter)
    DEVICE_ADAPTERS[name] = adapter
    return name
end

function register_device(name::AbstractString, adapter)
    return register_device(device_symbol(name), adapter)
end

function model_to_device(model, device = detect_device())
    name = device_symbol(device)
    if !haskey(DEVICE_ADAPTERS, name)
        @warn "model_to_device: no adapter registered for device, leaving the model untouched" device = name registered = sort!(collect(keys(DEVICE_ADAPTERS)))
        return model
    end
    return Flux.fmap(DEVICE_ADAPTERS[name], model)
end

function _count_layers(x)
    n_conv = 0
    n_bn = 0
    if x isa Conv
        n_conv += 1
    elseif x isa BatchNorm
        n_bn += 1
    elseif x isa Tuple || x isa NamedTuple
        for child in x
            c_conv, c_bn = _count_layers(child)
            n_conv += c_conv
            n_bn += c_bn
        end
    else
        for child in children_of(x)
            c_conv, c_bn = _count_layers(child)
            n_conv += c_conv
            n_bn += c_bn
        end
    end
    return (n_conv, n_bn)
end

function analyze_model(model)
    n_conv, n_bn = _count_layers(model)
    total = count_parameters(model)
    return (
        total_parameters = total,
        num_layers = n_conv + n_bn,
        num_conv = n_conv,
        num_batchnorm = n_bn,
        model_size_mb = round(total * sizeof(Float32) / 1024^2; digits = 3),
    )
end

function save_model(model, path::AbstractString; metadata = Dict{String, Any}())
    p = String(path)
    if !endswith(lowercase(p), ".bson")
        @error "save_model: only the .bson format is supported, nothing was written" path = p
        return nothing
    end
    try
        mkpath(dirname(abspath(p)))
        BSON.@save p model metadata
        return p
    catch err
        @error "save_model: failed to write the checkpoint" path = p exception = err
        return nothing
    end
end

function load_model(path::AbstractString)
    p = String(path)
    if !isfile(p)
        @error "load_model: checkpoint does not exist" path = p
        return nothing
    end
    try
        return BSON.load(p)
    catch err
        @error "load_model: failed to read the checkpoint" path = p exception = err
        return nothing
    end
end

function export_model(model, path::AbstractString; metadata = Dict{String, Any}())
    p = String(path)
    if endswith(lowercase(p), ".bson")
        return save_model(model, p; metadata = metadata)
    end
    @error "export_model: unsupported export format, falling back to the .bson checkpoint" path = p
    return save_model(model, splitext(p)[1] * ".bson"; metadata = metadata)
end

end # module Models