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

module BPLADetection

using Base: invokelatest
using Dates: now
using JSON
using Images: load, save
using ColorTypes: RGB

const VERSION = "1.0.0"

export main, run_web, run_serve, run_train, run_server, train_model, detect_image

include("Config.jl")
include("Models.jl")
include("Training.jl")
include("app.jl")
include("serve.jl")
include("__main__.jl")

using .Config, .Models, .Training, .App, .Serve, .Main
using .Serve: DetectionResult

run_server(;host="0.0.0.0",port=8000) = invokelatest(Serve.run_server, host=host, port=port)
train_model!(m,l;kwargs...) = invokelatest(Training.train_model!, m, l; kwargs...)

function main(;mode="web",input=nothing,output=nothing,host="0.0.0.0",port=7860)::Int
    @info "BPLA v$VERSION mode=$mode"
    
    if mode=="web" 
        invokelatest(App.run_server, host=host, port=port)
    elseif mode=="serve" 
        invokelatest(Serve.run_server, host=host, port=port)
    elseif mode=="train" 
        invokelatest(Training.train_augmented)
        
    elseif mode=="inference" 
        if input === nothing
            @error "Для режима inference нужно указать input (путь к изображению)"
            return 1
        end
        
        try
            @info "Загрузка модели..."
            config = InferenceConfig()
            engine = InferenceEngine(; config=config)
            
            @info "Загрузка изображения: $input"
            if !isfile(input)
                @error "Файл не найден: $input"
                return 1
            end
            
            img = load(input)
            
            @info "Запуск детекции..."
            result = Inference.predict(engine, img)
            
            @info "Результаты детекции:"
            @info "   Найдено объектов: $(size(result.boxes, 1))"
            for i in 1:min(5, size(result.boxes, 1))
                box = result.boxes[i, :]
                @info "   $i: [$(round(box[1], digits=1)), $(round(box[2], digits=1)), $(round(box[3], digits=1)), $(round(box[4], digits=1))] | score=$(round(result.scores[i], digits=3)) | class=$(result.labels[i])"
            end
            
            if output !== nothing
                @info "Сохранение результата в: $output"
                drawn = Inference.draw_results(img, result)
                save(output, drawn)
                @info "✅ Результат сохранён"
            end
            
            return 0
            
        catch e
            @error "Ошибка при инференсе: $e"
            @error stacktrace(catch_backtrace())
            return 1
        end
        
    elseif mode=="satellite"
        if input === nothing
            @error "Для режима satellite нужно указать IP спутниковой камеры"
            return 1
        end
        
        try
            @info "Подключение к спутниковой камере: $input"
            detector = SatelliteDetector(input)
            connect(detector)
            result = detect_on_satellite!(detector)
            @info "✅ Детекция со спутника завершена. Найдено людей: $(result["num_people"])"
            if output !== nothing
                @info "Сохранение результата в: $output"
                img_with_detections = create_annotated_image(result["image"], result["boxes"])
                save(output, img_with_detections)
                json_output = replace(output, r"\.[^.]+$" => ".json")
                open(json_output, "w") do f
                    JSON.pretty(f, Dict(
                        "num_people" => result["num_people"],
                        "boxes" => result["boxes"],
                        "scores" => result["scores"],
                        "labels" => result["labels"],
                        "timestamp" => string(now())
                    ))
                end
                @info "✅ Снимок сохранён в $output, метаданные в $json_output"
            end
            
            disconnect(detector)
            return 0
        catch e
            @error "Ошибка при спутниковой детекции: $e"
            return 1
        end
    elseif mode=="batch"
        if input === nothing || !isdir(input)
            @error "Для режима batch нужно указать директорию с изображениями"
            return 1
        end
        
        try
            @info "Пакетная обработка изображений из: $input"
            images = filter(f -> endswith(f, r"\.(jpg|png|jpeg)$"), readdir(input))
            @info "Найдено $(length(images)) изображений"
            config = InferenceConfig()
            engine = InferenceEngine(; config=config)
            results = []
            for (i, img_file) in enumerate(images)
                @info "[$i/$(length(images))] Обработка: $img_file"
                img_path = joinpath(input, img_file)
                img = load(img_path)
                result = Inference.predict(engine, img)
                push!(results, (img_file, result))
            end
            
            report_path = output !== nothing ? output : joinpath(input, "inference_report.json")
            save_inference_report(report_path, results)
            @info "✅ Отчёт сохранён в: $report_path"
            
            return 0
        catch e
            @error "Ошибка при пакетной обработке: $e"
            return 1
        end
        
    elseif mode=="profile"
        @info "Запуск профилирования производительности..."
        run_benchmark()
        return 0
    else 
        @error "Неизвестный режим: $mode"
        return 1 
    end
    return 0
end

function save_inference_report(path::String, results::Vector{Tuple{String, DetectionResult}})
    report = Dict(
        "timestamp" => string(now()),
        "total_images" => length(results),
        "results" => [
            Dict(
                "image" => img_file,
                "num_detections" => size(r.boxes, 1),
                "boxes" => r.boxes,
                "scores" => r.scores,
                "labels" => r.labels,
                "inference_time" => r.inference_time
            ) for (img_file, r) in results
        ]
    )
    
    open(path, "w") do f
        JSON.pretty(f, report)
    end
end

function run_benchmark()
    @info "Запуск бенчмарка..."
    @info "Тест 1: Загрузка модели"
    t1 = @elapsed begin
        config = InferenceConfig()
        engine = InferenceEngine(; config=config)
    end
    @info "   Время загрузки: $(round(t1, digits=3)) сек"
    @info "Тест 2: Инференс"
    img = rand(RGB{Float32}, 640, 640)
    t2 = @elapsed begin
        result = Inference.predict(engine, img)
    end
    @info "   Время инференса: $(round(t2*1000, digits=1)) мс"
    @info "   Найдено объектов: $(size(result.boxes, 1))"
    @info "Тест 3: Пакетная обработка (5 изображений)"
    images = [rand(RGB{Float32}, 640, 640) for _ in 1:5]
    
    t3 = @elapsed begin
        for img in images
            Inference.predict(engine, img)
        end
    end
    @info "   Общее время: $(round(t3*1000, digits=1)) мс"
    @info "   Среднее на изображение: $(round(t3*1000/5, digits=1)) мс"
    @info "✅ Бенчмарк завершён"
end

end  # module BPLADetection