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

module Main

include("Config.jl")
include("models/Models.jl")
include("Training.jl")
include("app.jl") 
include("serve.jl")

using Base: invokelatest
using ArgParse
using Logging
using Dates
using JSON
using BSON

export main

function parse_args()
    settings = ArgParseSettings()
    settings.description = "BPLA Person Detection - Детекция людей на снимках с БПЛА и Спутника"
    settings.add_help = true
    settings.prog = "bpla-detection"
    settings.version = "1.0.0"

    @add_arg_table! settings begin
        "--mode", "-m"
            help = "Режим работы: web, train, serve, mamaa, inference, benchmark, test, version"
            arg_type = String
            default = "web"
        "--host"
            help = "Хост для запуска сервера"
            arg_type = String
            default = "0.0.0.0"
        "--port", "-p"
            help = "Порт для запуска сервера"
            arg_type = Int
            default = 7860
        "--debug"
            help = "Включить режим отладки"
        "--output", "-o"
            help = "Выходной файл для результатов"
            arg_type = String
        "--device"
            help = "Устройство для инференса: gpu"
            arg_type = String
            default = "gpu"
        "--redis-host"
            help = "Хост Redis для кэширования"
            arg_type = String
            default = "localhost"
        "--rabbitmq-host"
            help = "Хост RabbitMQ для очередей"
            arg_type = String
            default = "localhost"
        "--metrics-port"
            help = "Порт для Prometheus метрик"
            arg_type = Int
            default = 8000
    end
    return ArgParse.parse_args(settings)
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

"""
Вывод информации о системе
"""
function print_system_info()
    println()
    println("📊 Системная информация:")
    println("  • Julia: $(VERSION)")
    println("  • Date: $(Dates.now())")
    println("  • CPU: $(Sys.cpu_info()[1].model)")
    println("  • Memory: $(round(Sys.total_memory() / 1024^3, digits=2)) GB")
    
    println("  • Backend: $(detect()) (Julia control-plane only)")
end

"""
Режим инференса
"""
function run_inference(args)
    @info "🖼️ Запуск инференса"
    
    input_path = args["input"]
    output_path = get(args, "output", nothing)
    model_path = args["model-path"]
    device = detect(args)
    
    @info "  • Вход: $input_path"
    @info "  • Модель: $model_path"
    @info "  • Устройство: $device"
    @info "  • Порог: $(args["conf-threshold"])"
    
    # Загрузка модели
    @info "Загрузка модели..."
    model = invokelatest(Models.load_model, model_path)
    model = invokelatest(Models.model_to_device, model, device)
    
    # Загрузка изображения
    @info "Загрузка изображения..."
    img = invokelatest(FileIO.load, input_path)
    
    # Инференс
    @info "Обработка..."
    start_time = time_ns()
    result = invokelatest(
        Inference.predict,
        model,
        img;
        conf_threshold=args["conf-threshold"],
        iou_threshold=args["iou-threshold"]
    )
    elapsed = (time_ns() - start_time) / 1_000_000
    
    @info "✅ Инференс завершён за $(round(elapsed, digits=2)) ms"
    @info "  • Найдено объектов: $(length(result.boxes))"
    @info "  • Люди: $(count(result.labels .== 0))"
    if output_path !== nothing
        if endswith(output_path, ".json")
            open(output_path, "w") do f
                invokelatest(JSON.write, f, result)
            end
            @info "  • Результат сохранён в JSON: $output_path"
        elseif endswith(output_path, ".jpg") || endswith(output_path, ".png")
            invokelatest(save, output_path, result.annotated_image)
            @info "  • Аннотированное изображение сохранено: $output_path"
        else
            invokelatest(BSON.bson, output_path, Dict("result" => result))
            @info "  • Результат сохранён в BSON: $output_path"
        end
    end
    
    return 0
end

"""
Режим обучения
"""
function run_training(args)
    @info "📚 Запуск обучения"
    
    device = detect(args)
    
    @info "  • Устройство: $device"
    @info "  • Эпохи: $(args["epochs"])"
    @info "  • Batch size: $(args["batch-size"])"
    @info "  • Learning rate: $(args["lr"])"
    @info "  • Data dir: $(args["data-dir"])"
    @info "  • Model dir: $(args["model-dir"])"
    
    success = invokelatest(
        Training.train_augmented,
        epochs=args["epochs"],
        batch_size=args["batch-size"],
        lr=args["lr"],
        resize_dim=args["resize-dim"],
        data_dir=args["data-dir"],
        model_dir=args["model-dir"],
        device=device,
        conf_threshold=args["conf-threshold"],
        iou_threshold=args["iou-threshold"]
    )
end

function show_version()
    println("BPLA Person Detection 1.0.0")
    print_system_info()
    return 0
end

function main()::Int
    try
        args = parse_args()
        if get(args, "debug", false)
            global_logger(ConsoleLogger(stderr, Logging.Debug))
            @debug "Режим отладки включен"
        end
        if get(args, "version", false)
            return show_version()
        end
        validate_args(args)
        print_system_info()
        mode = args["mode"]
        @info "🚀 Запуск в режиме: $mode"
        if mode == "web"
            @info "🌐 Запуск веб-интерфейса..."
            invokelatest(App.run_server, host=args["host"], port=args["port"])
        elseif mode == "train"
            return run_training(args)
        elseif mode == "serve"
            @info "🔌 Запуск сервера..."
            invokelatest(Serve.run_server, host=args["host"], port=args["port"])
        elseif mode == "version"
            return show_version()
        else
            println()
            println("Доступные режимы:")
            println("  web        - Запуск веб-интерфейса")
            println("  train      - Обучение модели")
            println("  serve      - Сервер")
            println("  version    - Информация о версии")
            return 1
        end
        return 0
    catch
        return 1
    finally
        cleanup()
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main())
end

end  # module Main