using Pkg

using CSV, DataFrames, CairoMakie

# Load the parser from generate_csv.jl
include("generate_csv.jl")  # defines module BenchmarkCSV
using .BenchmarkCSV: parse_log

function ensure_csv_from_log(input_path::String)::String
    if endswith(input_path, ".csv")
        return input_path
    elseif endswith(input_path, ".log")
        csv_path = replace(input_path, r"\.log$" => ".csv", count=1)
        println("↪ Detected log file. Generating CSV: $csv_path")

        log_text = read(input_path, String)
        df = parse_log(log_text)
        CSV.write(csv_path, df)
        return csv_path
    else
        error("Unsupported file type: must be .log or .csv")
    end
end

function plot_benchmark(csv_path::String, prefix::String)
    df = CSV.read(csv_path, DataFrame)

    types_to_plot = ["Float32", "Float64", "ComplexF32", "ComplexF64"]
    colors = Dict(
        "OpenBLAS" => :blue,
        "CUDA" => :green,
        "MAGMA+OpenBLAS" => :orange,
        "MKL" => :purple,
        "MAGMA+MKL" => :red,
        "MAGMA+NVPL" => :teal,
    )

    for tp in types_to_plot
        subdf = filter(row -> row.Type == tp, df)
        if isempty(subdf)
            @warn "No data for $tp — skipping."
            continue
        end

        fig = Figure(size=(800, 600))
        ax = Axis(fig[1, 1], title=tp, xlabel="Matrix size (N)", ylabel="Time (ms)", yscale=Makie.log10)

        for solver in sort(unique(subdf.Solver))
            sdata = filter(row -> row.Solver == solver, subdf)
            lines!(ax, sdata.N, sdata.Time_ms;
                   label=solver,
                   color=get(colors, solver, :black),
                   linewidth=2)
            scatter!(ax, sdata.N, sdata.Time_ms;
                     color=get(colors, solver, :black),
                     markersize=5)
        end

        axislegend(ax, position=:rb, framevisible=false)
        save("$(prefix)_$(tp).pdf", fig)
        println("✅ Saved: $(prefix)_$(tp).pdf")
    end
end

function main()
    if length(ARGS) < 1
        println("Usage: julia plot_benchmark.jl <benchmark.log or benchmark.csv>")
        return
    end

    Pkg.instantiate()

    input_file = ARGS[1]
    base = splitext(basename(input_file))[1]  # Remove .log or .csv
    csv_file = ensure_csv_from_log(input_file)

    plot_benchmark(csv_file, base)
end

main()