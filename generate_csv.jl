using Pkg
Pkg.instantiate()

module BenchmarkCSV

using DataFrames

export parse_log

function parse_log(log_text::String)
    data = DataFrame(N=Int[], Type=String[], Solver=String[], Time_ms=Float64[])
    current_n = 0
    current_type = ""
    current_solver = ""
    mkl_active = false
    nvpl_active = false

    for line in split(log_text, '\n')
        line = strip(line)

        if occursin("Testing N =", line)
            current_n = parse(Int, match(r"(\d+)", line).match)
        elseif occursin("Testing ", line) && occursin("=====", line)
            current_type = match(r"Testing (\w+)", line).captures[1]
        elseif occursin("Switching to MKL", line)
            mkl_active = true; nvpl_active = false
        elseif occursin("Switching to NVPL", line)
            nvpl_active = true; mkl_active = false
        elseif occursin("LinearAlgebra", line)
            current_solver = mkl_active ? "MKL" : "OpenBLAS"
        elseif occursin("CUDA", line)
            current_solver = "CUDA"
        elseif occursin("MAGMA", line)
            current_solver = mkl_active ? "MAGMA+MKL" :
                             nvpl_active ? "MAGMA+NVPL" : "MAGMA+OpenBLAS"
        else
            m = match(r"^\s*([\d.]+)\s*(ms|s)", line)
            if m !== nothing
                time = parse(Float64, m.captures[1])
                unit = m.captures[2]
                time_ms = unit == "s" ? time * 1000 : time
                push!(data, (current_n, current_type, current_solver, time_ms))
            end
        end
    end

    grouped = groupby(data, [:N, :Type, :Solver])
    return combine(grouped, :Time_ms => (x -> sum(x) / length(x)) => :Time_ms)
end

end # module

# Allow script usage
if abspath(PROGRAM_FILE) == @__FILE__
    
    using CSV, .BenchmarkCSV

    if length(ARGS) < 1
        println("Usage: julia generate_csv.jl <file.log>")
        exit(1)
    end

    input_path = ARGS[1]
    output_path = replace(input_path, r"\.log$" => ".csv", count=1)

    log_text = read(input_path, String)
    df = BenchmarkCSV.parse_log(log_text)
    CSV.write(output_path, df)

    println("✅ CSV written to: $output_path")
end