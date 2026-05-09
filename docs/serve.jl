using LiveServer

# Set the directory to "build"; assumes server.jl and the build directory are in the same folder
serve(; dir=joinpath(@__DIR__, "build"))
