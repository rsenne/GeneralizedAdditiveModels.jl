# docs/make.jl
using Documenter

# Make sure the package loads when building from docs/
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using GeneralizedAdditiveModels  

# Doctesting
DocMeta.setdocmeta!(GAM, :DocTestSetup, :(using GAM); recursive=true)

makedocs(
    sitename = "GeneralizedAdditiveModels.jl",
    modules  = [GeneralizedAdditiveModels],
    pages    = [
        "Home" => "index.md",
        "API Reference" => "api_reference.md",
    ],
    format   = Documenter.HTML(; prettyurls = get(ENV, "CI", "false") == "true"),
    authors  = "Trent Henderson",
    remotes  = nothing,
    checkdocs = :none,  # relax until docs are fleshed out
)

# for later
# deploydocs(repo = "github.com/hendersontrent/GeneralizedAdditiveModels.jl.git", devbranch = "main")
