using SimulationBasedInference
using PythonCall
using Documenter
using Literate

# ENV["LOCALDOCS"] = "true"

IS_LOCAL = haskey(ENV,"LOCALDOCS") && ENV["LOCALDOCS"] == "true"
if haskey(ENV, "GITHUB_ACTIONS")
       ENV["JULIA_DEBUG"] = "Documenter"
end

deployconfig = Documenter.auto_detect_deploy_system()

const modules = [
    SimulationBasedInference,
    SimulationBasedInference.PySBI,
];

examples_dir = joinpath(@__DIR__, "..", "examples")
examples_output_dir = joinpath(@__DIR__, "src", "examples")
# remove existing files
rm(examples_output_dir, recursive=true, force=true)
# recreate directory
mkpath(examples_output_dir)

makedocs(
       modules=modules,
       sitename="SimulationBasedInference.jl",
       authors="Brian Groenke, Kristoffer Aalstad",
       format=Documenter.HTML(
              prettyurls=!IS_LOCAL,
              canonical = "https://bgroenks96.github.io/SimulationBasedInference.jl/v0",
       ),
       warnonly=true, # don't fail when there are errors
       pages=[
        "Home" => "index.md",
        "API Reference" => "api.md",
       ],
)

# remove gitignore from build files
# rm(joinpath(@__DIR__, "build", ".gitignore"))

deploydocs(
       repo="github.com/bgroenks96/SimulationBasedInference.jl.git",
       push_preview = true,
       versions = ["v0" => "v^", "v#.#", "dev" => "dev"],
       deploy_config = deployconfig,
)