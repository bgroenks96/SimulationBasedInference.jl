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

linearode_example_doc = Literate.markdown(
       joinpath(examples_dir, "linearode", "linearode.jl"),
       examples_output_dir,
       execute=true,
       documenter=true
)

makedocs(
       modules=modules,
       sitename="SimulationBasedInference.jl",
       authors="Brian Groenke, Kristoffer Aalstad",
       format=Documenter.HTML(
              prettyurls=!IS_LOCAL,
              size_threshold=2*1024*1024, # 2 MiB
              size_threshold_warn=512*1024, # 512 KiB
              size_threshold_ignore=["examples/linearode.md"],
              canonical = "https://bgroenks96.github.io/SimulationBasedInference.jl/v0",
       ),
       warnonly=true, # don't fail when there are errors
       pages=[
              "Home" => "index.md",
              "Getting started" => [
                     "Ensemble inversion of a linear ODE" => "examples/linearode.md",
              ],
              "API Reference" => "api.md",
       ],
)

# remove gitignore from build files
rm(joinpath(@__DIR__, "build", ".gitignore"), force=true)

deploydocs(
       repo="github.com/bgroenks96/SimulationBasedInference.jl.git",
       push_preview = true,
       versions = ["v0" => "v^", "v#.#", "dev" => "dev"],
       deploy_config = deployconfig,
)