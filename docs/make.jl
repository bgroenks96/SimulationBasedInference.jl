using ArgParse
using SimulationBasedInference
using PythonCall
using Documenter
using Literate

s = ArgParseSettings()
@add_arg_table! s begin
    "--local", "-l"
       action = :store_true
       help = "Local docs build mode"
    "--draft", "-d"
       action = :store_true
       help = "Whether to build docs in draft mode, i.e. skipping execution of examples and doctests"
end
parsed_args = parse_args(ARGS, s)

IS_LOCAL = parsed_args["local"] || parse(Bool, get(ENV, "LOCALDOCS", "false"))
IS_DRAFT = parsed_args["draft"] || parse(Bool, get(ENV, "DRAFTDOCS", "false"))
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
       execute=!IS_DRAFT,
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
       doctest=!IS_DRAFT,
       draft=IS_DRAFT,
       warnonly=true, # don't fail when there are errors
       pages=[
              "Home" => "index.md",
              "Getting started" => [
                     "Ensemble inversion of a linear ODE" => "examples/linearode.md",
              ],
              "Problem interface" => [
                     "Observables" => "problems/observables.md",
                     "Forward problems" => "problems/forward_problem.md",
                     "Inference problems" => "problems/inference_problem.md",
              ],
              "Inference algorithms" => [
                     "Ensemble methods" => "inference/ensemble.md",
                     "MCMC" => "inference/mcmc.md",
                     "PySBI" => "inference/pysbi.md",
              ],
              "Utilities" => "utils.md",
              "API Reference" => [
                     "SimulationBasedInference" => "api/sbi.md",
                     "Emulators" => "api/emulators.md",
              ]
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