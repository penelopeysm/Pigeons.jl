# use single statement to avoid multiple precompile stages
using Pigeons,
    ADTypes,
    Distributions,
    DynamicPPL,
    Enzyme,
    FillArrays,
    ForwardDiff,
    LinearAlgebra,
    LogDensityProblems,
    LogDensityProblemsAD,
    Random,
    ReverseDiff,
    SplittableRandoms,
    SpecialFunctions,
    Test

is_windows_in_CI() = Sys.iswindows() && (get(ENV, "CI", "false") == "true")
