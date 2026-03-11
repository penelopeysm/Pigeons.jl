using Test
using Pigeons
using DynamicPPL
using Distributions
using Random
using SplittableRandoms
include("../../../ext/PigeonsDynamicPPLExt/state.jl")
include("../../../ext/PigeonsDynamicPPLExt/utils.jl")


@model function test_model()
    α ~ Beta(1, 2)
    β ~ Beta(2, 3)
    y ~ Binomial(10, α*β)
    return nothing
end

rng = MersenneTwister(2026)
model = test_model()
vi = DynamicPPL.VarInfo(model)

@testset "variables" begin
    println("typeof(vi) = ", typeof(vi))
    println("vi = ", vi)
    println("keys(vi) = ", keys(vi))
    for vn in keys(vi)
        println("vn = ", vn)
        println("  sym = ", DynamicPPL.getsym(vn))
        tv = getindex(vi.values, vn)
        val = DynamicPPL.get_internal_value(tv)
        println("  val = ", val)
        println("  typeof(val) = ", typeof(val))
        println("  eltype(val) = ", eltype(val))
    end

    # test Pigeons.variable()
    vars_of_Float64 = variables(vi, Float64)
    println("variables(vi, Float64) = ", vars_of_Float64)
    @test length(vars_of_Float64) == 2
    vars_of_int = variables(vi, Int64)
    println("variables(vi, Int64) = ", vars_of_int)
    @test length(vars_of_int) == 1
    vars_of_vector_of_Float64 = variables(vi, Vector{Float64})
    println("variables(vi, Vector{Float64}) = ", vars_of_vector_of_Float64)
    @test length(vars_of_vector_of_Float64) == 0

    syms = DynamicPPL.getsym.(keys(vi))
    var_values = Pigeons.variable(vi, :singleton_variable)
    println("Pigeons.variable() with no symbol specified: ", var_values)
    value_at_1 = Pigeons.variable(vi, syms[1])
    println("Pigeons.variable() with symbol specified : ", value_at_1)
    @test only(value_at_1) == var_values[1]
end

@testset "update_state!" begin
    syms = DynamicPPL.getsym.(keys(vi))
    updated_vi = Pigeons.update_state!(vi, syms[1], 1, 0.81)
    @test DynamicPPL.getindex_internal(updated_vi, :)[1] == 0.81
end


@testset "samples" begin
    # test extract_sample
    lp = Pigeons.TuringLogPotential(model)
    extracted_sample_values = Pigeons.extract_sample(vi, lp)
    println("samples = ", extracted_sample_values)
    #test sample_names
    extracted_sample_names = Pigeons.sample_names(vi, nothing)
    println("sample_names = ", extracted_sample_names)
    @test length(extracted_sample_values) == length(extracted_sample_names)

end


@testset "equality" begin
    @model function model_2()
        α ~ Beta(1, 2)
        β ~ Beta(2, 3)
        y ~ Binomial(10, α * β)
        return nothing
    end
    vi_2 = DynamicPPL.VarInfo(model_2())

    @test Pigeons._recursive_equal(vi, vi_2) == false
    @test Pigeons.recursive_equal(vi, vi_2) == false
    
end


@testset "explorer" begin
    #TODO
    # test slice_sample!()

    # test step!()
    
end

