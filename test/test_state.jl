import Pigeons
using DynamicPPL

@model function test_model()
    α ~ Beta(1, 2)
    β ~ Beta(2, 3)
    n ~ Poisson(3.0) + 1
    y ~ Binomial(n, α * β)
end

rng = SplittableRandom(2026)
model = test_model()
vi = DynamicPPL.VarInfo(model)

vector_state = DynamicPPL.internal_values_as_vector(vi)
println("test unflatten:", DynamicPPL.unflatten!!(vi, vector_state), "\n")

@testset "variables" begin
    println("typeof(vi) = ", typeof(vi))
    println("vi = ", vi)
    println("keys(vi) = ", keys(vi))
    for vn in keys(vi)
        println("vn = ", vn)
        println("  sym = ", Symbol(vn))
        tv = getindex(vi.values, vn)
        val = DynamicPPL.get_internal_value(tv)
        println("  val = ", val)
        println("  typeof(val) = ", typeof(val))
        println("  eltype(val) = ", eltype(val))
    end

    # test Pigeons.variable()
    vars_of_Float64 = Pigeons.continuous_variables(vi)
    println("variables(vi, Float64) = ", vars_of_Float64)
    @test length(vars_of_Float64) == 2 # α, β
    vars_of_int = Pigeons.discrete_variables(vi)
    println("variables(vi, Int64) = ", vars_of_int)
    @test length(vars_of_int) == 2 # n, y
    α_val = Pigeons.variable(vi, Symbol(keys(vi)[1]))
    β_val = Pigeons.variable(vi, Symbol(keys(vi)[2]))
    n_val = Pigeons.variable(vi, Symbol(keys(vi)[3]))
    y_val = Pigeons.variable(vi, Symbol(keys(vi)[4]))
    println("values: ", α_val, β_val, n_val, y_val)


    syms = DynamicPPL.getsym.(keys(vi))
    var_values = Pigeons.variable(vi, :singleton_variable)
    println("Pigeons.variable() with no symbol specified: ", var_values)
    value_at_1 = Pigeons.variable(vi, syms[1])
    println("Pigeons.variable() with symbol specified : ", value_at_1)
    @test only(value_at_1) == var_values[1]
end

@testset "update_state!" begin
    syms = Symbol.(keys(vi))
    updated_vi = Pigeons.update_state!(vi, syms[1], 1, 0.81)
    @test DynamicPPL.getindex_internal(updated_vi, :)[1] == 0.81
end


@testset "samples" begin
    # test extract_sample
    lp = TuringLogPotential(model)
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


@testset "invariance test" begin
    @model function gaussian_model()
        x ~ Normal(0, 1)
    end
    model = gaussian_model()
    res = Pigeons.invariance_test(TuringLogPotential(model), SliceSampler(), rng)
    println("res = ", res)
    @show res.pvalues
    @test res.passed

end


@testset "explorer" begin
    rng = SplittableRandom(2026)
    @model function cont_model(y)
        x ~ Normal(0, 1)
        y .~ Normal(x, 1)
    end
    dist = Normal(0.7, 1.0)
    y = rand(rng, dist, 1000)
    model = cont_model(y)

    vi = DynamicPPL.VarInfo(model)
    vi = last(DynamicPPL.init!!(
        rng,
        model,
        vi,
        DynamicPPL.InitFromPrior(),
        DynamicPPL.UnlinkAll(),
    ))
    log_potential = TuringLogPotential(model)
    h = SliceSampler()
    cached_lp = -Inf
    n = 100
    states = Vector{Float64}(undef, n)
    for i in 1:n
        replica = Pigeons.Replica(vi, 1, rng, (;), 1)
        cached_lp = Pigeons.slice_sample!(h, vi, log_potential, cached_lp, replica)
        # println("cached_lp = ", cached_lp)
        state = DynamicPPL.getindex_internal(vi, :)[1]
        # println("type of state:", typeof(state))
        # println("state:", state)
        states[i] = state
    end

    @test abs(mean(states) - 0.7) < 0.1
end

