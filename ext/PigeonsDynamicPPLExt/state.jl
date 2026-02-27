"""
Notes:
vi.values -> VarNamedTuple: (key, value) = (VarName, AbstractTransformedValue)
keys(vi) -> an iterator (collectable) of DynamicPPL.VarName objects
collect(keys(vi)) -> Vector{DynamicPPL.VarName}
VarName has: symbol + something else
so DynamicPPL.getsym(vn) -> symbol


"""

_varnames_for_symbol(vi::DynamicPPL.VarInfo, name::Symbol) =
    [vn for vn in keys(vi) if DynamicPPL.getsym(vn) === name]

# grouping of variables
# variables(vi::DynamicPPL.TypedVarInfo{<:NamedTuple{names}}, ::Type{T}) where {names,T} =
#     [name for (name, meta) in zip(names, vi.metadata) if eltype(meta.vals) <: T]


function variables(vi::DynamicPPL.VarInfo, ::Type{T}) where {T}
    syms = unique(DynamicPPL.getsym.(collect(keys(vi))))
    return [sym for sym in syms if all(vn ->
            DynamicPPL.getsym(vn) !== sym || begin
                tv = getindex(vi.values, vn)
                eltype(DynamicPPL.get_internal_value(tv)) <: T
            end,
        keys(vi)
    )]
end


Pigeons.continuous_variables(state::DynamicPPL.VarInfo) = variables(state::DynamicPPL.VarInfo, AbstractFloat)
Pigeons.discrete_variables(state::DynamicPPL.VarInfo) = variables(state::DynamicPPL.VarInfo, Integer)
function Pigeons.recorded_continuous_variables(vi::DynamicPPL.VarInfo)
    cvars = Pigeons.continuous_variables(vi)

    # allows us to handle samplers with adaptive preconditioners, which *may* be
    # used in fully cont models (no way of knowing here if they are actually used though)
    is_fully_continuous(vi) && push!(cvars, :singleton_variable)

    return cvars
end

# note: this returns unconstrained parameters when the varinfo is linked 
# (default in pigeons as of Jul-24), and constrained otherwise
# Pigeons.variable(state::DynamicPPL.TypedVarInfo, name::Symbol) = 
#     if name === :singleton_variable
#         state[:]
#     else
#         state.metadata[name].vals
#     end

Pigeons.variable(state::DynamicPPL.VarInfo, name::Symbol) =
    if name === :singleton_variable
        DynamicPPL.getindex_internal(state, :)
    else
        vns = _varnames_for_symbol(state, name)
        mapreduce(vn -> DynamicPPL.getindex_internal(state, vn), vcat, vns)
    end


# Pigeons.update_state!(vi::DynamicPPL.VarInfo, name::Symbol, index::Int, value) =
#     vi.metadata[name].vals[index] = value

function Pigeons.update_state!(vi::DynamicPPL.TypedVarInfo, ::Symbol, index::Int, value)
    vals = collect(DynamicPPL.getindex_internal(vi, :))  # flattened internal vector
    vals[index] = value
    return DynamicPPL.unflatten!!(vi, vals)
end



# From Turing.jl/src/utilities/helper.jl
ind2sub(v, i) = Tuple(CartesianIndices(v)[i])


# function Pigeons.extract_sample(state::DynamicPPL.TypedVarInfo, log_potential)
#     invlink_vi = DynamicPPL.invlink(state, Pigeons.turing_model(log_potential))
#     result = invlink_vi[:]
#     push!(result, log_potential(state))
#     return result
# end

function Pigeons.extract_sample(state::DynamicPPL.VarInfo, log_potential)
    model = Pigeons.turing_model(log_potential)
    invlink_vi = DynamicPPL.invlink!!(DynamicPPL.DynamicTransformation(), state, model)
    result = copy(DynamicPPL.getindex_internal(invlink_vi, :))
    push!(result, log_potential(state))
    return result
end

# function Pigeons.sample_names(state::DynamicPPL.TypedVarInfo, _)
#     result = Symbol[]
#     all_names = fieldnames(typeof(state.metadata))
#     for var_name in all_names
#         var = state.metadata[var_name].vals
#         if var isa Number || (var isa AbstractArray && length(var) == 1)
#             push!(result, var_name)
#         elseif var isa AbstractArray
#             # flatten vector names following Turing convention
#             for i in eachindex(var)
#                 var_and_index_name =
#                     Symbol(var_name, "[", join(ind2sub(size(var), i), ","), "]")
#                 push!(result, var_and_index_name)
#             end
#         else
#             error("don't know how to handle var `$var_name` of type $(typeof(var))")
#         end
#     end
#     push!(result, :log_density)
#     return result
# end

# returns e.g. [:x, :σ, :z] in the order they first appear in keys(vi)
function symbols_in_order(vi::DynamicPPL.VarInfo)
    vns = collect(keys(vi))                 # Vector{VarName}
    syms = map(DynamicPPL.getsym, vns)      # Vector{Symbol}
    return unique(syms)                     # keeps first-occurrence order
end

function Pigeons.sample_names(state::DynamicPPL.VarInfo, _)
    result = Symbol[]
    syms = symbols_in_order(state)
    for sym in syms
        var = Pigeons.variable(state, sym)  # 1D vector
        if var isa Number || (var isa AbstractArray && length(var) == 1)
            push!(result, sym)
        elseif var isa AbstractArray
            for i in eachindex(var)
                var_and_index_name = Symbol(sym, "[", join(ind2sub(size(var), i), ","), "]")
                push!(result, var_and_index_name)
            end
        else
            error("don't know how to handle var `$sym` of type $(typeof(var))")
        end
    end
    push!(result, :log_density)
    return result
end

#=
explorer implementations
=#
# function Pigeons.slice_sample!(h::SliceSampler, vi::DynamicPPL.TypedVarInfo, log_potential, cached_lp, replica)
#     for meta in vi.metadata
#         cached_lp = Pigeons.slice_sample!(h, meta.vals, log_potential, cached_lp, replica)
#     end
#     return cached_lp
# end

function Pigeons.slice_sample!(h::SliceSampler, vi::DynamicPPL.VarInfo, log_potential, cached_lp, replica)
    vals = copy(DynamicPPL.getindex_internal(vi, :)) # flattened 
    cached_lp = Pigeons.slice_sample!(h, vals, log_potential, cached_lp, replica)
    replica.state = DynamicPPL.unflatten!!(vi, vals)
    return cached_lp
end

function Pigeons.step!(explorer::Pigeons.GradientBasedSampler, replica, shared, vi::DynamicPPL.VarInfo)
    vector_state = Pigeons.get_buffer(replica.recorders.buffers, :flattened_vi, get_dimension(vi))
    flatten!(vi, vector_state)
    Pigeons.step!(explorer, replica, shared, vector_state)
    replica.state = DynamicPPL.unflatten!!(vi, vector_state)
end

#=
specialized equality checks
=#
# Pigeons.recursive_equal(a::DynamicPPL.TypedVarInfo, b::DynamicPPL.TypedVarInfo) =
#     # as of Nov 2023, DynamicPPL does not supply == for TypedVarInfo
#     length(a.metadata) == length(b.metadata) &&
#         sample_names(a,1) == sample_names(b,1) && # second argument is not used
#         a[:] == b[:]

Pigeons.recursive_equal(a::DynamicPPL.VarInfo, b::DynamicPPL.VarInfo) =
    length(a) == length(b) &&
    sample_names(a, 1) == sample_names(b, 1) && # second argument is not used
    DynamicPPL.getindex_internal(a, :) == DynamicPPL.getindex_internal(b, :)


# Pigeons.recursive_equal(
#     a::Union{TuringLogPotential,DynamicPPL.Model,DynamicPPL.ConditionContext}, 
#     b) = Pigeons._recursive_equal(a, b)

Pigeons.recursive_equal(
    a::Union{TuringLogPotential,DynamicPPL.Model},
    b) = Pigeons._recursive_equal(a, b)
