"""
Notes:
vi.values -> VarNamedTuple: (key, value) = (VarName, AbstractTransformedValue)
keys(vi) -> an iterator (collectable) of DynamicPPL.VarName objects, e.g. keys(vi) = VarName[α, β, y]
collect(keys(vi)) -> Vector{DynamicPPL.VarName}
VarName has: symbol + something else (could be AbstractPPL.Iden)
so DynamicPPL.getsym(vn) -> symbol of this VarName

DynamicPPL.getindex_internal(vi, vn) -> vector of values of each variable
DynamicPPL.setindex_internal!!(vi, val, vn) -> new vi with val replacing the old one


"""

# grouping of variables
function variables(vi::DynamicPPL.VarInfo, ::Type{T}) where {T}
    vns = keys(vi)
    syms = DynamicPPL.getsym.(vns)
    # return [sym for (sym, vn) in zip(syms, vns) if eltype(DynamicPPL.get_internal_value(getindex(vi.values, vn))) <: T]
    return [sym for (sym, vn) in zip(syms, vns) if eltype(DynamicPPL.getindex_internal(vi, vn)) <: T]
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
Pigeons.variable(state::DynamicPPL.VarInfo, name::Symbol) =
    if name === :singleton_variable
        DynamicPPL.getindex_internal(state, :)
    else
        vns = [vn for vn in keys(state) if DynamicPPL.getsym(vn) === name]
        mapreduce(vn -> DynamicPPL.getindex_internal(state, vn), vcat, vns)
    end


function Pigeons.update_state!(vi::DynamicPPL.VarInfo, name::Symbol, index::Int, value)
    vns = [vn for vn in keys(vi) if DynamicPPL.getsym(vn) === name]
    vn = vns[1]
    vals = DynamicPPL.getindex_internal(vi, vn)
    vals[index] = value
    return DynamicPPL.setindex_internal!!(vi, vals, vn)
end



# From Turing.jl/src/utilities/helper.jl
ind2sub(v, i) = Tuple(CartesianIndices(v)[i])


function Pigeons.extract_sample(state::DynamicPPL.VarInfo, log_potential)
    invlink_vi = DynamicPPL.invlink(state, Pigeons.turing_model(log_potential))
    result = invlink_vi[:] # result = DynamicPPL.getindex_internal(invlink_vi, :) is also acceptable
    push!(result, log_potential(state))
    return result
end

function Pigeons.sample_names(state::DynamicPPL.VarInfo, _)
    result = Symbol[]
    syms = DynamicPPL.getsym.(keys(state))
    for sym in syms
        var = Pigeons.variable(state, sym) 
        # TODO: test up tp this point, try different data types: scalar, vector, matrix, etc to see var.
        if var isa Number || (var isa AbstractArray && length(var) == 1) # TODO: is this long predicate necessary??
            push!(result, sym)
        elseif var isa AbstractArray
            for i in eachindex(var) # bug here: missing the last index
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
Pigeons._recursive_equal(a::DynamicPPL.VarInfo, b::DynamicPPL.VarInfo) =
    length(a) == length(b) &&
    sample_names(a, 1) == sample_names(b, 1) && # second argument of sample_names() is dummy
    DynamicPPL.getindex_internal(a, :) == DynamicPPL.getindex_internal(b, :)


Pigeons.recursive_equal(
    a::Union{TuringLogPotential,DynamicPPL.Model},
    b) = Pigeons._recursive_equal(a, b)
