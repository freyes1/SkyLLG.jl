"Abstract supertype for all spin states"
abstract type SpinState end

"""
SpinState1D <: SpinState

Type of a 1D spin chain
"""
struct SpinState1D <: SpinState
    "Vector of spins"
    spins::Vector{SVector{3,<: Number}}
    "Length of chain"
    N::Integer
    
    "Construct chain from vector"
    SpinState1D(s) = new(s, length(s))
end

# Generalizing indices to work with SpinStates

Base.size(S::SpinState1D) = (length(S.spins))

function Base.getindex(S::T, i::Int) where T <: SpinState
    return S.spins[i]
end

function Base.setindex!(S::T, v, i::Int) where T <: SpinState
    S.spins[i] = v
    return v
end

"""
SpinState2D <: SpinState

Type of a 2D spin layer
"""
struct SpinState2D <: SpinState
    "Matrix of spins"
    spins::Matrix{SVector{3,<: Number}}
    "Size of layer"
    N::Tuple{<: Integer,<: Integer}
    
    "Construct layer from matrix"
    SpinState2D(s) = new(s, size(s))
end

# Generalize indices to work with SpinState

Base.size(S::SpinState2D) = (size(S.spins))

Base.getindex(S::SpinState2D, i::Int, j::Int) = S.spins[i,j]

function Base.setindex!(S::SpinState2D, v, i::Int, j::Int) 
    S.spins[i,j] = v
    return v
end

"Create a copy of a spin state"
function Base.copy(S::T) where T <: SpinState
    x = T(copy(S.spins))
    return x
end

"Normalize each spin in a spin state to unity"
function LinearAlgebra.normalize!(S::T) where T <: SpinState
    for i in eachindex(S.spins)
        S.spins[i] = normalize(S.spins[i])
    end
end

"Type of a history of the time evolution of spin states"
struct SpinHistory
    "Times"
    times::Vector
    "State at a certain time"
    states::Vector{<:SpinState}
    
    "Construct a new SpinHistory with state s at time 0.0"
    function SpinHistory(s::T) where T <: SpinState
        new([0.0], [s])
    end
end

# Generalize indices to work with SpinHistory

function Base.getindex(S::SpinHistory, i::Int)
    # SpinHistory at index 0 defined to be the same as at index 1
    if i==0 return S.states[1]
    else return S.states[i] end
end

function Base.setindex!(S::SpinHistory, v::T, i::Int) where T<:SpinState
    S.states[i] = v
    return v
end

"Type of LLG parameters"
mutable struct LlgParams
    "Heisenberg exchange"
    J::Real
    "Easy z axis anisotropy"
    K1::Real
    "Easy y axis anisotropy"
    K2::Real
    "External magnetic field vector"
    B::SVector{3,Real}
    "Gilbert damping parameter"
    αG::Real
    "Use Non-Markovian kernel?"
    ost::Bool
    "Non-Markovian kernel"
    kernels::Union{AbstractMatrix, Nothing}
    "Sites subject to light"
    sites::Union{Integer, NTuple}
    "Jsd coupling"
    jsd::Real
end
