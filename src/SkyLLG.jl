module SkyLLG

using LinearAlgebra
using StaticArrays
using Integrals, Cubature

export SpinState1D, SpinState2D, SpinHistory, LlgParams
export create_initial_state, evolve!, reformat
export cut_kernel!
export KernelParams
export compute_kernel, compute_kernel_b, compute_kernel_p
export convert_to_triangular

include("SpinTypes.jl")
include("KernelTools.jl")
include("IntegrateKernel.jl")

const x̂ = SVector(1,0,0)
const ŷ = SVector(0,1,0)
const ẑ = SVector(0,0,1)

# =============== STATE TOOLS =================

"Create a SpinState of type T from a vector or matrix of spherical polar angles"
function create_initial_state(angs, T)
    stat = similar(angs, SVector{3, <: Number})
    for (i,a) in enumerate(angs)
        θ, ϕ = a
        stat[i] = SVector(sin(θ)*cos(ϕ), sin(θ)*sin(ϕ), cos(θ))
    end
    return T(stat)
end

"Get a copy of the states in hist between now - depth and now"
function get_state_past(hist::SpinHistory, site, now, depth)
    last = now-depth+1
    res = [hist[t][site] for t in last:now]
    return copy(res)
end

"""
    reformat(S::SpinHistory)

Reformat S into a matrix with columns given by:

[S.times s1x s1y s1z s2x s2y s2z ...]
"""
function reformat(S::SpinHistory)
    data = []
    for s = 1:length(S.times)
        row = Vector(reduce(vcat, S[s].spins))
        pushfirst!(row, S.times[s])
        push!(data, row)
    end
    
    res = hcat(data...)
    return copy(transpose(res))
end

# =============== AUXILIARY FUNCTIONS ================

"Integrates a function over a 1D interval using trapezoid method"
function trapezoid(y, dx)
    if length(y) < 2; return y[1]-y[1] end
    res = 0.5*(y[1] + y[end])
    res += sum(y[2:end-1])
    return dx*res
end

# =============== PHYSICAL FUNCTIONS =================

"""
    effective_field(state, p)

Compute the effective magnetic field at each site of a state. Heisenberg exchange
is treated using open boundary conditions. Anisotropy is easy axis if K>0.
"""
function effective_field(state::SpinState1D, p::LlgParams)
    field = SpinState1D(similar(state.spins))
    
    nsites = state.N
    
    for s = 1:nsites
        exch = @SVector zeros(3)
        
        if s > 1             # Has a neighbor on the left
            exch += p.J * state[s-1]
        end
        if s < nsites        # Has a neighbor on the right
            exch += p.J * state[s+1]
        end
        
        anis = 2*p.K[1]*state[s][1]*x̂ + 2*p.K[2]*state[s][2]*ŷ + 2*p.K[3]*state[s][3]*ẑ
        field[s] = -exch - anis - p.B
    end
    
    return field
end

function effective_field(state::SpinState2D, p::LlgParams)
    field = SpinState2D(similar(state.spins))
    
    nrows, ncols = state.N
    
    for c = 1:ncols
        for r = 1:nrows
            exch = @SVector zeros(3)

            # Check for neighbors and accumulate the exchange contribution
            if r > 1          # Has a neighbor above
                exch += p.J * state[r-1, c]
            end
            if r < nrows      # Has a neighbor below
                exch += p.J * state[r+1, c]
            end
            if c > 1          # Has a neighbor on the left
                exch += p.J * state[r, c-1]
            end
            if c < ncols      # Has a neighbor on the right
                exch += p.J * state[r, c+1]
            end

            anis = 2*p.K[1]*state[r,c][3]*x̂ + 2*p.K[2]*state[r,c][2]*ŷ + 2*p.K[3]*state[r,c][3]*ẑ
            
            # Store the result
            field[r, c] = -exch - anis - p.B
        end
    end
    
    return field
end

raw"Compute ``αG \\partial_t \mathbf{S}_n`` for all n"
function loc_damp(curr::SpinState, prev::SpinState, dt, p::LlgParams)
    ds = copy(curr)
    
    for (i,ps) in enumerate(prev.spins)
        ds[i] -= ps
        ds[i] *= p.αG/dt
    end
    
    return ds
end

raw"Compute ``\int_{t_0}^t dt' η(t,t')\mathbf{S}(t')`` where t is now and 
t_0 is now-depth. Uses trapezoid method."
function kernel_field(hist::SpinHistory, ker, now, depth, dt, p::LlgParams)
    conv = SpinState1D(fill((@SVector [0,0,0]), length(p.sites)))
    
    for i in p.sites
        state_past = get_state_past(hist, i, now, depth)
        state_past .*= ker[depth,1:depth]
        conv[i] = p.jsd^2*trapezoid(state_past, dt)
    end
    return conv
end

# =============== EVOLUTION FUNCTIONS =================

"""
    compute_fields(hist, curr, p, now, dt)

Compute all effective fields that spins are subject to. Always computes effective magnetic field and local damping field. Only computes non-Markovian field if ost 
in p is true."""
function compute_fields(hist::SpinHistory, curr::SpinState, p::LlgParams, now, dt, dto, depth)
    glob_fields = []
    loc_fields = []
    push!(glob_fields, effective_field(curr, p))
    
    prev = hist[now - 2]
    push!(glob_fields, loc_damp(curr, prev, dto, p))
    
    if p.ost 
        push!(loc_fields, kernel_field(hist, p.kernels, now-1, depth, dt, p)) 
    end
    
    return glob_fields, loc_fields
end

"Compute torques by taking cross product of spins and fields at each site"
function torques(state::T, p::LlgParams, glob_fields, loc_fields) where T <: SpinState
    tau = T(fill((@SVector [0,0,0]), state.N...))
    
    for b in glob_fields
        for (i,s) in enumerate(state.spins)
            tau[i] += s × b[i]
        end
    end
    
    for b in loc_fields
        for i in p.sites
            tau[i] += state.spins[i] × b[i]
        end
    end
    
    return tau
end

"Predict next SpinState"
function predict(state::T, ds::T, dt) where T <: SpinState
    
    prediction = copy(state)
    
    for (i,s) in enumerate(state.spins)
        prediction[i] += dt*ds[i]
    end
    
    normalize!(prediction)
    
    return prediction
end

"Correct prediction of next SpinState"
function correct(state::T, ds1::T, ds2::T, dt) where T<: SpinState
    
    correction = copy(state)
    
    for (i,s) in enumerate(state.spins)
        correction[i] += 0.5*dt*(ds1[i] + ds2[i])
    end
    
    normalize!(correction)
    
    return correction
end

"""
    evolve!(history, new_times, p)

Evolve history throughout new_times using LlgParams p and a predictor-corrector 
method (Heun). The new_times are added to the previous history.times so that 
new_times represents the increments. In other words, new_times[1] is the timestep
of the new_times.
"""
function evolve!(history::SpinHistory, new_times, p::LlgParams)
    tstep = new_times[1]
    t1 = length(history.times)
    
    # If t1==1, then it is the first time evolving history
    if t1==1 tstepold = tstep
    # tstepold is used for the time derivative of the first step
    else tstepold = history.times[t1]-history.times[t1-1] end
    
    for t = 1:lastindex(new_times)
        current = history[t1 + t - 1]
        
        fields = compute_fields(history, current, p, t+t1, tstep, tstepold, t)       
        torques1 = torques(current, p, fields...)
        
        prediction = predict(current, torques1, tstep)
        
        push!(history.states, prediction)
        
        fields = compute_fields(history, prediction, p, t+t1+1, tstep, tstepold, t+1)
        torques2 = torques(prediction, p, fields...)
        
        history[t1+t] = correct(current, torques1, torques2, tstep)
        
        # Change tstepold for tstep of new_times after 
        # initial time derivative is computed
        if t==1 tstepold=tstep end
    end
    
    next_times = fill(history.times[t1], length(new_times)) .+ new_times
    append!(history.times, next_times)
end

end
