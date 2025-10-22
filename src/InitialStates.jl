module InitialStates

using NonlinearSolve:NonlinearProblem, solve
using Distributions:Exponential, truncated

"""
    thermal_state_z(N, z, kbT, p::LlgParams)

Picks N initial angles from mean field Boltzmann distribution with ground state 
along z with coordination number z, temperature kbT, and LLG params given by p.
"""
function thermal_state_z(N, z, kbT, p)
    Jp= z*p.J + p.K1
    B = p.B[3]
    
    function mf_equation(bmf, p)
        jp, b, t = p

        return jp*coth(bmf/t)/2 - jp*t/(2*bmf) + b - bmf
    end
    
    prob = NonlinearProblem(mf_equation, 1, [Jp,B,kbT])
    sol = solve(prob)
    bmf = sol.u
    
    boltzmann = truncated(Exponential(1/(bmf/kbT)), upper=2);
    angles = [(acos.(ones() .- rand(boltzmann)), rand()*2π) for n in 1:N]
    
    return angles
end

"""
    simulated_annealing(T_init, T_final, hist::SpinHistory, p::LlgParams; fc=0.1, fh=5, Δtc=1e4, Δth=1000))

Computes ground state of system governed by params p by running many cooling/heating cycles until desired
T_final is reached. fc/fh determine how fast the temperature changes during cooling/heating periods lasting
Δtc/Δth time steps respectively. 

Details of the method in Franke, et al., Phys. Rev. B 106, 174428 (2022).
"""
function simulated_annealing(T_init, T_final, hist::SpinHistory,
        p::LlgParams; fc=0.1, fh=5, Δtc=1e4, Δth=1000)

    n_cyc = floor(Int, log(T_final/T_init)/log(fc*fh))

    println("The number of cycles is $n_cyc")

    for n=1:n_cyc
        for i=1:Δtc
            t = 0.1:0.1:0.1
            p.T = T_init*(i*(fc-1)/Δtc + 1)
            evolve!(hist, t, p)
        end

        for i=1:Δth
            t = 0.1:0.1:0.1
            p.T = T_init*fc*(i*(fh-1)/Δth + 1)
            evolve!(hist, t, p)
        end

        T_init *= fc*fh
        println("Cycle $n done. T=$T_init")
    end

    i = 1
    while p.T >= T_final
        t = 0.1:0.1:0.1
        p.T = T_init*(i*(fc-1)/Δtc + 1)
        evolve!(hist, t, p)
        i += 1
    end

    println("Final temperature is $(p.T)")

    return hist.states[end]
end


end
