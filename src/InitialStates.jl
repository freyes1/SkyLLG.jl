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

end
