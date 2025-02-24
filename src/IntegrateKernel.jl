"Type for parameters of the kernel"
mutable struct KernelParams
    "Hopping parameter"
    hop::Real
    "Chemical potential"
    μ::Real
    "Position where kernel is calculated"
    x::Real
    "Frequency of pulse"
    f::Real
    "Standard deviation of pulse"
    w::Real
    "Adimensional amplitude of pulse"
    z::Real
    "Center of pulse"
    t0::Real
    "Timestep"
    tstep::Real
    "Number of timesteps"
    n_time::Real
    "Pulse profile"
    pulse::Vector
    "Cos of the pulse"
    cosa::Vector
    "Sin of the pulse"
    sina::Vector
    
    function KernelParams(hop::Real, μ::Real, x::Real, f::Real, w::Real, z::Real, t0::Real, tstep::Real, n_time::Real)
        times = tstep:tstep:tstep*n_time
        pulse = create_pulse(times, f,w,z,t0);
        cosa, sina = integrate_light(times, pulse);
        
        return new(hop, μ, x, f, w, z, t0, tstep, n_time, pulse,cosa,sina)
    end
end

"Create pulse profile"
function create_pulse(t, f,w,z,t0)
    offset = fill(t0, length(t))
    envelope = exp.(-0.5*(t.-t0).^2/w^2)
    wave = cos.(2π*f*t)
    return z*envelope.*wave
end

"""
    integrate_light(times, pulse, p)

Integrate cosine and sine of pulse profiles using trapezoid method. 
Returns a time matrix that is upper triangular.
"""
function integrate_light(times, pulse)
    n_time = length(times)
    d = round(Integer, n_time * (n_time-1)/2)
    bigcosa = zeros(d)
    bigsina = zeros(d)

    cosa = cos.(pulse)
    sina = sin.(pulse)
    
    idx = 1
    for j = 1:n_time
        for i = j+1:n_time
            prob1 = SampledIntegralProblem(cosa[j:i], times[j:i])
            prob2 = SampledIntegralProblem(sina[j:i], times[j:i])
            sol1 = solve(prob1, TrapezoidalRule())
            sol2 = solve(prob2, TrapezoidalRule())
            bigcosa[idx] = sol1.u
            bigsina[idx] = sol2.u
            idx += 1
        end
    end
    
    return bigcosa, bigsina
end

"Compute bounds of integration domain at zero temperature"
function create_domain(p::KernelParams)
    kfermi = acos(p.μ/(2*p.hop))
    return ([-kfermi,-π], [kfermi,π])
end

"""
    big_integrand(out, bigk, p)

Create in-place function that returns a vector of integrands. The vector
should be reshaped into an uppper triangular matrix.
"""
function big_integrand(out, bigk, p::KernelParams)
    cosa = p.cosa
    sina = p.sina
    k, q = bigk
    x= p.x
            
    en = -2*p.hop*(cos(k) - cos(q))
    vn = -2*p.hop*(sin(k) - sin(q))

    for idx = 1:length(cosa)
        out[idx] = -4*sin((k-q)*x - en*cosa[idx] + vn*sina[idx])/(2π)^2
    end
end

"Integrates to obtain kernel in the form of upper triangular matrix"
function compute_kernel(p::KernelParams; convert=false)
    d = round(Integer, p.n_time*(p.n_time-1)/2)
    
    # The prototype has second dimension zero because the quadrature can figure out
    # on its own the appropriate batch number, so the zero is just a placeholder.
    prototype = zeros(d)
    domain = create_domain(p)
    
    # Here it uses the in-place batched version of the integrand
    func = IntegralFunction(big_integrand, prototype)
    prob = IntegralProblem(func, domain, p)
    
    # CubatureJLp is faster than CubatureJLh
    sol = solve(prob, CubatureJLp(); reltol = 1e-3, abstol = 2e-1)
    
    # println("Success? $(sol.retcode)")
    if convert res = convert_to_triangular(sol.u, p.n_time)
    else res = sol.u end
    
    return res
end

"Create in-place batched version of big_integrand."
function big_integrand_b(out, bigk, p::KernelParams)
    # The columns of the argument of the batched function are arguments for the  
    # original function. The output of the batched function is an array whose last 
    # dimension indexes the results. For example, out[:,1] is the result of the 
    # original function applied to in[:,1], and so on.
    
    cosa = p.cosa
    sina = p.sina
    x = p.x
    
    for aux in 1:size(bigk, 2)
        k, q = bigk[:,aux]
        en = -2*p.hop*(cos(k) - cos(q))
        vn = -2*p.hop*(sin(k) - sin(q))
        for idx = 1:length(p.cosa)
            out[idx, aux] = -4*sin((k-q)*x - en*cosa[idx] + vn*sina[idx])/(2π)^2
        end
    end
end

"Create in-place parallel batched version of big_integrand."
function big_integrand_bp(out, bigk, p::KernelParams)
    cosa = p.cosa
    sina = p.sina
    x= p.x
    
    # in_threads is for integrating. 4 is optimal in 1D
    in_threads = 4
    in_chunk = floor(Integer, size(bigk, 2)/in_threads)
    in_tasks = []
    
    for thr=1:in_threads
        start_in = (thr-1) * in_chunk + 1
        end_in = (thr == in_threads) ? size(bigk, 2) : thr * in_chunk
        
        in_task = Threads.@spawn begin
        for aux in start_in:end_in
            k, q = bigk[:,aux]
            en = -2*p.hop*(cos(k) - cos(q))
            vn = -2*p.hop*(sin(k) - cos(q))
            for idx = 1:length(cosa)
                out[idx, aux] = -4*sin((k-q)*x - en*cosa[idx] + vn*sina[idx])/(2π)^2
            end
        end 
        end
        push!(in_tasks, in_task)
    end
    
    fetch.(in_tasks)
end

"Batched integral to obtain kernel in the form of upper triangular matrix"
function compute_kernel_b(p::KernelParams; convert=false)
    d = round(Integer, p.n_time*(p.n_time-1)/2)
    
    # The prototype has second dimension zero because the quadrature can figure out
    # on its own the appropriate batch number, so the zero is just a placeholder.
    prototype = zeros(d, 0)
    domain = create_domain(p)
    
    # Here it uses the in-place batched version of the integrand
    func = BatchIntegralFunction(big_integrand_bp, prototype)
    prob = IntegralProblem(func, domain, p)
    
    # CubatureJLp is faster than CubatureJLh
    sol = solve(prob, CubatureJLp(); reltol = 1e-3, abstol = 2e-1)
    
    # println("Success? $(sol.retcode)")
    if convert res = convert_to_triangular(sol.u, p.n_time)
    else res = sol.u end
    
    return res
end

"Create in-place parallel batched version of big_integrand."
function batched_chunked_integrand(out, bigk, tp)
    p, start_out, end_out, out_ch = tp
    cosa = p.cosa
    sina = p.sina
    x= p.x
    
    # in_threads is for integrating. 4 is optimal in 1D
    in_threads = 4
    in_chunk = floor(Integer, size(bigk, 2)/in_threads)
    in_tasks = []
    
    for thr=1:in_threads
        start_in = (thr-1) * in_chunk + 1
        end_in = (thr == in_threads) ? size(bigk, 2) : thr * in_chunk
        
        in_task = Threads.@spawn begin
        for aux in start_in:end_in
            k, q = bigk[:,aux]
            en = -2*p.hop*(cos(k) - cos(q))
            vn = -2*p.hop*(sin(k) - sin(q))
            for idx = 1:out_ch
                out[idx, aux] = -4*sin((k-q)*x - en*cosa[start_out+idx-1] + vn*sina[start_out+idx-1])/(2π)^2
            end
        end 
        end
        push!(in_tasks, in_task)
    end
    
    fetch.(in_tasks)
end

"Integrates to obtain kernel in the form of upper triangular matrix"
function compute_kernel_p(p::KernelParams, tot_threads; ch_sz = 2e4, convert = false)    
    domain = create_domain(p)
    d = round(Integer, p.n_time*(p.n_time-1)/2)
    
    out_threads = round(Integer, tot_threads/4)
    n_chunks = max(out_threads, round(Integer, d/ch_sz))
    chunk_size = floor(Integer, d/n_chunks)
    chunk_series = round(Integer, n_chunks/out_threads)
    remainder = n_chunks - chunk_series*out_threads
    
    println("There are $n_chunks chunks between $out_threads threads"); flush(stdout)
    
    out_tasks = []
    
    for th=1:out_threads
        out_task = Threads.@spawn begin
            res_of_thread = []
            
            for ch=1:chunk_series + (th==out_threads)*remainder
                s = (ch-1)*chunk_size+(th-1)*chunk_size*chunk_series+1
                if (th==out_threads && ch==chunk_series + remainder)
                    e = d
                else e = s + chunk_size - 1 end
               
                chsz = e - s + 1
                prototype = zeros(chsz, 0)

                func = BatchIntegralFunction(batched_chunked_integrand, prototype)
                prob = IntegralProblem(func, $domain, (p, s, e, chsz))

                sol = solve(prob, CubatureJLp(); reltol = 1e-3, abstol = 2e-1)
                
                println("Chunk $(ch), thread $(th) was $(sol.retcode)ful.")
                flush(stdout)
                
                push!(res_of_thread, sol.u)
            end
            vcat(res_of_thread...)
        end
        push!(out_tasks, out_task)
    end
    
    res = fetch.(out_tasks)
    res = vcat(res...)
    
    if convert res = convert_to_triangular(res, p.n_time) end
    
    return res
end

"Convert vector into upper triangular matrix"
function convert_to_triangular(vec, d)
    idx = 1
    out = zeros(d, d)
    for j = 1:d
        for i = j+1:d
            out[i,j] = vec[idx]
            idx += 1
        end
    end
    return out
end

