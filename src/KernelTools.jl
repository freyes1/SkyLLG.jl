using ImageFiltering

function read_kernel(filename)
    full_kernel = []

    open(filename, "r") do file
        for line in eachline(file)
            # Split the line into individual numbers
            values = parse.(Float64, split(line))
            push!(full_kernel, values)
        end
    end

    # Fill with zeros to make it square
    for i in 1:length(full_kernel)
        while length(full_kernel[i]) < length(full_kernel[end])
            push!(full_kernel[i], 0)
        end
    end
    
    return transpose(hcat(full_kernel...))
end

function cut_kernel!(ker, center, std_dev; nsigma=2.5)
    n_time = size(ker)[1]
    for j in 1:(n_time)
        for i in 1:(n_time)
            # Calculate the conditions
            out_pulse = (abs(i-center) > std_dev * nsigma) || (abs(j-center) > std_dev * nsigma)

            # Set the value to 0 if out_pulse is true
            if out_pulse; ker[i, j] = 0; end
        end
    end 
end

function filter_kernel(ker; sigma=5)
    return imfilter(ker, Kernel.gaussian(5))
end
