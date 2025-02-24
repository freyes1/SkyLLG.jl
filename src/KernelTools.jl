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
