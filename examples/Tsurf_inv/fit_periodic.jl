using FFTW
using Statistics

function fit_sine(xdata::Vector{T}, ydata::Vector{T}; sample_rate=nothing) where {T<:Number}
    if sample_rate === nothing
        sample_rate = 1 / Statistics.mean(xdata[2:end] - xdata[1:end-1])
    end

    offset = (maximum(ydata)+minimum(ydata))/2
    ampl = (maximum(ydata) - minimum(ydata))/2
    fft = FFTW.rfft(ydata .- offset)
    fftfreq = FFTW.rfftfreq(length(ydata), sample_rate)

    # Determine dominant frequency and calculate phase shift.
    maxfreqloc = argmax(abs.(fft))
    maxfreq = fftfreq[maxfreqloc]
    shift = mod(atan(imag(fft[maxfreqloc]), real(fft[maxfreqloc])) + pi/2, 2pi)
    freq = maxfreq * 2pi

    return (; ampl, freq, shift, offset)
end
