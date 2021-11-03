using DifferentialEquations,LinearAlgebra,Distances, Plots
#note:sometimes if the input is too strong the solver seems to get stuck

Iflag(t) = (1.5 < t < 15.0) #to specify the duration of external input

gridsize = 40 #size of square grid

N = gridsize*gridsize #total number of neurons

function posi(L) # function to assign a Euclidean position to each neuron
   k = 0
   pmatrix = zeros(L*L,2)
   for i in 1:L
      for j in 1:L
         k = k +1
         pmatrix[k,:] = [i,j]
      end
   end
   return pmatrix
end


function f(u,p,t) #ODEs
   x = u[1:N] # activity of each neuron
   z = u[N+1:end] # recovery variable
   m = max.(x,0)
   inp, We, Wi, Az, rz, dec = p
   dx = (2.0 .- x).*(inp .* Iflag(t) .+ We*m) .- (x .+ 2.0).*(Wi*m + Az.*z) - dec.*x
   dz = rz .*(m .- z)
   [dx; dz]
end

posits = posi(gridsize)
distmatrix = pairwise(Euclidean(),posits';dims=2)

# change these parameters to get different patterns
AWee = 1.2 # height of exitatory kernel (connection matrix)
AWie = 0.4 # height of inhibitory kernel (connection matrix)
sWee = 2.5 # Width of Gaussian for exc kernel
sWie = 4.0 # Width of Gaussian for exc kernel
Az1 = 2.0 # strength of recovery variable inhibition
rz1 = 0.3 # rate of integration of recovery variable
decayrate = 2.0 # decay rate of x

Wee = AWee .*exp.(.-(distmatrix ./ sWee ).^2.0)
Wie = AWie.*exp.(.-(distmatrix ./ sWie ).^2.0)

# Initial conditions:

#x0 = rand(0.0:0.01:0.1,N)
x0 = zeros(N,)
z0 = zeros(N,)
u0 = [x0; z0]
rinp = 0.1 .* rand(0.0:0.1:1.0,N)

# Input pattern
imatrix = zeros(gridsize,gridsize)
midp = Int(gridsize/2)
imatrix[midp-1:midp+2,:] .= 0.1
imatrix[:,midp-1:midp+2] .= 0.1


#imatrix[midp-9:midp+10,:] .= 0.1
#imatrix[:,midp-4:midp+15] .= 0.1
inp = reshape(imatrix,(N,)) .+ rinp
#inp = zeros(N,)
#inp[150:175] .= 1.0

tspan = (0.0,40.0) # duration of simulation
p = inp, Wee, Wie, Az1, rz1, decayrate

#Solution
prob = ODEProblem(f,u0,tspan,p)
sol1 = solve(prob)

s = hcat(sol1.u...)
cmax = maximum(s[1:Int(N/2),:])
#Plotting
tL = length(sol1.t)
anim = @animate for i = 1:tL
   a = sol1.u[i] #note that these are not evenly spaced in time, because of the solver
   mat = reshape(a[1:N],(gridsize,gridsize))
   heatmap(mat, clim = (0.0,cmax), c= :thermal, border = :none, aspect_ratio = :equal, legend = :false)
end

gif(anim, "neuroblobs.gif", fps = 25)
