# We start by including `Ferrite` and `Test`, 
# to allow us to verify our implementation.
using Ferrite, Test

# Define a simple version of the cell values object, which only supports
# identity mapping of scalar interpolations, without embedding. 
struct SimpleCellValues{T, dim} <: Ferrite.AbstractCellValues
    N::Matrix{T}             # Precalculated shape values
    dNdξ::Matrix{Vec{dim,T}} # Precalculated shape gradients in the reference domain
    dNdx::Matrix{Vec{dim,T}} # Cache for shape gradients in the real domain
    M::Matrix{T}             # Precalculated geometric shape values
    dMdξ::Matrix{Vec{dim,T}} # Precalculated geometric shape gradients
    weights::Vector{T}       # Given quadrature weights in the reference domain
    detJdV::Vector{T}        # Cache for quadrature weights in the real domain
end;

# To make it easier to initiate this struct, we create a constructor function
# with the same input as `CellValues`. However, we skip some consistency checking here. 
function SimpleCellValues(qr::QuadratureRule, ip_fun::Interpolation, ip_geo::Interpolation, T=Float64)
    dim = Ferrite.getdim(ip_fun)
    ## Quadrature weights and coordinates (in reference cell)
    weights = Ferrite.getweights(qr)
    n_qpoints = length(weights)

    ## Function interpolation
    n_func_basefuncs = getnbasefunctions(ip_fun)
    N    = zeros(T,          n_func_basefuncs, n_qpoints)
    dNdx = zeros(Vec{dim,T}, n_func_basefuncs, n_qpoints)
    dNdξ = zeros(Vec{dim,T}, n_func_basefuncs, n_qpoints)

    ## Geometry interpolation
    n_geom_basefuncs = getnbasefunctions(ip_geo)
    M    = zeros(T,          n_geom_basefuncs, n_qpoints)
    dMdξ = zeros(Vec{dim,T}, n_geom_basefuncs, n_qpoints)

    ## Precalculate function and geometric shape values and gradients
    for (qp, ξ) in pairs(Ferrite.getpoints(qr))
        for i in 1:n_func_basefuncs
            dNdξ[i, qp], N[i, qp] = Ferrite.shape_gradient_and_value(ip_fun, ξ, i)
        end
        for i in 1:n_geom_basefuncs
            dMdξ[i, qp], M[i, qp] = Ferrite.shape_gradient_and_value(ip_geo, ξ, i)
        end
    end

    detJdV = zeros(T, n_qpoints)
    SimpleCellValues(N, dNdξ, dNdx, M, dMdξ, weights, detJdV)
end;

# We first define the `getnbasefunctions`
Ferrite.getnbasefunctions(cv::SimpleCellValues) = size(cv.N, 1);

# Before we define the `reinit!` function to calculate the cached 
# values `dNdx` and `detJdV` for the current cell
function Ferrite.reinit!(cv::SimpleCellValues, x::Vector{Vec{dim,T}}) where {dim,T}
    for (q_point, w) in pairs(cv.weights) # Loop over each quadrature point
        ## Calculate the jacobian, J
        J = zero(Tensor{2,dim,T})
        for i in eachindex(x)
            J += x[i] ⊗ cv.dMdξ[i, q_point]
        end
        ## Calculate the correct integration weight for the current q_point
        cv.detJdV[q_point] = det(J)*w
        ## map the shape gradients to the current geometry 
        Jinv = inv(J)
        for i in 1:getnbasefunctions(cv)
            cv.dNdx[i, q_point] = cv.dNdξ[i, q_point] ⋅ Jinv
        end
    end
end;

# To make our `SimpleCellValues` work in standard Ferrite code, we need to define how 
# to get the shape value and graident: 
Ferrite.shape_value(cv::SimpleCellValues, q_point::Int, i::Int) = cv.N[i, q_point]
Ferrite.shape_gradient(cv::SimpleCellValues, q_point::Int, i::Int) = cv.dNdx[i, q_point]

# We are now ready to test, so let's create an instance of our new and the regular cell values
qr = QuadratureRule{RefQuadrilateral}(2)
ip = Lagrange{RefQuadrilateral,1}()
simple_cv = SimpleCellValues(qr, ip, ip)
cv = CellValues(qr, ip, ip)

# The first thing to try is to reinitialize the cell values to a given cell
grid = generate_grid(Quadrilateral, (2,2))
x = getcoordinates(grid, 2)
reinit!(simple_cv, x)
reinit!(cv, x)

# If we now pretend we are inside an element, where we have a vector of element 
# degree of freedom values, we can check that the function values and gradients match
ue = rand(getnbasefunctions(simple_cv))
q_point = 2
@test function_value(cv, q_point, ue) ≈ function_value(simple_cv, q_point, ue)
@test function_gradient(cv, q_point, ue) ≈ function_gradient(simple_cv, q_point, ue)