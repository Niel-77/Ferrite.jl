using Ferrite, FerriteGmsh, SparseArrays

grid = togrid("logo.geo");

addfaceset!(grid, "top", x->x[2] ≈ 1.0)
addfaceset!(grid, "left", x->x[1] < 1e-6)
addfaceset!(grid, "bottom", x->x[2] < 1e-6);

dim = 2
order = 1 # linear interpolation
ip = Lagrange{RefTriangle, order}()^dim # vector valued interpolation
qr = QuadratureRule{RefTriangle}(1) # 1 quadrature point
cellvalues = CellValues(qr, ip);

dh = DofHandler(grid)
add!(dh, :u, ip)
close!(dh);

ch = ConstraintHandler(dh)
add!(ch, Dirichlet(:u, getfaceset(grid, "bottom"), (x, t) -> 0.0, 2))
add!(ch, Dirichlet(:u, getfaceset(grid, "left"), (x, t) -> 0.0, 1))
add!(ch, Dirichlet(:u, getfaceset(grid, "top"), (x, t) -> 0.1, 2))
close!(ch);

E = 200e3 # Young's modulus [MPa]
ν = 0.3 # Poisson's ratio [-]

λ = E*ν / ((1 + ν) * (1 - 2ν)) # 1st Lamé parameter
μ = E / (2(1 + ν)) # 2nd Lamé parameter
I = one(SymmetricTensor{2, dim}) # 2nd order unit tensor
II = one(SymmetricTensor{4, dim}) # 4th order symmetric unit tensor
∂σ∂ε = 2μ * II + λ * (I ⊗ I) # elastic stiffness tensor

function assemble_cell!(ke, cellvalues, ∂σ∂ε)
    fill!(ke, 0.0)

    n_basefuncs = getnbasefunctions(cellvalues)
    for q_point in 1:getnquadpoints(cellvalues)
        dΩ = getdetJdV(cellvalues, q_point)
        for i in 1:n_basefuncs
            ∇Nᵢ = shape_gradient(cellvalues, q_point, i)# shape_symmetric_gradient(cellvalues, q_point, i)
            for j in 1:n_basefuncs
                ∇ˢʸᵐNⱼ = shape_symmetric_gradient(cellvalues, q_point, j)
                ke[i, j] += (∂σ∂ε ⊡ ∇ˢʸᵐNⱼ) ⊡ ∇Nᵢ * dΩ
            end
        end
    end
end

function assemble_global!(K, dh, cellvalues, ∂σ∂ε)
    # Allocate the element stiffness matrix
    n_basefuncs = getnbasefunctions(cellvalues)
    ke = zeros(n_basefuncs, n_basefuncs)
    # Create an assembler
    assembler = start_assemble(K)
    # Loop over all cells
    for cell in CellIterator(dh)
        reinit!(cellvalues, cell) # update spatial derivatives based on element coordinates
        # Compute element contribution
        assemble_cell!(ke, cellvalues, ∂σ∂ε)
        # Assemble ke and fe into K and f
        assemble!(assembler, celldofs(cell), ke)
    end
    return K
end

K = create_sparsity_pattern(dh)
assemble_global!(K, dh, cellvalues, ∂σ∂ε);

f = zeros(ndofs(dh));

apply!(K, f, ch)
u = K \ f;

vtk_grid("linear_elasticity", dh) do vtk
    vtk_point_data(vtk, dh, u)
    vtk_cellset(vtk, grid) # export cellsets of grains for logo-coloring
end

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl
