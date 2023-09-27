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

struct Elasticity
    G::Float64
    K::Float64
end

function material_routine(material::Elasticity, ε::SymmetricTensor{2})
    (; G, K) = material
    stress(ε) = 2G * dev(ε) + K * tr(ε) * one(ε)
    ∂σ∂ε, σ = gradient(stress, ε, :all)
    return σ, ∂σ∂ε
end

E = 200e3 # Young's modulus [MPa]
ν = 0.3 # Poisson's ratio [-]
material = Elasticity(E/2(1+ν), E/3(1-2ν));

function assemble_cell!(ke, fe, cellvalues, material, ue)
    fill!(ke, 0.0)
    fill!(fe, 0.0)

    n_basefuncs = getnbasefunctions(cellvalues)
    for q_point in 1:getnquadpoints(cellvalues)
        # For each integration point, compute strain, stress and material stiffness
        ε = function_symmetric_gradient(cellvalues, q_point, ue)
        σ, ∂σ∂ε = material_routine(material, ε)

        dΩ = getdetJdV(cellvalues, q_point)
        for i in 1:n_basefuncs
            ∇Nᵢ = shape_gradient(cellvalues, q_point, i)# shape_symmetric_gradient(cellvalues, q_point, i)
            fe[i] += σ ⊡ ∇Nᵢ * dΩ # add internal force to residual
            for j in 1:n_basefuncs
                ∇ˢʸᵐNⱼ = shape_symmetric_gradient(cellvalues, q_point, j)
                ke[i, j] += (∂σ∂ε ⊡ ∇ˢʸᵐNⱼ) ⊡ ∇Nᵢ * dΩ
            end
        end
    end
end

function assemble_global!(K, f, a, dh, cellvalues, material)
    # Allocate the element stiffness matrix and element force vector
    n_basefuncs = getnbasefunctions(cellvalues)
    ke = zeros(n_basefuncs, n_basefuncs)
    fe = zeros(n_basefuncs)
    # Create an assembler
    assembler = start_assemble(K, f)
    # Loop over all cells
    for cell in CellIterator(dh)
        reinit!(cellvalues, cell) # update spatial derivatives based on element coordinates
        @views ue = a[celldofs(cell)]
        # Compute element contribution
        assemble_cell!(ke, fe, cellvalues, material, ue)
        # Assemble ke and fe into K and f
        assemble!(assembler, celldofs(cell), ke, fe)
    end
    return K, f
end

K = create_sparsity_pattern(dh)
f = zeros(ndofs(dh))
a = zeros(ndofs(dh))
assemble_global!(K, f, a, dh, cellvalues, material);

apply!(K, f, ch)
u = K \ f;

vtk_grid("linear_elasticity", dh) do vtk
    vtk_point_data(vtk, dh, u)
    vtk_cellset(vtk, grid)
end

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl
