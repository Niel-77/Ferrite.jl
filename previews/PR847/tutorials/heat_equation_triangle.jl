using Ferrite, SparseArrays

grid = generate_grid(Triangle, (20, 20));

ip = Lagrange{RefTriangle, 1}()
qr = QuadratureRule{RefTriangle}(2)
cellvalues = CellValues(qr, ip);

dh = DofHandler(grid)
add!(dh, :u, ip)
close!(dh);

K = create_sparsity_pattern(dh)

ch = ConstraintHandler(dh);

∂Ω = union(
    getfaceset(grid, "left"),
    getfaceset(grid, "right"),
    getfaceset(grid, "top"),
    getfaceset(grid, "bottom"),
);

dbc = Dirichlet(:u, ∂Ω, (x, t) -> 0)
add!(ch, dbc);

close!(ch)

function assemble_element!(Ke::Matrix, fe::Vector, cellvalues::CellValues)
    n_basefuncs = getnbasefunctions(cellvalues)
    # Reset to 0
    fill!(Ke, 0)
    fill!(fe, 0)
    # Loop over quadrature points
    for q_point in 1:getnquadpoints(cellvalues)
        # Get the quadrature weight
        dΩ = getdetJdV(cellvalues, q_point)
        # Loop over test shape functions
        for i in 1:n_basefuncs
            δu  = shape_value(cellvalues, q_point, i)
            ∇δu = shape_gradient(cellvalues, q_point, i)
            # Add contribution to fe
            fe[i] += δu * dΩ
            # Loop over trial shape functions
            for j in 1:n_basefuncs
                ∇u = shape_gradient(cellvalues, q_point, j)
                # Add contribution to Ke
                Ke[i, j] += (∇δu ⋅ ∇u) * dΩ
            end
        end
    end
    return Ke, fe
end

function assemble_global(cellvalues::CellValues, K::SparseMatrixCSC, dh::DofHandler)
    # Allocate the element stiffness matrix and element force vector
    n_basefuncs = getnbasefunctions(cellvalues)
    Ke = zeros(n_basefuncs, n_basefuncs)
    fe = zeros(n_basefuncs)
    # Allocate global force vector f
    f = zeros(ndofs(dh))
    # Create an assembler
    assembler = start_assemble(K, f)
    # Loop over all cels
    for cell in CellIterator(dh)
        # Reinitialize cellvalues for this cell
        reinit!(cellvalues, cell)
        # Compute element contribution
        assemble_element!(Ke, fe, cellvalues)
        # Assemble Ke and fe into K and f
        assemble!(assembler, celldofs(cell), Ke, fe)
    end
    return K, f
end

K, f = assemble_global(cellvalues, K, dh);

apply!(K, f, ch)
u = K \ f;

vtk_grid("heat_equation", dh) do vtk
    vtk_point_data(vtk, dh, u)
end

@show norm(u)/length(u)

function calculate_flux_lag(dh, dΩ, ip, a)
    qr = FaceQuadratureRule{RefTriangle}(2)
    fv = FaceValues(qr, ip, Lagrange{RefTriangle,1}())
    grid = dh.grid
    dofrange = dof_range(dh, :u)
    flux = 0.0
    dofs = celldofs(dh, 1)
    ae = zeros(length(dofs))
    x = getcoordinates(grid, 1)
    for (cellnr, facenr) in dΩ
        getcoordinates!(x, grid, cellnr)
        cell = getcells(grid, cellnr)
        celldofs!(dofs, dh, cellnr)
        map!(i->a[i], ae, dofs)
        reinit!(fv, x, facenr, cell)
        for q_point in 1:getnquadpoints(fv)
            dΓ = getdetJdV(fv, q_point)
            n = getnormal(fv, q_point)
            q = function_gradient(fv, q_point, ae, dofrange)
            flux -= (q ⋅ n)*dΓ
        end
    end
    return flux
end

flux = calculate_flux_lag(dh, ∂Ω, ip, u)
@show flux

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl
