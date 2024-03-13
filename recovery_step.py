from dolfin import *
from multiphenics import *
from helpers import *
import numpy as np
from mesh_generation import *
import matplotlib.pyplot as plt

#---------------------------------------------------------------------
# Global variables
#---------------------------------------------------------------------


#---------------------------------------------------------------------
# Setting up file names and paramerers
#---------------------------------------------------------------------

"""
Define file names
"""

# directory for file output
# dir = '/media/eg21388/data/fenics/stokes_elasticity/'
dir = '/home/matt/data/fenics/fpp2d_iterative/'


# output_eul_f = XDMFFile(dir + "eul_fluid.xdmf")
# output_eul_f.parameters["rewrite_function_mesh"] = False
# output_eul_f.parameters["functions_share_mesh"] = True
# output_eul_f.parameters["flush_output"] = True

"""
Solver parameters
"""
snes_solver_parameters = {"snes_solver": {"linear_solver": "mumps",
                                          "maximum_iterations": 50,
                                          "report": True,
                                          "absolute_tolerance": 1e-8,
                                          "error_on_nonconvergence": False}}

parameters["allow_extrapolation"] = True
parameters["form_compiler"]["representation"] = "uflacs"

parameters["ghost_mode"] = "shared_facet"

parameters["form_compiler"]["quadrature_degree"] = 5
parameters["form_compiler"]["optimize"]     = True
parameters["form_compiler"]["cpp_optimize"] = True



def recovery_step(layers, J_layer, mesh, subdomains, bdry, ids):

    s_all = np.logspace(-2, 0, 5)
    s_all = [1.]
    s = Constant(s_all[0])

    Os = []
    Sig = []
    for n in range(1, layers + 1):
        Os.append(generate_subdomain_restriction(mesh, subdomains, ids[f"layer{n}"]))
    
    for n in range(1, layers - 1):
        Sig.append(generate_interface_restriction(mesh, subdomains, {ids[f"layer{n}"], ids[f"layer{n+1}"]}))

    dx = Measure("dx", domain=mesh, subdomain_data=subdomains)
    ds = Measure("ds", domain=mesh, subdomain_data=bdry)
    dS = Measure("dS", domain=mesh, subdomain_data=bdry)


    #---------------------------------------------------------------------
    # elements, function spaces, and test/trial functions
    #---------------------------------------------------------------------
    P2 = VectorElement("CG", mesh.ufl_cell(), 2)
    P1 = FiniteElement("CG", mesh.ufl_cell(), 1)
    DGT = VectorElement("DGT", mesh.ufl_cell(), 1)

    elements = layers * [P2] + layers * [P1] + (layers - 2) * [DGT]

    restrict = Os + Os
    if layers > 2:
        restrict += Sig

    mixed_element = BlockElement(elements)
    V = BlockFunctionSpace(mesh, mixed_element, restrict = restrict)
    
    X = BlockFunction(V)
    Xl = list(block_split(X))
    U = Xl[0:layers]
    P = Xl[layers:2*layers]
    if layers > 2:
        Lam = Xl[2*layers:]

    # unknowns and test functions
    Y = BlockTestFunction(V)
    Yl = list(block_split(Y))
    U_ = Yl[0:layers]
    P_ = Yl[layers:2*layers]
    if layers > 2:
        Lam_ = Yl[2*layers:]

    
    Xt = BlockTrialFunction(V)

    #---------------------------------------------------------------------
    # boundary conditions
    #---------------------------------------------------------------------

    """
    Physical boundary conditions
    """

     # impose zero vertical solid displacement at the centreline axis
    bc_l = [DirichletBC(V.sub(0), Constant((0, 0)), bdry, ids["substrate"])]

    for n in range(1, layers+1):
        bc_l += [DirichletBC(V.sub(n-1).sub(0), Constant(0), bdry, ids[f"left{n}"])]
        bc_l += [DirichletBC(V.sub(n-1).sub(0), Constant(0), bdry, ids[f"right{n}"])]

    # Combine all BCs together
    bcs = BlockDirichletBC(bc_l)


    #---------------------------------------------------------------------
    # Define the model
    #---------------------------------------------------------------------

    I = Identity(2)

    """
    Solids problem
    """

    FUN = (3 * layers - 2) * [0]

    F = 2 * layers * [0]
    J = 2 * layers * [0]
    sigma = 2 * layers * [0]

    for n in range(layers):
        F[n] = inv(I - grad(U[n]))
        J[n] = det(F[n])

        sigma[n] = 1 / Constant(J_layer[n]) * (F[n] * F[n].T - I) - P[n] * I

        FUN[n] = -inner(sigma[n], grad(U_[n])) * dx(ids[f"layer{n+1}"])
        FUN[n+layers] = (J[n] - 1 + s * (1 - Constant(J_layer[n]))) * P_[n] * dx(ids[f"layer{n+1}"])

    for n in range(layers-2):
        FUN[n] += dot(Lam[n]("+"), U_[n]("+")) * dS(ids[f"top{n+1}"])
        FUN[n+1] -= dot(Lam[n]("-"), U_[n+1]("-")) * dS(ids[f"top{n+1}"])
        FUN[2*layers+n] = dot(avg(Lam_[n]), U[n]("+") - U[n+1]("+")) * dS(ids[f"top{n+1}"])


    JAC = block_derivative(FUN, X, Xt)

    #---------------------------------------------------------------------
    # set up the solver
    #---------------------------------------------------------------------

    # Initialize solver
    problem = BlockNonlinearProblem(FUN, X, bcs, JAC)
    solver = BlockPETScSNESSolver(problem)
    solver.parameters.update(snes_solver_parameters["snes_solver"])
    
    # extract solution components
    Xl = list(X.block_split())

    x = SpatialCoordinate(mesh)
    for n in range(layers):
        tmp = Expression(("0", "(1 - 1 / J) * x[1]"), J = J_layer[n], degree = 2)
        Xl[n].interpolate(tmp)  

    #---------------------------------------------------------------------
    # Set up code to save solid quanntities only on the solid domain and
    # fluid quantities only on the fluid domain
    #---------------------------------------------------------------------

    """
        Separate the meshes
    """
    # mesh_f = SubMesh(mesh, subdomains, fluid)
    # mesh_s = SubMesh(mesh, subdomains, solid)
    #
    # # Create function spaces for the velocity and displacement
    # Vf = VectorFunctionSpace(mesh_f, "CG", 1)
    # Vs = VectorFunctionSpace(mesh_s, "CG", 1)
    #
    # u_f_only = Function(Vf)
    # u_a_only = Function(Vf)
    # u_s_only = Function(Vs)

    # Python function to save solution for a given value
    # of epsilon
    # def save(eps):
    #     u_f_only = project(u_f, Vf)
    #     u_a_only = project(u_a, Vf)
    #     u_s_only = project(u_s, Vs)
    #
    #     u_f_only.rename("u_f", "u_f")
    #     u_a_only.rename("u_a", "u_a")
    #     u_s_only.rename("u_s", "u_s")
    #
    #     output_ale_f.write(u_f_only, eps)
    #     output_ale_f.write(u_a_only, eps)
    #     output_ale_s.write(u_s_only, eps)

    for ss in s_all:
        s.assign(ss)
        (its, conv) = solver.solve()

    # def def_mesh(m):
    #     x = m.coordinates()
    #     for n in range(len(x)):
    #         x[n] += Xl[0](x[n][0], x[n][1]) + Xl[1](x[n][0], x[n][1]) + Xl[2](x[n][0], x[n][1])
    #     return m

    # d_mesh = def_mesh(mesh)
    # plot(d_mesh)
    # plt.show()

    # surf = plot(det(inv(I - grad(Xl[0]))))
    # cb = plt.colorbar(surf)
    # cb.ax.ticklabel_format(style = 'plain')
    # plt.show()


    return conv, Xl