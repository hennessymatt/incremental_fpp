from fenics import *
from multiphenics import *
import numpy as np
from mesh_generation import *
from swelling_step import *
from recovery_step import *
import matplotlib.pyplot as plt

def boundary_array(x, y):
    return [np.array([ex, ey]) for (ex, ey) in zip(x, y)]

def solver():

    params = {"dt": 1e-1, "t_i": 0.05, "Q": 10e-1}
    # params = {"dt": 1e-1 / 2, "t_i": 1, "Q": 1e-1}

    total_layers = 20

    Nx = 30
    Ny = 3
    L = 30
    x = np.linspace(0, L, Nx)
    x[1:-1] *= (1 + 1e-2 * (2 * np.random.random(Nx-2) - 1))
    y = np.linspace(0, 1, Ny)

    bottom = boundary_array(x, np.zeros(Nx))

    right = total_layers * [0]
    top = total_layers * [0]
    left = total_layers * [0]

    J_layer = np.ones(total_layers)

    # right[0] = boundary_array(10 * np.ones(Ny), y)
    # top[0] = boundary_array(np.flip(x), np.ones(Nx))
    # left[0] = boundary_array(0 * y, np.flip(y))

    # right[1] = boundary_array(10 * np.ones(Ny), y + 1)
    # top[1] = boundary_array(np.flip(x), np.ones(Nx) + 1)
    # left[1] = boundary_array(0 * y, np.flip(y) + 1)

    # right[2] = boundary_array(10 * np.ones(Ny), y + 2)
    # top[2] = boundary_array(np.flip(x), np.ones(Nx) + 2)
    # left[2] = boundary_array(0 * y, np.flip(y) + 2)

    plt.figure()
    for layers in range(1, total_layers+1):

        print('========================================')
        print(f'Solving problem with {layers} layers')
        print('========================================')

        """
        FPP step
        """
        if layers == 1:
            x_f = np.flip(x)
            z_f = np.zeros(Nx)
        else:
            x_f = np.array([e[0] for e in top[layers-2]])
            z_f = np.array([e[1] for e in top[layers-2]])
        
        dz = np.exp(-z_f) / params["t_i"] * params["dt"]
        mean_dz = np.mean(dz)

        # plt.cla()
        # plt.plot(x_f, z_f)
        # plt.title(f'{layers} layers')
        # plt.pause(0.1)


        top[layers-1] = boundary_array(x_f, z_f + dz)
        right[layers-1] = boundary_array(Ny * [x_f[0]], np.linspace(z_f[0], z_f[0] + dz[0], Ny))
        left[layers-1] = boundary_array(Ny * [x_f[-1]], np.linspace(z_f[-1] + dz[-1], z_f[-1], Ny))

        """
        Build the mesh
        """
        print('--------Building the mesh-------------')
        mesh, subdomains, bdry, ids = build_mesh(layers, bottom, right, top, left)
    
        """
        Eulerian solve to recover the deformation
        """
        if layers > 1:
            print('--------Eulerian solve-----------')
            conv, X_eul = recovery_step(layers, J_layer[0:layers], mesh, subdomains, bdry, ids)
            U_eul = X_eul[0:layers]

            if not(conv):
                print('Eulerian solve failed to converge')
                return

        else:
            U_eul = None

        """
        Swelling step
        """
        print('--------Swelling step----------------')
        conv, X, J_layer[layers-1] = swelling_step(layers, J_layer[0:layers-1], mesh, subdomains, bdry, ids, params, U_eul)

        if not(conv):
            print('Swelling step failed to converge')
            return


        print(J_layer)
        U = X[0:layers]


        """
        Update the layer boundaries
        """
        for n in range(layers):
            for side in [right, top, left]:
                for i in range(len(side[n])):
                    tmp = U[n](side[n][i][0], side[n][i][1])
                    # print(f'x = {side[n][i][0]:.2f}, z = {side[n][i][1]:.2f}, U = {tmp[0]:.2e}, W = {tmp[1]:.2e}')
                    side[n][i] += tmp
    

        x_f_1 = np.array([e[0] for e in top[layers-1]])
        z_f_1 = np.array([e[1] for e in top[layers-1]])
        # print(f'post vol(0) = {-np.trapz(z_f_1, x_f_1)}')
        plt.clf()
        plt.plot(x_f_1, z_f_1)
        plt.pause(0.1)
        # plt.hlines([0.055], 0, 2, colors = "black", linestyles='dashed')





solver()
plt.show()