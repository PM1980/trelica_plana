import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def solve_truss(coordinates, connectivity, boundary_conditions, loads):
    # Número de nós e elementos
    num_nodes = coordinates.shape[0]
    num_elements = connectivity.shape[0]

    # Matriz de rigidez global
    K = np.zeros((2*num_nodes, 2*num_nodes))

    # Montagem da matriz de rigidez global
    for e in range(num_elements):
        node_i, node_j = connectivity[e]
        xi, yi = coordinates[node_i]
        xj, yj = coordinates[node_j]
        L = np.sqrt((xj-xi)**2 + (yj-yi)**2)
        c = (xj-xi)/L
        s = (yj-yi)/L
        k = np.array([[c*c, c*s, -c*c, -c*s],
                      [c*s, s*s, -c*s, -s*s],
                      [-c*c, -c*s, c*c, c*s],
                      [-c*s, -s*s, c*s, s*s]])
        dofs = [2*node_i, 2*node_i+1, 2*node_j, 2*node_j+1]
        for i in range(4):
            for j in range(4):
                K[dofs[i], dofs[j]] += k[i,j]

    # Aplicar condições de contorno
    for node, (ux, uy) in enumerate(boundary_conditions):
        if ux == 0:
            K[2*node, :] = 0
            K[2*node, 2*node] = 1
        if uy == 0:
            K[2*node+1, :] = 0
            K[2*node+1, 2*node+1] = 1

    # Resolver o sistema
    F = np.zeros(2*num_nodes)
    for node, (fx, fy) in enumerate(loads):
        F[2*node] = fx
        F[2*node+1] = fy

    U = np.linalg.solve(K, F)

    return U

def plot_truss(coordinates, connectivity):
    fig, ax = plt.subplots()
    for e in connectivity:
        node_i, node_j = e
        xi, yi = coordinates[node_i]
        xj, yj = coordinates[node_j]
        ax.plot([xi, xj], [yi, yj], 'b-')
    ax.scatter(coordinates[:, 0], coordinates[:, 1], color='r')
    ax.set_aspect('equal')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Geometria da Treliça')
    return fig

st.title('Solução de Treliças Planas')

st.header('Entrada de Dados')

# Exemplo de validação
st.subheader('Exemplo de Validação')
st.write("Coordenadas: [[0,0], [3,0], [0,4]]")
st.write("Conectividade: [[0,1], [1,2], [2,0]]")
st.write("Condições de Contorno: [[0,0], [0,0], [1,1]]")
st.write("Carregamentos: [[0,0], [0,-10], [0,0]]")

# Campos de entrada
coordinates = st.text_area('Coordenadas (formato: [[x1,y1], [x2,y2], ...]):', value='[[0,0], [3,0], [0,4]]')
connectivity = st.text_area('Conectividade (formato: [[n1,n2], [n3,n4], ...]):', value='[[0,1], [1,2], [2,0]]')
boundary_conditions = st.text_area('Condições de Contorno (formato: [[ux1,uy1], [ux2,uy2], ...]):', value='[[0,0], [0,0], [1,1]]')
loads = st.text_area('Carregamentos (formato: [[fx1,fy1], [fx2,fy2], ...]):', value='[[0,0], [0,-10], [0,0]]')

if st.button('Resolver'):
    try:
        # Converter entradas para arrays numpy
        coordinates = np.array(eval(coordinates))
        connectivity = np.array(eval(connectivity))
        boundary_conditions = np.array(eval(boundary_conditions))
        loads = np.array(eval(loads))

        # Resolver a treliça
        U = solve_truss(coordinates, connectivity, boundary_conditions, loads)

        # Mostrar resultados
        st.subheader('Resultados')
        st.write('Deslocamentos:')
        for i, u in enumerate(U.reshape(-1, 2)):
            st.write(f'Nó {i}: ({u[0]:.6f}, {u[1]:.6f})')

        # Plotar geometria
        st.subheader('Geometria da Treliça')
        fig = plot_truss(coordinates, connectivity)
        st.pyplot(fig)

    except Exception as e:
        st.error(f'Erro: {str(e)}')
