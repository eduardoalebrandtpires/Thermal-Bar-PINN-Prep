import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches

"""
PROJETO: EQUAÇÃO DO CALOR 2D (versão completa)

Objetivos:
1) Resolver numericamente a equação do calor em 2D
2) Verificar estabilidade do método de diferenças finitas
3) Comparar solução numérica com solução analítica
4) Analisar erro numérico
5) Testar convergência do método

Equação:
dT/dt = α (d²T/dx² + d²T/dy²)

Etapa futura:
Treinar uma PINN (Physics-Informed Neural Network)
para resolver a mesma equação e comparar desempenho
"""

# PARÂMETROS FÍSICOS
L = 1.0 #Barra
alpha = 0.01
T_quente = 100
T_fria = 0

# DISCRETIZAÇÃO
Nx = 60
Ny = 60
Nt = 2000
T_total = 2
dx = L/(Nx-1)
dy = L/(Ny-1)
dt = T_total/Nt

# ESTABILIDADE
r = alpha * dt / dx**2
print(f"Coeficiente r = {r:.4f}")
if r > 0.25:
    print("Instável! Diminua dt ou aumente dx")
else:
    print("Estável")

# MATRIZ
u = np.zeros((Nt, Nx, Ny))
u[0,:,:] = 0  # condição inicial

# CONDIÇÕES DE CONTORNO
u[:,0,:] = T_quente   # esquerda
u[:,-1,:] = T_fria    # direita
u[:,:,0] = 0          # baixo
u[:,:,-1] = 0         # cima

# SIMULAÇÃO
for n in range(Nt-1):
    for i in range(1, Nx-1):
        for j in range(1, Ny-1):
            u[n+1,i,j] = u[n,i,j] + r * (
                u[n,i+1,j] + u[n,i-1,j] +
                u[n,i,j+1] + u[n,i,j-1] -
                4*u[n,i,j]
            )

# SOLUÇÃO ANALÍTICA ESTACIONÁRIA (linear entre bordas esquerda e direita)
x = np.linspace(0,L,Nx)
y = np.linspace(0,L,Ny)
X,Y = np.meshgrid(x,y,indexing='ij')
u_analitica = T_quente + (T_fria - T_quente) * X/L

# CÁLCULO DE ERRO
erro = np.abs(u[-1] - u_analitica)
erro_medio = np.mean(erro)
print(f"Erro médio entre solução numérica e analítica: {erro_medio:.4f}")

# PLOT FINAL
fig, ax = plt.subplots(figsize=(6,5))
im = ax.imshow(u[-1], cmap='hot', extent=[0,L,0,L])

# DESENHO DA BARRA
rect = patches.Rectangle((0,0), L, L, linewidth=2, edgecolor='blue', facecolor='none')
ax.add_patch(rect)


plt.colorbar(im, label="Temperatura")
plt.title("Distribuição de Temperatura 2D (Final) com Barra")
plt.xlabel("x")
plt.ylabel("y")
plt.tight_layout()
plt.show()

# ANIMAÇÃO
fig, ax = plt.subplots()
im = ax.imshow(u[0], cmap='hot', extent=[0,L,0,L])
rect = patches.Rectangle((0,0), L, L, linewidth=2, edgecolor='blue', facecolor='none')
ax.add_patch(rect)


def update(frame):
    im.set_array(u[frame])
    ax.set_title(f"Tempo = {frame*dt:.2f}")
    return [im]

anim = FuncAnimation(fig, update, frames=range(0, Nt, 20), interval=50)
plt.show()

# FUNÇÃO PARA TESTAR CONVERGÊNCIA
def teste_convergencia(dx_list, dt_list):
    erros = []
    for dx_test, dt_test in zip(dx_list, dt_list):
        Nx_test = int(L/dx_test)+1
        Ny_test = Nx_test
        Nt_test = int(T_total/dt_test)
        r_test = alpha*dt_test/dx_test**2
        u_test = np.zeros((Nt_test, Nx_test, Ny_test))
        u_test[0,:,:] = 0
        u_test[:,0,:] = T_quente
        u_test[:,-1,:] = T_fria
        u_test[:,:,0] = 0
        u_test[:,:,-1] = 0
        for n in range(Nt_test-1):
            for i in range(1,Nx_test-1):
                for j in range(1,Ny_test-1):
                    u_test[n+1,i,j] = u_test[n,i,j] + r_test*(
                        u_test[n,i+1,j] + u_test[n,i-1,j] +
                        u_test[n,i,j+1] + u_test[n,i,j-1] -
                        4*u_test[n,i,j]
                    )
        # Solução analítica
        x_test = np.linspace(0,L,Nx_test)
        X_test,_ = np.meshgrid(x_test,x_test,indexing='ij')
        u_analitica_test = T_quente + (T_fria - T_quente)*X_test/L
        erro_test = np.mean(np.abs(u_test[-1] - u_analitica_test))
        erros.append(erro_test)
    return erros

# EXEMPLO DE TESTE DE CONVERGÊNCIA
dx_list = [0.1,0.05,0.025]
dt_list = [T_total/200, T_total/400, T_total/800]
erros_conv = teste_convergencia(dx_list, dt_list)
print("Erros médios para diferentes resoluções (teste de convergência):", erros_conv)
