import numpy as np
import matplotlib.pyplot as plt
"""
OBJETIVO DO PROJETO

1) Resolver numericamente a equação do calor em 1D 
2) Verificar estabilidade do método de diferenças finitas 
3) Comparar solução numérica com solução analítica 
4) Analisar erro numérico 
5) Testar convergência do método 

Etapa futura:
Treinar uma PINN (Physics-Informed Neural Network)
para resolver a mesma equação e comparar desempenho
"""

# PROJETO: PROPAGAÇÃO DE CALOR EM UMA BARRA DE FERRO
# Ideia: Ver o calor andando na barra até virar uma reta.
# Depois vou tentar comparar isso com PINNs pra ver se é melhor.


L = 1.0
alpha_real = 2.3e-5
alpha = 0.01 # valor tunado pq o real demora 12h          

T_quente = 100
T_fria = 0

#  CHECK DA FÍSICA 
# No começo achei que o códigoi tava bugado. O calor não andava.
# Percebi que o tempo real ≈ L² / α. Pro ferro dá umas 12 horas.
# Então aumentei o alpha 
t_char = L**2 / alpha_real
print(f"Tempo físico real  ≈ {t_char:.0f} s")

# DISCRETIZAÇÃO DA BARRA 
# Quebrei a barra em vários pedaços (o veterano da faculdade me deu a letra).
# Mais pedaço = mais preciso, mas o PC chora.
Nx = 60
Nt = 4000
T_total = 5

dx = L/(Nx-1)
dt = T_total/Nt
x = np.linspace(0, L, Nx)

# VERIFICAÇÃO DE ESTABILIDADE (Dica da Rita) 
# Se r > 0.5 a solução explode e vira infinito.
r = alpha * dt / dx**2
print(f"Coeficiente r = {r:.4f}")

if r > 0.5:
    print("Calma, calma! Vai estabilizar! Vai estabilizar logo! BOOMM")
else:
    print("O poder do sol... na palma da minha mão.")

#  MATRIZ DE TEMPERATURA 
# Aqui guardei a porra tuda. u[tempo, posição]
u = np.zeros((Nt, Nx))

# Condição inicial: Barra fria 
u[0,:] = 0

# Condições de contorno: Pontas fixas (Quente na esquerda, Fria na direita)
u[:,0] = T_quente
u[:,-1] = T_fria

# LOOP DA SIMULAÇÃO (DIFERENÇAS FINITAS) 
# Cada ponto recebe calor dos vizinhos 
for n in range(0, Nt-1):
    for i in range(1, Nx-1):
        u[n+1,i] = u[n,i] + r*(u[n,i+1] - 2*u[n,i] + u[n,i-1])

#  SOLUÇÃO TEÓRICA (RETA) 
# No estado estacionário, vira uma reta.
u_estacionario = T_quente + (T_fria - T_quente)*x/L

#  ANÁLISE DE ERRO 
# Saquei que o erro é intrínseco pq o PC aproxima as derivadas
erro = np.abs(u[-1] - u_estacionario)
print("\nANÁLISE DE ERRO")
print("Erro médio =", np.mean(erro))

#  GRÁFICOS
plt.figure(figsize=(8,5))
plt.plot(x, u[0], label="Início")
plt.plot(x, u[Nt//2], label="Meio")
plt.plot(x, u[-1], 'r', label="Final numérico")
plt.plot(x, u_estacionario, '--k', label="Reta teórica")
plt.title("Propagação de calor na barra")
plt.legend()
plt.grid(True)
plt.show()

# Gráfico do Erro
plt.figure(figsize=(8,4))
plt.plot(x, erro, color='orange')
plt.title("Erro da solução numérica")
plt.show()

# Gráfico da Energia
energia = np.sum(u, axis=1)*dx
plt.figure(figsize=(8,4))
plt.plot(np.linspace(0, T_total, Nt), energia)
plt.title("Energia térmica total")
plt.show()

# Mapa de Calor (padrão de paper)
plt.figure(figsize=(10,5))
plt.imshow(u, aspect='auto', cmap='hot', extent=[0, L, T_total, 0])
plt.colorbar(label="Temp")
plt.title("Mapa de calor da barra")
plt.show()

