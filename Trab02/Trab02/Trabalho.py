import numpy as np

def gauss_seidel(matriz_coef, matriz_results, e = 0.005, iter = 0):
    dim = matriz_coef.shape[0]
    erro = 100
    vetor_x_inicial = np.zeros(dim)
    
    while(erro > e and iter <=100):
        vetor_calculo_erro = vetor_x_inicial.copy() 
        iter += 1            
        for i in range(dim):
            soma = 0
            for j in range(dim):
                if j != i:
                    soma += matriz_coef[i, j] * vetor_x_inicial[j]
            vetor_x_inicial[i] = (matriz_results[i] - soma) / matriz_coef[i, i]
        termo_superior = np.max(np.abs(vetor_x_inicial - vetor_calculo_erro))
        termo_inferior = np.max(np.abs(vetor_x_inicial))
        erro = termo_superior / termo_inferior

    return vetor_x_inicial, iter        

def gauss_jacobi(matriz_coef, matriz_results, e = 0.005, iter = 0):
    dim = matriz_coef.shape[0]
    erro = 100
    vetor_x_inicial = np.zeros(dim)
    
    while(erro > e and iter <=100):
        vetor_calculo_erro = vetor_x_inicial.copy()
        iter += 1            
        for i in range(dim):
            soma = 0
            for j in range(dim):
                if j != i:
                    soma += matriz_coef[i, j] * vetor_calculo_erro[j]
            vetor_x_inicial[i] = (matriz_results[i] - soma) / matriz_coef[i, i]
        termo_superior = np.max(np.abs(vetor_x_inicial - vetor_calculo_erro))
        termo_inferior = np.max(np.abs(vetor_x_inicial))
        erro = termo_superior / termo_inferior

    return vetor_x_inicial, iter        

def calcular_inversa_metodo(matriz_a, e):
    dim = matriz_a.shape[0]
    identidade = np.eye(dim)
    saida = np.zeros((dim, dim))
    iteracoes = list()
    for i in range(dim):
        saida[:, i], iter = gauss_seidel(matriz_a, identidade[:,i], e)
        iteracoes.append(iter)
    return saida, iteracoes    

def calcular_deslocamentos(matriz_a, matriz_b, n, e):
    inversa, lista_iter = calcular_inversa_metodo(matriz_a, e)
    d = np.abs(inversa @ matriz_b)
    return d[0:n], lista_iter[0:n], inversa

# BLOCO DE EXECUÇÃO INTERATIVA 
if __name__ == "__main__":
    print("=" * 70)
    print("      INICIALIZANDO PROGRAMA DE CÁLCULO NUMÉRICO - TEMA 2      ")
    print("=" * 70)
    
    # 1. GERAÇÃO AUTOMÁTICA DOS CENÁRIOS DO QUADRO RESPOSTA (ITEM D)
    A1 = np.array([[5.0, 3.0, 1.0], [5.0, 6.0, 1.0], [1.0, 6.0, 7.0]])
    b1 = np.array([1.0, 2.0, 3.0])
    A2 = A1.copy()
    b2 = np.array([3.0, 5.0, 2.0])
    A3 = np.array([[1.0, 5.0, 2.0], [4.0, 1.0, 3.0], [1.0, 2.0, 1.0]])
    
    cenarios = [
        ("1: CALIBRAÇÃO PADRÃO (Dados do Enunciado)", A1, b1),
        ("2: SOBRECARGA FÍSICA (Variação de b)", A1, b2),
        ("3: INSTABILIDADE NUMÉRICA (Variação de A)", A3, b1)
    ]
    
    precisao_padrao = 0.005
    n_padrao = 3
    
    print("\n>>> EXECUÇÃO AUTOMÁTICA DOS CENÁRIOS PADRÃO (ITEM D)...\n")
    
    for nome, m_A, v_b in cenarios:
        print("-" * 60)
        print(f" Cenário {nome} ")
        print("-" * 60)
        
        # Processamento Gauss-Seidel (Via sua função original)
        d_seidel, iter_seidel, _ = calcular_deslocamentos(m_A, v_b, n_padrao, precisao_padrao)
        max_seidel = np.max(d_seidel)
        status_seidel = "🚨 DANOS ESTRUTURAIS!" if max_seidel > 0.4 else "✅ SEGURO"
        
        # Processamento Gauss-Jacobi (Via loop paralelo modular)
        identidade = np.eye(n_padrao)
        inv_jacobi = np.zeros((n_padrao, n_padrao))
        iter_jacobi = []
        for idx in range(n_padrao):
            inv_jacobi[:, idx], col_iter = gauss_jacobi(m_A, identidade[:, idx], precisao_padrao)
            iter_jacobi.append(col_iter)
        d_jacobi = np.abs(inv_jacobi @ v_b)
        max_jacobi = np.max(d_jacobi)
        status_jacobi = "🚨 DANOS ESTRUTURAIS!" if max_jacobi > 0.4 else "✅ SEGURO"
        
        print(f" [GAUSS-SEIDEL] Iterações por coluna: {iter_seidel} | Deslocamentos: {d_seidel}")
        print(f"                Status Estrutural: {status_seidel}")
        print(f" [GAUSS-JACOBI] Iterações por coluna: {iter_jacobi} | Deslocamentos: {d_jacobi}")
        print(f"                Status Estrutural: {status_jacobi}\n")

    print("=" * 70)
    print(" QUADRO-RESPOSTA PADRÃO EXECUTADO COM SUCESSO! ")
    print("=" * 70)
    
    # 2. MENU INTERATIVO PARA INSERÇÃO DE NOVOS CASOS (CUMPRE ENTRADA DE DADOS)
    while True:
        opcao = input("\nDeseja analisar um NOVO caso customizado? (S/N): ").strip().upper()
        if opcao != 'S':
            print("\nEncerrando o programa.")
            input("Pressione Enter para fechar a janela...")
            break
            
        print("\n--- INSERÇÃO DE DADOS CUSTOMIZADOS ---")
        try:
            n = int(input("Digite o número de deslocamentos (n): "))
            e = float(input("Digite a precisão desejada (e): "))
            
            print(f"\nDigite as {n} linhas da matriz A (separe os números por espaço):")
            linhas_A = []
            for i in range(n):
                linha = list(map(float, input(f"  Linha {i+1}: ").split()))
                if len(linha) != n:
                    raise ValueError(f"Cada linha precisa ter exatamente {n} elementos.")
                linhas_A.append(linha)
            A_custom = np.array(linhas_A)
            
            print(f"\nDigite os {n} elementos do vetor b (separados por espaço):")
            b_custom = np.array(list(map(float, input("  Vetor b: ").split())))
            if len(b_custom) != n:
                raise ValueError(f"O vetor b precisa ter exatamente {n} elementos.")
                
            print("\n>>> PROCESSANDO NOVO SISTEMA LINEAR...")
            
            d_s_cust, iter_s_cust, inv_s_cust = calcular_deslocamentos(A_custom, b_custom, n, e)
            
            I_cust = np.eye(n)
            inv_j_cust = np.zeros((n, n))
            iter_j_cust = []
            for idx in range(n):
                inv_j_cust[:, idx], c_it = gauss_jacobi(A_custom, I_cust[:, idx], e)
                iter_j_cust.append(c_it)
            d_j_cust = np.abs(inv_j_cust @ b_custom)
            
            print("\n" + "-" * 60)
            print(" RESULTADOS DO SEU CASO CUSTOMIZADO ")
            print("-" * 60)
            print(f"Matriz Inversa A^-1 obtida:\n{inv_s_cust}\n")
            print(f"[GAUSS-SEIDEL] Iterações por coluna: {iter_s_cust}")
            print(f"[GAUSS-SEIDEL] Deslocamentos: {d_s_cust}")
            print(f"Status Estrutural (Seidel): {'🚨 DANOS ESTRUTURAIS!' if np.max(d_s_cust) > 0.4 else '✅ SEGURO'}")
            print("-" * 60)
            print(f"[GAUSS-JACOBI] Iterações por coluna: {iter_j_cust}")
            print(f"[GAUSS-JACOBI] Deslocamentos: {d_j_cust}")
            print(f"Status Estrutural (Jacobi): {'🚨 DANOS ESTRUTURAIS!' if np.max(d_j_cust) > 0.4 else '✅ SEGURO'}")
            print("-" * 60)
            
        except Exception as err:
            print(f"\n❌ Erro na entrada de dados: {err}. Certifique-se de usar apenas números e espaços.")