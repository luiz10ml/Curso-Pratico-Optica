# Curso-Pratico-Optica

![Inatel](https://img.shields.io/badge/Instituição-Inatel-blue)
![Nível](https://img.shields.io/badge/Nível-Graduação-success)
![Área](https://img.shields.io/badge/Área-Telecomunicações-informational)

---

# 📡 Curso Prático: Predistorção Digital (DPD) com Redes Neurais

Este repositório contém o material didático para implementação de uma **Predistorção Digital (DPD)** utilizando Redes Neurais do tipo **MLP (Multi-Layer Perceptron)** para linearizar um **Modulador Mach-Zehnder (MZM)** em sistemas **Radio-over-Fiber (RoF)**.

---

# 📖 1. Introdução e Setup

<img src="/figuras/rofDiagram.png" width="900px"> 


## 📡 Representação Física vs. Modelo Matemático

O diagrama acima apresenta **duas formas equivalentes** de representar um sistema **Radio-over-Fiber (RoF)**.

---

### 🔬 Sistema Físico Real (Diagrama da Esquerda)

À esquerda, temos o sistema físico completo, composto por:

- Transmissor SDR (Tx)  
- Amplificador de Potência (PA)  
- Laser (LD)  
- Modulador Mach-Zehnder (MZM)  
- Caminho Óptico  
- Fotodiodo (PD)  
- Receptor (Rx)  

O funcionamento ocorre da seguinte forma:

1. O transmissor gera o sinal elétrico.
2. O sinal é amplificado pelo PA.
3. O MZM converte o sinal elétrico em modulação óptica.
4. O sinal óptico se propaga pelo enlace.
5. O fotodiodo reconverte o sinal óptico para o domínio elétrico.

Durante esse processo surgem **não-linearidades**, principalmente associadas ao MZM.  
Como consequência, o sinal de saída não é uma cópia perfeita do sinal de entrada.

---

### 📐 Modelo Matemático Equivalente (Diagrama da Direita)

À direita, o mesmo sistema é representado por um **modelo polinomial sem memória**, que descreve a relação entre entrada e saída:

<img src="/figuras/modelo.png" width="200px"> 

Onde:

- `v_n` → amostra de entrada  
- `z_n` → amostra de saída  
- `h_j` → coeficientes complexos do modelo  
- `J` → ordem do polinômio  

#### 📌 O que significa "sem memória"?

Significa que a saída no instante `n` depende apenas da entrada naquele mesmo instante.  
Não há dependência de amostras anteriores (`v_{n-1}`, `v_{n-2}`, etc.).

Se os coeficientes `h_j` forem corretamente estimados a partir de medições reais, esse modelo consegue reproduzir com alta fidelidade o comportamento do sistema físico completo, conforme ilustram as Figuras abaixo:

<img src="/figuras/desempenho.png" width="900px"> 


---

## 🎯 Por que usar um modelo em vez do sistema físico?

Substituir o sistema físico por um modelo matemático traz vantagens fundamentais, especialmente em ambiente acadêmico:

- ✅ Permite gerar amostras ilimitadas sem laboratório  
- ✅ Reduz drasticamente custos (equipamentos ópticos são caros)  
- ✅ Garante reprodutibilidade total dos experimentos  
- ✅ Facilita o treinamento de algoritmos de DPD  
- ✅ Permite explorar diferentes níveis de não-linearidade alterando apenas os coeficientes  

Em outras palavras:

> O modelo substitui um sistema físico complexo por uma representação matemática compacta, viabilizando simulações, testes e desenvolvimento de algoritmos de forma rápida, controlada e acessível.

É exatamente esse modelo que será utilizado neste curso para treinar uma rede neural MLP capaz de realizar a **Predistorção Digital (DPD)** e compensar os efeitos não-lineares do modulador.

---


Preparação do ambiente no Google Colab:

```python
# Instalação da biblioteca de modulação
!pip install ModulationPy -q

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from ModulationPy import QAMModem
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from scipy.signal import welch
import os


# Early Stopping
callback_dpd = tf.keras.callbacks.EarlyStopping(
    monitor='loss',
    patience=50,
    min_delta=1e-9,
    restore_best_weights=True
)
```

# ⚙️ 2. Teste da GPU

Preparação do ambiente no Google Colab:

```python
# %% Verificação da GPU
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  print('Connected without GPU Device')
else: 
  print('Connected with GPU Device')
```

---

# ⚙️ 3. Parâmetros do Sistema OFDM

| Parâmetro | Valor | Descrição |
|-----------|--------|-----------|
| K | 2048 | Tamanho da FFT |
| NUM_BLOCOS | 10 | Quantidade de blocos |
| MOD_ORDER | 16 | Modulação 16-QAM |
| SNR_DB | 45 | Relação Sinal-Ruído |
| J | 2 | Ordem da não-linearidade |

```python
# --- CONFIGURAÇÕES DO SISTEMA ---
K = 2048                                  # Número de subportadoras (Tamanho da FFT)
NUM_BLOCOS = 10                           # Quantidade de blocos OFDM para processamento
SUBPORT_ATIVAS = np.arange(-200, 201, 1)  # Espectro ocupado
MOD_ORDER = 16                            # 16-QAM
SNR_DB = 45                               # Relação Sinal-Ruído do canal
J = 2                                     # Ordem do modelo polinomial do modulador (Não-linearidade)
```

---

# 🧠 4. Funções de Apoio

```python
def suavizar_espectro(vetor_db, janela=20):
    """Aplica uma média móvel robusta para deixar o gráfico da DEP bem suave."""
    return np.convolve(vetor_db, np.ones(janela)/janela, mode='same')

def modular_qam(bits_ou_indices, ordem):
    """Realiza a modulação QAM e normaliza a potência unitária."""
    modem = QAMModem(ordem, bin_input=False, soft_decision=False, bin_output=False)
    simbolos = modem.modulate(bits_ou_indices)
    potencia_media = np.mean(np.abs(simbolos)**2)
    simbolos_norm = simbolos * np.sqrt(1 / potencia_media)
    return simbolos_norm, modem

def mapear_ofdm(simbolos_qam, indices_ativos, tamanho_fft):
    """Aloca os símbolos QAM nas subportadoras específicas da FFT."""
    espectro = np.zeros(tamanho_fft, dtype=complex)
    espectro[indices_ativos] = simbolos_qam
    return espectro

def canal_awgn(sinal, snr_db, pot_referencia):
    """Adiciona ruído gaussiano branco (AWGN) baseado na SNR desejada."""
    sigma2 = pot_referencia * 10**(-snr_db/10)
    ruido = np.sqrt(sigma2/2) * (np.random.randn(*sinal.shape) + 1j*np.random.randn(*sinal.shape))
    return sinal + ruido

def modelo_mzm(coeficientes, sinal_in, ordem):
    """Representa o comportamento não-linear do Modulador Mach-Zehnder."""
    # Matriz onde cada coluna é o sinal elevado a uma potência ímpar (comum em RF)
    X = np.column_stack([sinal_in * np.abs(sinal_in)**k for k in range(ordem)])
    return X.dot(coeficientes)

def calcular_evm(simbolos_est, simbolos_ref):
    """Calcula o Error Vector Magnitude em porcentagem."""
    erro = simbolos_est - simbolos_ref
    return np.sqrt(np.mean(np.abs(erro)**2) / np.mean(np.abs(simbolos_ref)**2)) * 100

def calcular_mer(simbolos_est, simbolos_ref):
    """Calcula o Modulation Error Ratio em dB."""
    erro = simbolos_est - simbolos_ref
    p_sinal = np.mean(np.abs(simbolos_ref)**2)
    p_erro = np.mean(np.abs(erro)**2)
    return 10 * np.log10(p_sinal / p_erro)
```

---

# 📊 5. Carregando os Coeficientes do Modelo Polinomial

<img src="/figuras/data.png" width="900px"> 
---

## 🧪 Coleta de Dados Experimentais

A figura acima mostra o setup experimental utilizado para a coleta de dados do sistema **RoF**.

O objetivo principal do experimento foi coletar dois conjuntos de amostras:

- 🔹 **Sinal de entrada sem distorção (TX Data)**  
- 🔹 **Sinal de saída distorcido (RX Data)**  

O sinal transmitido pelo SDR é aplicado ao PA e ao Modulador Mach-Zehnder (MZM), propagando-se pelo enlace óptico.  
No receptor, o fotodiodo converte novamente o sinal para o domínio elétrico, resultando em uma versão distorcida do sinal original.

Esses dois sinais (entrada e saída) são armazenados para posterior modelagem do sistema.

---

## ⏱️ Alinhamento Temporal (Sincronização)

Após a coleta, as sequências TX e RX não estão perfeitamente alinhadas no tempo devido a:

- Atrasos do enlace
- Atrasos de hardware
- Processamento interno dos dispositivos

Para corrigir isso, é necessário realizar **sincronização temporal** utilizando **correlação cruzada** entre o sinal transmitido e o sinal recebido.

Um **preambulo conhecido** é inserido no início do sinal transmitido, permitindo realizar o alinhamento temporal.

Para estimar `h`, utilizamos a Equação Normal dos Mínimos Quadrados:

<img src="/figuras/coeficientes.png" width="150px"> 


Onde:

- `V^H` é a transposta conjugada de `V`

- `h` é o vetor estimado de coeficientes

Essa solução minimiza o erro quadrático médio entre o modelo e o sistema real.



```python
# --- GERAÇÃO DE DADOS E CANAL (CENÁRIO SEM DPD) ---


# Link para o arquivo RAW no seu GitHub
# Substitua 'SEU_USUARIO' e 'SEU_REPOSITORIO' pelos seus dados reais
url_coef = "https://raw.githubusercontent.com/luiz10ml/Curso-Pratico-Optica/main/coeficientes/coef"


# Faz o download do arquivo para o ambiente do Colab
if not os.path.exists("coef"):
    !wget {url_coef} -O coef

# Carregamento dos dados
coef_mzm = np.fromfile("coef", dtype=np.complex64)

print(f"Coeficientes carregados com sucesso!: {coef_mzm}")

```
---
# 📊 6. Geração dos dados
```python
# Vetores para armazenar o sinal completo
sinal_tx_total = np.zeros(NUM_BLOCOS * K, dtype=complex)

for i in range(NUM_BLOCOS):
    p_tx_dbm = np.random.randint(-5, 16)
    p_tx_linear = 10**(p_tx_dbm/10) * 1e-3
    
    indices = np.random.randint(0, MOD_ORDER, size=len(SUBPORT_ATIVAS))
    qam_norm, _ = modular_qam(indices, MOD_ORDER)
    
    espectro_mapeado = mapear_ofdm(qam_norm, SUBPORT_ATIVAS, K)
    sinal_tempo = np.fft.ifft(espectro_mapeado) * np.sqrt(K)
    
    escala = np.sqrt(p_tx_linear / np.mean(np.abs(sinal_tempo)**2))
    sinal_tx_total[i*K : (i+1)*K] = sinal_tempo * escala

# Passagem pelo Modulador Não-Linear (MZM) e Canal
sinal_distorcido = modelo_mzm(coef_mzm, sinal_tx_total, J)
sinal_recebido = canal_awgn(sinal_distorcido, SNR_DB, np.mean(np.abs(sinal_tx_total)**2))
```

---

# 🤖 7. Arquitetura e Treinamento da Rede Neural (DPD)

<img src="/figuras/train.png" width="900px"> 


Após obter o par de sinais **entrada sem distorção** e **saída distorcida** (já sincronizados no tempo), treinamos uma rede neural do tipo **MLP (Multi-Layer Perceptron)** para aprender a operação inversa da distorção.

A ideia é simples:

- **Entrada da MLP (features):** amostras do sinal **distorcido** no receptor  
- **Saída desejada (labels):** amostras do sinal **original sem distorção** no transmissor

Em outras palavras, a MLP recebe o sinal “estragado” (distorcido) e aprende a estimar qual era o sinal original antes do enlace RoF.

Como os sinais são complexos, o processamento é feito separando:

- Parte real (I)
- Parte imaginária (Q)

Assim, cada amostra complexa vira um vetor real de 2 dimensões:


### 🎯 Função de Custo (Erro Quadrático Médio)

Os coeficientes (pesos) da MLP são ajustados minimizando o **Erro Quadrático Médio (MSE)** entre:

- o **label** (sinal original sem distorção), e
- a **estimativa produzida pela MLP** na saída

A função custo usada no treinamento é o erro

```math
J_e(\mathbf{w}) = \frac{1}{N}\left\|\mathbf{v} - \hat{\mathbf{v}}\right\|^2
```


```python
X_train = np.c_[sinal_recebido.real, sinal_recebido.imag]
y_train = np.c_[sinal_tx_total.real, sinal_tx_total.imag]

model_dpd = Sequential([
    Dense(2048, activation='relu', input_shape=(2,)),
    Dense(1024, activation='relu'),
    Dense(2)
])

model_dpd.compile(optimizer='adam', loss='mse')

model_dpd.fit(
    X_train,
    y_train,
    epochs=200,
    batch_size=K,
    verbose=2,
    callbacks=[callback_dpd]
)
```

---

# 🏁 8. Teste e Validação

```python
print("\n--- Avaliando Performance ---")
p_teste_dbm = 10 
p_teste_lin = 10**(p_teste_dbm/10) * 1e-3

indices_teste = np.random.randint(0, MOD_ORDER, size=len(SUBPORT_ATIVAS))
qam_teste, modem_obj = modular_qam(indices_teste, MOD_ORDER)
sinal_ofdm_teste = np.fft.ifft(mapear_ofdm(qam_teste, SUBPORT_ATIVAS, K)) * np.sqrt(K)
sinal_ofdm_teste *= np.sqrt(p_teste_lin / np.mean(np.abs(sinal_ofdm_teste)**2))

# 1. Caso SEM DPD
saida_sem_dpd = modelo_mzm(coef_mzm, sinal_ofdm_teste, J)
energia_saida = np.mean(abs(saida_sem_dpd)**2)
saida_sem_dpd = saida_sem_dpd*np.sqrt(np.mean(abs(sinal_ofdm_teste)**2)/energia_saida)

# 2. Caso COM DPD
sinal_entrada_mlp = np.c_[sinal_ofdm_teste.real, sinal_ofdm_teste.imag]
sinal_pre_distorcido_raw = model_dpd.predict(sinal_entrada_mlp, verbose=0)
sinal_pre_distorcido = sinal_pre_distorcido_raw[:,0] + 1j*sinal_pre_distorcido_raw[:,1]
saida_com_dpd = modelo_mzm(coef_mzm, sinal_pre_distorcido, J)

# --- DEMODULAÇÃO E CÁLCULO DE MÉTRICAS ---
def processar_receptor(sinal_rx, ref_qam):
    rx_f = np.fft.fft(sinal_rx) / np.sqrt(K)
    qam_rx = rx_f[SUBPORT_ATIVAS]
    qam_rx *= np.sqrt(np.mean(np.abs(ref_qam)**2) / np.mean(np.abs(qam_rx)**2))
    evm_val = calcular_evm(qam_rx, ref_qam)
    mer_val = calcular_mer(qam_rx, ref_qam)
    return qam_rx, evm_val, mer_val

qam_sem, evm_sem, mer_sem = processar_receptor(saida_sem_dpd, qam_teste)
qam_com, evm_com, mer_com = processar_receptor(saida_com_dpd, qam_teste)
```

---
# 🏁 9. Análise Espectral

```python
f_ref, p_ref = welch(sinal_ofdm_teste, fs=1.0, window='hann', nperseg=K, return_onesided=False)
f_sem, p_sem = welch(saida_sem_dpd, fs=1.0, window='hann', nperseg=K, return_onesided=False)
f_com, p_com = welch(saida_com_dpd, fs=1.0, window='hann', nperseg=K, return_onesided=False)

# Suavização e centralização (Janela aumentada para suavizar mais)
db_ref = np.fft.fftshift(suavizar_espectro(10*np.log10(p_ref + 1e-12), janela=41))
db_sem = np.fft.fftshift(suavizar_espectro(10*np.log10(p_sem + 1e-12), janela=41))
db_com = np.fft.fftshift(suavizar_espectro(10*np.log10(p_com + 1e-12), janela=41))
f_plot = np.fft.fftshift(f_sem)
```

---
# 🏁 10. Plotando os Resultados

```python
plt.figure(figsize=(12, 10))

# Plot 1: Constelação Sem DPD (Topo Esquerda)
plt.subplot(221)
plt.scatter(qam_sem.real, qam_sem.imag, s=5, label=f'Sem DPD ({evm_sem:.1f}%)')
plt.title("Constelação: Sem DPD")
plt.grid(); plt.legend(loc='upper right')

# Plot 2: Constelação Com DPD (Topo Direita)
plt.subplot(222)
plt.scatter(qam_com.real, qam_com.imag, s=5, color='green', label=f'Com DPD ({evm_com:.1f}%)')
plt.title("Constelação: Com MLP-DPD")
plt.grid(); plt.legend(loc='upper right')

# Plot 3: Densidade Espectral de Potência (Base - Ocupando as duas colunas)
plt.subplot(212)
plt.plot(f_plot, db_ref, color='black', label='Ideal (Referência)', linestyle='--', alpha=0.6)
plt.plot(f_plot, db_sem, label='Sem DPD (Distorcido)', alpha=0.8)
plt.plot(f_plot, db_com, label='Com MLP-DPD (Linearizado)', color='green', linewidth=2)
plt.title("Densidade Espectral de Potência (Suavizada)")
plt.xlabel("Frequência Normalizada")
plt.ylabel("Magnitude (dB)")
plt.ylim(np.max(db_ref)-60, np.max(db_ref)+5)
plt.grid(); plt.legend()

plt.tight_layout()
plt.show()

print(f"RESULTADO FINAL ({p_teste_dbm} dBm):")
print(f"Sem DPD -> MER: {mer_sem:.2f} dB | EVM: {evm_sem:.2f}%")
print(f"Com DPD -> MER: {mer_com:.2f} dB | EVM: {evm_com:.2f}%")
```

---

# 🚀 Como usar

1. Abra o Google Colab  
2. Copie os blocos em células separadas  
3. Execute em ordem  

---

> 💡 **Exercicios para casa:**  
Altere o número de neurônios ou trocar `relu` por `tanh`, `elu`, `selu`, `LeakyReLU` e observar o impacto no desempenho.


## 📚 Referências Bibliográficas

[1] Luiz. A. M. Pereira; Luciano. L. Mendes; Mitchell A. Cox; Arismar Cerqueira S. Jr.. Open dataset initiative for machine learning-based linearization in analog radio over fiber systems. Optics Communications, v. 590, p. 131949, 2025. ISSN 0030-4018. DOI: 10.1016/j.optcom.2025.131949. Disponível em: https://www.sciencedirect.com/science/article/pii/S0030401825004778




Prof. Dr. Luiz Augusto Melo Pereira

luiz.melo@inatel.br