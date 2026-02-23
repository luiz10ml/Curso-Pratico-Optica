# Curso-Pratico-Optica

![Inatel](https://img.shields.io/badge/Instituição-Inatel-blue)
![Nível](https://img.shields.io/badge/Nível-Graduação-success)
![Área](https://img.shields.io/badge/Área-Telecomunicações-informational)

---

# 📡 Curso Prático: Predistorção Digital (DPD) com Redes Neurais

Este repositório contém o material didático para implementação de uma **Predistorção Digital (DPD)** utilizando Redes Neurais do tipo **MLP (Multi-Layer Perceptron)** para linearizar um **Modulador Mach-Zehnder (MZM)** em sistemas **Radio-over-Fiber (RoF)**.

---

# 📖 1. Introdução e Setup

Preparação do ambiente no Google Colab:

```python
# Instalação da biblioteca de modulação
!pip install ModulationPy

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from ModulationPy import QAMModem
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from scipy.signal import welch

# Early Stopping
callback_dpd = tf.keras.callbacks.EarlyStopping(
    monitor='loss',
    patience=50,
    min_delta=1e-9,
    restore_best_weights=True
)
```

---

# ⚙️ 2. Parâmetros do Sistema OFDM

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

# 🧠 3. Funções de Apoio

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

# 📊 4. Geração de Dados

```python
filePath = "/content/coef"
coef_mzm = np.fromfile(filePath, dtype=np.complex64)

sinal_tx_total = np.zeros(NUM_BLOCOS * K, dtype=complex)

for i in range(NUM_BLOCOS):
    p_tx_linear = 10**(np.random.randint(-5, 16)/10) * 1e-3
    indices = np.random.randint(0, MOD_ORDER, size=len(SUBPORT_ATIVAS))

    qam_norm, _ = modular_qam(indices, MOD_ORDER)
    espectro_mapeado = mapear_ofdm(qam_norm, SUBPORT_ATIVAS, K)
    sinal_tempo = np.fft.ifft(espectro_mapeado) * np.sqrt(K)

    escala = np.sqrt(p_tx_linear / np.mean(np.abs(sinal_tempo)**2))
    sinal_tx_total[i*K : (i+1)*K] = sinal_tempo * escala

sinal_distorcido = modelo_mzm(coef_mzm, sinal_tx_total, J)
sinal_recebido = canal_awgn(
    sinal_distorcido,
    SNR_DB,
    np.mean(np.abs(sinal_tx_total)**2)
)
```

---

# 🤖 5. Arquitetura da Rede Neural (DPD)

```python
X_train = np.c_[sinal_recebido.real, sinal_recebido.imag]
y_train = np.c_[sinal_tx_total.real, sinal_tx_total.imag]

model_dpd = Sequential([
    Dense(2048, activation='relu', input_shape=(2,)),
    Dense(2048, activation='relu'),
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

# 🏁 6. Validação

```python
sinal_entrada_mlp = np.c_[
    sinal_ofdm_teste.real,
    sinal_ofdm_teste.imag
]

sinal_pre_distorcido_raw = model_dpd.predict(
    sinal_entrada_mlp,
    verbose=0
)

sinal_pre_distorcido = (
    sinal_pre_distorcido_raw[:, 0]
    + 1j * sinal_pre_distorcido_raw[:, 1]
)

saida_com_dpd = modelo_mzm(
    coef_mzm,
    sinal_pre_distorcido,
    J
)
```

---

# 🚀 Como usar

1. Abra o Google Colab  
2. Copie os blocos em células separadas  
3. Faça upload do arquivo `coef`  
4. Execute em ordem  

---

> 💡 **Dica Didática:**  
Peça aos alunos para alterar o número de neurônios ou trocar `relu` por `tanh` e observar o impacto na DEP.