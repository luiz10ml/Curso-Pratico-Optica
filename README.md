# Curso-Pratico-Optica
![Inatel](https://img.shields.io/badge/Instituição-Inatel-blue)
![Nível](https://img.shields.io/badge/Nível-Graduação-success)
![Área](https://img.shields.io/badge/Área-Telecomunicações-informational)


📡 Curso Prático: Predistorção Digital (DPD) com Redes NeuraisEste repositório contém o material didático para a implementação de uma Predistorção Digital (DPD) utilizando Redes Neurais do tipo MLP (Multi-Layer Perceptron) para linearizar um Modulador Mach-Zehnder (MZM) em sistemas Radio-over-Fiber (RoF).📖 1. Introdução e SetupPrimeiro, precisamos preparar nosso ambiente no Google Colab instalando a biblioteca necessária para modulação e importando as ferramentas de álgebra e Deep Learning.Python# Instalação da biblioteca de modulação
!pip install ModulationPy

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from ModulationPy import QAMModem
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from scipy.signal import welch

# Configuração de parada antecipada (Early Stopping) para otimizar o tempo de treino
callback_dpd = tf.keras.callbacks.EarlyStopping(
    monitor='loss', patience=50, min_delta=1e-9, restore_best_weights=True
)
⚙️ 2. Parâmetros do Sistema OFDMAqui definimos as características do sinal que será transmitido. O sinal OFDM é a base das comunicações 4G/5G.ParâmetroValorDescriçãoK2048Tamanho da FFT (Número de subportadoras)NUM_BLOCOS10Quantidade de blocos para o datasetMOD_ORDER16Modulação 16-QAMSNR_DB45Relação Sinal-Ruído (dB)J2Ordem da não-linearidade do MZMPythonK = 2048                
NUM_BLOCOS = 10         
SUBPORT_ATIVAS = np.arange(-200, 201, 1)  
MOD_ORDER = 16          
SNR_DB = 45             
J = 2                   
🧠 3. Funções de Apoio (O Coração do Sistema)Para que o código seja modular, criamos funções que simulam cada etapa da cadeia de comunicação.Modular QAM: Transforma bits em símbolos complexos.Modelo MZM: Simula a distorção física do componente óptico.Suavizar Espectro: Limpa o gráfico da Densidade Espectral de Potência (DEP) para melhor visualização.Pythondef suavizar_espectro(vetor_db, janela=41):
    """Aplica uma média móvel para suavizar o ruído no gráfico da DEP."""
    return np.convolve(vetor_db, np.ones(janela)/janela, mode='same')

def modelo_mzm(coeficientes, sinal_in, ordem):
    """Matriz de potências ímpares para simular distorção de amplitude e fase."""
    X = np.column_stack([sinal_in * np.abs(sinal_in)**k for k in range(ordem)])
    return X.dot(coeficientes)

def calcular_evm(simbolos_est, simbolos_ref):
    """Calcula o Error Vector Magnitude (EVM)."""
    erro = simbolos_est - simbolos_ref
    return np.sqrt(np.mean(np.abs(erro)**2) / np.mean(np.abs(simbolos_ref)**2)) * 100
📊 4. Geração de Dados e Canal Não-LinearNesta etapa, carregamos os coeficientes reais do dispositivo e geramos o sinal OFDM. Note que o sinal passará pelo modelo_mzm, o que causará o espalhamento espectral e a deformação da constelação.Python# Substitua o caminho pelo local onde seu arquivo 'coef' está no Colab
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

# Simulação da distorção física
sinal_distorcido = modelo_mzm(coef_mzm, sinal_tx_total, J)
sinal_recebido = canal_awgn(sinal_distorcido, SNR_DB, np.mean(np.abs(sinal_tx_total)**2))
🤖 5. Arquitetura da Rede Neural (A Solução DPD)A Rede Neural atua como o predistorçor. Ela aprende a função inversa do MZM. Se o MZM comprime o sinal, a rede aprende a expandi-lo preventivamente.Python# Preparação dos dados: Convertendo números complexos em colunas Real e Imaginária
X_train = np.c_[sinal_recebido.real, sinal_recebido.imag]
y_train = np.c_[sinal_tx_total.real, sinal_tx_total.imag]

# Definição da MLP (Multi-Layer Perceptron)
model_dpd = Sequential([
    Dense(2048, activation='relu', input_shape=(2,)),
    Dense(2048, activation='relu'),
    Dense(2) 
])

model_dpd.compile(optimizer='adam', loss='mse')
model_dpd.fit(X_train, y_train, epochs=200, batch_size=K, verbose=2, callbacks=[callback_dpd])
🏁 6. Validação e Comparação de ResultadosPor fim, comparamos o sinal que não recebeu tratamento com o sinal que passou pela nossa Rede Neural.Python# Aplicação da DPD treinada
sinal_entrada_mlp = np.c_[sinal_ofdm_teste.real, sinal_ofdm_teste.imag]
sinal_pre_distorcido_raw = model_dpd.predict(sinal_entrada_mlp, verbose=0)
sinal_pre_distorcido = sinal_pre_distorcido_raw[:,0] + 1j*sinal_pre_distorcido_raw[:,1]
saida_com_dpd = modelo_mzm(coef_mzm, sinal_pre_distorcido, J)
🚀 Como usar este repositórioAbra o Google Colab.Copie os blocos de código deste README em células separadas.Faça o upload do arquivo de coeficientes (coef) para o ambiente do Colab.Execute as células em ordem e observe a limpeza do espectro de rádio![!TIP]Dica Didática: Peça para os alunos alterarem o número de neurônios na camada Dense ou trocarem a função de ativação de relu para tanh e observarem o impacto no gráfico de DEP!