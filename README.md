<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/CUDA-GPU%20Accelerated-76B900?style=for-the-badge&logo=nvidia&logoColor=white" alt="CUDA"/>
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/License-MIT-blue?style=for-the-badge" alt="License"/>
</p>

<h1 align="center">
  🔬 D²NN Digital Twin
  <br>
  <sub>Gêmeo Digital para Computação Óptica Generativa</sub>
</h1>

<p align="center">
  <strong>Simulador de alta fidelidade para Redes Neurais Difrativas (D²NN) com física rigorosa de propagação óptica</strong>
</p>

<p align="center">
  <a href="#-demonstração">Demonstração</a> •
  <a href="#-sobre-o-projeto">Sobre</a> •
  <a href="#-features">Features</a> •
  <a href="#-resultados">Resultados</a> •
  <a href="#-instalação">Instalação</a> •
  <a href="#-uso">Uso</a> •
  <a href="#-referências">Referências</a>
</p>

---

## 🎬 Demonstração

<p align="center">
  <a href="https://youtu.be/ydwNh7mtjvU?si=wefclF5w_i941yKA">
    <img src="https://img.shields.io/badge/▶️%20Assistir%20Demo-YouTube-FF0000?style=for-the-badge&logo=youtube&logoColor=white" alt="YouTube Demo"/>
  </a>
</p>

<table>
  <tr>
    <td align="center"><b>🔬 FADER COUNT</b><br><sub>9 Estágios de Propagação</sub></td>
    <td align="center"><b>🎬 MICHAEL JACKSON</b><br><sub>52 Frames Animados</sub></td>
  </tr>
  <tr>
    <td align="center">
      <code>INPUT → L1 → L2 → L3 → L4 → L5 → OUTPUT</code><br>
      <sub>Geração de texto "FADER COUNT" + dígitos 0-9</sub>
    </td>
    <td align="center">
      <code>GIF 52 frames → Silhueta Óptica</code><br>
      <sub>Animação contínua via difração de luz</sub>
    </td>
  </tr>
</table>

---

## 📖 Sobre o Projeto

Este projeto apresenta um **Gêmeo Digital** completo para simulação de **Redes Neurais Difrativas (D²NN)** — uma arquitetura revolucionária que substitui operações matriciais por **propagação física de luz**, alcançando inferências à velocidade da luz (~299.792 km/s) com consumo energético próximo de zero.

### 🎯 Motivação

| Problema | Solução D²NN |
|----------|--------------|
| GPUs consomem ~400W por inferência | Luz propaga sem consumo (passivo) |
| Latência de ms em redes profundas | Inferência em **nanosegundos** |
| Aquecimento por efeito Joule | Fótons não geram calor |
| Fabricação física custa ~USD 40.000+ | **Simulação digital a custo zero** |

### 🧠 Conceito: IA ↔ Óptica

```
┌─────────────────────┬──────────────────────────────────┐
│ Conceito de IA      │ Equivalente Óptico               │
├─────────────────────┼──────────────────────────────────┤
│ Neurônio            │ Pixel do SLM (modulador de luz)  │
│ Pesos               │ Espessura/fase da máscara DOE    │
│ Forward Pass        │ Propagação física (difração)     │
│ Backpropagation     │ Método adjunto (gradiente)       │
│ Inferência (ms)     │ Passagem de luz (~ns)            │
└─────────────────────┴──────────────────────────────────┘
```

---

## ✨ Features

### 📊 Visualização (V8)
- ✅ **Energy Scaling**: `sc = target.mean() / output.mean()`
- ✅ GIFs de alta qualidade com padrões claros
- ✅ Métricas SSIM/PSNR/MSE em tempo real
- ✅ Mapas de transmissão coloridos (viridis)
- ✅ Visualização de fluxo de energia por camada

### 🔬 Física Rigorosa (V18)
- ✅ Parâmetros físicos completos (λ=633nm, n=1.46, 20mW)
- ✅ Medições de potência em **mW** durante treinamento
- ✅ Cálculo de eficiência óptica real
- ✅ Estimativa de SNR (shot noise limited)
- ✅ Exportação DOE 16-bit TIFF para litografia
- ✅ Relatórios de especificação de fabricação

### 🚀 Performance
- ✅ Aceleração GPU via CUDA/TF32
- ✅ Mixed Precision Training (AMP)
- ✅ Scheduler adaptativo (ReduceLROnPlateau)

---

## 📐 Arquitetura do Sistema

```
                    ┌─────────────────────────────────────────────┐
                    │         LASER HeNe (633nm, 20mW)            │
                    └─────────────────┬───────────────────────────┘
                                      │
                                      ▼
    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
    │    L1    │───▶│    L2    │───▶│    L3    │───▶│    L4    │───▶│    L5    │
    │  T≈44%   │    │  T≈45%   │    │  T≈42%   │    │  T≈30%   │    │  T≈10%   │
    │   4cm    │    │   4cm    │    │   4cm    │    │   4cm    │    │   20cm   │
    └──────────┘    └──────────┘    └──────────┘    └──────────┘    └──────────┘
                                                                         │
                                                                         ▼
                                                                   ┌──────────┐
                                                                   │  SENSOR  │
                                                                   │  256×256 │
                                                                   └──────────┘
```

**Especificações Técnicas:**
- **Grid**: 256×256 pixels
- **Pixel Size**: 20µm
- **Wavelength**: 633nm (HeNe vermelho)
- **Substrate**: SiO₂ (n=1.46)
- **Layers**: 5 DOEs (Diffractive Optical Elements)

---

## 📊 Resultados

### Métricas de Qualidade

| Experimento | SSIM ↑ | PSNR (dB) ↑ | MSE ↓ | Tempo | Frames |
|-------------|--------|-------------|-------|-------|--------|
| **Fader Count** | 0.7027 | 22.28 | 0.0062 | 153s | 11 |
| **Michael Jackson** | 0.6053 | 22.54 | 0.0058 | 768s | 52 |

### Convergência do Treinamento

```
Loss: 0.5584 → 0.0494 (Fader Count)  │  Redução: 91.1%
Loss: 0.0209 → 0.0054 (MJ)          │  Redução: 74.2%
```

### Análise de Energia (Fader Count)

```
📊 FLUXO DE ENERGIA ATRAVÉS DAS CAMADAS:

 Input:    [████████████████████] 100.00%
 Após L1:  [█████░░░░░░░░░░░░░░░]  27.98%
 Após L2:  [█░░░░░░░░░░░░░░░░░░░]   9.11%
 Após L3:  [░░░░░░░░░░░░░░░░░░░░]   2.06%
 Após L4:  [░░░░░░░░░░░░░░░░░░░░]   0.15%
 Após L5:  [░░░░░░░░░░░░░░░░░░░░]   0.00%

⚡ Transmissão Total: 0.25% (energia redistribuída para formar padrão)
```

### Inovação: Inversão de Target

| Problema Original | Solução Implementada |
|-------------------|---------------------|
| MJ claro (93%) sobre fundo escuro | Target invertido (7% energia) |
| Bloquear 93% da luz = inviável | Iluminar fundo = eficiente |
| **Demanda energética: 93.9%** | **Demanda energética: 7.5%** |

---

## 🛠️ Instalação

### Requisitos
- Python 3.10+
- CUDA-capable GPU (recomendado)
- 4GB+ VRAM

### Setup

```bash
# Clone o repositório
git clone https://github.com/seu-usuario/d2nn-digital-twin.git
cd d2nn-digital-twin

# Crie ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou: venv\Scripts\activate  # Windows

# Instale dependências
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy matplotlib imageio scikit-image pillow
```

---

## 🚀 Uso

### Google Colab (Recomendado)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Diyanderson/d2nn-digital-twin/blob/main/D2NN_DigitalTwin_OpticalComputing.ipynb)

1. Clique no badge acima para abrir diretamente no Colab
2. Selecione **Runtime → Change runtime type → GPU**
3. Execute todas as células sequencialmente

### Local

```bash
# Execute o notebook
jupyter notebook D2NN_DigitalTwin_OpticalComputing.ipynb
```

### Ajuste de Hiperparâmetros

```python
# Balanço Fase × Amplitude (init_bias)
# -3.0 → Amplitude extrema (absorção máxima)
#  0.0 → Híbrido equilibrado (50/50)
# +3.0 → Fase extrema (transmissão máxima)

PARAMS.init_bias = 1.0  # Padrão: DOE de fase
```

---

## 📁 Estrutura de Arquivos Gerada

```
optical_fader/
├── input.gif                    # Input animado (11 frames)
├── output.gif                   # Output (inferno colormap)
├── target.gif                   # Target animado
├── L1.gif ... L5.gif            # Fases por camada (HSV)
├── metrics.png                  # Gráficos SSIM/PSNR/MSE
├── transmission_maps.png        # Mapas de transmissão
├── final_L{1-5}_phase_16bit.tiff    # DOEs para litografia
├── final_L{1-5}_amp_8bit.png        # Amplitude visualizada
├── final_fabrication_specs.txt      # Specs de fabricação
└── model_fader_v19.pth              # Modelo PyTorch

optical_mj/
└── (mesma estrutura, 52 frames)
```

---

## 📚 Referências

1. **Chen, S. et al.** "Optical generative models." *Nature* 644, 903–910 (2025).  
   DOI: [10.1038/s41586-025-08519-4](https://www.nature.com/articles/s41586-025-08519-4)

2. **Lin, X. et al.** "All-optical machine learning using diffractive deep neural networks." *Science* 361, 1004–1008 (2018).  
   DOI: [10.1126/science.aat8084](https://www.science.org/doi/10.1126/science.aat8084)

3. **Goodman, J.W.** *Introduction to Fourier Optics.* 3rd ed. Roberts and Company Publishers, 2005.

---

## 👤 Autor

<table>
  <tr>
    <td align="center">
      <a href="https://github.com/Diyanderson">
        <img src="https://github.com/Diyanderson.png" width="100px;" alt="Foto do Autor"/><br>
        <sub><b>Anderson Alves dos Santos</b></sub>
      </a>
      <br>
      <sub>Inteligência Artificial & Machine Learning</sub>
      <br>
      <sub>Centro Universitário Leonardo da Vinci (Uniasselvi)</sub>
    </td>
  </tr>
</table>

### 📫 Contato

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://br.linkedin.com/in/anderson-alves-dos-santos-78048388)
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Diyanderson)
[![Email](https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:diyanderson@gmail.com)

---

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.

---

<p align="center">
  <sub>⚡ Desenvolvido com PyTorch + CUDA | 🔬 Simulando os computadores de luz do futuro</sub>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Made%20with-❤️-red?style=flat-square" alt="Made with Love"/>
  <img src="https://img.shields.io/badge/Powered%20by-Light-yellow?style=flat-square" alt="Powered by Light"/>
</p>
