<!-- README: Configuração de Credenciais da HistoLens -->

# Setup de Credenciais - HistoLens SmartScope AI

## 1. Google Gemini API (Maestro Router)

O **Maestro Router** usa o modelo `gemini-2.0-flash-exp` do Google para classificar perguntas médicas.

### Opções de Configuração:

#### Opção A: Variável de Ambiente (RECOMENDADA)
```powershell
# No PowerShell (sessão atual)
$env:GOOGLE_API_KEY="sua_chave_aqui"

# Permanente no Windows (roda uma única vez)
setx GOOGLE_API_KEY "sua_chave_aqui"

# Depois reabra o terminal
```

#### Opção B: Passar como Parâmetro (Desenvolvimento)
```python
from maestro import MaestroRouter

router = MaestroRouter(api_key="sua_chave_aqui")
```

#### Opção C: Arquivo `.env` (com `python-dotenv`)
```bash
pip install python-dotenv
```

Crie arquivo `.env` na raiz do projeto:
```
GOOGLE_API_KEY=sua_chave_aqui
```

No código:
```python
from dotenv import load_dotenv
load_dotenv()
# Depois use MaestroRouter() normalmente
```

### Como Obter a Chave:

1. Acesse: https://aistudio.google.com/apikey
2. Clique em "Get API key"
3. Crie um novo projeto ou escolha um existente
4. Copie a chave e use uma das opções acima

---

## 2. Google Cloud Speech-to-Text (STT) + Text-to-Speech (TTS)

Já configurado com `GOOGLE_APPLICATION_CREDENTIALS` apontando para o arquivo JSON da Service Account.

**Arquivo**: `C:\Users\erick\Downloads\gen-lang-client-0140605921-a4e0dcd0ba63.json`

Defina antes de rodar o microscopyo:
```powershell
$env:GOOGLE_APPLICATION_CREDENTIALS="C:\Users\erick\Downloads\gen-lang-client-0140605921-a4e0dcd0ba63.json"
```

---

## 3. Hugging Face + PyTorch (MedGemma) - CRÍTICO: USE CONDA

Este é o maior gargalo do projeto. **NÃO USE PIP para instalar PyTorch** — a resolução de dependências falha completamente. Use **Conda** (Miniconda ou Anaconda).

### Por que Conda?

PyTorch, CUDA, cuDNN e bibliotecas C++ subjacentes têm dependências binárias (wheels) complexas. O pip não consegue resolver essa árvore corretamente em ambientes Windows/Linux. Conda tem solvedores mais robustos para essas dependências críticas.

## VS Code não reconhece `conda` no terminal (Windows)

No PowerShell fora do VS Code (uma vez só):
```powershell
conda init powershell
```

Depois:
1. Feche **todo** o VS Code
2. Abra o VS Code novamente
3. No terminal integrado, rode:

```powershell
conda activate histolens
```

Se ainda falhar, selecione o interpretador do conda em:
**Python: Select Interpreter** → ambiente `histolens`.

---

## Setup Manual (Passo-a-Passo):

### 1. Crie e Ative Ambiente Conda
```powershell
# Download de https://docs.conda.io/projects/miniconda/en/latest/
# Ou via Chocolatey:
choco install miniconda3
```

#### 2. Crie ambiente NOVO (separado do atual .venv)
```powershell
# Cria ambiente isolado para HistoLens
conda create -n histolens python=3.10

# Ativa o ambiente
conda activate histolens
```

#### 2. Instale PyTorch moderno (CRÍTICO!)
```powershell
# Para GPU NVIDIA (CUDA 12.1):
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Para CPU only (laptop/sem GPU):
conda install pytorch torchvision torchaudio cpuonly -c pytorch

# Para GPU AMD (ROCm):
conda install pytorch torchvision torchaudio rocm/label/dev -c pytorch

# Para Mac:
conda install pytorch torchvision torchaudio -c pytorch
```

#### 3. Instale as dependências Python no ambiente conda
```powershell
# No ambiente conda ativo:
pip install google-cloud-speech google-cloud-texttospeech google-generativeai transformers pillow opencv-python openslide-python openslide-bin sounddevice numpy python-dotenv requests
```

#### 4. Configure Hugging Face
```powershell
# Defina seu token de acesso (você já tem!)
$env:HF_TOKEN="seu_token_do_hugging_face"

# Ou permanente:
setx HF_TOKEN "seu_token"
```

#### 5. Teste a instalação
```powershell
# Dentro do ambiente conda
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "from transformers import AutoModelForCausalLM; print('✅ Transformers OK')"
```

### Estrutura de Ambientes:

```
.venv/                    # Ambiente original (pip) - mantém STT/TTS/Gemini
histolens (conda)/        # Novo ambiente - PyTorch + MedGemma
```

Você pode ter **ambos** rodando em paralelo! A .venv continua com o Google Cloud, o conda é só para MedGemma.

### Rodar HistoLens com Conda:

```powershell
# Ativa ambiente conda
conda activate histolens

# Configure APIs
$env:GOOGLE_API_KEY="sua_chave_gemini"
$env:GOOGLE_APPLICATION_CREDENTIALS="C:\Users\erick\Downloads\gen-lang-client-0140605921-a4e0dcd0ba63.json"
$env:HF_TOKEN="seu_token_hugging_face"

# Rode
cd C:\Users\erick\Downloads\HistoLens
python microscopyo.py
```

### Resolução de Problemas:

**"CUDA out of memory"**
- Modelo 27B precisa ~24GB VRAM (RTX 4090, A100)
- Com menos VRAM: use `load_in_8bit=True` (já configurado no código)
- Última resort: use CPU (muito lento)

**"No module named 'torch'"**
- Certificar que está no ambiente conda correto: `conda info --envs`
- Verificar: `python -c "import torch"`

**"CUDA capability sm_35"**
- GPU muito antiga, não suporta PyTorch moderno
- Use CPU ou upgrade GPU

### Repositórios Reais dos Modelos (quando forem publicados):

Ainda não temos os repos públicos exatos, mas a estrutura de código em [medgemma.py](medgemma.py) está pronta para:
- `google/medgemma-4b` (visão)
- `google/medgemma-27b` (teoria)

Quando oficializarem, é só trocar as strings `model_id`.

---

## Roteiro de Integração:

```
[✅] STT + TTS (Google Cloud - pronto)
[✅] Maestro Router (Gemini 2.0 - pronto)
[✅] MedGemma 4B (Visão - código estruturado)
[✅] MedGemma 27B (Teoria - código estruturado)
[❌] Orquestração Completa
[❌] Modo Produção
```

---

## Próximos Passos:

1. **Ativar Conda no terminal do VS Code** (uma vez):
   ```powershell
   conda init powershell
   # Fechar e reabrir o VS Code
   ```

2. **Instalar/validar stack no ambiente `histolens`**:
   ```powershell
   conda activate histolens
   conda install pytorch -c pytorch  # Ou com CUDA/ROCm conforme sua GPU
   pip install google-cloud-speech google-cloud-texttospeech google-generativeai transformers pillow opencv-python openslide-python openslide-bin sounddevice numpy python-dotenv requests
   ```

3. **Teste MedGemma**:
   ```powershell
   conda activate histolens
   python medgemma.py
   ```

4. **Integração no Microscopyo** (próxima fase):
   - Conectar Maestro → MedGemma 4B/27B
   - Adicionar screenshots automáticos
   - Feedback de áudio "processando..."

---

**Status**: Toda infraestrutura de código está pronta. Próximo foco: testes de hardware e integração completa. 🚀
