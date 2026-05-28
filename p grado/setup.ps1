# setup.ps1 — Instalación del entorno del proyecto
# Ejecutar con: powershell -ExecutionPolicy Bypass -File setup.ps1

Write-Host "============================================" -ForegroundColor Cyan
Write-Host " Sistema de Análisis Estadístico de Imágenes" -ForegroundColor Cyan
Write-Host " Instalación automática del entorno" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# 1. Verificar / instalar Python 3.11
$pythonPath = (Get-Command python -ErrorAction SilentlyContinue)?.Source
if (-not $pythonPath) {
    Write-Host "Instalando Python 3.11 via winget..." -ForegroundColor Yellow
    winget install Python.Python.3.11 --silent --accept-package-agreements --accept-source-agreements
    # Reload PATH
    $env:PATH = [System.Environment]::GetEnvironmentVariable("PATH", "Machine") + ";" +
                [System.Environment]::GetEnvironmentVariable("PATH", "User")
} else {
    Write-Host "Python encontrado: $pythonPath" -ForegroundColor Green
}

# 2. Crear entorno virtual
if (-not (Test-Path "venv")) {
    Write-Host "Creando entorno virtual..." -ForegroundColor Yellow
    python -m venv venv
}
Write-Host "Activando entorno virtual..." -ForegroundColor Yellow
& "venv\Scripts\Activate.ps1"

# 3. Actualizar pip
Write-Host "Actualizando pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip --quiet

# 4. Instalar PyTorch CPU (más pequeño y compatible con AMD)
Write-Host "Instalando PyTorch CPU..." -ForegroundColor Yellow
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --quiet

# 5. Instalar resto de dependencias
Write-Host "Instalando dependencias del proyecto..." -ForegroundColor Yellow
pip install -r requirements.txt --quiet

Write-Host ""
Write-Host "============================================" -ForegroundColor Green
Write-Host " Instalación completada exitosamente" -ForegroundColor Green
Write-Host " Ejecuta: .\run.ps1  para iniciar la app" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green
