# run.ps1 — Iniciar la aplicación Streamlit
# Ejecutar con: powershell -ExecutionPolicy Bypass -File run.ps1

$venvActivate = "venv\Scripts\Activate.ps1"
if (Test-Path $venvActivate) {
    & $venvActivate
} else {
    Write-Host "Entorno virtual no encontrado. Ejecuta setup.ps1 primero." -ForegroundColor Red
    exit 1
}

Write-Host "Iniciando Sistema de Análisis Estadístico de Imágenes..." -ForegroundColor Cyan
Write-Host "Abre tu navegador en: http://localhost:8501" -ForegroundColor Green
streamlit run Home.py
