# Video Transcriber

Herramienta para transcribir y traducir videos usando Groq API con Whisper y modelos de traducción.

## Características

- Transcripción de audio usando Whisper
- Traducción automática al español
- Procesamiento por lotes de múltiples videos
- Modelos configurables
- Guardado opcional de resultados

## Instalación

```bash
pip install moviepy groq langchain-groq python-dotenv
```

## Configuración

Obtén tu API key gratuita de Groq: https://console.groq.com/
Crea un archivo .env:

```bash
GROQ_API_KEY=tu_api_key_aqui
```

## Uso

```bash
# Solo visualización
python video_transcriber.py ./directorio_videos

# Guardar resultados
python video_transcriber.py ./directorio_videos ./directorio_resultados

# Modelos personalizados
python video_transcriber.py ./directorio_videos ./directorio_resultados whisper-large-v3-turbo openai/gpt-oss-20b
```bash

## Formatos soportados
- MP4, AVI, MOV, MKV, WEBM, M4V

