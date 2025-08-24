# Video Transcriber

Herramienta para transcribir y traducir videos usando Groq API con Whisper y modelos de traducción.

## Características

- Transcripción de audio usando Whisper
- Traducción automática de la transcripción al español
- Procesamiento por lotes de múltiples videos
- Modelos y prompts configurables
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
```

## Formatos soportados
- MP4, AVI, MOV, MKV, WEBM, M4V

## Modelos
Por defecto el script utiliza *whisper-large-v3* como modelo de transcripción.
Y *openai/gpt-oss-120b* como modelo de traducción

Puedes cambiar los modelos al utilizar el script por consola, como se indica más arriba
O también al utilizar la función en python, mediante la siguiente función

```python
transcriber = VideoTranscriber()
transcriber.set_transcription_model("whisper-large-v3-turbo")
transcriber.set_translation_model("openai/gpt-oss-20b")
```

## Prompt
El prompt por defecto tiene este formato

> Traduce el siguiente texto del inglés al español. 
> Mantén el formato original y asegúrate de que la traducción sea natural y fluida.
> Devuelve únicamente el contenido de la traducción
>
> Texto a traducir:
> {text}
>
> Traducción al español:
>

Puedes modificar este prompt, por ejemplo, del siguiente modo

```python
transcriber = VideoTranscriber()
new_translation_prompt = """
Traducir a un español neutro con atención a los términos técnicos del campo disciplinar

Texto a traducir:
{text}

Traducción:
"""

transcriber.set_translation_prompt(new_translation_prompt)
```
