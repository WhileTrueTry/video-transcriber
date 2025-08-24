


import os
import sys
import random
import time
from pathlib import Path
from moviepy import VideoFileClip
from groq import Groq
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import tempfile
import logging



# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)



# Cargar variables de entorno
load_dotenv()




class VideoTranscriber:
    def __init__(self, whisper_model='whisper-large-v3', translation_model="openai/gpt-oss-120b", groq_api_key=None):
        """
        Inicializa el transcriptor de videos
        
        Args:
            whisper_model (str): Modelo de Whisper para transcripción (por defecto: 'whisper-large-v3')
            translation_model (str): Modelo para traducción (por defecto: "openai/gpt-oss-120b")
            groq_api_key (str): API key de Groq. Si no se proporciona, se buscará en variables de entorno
        """
        self.groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY no encontrada. Configúrala como variable de entorno o pásala como parámetro")
        
        # Configurar modelos
        self.whisper_model = whisper_model
        self.translation_model = translation_model
        
        # Prompt por defecto para traducción
        self.translation_prompt = """
        Traduce el siguiente texto del inglés al español. 
        Eventualmente puede haber texto en árabe también. Asegurate de traducirlo correctamente
        Mantén el formato original y asegúrate de que la traducción sea natural y fluida.
        Devuelve únicamente el contenido de la traducción
        
        Texto a traducir:
        {text}
        
        Traducción al español:
        """
        
        # Inicializar clientes de Groq
        self._initialize_clients()
    
    
    
    def _initialize_clients(self):
        """Inicializa los clientes de Groq"""
        self.whisper_client = Groq(api_key=self.groq_api_key)
        
        self.translator_client = ChatGroq(
            api_key=self.groq_api_key,
            model_name=self.translation_model
        )
    
    
    
    def set_transcription_model(self, whisper_model):
        """
        Cambia el modelo de transcripción
        
        Args:
            whisper_model (str): Nuevo modelo de Whisper a utilizar
        """
        self.whisper_model = whisper_model
        logger.info(f"Modelo de transcripción cambiado a: {whisper_model}")
    
    
    
    def set_translation_model(self, translation_model):
        """
        Cambia el modelo de traducción y reinstancia el cliente
        
        Args:
            translation_model (str): Nuevo modelo de traducción a utilizar
        """
        self.translation_model = translation_model
        # Reinstanciar el cliente de traducción con el nuevo modelo
        self.translator_client = ChatGroq(
            api_key=self.groq_api_key,
            model_name=self.translation_model
        )
        logger.info(f"Modelo de traducción cambiado a: {translation_model}")
    
    
    
    def set_translation_prompt(self, prompt):
        """
        Cambia el prompt de traducción
        
        Args:
            prompt (str): Nuevo prompt para traducción. Debe incluir {text} donde se insertará el texto a traducir
        """
        if "{text}" not in prompt:
            raise ValueError("El prompt debe incluir {text} donde se insertará el texto a traducir")
        
        self.translation_prompt = prompt
        logger.info("Prompt de traducción actualizado")
    
    
    
    def extract_audio(self, video_path):
        """
        Extrae el audio de un video y lo guarda como archivo temporal
        
        Args:
            video_path (str): Ruta al archivo de video
            
        Returns:
            str: Ruta al archivo de audio temporal
        """
        try:
            logger.info(f"Extrayendo audio de: {video_path}")
            video = VideoFileClip(video_path)
            
            # Crear archivo temporal para el audio
            temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            audio_path = temp_audio.name
            temp_audio.close()
            
            # Extraer audio
            video.audio.write_audiofile(audio_path, logger=None)
            video.close()
            
            logger.info(f"Audio extraído exitosamente: {audio_path}")
            return audio_path
            
        except Exception as e:
            logger.error(f"Error extrayendo audio de {video_path}: {str(e)}")
            raise
    
    
    
    def transcribe_audio(self, audio_path, intento=0):
        """
        Transcribe audio usando Whisper a través de Groq API
        
        Args:
            audio_path (str): Ruta al archivo de audio
            intento (int): Número de intento actual
            
        Returns:
            str: Texto transcrito
        """
        
        if intento > 10: 
            logger.error("FALLO TRANSCRIBIENDO EN CANTIDAD DE INTENTOS")
            return 'FALLO TRANSCRIBIENDO EN CANTIDAD DE INTENTOS'
 
        try:
            logger.info(f"Transcribiendo audio: {audio_path}")
            
            with open(audio_path, 'rb') as audio_file:
                transcription = self.whisper_client.audio.transcriptions.create(
                    file=(audio_path, audio_file.read()),
                    model=self.whisper_model,
                    prompt="",
                    response_format="text"    
                )
            
            text = transcription.content if hasattr(transcription, 'content') else str(transcription)
            logger.info("Transcripción completada exitosamente")
            return text
            
        except Exception as e:
            logger.info(f"Intento {intento + 1}")
            logger.error(f"Error transcribiendo audio {audio_path}: {str(e)}")
            retry_after_time = self._get_retry_time(e)
            time.sleep(retry_after_time)
            return self.transcribe_audio(audio_path, intento + 1)
    
    
    
    def translate_text(self, text, intento=0):
        """
        Traduce texto usando el modelo y prompt configurados
        
        Args:
            text (str): Texto a traducir
            intento (int): Número de intento actual
            
        Returns:
            str: Texto traducido
        """
        
        if intento > 10: 
            logger.error("FALLO TRADUCIENDO EN CANTIDAD DE INTENTOS")
            return 'FALLO TRADUCIENDO EN CANTIDAD DE INTENTOS'
        
        try:
            logger.info("Traduciendo texto...")
            
            # Usar el prompt personalizable
            prompt = self.translation_prompt.format(text=text)
            
            response = self.translator_client.invoke(prompt)
            translation = response.content if hasattr(response, 'content') else str(response)
            
            logger.info("Traducción completada exitosamente")
            return translation
            
        except Exception as e:
            logger.info(f"Intento {intento + 1}")
            logger.error(f"Error traduciendo texto: {str(e)}")
            retry_after_time = self._get_retry_time(e)
            time.sleep(retry_after_time)
            return self.translate_text(text, intento + 1)
    
    
    
    def save_text(self, text, file_path):
        """
        Guarda texto en un archivo
        
        Args:
            text (str): Texto a guardar
            file_path (str): Ruta donde guardar el archivo
        """
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(text)
            logger.info(f"Archivo guardado: {file_path}")
        except Exception as e:
            logger.error(f"Error guardando archivo {file_path}: {str(e)}")



    def process_video(self, video_path, results_path=None, save=True, return_results=False):
        """
        Procesa un video completo: extrae audio, transcribe y traduce
        
        Args:
            video_path (str): Ruta al archivo de video
            results_path (str): Directorio donde guardar los resultados (opcional)
            save (bool): Si True, guarda los archivos de transcripción y traducción
            return_results (bool): Si True, retorna los resultados como diccionario
            
        Returns:
            dict or None: Diccionario con transcripción y traducción si return_results=True
                {
                    'transcription': str,
                    'translation': str,
                    'video_path': str
                }
        """
        # Verificar que el archivo existe
        if not os.path.exists(video_path):
            logger.error(f"Archivo no encontrado: {video_path}")
            raise FileNotFoundError(f"Archivo no encontrado: {video_path}")
        
        # Si se va a guardar, verificar que el directorio de resultados existe
        if save and results_path:
            os.makedirs(results_path, exist_ok=True)
        
        try:
            logger.info(f"Procesando video: {video_path}")
            
            # Extraer audio
            audio_path = self.extract_audio(video_path)
            
            try:
                # Transcribir
                transcription = self.transcribe_audio(audio_path)
                
                # Verificar si hubo error en la transcripción
                if "FALLO TRANSCRIBIENDO" in transcription:
                    logger.error("Error en transcripción por límite de intentos")
                    if return_results:
                        return {
                            'transcription': transcription,
                            'translation': None,
                            'video_path': video_path
                        }
                    return None
                
                # Traducir
                translation = self.translate_text(transcription)
                
                # Obtener nombre del archivo sin extensión
                video_filename = Path(video_path).stem
                
                # Guardar archivos si se solicita
                if save and results_path:
                    transcription_file = os.path.join(results_path, f"{video_filename}_transcription.txt")
                    translation_file = os.path.join(results_path, f"{video_filename}_translation.txt")
                    
                    self.save_text(transcription, transcription_file)
                    self.save_text(translation, translation_file)
                
                logger.info(f"Video procesado exitosamente: {video_path}")
                
                # Retornar resultados si se solicita
                if return_results:
                    return {
                        'transcription': transcription,
                        'translation': translation,
                        'video_path': video_path
                    }
                
            finally:
                # Limpiar archivo temporal de audio
                if os.path.exists(audio_path):
                    os.unlink(audio_path)


                    
        except Exception as e:
            logger.error(f"Error procesando video {video_path}: {str(e)}")
            raise


    
    def _get_retry_time(self, e):
        """
        Calcula el tiempo de espera para reintentar después de un error
        
        Args:
            e (Exception): Excepción que causó el error
            
        Returns:
            float: Tiempo de espera en segundos
        """
        retry_after = 60  # Tiempo base por defecto
        
        if hasattr(e, 'response') and hasattr(e.response, 'headers') and 'retry-after' in e.response.headers:
            try:
                retry_after = int(e.response.headers['retry-after']) + 2
                logger.info(f"API sugiere esperar {retry_after} segundos.")
            except (ValueError, TypeError):
                logger.info("No se pudo convertir 'retry-after' a un número entero.")
                retry_after = 200

        jitter = random.uniform(0, 5)  # Pequeño jitter para evitar colisiones
        wait_time = retry_after + jitter
        logger.info(f"Reintentando en {wait_time:.2f} segundos...")
        return wait_time






# Ejemplo de uso
if __name__ == "__main__":
    def print_usage():
        """Imprime las instrucciones de uso"""
        print("\n" + "="*80)
        print("USO DEL SCRIPT:")
        print("="*80)
        print("Opción 1 - Solo directorio de videos (modelos por defecto, solo mostrar):")
        print("  python script.py <directorio_videos>")
        print("\nOpción 2 - Directorio de videos + directorio de resultados:")
        print("  python script.py <directorio_videos> <directorio_resultados>")
        print("\nOpción 3 - Directorio + modelos personalizados (solo mostrar):")
        print("  python script.py <directorio_videos> <modelo_transcripcion> <modelo_traduccion>")
        print("\nOpción 4 - Completa (directorio + resultados + modelos):")
        print("  python script.py <directorio_videos> <directorio_resultados> <modelo_transcripcion> <modelo_traduccion>")
        print("\nEjemplos:")
        print("  python script.py ./videos")
        print("  python script.py ./videos ./resultados")
        print("  python script.py ./videos whisper-large-v3-turbo llama-3.1-70b-versatile")
        print("  python script.py ./videos ./resultados whisper-large-v3-turbo llama-3.1-70b-versatile")
        print("\nModelos disponibles de transcripción:")
        print("  - whisper-large-v3 (por defecto)")
        print("  - whisper-large-v3-turbo")
        print("\nModelos disponibles de traducción:")
        print("  - openai/gpt-oss-120b (por defecto)")
        print("  - openai/gpt-oss-20b")
        print("  - llama-3.1-70b-versatile")
        print("="*80)
        print("Podés encontrar todos los modelos disponibles en la documentación oficial de Groq:")
        print("https://console.groq.com/docs/models")
        print("="*80)
        print("\nNOTA:")
        print("  - Si no especificas directorio de resultados, solo se mostrarán las transcripciones/traducciones")
        print("  - Si especificas directorio de resultados, se crearán archivos .txt automáticamente")
        print("="*80)
    
    try:
        # Verificar número de argumentos
        if len(sys.argv) < 2:
            print("❌ ERROR: Debes proporcionar al menos el directorio de videos")
            print_usage()
            sys.exit(1)
        
        if len(sys.argv) > 5:
            print("❌ ERROR: Demasiados argumentos")
            print_usage()
            sys.exit(1)
        
        # Variables por defecto
        video_directory = None
        results_directory = None
        whisper_model = 'whisper-large-v3'
        translation_model = "openai/gpt-oss-120b"
        save_files = False
        
        # Procesar argumentos según el número
        if len(sys.argv) == 2:
            # Solo directorio de videos
            video_directory = sys.argv[1]
            print(f"📁 Directorio videos: {video_directory}")
            print(f"🎤 Modelo transcripción: {whisper_model} (por defecto)")
            print(f"🌐 Modelo traducción: {translation_model} (por defecto)")
            print("📋 Modo: Solo mostrar resultados")
            
        elif len(sys.argv) == 3:
            # Dos opciones posibles: videos + resultados O videos + modelo_transcripción
            # Verificamos si el segundo argumento es un directorio o un modelo
            if os.path.isdir(sys.argv[2]) or not os.path.exists(sys.argv[2]):
                # Es un directorio de resultados (existente o a crear)
                video_directory = sys.argv[1]
                results_directory = sys.argv[2]
                save_files = True
                print(f"📁 Directorio videos: {video_directory}")
                print(f"💾 Directorio resultados: {results_directory}")
                print(f"🎤 Modelo transcripción: {whisper_model} (por defecto)")
                print(f"🌐 Modelo traducción: {translation_model} (por defecto)")
                print("📋 Modo: Guardar archivos")
            else:
                print("❌ ERROR: Formato incorrecto para 2 argumentos")
                print("💡 Para 2 argumentos usa: <directorio_videos> <directorio_resultados>")
                print_usage()
                sys.exit(1)
                
        elif len(sys.argv) == 4:
            # videos + modelo_transcripción + modelo_traducción (solo mostrar)
            video_directory = sys.argv[1]
            whisper_model = sys.argv[2]
            translation_model = sys.argv[3]
            print(f"📁 Directorio videos: {video_directory}")
            print(f"🎤 Modelo transcripción: {whisper_model}")
            print(f"🌐 Modelo traducción: {translation_model}")
            print("📋 Modo: Solo mostrar resultados")
            
        elif len(sys.argv) == 5:
            # videos + resultados + modelo_transcripción + modelo_traducción
            video_directory = sys.argv[1]
            results_directory = sys.argv[2]
            whisper_model = sys.argv[3]
            translation_model = sys.argv[4]
            save_files = True
            print(f"📁 Directorio videos: {video_directory}")
            print(f"💾 Directorio resultados: {results_directory}")
            print(f"🎤 Modelo transcripción: {whisper_model}")
            print(f"🌐 Modelo traducción: {translation_model}")
            print("📋 Modo: Guardar archivos")
        
        # Verificar que el directorio de videos existe
        if not os.path.exists(video_directory):
            print(f"❌ ERROR: El directorio de videos '{video_directory}' no existe")
            sys.exit(1)
        
        if not os.path.isdir(video_directory):
            print(f"❌ ERROR: '{video_directory}' no es un directorio")
            sys.exit(1)
        
        # Crear directorio de resultados si es necesario
        if save_files and results_directory:
            try:
                os.makedirs(results_directory, exist_ok=True)
                print(f"✅ Directorio de resultados preparado: {results_directory}")
            except Exception as e:
                print(f"❌ ERROR: No se pudo crear el directorio de resultados: {e}")
                sys.exit(1)
        
        # Buscar archivos de video en el directorio
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v']
        video_files = []
        
        for ext in video_extensions:
            video_files.extend(Path(video_directory).glob(f"*{ext}"))
            video_files.extend(Path(video_directory).glob(f"*{ext.upper()}"))
        
        if not video_files:
            print(f"❌ ERROR: No se encontraron archivos de video en '{video_directory}'")
            print(f"📋 Extensiones soportadas: {', '.join(video_extensions)}")
            sys.exit(1)
        
        print(f"✅ Encontrados {len(video_files)} archivo(s) de video")
        print("-" * 60)
        
        # Crear instancia del transcriptor
        transcriber = VideoTranscriber(
            whisper_model=whisper_model,
            translation_model=translation_model
        )
        
        # Procesar cada video
        processed_count = 0
        error_count = 0
        
        for i, video_file in enumerate(video_files, 1):
            print(f"\n🎬 Procesando video {i}/{len(video_files)}: {video_file.name}")
            
            try:
                # Procesar video con las opciones correspondientes
                resultado = transcriber.process_video(
                    str(video_file),
                    results_path=results_directory if save_files else None,
                    save=save_files,
                    return_results=True  # Siempre retornar para mostrar preview
                )
                
                if resultado and resultado['transcription'] and not resultado['transcription'].startswith('FALLO'):
                    print(f"✅ Video procesado exitosamente: {video_file.name}")
                    
                    if save_files:
                        print(f"💾 Archivos guardados en: {results_directory}")
                    
                    # Mostrar una muestra de los resultados
                    transcription_preview = resultado['transcription'][:100] + "..." if len(resultado['transcription']) > 100 else resultado['transcription']
                    translation_preview = resultado['translation'][:100] + "..." if len(resultado['translation']) > 100 else resultado['translation']
                    
                    print(f"📝 Transcripción (muestra): {transcription_preview}")
                    print(f"🌐 Traducción (muestra): {translation_preview}")
                    processed_count += 1
                else:
                    print(f"⚠️  Error en transcripción: {video_file.name}")
                    error_count += 1
                
            except Exception as e:
                print(f"❌ Error procesando {video_file.name}: {str(e)}")
                error_count += 1
        
        print("\n" + "="*60)
        print("📊 RESUMEN DEL PROCESAMIENTO:")
        print("="*60)
        print(f"✅ Videos procesados exitosamente: {processed_count}")
        print(f"❌ Videos con errores: {error_count}")
        print(f"📁 Total de videos encontrados: {len(video_files)}")
        if save_files:
            print(f"💾 Archivos guardados en: {results_directory}")
        else:
            print("📋 Modo: Solo visualización (no se guardaron archivos)")
        print("="*60)
        
    except ValueError as e:
        if "GROQ_API_KEY" in str(e):
            print("❌ ERROR DE CONFIGURACIÓN: API Key no encontrada")
            print("\n🔧 Para configurar tu API Key:")
            print("1. Obtén tu API key gratuita de Groq en: https://console.groq.com/")
            print("2. Crea un archivo .env en la misma carpeta del script con:")
            print("   GROQ_API_KEY=tu_api_key_aqui")
            print("3. O configura la variable de entorno GROQ_API_KEY")
        else:
            print(f"❌ Error de configuración: {e}")
        sys.exit(1)
        
    except KeyboardInterrupt:
        print("\n⚠️  Proceso interrumpido por el usuario")
        sys.exit(1)
        
    except Exception as e:
        print(f"❌ Error inesperado: {e}")
        logger.error(f"Error inesperado: {e}")
        sys.exit(1)