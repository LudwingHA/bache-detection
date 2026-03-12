import cv2
import pytesseract
from PIL import Image
import re
import json
from ultralytics import YOLO
import numpy as np
from pathlib import Path
import logging
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, List, Dict
import time
from datetime import datetime

# ================================
# CONFIGURACIÓN
# ================================
@dataclass
class Config:
    """Configuración centralizada del sistema"""
    video_path: str = "./videos-m/test3.mp4"
    model_path: str = "./best.pt"
    report_name: str = "reporte_baches_geolocalizados.json"
    confidence_threshold: float = 0.3
    ocr_roi: Tuple[int, int, int, int] = (0, 150, 0, 800)  # (y_start, y_end, x_start, x_end)
    gps_update_frequency: int = 5  # Actualizar GPS cada N frames
    save_video_output: bool = False
    output_video_path: str = "./output_detecciones.mp4"

class LoggerMixin:
    """Mixin para logging consistente"""
    @property
    def logger(self):
        if not hasattr(self, '_logger'):
            self._logger = logging.getLogger(self.__class__.__name__)
        return self._logger

class GPSProcessor(LoggerMixin):
    """Procesador de coordenadas GPS desde OCR"""
    
    def __init__(self, config: Config):
        self.config = config
        self.last_valid_coords: Optional[Tuple[float, float]] = None
        self.frame_counter = 0
        self.setup_logging()
    
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def dms_a_decimal(self, dms_str: str) -> Optional[float]:
        """Convierte coordenadas DMS a decimal"""
        patron = r"(\d+)°(\d+)'([\d.]+)\"?([NSEW])"
        match = re.search(patron, dms_str.strip())
        if not match:
            return None
        
        try:
            grados, minutos, segundos = float(match.group(1)), float(match.group(2)), float(match.group(3))
            direccion = match.group(4)
            decimal = grados + (minutos / 60) + (segundos / 3600)
            
            if direccion in ["S", "W"]:
                decimal *= -1
            return round(decimal, 6)
        except ValueError as e:
            self.logger.error(f"Error convirtiendo coordenada {dms_str}: {e}")
            return None
    
    def preprocess_image(self, frame: np.ndarray) -> np.ndarray:
        """Preprocesamiento optimizado para OCR"""
        # Extraer ROI
        y1, y2, x1, x2 = self.config.ocr_roi
        roi = frame[y1:y2, x1:x2]
        
        if roi.size == 0:
            return None
        
        # Mejorar calidad para OCR
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Aplicar filtros para mejorar texto
        denoised = cv2.fastNlMeansDenoising(gray, h=30)
        _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Redimensionar para mejorar OCR
        height, width = binary.shape
        if width < 800:
            scale = 800 / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            binary = cv2.resize(binary, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        return binary
    
    def extract_coordinates(self, frame: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
        """Extrae coordenadas del frame con caché para mejorar rendimiento"""
        self.frame_counter += 1
        
        # Solo procesar OCR cada N frames para optimizar
        if self.last_valid_coords and self.frame_counter % self.config.gps_update_frequency != 0:
            return self.last_valid_coords
        
        processed_img = self.preprocess_image(frame)
        if processed_img is None:
            return self.last_valid_coords if self.last_valid_coords else (None, None)
        
        # Configurar pytesseract para mejor precisión
        custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789°\'".NSWE '
        texto = pytesseract.image_to_string(
            Image.fromarray(processed_img), 
            config=custom_config
        )
        
        patron = r"\d+°\d+'\d+\.\d+\"?[NSEW]"
        coincidencias = re.findall(patron, texto)
        
        if len(coincidencias) >= 2:
            lat = self.dms_a_decimal(coincidencias[0])
            lon = self.dms_a_decimal(coincidencias[1])
            if lat is not None and lon is not None:
                self.last_valid_coords = (lat, lon)
                return lat, lon
        
        return self.last_valid_coords if self.last_valid_coords else (None, None)

class BacheDetector(LoggerMixin):
    """Detector de baches usando YOLO con tracking"""
    
    def __init__(self, config: Config):
        self.config = config
        self.model = YOLO(config.model_path)
        self.reporte: Dict[int, dict] = {}
        self.setup_output_video()
    
    def setup_output_video(self):
        """Configura el video de salida si está habilitado"""
        self.video_writer = None
        if self.config.save_video_output:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(
                self.config.output_video_path, 
                fourcc, 
                20.0, 
                (1920, 1080)
            )
    
    def process_frame(self, frame: np.ndarray, lat: float, lon: float) -> np.ndarray:
        """Procesa un frame individual con detecciones"""
        # Ejecutar detección con tracking
        results = self.model.track(
            frame, 
            persist=True, 
            conf=self.config.confidence_threshold,
            verbose=False,
            iou=0.5
        )
        
        # Procesar detecciones si existen
        if results[0].boxes.id is not None and lat is not None and lon is not None:
            ids = results[0].boxes.id.int().cpu().tolist()
            clases = results[0].boxes.cls.int().cpu().tolist()
            confidences = results[0].boxes.conf.cpu().tolist()
            
            for b_id, cls, conf in zip(ids, clases, confidences):
                if b_id not in self.reporte:
                    self.reporte[b_id] = {
                        "bache_id": b_id,
                        "tipo": self.model.names[cls],
                        "confianza": round(conf, 3),
                        "lat": lat,
                        "lon": lon,
                        "timestamp": datetime.now().isoformat(),
                        "google_maps": f"https://www.google.com/maps?q={lat},{lon}"
                    }
                    self.logger.info(f"📍 Nuevo bache detectado! ID: {b_id} en ({lat}, {lon})")
        
        # Anotar frame
        annotated_frame = results[0].plot()
        self._add_overlay_info(annotated_frame, lat, lon)
        
        return annotated_frame
    
    def _add_overlay_info(self, frame: np.ndarray, lat: float, lon: float):
        """Agrega información superpuesta al frame"""
        # Mostrar coordenadas actuales
        coord_text = f"GPS: {lat:.6f}, {lon:.6f}" if lat and lon else "GPS: No disponible"
        cv2.putText(frame, coord_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Mostrar contador de detecciones
        detections_text = f"Baches detectados: {len(self.reporte)}"
        cv2.putText(frame, detections_text, (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    def save_report(self):
        """Guarda el reporte en formato JSON"""
        report_path = Path(self.config.report_name)
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(list(self.reporte.values()), f, indent=4, ensure_ascii=False)
        self.logger.info(f"📁 Reporte guardado en: {report_path}")

class VideoProcessor(LoggerMixin):
    """Procesador principal de video"""
    
    def __init__(self, config: Config):
        self.config = config
        self.gps_processor = GPSProcessor(config)
        self.bache_detector = BacheDetector(config)
        self.setup_logging()
    
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def run(self):
        """Ejecuta el pipeline completo de procesamiento"""
        cap = cv2.VideoCapture(self.config.video_path)
        if not cap.isOpened():
            self.logger.error(f"No se pudo abrir el video: {self.config.video_path}")
            return
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        self.logger.info(f"🚀 Iniciando procesamiento - {total_frames} frames, {fps:.2f} FPS")
        
        start_time = time.time()
        frame_count = 0
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Mostrar progreso cada 100 frames
                if frame_count % 100 == 0:
                    progress = (frame_count / total_frames) * 100
                    elapsed = time.time() - start_time
                    self.logger.info(f"Progreso: {progress:.1f}% - {frame_count}/{total_frames} frames - Tiempo: {elapsed:.1f}s")
                
                # Extraer coordenadas
                lat, lon = self.gps_processor.extract_coordinates(frame)
                
                # Procesar frame con detecciones
                annotated_frame = self.bache_detector.process_frame(frame, lat, lon)
                
                # Guardar frame si está habilitado
                if self.bache_detector.video_writer:
                    self.bache_detector.video_writer.write(annotated_frame)
                
                # Mostrar resultado
                cv2.imshow("Sistema de Deteccion Vial", annotated_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.logger.info("Procesamiento interrumpido por el usuario")
                    break
                    
        except Exception as e:
            self.logger.error(f"Error durante el procesamiento: {e}", exc_info=True)
        
        finally:
            # Limpieza
            cap.release()
            cv2.destroyAllWindows()
            if self.bache_detector.video_writer:
                self.bache_detector.video_writer.release()
            
            # Guardar resultados
            self.bache_detector.save_report()
            
            elapsed_time = time.time() - start_time
            self.logger.info(f"\n✅ Proceso finalizado en {elapsed_time:.2f} segundos")
            self.logger.info(f"📊 Estadísticas:")
            self.logger.info(f"   - Frames procesados: {frame_count}")
            self.logger.info(f"   - Baches únicos detectados: {len(self.bache_detector.reporte)}")
            self.logger.info(f"   - FPS promedio: {frame_count/elapsed_time:.2f}")

def main():
    """Función principal"""
    # Cargar configuración (podría venir de un archivo externo)
    config = Config()
    
    # Crear y ejecutar procesador
    processor = VideoProcessor(config)
    processor.run()

if __name__ == "__main__":
    main()