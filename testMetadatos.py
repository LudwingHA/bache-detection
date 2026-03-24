#!/usr/bin/env python3
# extract_frame_metadata.py

import subprocess
import json
import os
import sys
from datetime import datetime

def extract_metadata_with_exiftool(video_file):
    """Extrae metadatos del video usando exiftool"""
    
    if not os.path.exists(video_file):
        print(f"Error: El archivo {video_file} no existe")
        return None
    

    output_dir = f"metadata_{os.path.splitext(os.path.basename(video_file))[0]}"
    os.makedirs(output_dir, exist_ok=True)
    

    print("Extrayendo metadatos completos...")
    cmd = ["exiftool", "-j", "-G1", "-struct", "-api", "LargeFileSupport=1", video_file]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        metadata = json.loads(result.stdout)
        

        json_file = os.path.join(output_dir, "complete_metadata.json")
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        print(f"✓ JSON guardado: {json_file}")
        

        frame_info = {
            "video_file": video_file,
            "extraction_date": datetime.now().isoformat(),
            "frame_metadata": {}
        }
        

        frame_fields = [
            "VideoFrameRate", "VideoFrameCount", "ImageWidth", "ImageHeight",
            "Duration", "FrameRate", "VideoFrameSize", "Timecode",
            "MediaCreateDate", "MediaModifyDate", "TrackCreateDate",
            "GPSPosition", "GPSAltitude", "GPSLatitude", "GPSLongitude"
        ]
        

        for field in frame_fields:
            try:
                cmd_field = ["exiftool", "-" + field, video_file]
                field_result = subprocess.run(cmd_field, capture_output=True, text=True, check=True)
                if field_result.stdout.strip():
                    frame_info["frame_metadata"][field] = field_result.stdout.strip()
            except:
                pass
        

        txt_file = os.path.join(output_dir, "frame_metadata.txt")
        with open(txt_file, "w", encoding="utf-8") as f:
            f.write(f"=== METADATOS DE FRAMES ===\n")
            f.write(f"Archivo: {video_file}\n")
            f.write(f"Fecha extracción: {datetime.now().isoformat()}\n\n")
            
            for key, value in frame_info["frame_metadata"].items():
                f.write(f"{key}: {value}\n")
        
        print(f"✓ Metadatos de frames guardados: {txt_file}")
        

        try:
            cmd_timecode = ["exiftool", "-Timecode", "-AllDates", video_file]
            timecode_result = subprocess.run(cmd_timecode, capture_output=True, text=True, check=True)
            timecode_file = os.path.join(output_dir, "timecodes.txt")
            with open(timecode_file, "w", encoding="utf-8") as f:
                f.write("=== TIMECODES Y FECHAS ===\n")
                f.write(timecode_result.stdout)
            print(f"✓ Timecodes guardados: {timecode_file}")
        except:
            pass
        
        print(f"\n✅ Todos los metadatos guardados en: {output_dir}/")
        

        print("\n=== RESUMEN DE FRAMES ===")
        for key in ["VideoFrameRate", "VideoFrameCount", "ImageWidth", "ImageHeight", "Duration"]:
            if key in frame_info["frame_metadata"]:
                print(f"{key}: {frame_info['frame_metadata'][key]}")
        
        return metadata
        
    except subprocess.CalledProcessError as e:
        print(f"Error ejecutando exiftool: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error parseando JSON: {e}")
        return None

def main():
    if len(sys.argv) < 2:
        print("Uso: python extract_frame_metadata.py <archivo_video>")
        print("Ejemplo: python extract_frame_metadata.py video.mp4")
        sys.exit(1)
    
    video_file = sys.argv[1]
    extract_metadata_with_exiftool(video_file)

if __name__ == "__main__":
    main()