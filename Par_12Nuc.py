import cv2
from PIL import Image
import numpy as np
import time
import os

from multiprocessing import Pool, freeze_support

NUM_NUCLEOS = 12

video_entrada = "video_entrada.webm"
video_salida = "videoBN_12_nucleos.mp4"

def configurar_proceso():

    cv2.setNumThreads(1)

def convertir_frame(frame):

    frame_gris = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    imagen_pillow = Image.fromarray(frame_gris)

    frame_gris = np.asarray(imagen_pillow)

    frame_final = cv2.cvtColor(
        frame_gris,
        cv2.COLOR_GRAY2BGR
    )

    return frame_final


if __name__ == "__main__":

    freeze_support()

    tiempo_inicio = time.time()

    carpeta = os.path.dirname(
        os.path.abspath(__file__)
    )

    video_entrada = os.path.join(
        carpeta,
        video_entrada
    )

    video_salida = os.path.join(
        carpeta,
        video_salida
    )

    print(" PROCESAMIENTO DE VIDEO - 12 NÚCLEOS")
    print(" Conversión a blanco y negro")
    
    video = cv2.VideoCapture(video_entrada)

    if not video.isOpened():

        print("No se pudo abrir el video:")
        print(video_entrada)

        exit()

    fps = video.get(cv2.CAP_PROP_FPS)

    ancho = int(
        video.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    alto = int(
        video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    cantidad_frames = int(
        video.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    duracion = (
        cantidad_frames / fps
        if fps > 0
        else 0
    )

    print(f"Resolución: {ancho} x {alto}")
    print(f"FPS originales: {fps:.2f}")
    print(f"Frames totales: {cantidad_frames}")
    print(f"Duración: {duracion / 60:.2f} minutos")
    print(f"Núcleos utilizados: {NUM_NUCLEOS}")
    print()

    codec = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    salida = cv2.VideoWriter(
        video_salida,
        codec,
        fps,
        (ancho, alto)
    )

    if not salida.isOpened():

        print("No se pudo crear el video de salida.")

        video.release()

        exit()
    tamanio_lote = NUM_NUCLEOS

    frame_actual = 0

    with Pool(
        processes=NUM_NUCLEOS,
        initializer=configurar_proceso
    ) as pool:

        while True:

            frames = []

            for _ in range(tamanio_lote):

                ret, frame = video.read()

                if not ret:
                    break

                frames.append(frame)

            if not frames:
                break

            frames_procesados = pool.map(
                convertir_frame,
                frames,
                chunksize=1
            )

            for frame_procesado in frames_procesados:

                salida.write(
                    frame_procesado
                )

                frame_actual += 1

            tiempo_actual = time.time()

            tiempo_transcurrido = (
                tiempo_actual -
                tiempo_inicio
            )

            porcentaje = (
                frame_actual /
                cantidad_frames
            ) * 100

            velocidad = (
                frame_actual /
                tiempo_transcurrido
                if tiempo_transcurrido > 0
                else 0
            )

            frames_restantes = (
                cantidad_frames -
                frame_actual
            )

            tiempo_restante = (
                frames_restantes /
                velocidad
                if velocidad > 0
                else 0
            )

            print(
                f"\r"
                f"Progreso: {porcentaje:6.2f}% | "
                f"Frames: {frame_actual}/{cantidad_frames} | "
                f"Velocidad: {velocidad:.2f} FPS | "
                f"Tiempo restante: "
                f"{tiempo_restante / 60:.2f} min",
                end="",
                flush=True
            )

    video.release()
    salida.release()

    tiempo_total = (
        time.time() -
        tiempo_inicio
    )

    print("\n")
    print("PROCESAMIENTO TERMINADO")

    print(
        f"Frames procesados: "
        f"{frame_actual}"
    )

    print(
        f"Tiempo total: "
        f"{tiempo_total:.2f} segundos"
    )

    print(
        f"Tiempo total: "
        f"{tiempo_total / 60:.2f} minutos"
    )

    print(
        f"Velocidad promedio: "
        f"{frame_actual / tiempo_total:.2f} FPS"
    )
    print(
        f"Núcleos utilizados: "
        f"{NUM_NUCLEOS}"
    )
    print(
        f"Video generado: "
        f"{video_salida}"
    )