import cv2
from PIL import Image
import numpy as np
import time
import os


video_entrada = "video_entrada.webm"
video_salida = "videoB_N.mp4"

carpeta = os.path.dirname(os.path.abspath(__file__))

video_entrada = os.path.join(carpeta, video_entrada)
video_salida = os.path.join(carpeta, video_salida)


print("   PROCESAMIENTO DE VIDEO")
print("   CONVERSIÓN A BLANCO Y NEGRO")

tiempo_inicio = time.time()

video = cv2.VideoCapture(video_entrada)


if not video.isOpened():
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

duracion = cantidad_frames / fps

print("\nInformación del video:")
print(f"Resolución: {ancho} x {alto}")
print(f"FPS: {fps:.2f}")
print(f"Frames: {cantidad_frames}")
print(f"Duración: {duracion / 60:.2f} minutos")

fourcc = cv2.VideoWriter_fourcc(*"mp4v")


salida = cv2.VideoWriter(
    video_salida,
    fourcc,
    fps,
    (ancho, alto),
    True
)


if not salida.isOpened():

    print("ERROR: No se pudo crear el video de salida.")

    video.release()

    exit()


print("\nProcesando video...\n")


frame_actual = 0


while True:

    correcto, frame = video.read()

    if not correcto:
        break

    frame_gris = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )
    imagen_pillow = Image.fromarray(
        frame_gris
    )

    frame_gris = np.asarray(
        imagen_pillow
    )

    frame_final = cv2.cvtColor(
        frame_gris,
        cv2.COLOR_GRAY2BGR
    )
    salida.write(frame_final)


    frame_actual += 1
    if frame_actual % 100 == 0:

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
        )


        frames_restantes = (
            cantidad_frames -
            frame_actual
        )


        tiempo_restante = (
            frames_restantes /
            velocidad
        )


        print(
            f"\r"
            f"Progreso: {porcentaje:6.2f}% | "
            f"Frames: {frame_actual}/{cantidad_frames} | "
            f"Velocidad: {velocidad:.2f} FPS | "
            f"Tiempo restante: "
            f"{tiempo_restante / 60:.2f} min",
            end=""
        )

video.release()
salida.release()

tiempo_fin = time.time()

tiempo_total = (
    tiempo_fin -
    tiempo_inicio
)
print("\n\n")
print("       PROCESAMIENTO TERMINADO")

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

print("\nVideo generado:")

print(video_salida)
