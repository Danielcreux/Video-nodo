python -m pip install fastapi uvicorn pydantic moviepy

uvicorn app:APP --reload

http://127.0.0.1:8000

- Carpeta esperada

- Crea dentro del mismo directorio de app.py:

WORKSPACE/
  assets/
    image.png
    audio.mp3
    a.mp3
    b.mp3
  notas/
    entrada.txt
  out/