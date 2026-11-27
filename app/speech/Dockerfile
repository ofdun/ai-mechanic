FROM python:3.12-slim-bookworm
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir faster-whisper==1.2.1 fastapi==0.116.1 uvicorn==0.35.0
ARG MODEL_REVISION=ebe41f70d5b6dfa9166e2c581c45c9c0cfc57b66
RUN python -c "import os, pathlib, urllib.request; p=pathlib.Path('/model'); p.mkdir(); [urllib.request.urlretrieve('https://huggingface.co/Systran/faster-whisper-base/resolve/'+os.environ['MODEL_REVISION']+'/'+f,p/f) for f in ('config.json','model.bin','tokenizer.json','vocabulary.txt')]"
COPY service.py .
USER nobody
CMD ["uvicorn","service:app","--host","0.0.0.0","--port","8003","--no-access-log"]
