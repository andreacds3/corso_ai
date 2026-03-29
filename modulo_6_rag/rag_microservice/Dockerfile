FROM pytorch/pytorch:2.11.0-cuda13.0-cudnn9-devel

# 1. Imposta subito la working directory
WORKDIR /usr/src/app

# 2. Copia SOLO il file dei requisiti nella cartella corrente (.)
# Questo layer verrà ricalcolato SOLO se il file requirements.txt cambia
COPY ./requirements.txt .
RUN apt-get update
RUN apt install python3.12-venv -y
# 3. Installa le dipendenze
RUN python -m venv --system-site-packages /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY ./requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt
COPY . .