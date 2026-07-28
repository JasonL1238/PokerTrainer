ARG TEXASSOLVER_COMMIT=42313c9cce96130d2341a8fc265160f580956054

FROM debian:bookworm-slim AS texassolver-builder
ARG TEXASSOLVER_COMMIT

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential ca-certificates cmake git libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --branch console https://github.com/bupticybee/TexasSolver.git /src/TexasSolver \
    && cd /src/TexasSolver \
    && git checkout "${TEXASSOLVER_COMMIT}" \
    && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build --target install --parallel 2

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8501 \
    POKER_DB_PATH=/data/poker_tracker.db \
    POKER_DATA_DIR=/data \
    POKERTRAINER_REQUIRE_AUTH=true \
    POKERTRAINER_SOLVER_THREADS=2 \
    POKERTRAINER_SOLVER_MEMORY_GB=8 \
    TEXAS_SOLVER_PATH=/opt/texassolver/console_solver \
    YOLO_CONFIG_DIR=/data/.ultralytics

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system --gid 10001 pokertrainer \
    && useradd --system --uid 10001 --gid pokertrainer --create-home pokertrainer

WORKDIR /app

COPY --from=texassolver-builder /src/TexasSolver/install/console_solver /opt/texassolver/console_solver
COPY --from=texassolver-builder /src/TexasSolver/install/resources /opt/texassolver/resources
COPY --from=texassolver-builder /src/TexasSolver/LICENSE /opt/texassolver/LICENSE

COPY requirements.txt requirements-cv.txt ./
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt -r requirements-cv.txt \
    && python -m pip install --no-deps ultralytics==8.3.203

COPY app.py ./
COPY poker_tracker ./poker_tracker
COPY cv_lab/scripts ./cv_lab/scripts
COPY cv_lab/models/region_spine_v1.pt ./cv_lab/models/region_spine_v1.pt
COPY cv_lab/models/card_cls_v1.pt ./cv_lab/models/card_cls_v1.pt
COPY cv_lab/models/card_templates.npz ./cv_lab/models/card_templates.npz
COPY cv_lab/models/ocr_templates.npz ./cv_lab/models/ocr_templates.npz
COPY cv_lab/models/pot_digits.npz ./cv_lab/models/pot_digits.npz
COPY .streamlit ./.streamlit

RUN mkdir -p /data \
    && chown -R pokertrainer:pokertrainer /app /data

USER pokertrainer
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=45s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('PORT', '8501') + '/_stcore/health', timeout=3)"

CMD ["sh", "-c", "streamlit run app.py --server.port=${PORT} --server.address=0.0.0.0 --server.headless=true"]
