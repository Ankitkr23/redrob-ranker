# Self-contained sandbox / reproduction image (spec Section 10.5 alternative to a
# hosted sandbox). Builds and runs unmodified. Lean by design: it uses the
# portable TF-IDF backend, so no torch / model download / network is needed.
#
#   Build:   docker build -t redrob-ranker .
#
#   Demo (ranks the bundled 50-candidate sample, prints the CSV):
#            docker run --rm redrob-ranker
#
#   Reproduce on the full pool (mount your candidates file + output dir):
#            docker run --rm -v "$PWD":/data redrob-ranker \
#              --candidates /data/candidates.jsonl --out /data/submission.csv \
#              --semantic-backend tfidf
#
FROM python:3.12-slim

WORKDIR /app

# CPU-only, no network at runtime — everything is installed at build time.
COPY requirements-core.txt ./
RUN pip install --no-cache-dir -r requirements-core.txt

COPY src ./src
COPY data/sample_candidates.json data/job_description.txt ./data/
COPY rank.py ./

# Default run: rank the bundled sample end-to-end and print the ranked CSV.
CMD ["sh", "-c", "python rank.py --candidates data/sample_candidates.json --out /tmp/ranking.csv --semantic-backend tfidf --top-n 25 && echo '----- ranking.csv -----' && cat /tmp/ranking.csv"]
