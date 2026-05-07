#!/bin/bash
# Run dopo che MSTL annuale ha finito.
# Lancia i nuovi approcci: E (ILR-only) e D2 aggiornato (COMBO+ILR annuale)

set -e
cd "$(dirname "$0")/.."

echo "=== 1. Create ILR-only embedding (Approccio E) ==="
python scripts/create_ilr_embedding.py

echo ""
echo "=== 2. Step03 + Step04 + Step05 per ILR ==="
python pipeline/step03_pca.py --exp ILR
python pipeline/step04_tomato.py --exp ILR
python pipeline/step05_cluster_quality.py --exp ILR

echo ""
echo "=== 3. Update COMBO+ILR embedding (Approccio D con MSTL annuale) ==="
python scripts/create_combo_ilr_embedding.py

echo ""
echo "=== 4. Step03 + Step04 + Step05 per D2 ==="
python pipeline/step03_pca.py --exp D2
python pipeline/step04_tomato.py --exp D2
python pipeline/step05_cluster_quality.py --exp D2

echo ""
echo "=== Done ==="
