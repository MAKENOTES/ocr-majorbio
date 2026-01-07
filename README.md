# OCR Project (DBNet + CRNN-CTC)

Two-stage OCR pipeline for scanned PDFs/images:
1) PDF -> page images (PyMuPDF)
2) Text detection (DBNet, line-level)
3) Crop lines -> text recognition (CRNN + CTC)
4) Structured JSON output

## Project Layout
- configs/: training and pipeline configs
- data/: pdfs, images, labels, crops, vocab
- detection/: DBNet skeleton + dataset
- recognition/: CRNN-CTC skeleton + dataset
- pipeline/: end-to-end orchestration
- tools/: rendering, visualization, dataset helpers

## Minimal Pipeline Usage
```bash
python main.py --pdf data/pdfs/example.pdf --config configs/pipeline.yaml
```
Output JSON: `outputs/final/result.json`

## Data Format
### Detection labels (ICDAR Localization)
`data/det_labels/{image_stem}.txt` with lines:
```
x1,y1,x2,y2,x3,y3,x4,y4,transcription
```

### Recognition labels
`data/rec_labels/labels.txt` with lines:
```
image_name.png<TAB>text
```

## Suggested Low-Cost Labeling (your scenario)
- Detection: 200-500 pages (line-level boxes)
- Recognition: 10k-30k cropped lines with text
- Rotation: not required; keep rotation augmentation at 0~3 degrees

## Notes
- `detection/train.py` and `recognition/train.py` are stubs for integrating full training.
- Replace model skeletons with full DBNet/CRNN implementations when ready.

## MMOCR DBNet Training (recommended)
1) Convert ICDAR txt labels to MMOCR JSON:
```bash
python tools/convert_icdar_to_mmocr.py \
  --images_dir data/images \
  --labels_dir data/det_labels \
  --label_prefix gt_ \
  --out_train_json data/mmocr/textdet_train.json \
  --out_val_json data/mmocr/textdet_val.json \
  --val_ratio 0.1
```

2) Install MMOCR deps (mmengine, mmcv, mmdet) and MMOCR:
```bash
python -m pip install -U openmim
python -m mim install "mmcv>=2.0.0rc4,<2.2.0" "mmdet>=3.0.0rc5,<3.4.0" "mmengine>=0.7.0,<1.1.0"
python -m pip install -e third_party/mmocr
```

3) Train DBNet:
```bash
python detection/train_mmocr.py --config configs/mmocr_dbnet_resnet18_contract.py --work_dir outputs/det_preds/mmocr_dbnet
```
