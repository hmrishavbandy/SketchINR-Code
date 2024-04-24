# SketchINR
A First Look into Sketches as Implicit Neural Representations

- Before training, change the root directory here https://github.com/hmrishavbandy/SketchINR-Code/blob/707bf1f0be69b70b097075c2bf7f9bd456c59913/single_fit.py#L248
  to your dataset folder. The folder (for QuickDraw) should look like
  ```
  .
  └── Datasets
      └── QuickDraw
           ├── QuickDraw_Keys.pickle
           ├── QuickDraw_TestData/
           ├── QuickDraw_TrainData/
           └── QuickDraw_ValidData/
  ```

- Training: `python3 single_fit.py`

# ToDo
- Add requirements.txt for python dependencies
- Upload QD Dataset
