# Todo
## Masking
- ~~Save masked image in the data folder~~ **(Ingrid)**
- ~~Replace cv2 with skimage~~ **(Noa)**
- ~~Add detection of beamstop~~ **(Adnan)**
- ~~Improve automatic detection of lines by adding a contrast requirement~~
- ~~Combine `findLines` and `adnan_test`. Change from `cv2` to `skimmage`.~~ **(Noa)**
- ~~Add user input for thresholds.~~ **(Noa)**
- ~~Find vertical lines~~ **(Ingrid)**
- Select multiple areas to mask **(Adnan?)**
- Manual image cropping to remove matlab frame **(Adnan?)**
- Add detection of beamstop holder. Hard because the beamstop is not so dark. Possible solutions:
  - Rotated input image. Does not work when the image pattern is not circular.
  - Image segmentation


## The model
- ~~Remove frame from the output of the DIP **(Noa)**
- See how the model works on the beamstop
-  Experiment with hyperparemeters and different models
   - Vase, Kate etc
   - reg_noise_std
   - LR
- Add early stopping
