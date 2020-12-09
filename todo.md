# Todo
## Masking
- ~~Save masked image in the data folder~~ **(Ingrid)**
- ~~Replace cv2 with skimage~~ **(Noa)**
- ~~Add detection of beamstop~~ **(Adnan)**
- ~~Improve automatic detection of lines by adding a contrast requirement~~
- ~~Combine `findLines` and `adnan_test`. Change from `cv2` to `skimmage`.~~ **(Noa)**
- ~~Add user input for thresholds.~~ **(Noa)**
- ~~Find vertical lines~~ **(Ingrid)**
- Select multiple areas to mask (beamstop holder specifically) **Adnan**
- Manual image cropping to remove matlab frame


## Inpainting
- ~~Remove frame from the output of the DIP **(Noa)**~~
- ~~See how the model works on the beamstop (Works bad) **(Noa)**~~
-  Experiment with hyperparemeters and different models **(Noa)**
   - Vase, Kate etc
   - reg_noise_std
   - LR, LR decay
- Create radial and angular intensity plots
- Add parsing of input image path and creation of folder structure?
- *Combine DIP result with original image*
- *Add early stopping*


