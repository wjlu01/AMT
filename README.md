# A Text-Image Pair Is not Enough: Language-Vision Relation Inference with Auxiliary Modality Translation
By Wenjie Lu, Dong Zhang, Shoushan Li, Guodong Zhou.

This is the pytorch implemention for "A Text-Image Pair Is not Enough: Language-Vision Relation Inference with Auxiliary Modality Translation". 

## Environment
- python = 3.7
- torch = 1.9.0+cu111
- pytorch-pretrained-bert = 0.6.2
- ```pip install torch torchvision sklearn numpy tqdm matplotlib```

## Dataset
Download [Bloomberg's images](https://drive.google.com/file/d/1KkeMBo32gDEfrbmUjTEX4whpdda_wrYn/view?usp=sharing) and [Clue's images](https://drive.google.com/file/d/1H2rE8DJtHFgNnlGO8CSe6vLJmTtvtifp/view?usp=sharing).

We expect the directory structure to be the following:
```
amt/data/
  bloomberg/
    rel_img/ #bloomberg images
  clue/
    rel_img/ #clue images
```

## Generated text from images
The generated text used in the model has been placed in the .jsonl files for each dataset. 

You can generated text from images by [CATR](https://github.com/saahiluppal/catr/tree/fac82f9b4004b1dd39ccf89760b758ad19a2dbee):

1) Clone the [CATR](https://github.com/saahiluppal/catr/tree/fac82f9b4004b1dd39ccf89760b758ad19a2dbee) repository locally and install all the requirements for CATR.
2) Move `caption_multi.py` into the cloned CATR.
3) Set the parameter "path" to indicate where the image is stored and move images to the path.
4) run ```python caption_multi.py```

## Model Training
- Train, evaluate, and test the three tasks in Bloomberg.
```
sh shs/run_bloomberg.sh
```

- Train, evaluate, and test Clue.
```
sh shs/run_clue.sh
```

## Model Testing
### Download Checkpoints
We provide links to download our checkpoints.
- [Finetuned checkpiont for Image task of Bloomberg](https://drive.google.com/drive/folders/1zSt7EYVckR55qwbF4IrJvhUakultyQ8X?usp=sharing)
- [Finetuned checkpiont for Text task of Bloomberg](https://drive.google.com/drive/folders/16gRsz0jM_zTMm-MUq9j-k75TPNvlr56Q?usp=sharing)
- [Finetuned checkpiont for Text+Image task of Bloomberg](https://drive.google.com/drive/folders/1AEWuaL6Z9QST4bUi-YTQ3DVytRZ4rmJ7?usp=sharing)
- [Finetuned checkpiont for Clue](https://drive.google.com/drive/folders/121kAaCyT1xECZKESFlAtchc9fYZaRK1W?usp=sharing)

We expect the directory structure to be the following:
```
save/AMT/
  clue/multi_label/model/ #Clue checkpoint
  bloomberg/img_label/model/ #Bloomberg's image task checkpoint
  bloomberg/txt_label/model/ #Bloomberg's txt task checkpoint
  bloomberg/it_label/model/ #Bloomberg's text+image task checkpoint
```

### Testing
- Test the three tasks in Bloomberg.
```
sh shs/test_bloomberg.sh
```

- Test Clue.
```
sh shs/test_clue.sh
```

## Acknowledgement
The code is based on the [MMBT](https://github.com/facebookresearch/mmbt) and [CATR](https://github.com/saahiluppal/catr/tree/fac82f9b4004b1dd39ccf89760b758ad19a2dbee). Thanks for their great works.
