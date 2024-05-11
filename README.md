# Usage

**Parameters**

--model vae (default), pairwise, cost
<br>
--retrain 'path/to/data', (default) ''''
<br>
--name 'name-of-themodel', (default) ''''
<br>
--lr '[1e-6, 1e-3]', (default) '[1e-6, 0.1]'
<br>
--epochs 10, (default) 100
<br>
--trials 15, (default) 25

Example use model.py --model <model_name> --retrain <retrain>

```bash
python main.py --model pairwise --retrain src/Data/pairwise-encodings.txt --name model-name
```

---
Thanks to Ryan Marcus for the implementation of ["Tree Convolution"](https://github.com/RyanMarcus/TreeConvolution) that was used in the training script.