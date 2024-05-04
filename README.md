# Usage

**Parameters**

--model vae (default), pairwise, cost
<br>
--retrain "path/to/data", (default) ""
<br>
--name "name-of-themodel", (default) ""

Example use model.py --model <model_name> --retrain <retrain>

```bash
python main.py --model pairwise --retrain src/Data/pairwise-encodings.txt --name model-name
```

---
Thanks a lot for the implementation of ["Tree Convolution"](https://github.com/RyanMarcus/TreeConvolution) used in the script.