import numpy as np
import torch.onnx
import torch.nn.functional as F
import onnx
import onnxruntime

def export_model(model, x, model_name) -> None:
    ort_input = x
    if type(x[0]) == list:
        ort_input = sum(x, [])

    amount_inputs = len(ort_input)
    inputs = [f"input{i+1}" for i in range(amount_inputs)]
    print(f"Inputs: {inputs}")
    axes = {f"input{i+1}": {0: "batch_size", 1: "height", 2: "width"} for i in range(amount_inputs)}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")

    if "vae" in model_name:
        model.training = False

    print(f"Now exporting {model_name}", flush=True)
    model.eval()
    model.training = False
    model = model.to("cpu")

    for i in range(len(x)):
        x[i] = x[i].to("cpu")

    torch.onnx.export(
        model,
        args=(x),
        f=model_name,
        export_params=True,
        opset_version=20,
        do_constant_folding=True,
        input_names=inputs,
        output_names=["output"],
        dynamic_axes=axes,
    )

    print("Finished exporting model", flush=True)

    torch_out = model(x).to("cpu")
    onnx_model = onnx.load(model_name)
    onnx.checker.check_model(onnx_model)

    ort_session = onnxruntime.InferenceSession(
        model_name, providers=["CPUExecutionProvider"]
    )
    options = ort_session.get_session_options()
    options.intra_op_num_threads = 1

    def to_numpy(tensor):
        return (
            #tensor.detach().to(device).cpu().numpy()
            tensor.detach().cpu().numpy()
            if tensor.requires_grad
            #else tensor.to(device).cpu().numpy()
            else tensor.cpu().numpy()
        )

    ort_inputs = {}
    for i, input in enumerate(ort_session.get_inputs()):
        print(f"ORT Input: {input}")
        # might need padding:
        """
        if input.name == 'input1':
            if input.shape[2] > x[i].shape[2]:
                pad_len = input.shape[2] - x[i].shape[2]
                print(f"Padding {x[i].shape} to {input.shape}")
                x[i] = F.pad(x[i], (0, pad_len))  # pad last dim
                print(f"Padded {x[i].shape}")

        if input.name == 'input2':
            if input.shape[1] > x[i].shape[1]:
                pad_len = input.shape[1] - x[i].shape[1]
                x[i] = F.pad(x[i], (0, 0, 0, pad_len))  # pad middle dim
                print(f"Padded again {x[i].shape}")
        """
        ort_inputs[input.name] = to_numpy(x[i])

    print(f"Checking the output of the model {model_name}", flush=True)
    ort_outs = ort_session.run(None, ort_inputs)
    np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)
    print("All good!")
    print(f"Final inference: {ort_outs[0][0]}")
