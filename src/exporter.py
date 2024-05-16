import numpy as np
import torch.onnx
import onnx
import onnxruntime


def export_model(model, x, model_name) -> None:
    ort_input = x
    if type(x[0]) == list: 
        ort_input = sum(x, [])
    amount_inputs = len(ort_input)
    inputs = [f"input{i+1}" for i in range(amount_inputs)]
    axes = {f"input{i+1}": {0: "batch"} for i in range(amount_inputs)}

    if "vae" in model_name:
        model.training = False

    print(f"Now exporting {model_name}", flush=True)
    model.eval()
    torch.onnx.export(
        model,
        args=(x),
        f=model_name,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=inputs,
        output_names=["output"],
        dynamic_axes=axes,
    )
    print("Finished exporting model", flush=True)
    torch_out = model(x)
    onnx_model = onnx.load(model_name)
    onnx.checker.check_model(onnx_model)
    ort_session = onnxruntime.InferenceSession(
        model_name, providers=["CPUExecutionProvider"]
    )
    options = ort_session.get_session_options()
    options.intra_op_num_threads = 1

    def to_numpy(tensor):
        print("tensor: " + tensor)
        return (
            tensor.detach().cpu().numpy()
            if tensor.requires_grad
            else tensor.cpu().numpy()
        )

    ort_inputs = {}
    for i, input in enumerate(ort_session.get_inputs()):
        ort_inputs[input.name] = to_numpy(ort_input[i])

    print("Checking the output of the model", flush=True)
    ort_outs = ort_session.run(None, ort_inputs)
    np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)
    print("All good!")
