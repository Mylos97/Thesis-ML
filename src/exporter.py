import numpy as np
import torch.onnx

def export_model(model, x, model_name="super_resolution.onnx") -> None:
    model.eval()
    torch_out = model(x)
    model_name = model_name if '.onnx' in model_name else model_name + '.onnx' 
    torch.onnx.export(model,               # model being run
                    args=(x),                         # model input (or a tuple for multiple inputs)
                    f=model_name,   # where to save the model (can be a file or file-like object)
                    export_params=True,        # store the trained parameter weights inside the model file
                    opset_version=10,          # the ONNX version to export the model to
                    do_constant_folding=True,  # whether to execute constant folding for optimization
                    input_names = ['input'],   # the model's input names
                    output_names = ['output'], # the model's output names
                    dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                    'output' : {0 : 'batch_size'}})

    import onnx

    onnx_model = onnx.load(model_name)
    onnx.checker.check_model(onnx_model)
    
    import onnxruntime

    ort_session = onnxruntime.InferenceSession(model_name, providers=["CPUExecutionProvider"])

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x[0]), ort_session.get_inputs()[1].name: to_numpy(x[1])}
    ort_outs = ort_session.run(None, ort_inputs)

    np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)
    
    