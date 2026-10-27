"""Test double for omnivoice.models.omnivoice_flashinfer."""

FAIL = False


def apply_flashinfer(model, cuda_graph_buckets=None, **kwargs):
    if FAIL:
        raise RuntimeError("no kernel image is available for execution on the device")
    model.graph_buckets = cuda_graph_buckets
    return model
