def get_input_dict(imgs, point_coords=None, point_labels=None,
                   boxes=None, mask_inputs=None):
    """
    Build per-sample input dicts for SAM (polyp 2D pipeline).

    The polyp pipeline resizes every image to a fixed (1024, 1024)
    inside `polyp_transform.py`, and keeps both predictions and ground
    truth in that same spatial frame. As a result, `original_size` and
    `image_size` are never consumed by `sam_call`, and there is no
    dependency on `MetaTensor` metadata.

    Parameters
    ----------
    imgs : torch.Tensor
        Batch of images, shape (B, C, H, W).
        For polyp: (B, 3, 1024, 1024), already ImageNet-normalized.

    point_coords, point_labels, boxes, mask_inputs : optional, batched
        Optional SAM prompts. When provided, the first dim MUST equal B.
        Each is attached PER-SAMPLE (indexed by `i`), matching SAM's
        native API. Not used in the polyp training pipeline (which
        relies on dense embeddings from the prompt learner) but kept
        for API completeness and future extensions.

    Returns
    -------
    list of dict
        One dict per sample, with the key 'image' and any of
        'point_coords', 'point_labels', 'boxes', 'mask_inputs' if given.
    """
    B = imgs.shape[0]
    batched_input = []

    for i, img in enumerate(imgs):
        single_input = {"image": img}

        if point_coords is not None:
            _assert_batch(point_coords, B, "point_coords")
            single_input["point_coords"] = _per_sample(point_coords, i, ndim=2)

        if point_labels is not None:
            _assert_batch(point_labels, B, "point_labels")
            single_input["point_labels"] = _per_sample(point_labels, i, ndim=1)

        if boxes is not None:
            _assert_batch(boxes, B, "boxes")
            single_input["boxes"] = _per_sample(boxes, i, ndim=2)

        if mask_inputs is not None:
            _assert_batch(mask_inputs, B, "mask_inputs")
            single_input["mask_inputs"] = mask_inputs[i]

        batched_input.append(single_input)

    return batched_input


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _assert_batch(x, B, name):
    """Assert that a prompt tensor has first dim == B."""
    if not hasattr(x, "shape"):
        raise TypeError(
            f"Prompt '{name}' must be a tensor-like with .shape, "
            f"got {type(x).__name__}"
        )
    if x.shape[0] != B:
        raise ValueError(
            f"Prompt '{name}' batch size mismatch: expected {B}, "
            f"got {x.shape[0]}"
        )


def _per_sample(x, i, ndim):
    """Extract sample i from a batched prompt. If the sample already has
    the final ndim, prepend a singleton so SAM sees shape (1, N, ...)."""
    s = x[i]
    if s.dim() == ndim:
        return s.unsqueeze(0)
    return s