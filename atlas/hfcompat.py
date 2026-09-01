"""One transformers behaviour change, in one place.

**The symptom.** Searching frames by phrase failed with

    encode failed: AttributeError: 'BaseModelOutputWithPooling' object has no
    attribute 'norm'

and the interface rendered that as *"that search needs a model … it is not
installed here"*. The diagnosis was wrong in the way that costs the most time:
the model was installed, downloaded, and loaded successfully — the log line
`image query encoder ready — openai/clip-vit-large-patch14 on cpu in 7s` was
printed seconds earlier. Nothing needed installing. The call after the load was
reading a field that had moved.

**What moved.** Up to transformers 4.x, `CLIPModel.get_text_features()` and
`get_image_features()` returned a bare tensor of shape `(batch, projection_dim)`
— the projected embedding, ready to normalise. In 5.x they return the whole
`BaseModelOutputWithPooling`, with the projection written *into* it::

    text_outputs = self.text_model(...)                     # transformers 5.x
    pooled_output = text_outputs.pooler_output
    text_outputs.pooler_output = self.text_projection(pooled_output)
    return text_outputs

So the embedding is still there, still projected, still correct — it is
`.pooler_output` now instead of the return value. `SiglipModel` returns its
`text_model`/`vision_model` output directly, where `pooler_output` is the head
output and is already the embedding, so the same rule reads both.

**Why a shared module rather than four edits.** Four call sites across two
modules and both repos build query vectors and index vectors with these two
methods, and they must agree: a query encoded one way and frames encoded another
do not live in the same space, and the search does not fail, it just ranks
wrongly. One rule, imported, cannot drift apart. Stdlib-only and importing no
part of the application, for the reason `subproc.py` gives.

**Why not pin transformers instead.** The index side runs on Kaggle, whose image
we do not control and which upgrades under us; the query side runs on a laptop
where torch is sometimes blocked outright. A version floor that has to hold in
both places is a version floor that will be wrong in one of them. Reading
whichever shape arrives costs six lines and survives the next change of mind.

**Why the tuple is refused rather than indexed.** `@can_return_tuple` turns the
output into `output.to_tuple()` when `return_dict=False`, and `to_tuple()` skips
`None` fields — so the position of `pooler_output` depends on which optional
outputs were requested, and `last_hidden_state` sits ahead of it. Guessing an
index and landing on `last_hidden_state` yields an unprojected vector that
normalises to length 1 and ranks with total confidence in a space nothing else
occupies. That is the failure `vsearch._load_onnx` already refuses an export for,
and it is refused here for the same reason: nobody in this tree sets
`return_dict=False`, and if somebody starts, an error naming the cause is worth
more than a search that quietly returns the wrong frames.
"""


def projected(out, what: str = "features"):
    """The projected embedding tensor, whatever shape the call handed back.

    `out` is the return value of `get_text_features` / `get_image_features`.
    Accepts the 5.x `ModelOutput` and the 4.x bare tensor; raises on the tuple.
    """
    pooled = getattr(out, "pooler_output", None)
    if pooled is not None:
        return pooled                       # transformers 5.x — already projected
    if isinstance(out, (tuple, list)):
        raise TypeError(
            f"{what} came back as a {type(out).__name__} of {len(out)} — that is "
            "transformers' `return_dict=False` form, where the projected "
            "embedding's position is not fixed. Nothing here sets that; set "
            "`config.return_dict = True` on the model rather than indexing it")
    return out                              # transformers 4.x — the tensor itself
