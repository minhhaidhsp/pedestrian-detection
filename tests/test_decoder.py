import torch

from models.decoder import RTDETRTransformer

HIDDEN_DIM = 128
FEAT_STRIDES = [4, 8, 16]  # P2, P3, P4 (P5 dropped)
INPUT_SIZE = 640
RESOLUTIONS = [INPUT_SIZE // s for s in FEAT_STRIDES]  # [160, 80, 40]
NUM_QUERIES = 300
NUM_CLASSES = 1


def _make_decoder(**overrides):
    kwargs = dict(
        num_classes=NUM_CLASSES,
        hidden_dim=HIDDEN_DIM,
        num_queries=NUM_QUERIES,
        feat_channels=[HIDDEN_DIM, HIDDEN_DIM, HIDDEN_DIM],
        feat_strides=FEAT_STRIDES,
        num_levels=3,
        nhead=4,
        num_decoder_layers=2,
        dim_feedforward=256,
    )
    kwargs.update(overrides)
    return RTDETRTransformer(**kwargs)


def _make_feats(batch):
    return [torch.randn(batch, HIDDEN_DIM, r, r) for r in RESOLUTIONS]


def test_decoder_forward_output_keys_and_shapes():
    torch.manual_seed(0)
    decoder = _make_decoder()
    decoder.eval()
    feats = _make_feats(batch=2)

    with torch.no_grad():
        out = decoder(feats)

    assert set(out.keys()) == {"pred_logits", "pred_boxes", "enc_pred_logits", "enc_pred_boxes"}
    assert out["pred_logits"].shape == (2, NUM_QUERIES, NUM_CLASSES)
    assert out["pred_boxes"].shape == (2, NUM_QUERIES, 4)
    # encoder query-selection proposals (see models/decoder.py DESIGN DECISION (2))
    assert out["enc_pred_logits"].shape == (2, NUM_QUERIES, NUM_CLASSES)
    assert out["enc_pred_boxes"].shape == (2, NUM_QUERIES, 4)


def test_pred_boxes_in_unit_range():
    torch.manual_seed(1)
    decoder = _make_decoder()
    decoder.eval()
    feats = _make_feats(batch=2)

    with torch.no_grad():
        out = decoder(feats)

    assert torch.all(out["pred_boxes"] >= 0.0)
    assert torch.all(out["pred_boxes"] <= 1.0)


def test_no_nan_or_inf():
    torch.manual_seed(2)
    decoder = _make_decoder()
    decoder.eval()
    feats = _make_feats(batch=2)

    with torch.no_grad():
        out = decoder(feats)

    assert torch.isfinite(out["pred_logits"]).all()
    assert torch.isfinite(out["pred_boxes"]).all()
    assert torch.isfinite(out["enc_pred_logits"]).all()
    assert torch.isfinite(out["enc_pred_boxes"]).all()


def test_batch_size_one_and_two():
    torch.manual_seed(3)
    decoder = _make_decoder()
    decoder.eval()

    for batch in (1, 2):
        feats = _make_feats(batch=batch)
        with torch.no_grad():
            out = decoder(feats)
        assert out["pred_logits"].shape[0] == batch
        assert out["pred_boxes"].shape[0] == batch


def test_num_queries_matches_config():
    decoder = _make_decoder(num_queries=300)
    assert decoder.num_queries == 300
    feats = _make_feats(batch=1)
    decoder.eval()
    with torch.no_grad():
        out = decoder(feats)
    assert out["pred_boxes"].shape[1] == 300


def test_backward_flows_gradients_to_input_features_and_parameters():
    torch.manual_seed(4)
    decoder = _make_decoder()
    decoder.train()

    feats = [torch.randn(2, HIDDEN_DIM, r, r, requires_grad=True) for r in RESOLUTIONS]

    out = decoder(feats)
    # Sum ALL four outputs -- including enc_pred_logits/enc_pred_boxes, the
    # encoder query-selection proposals. Without them in the loss,
    # enc_score_head/enc_output/enc_bbox_head would show up as dead (verified
    # during Buoc 5 development); the real training loss (Buoc 6) supervises
    # both the decoder output and this encoder proposal output, so this test
    # mirrors that instead of just the decoder's final output.
    loss = (
        out["pred_logits"].sum()
        + out["pred_boxes"].sum()
        + out["enc_pred_logits"].sum()
        + out["enc_pred_boxes"].sum()
    )
    loss.backward()

    for i, feat in enumerate(feats):
        assert feat.grad is not None, f"no gradient reached input feature scale {i}"
        assert torch.isfinite(feat.grad).all()
        assert feat.grad.abs().sum() > 0, f"gradient at input feature scale {i} is all zero"

    missing_grad = [name for name, param in decoder.named_parameters() if param.requires_grad and param.grad is None]
    assert not missing_grad, f"parameters with no gradient (dead branch?): {missing_grad}"
