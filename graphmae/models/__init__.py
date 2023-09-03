from .edcoder import PreModel


def build_model(args):
    num_heads = args.num_heads
    num_out_heads = args.num_out_heads
    num_hidden = args.num_hidden
    num_projector_hidden = args.num_projector_hidden
    num_projector = args.num_projector
    num_layers = args.num_layers
    residual = args.residual
    attn_drop = args.attn_drop
    in_drop = args.in_drop
    norm = args.norm
    negative_slope = args.negative_slope
    encoder_type = args.encoder
    decoder_type = args.decoder
    mask_rate = args.mask_rate
    temperature = args.temperature
    augmentation = args.augmentation
    drop_node_rate = args.drop_node_rate
    drop_edge_rate = args.drop_edge_rate
    drop_feature_rate = args.drop_feature_rate
    replace_rate = args.replace_rate

    activation = args.activation
    loss_fn = args.loss_fn
    loss_weight = args.loss_weight
    alpha_l = args.alpha_l
    concat_hidden = args.concat_hidden
    num_features = args.num_features
    mu = args.mu
    nu = args.nu
    aggr = args.aggr

    model = PreModel(
        in_dim=num_features,
        num_hidden=num_hidden,
        num_projector_hidden=num_projector_hidden,
        num_projector=num_projector,
        num_layers=num_layers,
        nhead=num_heads,
        nhead_out=num_out_heads,
        activation=activation,
        feat_drop=in_drop,
        attn_drop=attn_drop,
        negative_slope=negative_slope,
        residual=residual,
        encoder_type=encoder_type,
        decoder_type=decoder_type,
        mask_rate=mask_rate,
        temperature=temperature,
        norm=norm,
        loss_fn=loss_fn,
        loss_weight=loss_weight,
        mu=mu,
        nu=nu,
        augmentation=augmentation,
        drop_node_rate=drop_node_rate,
        drop_edge_rate=drop_edge_rate,
        drop_feature_rate=drop_feature_rate,
        replace_rate=replace_rate,
        alpha_l=alpha_l,
        concat_hidden=concat_hidden,
        aggr=aggr,
    )
    return model
