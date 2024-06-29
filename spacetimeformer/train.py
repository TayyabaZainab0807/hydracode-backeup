from argparse import ArgumentParser
import random
import sys
import warnings
import os
import uuid
import torchinfo
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pytorch_lightning as pl
import torch
import pickle
torch.cuda.empty_cache()
import spacetimeformer as stf
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import shutil

from contextlib import redirect_stdout



_MODELS = ["spacetimeformer", "mtgnn","bilstm", "heuristic", "lstm", "lstnet", "linear", "s4"]
 
_DSETS = [
    "asos",
    "metr-la",
    "pems-bay",
    "exchange",
    "precip",
    "toy2",
    "solar_energy",
    "syn",
    "mnist",
    "cifar",
    "copy",
    "cont_copy",
    "m4",
    "wiki",
    "ettm1",
    "weather",
    "monash",
    "hangzhou",
    "traffic",
]


def create_parser():
    model = sys.argv[1]
    dset = sys.argv[2]

    # Throw error now before we get confusing parser issues
    assert (
        model in _MODELS
    ), f"Unrecognized model (`{model}`). Options include: {_MODELS}"
    assert dset in _DSETS, f"Unrecognized dset (`{dset}`). Options include: {_DSETS}"

    parser = ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("dset")

    if dset == "precip":
        stf.data.precip.GeoDset.add_cli(parser)
        stf.data.precip.CONUS_Precip.add_cli(parser)
    elif dset == "metr-la" or dset == "pems-bay":
        stf.data.metr_la.METR_LA_Data.add_cli(parser)
    elif dset == "syn":
        stf.data.synthetic.SyntheticData.add_cli(parser)
        stf.data.CSVTorchDset.add_cli(parser)
    elif dset == "mnist":
        stf.data.image_completion.MNISTDset.add_cli(parser)
    elif dset == "cifar":
        stf.data.image_completion.CIFARDset.add_cli(parser)
    elif dset == "copy":
        stf.data.copy_task.CopyTaskDset.add_cli(parser)
    elif dset == "cont_copy":
        stf.data.cont_copy_task.ContCopyTaskDset.add_cli(parser)
    elif dset == "m4":
        stf.data.m4.M4TorchDset.add_cli(parser)
    elif dset == "wiki":
        stf.data.wiki.WikipediaTorchDset.add_cli(parser)
    elif dset == "monash":
        stf.data.monash.MonashDset.add_cli(parser)
    elif dset == "hangzhou":
        stf.data.metro.MetroData.add_cli(parser)
    else:
        stf.data.CSVTimeSeries.add_cli(parser)
        stf.data.CSVTorchDset.add_cli(parser)
        
    stf.data.DataModule.add_cli(parser)

    if model == "lstm":
        stf.lstm_model.LSTM_Forecaster.add_cli(parser)
        stf.callbacks.TeacherForcingAnnealCallback.add_cli(parser)
    elif model == "lstnet":
        stf.lstnet_model.LSTNet_Forecaster.add_cli(parser)
    elif model == "bilstm":
        stf.bilstm_model.bilstm_Forecaster.add_cli(parser)
    elif model == "mtgnn":
        stf.mtgnn_model.MTGNN_Forecaster.add_cli(parser)
    elif model == "heuristic":
        stf.heuristic_model.Heuristic_Forecaster.add_cli(parser)
    elif model == "spacetimeformer":
        stf.spacetimeformer_model.Spacetimeformer_Forecaster.add_cli(parser)
    elif model == "linear":
        stf.linear_model.Linear_Forecaster.add_cli(parser)
    elif model == "s4":
        stf.s4_model.S4_Forecaster.add_cli(parser)

    stf.callbacks.TimeMaskedLossCallback.add_cli(parser)

    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--plot_samples", type=int, default=8)
    parser.add_argument("--testing", type=int, default=0)
    parser.add_argument("--attn_plot", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--file_name", type=str, required=True)
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--col", type=str)
    parser.add_argument("--model_location", type=str)
    parser.add_argument("--accumulate", type=int, default=1)
    parser.add_argument("--val_check_interval", type=float, default=1.0)
    parser.add_argument("--limit_val_batches", type=float, default=1.0)
    parser.add_argument("--no_earlystopping", action="store_true")
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument(
        "--trials", type=int, default=1, help="How many consecutive trials to run"
    )

    if len(sys.argv) > 3 and sys.argv[3] == "-h":
        parser.print_help()
        sys.exit(0)

    return parser


def create_model(config):
    x_dim, yc_dim, yt_dim = None, None, None
    if config.dset == "metr-la":
        x_dim = 2
        yc_dim = 207
        yt_dim = 207
    elif config.dset == "pems-bay":
        x_dim = 2
        yc_dim = 325
        yt_dim = 325
    elif config.dset == "precip":
        x_dim = 2
        yc_dim = 49
        yt_dim = 49
    elif config.dset == "asos":
        x_dim = 6
        yc_dim = 1
        yt_dim = 1
    elif config.dset == "solar_energy":
        x_dim = 6
        yc_dim = 5
        yt_dim = 5
    elif config.dset == "exchange":
        x_dim = 6
        yc_dim = 8
        yt_dim = 8
    elif config.dset == "toy2":
        x_dim = 6
        yc_dim = 20
        yt_dim = 20
    elif config.dset == "syn":
        x_dim = 5
        yc_dim = 20
        yt_dim = 20
    elif config.dset == "mnist":
        x_dim = 1
        yc_dim = 28
        yt_dim = 28
    elif config.dset == "cifar":
        x_dim = 1
        yc_dim = 3
        yt_dim = 3
    elif config.dset == "copy" or config.dset == "cont_copy":
        x_dim = 1
        yc_dim = config.copy_vars
        yt_dim = config.copy_vars
    elif config.dset == "m4":
        x_dim = 4
        yc_dim = 1
        yt_dim = 1
    elif config.dset == "wiki":
        x_dim = 2
        yc_dim = 1
        yt_dim = 1
    elif config.dset == "monash":
        x_dim = 4
        yc_dim = 1
        yt_dim = 1
    elif config.dset == "ettm1":
        x_dim = 4
        yc_dim = 7
        yt_dim = 7
    elif config.dset == "weather":
        x_dim = 3
        yc_dim = 21
        yt_dim = 21
    elif config.dset == "hangzhou":
        x_dim = 4
        yc_dim = 160
        yt_dim = 160
    elif config.dset == "traffic":
        x_dim = 2
        yc_dim = 862
        yt_dim = 862
    assert x_dim is not None
    assert yc_dim is not None
    assert yt_dim is not None

    if config.model == "lstm":
        forecaster = stf.lstm_model.LSTM_Forecaster(
            # encoder
            d_x=x_dim,
            d_yc=yc_dim,
            d_yt=yt_dim,
            time_emb_dim=config.time_emb_dim,
            hidden_dim=config.hidden_dim,
            n_layers=config.n_layers,
            dropout_p=config.dropout_p,
            # training
            learning_rate=config.learning_rate,
            teacher_forcing_prob=config.teacher_forcing_start,
            l2_coeff=config.l2_coeff,
            loss=config.loss,
            linear_window=config.linear_window,
            use_revin=config.use_revin,
            linear_shared_weights=config.linear_shared_weights,
            use_seasonal_decomp=config.use_seasonal_decomp,
        )
        
        

    elif config.model == "heuristic":
        forecaster = stf.heuristic_model.Heuristic_Forecaster(
            d_x=x_dim,
            d_yc=yc_dim,
            d_yt=yt_dim,
            context_points=config.context_points,
            target_points=config.target_points,
            loss=config.loss,
            method=config.method,
        )
    elif config.model == "mtgnn":
        forecaster = stf.mtgnn_model.MTGNN_Forecaster(
            d_x=x_dim,
            d_yc=yc_dim,
            d_yt=yt_dim,
            context_points=config.context_points,
            target_points=config.target_points,
            gcn_depth=config.gcn_depth,
            dropout_p=config.dropout_p,
            node_dim=config.node_dim,
            dilation_exponential=config.dilation_exponential,
            conv_channels=config.conv_channels,
            subgraph_size=config.subgraph_size,
            skip_channels=config.skip_channels,
            end_channels=config.end_channels,
            residual_channels=config.residual_channels,
            layers=config.layers,
            propalpha=config.propalpha,
            tanhalpha=config.tanhalpha,
            learning_rate=config.learning_rate,
            kernel_size=config.kernel_size,
            l2_coeff=config.l2_coeff,
            time_emb_dim=config.time_emb_dim,
            loss=config.loss,
            linear_window=config.linear_window,
            linear_shared_weights=config.linear_shared_weights,
            use_seasonal_decomp=config.use_seasonal_decomp,
            use_revin=config.use_revin,
        )
    elif config.model == "bilstm":
        forecaster = stf.bilstm_model.bilstm_Forecaster(
            d_x=x_dim,
            d_yc=yc_dim,
            d_yt=yt_dim,
            context_points=config.context_points,
            hidRNN=config.hidRNN,
            hidCNN=config.hidCNN,
            CNN_kernel=config.CNN_kernel,
            dropout_p=config.dropout_p,
            output_fun=config.output_fun,
            learning_rate=config.learning_rate,
            l2_coeff=config.l2_coeff,
            loss=config.loss,
            linear_window=config.linear_window,
            use_revin=config.use_revin,
        )
    elif config.model == "lstnet":
        forecaster = stf.lstnet_model.LSTNet_Forecaster(
            d_x=x_dim,
            d_yc=yc_dim,
            d_yt=yt_dim,
            context_points=config.context_points,
            hidRNN=config.hidRNN,
            hidCNN=config.hidCNN,
            hidSkip=config.hidSkip,
            CNN_kernel=config.CNN_kernel,
            skip=config.skip,
            dropout_p=config.dropout_p,
            output_fun=config.output_fun,
            learning_rate=config.learning_rate,
            l2_coeff=config.l2_coeff,
            loss=config.loss,
            linear_window=config.linear_window,
            use_revin=config.use_revin,
        )
        
        print(x_dim)
        print(yc_dim)
        print(yt_dim)
        print(config.output_fun)
        print(config.hidCNN)
        print(config.hidRNN)
    elif config.model == "spacetimeformer":
        if hasattr(config, "context_points") and hasattr(config, "target_points"):
            max_seq_len = config.context_points + config.target_points
        elif hasattr(config, "max_len"):
            max_seq_len = config.max_len
        else:
            raise ValueError("Undefined max_seq_len")
        forecaster = stf.spacetimeformer_model.Spacetimeformer_Forecaster(
            d_x=x_dim,
            d_yc=yc_dim,
            d_yt=yt_dim,
            max_seq_len=max_seq_len,
            start_token_len=config.start_token_len,
            attn_factor=config.attn_factor,
            d_model=config.d_model,
            d_queries_keys=config.d_qk,
            d_values=config.d_v,
            n_heads=config.n_heads,
            e_layers=config.enc_layers,
            d_layers=config.dec_layers,
            d_ff=config.d_ff,
            dropout_emb=config.dropout_emb,
            dropout_attn_out=config.dropout_attn_out,
            dropout_attn_matrix=config.dropout_attn_matrix,
            dropout_qkv=config.dropout_qkv,
            dropout_ff=config.dropout_ff,
            pos_emb_type=config.pos_emb_type,
            use_final_norm=not config.no_final_norm,
            global_self_attn=config.global_self_attn,
            local_self_attn=config.local_self_attn,
            global_cross_attn=config.global_cross_attn,
            local_cross_attn=config.local_cross_attn,
            performer_kernel=config.performer_kernel,
            performer_redraw_interval=config.performer_redraw_interval,
            attn_time_windows=config.attn_time_windows,
            use_shifted_time_windows=config.use_shifted_time_windows,
            norm=config.norm,
            activation=config.activation,
            init_lr=config.init_lr,
            base_lr=config.base_lr,
            warmup_steps=config.warmup_steps,
            decay_factor=config.decay_factor,
            initial_downsample_convs=config.initial_downsample_convs,
            intermediate_downsample_convs=config.intermediate_downsample_convs,
            embed_method=config.embed_method,
            l2_coeff=config.l2_coeff,
            loss=config.loss,
            class_loss_imp=config.class_loss_imp,
            recon_loss_imp=config.recon_loss_imp,
            time_emb_dim=config.time_emb_dim,
            null_value=config.null_value,
            pad_value=config.pad_value,
            linear_window=config.linear_window,
            use_revin=config.use_revin,
            linear_shared_weights=config.linear_shared_weights,
            use_seasonal_decomp=config.use_seasonal_decomp,
            use_val=not config.no_val,
            use_time=not config.no_time,
            use_space=not config.no_space,
            use_given=not config.no_given,
            recon_mask_skip_all=config.recon_mask_skip_all,
            recon_mask_max_seq_len=config.recon_mask_max_seq_len,
            recon_mask_drop_seq=config.recon_mask_drop_seq,
            recon_mask_drop_standard=config.recon_mask_drop_standard,
            recon_mask_drop_full=config.recon_mask_drop_full,
        )
    elif config.model == "linear":
        forecaster = stf.linear_model.Linear_Forecaster(
            d_x=x_dim,
            d_yc=yc_dim,
            d_yt=yt_dim,
            context_points=config.context_points,
            learning_rate=config.learning_rate,
            l2_coeff=config.l2_coeff,
            loss=config.loss,
            linear_window=config.linear_window,
            linear_shared_weights=config.linear_shared_weights,
            use_revin=config.use_revin,
            use_seasonal_decomp=config.use_seasonal_decomp,
        )
    elif config.model == "s4":
        forecaster = stf.s4_model.S4_Forecaster(
            context_points=config.context_points,
            target_points=config.target_points,
            d_state=config.d_state,
            d_model=config.d_model,
            d_x=x_dim,
            d_yc=yc_dim,
            d_yt=yt_dim,
            layers=config.layers,
            time_emb_dim=config.time_emb_dim,
            channels=config.channels,
            dropout_p=config.dropout_p,
            learning_rate=config.learning_rate,
            l2_coeff=config.l2_coeff,
            loss=config.loss,
            linear_window=config.linear_window,
            linear_shared_weights=config.linear_shared_weights,
            use_revin=config.use_revin,
            use_seasonal_decomp=config.use_seasonal_decomp,
        )

    return forecaster


def create_dset(config):
    INV_SCALER = lambda x: x
    SCALER = lambda x: x
    NULL_VAL = None
    PLOT_VAR_IDXS = None
    PLOT_VAR_NAMES = None
    PAD_VAL = None

    if config.dset == "metr-la" or config.dset == "pems-bay":
        if config.dset == "pems-bay":
            assert (
                "pems_bay" in config.data_path
            ), "Make sure to switch to the pems-bay file!"
        data = stf.data.metr_la.METR_LA_Data(config.data_path)
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.metr_la.METR_LA_Torch,
            dataset_kwargs={"data": data},
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = data.inverse_scale
        SCALER = data.scale
        NULL_VAL = 0.0

    elif config.dset == "hangzhou":
        data = stf.data.metro.MetroData(config.data_path)
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.metro.MetroTorch,
            dataset_kwargs={"data": data},
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = data.inverse_scale
        SCALER = data.scale
        NULL_VAL = 0.0

    elif config.dset == "precip":
        dset = stf.data.precip.GeoDset(dset_dir=config.dset_dir, var="precip")
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.precip.CONUS_Precip,
            dataset_kwargs={
                "dset": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        NULL_VAL = -1.0
    elif config.dset == "syn":
        dset = stf.data.synthetic.SyntheticData(config.data_path)
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
    elif config.dset in ["mnist", "cifar"]:
        if config.dset == "mnist":
            config.target_points = 28 - config.context_points
            datasetCls = stf.data.image_completion.MNISTDset
            PLOT_VAR_IDXS = [18, 24]
            PLOT_VAR_NAMES = ["18th row", "24th row"]
        else:
            config.target_points = 32 * 32 - config.context_points
            datasetCls = stf.data.image_completion.CIFARDset
            PLOT_VAR_IDXS = [0]
            PLOT_VAR_NAMES = ["Reds"]
        DATA_MODULE = stf.data.DataModule(
            datasetCls=datasetCls,
            dataset_kwargs={"context_points": config.context_points},
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
    elif config.dset == "copy":
        # set these manually in case the model needs them
        config.context_points = config.copy_length + int(
            config.copy_include_lags
        )  # seq + lags
        config.target_points = config.copy_length
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.copy_task.CopyTaskDset,
            dataset_kwargs={
                "length": config.copy_length,
                "copy_vars": config.copy_vars,
                "lags": config.copy_lags,
                "mask_prob": config.copy_mask_prob,
                "include_lags": config.copy_include_lags,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
    elif config.dset == "cont_copy":
        # set these manually in case the model needs them
        config.context_points = config.copy_length + int(
            config.copy_include_lags
        )  # seq + lags
        config.target_points = config.copy_length
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.cont_copy_task.ContCopyTaskDset,
            dataset_kwargs={
                "length": config.copy_length,
                "copy_vars": config.copy_vars,
                "lags": config.copy_lags,
                "include_lags": config.copy_include_lags,
                "magnitude_matters": config.copy_mag_matters,
                "freq_shift": config.copy_freq_shift,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
    elif config.dset == "m4":
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.m4.M4TorchDset,
            dataset_kwargs={
                "data_path": config.data_path,
                "resolutions": args.resolutions,
                "max_len": args.max_len,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            collate_fn=stf.data.m4.pad_m4_collate,
            overfit=args.overfit,
        )
        NULL_VAL = -1.0
        PAD_VAL = -1.0

    elif config.dset == "wiki":
        DATA_MODULE = stf.data.DataModule(
            stf.data.wiki.WikipediaTorchDset,
            dataset_kwargs={
                "data_path": config.data_path,
                "forecast_duration": args.forecast_duration,
                "max_len": args.max_len,
            },
            batch_size=args.batch_size,
            workers=args.workers,
            collate_fn=stf.data.wiki.pad_wiki_collate,
            overfit=args.overfit,
        )
        NULL_VAL = -1.0
        PAD_VAL = -1.0
        SCALER = stf.data.wiki.WikipediaTorchDset.scale
        INV_SCALER = stf.data.wiki.WikipediaTorchDset.inverse_scale
    elif config.dset == "monash":
        root_dir = config.root_dir
        DATA_MODULE = stf.data.monash.monash_dloader.make_monash_dmodule(
            root_dir=root_dir,
            max_len=config.max_len,
            include=config.include,
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=config.overfit,
        )
        NULL_VAL = -64.0
        PAD_VAL = -64.0
    elif config.dset == "ettm1":
        target_cols = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
        dset = stf.data.CSVTimeSeries(
            data_path=config.data_path,
            target_cols=target_cols,
            ignore_cols=[],
            val_split=4.0 / 20,  # from informer
            test_split=4.0 / 20,  # from informer
            time_col_name="date",
            time_features=["month", "day", "weekday", "hour"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
        # PAD_VAL = -32.0
        PLOT_VAR_NAMES = target_cols
        PLOT_VAR_IDXS = [i for i in range(len(target_cols))]
    elif config.dset == "weather":
        data_path = config.data_path
        dset = stf.data.CSVTimeSeries(
            data_path=config.data_path,
            target_cols=[],
            ignore_cols=[],
            # paper says 7:1:2 split
            val_split=1.0 / 10,
            test_split=2.0 / 10,
            time_col_name="date",
            time_features=["day", "hour", "minute"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
        PLOT_VAR_NAMES = ["OT", "p (mbar)", "raining (s)"]
        PLOT_VAR_IDXS = [20, 0, 15]
    else:
        time_col_name = "Datetime"
        data_path = config.data_path
        time_features = ["year", "month", "day", "weekday", "hour", "minute"]
        if config.dset == "asos":
            if data_path == "auto":
                data_path = "./data/temperature-v1.csv"
            #target_cols = ["ABI", "AMA", "ACT", "ALB", "JFK", "LGA"]
            target_cols =[config.col]
        elif config.dset == "solar_energy":
            if data_path == "auto":
                data_path = "./data/solar_AL_converted.csv"
            target_cols = ["3","8", "14", "19", "22"]
            #target_cols = [str(i) for i in range(137)]
        elif "toy" in config.dset:
            if data_path == "auto":
                if config.dset == "toy2":
                    data_path = "./data/toy_dset2.csv"
                else:
                    raise ValueError(f"Unrecognized toy dataset {config.dset}")
            target_cols = [f"D{i}" for i in range(1, 21)]
        elif config.dset == "exchange":
            if data_path == "auto":
                data_path = "./data/exchange_rate_converted.csv"
            target_cols = [
                "Australia",
                "United Kingdom",
                "Canada",
                "Switzerland",
                "China",
                "Japan",
                "New Zealand",
                "Singapore",
            ]
        elif config.dset == "traffic":
            if data_path == "auto":
                data_path = "./data/traffic.csv"
            target_cols = [f"Lane {i}" for i in range(862)]
            time_col_name = "FakeTime"
            time_features = ["month", "day"]

        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols="all",
            time_col_name=time_col_name,
            time_features=time_features,
            val_split=0.2,
            test_split=0.2,
        )

        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None

    return (
        DATA_MODULE,
        INV_SCALER,
        SCALER,
        NULL_VAL,
        PLOT_VAR_IDXS,
        PLOT_VAR_NAMES,
        PAD_VAL,
    )


def create_callbacks(config, save_dir):
    
    filename = f"{config.run_name}_" + str(uuid.uuid1()).split("-")[0]
    model_ckpt_dir = os.path.join(save_dir, filename)
    config.model_ckpt_dir = model_ckpt_dir
    saving = pl.callbacks.ModelCheckpoint(
        dirpath=model_ckpt_dir,
        monitor="val/loss",
        mode="min",
        filename=f"{config.run_name}" + "{epoch:02d}",
        save_top_k=1,
        auto_insert_metric_name=True,
    )
    callbacks = [saving]

    if not config.no_earlystopping:
        callbacks.append(
            pl.callbacks.early_stopping.EarlyStopping(
                monitor="val/loss",
                patience=config.patience,
            )
        )

    if config.wandb:
        callbacks.append(pl.callbacks.LearningRateMonitor())

    if config.model == "lstm":
        callbacks.append(
            stf.callbacks.TeacherForcingAnnealCallback(
                start=config.teacher_forcing_start,
                end=config.teacher_forcing_end,
                steps=config.teacher_forcing_anneal_steps,
            )
        )
    if config.time_mask_loss:
        callbacks.append(
            stf.callbacks.TimeMaskedLossCallback(
                start=config.time_mask_start,
                end=config.time_mask_end,
                steps=config.time_mask_anneal_steps,
            )
        )
    return callbacks


def main(args):
    #when you want to load a model and only test it set this load_test variable data as 1
    load_test=args.testing
    log_dir = os.getenv("STF_LOG_DIR")
    if log_dir is None:
        log_dir = "./data/STF_LOG_DIR"
        print("Using default wandb log dir path of ./data/STF_LOG_DIR. This can be adjusted with the environment variable `STF_LOG_DIR`" )
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if args.wandb:
        import wandb

        project = os.getenv("STF_WANDB_PROJ")
        entity = os.getenv("STF_WANDB_ACCT")
        assert (
            project is not None and entity is not None
        ), "Please set environment variables `STF_WANDB_ACCT` and `STF_WANDB_PROJ` with \n\
            your wandb user/organization name and project title, respectively."
        experiment = wandb.init(
            project=project,
            entity=entity,
            config=args,
            dir=log_dir,
            reinit=True,
        )
        config = wandb.config
        wandb.run.name = args.run_name
        wandb.run.save()
        logger = pl.loggers.WandbLogger(
            experiment=experiment,
            save_dir=log_dir,
        )

    # Dset
    (
        data_module,
        inv_scaler,
        scaler,
        null_val,
        plot_var_idxs,
        plot_var_names,
        pad_val,
    ) = create_dset(args)

    # Model
    args.null_value = null_val
    args.pad_value = pad_val
    forecaster = create_model(args)
    if (load_test==1 or load_test==3):
        #path to the saved model, you want to test
        #checkpoint_path = "data/STF_LOG_DIR/transformer_48_al_solar_1b4ab592/transformer_48_al_solarepoch=04.ckpt"
        checkpoint_path = args.model_location
        checkpoint_path = "data/STF_LOG_DIR/ettm1-spatiotemporal_f7d658c8/ettm1-spatiotemporalepoch=06.ckpt"
        #checkpoint_path = "data/STF_LOG_DIR/lstnet_my_3_withlinearwindow_paper_al_solar_bf48055e/lstnet_my_3_withlinearwindow_paper_al_solarepoch=38.ckpt"
        #checkpoint_path = "data/STF_LOG_DIR/lstnet_my_6_withlinearwindow_paper_al_solar_bf27e706/lstnet_my_6_withlinearwindow_paper_al_solarepoch=20.ckpt"
        #checkpoint_path = "data/STF_LOG_DIR/lstnet_my_12_withlinearwindow_paper_al_solar_bf2d3742/lstnet_my_12_withlinearwindow_paper_al_solarepoch=11.ckpt"
        #checkpoint_path = "data/STF_LOG_DIR/lstnet_my_24_withlinearwindow_paper_al_solar_bf78de4a/lstnet_my_24_withlinearwindow_paper_al_solarepoch=05.ckpt"
        #checkpoint_path = "data/STF_LOG_DIR/lstnet_my_48_withlinearwindow_paper_al_solar_bf1d8d6a/lstnet_my_48_withlinearwindow_paper_al_solarepoch=05.ckpt"
        #checkpoint_path = "data/STF_LOG_DIR/lstnet_my_96_withlinearwindow_paper_al_solar_bf2e389a/lstnet_my_96_withlinearwindow_paper_al_solarepoch=00.ckpt"
        forecaster = forecaster.load_from_checkpoint(checkpoint_path=checkpoint_path, map_location=torch.device('cuda'))
        forecaster.eval()
 
    forecaster.set_inv_scaler(inv_scaler)
    forecaster.set_scaler(scaler)
    forecaster.set_null_value(null_val)

    # Callbacks
    callbacks = create_callbacks(args, save_dir=log_dir)
    test_samples = next(iter(data_module.test_dataloader()))

    if args.wandb and args.plot:
        callbacks.append(
            stf.plot.PredictionPlotterCallback(
                test_samples,
                var_idxs=plot_var_idxs,
                var_names=plot_var_names,
                pad_val=pad_val,
                total_samples=min(args.plot_samples, args.batch_size),
            )
        )

    if args.wandb and args.dset in ["mnist", "cifar"] and args.plot:
        callbacks.append(
            stf.plot.ImageCompletionCallback(
                test_samples,
                total_samples=min(16, args.batch_size),
                mode="left-right" if config.dset == "mnist" else "flat",
            )
        )

    if args.wandb and args.dset == "copy" and args.plot:
        callbacks.append(
            stf.plot.CopyTaskCallback(
                test_samples,
                total_samples=min(16, args.batch_size),
            )
        )

    if args.wandb and args.model == "spacetimeformer" and args.attn_plot:

        callbacks.append(
            stf.plot.AttentionMatrixCallback(
                test_samples,
                layer=0,
                total_samples=min(16, args.batch_size),
            )
        )

    if args.wandb:
        config.update(args)
        logger.log_hyperparams(config)

    if args.val_check_interval <= 1.0:
        val_control = {"val_check_interval": args.val_check_interval}
    else:
        val_control = {"check_val_every_n_epoch": int(args.val_check_interval)}

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=args.gpus,
        strategy="dp",
        callbacks=callbacks,
        logger=logger if args.wandb else None,
        gradient_clip_val=args.grad_clip_norm,
        gradient_clip_algorithm="norm",
        overfit_batches=20 if args.debug else 0,
        accumulate_grad_batches=args.accumulate,
        sync_batchnorm=True,
        limit_val_batches=args.limit_val_batches,
        max_epochs=10,
        **val_control,
    )

    
    #model_info = torchinfo.summary(forecaster)
    with open(args.file_name, 'a') as f:
        with redirect_stdout(f):
            model_info = torchinfo.summary(forecaster)

    
    
    # Train
    #torch.cuda.set_per_process_memory_fraction()
    if(load_test==0):
        trainer.fit(forecaster, datamodule=data_module)
        trainer.test(datamodule=data_module, ckpt_path="best")
        with open(args.file_name, 'a') as f:
            with redirect_stdout(f):
                print("Test/loss", str(trainer.logged_metrics["test/loss"].item()))
                print("Test/mae", str(trainer.logged_metrics["test/mae"].item()))
                print("Test/mse", str(trainer.logged_metrics["test/mse"].item()))
                print("Test/mape", str(trainer.logged_metrics["test/mape"].item()))
                print("Test/norm_mae", str(trainer.logged_metrics["test/norm_mae"].item()))
                print("Test/norm_mse", str(trainer.logged_metrics["test/norm_mse"].item()))
                print("Test/smape", str(trainer.logged_metrics["test/smape"].item()))
                print("******************************************************************************************")
                print("******************************************************************************************")




    elif(load_test==1):
        #testing prediction on prediction with a step of 1

        #trainer.test(forecaster, datamodule=data_module)
        #print("Test/mae", str(trainer.logged_metrics["test/mae"].item()))
        #print("Test/mse", str(trainer.logged_metrics["test/mse"].item()))   

        process_batches(forecaster, data_module, prediction_on_prediction=True, save_dir='results_pred')
        plot_and_save(save_dir='results_pred', filename='predicted_vs_actual_pred_pred.png')

        process_batches(forecaster, data_module, prediction_on_prediction=False, save_dir='results_actual')
        plot_and_save(save_dir='results_actual', filename='predicted_vs_actual_data_pred.png')

        

    elif(load_test == 2):
        #this plot is to see if the starting predictions are good compared to the end prediction of a prediction window. and creat a cloud plot
        prediction_length = pred_length = args.target_points

        mae_values = [[] for _ in range(prediction_length)]
        max_values = np.zeros(prediction_length)
        all_yt_pred = [[] for _ in range(prediction_length)]
        test_dataloader = data_module.test_dataloader()

        for batch_idx, batch in tqdm(enumerate(test_dataloader), desc="Processing batches"):
            xc, yc, xt, yt = batch
            yt_pred, yt_lab = forecaster.predict(xc, yc, xt, yt)

            for i in range(prediction_length):
                # Calculate MAE for each prediction point across all features
                mae = np.abs(yt[:, i, :] - yt_pred[:, i, :]).mean(axis=0).mean()
                mae_values[i].append(mae)
                
                # Update maximum value for each prediction point
                max_values[i] = max(max_values[i], yt_pred[:, i, :].max())
                
                # Collect all predictions for cloud plot
                all_yt_pred[i].extend(yt_pred[:, i, :].flatten())

            if batch_idx == 1000:
                break

        # Calculate mean and standard deviation for each prediction point
        mean_mae = [np.mean(maes) for maes in mae_values]
        std_mae = [np.std(maes) for maes in mae_values]

        # Plot cloud plot
        plt.figure(figsize=(10, 6))
        for i in range(prediction_length):
            plt.scatter([i] * len(all_yt_pred[i]), all_yt_pred[i], alpha=0.1, color='blue')

        # Plot mean MAE with error bars representing the standard deviation
        plt.errorbar(range(prediction_length), mean_mae, yerr=std_mae, fmt='-o', capsize=5, color='green', label='Mean MAE')

        # Plot maximum predicted value
        plt.plot(range(prediction_length), max_values, 'r--', label='Max Predicted Value')
        
        plt.xlabel('Prediction Point')
        plt.ylabel('Value')
        plt.title('Cloud Plot with MAE and Maximum Predicted Value for Each Prediction Point')
        plt.legend()
        plt.grid(True)
        plt.savefig('cloud_plot_with_mae_and_max.png', dpi=300, bbox_inches='tight')
        plt.close() #adding a git commit
        
 
    elif(load_test == 3):
        #process_batches(forecaster, data_module, prediction_on_prediction=False, save_dir='results_actual')
        #plot_and_save(save_dir='results_actual', filename='predicted_vs_actual_data_pred.png')
        
        pred_length = args.target_points
        #process_replace(forecaster, data_module, prediction_on_prediction=False, step_size=1, save_dir='results_actual', pred_length=pred_length,replace_data=False)
        #plot_and_save(save_dir='results_actual', filename='predicted_vs_actual_pred_pred.png')

        #process_replace(forecaster, data_module, prediction_on_prediction=True, step_size=1, save_dir='results_actual', pred_length=pred_length,replace_data=False)
        #plot_and_save(save_dir='results_actual', filename='predicted_vs_actual_pred_pred.png')

        process_replace(forecaster, data_module, prediction_on_prediction=False, step_size=pred_length, save_dir='results_actual', pred_length=pred_length,replace_data=False)
        plot_and_save(save_dir='results_actual', filename='predicted_vs_actual_pred_pred.png')

        #process_replace(forecaster, data_module, prediction_on_prediction=True, step_size=pred_length, save_dir='results_actual', pred_length=pred_length,replace_data=False)
        #plot_and_save(save_dir='results_actual', filename='predicted_vs_actual_pred_pred.png')

        process_replace(forecaster, data_module, prediction_on_prediction=True, step_size=pred_length, save_dir='results_actual', pred_length=pred_length,replace_data=True)
        plot_and_save(save_dir='results_actual', filename='predicted_vs_actual_pred_pred.png')


   
 
    else:
        
        #just doing prediction on actial data without training
        all_yt_pred = []
        all_yt_lab = []
        test_dataloader = data_module.test_dataloader()
        pred_length = args.target_points
        print("Testing for model_",str(pred_length)) 
        # Perform predictions outside the loop and append them to the lists
        for batch_idx, batch in tqdm(enumerate(test_dataloader),desc="Processing batches"):
            xc, yc, xt, yt = batch
            yt_pred, = forecaster(xc, yc, xt, yt)
            all_yt_pred.append(yt_pred)
            all_yt_lab.append(yt)
            #scaled values

        
        #all_yt_pred = np.concatenate(all_yt_pred)
        #all_yt_lab = np.concatenate(all_yt_lab)
        #print(all_yt_lab.shape)
        # Calculate mean absolute error
        #avg_abs_error = mean_absolute_error(all_yt_lab.flatten(), all_yt_pred.flatten())
        if(args.dset == "ettm1"):
            eval_ettm1(all_yt_pred,all_yt_lab, pred_length)

        
        if(args.dset == "solar_energy"):
            eval_solar_energy(all_yt_pred,all_yt_lab, pred_length)
            #Testing abs variation with relative error as 5%
            


    if args.wandb:
        experiment.finish()




def save_to_disk(data, filename):
    np.save(filename, data)

def process_batches(forecaster, data_module, prediction_on_prediction=False, save_dir='results'):
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    
    # Create a new directory
    os.makedirs(save_dir)

    prev_output = None

    xc=0 
    yc=0 
    xt=0 
    yt=0

    test_loader = data_module.test_dataloader()
    for batch_idx, batch in tqdm(enumerate(test_loader), desc="Processing batches"):
        if prediction_on_prediction and (batch_idx != 0):
            xc, _, xt, yt = batch
            yc = yc[:, 1:, :]
            yc = torch.cat([yc, prev_output], dim=1)
        else:
            xc, yc, xt, yt = batch

        with torch.no_grad():
            yt_pred = forecaster(xc, yc, xt, yt)
            yt_pred = yt_pred[0]
            prev_output = yt_pred[:, :1, :]

        # Save predictions and labels to disk
        
        yt_pred_unscaled = forecaster._inv_scaler(yt_pred)
        yt_unscaled = forecaster._inv_scaler(yt)
        
        

        save_to_disk(yt_pred_unscaled.detach().cpu().numpy(), os.path.join(save_dir, f'yt_pred_{batch_idx}.npy'))
        save_to_disk(yt_unscaled.detach().cpu().numpy(), os.path.join(save_dir, f'yt_lab_{batch_idx}.npy'))

        del yt_pred
        torch.cuda.empty_cache()

        if batch_idx >= 100:
            break


def is_slice_contained(slice1, slice2):
    # Convert slices to numpy arrays for comparison
    slice1_np = slice1.numpy() if isinstance(slice1, torch.Tensor) else slice1
    slice2_np = slice2.numpy() if isinstance(slice2, torch.Tensor) else slice2
    
    # Get shapes of slices
    shape1 = slice1_np.shape
    shape2 = slice2_np.shape

    # Check if slice2 can fit within slice1
    if shape2[1] > shape1[1]:
        return False, None

    # Iterate through possible subarrays in slice1
    for i in range(shape1[1] - shape2[1] + 1):
        sub_slice1 = slice1_np[:, i:i + shape2[1], :]
        
        # Check if sub_slice1 matches slice2
        if np.allclose(sub_slice1, slice2_np, atol=1e-3):
            return True, i

    return False, None

def find_and_replace(xc, yc, data_0,data_1, atol=1e-4):
    
    batch_size, seq_len, feature_size = xc.size()
    print("batch_size",batch_size)
    print("seq_len",seq_len)
    print("feature_size",feature_size)

    print("xc",xc)
    data_len = data_0.size(1)

    print("data_len",data_len)
    print("data_0",data_0)

    found, index = is_slice_contained(xc,data_0)
    print(found)
    if(found==True):
        print("before yc", yc)
        yc[:, index:index+data_len, :] = data_1 
        print("after yc", yc)
            
    return yc, found



def process_replace(forecaster, data_module, prediction_on_prediction=False, step_size=24, save_dir='results', pred_length=24,replace_data=True):

    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)

    os.makedirs(save_dir)

    file_path = '/home/tza/spacetimeformer/spacetimeformer/data/stored_data_datasetettm1_col-1_pred_len12_abs1.7_rel0.0_time4.pkl'
    loaded_data = []
    pickel_index=0

    # Load data from pickle file
    with open(file_path, 'rb') as f:
        try:
            while True:
                data = pickle.load(f)
                loaded_data.append(data)
        except EOFError:
            pass
    len_yt_pred=pred_length
    prev_output=None
    test_loader = data_module.test_dataloader()
    for batch_idx, batch in tqdm(enumerate(test_loader), desc="Processing batches"):
        if (batch_idx % step_size) != 0:  # my step size is prediction length
            continue
        if(prediction_on_prediction):
            if batch_idx == 0:
                xc, yc, xt, yt = batch  # for the first batch use data
            else:
                xc, _, xt, yt = batch
                yc=yc[:, len_yt_pred:, :]
                yc = torch.cat((yc, prev_output), dim=1)# we are doing predictions on predictions
        else:
            xc, yc, xt, yt = batch
        
        print("scaled",forecaster._inv_scaler(xc))

        # Check and replace yc if data[0] is contained in xc
        if(replace_data):
            yc, status =find_and_replace(xc, yc, loaded_data[pickel_index][0], loaded_data[pickel_index][1])
            if status==True:
                pickel_index+=1
        with torch.no_grad():
            yt_pred = forecaster(xc, yc, xt, yt)
            prev_output = yt_pred[0]

        # Save predictions and labels to disk
        
        yt_pred_unscaled = forecaster._inv_scaler(prev_output)
        yt_unscaled = forecaster._inv_scaler(yt)

        save_to_disk(yt_pred_unscaled.detach().cpu().numpy(), os.path.join(save_dir, f'yt_pred_{batch_idx}.npy'))
        save_to_disk(yt_unscaled.detach().cpu().numpy(), os.path.join(save_dir, f'yt_lab_{batch_idx}.npy'))

        del yt_pred
        torch.cuda.empty_cache()


        if batch_idx >= 1000:
            break



import matplotlib.pyplot as plt
import torch

def load_from_disk(filename):
    return np.load(filename)

def plot_and_save(save_dir, filename):
    all_yt_pred = []
    all_yt_lab = []

    # Load all the saved numpy files
    for file in sorted(os.listdir(save_dir)):
        if file.startswith('yt_pred_'):
            all_yt_pred.append(load_from_disk(os.path.join(save_dir, file)))
        elif file.startswith('yt_lab_'):
            all_yt_lab.append(load_from_disk(os.path.join(save_dir, file)))



    all_yt_pred_concat = np.concatenate(all_yt_pred, axis=0)
    all_yt_lab_concat = np.concatenate(all_yt_lab, axis=0)

    mae = mean_absolute_error(all_yt_lab_concat.flatten(), all_yt_pred_concat.flatten())
    print("Mean Absolute Error (MAE):", mae)

    # Compute Mean Squared Error (MSE)
    mse = mean_squared_error(all_yt_lab_concat.flatten(), all_yt_pred_concat.flatten())
    print("Mean Squared Error (MSE):", mse)

    '''
    print("Shape of concatenated predictions:", all_yt_pred_concat.shape)
    print("Shape of concatenated labels:", all_yt_lab_concat.shape)

    num_features = all_yt_pred_concat.shape[2]
    num_plots = 5  # Set to num_features if you want to plot all features
    
    # Compute the mean along the sequence dimension
    all_yt_pred_mean = np.mean(all_yt_pred_concat, axis=1)
    all_yt_lab_mean = np.mean(all_yt_lab_concat, axis=1)


    
    # Plotting
    plt.figure(figsize=(10, 5 * num_plots))
    for i in range(num_plots):
        plt.subplot(num_plots, 1, i + 1)
        plt.plot(all_yt_pred_mean[:, i], label='Predicted')
        plt.plot(all_yt_lab_mean[:, i], label='Actual')
        plt.xlabel('Time/Sequence Length')
        plt.ylabel('Value')
        plt.title(f'Feature {i + 1}: Predicted vs Actual')
        plt.legend()
    
    # Adjust layout to prevent overlapping of subplots
    plt.tight_layout()
    
    # Save the plot as PNG
    plt.savefig(filename)
    plt.close()'''


def count_consecutive_ones(binary_array, t):
    num_rows, num_cols = binary_array.shape
    consecutive_counts = np.zeros((num_cols,), dtype=int)

    for col in range(num_cols):
        consecutive_count = 0
        for row in range(num_rows - 1 ): # put - 1 here as we dont want to include last row as if there is a default msg sent after pred_length, need to test
            if binary_array[row, col] == 1:
                consecutive_count += 1
            else:
                consecutive_count = 0

            if consecutive_count == t  :
                # Check if there are at least t remaining rows below the current row
                consecutive_counts[col] += 1
                consecutive_count=0

    return consecutive_counts



def eval_ettm1(all_yt_pred,all_yt_lab, pred_len):
    best_case=0
    total_values=0
    abs = [0.5,1,1.5,2,2.5]
    rel = 0.05
    t=1
    output_abs = [None] * len(abs)
    msgs = [None] * len(abs)
    for i in range(len(abs)):
        output_abs[i],best_case,total_values,msgs[i] = evaluate_alsolar(abs[i],rel,t, all_yt_pred, all_yt_lab)
    print("If we only communicate after a wrong prediction")
    result_accuracy = [(value /total_values)*100 for value in output_abs]
    result_packets = [((value +(total_values/pred_len)) / total_values) *100 for value in msgs]
    print("total predictions: ",str(total_values))
    print("Best case msgs generated (%): ", str ((best_case/total_values)*100))
    print("How bad the predictions are (%):")
    print(result_accuracy)
    print("massages generated (%):") 
    print(result_packets)
    best_case=0
    total_values=0

    rel = [0.01, 0.03, 0.05, 0.10, 0.25]
    abs = 1.5
    t=1
    output_rel = [None] * len(rel)
    for i in range(len(rel)):
        output_rel[i],best_case,total_values,msgs[i] = evaluate_alsolar(abs,rel[i],t, all_yt_pred, all_yt_lab)
    print("If we only communicate after a wrong prediction")
    result_accuracy=[((value / total_values))*100 for value in output_rel]
    result_packets = [((value +(total_values/pred_len)) / total_values) *100 for value in msgs]
    print("Best case msgs generated (%): ", str ((best_case/total_values)*100))
    print("How bad the predictions are (%):")
    print(result_accuracy)
    print("massages generated (%):") 
    print(result_packets)
    #Print("If we communicate after a wrong prediction + every prediction length") #this does not consider the intersection so currently removing this.
    #print((output_rel+best_case)/total_values)
    
    
    #Testing time- tolerance for 1= no tolerence to 5= 5 consective wrong predictions%
    best_case=0
    total_values=0
    t = [1,2,3,4,5]
    abs = 1.5
    rel=0.05
    
    output_time = [None] * len(t)
    for i in range(len(t)):
        a,b,total_values,output_time[i] = evaluate_alsolar(abs,rel,t[i], all_yt_pred, all_yt_lab)
    print("If we only communicate after t consective wrong prediction")
    result_packets=[((value +(total_values/pred_len)) / total_values) *100 for value in output_time]
    
    print("massages generated (%):") 
    print(result_packets)
    
    


def eval_solar_energy(all_yt_pred,all_yt_lab, pred_len):
    best_case=0
    total_values=0
    abs = [1,1.5,2.5,3.5,4]
    rel = 0.05
    t=1
    output_abs = [None] * len(abs)
    msgs = [None] * len(abs)
    for i in range(len(abs)):
        output_abs[i],best_case,total_values,msgs[i] = evaluate_alsolar(abs[i],rel,t, all_yt_pred, all_yt_lab)
    print("If we only communicate after a wrong prediction")
    result_accuracy = [(value /total_values)*100 for value in output_abs]
    result_packets = [((value +(total_values/pred_len)) / total_values) *100 for value in msgs]
    print("total predictions: ",str(total_values))
    print("Best case msgs generated (%): ", str ((best_case/total_values)*100))
    print("How bad the predictions are (%):")
    print(result_accuracy)
    print("massages generated (%):") 
    print(result_packets)
    best_case=0
    total_values=0
    rel = [0.01, 0.03, 0.05, 0.10, 0.25]
    abs = 2.5
    t=1
    output_rel = [None] * len(rel)
    for i in range(len(rel)):
        output_rel[i],best_case,total_values,msgs[i] = evaluate_alsolar(abs,rel[i],t, all_yt_pred, all_yt_lab)
    print("If we only communicate after a wrong prediction")
    result_accuracy=[((value / total_values))*100 for value in output_rel]
    result_packets = [((value +(total_values/pred_len)) / total_values) *100 for value in msgs]
    print("Best case msgs generated (%): ", str ((best_case/total_values)*100))
    print("How bad the predictions are (%):")
    print(result_accuracy)
    print("massages generated (%):") 
    print(result_packets)
    #Print("If we communicate after a wrong prediction + every prediction length") #this does not consider the intersection so currently removing this.
    #print((output_rel+best_case)/total_values)
    
    
    #Testing time- tolerance for 1= no tolerence to 5= 5 consective wrong predictions%
    best_case=0
    total_values=0
    t = [1,2,3,4,5]
    abs = 2.5
    rel=0.05
    
    output_time = [None] * len(t)
    for i in range(len(t)):
        a,b,total_values,output_time[i] = evaluate_alsolar(abs,rel,t[i], all_yt_pred, all_yt_lab)
    print("If we only communicate after t consective wrong prediction")
    result_packets=[((value +(total_values/pred_len)) / total_values) *100 for value in output_time]
    
    print("massages generated (%):") 
    print(result_packets)



def evaluate_alsolar(abs, rel, t, all_yt_pred, all_yt_lab):
    

    total_values=0
    total_ones=0
    total_non_zero_sum=0
    abs_error = abs
    rel_error = rel
    
    for batch_idx in (range(len(all_yt_pred))):
        
        yt_pred = all_yt_pred[batch_idx]
        yt_lab = all_yt_lab[batch_idx]
        
        abs_differences = torch.abs(yt_pred - yt_lab)
        rel_differences = torch.abs(yt_lab * rel_error)

        yt_pred_int = yt_pred.to(dtype=torch.int)  # Convert yt_pred to integers
        yt_lab_int = yt_lab.to(dtype=torch.int)    # Convert yt_lab to integers
        
        max_epsilon = torch.maximum(torch.tensor(abs_error), rel_differences)

        lower_bound = yt_lab - max_epsilon
        upper_bound = yt_lab + max_epsilon
        
        binary_array = torch.where((yt_pred_int >= lower_bound) & (yt_pred_int <= upper_bound), torch.tensor(0), torch.tensor(1))
        
        total_values = total_values + binary_array.numel()  # Total number of values in the binary array
        total_ones = total_ones+ binary_array.sum().item()

        

        #TODO: may be i will have to change the shape of results, depending on the dataset, currently it works for al solar
        results = np.zeros((binary_array.shape[0], binary_array.shape[2]), dtype=int)
        for i in range(binary_array.shape[0]):
            results[i,:] = count_consecutive_ones(binary_array[i,:,:], t)

        total_non_zero_sum = total_non_zero_sum + np.sum(results[results != 0])
    
    features = all_yt_pred[0].shape[2]
    pred_len = all_yt_pred[0].shape[1]

    #print(features, pred_len)

    best_case = total_values/(features*pred_len)
    #print("best case senario msgs: ",str(best_case))
    #print("worst case senario msgs: ",str(total_values))
    #total_ones/ total_values does not include the msgs after every batch
    return  total_ones, best_case, total_values, total_non_zero_sum #total wrong predictions

  

if __name__ == "__main__":
    # CLI
    parser = create_parser()
    args = parser.parse_args()
    file = args.file_name
    run_command = ' '.join(sys.argv)
    with open(file, 'a') as f:
        f.write("Run command: {}\n".format(run_command))
    for trial in range(args.trials):
        main(args)
