import logging
import numpy as np
from tqdm import tqdm
import torch

from graphmae.utils import (
    build_args,
    create_optimizer,
    set_random_seed,
    TBLogger,
    get_current_lr,
    load_best_configs,
)
from graphmae.datasets.data_util import load_dataset
from graphmae.evaluation import node_classification_evaluation, link_prediction_for_transductive, \
    clustering_for_transductive, SVM_node_classiifcation
from graphmae.models import build_model


logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)


def pretrain(task, model, graph, feat, optimizer, max_epoch, device, scheduler, num_classes, lr_f, weight_decay_f,
             max_epoch_f, linear_prob, max_epoch_lp, lr_lp, weight_decay_lp, logger=None):
    logging.info("start training..")
    graph = graph.to(device)
    x = feat.to(device)
    epoch_iter = tqdm(range(max_epoch))

    for epoch in epoch_iter:
        model.train()
        loss, loss_dict = model(graph, x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        epoch_iter.set_description(f"# Epoch {epoch}: train_loss: {loss.item():.4f}")
        if logger is not None:
            loss_dict["lr"] = get_current_lr(optimizer)
            logger.note(loss_dict, step=epoch)

        if (epoch + 1) % 100 == 0:
            if task == 'cl':
                node_classification_evaluation(model, graph, x, num_classes, lr_f, weight_decay_f, max_epoch_f,
                                               device, linear_prob, mute=True)
            elif task == 'lp':
                link_prediction_for_transductive(model, graph, x, lr_lp, weight_decay_lp, max_epoch_lp,
                                                 device, test_ratio=0.1, mute=True)
            elif task == 'cls':
                clustering_for_transductive(model, graph, x, num_classes)

    return model


def main(args):
    device = args.device if args.device >= 0 else "cpu"
    seeds = args.seeds
    dataset_name = args.dataset
    task = args.task
    max_epoch = args.max_epoch
    max_epoch_f = args.max_epoch_f
    max_epoch_lp = args.max_epoch_lp
    num_hidden = args.num_hidden
    num_layers = args.num_layers
    encoder_type = args.encoder
    decoder_type = args.decoder
    replace_rate = args.replace_rate
    optim_type = args.optimizer 
    loss_fn = args.loss_fn
    lr = args.lr
    weight_decay = args.weight_decay
    lr_f = args.lr_f
    weight_decay_f = args.weight_decay_f
    lr_lp = args.lr_lp
    weight_decay_lp = args.weight_decay_lp
    linear_prob = args.linear_prob
    load_model = args.load_model
    save_model = args.save_model
    logs = args.logging
    use_scheduler = args.scheduler

    graph, (num_features, num_classes) = load_dataset(dataset_name)
    args.num_features = num_features

    acc_list = []
    estp_acc_list = []
    auc_list = []
    estp_auc_list = []
    nmi_list = []
    cls_acc_list = []

    for i, seed in enumerate(seeds):
        print(f"####### Run {i} for seed {seed}")
        set_random_seed(seed)

        if logs:
            logger = TBLogger(name=f"{dataset_name}_loss_{loss_fn}_rpr_{replace_rate}_nh_{num_hidden}_nl_{num_layers}_lr_{lr}_mp_{max_epoch}_mpf_{max_epoch_f}_wd_{weight_decay}_wdf_{weight_decay_f}_{encoder_type}_{decoder_type}")
        else:
            logger = None

        model = build_model(args)

        model.to(device)
        optimizer = create_optimizer(optim_type, model, lr, weight_decay)

        if use_scheduler:
            logging.info("Use schedular")
            scheduler = lambda epoch :( 1 + np.cos((epoch) * np.pi / max_epoch) ) * 0.5
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler)
        else:
            scheduler = None
            
        x = graph.ndata["feat"]
        if not load_model:
            model = pretrain(task, model, graph, x, optimizer, max_epoch, device, scheduler, num_classes, lr_f,
                             weight_decay_f, max_epoch_f, linear_prob, max_epoch_lp, lr_lp, weight_decay_lp, logger)
            model = model.cpu()

        if load_model:
            logging.info("Loading Model ... ")
            model.load_state_dict(torch.load("clustering.pt"))
        if save_model:
            logging.info("Saveing Model ...")
            torch.save(model.state_dict(), "checkpoint.pt")

        model = model.to(device)
        model.eval()

        if task == 'cl':
            # final_acc = SVM_node_classiifcation(model, graph, x, device)
            final_acc, estp_acc = node_classification_evaluation(model, graph, x, num_classes, lr_f, weight_decay_f, max_epoch_f, device, linear_prob)
            acc_list.append(final_acc)
            estp_acc_list.append(estp_acc)
        elif task == 'lp':
            final_auc, estp_auc = link_prediction_for_transductive(model, graph, x, lr_lp, weight_decay_lp, max_epoch_lp,
                                                                   device, test_ratio=0.1, mute=True)
            auc_list.append(final_auc)
            estp_auc_list.append(estp_auc)
        elif task == 'cls':
            graph = graph.to(device)
            x = x.to(device)
            final_nmi, final_acc = clustering_for_transductive(model, graph, x, num_classes)
            nmi_list.append(final_nmi)
            cls_acc_list.append(final_acc)

        if logger is not None:
            logger.finish()
    if task == 'cl':
        final_acc, final_acc_std = np.mean(acc_list), np.std(acc_list)
        estp_acc, estp_acc_std = np.mean(estp_acc_list), np.std(estp_acc_list)
        print(f"# final_acc: {final_acc:.4f}±{final_acc_std:.4f}")
        print(f"# early-stopping_acc: {estp_acc:.4f}±{estp_acc_std:.4f}")
        print('acc_list: ', acc_list)
    elif task == 'lp':
        final_auc, final_auc_std = np.mean(auc_list), np.std(auc_list)
        estp_auc, estp_auc_std = np.mean(estp_auc_list), np.std(estp_auc_list)
        print(f"# final_auc: {final_auc:.4f}±{final_auc_std:.4f}")
        print(f"# early-stopping_auc: {estp_auc:.4f}±{estp_auc_std:.4f}")
    elif task == 'cls':
        final_nmi, final_nmi_std = np.mean(nmi_list), np.std(nmi_list)
        final_acc, final_acc_std = np.mean(cls_acc_list), np.std(cls_acc_list)
        print(f"# final_nmi: {final_nmi:.4f}±{final_nmi_std:.4f}")
        print(f"# final_acc: {final_acc:.4f}±{final_acc_std:.4f}")
        print(nmi_list)
        print(cls_acc_list)


if __name__ == "__main__":
    args = build_args()
    task = args.task
    if args.use_cfg:
        if task == 'cls':
            args = load_best_configs(args, "configs_cls.yml")
        elif task == 'lp':
            args = load_best_configs(args, "configs_lp.yml")
        else:
            args = load_best_configs(args, "configs.yml")
    print(args)

    main(args)
