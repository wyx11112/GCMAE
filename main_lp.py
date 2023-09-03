import copy
import logging
import numpy as np
from tqdm import tqdm
import torch
import nni
from nni.utils import merge_parameter
import dgl

from graphmae.utils import (
    build_args,
    create_optimizer,
    set_random_seed,
    TBLogger,
    get_current_lr,
    load_best_configs
)
from graphmae.datasets.data_util import load_dataset
from graphmae.evaluation import node_classification_evaluation, link_prediction_for_transductive, \
    clustering_for_transductive, compute_auc, DotPredictor
from graphmae.models import build_model

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)


def pretrain(model, decoder, graph, x, optimizer, max_epoch, device, scheduler, logger=None, mute=False):
    logging.info("start training..")

    decoder.to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    graph = graph.to(device)
    x = x.to(device)

    num_nodes = graph.number_of_nodes()
    u, v = graph.edges()
    num_edges = graph.number_of_edges()
    eids = torch.randperm(num_edges).to(device)
    test_size = int(eids.shape[0] * 0.1)
    test_pos_u, test_pos_v = u[eids[:test_size]], v[eids[:test_size]]
    train_pos_u, train_pos_v = u[eids], v[eids]

    num_samples = num_edges
    neg_u, neg_v = dgl.sampling.global_uniform_negative_sampling(graph, num_samples, exclude_self_loops=True)
    neg_eids = torch.randperm(num_edges)
    test_neg_u, test_neg_v = neg_u[neg_eids[:test_size]], neg_v[neg_eids[:test_size]]
    train_neg_u, train_neg_v = neg_u[neg_eids], neg_v[neg_eids]

    # train_g = dgl.remove_edges(graph, eids[:test_size])
    train_g = graph
    # train_g = dgl.add_self_loop(train_g)
    # test_g = graph.edge_subgraph(eids[:test_size])
    train_pos_g = dgl.graph((train_pos_u, train_pos_v), num_nodes=num_nodes)
    train_neg_g = dgl.graph((train_neg_u, train_neg_v), num_nodes=num_nodes)
    test_pos_g = dgl.graph((test_pos_u, test_pos_v), num_nodes=num_nodes)
    test_neg_g = dgl.graph((test_neg_u, test_neg_v), num_nodes=num_nodes)

    best_test_auc = 0
    best_epoch = 0
    best_model = None

    epoch_iter = tqdm(range(max_epoch))
    for epoch in epoch_iter:
        model.train()
        loss, loss_dict = model(graph, x)
        out = model.embed(train_g, train_g.ndata["feat"])
        pos_score = decoder(train_pos_g, out)
        neg_score = decoder(train_neg_g, out)
        loss_lp = criterion(pos_score, neg_score)
        loss = loss + loss_lp
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
            with torch.no_grad():
                model.eval()
                out = model.embed(train_g, train_g.ndata["feat"])
                pos_score_test = decoder(test_pos_g, out)
                neg_score_test = decoder(test_neg_g, out)
                test_auc = compute_auc(pos_score_test, neg_score_test)
                # test_auc = compute_ap(pos_score_test, neg_score_test)
                pos_score_train = decoder(train_pos_g, out)
                neg_score_train = decoder(train_neg_g, out)
                train_auc = compute_auc(pos_score_train, neg_score_train)
                # train_auc = compute_ap(pos_score_train, neg_score_train)
                test_loss = criterion(pos_score_test, neg_score_test)
            if test_auc >= best_test_auc:
                best_test_auc = test_auc
                # print('best', best_test_auc)
                best_epoch = epoch
                # print(best_epoch)
                best_model = copy.deepcopy(decoder)

            if not mute:
                epoch_iter.set_description(
                    f"# Epoch: {epoch}, train_loss:{loss.item(): .4f}, train_auc:{train_auc: .4f}, "
                    f"test_loss:{test_loss.item(): .4f}, test_auc:{test_auc: .4f}")

            best_model.eval()
            with torch.no_grad():
                pos_score = best_model(test_pos_g, out)
                neg_score = best_model(test_neg_g, out)
                estp_test_auc = compute_auc(pos_score, neg_score)
                # estp_test_auc = compute_ap(pos_score, neg_score)
            if mute:
                print(
                    f"# IGNORE: --- TestAUC: {test_auc:.4f}, early-stopping-TestAUC: {estp_test_auc:.4f} in epoch {best_epoch} --- ")
            else:
                print(
                    f"--- TestAUC: {test_auc:.4f}, early-stopping-TestAUC: {estp_test_auc:.4f} in epoch {best_epoch} --- ")

    # (final_acc, es_acc, best_acc)
    return test_auc, estp_test_auc

def main(args):
    device = args.device if args.device >= 0 else "cpu"
    seeds = args.seeds
    dataset_name = args.dataset
    max_epoch = args.max_epoch
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
    lr_lp = args.lr_lp
    weight_decay_lp = args.weight_decay_lp
    logs = args.logging
    use_scheduler = args.scheduler

    graph, (num_features, num_classes) = load_dataset(dataset_name)
    args.num_features = num_features

    auc_list = []
    estp_auc_list = []

    for i, seed in enumerate(seeds):
        print(f"####### Run {i} for seed {seed}")
        set_random_seed(seed)

        model = build_model(args)
        decoder = DotPredictor()
        # print(model)
        optimizer = torch.optim.Adam(
            list(decoder.parameters()) + list(model.parameters()), lr=lr, weight_decay=weight_decay
        )
        model.to(device)

        if use_scheduler:
            logging.info("Use schedular")
            scheduler = lambda epoch: (1 + np.cos((epoch) * np.pi / max_epoch)) * 0.5
            # scheduler = lambda epoch: epoch / warmup_steps if epoch < warmup_steps \
            # else ( 1 + np.cos((epoch - warmup_steps) * np.pi / (max_epoch - warmup_steps))) * 0.5
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler)
        else:
            scheduler = None

        x = graph.ndata["feat"]

        final_auc, estp_auc = pretrain(model,decoder, graph, x, optimizer, max_epoch, device, scheduler, )
        # model = model.cpu()
        #
        # model = model.to(device)
        # model.eval()
        #
        # final_auc, estp_auc = link_prediction_for_transductive(model, graph, x, lr_lp, weight_decay_lp,
        #                                                        max_epoch_lp,
        #                                                        device, test_ratio=0.1, mute=True)
        nni.report_final_result({'default': final_auc})
        auc_list.append(final_auc)
        estp_auc_list.append(estp_auc)
        #
        # if logger is not None:
        #     logger.finish()

    final_auc, final_auc_std = np.mean(auc_list), np.std(auc_list)
    estp_auc, estp_auc_std = np.mean(estp_auc_list), np.std(estp_auc_list)
    print(f"# final_auc: {final_auc:.4f}±{final_auc_std:.4f}")
    print(f"# early-stopping_auc: {estp_auc:.4f}±{estp_auc_std:.4f}")


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    args = build_args()
    args = load_best_configs(args, "configs_lp.yml")

    nni_params = nni.get_next_parameter()
    args = merge_parameter(args, nni_params)
    print(args)

    main(args)
