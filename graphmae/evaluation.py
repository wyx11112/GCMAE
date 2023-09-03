import copy
from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np

import dgl
import dgl.function as fn

from graphmae.utils import create_optimizer, accuracy
from sklearn.metrics import roc_auc_score, accuracy_score, adjusted_rand_score, confusion_matrix, f1_score
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score as NMI
from sklearn.model_selection import KFold
from sklearn import svm
from munkres import Munkres


def compute_auc(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score])
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]
    )
    return roc_auc_score(labels.cpu().numpy(), scores.cpu().numpy())


def evaluate_link_prediction(pred, h, test_pos_g, test_neg_g, train_pos_g, train_neg_g):
    with torch.no_grad():
        pos_score = pred(test_pos_g, h)
        neg_score = pred(test_neg_g, h)
        print("Test AUC", compute_auc(pos_score, neg_score))
        pos_score = pred(train_pos_g, h)
        neg_score = pred(train_neg_g, h)
        print("Train AUC", compute_auc(pos_score, neg_score))


def calculate_cost_matrix(C, n_clusters):
    cost_matrix = np.zeros((n_clusters, n_clusters))

    # cost_matrix[i,j] will be the cost of assigning cluster i to label j
    for j in range(n_clusters):
        s = np.sum(C[:, j])  # number of examples in cluster i
        for i in range(n_clusters):
            t = C[i, j]
            cost_matrix[j, i] = s-t
    return cost_matrix


def get_cluster_labels_from_indices(indices):
    n_clusters = len(indices)
    clusterLabels = np.zeros(n_clusters)
    for i in range(n_clusters):
        clusterLabels[i] = indices[i][1]
    return clusterLabels


def get_y_preds(cluster_assignments, y_true, n_clusters):
    '''
    Computes the predicted labels, where label assignments now
    correspond to the actual labels in y_true (as estimated by Munkres)

    cluster_assignments:    array of labels, outputted by kmeans
    y_true:                 true labels
    n_clusters:             number of clusters in the dataset

    returns:    a tuple containing the accuracy and confusion matrix,
                in that order
    '''
    confusion = confusion_matrix(y_true, cluster_assignments, labels=None)
    # compute accuracy based on optimal 1:1 assignment of clusters to labels
    cost_matrix = calculate_cost_matrix(confusion, n_clusters)
    indices = Munkres().compute(cost_matrix)
    kmeans_to_true_cluster_labels = get_cluster_labels_from_indices(indices)
    y_pred = kmeans_to_true_cluster_labels[cluster_assignments]
    return y_pred


def clustering_for_transductive(model, graph, x, num_classes):
    model.eval()
    X = model.embed(graph, x)
    labels = graph.ndata["label"]
    labels = labels.cpu().detach().numpy()
    X = X.cpu().detach().numpy()
    pred = KMeans(n_clusters=num_classes, max_iter=100, n_init=10, init='k-means++', algorithm='auto', random_state=0).fit_predict(X)
    y_pred = get_y_preds(pred, labels, num_classes)
    nmi = NMI(labels, y_pred)
    acc = adjusted_rand_score(labels, y_pred)
    print(f"--- Clustering NMI: {nmi:.4f}, Clustering ACC: {acc:.4f} ")
    return nmi, acc


def link_prediction_for_transductive(model, graph, x, lr_lp, weight_decay_lp, max_epoch, device, test_ratio=0.1, mute=False):
    model.eval()
    decoder = DotPredictor()
    decoder.to(device)
    optimizer = torch.optim.Adam(
        [{'params': decoder.parameters()}], lr=lr_lp, weight_decay=weight_decay_lp
    )
    criterion = torch.nn.BCEWithLogitsLoss()
    graph = graph.to(device)
    x = x.to(device)

    num_nodes = graph.number_of_nodes()
    u, v = graph.edges()
    num_edges = graph.number_of_edges()
    eids = torch.randperm(num_edges).to(device)
    test_size = int(eids.shape[0] * test_ratio)
    test_pos_u, test_pos_v = u[eids[:test_size]], v[eids[:test_size]]
    train_pos_u, train_pos_v = u[eids[test_size:]], v[eids[test_size:]]

    num_samples = num_edges
    neg_u, neg_v = dgl.sampling.global_uniform_negative_sampling(graph, num_samples, exclude_self_loops=True)
    neg_eids = torch.randperm(num_edges)
    test_neg_u, test_neg_v = neg_u[neg_eids[:test_size]], neg_v[neg_eids[:test_size]]
    train_neg_u, train_neg_v = neg_u[neg_eids[test_size:]], neg_v[neg_eids[test_size:]]

    train_g = dgl.remove_edges(graph, eids[:test_size])
    train_g = dgl.add_self_loop(train_g)
    # test_g = graph.edge_subgraph(eids[:test_size])
    train_pos_g = dgl.graph((train_pos_u, train_pos_v), num_nodes=num_nodes)
    train_neg_g = dgl.graph((train_neg_u, train_neg_v), num_nodes=num_nodes)
    test_pos_g = dgl.graph((test_pos_u, test_pos_v), num_nodes=num_nodes)
    test_neg_g = dgl.graph((test_neg_u, test_neg_v), num_nodes=num_nodes)

    best_test_auc = 0
    best_epoch = 0
    best_model = None

    if not mute:
        epoch_iter = tqdm(range(max_epoch))
    else:
        epoch_iter = range(max_epoch)

    for epoch in epoch_iter:
        model.train()
        out = model.embed(train_g, train_g.ndata["feat"])
        pos_score = decoder(train_pos_g, out)
        neg_score = decoder(train_neg_g, out)
        loss = criterion(pos_score, neg_score)
        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3)
        optimizer.step()

        with torch.no_grad():
            model.eval()
            out = model.embed(train_g, train_g.ndata["feat"])
            pos_score_test = decoder(test_pos_g, out)
            neg_score_test = decoder(test_neg_g, out)
            test_auc = compute_auc(pos_score_test, neg_score_test)
            pos_score_train = decoder(train_pos_g, out)
            neg_score_train = decoder(train_neg_g, out)
            train_auc = compute_auc(pos_score_train, neg_score_train)
            test_loss = criterion(pos_score_test, neg_score_test)
        if test_auc >= best_test_auc:
            best_test_auc = test_auc
            # print('best', best_test_auc)
            best_epoch = epoch
            best_model = copy.deepcopy(decoder)

        if not mute:
            epoch_iter.set_description(
                f"# Epoch: {epoch}, train_loss:{loss.item(): .4f}, train_auc:{train_auc: .4f}, "
                f"test_loss:{test_loss.item(): .4f}, test_auc:{test_auc: .4f}")

        best_model.eval()
        with torch.no_grad():
            pos_score = best_model(test_pos_g, out)
            neg_score = best_model(test_neg_g, out)
            test_auc = compute_auc(pos_score, neg_score)
        if mute:
            print(
                f"# IGNORE: --- TestAUC: {test_auc:.4f}, early-stopping-TestAUC: {best_test_auc:.4f} in epoch {best_epoch} --- ")
        else:
            print(
                f"--- TestAUC: {test_auc:.4f}, early-stopping-TestAUC: {best_test_auc:.4f} in epoch {best_epoch} --- ")

        # (final_acc, es_acc, best_acc)
        return test_auc, best_test_auc


def reconstruct_adj_evaluation(model, graph, x, device):
    model.eval()
    with torch.no_grad():
        re_adj = model.reconstruct_adj(graph.to(device), x.to(device))
        original_adj = graph.adj().to_dense()
        original_adj = original_adj.to(device)
        adj_sim = torch.cosine_similarity(original_adj, re_adj, dim=1)
        return adj_sim, original_adj, re_adj


def linear_probing_for_large(model, labels, feat, optimizer, max_epoch, device):
    criterion = torch.nn.CrossEntropyLoss()
    x = feat.to(device)
    acc = 0

    for epoch in range(max_epoch):
        model.train()
        out = model(x, x)
        loss = criterion(out, labels)
        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3)
        optimizer.step()
        acc = accuracy(out, labels)
        print(f"--- Val_ACC: {acc:.4f} ---")

    return acc


def large_node_classification_evaluation(model, graph, labels, split_dix, loader, num_classes, lr_f, weight_decay_f,
                                         max_epoch_f, device,):
    model.eval()
    y = []
    x_emb = None
    with torch.no_grad():
        embeddings = model.sage_embed(graph, device, 1024)
    in_feat = embeddings.shape[1]
    print('emb:', embeddings.shape)
    print(1/0)
    encoder = LogisticRegression(in_feat, num_classes)
    num_finetune_params = [p.numel() for p in encoder.parameters() if p.requires_grad]
    print(f"num parameters for finetuning: {sum(num_finetune_params)}")
    encoder.to(device)
    optimizer_f = create_optimizer("adam", encoder, lr_f, weight_decay_f)
    feat = torch.cat(x_emb)
    labels = torch.cat(y)
    final_acc = linear_probing_for_large(encoder, labels, feat, optimizer_f, max_epoch_f, device)
    return final_acc


def node_classification_evaluation(model, graph, x, num_classes, lr_f, weight_decay_f, max_epoch_f, device, linear_prob=True, mute=False):
    model.eval()
    if linear_prob:
        with torch.no_grad():
            x = model.embed(graph.to(device), x.to(device))
            in_feat = x.shape[1]
        encoder = LogisticRegression(in_feat, num_classes)
    else:
        encoder = model.encoder
        encoder.reset_classifier(num_classes)

    num_finetune_params = [p.numel() for p in encoder.parameters() if p.requires_grad]
    if not mute:
        print(f"num parameters for finetuning: {sum(num_finetune_params)}")
    
    encoder.to(device)
    optimizer_f = create_optimizer("adam", encoder, lr_f, weight_decay_f)
    final_acc, estp_acc = linear_probing_for_transductive_node_classiifcation(encoder, graph, x, optimizer_f, max_epoch_f, device, mute)
    return final_acc, estp_acc


def linear_probing_for_transductive_node_classiifcation(model, graph, feat, optimizer, max_epoch, device, mute=False):
    criterion = torch.nn.CrossEntropyLoss()

    graph = graph.to(device)
    x = feat.to(device)

    train_mask = graph.ndata["train_mask"]
    val_mask = graph.ndata["val_mask"]
    test_mask = graph.ndata["test_mask"]
    labels = graph.ndata["label"]

    best_val_acc = 0
    best_val_epoch = 0
    best_model = None

    if not mute:
        epoch_iter = tqdm(range(max_epoch))
    else:
        epoch_iter = range(max_epoch)

    for epoch in epoch_iter:
        model.train()
        out = model(graph, x)
        loss = criterion(out[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3)
        optimizer.step()

        with torch.no_grad():
            model.eval()
            pred = model(graph, x)
            val_loss = criterion(pred[val_mask], labels[val_mask])
            test_loss = criterion(pred[test_mask], labels[test_mask])
            val_acc = accuracy(pred[val_mask], labels[val_mask])
            test_acc = accuracy(pred[test_mask], labels[test_mask])
            # val_acc = f1_score(labels[val_mask].cpu().numpy(), np.argmax(pred[val_mask].cpu().numpy(), axis=1), average='micro')
            # test_acc = f1_score(labels[test_mask].cpu().numpy(), np.argmax(pred[test_mask].cpu().numpy(), axis=1), average='micro')

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_val_epoch = epoch
            best_model = copy.deepcopy(model)

        if not mute:
            epoch_iter.set_description(f"# Epoch: {epoch}, train_loss:{loss.item(): .4f}, val_loss:{val_loss.item(): .4f}, val_acc:{val_acc}, test_loss:{test_loss.item(): .4f}, test_acc:{test_acc: .4f}")

    best_model.eval()
    with torch.no_grad():
        pred = best_model(graph, x)
        estp_test_acc = accuracy(pred[test_mask], labels[test_mask])
    if mute:
        print(f"# IGNORE: --- TestAcc: {test_acc:.4f}, early-stopping-TestAcc: {estp_test_acc:.4f}, Best ValAcc: {best_val_acc:.4f} in epoch {best_val_epoch} --- ")
    else:
        print(f"--- TestAcc: {test_acc:.4f}, early-stopping-TestAcc: {estp_test_acc:.4f}, Best ValAcc: {best_val_acc:.4f} in epoch {best_val_epoch} --- ")

    # (final_acc, es_acc, best_acc)
    return test_acc, estp_test_acc


def SVM_node_classiifcation(model, graph, x, device):
    accs = []
    model.eval()
    feature = model.embed(graph, x)
    feature = feature.data.cpu().numpy()
    labels = graph.ndata["label"]
    labels = labels.view(-1)
    kf = KFold(n_splits=5, random_state=42, shuffle=True)
    for train_index, test_index in kf.split(feature):
        train_X, train_y = feature[train_index], labels[train_index]
        test_X, test_y = feature[test_index], labels[test_index]
        clf = svm.SVC(kernel='rbf', decision_function_shape='ovo')
        clf.fit(train_X, train_y)
        preds = clf.predict(test_X)
        acc = accuracy_score(test_y, preds)
        accs.append(acc)
    accs = np.array(accs)
    accs = np.mean(accs)
    return accs


class LogisticRegression(nn.Module):
    def __init__(self, num_dim, num_class):
        super().__init__()
        self.linear = nn.Linear(num_dim, num_class)
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, g, x, *args):
        logits = self.linear(x)
        return logits


class DotPredictor(nn.Module):
    def forward(self, g, h):
        with g.local_scope():
            g.ndata["h"] = h
            g.apply_edges(fn.u_dot_v("h", "h", "score"))
            return g.edata["score"][:, 0]