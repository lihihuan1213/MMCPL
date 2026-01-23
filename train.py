import numpy as np
import torch
import time
import torch.nn as nn
import pandas as pd
import argparse
from model import Net
from utils.set_logger import set_logger
from sklearn.utils import shuffle
from torch import optim as optima
from utils.set_seed import set_seed
from utils.eval_performance import evaluate
from utils.load_DGCN import generate_adj_matrices
from utils.extract_clip_emb import clip_embedding
from utils.skill_toresource import preprocess_skill_to_resource
from test import test


parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=40, help='number of epochs')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-3, help='weight decay')
parser.add_argument('--seed', type=int, default=25, help='random seed')
# parser.add_argument('--dataset', type=str, default='Lecturebank', help='dataset name')
# parser.add_argument('--dataset', type=str, default='UC', help='dataset name')
parser.add_argument('--dataset', type=str, default='MOOC', help='dataset name')

args = parser.parse_args()

logger = set_logger(args.dataset)

def train(args):
    set_seed(args.seed)
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    logger.info('Dataset: ' + args.dataset + '\n')

    logger.info("Read data complete!")
    train_data_df = pd.read_csv(f"./data1/{args.dataset}/train2.csv", header=0)
    val_data_df = pd.read_csv(f"./data1/{args.dataset}/val2.csv", header=0)

    train_data = [tuple(x) for x in train_data_df.to_numpy()]
    val_data = [tuple(x) for x in val_data_df.to_numpy()]

    edge_file = f"./data1/{args.dataset}/cs_edges1.csv"
    index_file = f"./data1/{args.dataset}/resources_index1.csv"
    adj_out, adj_in = generate_adj_matrices(edge_file, index_file)
    adj_out = adj_out.to(device)
    adj_in = adj_in.to(device)

    clip_path = f"./data1/{args.dataset}/skill_clip_embedding.pt"
    text_emb, img_emb = clip_embedding(clip_path)
    text_emb = text_emb.to(device)
    img_emb = img_emb.to(device)

    contain_path = f"./data1/{args.dataset}/contain.csv"
    skill_resource_dict = preprocess_skill_to_resource(contain_path)

    resources_pt = torch.load(f"./data1/{args.dataset}/resource_embs.pt", map_location=device, weights_only=True)
    resources_embed = torch.stack(list(resources_pt.values()), dim=0)

    model = Net(adj_out, adj_in, img_emb, text_emb, resources_embed, skill_resource_dict).to(device)
    optimizer = optima.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.BCELoss()
    logger.info("Training!!!")
    best_val_auc = 0.0

    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        logger.info(f'Epoch {epoch + 1}/{args.epochs}, lr = {optimizer.param_groups[0]["lr"]}')

        # Shuffle training data for each epoch
        X_train = np.array(shuffle(train_data, random_state=args.seed))
        sum_total_loss = 0.0

        model.train()
        batch_idx = 0

        for i in range(X_train.shape[0] // args.batch_size):
            x = X_train[batch_idx * args.batch_size: batch_idx * args.batch_size + args.batch_size]
            batch_idx += 1
            c1, c2 = x[:, 0], x[:, 1]
            target = x[:, -1]
            target = torch.tensor(target, dtype=torch.float32).to(device)

            optimizer.zero_grad()
            # Forward pass
            y_pred, cl_loss, loss_intra = model(c1, c2)
            y_pred = y_pred.squeeze(1)
            loss = criterion(y_pred, target) + 0.1 * cl_loss + 0.2 * loss_intra 
            sum_total_loss += loss
            loss.backward()
            optimizer.step()

        # Calculate average loss for the epoch
        average_loss = (sum_total_loss / batch_idx)
        logger.info(f"Average train loss for epoch {epoch + 1}/{args.epochs}: {average_loss}")

        # Evaluate on training data
        train_metrics = evaluate(model, train_data, args.batch_size, device)
        logger.info(f"Train metrics: ACC = {train_metrics['ACC']:.4f}, F1 = {train_metrics['F1']:.4f}, "
                    f"Precision = {train_metrics['Precision']:.4f}, Recall = {train_metrics['Recall']:.4f}, "
                    f"AUC = {train_metrics['AUC']:.4f}")

        val_metrics = evaluate(model, val_data, args.batch_size, device)
        logger.info(f"Validation metrics: ACC = {val_metrics['ACC']:.4f}, F1 = {val_metrics['F1']:.4f}, "
                    f"Precision = {val_metrics['Precision']:.4f}, Recall = {val_metrics['Recall']:.4f}, "
                    f"AUC = {val_metrics['AUC']:.4f}")

        # Save the model if AUC improves on the validation set
        if val_metrics['AUC'] > best_val_auc:
            best_val_auc = val_metrics['AUC']
            logger.info(f"Saved new best model at epoch {epoch + 1} with val_auc = {best_val_auc:.4f}!!!")
            model_name = f"./best_model/{args.dataset}/{args.dataset}_best_net.pt"
            torch.save({
                'model': model.state_dict(),
            }, model_name)
            logger.info(f"Model parameters saved to {model_name}!")

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        logger.info(f"Epoch {epoch + 1} duration: {epoch_duration:.2f} seconds\n")


if __name__ == '__main__':
    start_time = time.time()
    train(args)
    end_time = time.time()
    total_time = end_time - start_time
    logger.info(f"Total time: {total_time:.2f} seconds\n")
