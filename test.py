import time
import argparse
import pandas as pd
import datetime
from model import Net
from utils.eval_performance import evaluate_test
from utils.load_DGCN import generate_adj_matrices
from utils.skill_toresource import preprocess_skill_to_resource
from utils.extract_clip_emb import clip_embedding
import torch


parser = argparse.ArgumentParser()
# parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
# parser.add_argument('--seed', type=int, default=25, help='Random seed')
# parser.add_argument('--dataset', type=str, default='Lecturebank', help='dataset name')
# parser.add_argument('--dataset', type=str, default='UC', help='dataset name')
parser.add_argument('--dataset', type=str, default='MOOC', help='dataset name.')
args = parser.parse_args()

def test(args):
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    print('Dataset: ' + args.dataset + '\n')

    print("Read test data complete!")
    test_data_df = pd.read_csv(f'./data1/{args.dataset}/test2.csv', header=0)
    test_data = [tuple(x) for x in test_data_df.to_numpy()]

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
    model_name = f"./best_model/{args.dataset}/{args.dataset}_best_net.pt"
    checkpoint = torch.load(model_name,weights_only=True)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    print("Testing!!!")
    test_metrics = evaluate_test(model, test_data, args.batch_size, device, save_path=f"./data1/{args.dataset}/predictions_with_concepts.csv")
    print(f"Test metrics: ACC = {test_metrics['ACC']:.4f}, F1 = {test_metrics['F1']:.4f}, "
          f"Precision = {test_metrics['Precision']:.4f}, Recall = {test_metrics['Recall']:.4f}, "
          f"AUC = {test_metrics['AUC']:.4f}")
    with open(f"./results/{args.dataset}/{args.dataset}.txt", "a") as f:
        f.write(f'Time: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
        f.write(f"ACC = {test_metrics['ACC']:.4f}, F1 = {test_metrics['F1']:.4f}, "
                f"Precision = {test_metrics['Precision']:.4f}, Recall = {test_metrics['Recall']:.4f}, "
                f"AUC = {test_metrics['AUC']:.4f}\n")



if __name__ == '__main__':
    start_time = time.time()
    test(args)
    end_time = time.time()
    total_time = end_time - start_time  # Calculate the total test time
    print(f'Test time: {total_time:.2f} seconds\n')