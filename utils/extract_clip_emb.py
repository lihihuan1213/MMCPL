import torch

def clip_embedding(pt_path):
    data = torch.load(pt_path, weights_only=True)

    text_embedding = []
    image_embedding = []

    for item in data:
        text_embedding.append(item['text_embedding'])
        image_embedding.append(item['image_embedding'])

    text_matrix = torch.stack(text_embedding)
    image_matrix = torch.stack(image_embedding)

    return text_matrix, image_matrix
