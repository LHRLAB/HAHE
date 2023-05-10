import torch

def build_graph(vocabulary, examples, hyperedge_dropout, device):
    selected = int((1 - hyperedge_dropout) * len(examples))
    examples = examples[:selected]
    s, t = [], []
    for hyperedge, example in enumerate(examples):
        L = [example.head, example.tail]
        if example.auxiliary_info:
            for i in example.auxiliary_info.values():
                L += list(i)
        for entity in vocabulary.convert_tokens_to_ids(L):
            s.append(hyperedge)
            t.append(entity)
    forward_edge = torch.tensor([t, s], dtype=torch.long).to(device)
    backward_edge = torch.tensor([s, t], dtype=torch.long).to(device)
    return forward_edge, backward_edge, len(examples)
