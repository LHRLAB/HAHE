import torch
import torch.nn as nn
import math
from torch_geometric.nn import GATv2Conv

class Global(nn.Module):
    def __init__(self, forward_edge, backward_edge, num_hyperedges, num_nodes, dim, use_global, num_layers, heads, dropout, activation):
        super(Global, self).__init__()
        self.layers0 = nn.ModuleList()
        self.layers1 = nn.ModuleList()
        for _ in range(num_layers):
            self.layers0.append(GATv2Conv(dim, dim // heads, heads=heads, dropout=dropout))
            self.layers1.append(GATv2Conv(dim, dim // heads, heads=heads, dropout=dropout))
        self.num_layers = num_layers
        self.forward_edge = forward_edge
        self.backward_edge = backward_edge
        self.node_embedding = nn.parameter.Parameter(torch.Tensor(num_nodes, dim))
        self.hyperedge_embedding = nn.parameter.Parameter(torch.Tensor(num_hyperedges, dim))
        if activation == "gelu":
            self.activate = nn.GELU()
        elif activation == "relu":
            self.activate = nn.ReLU()
        elif activation == 'elu':
            self.activate = nn.ELU()
        elif activation == 'tanh':
            self.activate = nn.Tanh()
        self.use_global = use_global

    def forward(self):
        node_embedding = self.node_embedding
        if self.use_global is True:
            hyperedge_embedding = self.hyperedge_embedding.to(node_embedding.device)
            for i in range(self.num_layers):
                tmp = self.layers0[i]((node_embedding, hyperedge_embedding), self.forward_edge.to(node_embedding.device))
                tmp = self.activate(tmp)
                hyperedge_embedding = hyperedge_embedding + tmp
                tmp = self.layers1[i]((hyperedge_embedding, node_embedding), self.backward_edge.to(node_embedding.device))
                tmp = self.activate(tmp)
                node_embedding = node_embedding + tmp
        return node_embedding

class PrepareForMultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim: int, heads: int, bias: bool, use_node: bool) -> None:
        super().__init__()
        self.heads = heads
        self.use_node = use_node       

        if self.use_node is True:
            self.layer_s=nn.Linear(hidden_dim,hidden_dim)
            self.layer_r=nn.Linear(hidden_dim,hidden_dim)
            self.layer_o=nn.Linear(hidden_dim,hidden_dim)
            self.layer_a=nn.Linear(hidden_dim,hidden_dim)
            self.layer_v=nn.Linear(hidden_dim,hidden_dim)
        else:
            self.linear = nn.Linear(hidden_dim, hidden_dim, bias=bias)


    def forward(self, x : torch.Tensor):
        shape = x.shape[:-1]

        if self.use_node is False:
            x = self.linear(x)
        else:
            device=x.device
            max_seq_len=x.size(1)
            mask_s = torch.tensor([1]+[0]*(max_seq_len-1)).to(device)
            mask_r = torch.tensor([0,1]+[0]*(max_seq_len-2)).to(device)
            mask_o = torch.tensor([0,0,1]+[0]*(max_seq_len-3)).to(device)
            mask_a = torch.tensor([0,0,0]+[1,0]*int(((max_seq_len-3)/2))).to(device)
            mask_v = torch.tensor([0,0,0]+[0,1]*int(((max_seq_len-3)/2))).to(device)

            x_s=self.layer_s(torch.mul(x,mask_s[:,None].expand(-1,x.size(-1))))
            x_r=self.layer_r(torch.mul(x,mask_r[:,None].expand(-1,x.size(-1))))
            x_o=self.layer_o(torch.mul(x,mask_o[:,None].expand(-1,x.size(-1))))
            x_a=self.layer_a(torch.mul(x,mask_a[:,None].expand(-1,x.size(-1))))
            x_v=self.layer_v(torch.mul(x,mask_v[:,None].expand(-1,x.size(-1))))
                            
            x=(x_s+x_r+x_o+x_a+x_v) 
      
        return x.reshape(*shape, self.heads, -1)

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim: int, heads: int, dropout_prob: float, use_edge: bool, remove_mask: bool, bias: bool, use_node: bool) -> None:
        super().__init__()
        assert hidden_dim % heads == 0
        self.dim = hidden_dim // heads
        self.heads = heads
        self.query = PrepareForMultiHeadAttention(hidden_dim, heads, bias, use_node)
        self.key = PrepareForMultiHeadAttention(hidden_dim, heads, bias, use_node)
        self.value = PrepareForMultiHeadAttention(hidden_dim, heads, True, use_node)
        self.pos = PrepareForMultiHeadAttention(hidden_dim, heads, True, use_node)
        self.softmax = nn.Softmax(dim=-1)
        self.output = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.use_edge = use_edge
        self.remove_mask = remove_mask
        self.scale = 1 / math.sqrt(self.dim)
        # trasformer-xl
        self.r_w_bias = nn.Parameter(torch.Tensor(heads, self.dim)) # u
        self.r_r_bias = nn.Parameter(torch.Tensor(heads, self.dim)) # v

    def get_mask(self, graph: torch.Tensor):
        return graph.unsqueeze(1).repeat(1, self.heads, 1, 1)

    def forward(self, *, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                        graph: torch.Tensor, edge_key: torch.Tensor, edge_value: torch.Tensor, edge_query: torch.Tensor):
        # query/key/value: (batch, seq_len, hidden_dim)
        # graph: (batch, kinds, query, key)
        shape = query.shape[:-1]
        query = self.query(query)   # (batch, seq_len, head, hidden)
        key = self.key(key)         # (batch, seq_len, head, hidden)
        value = self.value(value)   # (batch, seq_len, head, hidden)
        seq_len = query.size(1)
        if self.use_edge is True:
            scores = torch.einsum("bqhd,bkhd->bhqk", query, key) + torch.einsum("bqhd,bqkd->bhqk", query, edge_key) + torch.einsum("bkqd,bkhd->bhqk", edge_query, key) + torch.einsum("bkqd,bqkd->bqk", edge_query, edge_key).unsqueeze(1)
            scores = scores * self.scale
            mask = self.get_mask(graph)
            if self.remove_mask is True:
                for i in range(3,seq_len,2):
                    if i==3:
                        mask[:,:,i:(i+2),(i+2):]=False
                    elif i==(seq_len-2):
                        mask[:,:,i:(i+2),3:i]=False
                    else:
                        mask[:,:,i:(i+2),(i+2):]=False
                        mask[:,:,i:(i+2),3:i]=False     
            scores = scores.masked_fill(mask == 0, -100000)
            attn = self.softmax(scores)
            attn = self.dropout(attn)
            x = torch.einsum("bhqk,bkhd->bqhd", attn, value) + torch.einsum("bhqk,bqkd->bqhd", attn, edge_value)
            x = x.reshape(*shape, -1)
        else:
            scores = torch.einsum("bqhd,bkhd->bhqk", query, key)
            scores *= self.scale
            mask = self.get_mask(graph)
            if self.remove_mask is True:
                for i in range(3,seq_len,2):
                    if i==3:
                        mask[:,:,i:(i+2),(i+2):]=False
                    elif i==(seq_len-2):
                        mask[:,:,i:(i+2),3:i]=False
                    else:
                        mask[:,:,i:(i+2),(i+2):]=False
                        mask[:,:,i:(i+2),3:i]=False  
            scores = scores.masked_fill(mask == 0, -100000)
            attn = self.softmax(scores)
            attn = self.dropout(attn)
            x = torch.einsum("bhqk,bkhd->bqhd", attn, value)
            x = x.reshape(*shape, -1)

        return self.output(x)  # (batch, query, hidden_dim)

class FeedForward(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, activation) -> None:
        super().__init__()
        act = None
        if activation == "gelu":
            act = nn.GELU()
        elif activation == "relu":
            act = nn.ReLU()
        elif activation == 'elu':
            act = nn.ELU()
        elif activation == 'tanh':
            act = nn.Tanh()
        self.layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            act,
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.layer(x)
        
class TransformerLayer(nn.Module):
    def __init__(self, hidden_dim: int, heads: int, dropout_prob: float, activation: str, use_edge: bool, remove_mask: bool, use_node: bool, bias=True, times=2) -> None:
        super().__init__()
        self.norm_attention = nn.LayerNorm(hidden_dim)
        self.attention = MultiHeadAttention(hidden_dim, heads, dropout_prob, use_edge, remove_mask, bias, use_node)
        self.dropout = nn.Dropout(dropout_prob)
        self.norm_ffn = nn.LayerNorm(hidden_dim)
        self.ffn = FeedForward(hidden_dim, hidden_dim * times, hidden_dim, activation)

    def forward(self, x: torch.Tensor, graph: torch.Tensor, edge_key: torch.Tensor, edge_value: torch.Tensor, edge_query: torch.Tensor):
        attn = self.attention(query=x, key=x, value=x, graph=graph, edge_key=edge_key, edge_value=edge_value, edge_query=edge_query)
        x = self.norm_attention(x + self.dropout(attn))
        ff = self.ffn(x)
        x = self.norm_ffn(x + self.dropout(ff))
        return x

class Transformer(nn.Module):
    def __init__(self, graph, vocab_size: int, local_layers: int, global_layers: int, hidden_dim: int,
            local_heads: int, global_heads: int, use_global: bool, local_dropout: float, global_dropout: float, 
            decoder_activation: str, global_activation: str, use_edge: bool, remove_mask: bool, use_node: bool, bias=True, times=2) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layers = nn.ModuleList()
        for _ in range(local_layers):
            self.layers.append(TransformerLayer(hidden_dim, local_heads, local_dropout, decoder_activation, use_edge, remove_mask, use_node, bias, times=times))
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.input_dropout = nn.Dropout(p=local_dropout)
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_act = nn.GELU()
        self.output_linear = nn.Linear(hidden_dim, hidden_dim)
        self.output_bias = torch.nn.parameter.Parameter(torch.zeros(vocab_size))
        self.edge_query_embedding = nn.Embedding(14, hidden_dim // local_heads, padding_idx=0)
        self.edge_key_embedding = nn.Embedding(14, hidden_dim // local_heads, padding_idx=0)
        self.edge_value_embedding = nn.Embedding(14, hidden_dim // local_heads, padding_idx=0)
        self.globl = Global(*graph, vocab_size, hidden_dim, use_global, global_layers, global_heads, global_dropout, global_activation)
        self.init_params()
    def init_params(self):
        for name, param in self.named_parameters():
            if "norm" in name:
                continue
            elif "bias" in name:
                nn.init.zeros_(param)
            elif "weight" in name or "att" in name or "embedding" in name:
                nn.init.normal_(param, mean=0, std=0.02)
            else:
                raise TypeError("Invalid Parameters")
    def forward(self, input_ids, input_mask, mask_position, mask_output, edge_labels):
        embedding = self.globl().to(input_ids.device)
        x = torch.nn.functional.embedding(input_ids, embedding)
        x = self.input_dropout(self.input_norm(x))
        edge_query = self.edge_query_embedding(edge_labels)
        edge_key = self.edge_key_embedding(edge_labels)
        edge_value = self.edge_value_embedding(edge_labels)

        for layer in self.layers:
            x = layer(x, input_mask, edge_key, edge_value, edge_query)
        x = x[torch.arange(x.shape[0]), mask_position]
        x = self.output_linear(x)  # x(batch_size, hiddem_dim)
        x = self.output_act(x)
        x = self.output_norm(x)
        y = torch.mm(x, embedding.transpose(0, 1)) + self.output_bias
        y = y.masked_fill(mask_output == 0, -100000)
        return y
