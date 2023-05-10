import json
import collections
import numpy as np
import torch.utils.data.dataset as Dataset

class Vocabulary(object):
    def __init__(self, vocab_file, num_relations, num_entities):
        self.vocab = self.load_vocab(vocab_file)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.num_relations = num_relations
        self.num_entities = num_entities
        assert len(self.vocab) == self.num_relations + self.num_entities + 2
    def load_vocab(self, vocab_file):
        """
        Load a vocabulary file into a dictionary.
        """
        vocab = collections.OrderedDict()
        fin = open(vocab_file, encoding='utf-8')
        for num, line in enumerate(fin):
            items = line.strip().split("\t")
            if len(items) > 2:
                break
            token = items[0]
            index = items[1] if len(items) == 2 else num
            token = token.strip()
            vocab[token] = int(index)
        return vocab
    def convert_by_vocab(self, vocab, items):
        """
        Convert a sequence of [tokens|ids] using the vocab.
        """
        output = []
        for item in items:
            output.append(vocab[item])
        return output
    def convert_tokens_to_ids(self, tokens):
        return self.convert_by_vocab(self.vocab, tokens)
    def convert_ids_to_tokens(self, ids):
        return self.convert_by_vocab(self.inv_vocab, ids)
    def __len__(self):
        return len(self.vocab)

class NaryExample(object):  # instance of N-ary
    def __init__(self,
                 arity,
                 head,
                 relation,
                 tail,
                 auxiliary_info=None):
        self.arity = arity
        self.head = head
        self.relation = relation
        self.tail = tail
        self.auxiliary_info = auxiliary_info

class NaryFeature(object):  # samples for training
    def __init__(self,
                 feature_id,
                 example_id,
                 input_tokens,
                 input_ids,
                 input_mask,
                 mask_position,
                 mask_label,
                 mask_type,
                 arity):
        """
        Construct NaryFeature.
        Args:
            feature_id: unique feature id
            example_id: corresponding example id
            input_tokens: input sequence of tokens
            input_ids: input sequence of ids
            input_mask: input sequence mask
            mask_position: position of masked token
            mask_label: label of masked token
            mask_type: type of masked token,
                1 for entities (values) and -1 for relations (attributes)
            arity: arity of the corresponding example
        """
        self.feature_id = feature_id        
        self.example_id = example_id        
        self.input_tokens = input_tokens    
        self.input_ids = input_ids          
        self.input_mask = input_mask        
        self.mask_position = mask_position  
        self.mask_label = mask_label        
        self.mask_type = mask_type          
        self.arity = arity

def read_examples(input_file, max_arity):
    """
    Read a n-ary json file into a list of NaryExample.
    """
    examples, total_instance = [], 0
    with open(input_file, "r", encoding='utf-8') as fr:
        for line in fr.readlines():
            obj = json.loads(line.strip())
            arity = obj["N"]
            relation = obj["relation"]
            head = obj["subject"]
            tail = obj["object"]

            auxiliary_info = None
            if arity > 2:
                auxiliary_info = collections.OrderedDict()
                # store attributes in alphabetical order
                for attribute in sorted(obj.keys()):
                    if attribute in ("N", "relation", "subject", "object"):
                        continue
                    # store corresponding values in alphabetical order
                    auxiliary_info[attribute] = sorted(obj[attribute])
            if arity <= max_arity:
                example = NaryExample(
                    arity=arity,
                    head=head,
                    relation=relation,
                    tail=tail,
                    auxiliary_info=auxiliary_info)
                examples.append(example)
                total_instance += (2 * (arity - 2) + 3)
    return examples, total_instance

def convert_examples_to_features(examples, vocabulary, max_arity, max_seq_length):
    """
    Convert a set of NaryExample into a set of NaryFeature. Each single
    NaryExample is converted into (2*(n-2)+3) NaryFeature, where n is
    the arity of the given example.
    """
    max_aux = max_arity - 2
    assert max_seq_length == 2 * max_aux + 3, \
        "Each input sequence contains relation, head, tail, " \
        "and max_aux attribute-value pairs."

    features = []
    feature_id = 0
    for (example_id, example) in enumerate(examples):
        # get original input tokens and input mask
        hrt = [ example.head, example.relation, example.tail ] 
        hrt_mask = [1, 1, 1]

        aux_q = []
        aux_q_mask = []
        aux_values = []
        aux_values_mask = []
        if example.auxiliary_info is not None:
            for attribute in example.auxiliary_info.keys():
                for value in example.auxiliary_info[attribute]:
                    aux_q.append(attribute)
                    aux_q.append(value)
                    aux_q_mask.append(1)
                    aux_q_mask.append(1)
        while len(aux_q) < max_aux*2:
            aux_q.append("[PAD]")
            aux_q.append("[PAD]")
            aux_q_mask.append(0)
            aux_q_mask.append(0)
        assert len(aux_q) == max_aux*2

        orig_input_tokens = hrt + aux_q
        orig_input_mask = hrt_mask + aux_q_mask
        assert len(orig_input_tokens) == max_seq_length and len(orig_input_mask) == max_seq_length

        # generate a feature by masking each of the tokens
        for mask_position in range(max_seq_length):
            if orig_input_tokens[mask_position] == "[PAD]":
                continue
            mask_label = vocabulary.vocab[orig_input_tokens[mask_position]]
            mask_type = 1 if mask_position % 2== 0 else -1

            input_tokens = orig_input_tokens[:]
            input_tokens[mask_position] = "[MASK]"
            input_ids = vocabulary.convert_tokens_to_ids(input_tokens)
            assert len(input_tokens) == max_seq_length and len(input_ids) == max_seq_length

            feature = NaryFeature(
                feature_id=feature_id,
                example_id=example_id,
                input_tokens=input_tokens,
                input_ids=input_ids,
                input_mask=orig_input_mask,
                mask_position=mask_position,
                mask_label=mask_label,
                mask_type=mask_type,
                arity=example.arity)
            features.append(feature)
            feature_id += 1

    return features

class MultiDataset(Dataset.Dataset):
    def __init__(self, vocabulary: Vocabulary, examples, max_arity=2, max_seq_length=3):
        self.examples = examples
        self.vocabulary = vocabulary
        self.max_arity = max_arity
        self.max_seq_length = max_seq_length
        self.features = convert_examples_to_features(
            examples=self.examples,
            vocabulary=self.vocabulary,
            max_arity=self.max_arity,
            max_seq_length=self.max_seq_length)
        self.multidataset = []
        for feature in self.features:
            feature_out = [feature.input_ids] + [feature.input_mask] + \
                [feature.mask_position] + [feature.mask_label] + [feature.mask_type]
            self.multidataset.append(feature_out)
    def __len__(self):
        return len(self.multidataset)
    def __getitem__(self,index):        
        x = self.multidataset[index]
        batch_data = prepare_batch_data(x, self.vocabulary, self.max_arity, self.max_seq_length)
        return batch_data
        
def prepare_batch_data(inst, vocabulary: Vocabulary, max_arity, max_seq_length):
    # inst: [input_ids, input_mask, mask_position, mask_label, query_type]
    input_ids = np.array(inst[0]).astype("int64")
    input_mask = np.array(inst[1]).astype("int64")
    mask_position = np.array(inst[2]).astype("int64")
    mask_label = np.array(inst[3]).astype("int64")
    query_type = np.array(inst[4]).astype("int64")
    input_mask = np.outer(input_mask, input_mask).astype("bool")
    
    # edge labels between input nodes (used for GRAN-hete)
    #     0: no edge
    #     1: relation-subject
    #     2: relation-object
    #     3: relation-attribute
    #     4: attribute-value
    edge_labels = []
    max_aux = max_arity - 2
    edge_labels.append([0, 1, 2] + [3,4] * max_aux )
    edge_labels.append([1, 0, 5] + [6,7] * max_aux )
    edge_labels.append([2, 5, 0] + [8,9] * max_aux )
    for idx in range(max_aux):
        edge_labels.append([3,6,8] + [11,12] * idx + [0,10] + [11,12] * (max_aux - idx - 1))
        edge_labels.append([4,7,9] + [12,13] * idx + [10,0] + [12,13] * (max_aux - idx - 1))
    edge_labels = np.asarray(edge_labels).astype("int64")
    mask_output = np.zeros(len(vocabulary.vocab)).astype("bool")
    if query_type == -1:
        mask_output[2:2+vocabulary.num_relations] = True
    else:
        mask_output[2+vocabulary.num_relations:] = True
    
    return input_ids, input_mask, mask_position, mask_label, mask_output, edge_labels, query_type