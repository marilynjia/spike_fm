"""Replace POYO's unit lookup table with an encoder of unit features.

POYO uses ``model.unit_emb`` in three ways: ``tokenizer(unit_ids)`` maps unit ids
to global indices, ``is_lazy()`` checks that it is ready, and calling it on the
indices returns one embedding per spike. UnitFeatureEncoder keeps the same
interface, but computes each embedding from the unit's features (unit_features.py)
with a small MLP instead of looking it up, so POYO itself is unchanged.

Usage (after ``model.init_vocabs(dataset)``):
    model.unit_emb = UnitFeatureEncoder.from_model(model, dataset)
"""

from collections import OrderedDict

import torch
from hydra.utils import instantiate
from torch import nn

from poyo.unit_features import build_feature_table


class UnitFeatureEncoder(nn.Module):
    def __init__(self, vocab, features, dim, hidden_dim=128):
        super().__init__()
        self.vocab = OrderedDict(vocab)
        self.register_buffer("features", torch.as_tensor(features, dtype=torch.float32))
        self.mlp = nn.Sequential(
            nn.Linear(self.features.shape[1], hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    @classmethod
    def from_model(cls, model, dataset, hidden_dim=128, waveforms=False):
        """Build from a POYO whose vocabularies were initialized on `dataset`."""
        vocab = model.unit_emb.vocab
        features = build_feature_table(dataset, vocab, waveforms)
        return cls(vocab, features, model.unit_emb.embedding_dim, hidden_dim)

    def tokenizer(self, words):
        if isinstance(words, str):
            return self.vocab[words]
        return [self.vocab[w] for w in words]

    def is_lazy(self):
        return False

    def forward(self, input):
        return self.mlp(self.features[input])


def load_with_encoder(model_cfg, checkpoint, dataset, hidden_dim=128, waveforms=False):
    """Load a checkpoint trained with UnitFeatureEncoder.

    POYO.load_pretrained cannot do this: it rebuilds the lookup table, which has
    different weights. The vocabularies are rebuilt from `dataset`, so it must
    hold the same recordings as during training.
    """
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"]
    model = instantiate(model_cfg, dim_out=state_dict["readout.weight"].size(0))
    model.init_vocabs(dataset)
    model.unit_emb = UnitFeatureEncoder.from_model(model, dataset, hidden_dim, waveforms)
    model.load_state_dict(state_dict)
    return model
