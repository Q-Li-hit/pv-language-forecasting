"""Sequence-model checks using numerical inputs and a small causal module."""
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np

try:
    import torch
    from torch import nn
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "Install the sequence extra to test PyTorch models.")
class SequenceTests(unittest.TestCase):
    def setUp(self):
        from pvforecast.sequence import ForecastModel, SequenceConfig

        self.Model = ForecastModel
        self.cfg = SequenceConfig(
            d_model=8, heads=2, layers=1, dropout=0.0,
            patch_length=4, patch_stride=2, time_llm_prototypes=5,
            full_backbone_layers=2,
        )
        torch.manual_seed(18)
        self.history = torch.randn(3, 12)
        self.future = torch.randn(3, 4, 6)
        self.ids = torch.tensor([[1, 2, 3], [1, 4, 0], [1, 0, 0]])
        self.mask = self.ids.ne(0).to(torch.long)

    def tiny_model(self, **kwargs):
        class CausalModule(nn.Module):
            def __init__(self, config=None):
                super().__init__()
                self.config = config or types.SimpleNamespace(num_hidden_layers=2, use_cache=False)
                self.embedding = nn.Embedding(24, 12)
                self.linear = nn.Linear(12, 12)

            def get_input_embeddings(self):
                return self.embedding

            def forward(self, inputs_embeds, attention_mask, **kwargs):
                states = torch.tanh(self.linear(inputs_embeds)) * attention_mask.unsqueeze(-1)
                return types.SimpleNamespace(last_hidden_state=states.cumsum(dim=1))

        return self.Model(
            "time_llm_qwen", 12, 6, 4, config=self.cfg,
            backbone=CausalModule(), random_backbone_factory=CausalModule,
            **kwargs,
        )

    def test_numeric_models_train_without_transformers(self):
        for variant in ("dense_numeric", "patchtst"):
            with self.subTest(variant=variant), patch.dict(sys.modules, {"transformers": None}):
                model = self.Model(variant, 12, 6, 4, config=self.cfg)
                result = model(self.history, self.future)
                self.assertEqual(tuple(result.shape), (3, 4))
                loss = result.square().mean()
                loss.backward()
                self.assertTrue(torch.isfinite(result).all())
                self.assertGreater(model.head.weight.grad.abs().sum().item(), 0)

    def test_frozen_language_module_passes_gradients_to_task_parameters(self):
        model = self.tiny_model().train()
        self.assertFalse(model.backbone.training)
        result = model(self.history, self.future, self.ids, self.mask)
        result.square().mean().backward()
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in model.backbone.parameters()))
        self.assertGreater(model.patch_embed.weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.source_mapping.weight.grad.abs().sum().item(), 0)

    def test_original_task_keys_load_and_restore_predictions(self):
        model = self.tiny_model().eval()
        with torch.no_grad():
            expected = model(self.history, self.future, self.ids, self.mask)
        state = model.trainable_state()
        self.assertIn("source_mapping.weight", state)
        self.assertIn("reprogram_output.weight", state)
        self.assertFalse(any(key.startswith("backbone.") for key in state))
        with torch.no_grad():
            model.head.bias.add_(4)
        model.load_trainable_state(state)
        with torch.no_grad():
            actual = model(self.history, self.future, self.ids, self.mask)
        self.assertTrue(torch.equal(expected, actual))
        with self.assertRaises(ValueError):
            model.load_trainable_state({})

    def test_cache_preserves_predictions_and_releases_tensors(self):
        model = self.tiny_model().eval()
        with torch.no_grad():
            expected = model(self.history, self.future, self.ids, self.mask)
            with model.inference_cache():
                actual = model(self.history, self.future, self.ids, self.mask)
                self.assertTrue(torch.equal(expected, actual))
            self.assertIsNone(model._projection_cache)
            with model.inference_cache():
                model.source_mapping.weight.add_(1)
                with self.assertRaises(RuntimeError):
                    model(self.history, self.future, self.ids, self.mask)
        self.assertIsNone(model._projection_cache)
        with self.assertRaises(RuntimeError):
            with model.train().inference_cache():
                pass

    def test_cache_rejects_gradient_enabled_forward(self):
        model = self.tiny_model().eval()
        with model.inference_cache():
            with self.assertRaises(RuntimeError):
                model(self.history, self.future, self.ids, self.mask)
        with model.inference_cache():
            model.load_trainable_state(model.trainable_state())
            self.assertIsNone(model._projection_cache)
        with model.inference_cache():
            model.to("cpu")
            self.assertIsNone(model._projection_cache)

    def test_random_control_retains_task_initialization_and_rng(self):
        torch.manual_seed(123)
        pretrained = self.tiny_model()
        after_pretrained = torch.random.get_rng_state().clone()
        torch.manual_seed(123)
        random = self.tiny_model(random_backbone=True, head_seed=42)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), after_pretrained))
        self.assertTrue(all(torch.equal(value, random.trainable_state()[key]) for key, value in pretrained.trainable_state().items()))
        self.assertFalse(torch.equal(pretrained.backbone.embedding.weight, random.backbone.embedding.weight))
        torch.manual_seed(123)
        repeated = self.tiny_model(random_backbone=True, head_seed=42)
        self.assertTrue(torch.equal(random.backbone.embedding.weight, repeated.backbone.embedding.weight))

    def test_statistical_and_domain_prompts_preserve_original_text(self):
        from pvforecast.sequence import DOMAIN_PREFIX, build_statistical_prompts

        values = np.array([[0., 1., 2., 3., 4., 5.]])
        basic = build_statistical_prompts(values, "rolling")[0]
        self.assertIn("min 0.0, max 5.0, median 2.5 MW; trend up", basic)
        self.assertIn("12 future five-minute windows", basic)
        domain = build_statistical_prompts(values, "rolling", domain=True)[0]
        self.assertEqual(domain, DOMAIN_PREFIX + basic)
        with self.assertRaises(ValueError):
            build_statistical_prompts(values, "intraday")

    def test_tokenization_rejects_truncation(self):
        from pvforecast.sequence import tokenize_prompts

        class Tokenizer:
            def encode(self, text):
                return list(range(len(text.split())))

            def __call__(self, texts, *, max_length, truncation, **kwargs):
                assert truncation is False
                ids = np.zeros((len(texts), max_length), dtype=int)
                mask = np.zeros_like(ids)
                for index, text in enumerate(texts):
                    count = len(self.encode(text))
                    ids[index, :count] = np.arange(1, count + 1)
                    mask[index, :count] = 1
                return {"input_ids": ids, "attention_mask": mask}

        values = np.arange(12, dtype=float)[None, :]
        ids, mask = tokenize_prompts(values, "day_ahead", Tokenizer())
        self.assertEqual(ids.dtype, np.int32)
        self.assertEqual(mask.dtype, np.int8)
        with self.assertRaises(ValueError):
            tokenize_prompts(values, "day_ahead", Tokenizer(), max_length=3)
        with self.assertRaises(ValueError):
            tokenize_prompts(values, "day_ahead", Tokenizer(), max_length=0)


if __name__ == "__main__":
    unittest.main()
