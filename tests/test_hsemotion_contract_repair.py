"""Synthetic HSEmotion contract regressions. Never loads weights or datasets."""
import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


class SyntheticSession:
    def __init__(self, logits, valence=0.25, arousal=-0.5):
        self.row = np.array([list(logits) + [valence, arousal]], dtype=np.float32)
        self.tensor = None

    def get_inputs(self):
        return [SimpleNamespace(name="synthetic")]

    def run(self, _outputs, inputs):
        self.tensor = inputs["synthetic"].copy()
        return [self.row.copy()]


class HSEmotionContractRepairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = Path(__file__).resolve().parents[1] / "emolab/adapters/claude_hsemotion.py"
        spec = importlib.util.spec_from_file_location("synthetic_hse_contract_under_test", source)
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def adapter(self, logits=None, valence=0.25, arousal=-0.5):
        adapter = self.module.HSEmotionAdapter(model_path="SYNTHETIC_DO_NOT_OPEN")
        adapter._session = SyntheticSession([0] * 8 if logits is None else logits,
                                           valence, arousal)
        return adapter

    def image(self):
        return np.zeros((4, 4, 3), dtype=np.uint8)

    def test_affect_outputs_do_not_enter_expression_softmax(self):
        first = self.adapter(valence=0, arousal=0).predict_proba(self.image())
        second = self.adapter(valence=50, arousal=-50).predict_proba(self.image())
        np.testing.assert_allclose(list(first.probabilities.values()),
                                   list(second.probabilities.values()), atol=1e-7)

    def test_supported_distribution_is_normalised(self):
        result = self.adapter().predict_proba(self.image())
        self.assertAlmostEqual(sum(result.probabilities.values()), 1.0, places=6)
        self.assertEqual(set(result.probabilities), set(result.metadata.supported_classes))

    def test_full_native_distribution_keeps_contempt(self):
        result = self.adapter().predict_proba(self.image())
        native = result.metadata.provenance["native8_probabilities"]
        self.assertEqual(len(native), 8)
        self.assertIn("contempt", native)
        self.assertEqual(set(native), set(self.module.HSEmotionAdapter.native_labels))
        np.testing.assert_allclose(list(native.values()), [0.125] * 8, atol=1e-7)
        self.assertAlmostEqual(result.metadata.provenance["retained_mass"], 0.875)
        self.assertTrue(result.metadata.provenance["conditional_fer7"])

    def test_contempt_dominant_conditional_is_not_fabricated_uniform(self):
        logits = np.array([1, 40, -2, 3, 0, -4, 2, -1], dtype=np.float64)
        adapter = self.adapter(logits)
        result = adapter.predict_proba(self.image())
        names = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
        indices = [0, 2, 3, 4, 6, 7, 5]
        scaled = logits[indices] / adapter.temperature
        expected = np.exp(scaled - scaled.max())
        expected /= expected.sum()
        np.testing.assert_allclose([result.probabilities[k] for k in names], expected,
                                   rtol=1e-5, atol=1e-6)
        self.assertGreater(result.metadata.provenance["retained_mass"], 0)

    def test_zero_retained_mass_fails_closed(self):
        with self.assertRaises(ValueError):
            self.adapter([0, 1e6, 0, 0, 0, 0, 0, 0]).predict_proba(self.image())

    def test_existing_face_rgb_keyword_remains_supported(self):
        result = self.adapter().predict_proba(face_rgb=self.image())
        self.assertAlmostEqual(sum(result.probabilities.values()), 1, places=6)

    def test_bgr_boundary_preserves_red_pixel(self):
        adapter = self.adapter()
        image = self.image()
        image[:, :, 2] = 255
        adapter.predict_proba(image)
        np.testing.assert_allclose(adapter._session.tensor[0, :, 0, 0],
                                   [2.2489082969, -2.0357142857, -1.8044444444],
                                   atol=1e-6)

    def test_valence_arousal_and_frozen_temperature_are_preserved(self):
        adapter = self.adapter(valence=0.25, arousal=-0.5)
        result = adapter.predict_proba(self.image())
        self.assertEqual(result.metadata.provenance["valence"], 0.25)
        self.assertEqual(result.metadata.provenance["arousal"], -0.5)
        self.assertEqual(adapter.temperature, 1.3407638433649585)


if __name__ == "__main__":
    unittest.main()
