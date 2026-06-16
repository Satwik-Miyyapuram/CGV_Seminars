import unittest
import sys
import os
# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import tempfile
import shutil
from pathlib import Path
from plyfile import PlyData
from tools.pack_for_web import (
    restrict_primitives,
    export_to_ply,
    export_combined_ply,
    load_model_state,
    extract_primitives,
    PRIM_TYPE_SURFEL,
    PRIM_TYPE_GAUSSIAN,
)

class TestPacking(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_restrict_primitives(self):
        # Create dummy tensors representing 5 surfels/gaussians
        tensor_dict = {
            "means": torch.tensor([[0., 0., 0.], [1., 1., 1.], [2., 2., 2.], [3., 3., 3.], [4., 4., 4.]]),
            "scales": torch.tensor([[1., 1., 1.]] * 5),
            "quats": torch.tensor([[1., 0., 0., 0.]] * 5),
            # Opacities are pre-sigmoid: 2.0, 1.0, 4.0, 3.0, 0.0
            # Sliced order by opacity descending: index 2 (4.0), index 3 (3.0), index 0 (2.0), index 1 (1.0), index 4 (0.0)
            "opacities": torch.tensor([[2.0], [1.0], [4.0], [3.0], [0.0]]),
            "features_dc": torch.tensor([[0.5, 0.5, 0.5]] * 5),
        }

        # Restrict to top 3
        restricted = restrict_primitives(tensor_dict, 3)

        # The count should be 3
        self.assertEqual(restricted["means"].shape[0], 3)
        # Check that the kept ones are indeed the ones with highest opacity (indices 2, 3, 0)
        # Expected means: [2, 2, 2] (index 2), [3, 3, 3] (index 3), [0, 0, 0] (index 0)
        expected_means = torch.tensor([[2., 2., 2.], [3., 3., 3.], [0., 0., 0.]])
        self.assertTrue(torch.allclose(restricted["means"], expected_means))

        # Check expected opacities
        expected_opacities = torch.tensor([[4.0], [3.0], [2.0]])
        self.assertTrue(torch.allclose(restricted["opacities"], expected_opacities))

    def test_export_to_ply(self):
        # Verify export runs and writes file successfully
        tensor_dict = {
            "means": torch.tensor([[0., 0., 0.], [1., 1., 1.]]),
            "scales": torch.tensor([[1., 1., 1.], [2., 2., 2.]]),
            "quats": torch.tensor([[1., 0., 0., 0.], [1., 0., 0., 0.]]),
            "opacities": torch.tensor([[1.0], [0.5]]),
            "features_dc": torch.tensor([[0.5, 0.5, 0.5], [0.1, 0.2, 0.3]]),
        }
        out_file = Path(self.test_dir) / "test.ply"
        export_to_ply(tensor_dict, out_file)
        self.assertTrue(out_file.exists())
        self.assertTrue(out_file.stat().st_size > 0)

    def _make_primitives(self, n, sh_rest_bases=15):
        return {
            "means": torch.randn(n, 3),
            "scales": torch.randn(n, 3),
            "quats": torch.randn(n, 4),
            "opacities": torch.randn(n, 1),
            "features_dc": torch.randn(n, 3),
            "features_rest": torch.randn(n, sh_rest_bases, 3),
        }

    def test_export_combined_ply_tags_prim_type(self):
        surfels = self._make_primitives(3)
        gaussians = self._make_primitives(5)
        out_file = Path(self.test_dir) / "scene.ply"
        export_combined_ply(surfels, gaussians, out_file, sh_degree=3)

        self.assertTrue(out_file.exists())
        ply = PlyData.read(str(out_file))
        verts = ply["vertex"]

        # All 8 primitives present, tagged correctly (surfels first, then gaussians).
        self.assertEqual(len(verts), 8)
        prim = verts["prim_type"]
        self.assertEqual(int((prim == PRIM_TYPE_SURFEL).sum()), 3)
        self.assertEqual(int((prim == PRIM_TYPE_GAUSSIAN).sum()), 5)
        np.testing.assert_array_equal(prim[:3], PRIM_TYPE_SURFEL)
        np.testing.assert_array_equal(prim[3:], PRIM_TYPE_GAUSSIAN)

        # SH rest coefficients present at degree 3 (15 bases * 3 channels = 45).
        self.assertIn("f_rest_44", verts.data.dtype.names)
        self.assertNotIn("f_rest_45", verts.data.dtype.names)

    def test_export_combined_ply_handles_empty_gaussians(self):
        surfels = self._make_primitives(4)
        gaussians = {k: v[:0] for k, v in self._make_primitives(0, sh_rest_bases=15).items()}
        out_file = Path(self.test_dir) / "scene.ply"
        export_combined_ply(surfels, gaussians, out_file, sh_degree=3)

        ply = PlyData.read(str(out_file))
        verts = ply["vertex"]
        self.assertEqual(len(verts), 4)
        np.testing.assert_array_equal(verts["prim_type"], PRIM_TYPE_SURFEL)

    def test_export_combined_sh_degree_zero_has_no_rest(self):
        surfels = self._make_primitives(2)
        gaussians = self._make_primitives(2)
        out_file = Path(self.test_dir) / "scene.ply"
        export_combined_ply(surfels, gaussians, out_file, sh_degree=0)

        ply = PlyData.read(str(out_file))
        names = ply["vertex"].data.dtype.names
        self.assertFalse(any(n.startswith("f_rest_") for n in names))

    def test_opacity_stored_raw_logit(self):
        # The @mkkellogg loader applies sigmoid on load, so we must store raw logit.
        opac = torch.tensor([[-3.0], [0.0], [5.0]])
        tensor_dict = {
            "means": torch.zeros(3, 3),
            "scales": torch.zeros(3, 3),
            "quats": torch.tensor([[1., 0., 0., 0.]] * 3),
            "opacities": opac,
            "features_dc": torch.zeros(3, 3),
        }
        out_file = Path(self.test_dir) / "raw.ply"
        export_to_ply(tensor_dict, out_file, sh_degree=0)
        verts = PlyData.read(str(out_file))["vertex"]
        np.testing.assert_allclose(np.array(verts["opacity"]), opac.squeeze().numpy(), atol=1e-5)

    def test_non_finite_primitives_dropped(self):
        prims = self._make_primitives(5)
        prims["scales"][2, 1] = float("nan")
        prims["means"][4, 0] = float("inf")
        out_file = Path(self.test_dir) / "finite.ply"
        export_to_ply(prims, out_file, sh_degree=3)
        verts = PlyData.read(str(out_file))["vertex"]
        # rows 2 and 4 dropped -> 3 remain, all finite
        self.assertEqual(len(verts), 3)
        for n in verts.data.dtype.names:
            self.assertTrue(np.isfinite(np.array(verts[n])).all(), f"{n} has non-finite values")

    def test_load_model_state_ckpt_and_pth(self):
        # Build a minimal model-style state dict for both formats.
        base = {
            "surfel.means": torch.randn(3, 3),
            "surfel.quats": torch.randn(3, 4),
            "surfel.scales": torch.randn(3, 3),
            "surfel.opacities": torch.randn(3, 1),
            "surfel.features_dc": torch.randn(3, 3),
            "surfel.features_rest": torch.randn(3, 15, 3),
            "gaussian.means": torch.randn(2, 3),
            "gaussian.quats": torch.randn(2, 4),
            "gaussian.scales": torch.randn(2, 3),
            "gaussian.opacities": torch.randn(2, 1),
            "gaussian.features_dc": torch.randn(2, 3),
            "gaussian.features_rest": torch.randn(2, 15, 3),
        }

        # .pth: raw model state dict, no prefix.
        pth_path = Path(self.test_dir) / "milestone.pth"
        torch.save(base, pth_path)
        state, prefix = load_model_state(pth_path)
        self.assertEqual(prefix, "")
        surfels = extract_primitives(state, prefix, "surfel")
        self.assertEqual(surfels["means"].shape[0], 3)

        # .ckpt: wrapped under "pipeline" with "_model." prefix.
        ckpt = {"pipeline": {f"_model.{k}": v for k, v in base.items()}}
        ckpt_path = Path(self.test_dir) / "step.ckpt"
        torch.save(ckpt, ckpt_path)
        state, prefix = load_model_state(ckpt_path)
        self.assertEqual(prefix, "_model.")
        gaussians = extract_primitives(state, prefix, "gaussian")
        self.assertEqual(gaussians["means"].shape[0], 2)

if __name__ == "__main__":
    unittest.main()
