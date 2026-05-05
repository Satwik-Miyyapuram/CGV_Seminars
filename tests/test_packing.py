import unittest
import numpy as np
import os
import shutil
import tempfile
import zstandard as zstd
from tools.pack_for_web import pack_checkpoint, quantize_sh

class TestPacking(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_quantize_sh(self):
        # SH coefficients usually range roughly -1 to 1 but can be larger
        sh = np.array([0.5, -0.2, 0.1, 1.5], dtype=np.float32)
        quantized = quantize_sh(sh)
        self.assertEqual(quantized.dtype, np.float16)
        # Check if values are reasonably close after quantization
        np.testing.assert_allclose(quantized.astype(np.float32), sh, atol=1e-3)

    def test_pack_checkpoint(self):
        # Mock data: 100 surfels, each with some float32 properties
        num_primitives = 100
        # For simplicity, let's say a primitive is just position(3) and SH(4)
        positions = np.random.rand(num_primitives, 3).astype(np.float32)
        sh = np.random.rand(num_primitives, 4).astype(np.float32)
        
        output_path = os.path.join(self.test_dir, "packed.bin")
        pack_checkpoint(output_path, {'pos': positions, 'sh': sh}, {})
        
        self.assertTrue(os.path.exists(output_path))
        
        # Decompress and check size
        dctx = zstd.ZstdDecompressor()
        with open(output_path, 'rb') as f:
            compressed_data = f.read()
            decompressed_data = dctx.decompress(compressed_data)
            
        # positions (100*3*4 bytes) + quantized sh (100*4*2 bytes) = 1200 + 800 = 2000
        expected_size = num_primitives * 3 * 4 + num_primitives * 4 * 2
        self.assertEqual(len(decompressed_data), expected_size)

if __name__ == "__main__":
    unittest.main()
