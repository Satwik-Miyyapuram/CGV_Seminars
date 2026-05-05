import unittest
import os
import json
import shutil
import tempfile
from tools.run_optim import save_checkpoint, AtomicWriter

class TestOptimDriver(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.checkpoint_dir = os.path.join(self.test_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_atomic_write(self):
        filepath = os.path.join(self.test_dir, "test.txt")
        content = "hello world"
        with AtomicWriter(filepath) as f:
            f.write(content)
        
        self.assertTrue(os.path.exists(filepath))
        with open(filepath, "r") as f:
            self.assertEqual(f.read(), content)

    def test_save_checkpoint_logging(self):
        # Mock data
        iteration = 5000
        data = {"iter": iteration, "loss": 0.05}
        
        save_checkpoint(self.checkpoint_dir, iteration, {}, data)
        
        json_path = os.path.join(self.checkpoint_dir, f"iter_{iteration:05d}.json")
        self.assertTrue(os.path.exists(json_path))
        
        with open(json_path, "r") as f:
            saved_data = json.load(f)
            self.assertEqual(saved_data["iter"], iteration)
            self.assertEqual(saved_data["loss"], 0.05)

if __name__ == "__main__":
    unittest.main()
