import unittest


class TestTorchStage(unittest.TestCase):
    def test_torch_available_is_false_when_torch_missing(self):
        from oxidata.torch_stage import torch_available

        self.assertFalse(torch_available())

    def test_tensor_tree_to_torch_requires_torch(self):
        from oxidata.torch_stage import tensor_tree_to_torch

        with self.assertRaises(RuntimeError):
            tensor_tree_to_torch({"x": 1})

    def test_stage_tree_to_device_requires_torch(self):
        from oxidata.torch_stage import stage_tree_to_device

        with self.assertRaises(RuntimeError):
            stage_tree_to_device({"x": 1}, "cuda")


if __name__ == "__main__":
    unittest.main()
