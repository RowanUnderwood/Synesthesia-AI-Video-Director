import unittest
from unittest import mock

import gpu_power


SAMPLE_SMI = """0, NVIDIA GeForce RTX 5090, 450.00, 575.00, 400.00, 600.00
1, NVIDIA GeForce RTX 4090, 350.00, 450.00, 10.00, 479.00
2, NVIDIA GeForce RTX 3090, 280.00, 350.00, 100.00, 385.00"""


class GpuPowerTests(unittest.TestCase):
    def test_query_and_role_resolution_use_names_not_cuda_order(self):
        with mock.patch.object(gpu_power, "_run", return_value=(0, SAMPLE_SMI)):
            cards = gpu_power.query_gpus()
        roles = gpu_power.resolve_roles(cards)
        self.assertEqual(roles["5090"]["index"], 0)
        self.assertEqual(roles["4090"]["index"], 1)
        self.assertEqual(roles["3090"]["index"], 2)

    def test_clamp_uses_hardware_bounds(self):
        with mock.patch.object(gpu_power, "_run", return_value=(0, SAMPLE_SMI)):
            cards = gpu_power.query_gpus()
        self.assertEqual(gpu_power.clamp_watts("5090", 200, cards), 400)
        self.assertEqual(gpu_power.clamp_watts("4090", 999, cards), 479)
        self.assertEqual(gpu_power.clamp_watts("3090", 280, cards), 280)

    def test_apply_builds_nvidia_smi_targets_from_resolved_names(self):
        with mock.patch.object(gpu_power, "_run", return_value=(0, SAMPLE_SMI)), \
             mock.patch.object(gpu_power, "_dispatch", return_value=(True, "ok")) as dispatch:
            ok, _message = gpu_power.apply_limits({"5090": 450, "4090": 350, "3090": 280})
        self.assertTrue(ok)
        self.assertEqual(dispatch.call_args.args[0], "apply")
        self.assertEqual(dispatch.call_args.args[1], [
            {"index": 0, "watts": 450},
            {"index": 1, "watts": 350},
            {"index": 2, "watts": 280},
        ])

    def test_public_query_degrades_when_nvidia_smi_is_missing(self):
        with mock.patch.object(gpu_power, "_run", return_value=(127, "not found")):
            self.assertEqual(gpu_power.query_gpus(), [])
            self.assertIn("unavailable", gpu_power.current_state())


if __name__ == "__main__":
    unittest.main()
