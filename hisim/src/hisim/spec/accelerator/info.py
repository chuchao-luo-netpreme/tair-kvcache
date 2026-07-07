from hisim.spec.accelerator.base import AcceleratorInfo


class NVIDIA:
    NVIDIA_H100 = AcceleratorInfo.from_dict(
        config={
            "name": "NVIDIA H100",
            "device_alias": ["H100", "h100_sxm"],
            "tflops": {
                "FP8_TENSOR": 3958,
                "INT8_TENSOR": 3958,
                "FP16_TENSOR": 1979,
                "BF16_TENSOR": 1979,
                "FP32": 67,
            },
            "hbm_capacity_gb": 80,
            "hbm_bandwidth_gb": 3350,
            "inter_node_bandwidth_gb": 400,
            "intra_node_bandwidth_gb": 450,
            "vendor": "NVIDIA",
            "ref": "https://www.nvidia.com/en-us/data-center/h100/",
        },
        save_to_registry=True,
    )

    NVIDIA_H200 = AcceleratorInfo.from_dict(
        config={
            "name": "NVIDIA H200",
            "device_alias": ["H200", "h200_sxm"],
            "tflops": {
                "FP8_TENSOR": 3958,
                "INT8_TENSOR": 3958,
                "FP16_TENSOR": 1979,
                "BF16_TENSOR": 1979,
                "FP32": 67,
            },
            "hbm_capacity_gb": 141,
            "hbm_bandwidth_gb": 4800,
            "inter_node_bandwidth_gb": 400,
            "intra_node_bandwidth_gb": 450,
            "vendor": "NVIDIA",
            "ref": "https://www.nvidia.com/en-us/data-center/h200/",
        },
        save_to_registry=True,
    )

    NVIDIA_B200 = AcceleratorInfo.from_dict(
        config={
            "name": "NVIDIA B200",
            "device_alias": ["B200", "b200_sxm"],
            "tflops": {
                "FP4_TENSOR": 9000,
                "FP8_TENSOR": 4500,
                "INT8_TENSOR": 4500,
                "FP16_TENSOR": 2250,
                "BF16_TENSOR": 2250,
                "FP32": 90,
            },
            "hbm_capacity_gb": 180,
            "hbm_bandwidth_gb": 8000,
            "inter_node_bandwidth_gb": 50,
            "intra_node_bandwidth_gb": 900,
            "vendor": "NVIDIA",
            "ref": "https://www.nvidia.com/en-us/data-center/b200/",
        },
        save_to_registry=True,
    )

    NVIDIA_GB300 = AcceleratorInfo.from_dict(
        config={
            "name": "NVIDIA GB300",
            "device_alias": ["GB300", "gb300"],
            "tflops": {
                "FP4_TENSOR": 15000,
                "FP8_TENSOR": 5000,
                "INT8_TENSOR": 165,
                "FP16_TENSOR": 2500,
                "BF16_TENSOR": 2500,
                "FP32": 83,
            },
            "hbm_capacity_gb": 298,
            "hbm_bandwidth_gb": 8000,
            "inter_node_bandwidth_gb": 100,
            "intra_node_bandwidth_gb": 900,
            "vendor": "NVIDIA",
            "ref": "https://www.nvidia.com/en-us/data-center/gb300-nvl72/",
        },
        save_to_registry=True,
    )
