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

    NVIDIA_H20 = AcceleratorInfo.from_dict(
        config={
            "name": "NVIDIA H20",
            "device_alias": ["H20", "h20_sxm"],
            "tflops": {
                "FP8_TENSOR": 296,
                "INT8_TENSOR": 296,
                "FP16_TENSOR": 148,
                "BF16_TENSOR": 148,
                "FP32": 74,
            },
            "hbm_capacity_gb": 96,
            "hbm_bandwidth_gb": 4022,
            "inter_node_bandwidth_gb": 64,
            "intra_node_bandwidth_gb": 450,
            "vendor": "NVIDIA",
            "ref": "https://viperatech.com/product/nvidia-hgx-h20",
        },
        save_to_registry=True,
    )
