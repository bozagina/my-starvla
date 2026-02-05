


def get_vlm_model(config):

    vlm_name = config.framework.mapanything_llava3d.base_vlm

    if "Qwen2.5-VL" in vlm_name or "nora" in vlm_name.lower():
        from .QWen2_5 import _QWen_VL_Interface 
        return _QWen_VL_Interface(config)
    elif "Qwen3-VL" in vlm_name:
        from .QWen3 import _QWen3_VL_Interface

        return _QWen3_VL_Interface(config)
    elif "florence" in vlm_name.lower():
        from .Florence2 import _Florence_Interface 
        return _Florence_Interface(config)
    elif "mapanything_llava3d" in vlm_name.lower():
        from .MapAnythingLlava3D import _MapAnythingLlava3D_Interface
        return _MapAnythingLlava3D_Interface(config)
    else:
        raise NotImplementedError(f"VLM model {vlm_name} not implemented")


