


def get_vlm_model(config):

    fw = getattr(config, "framework", None)
    vlm_name = None
    if fw is not None:
        fw_name = getattr(fw, "name", None)
        if fw_name in ("QwenPI", "QwenOFT", "QwenFast"):
            qwenvl_cfg = getattr(fw, "qwenvl", None)
            if qwenvl_cfg is not None:
                vlm_name = getattr(qwenvl_cfg, "base_vlm", None)
        elif fw_name == "MapAnythingLlava3DPI":
            ma_cfg = getattr(fw, "mapanything_llava3d", None)
            if ma_cfg is not None:
                vlm_name = getattr(ma_cfg, "base_vlm", None)

    if not vlm_name:
        ma_cfg = getattr(getattr(fw, "mapanything_llava3d", None), "base_vlm", None) if fw is not None else None
        qw_cfg = getattr(getattr(fw, "qwenvl", None), "base_vlm", None) if fw is not None else None
        vlm_name = ma_cfg or qw_cfg

    if "Qwen2.5-VL" in vlm_name or "nora" in vlm_name.lower():
        from .QWen2_5 import _QWen_VL_Interface 
        return _QWen_VL_Interface(config)
    elif "Qwen3-VL" in vlm_name:
        from .QWen3 import _QWen3_VL_Interface

        return _QWen3_VL_Interface(config)
    elif "florence" in vlm_name.lower():
        from .Florence2 import _Florence_Interface 
        return _Florence_Interface(config)
    elif "mapanything_llava3d" in vlm_name.lower() or "mapanythingllava3d" in vlm_name.lower():
        from .MapAnythingLlava3D import _MapAnythingLlava3D_Interface
        return _MapAnythingLlava3D_Interface(config)
    else:
        raise NotImplementedError(f"VLM model {vlm_name} not implemented")
