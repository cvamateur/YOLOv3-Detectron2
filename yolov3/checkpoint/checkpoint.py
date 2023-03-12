from detectron2.checkpoint import DetectionCheckpointer

from .load_darknet import load_darknet_weights_to_dict


class YOLOV3Checkpointer(DetectionCheckpointer):
    """
    Same as :class:`DetectionCheckpointer`, but can load official darknet weights.
    """

    def _load_file(self, filename):
        if filename.endswith(".weights"):
            if "darknet53" in filename:
                darknet_convs = self.model.backbone.bottom_up.darknet_modules()
                field = "backbone.bottom_up"
            elif "yolov3" in filename:
                darknet_convs = self.model.backbone.darknet_modules()
                field = "backbone"
            else:
                raise RuntimeError("weights invalid")

            with self.path_manager.open(filename, "rb") as f:
                loaded = load_darknet_weights_to_dict(f, darknet_convs, field)
        else:
            loaded = super()._load_file(filename)  # load native pth checkpoint
        if "model" not in loaded:
            loaded = {"model": loaded}
        return loaded
