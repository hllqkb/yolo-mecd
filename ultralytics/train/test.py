# 重要：检验图片和标签是否匹配u can also use labelImg to check the label.
from ultralytics.utils import verify_image_label
verify_image_label("/path/to/train/images/img001.jpg", "/path/to/train/labels/img001.txt")