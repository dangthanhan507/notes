---
layout: default
title: Image Augmentation (Torchvision)
parent: Code Notes
nav_order: 3
---

# Image Augmentation with Torchvision

https://docs.pytorch.org/vision/main/transforms.html

Torchvision now has v1 and v2 transforms. It is recommended to try v2 now. Here is an example of how to use v2 transforms for image augmentation:


```python
from torchvision.transforms import v2
transforms = v2.Compose([
    v2.ToImage(),  # Convert to tensor, only needed if you had a PIL image
    v2.ToDtype(torch.uint8, scale=True),  # optional, most input are already uint8 at this point
    # ...
    v2.RandomResizedCrop(size=(224, 224), antialias=True),  # Or Resize(antialias=True)
    # ...
    v2.ToDtype(torch.float32, scale=True),  # Normalize expects float input
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

reasons to use v2:
- faster
- supports arbitrary inputs (dicts, list, tuples)
- support is moved to v2 now.
- can transform bbox masks and videos.
- backwards compatible with v1 transforms (no loss).




