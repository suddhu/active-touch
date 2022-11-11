import os
from PIL import Image
import numpy as np
import trimesh
import warnings

# warnings.filterwarnings("ignore")

import meshplot as mp

mp.offline()
from pterotactyl.simulator.scene import sampler
from pterotactyl.simulator.physics import grasping
from pterotactyl.utility import utils
import pterotactyl.objects as objects

OBJ_LOCATION = os.path.join(os.path.dirname(objects.__file__), "test_objects/0")
print(OBJ_LOCATION)
batch = [OBJ_LOCATION]

verts, faces = utils.load_mesh_touch(OBJ_LOCATION + ".obj")
plot = mp.plot(verts.data.cpu().numpy(), faces.data.cpu().numpy())


s = sampler.Sampler(grasping.Agnostic_Grasp, bs=1, vision=True, resolution=[256, 256])
s.load_objects(batch, from_dataset=False, scale=2.6)


action = [30]
parameters = [[[0.3, 0.3, 0.3], [60, 0, 135]]]
signals = s.sample(
    action,
    touch=True,
    touch_point_cloud=True,
    vision=True,
    vision_occluded=True,
    parameters=parameters,
)


img_vision = Image.fromarray(signals["vision"][0])
img_vision.show()

img_vision_grasp = Image.fromarray(signals["vision_occluded"][0])
img_vision_grasp.show()


image = np.zeros((121 * 4, 121 * 2, 3)).astype(np.uint8)
for i in range(4):
    print(f'Finger {i} has status {signals["touch_status"][0][i]}')
    touch = signals["touch_signal"][0][i].data.numpy().astype(np.uint8)
    image[i * 121 : i * 121 + 121, :121] = touch
    depth = utils.visualize_depth(signals["depths"][0][i].data.numpy()).reshape(
        121, 121, 1
    )
    image[i * 121 : i * 121 + 121, 121:] = depth
print(" ")
print("     TOUCH         DEPTH")
Image.fromarray(image).show()
