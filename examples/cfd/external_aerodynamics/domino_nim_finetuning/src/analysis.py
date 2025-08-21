from physicsnemo.utils.domino.utils import *
import numpy as np
# vol_factors = np.array([[ 2.1508515,  1.0027921,  1.0663894,  1.1288369,  0.05063211, 0.00381244], 
#                 [-1.9028450e+00, -1.0032533e+00, -1.0505041e+00, -1.4412953e+00,
#   1.5563720e-18, -2.7427445e-20]], dtype=np.float32)

# surf_factors = np.array([[ 0.98881036, 0.00550783, 0.00854675, 0.00452144], 
#                 [-2.4203062,  -0.00740275, -0.00848471, -0.00448634]], dtype=np.float32)

# 

# filepath = "/lustre/rranade/modulus_dev/data/drivaer_data_finetune_processed/"
filepath = "/lustre/rranade/modulus_dev/data/drivaer_data_baseline_processed/"
filenames = get_filenames(filepath)

max_geom = -1e6*np.ones((3))
min_geom = 1e6*np.ones((3))
for f_ct, j in enumerate(filenames):
    # print(f_ct, j)
    data_dict = np.load(filepath+j, allow_pickle=True).item()
    # print(data_dict.keys())
    stl_vertices = data_dict["stl_coordinates"]
    stl_centers = data_dict["stl_centers"]
    mesh_indices_flattened = data_dict["stl_faces"]
    stl_sizes = data_dict["stl_areas"]
    vol_fields = data_dict["volume_fields"]
    surf_fields = data_dict["surface_fields"]
    surface_sizes = data_dict["surface_areas"]
    # stream_velocity = data_dict["stream_velocity"]
    # air_density = data_dict["air_density"]
    stream_velocity = 38.89
    air_density = 1.226
    length_scale = np.amax(np.amax(stl_vertices, 0) - np.amin(stl_vertices, 0))

    # vol_fields[:, :3] = vol_fields[:, :3] / (stream_velocity)
    # vol_fields[:, 3] = vol_fields[:, 3] / (stream_velocity*stream_velocity*air_density)
    # # vol_fields[:, 4] = vol_fields[:, 4] / (stream_velocity*stream_velocity*air_density)
    # vol_fields[:, 4] = vol_fields[:, 4] / (stream_velocity*length_scale)

    # surf_fields = surf_fields / (stream_velocity*stream_velocity*air_density)

    if f_ct == 0:
        vol_fields_max = np.amax(vol_fields, 0)
        vol_fields_min = np.amin(vol_fields, 0)
    else:
        # if vmax[0] < 2.5 and abs(vmax[3]) < 2.5 and abs(vmin[3]) < 2.5:
        vol_fields_max1 = np.amax(vol_fields, 0)
        vol_fields_min1 = np.amin(vol_fields, 0)

        for k in range(vol_fields.shape[-1]):
            if vol_fields_max1[k] > vol_fields_max[k]:
                vol_fields_max[k] = vol_fields_max1[k]

            if vol_fields_min1[k] < vol_fields_min[k]:
                vol_fields_min[k] = vol_fields_min1[k]

    
    if f_ct == 0:
        surf_fields_max = np.amax(surf_fields, 0)
        surf_fields_min = np.amin(surf_fields, 0)
    else:
        surf_fields_max1 = np.amax(surf_fields, 0)
        surf_fields_min1 = np.amin(surf_fields, 0)

        for k in range(surf_fields.shape[-1]):
            if surf_fields_max1[k] > surf_fields_max[k]:
                surf_fields_max[k] = surf_fields_max1[k]

            if surf_fields_min1[k] < surf_fields_min[k]:
                surf_fields_min[k] = surf_fields_min1[k]
    print(f_ct, j, vol_fields_max, vol_fields_min, surf_fields_max, surf_fields_min)
    if f_ct == 4:
        break

print("Max:", max_geom, "Min:", min_geom)
print("Max vol:", vol_fields_max, "Min vol:", vol_fields_min)
print("Max surf:", surf_fields_max, "Min surf:", surf_fields_min)
vol_factors = np.array([vol_fields_max, vol_fields_min])
surf_factors = np.array([surf_fields_max, surf_fields_min])
# np.save("/lustre/rranade/modulus_dev/modulus/physicsnemo/examples/cfd/external_aerodynamics/domino_nim_finetuning/outputs/AWS_Dataset_Finetune/volume_scaling_factors.npy", vol_factors)
# np.save("/lustre/rranade/modulus_dev/modulus/physicsnemo/examples/cfd/external_aerodynamics/domino_nim_finetuning/outputs/AWS_Dataset_Finetune/surface_scaling_factors.npy", surf_factors)
np.save("/lustre/rranade/modulus_dev/modulus/physicsnemo/examples/cfd/external_aerodynamics/domino_nim_finetuning/outputs/AWS_Dataset_Baseline/volume_scaling_factors.npy", vol_factors)
np.save("/lustre/rranade/modulus_dev/modulus/physicsnemo/examples/cfd/external_aerodynamics/domino_nim_finetuning/outputs/AWS_Dataset_Baseline/surface_scaling_factors.npy", surf_factors)
