import numpy as np

hplenum = 1.4 
zdepth2_back = hplenum*np.sin(2*np.pi*5/360) * -0.5

print(zdepth2_back)

fuel_injector_bottom_centerline_height = 0.0205 
fuel_injector_top_centerline_height = 0.0235 
secondary_inlet_bottom_centerline_height = 0.03

fuel_injector_top_z_front = fuel_injector_top_centerline_height*np.sin(2*np.pi*(5/360)) * 0.5
fuel_injector_top_z_back = fuel_injector_top_centerline_height*np.sin(2*np.pi*(5/360)) * -0.5
fuel_injector_bottom_z_front =fuel_injector_bottom_centerline_height *np.sin(2*np.pi*(5/360)) * 0.5
fuel_injector_bottom_z_back = fuel_injector_bottom_centerline_height*np.sin(2*np.pi*(5/360)) * -0.5
secondary_inlet_bottom_z_front = secondary_inlet_bottom_centerline_height*np.sin(2*np.pi*(5/360)) * 0.5
secondary_inlet_bottom_z_back = secondary_inlet_bottom_centerline_height*np.sin(2*np.pi*(5/360)) * -0.5


output = f"""
fuel_injector_top_z_front {fuel_injector_top_z_front};
fuel_injector_top_z_back {fuel_injector_top_z_back};
fuel_injector_bottom_z_front {fuel_injector_bottom_z_front};
fuel_injector_bottom_z_back {fuel_injector_bottom_z_back};
secondary_inlet_bottom_z_front {secondary_inlet_bottom_z_front};
secondary_inlet_bottom_z_back {secondary_inlet_bottom_z_back};
"""

print(output)

fuel_injector_top = fuel_injector_top_centerline_height * np.cos(2*np.pi*(5/360))
fuel_injector_bottom = fuel_injector_bottom_centerline_height * np.cos(2*np.pi*(5/360))
secondary_inlet_bottom = secondary_inlet_bottom_centerline_height * np.cos(2*np.pi*(5/360))

output = f"""
fuel_injector_top {fuel_injector_top};
fuel_injector_bottom {fuel_injector_bottom};
secondary_inlet_bottom {secondary_inlet_bottom};
"""
print(output)
