import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # safe even if no display
import matplotlib.pyplot as plt
import bmi_cfe

# Optional: allow passing config path: python run_cfe_bmi.py cat58_config_cfe.json
import sys
cfg = sys.argv[1] if len(sys.argv) > 1 else "./cat58_config_cfe.json"

cfe_instance = bmi_cfe.BMI_CFE(cfg)
cfe_instance.initialize()

df_forcing = pd.read_csv(cfe_instance.forcing_file)

outputs = cfe_instance.get_output_var_names()
output_lists = {name: [] for name in outputs}

for precip in df_forcing["APCP_surface"]:
    # APCP_surface is in mm (kg/m2); divide by 1000 to convert to meters,
    # matching the C code (bmi_cfe.c line 1555: precip_kg_per_m2 / 1000)
    cfe_instance.set_value(
        "atmosphere_water__time_integral_of_precipitation_mass_flux",
        precip / 1000.0
    )
    cfe_instance.update()

    for name in outputs:
        val = cfe_instance.get_value(name)
        # make sure we store a scalar
        val = float(np.asarray(val).ravel()[0])
        output_lists[name].append(val)

# --- AFTER LOOP: SAVE + PRINT ---
out_df = pd.DataFrame(output_lists)
out_df.insert(0, "timestep", np.arange(len(out_df)))

out_csv = "cfe_bmi_outputs.csv"
out_df.to_csv(out_csv, index=False)

print("Wrote:", out_csv)
print(out_df.head())

# Optional: quick plot of first output variable
first_var = outputs[0] if outputs else None
if first_var:
    plt.figure()
    plt.plot(out_df[first_var])
    plt.title(first_var)
    plt.xlabel("timestep")
    plt.savefig("cfe_output_plot.png", dpi=200, bbox_inches="tight")
    print("Wrote: cfe_output_plot.png")

cfe_instance.finalize()
