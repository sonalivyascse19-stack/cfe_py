import time
import numpy as np
import pandas as pd
import sys
from scipy.integrate import odeint
import math

class CFE():
    def __init__(self):
        super(CFE, self).__init__()
        
    # ____________________________________________________________________________________
    def calculate_input_rainfall_and_PET(self, cfe_state):
        """
        Calculate input rainfall and PET 
        """
        cfe_state.potential_et_m_per_timestep = cfe_state.potential_et_m_per_s * cfe_state.time_step_size
        cfe_state.reduced_potential_et_m_per_timestep = cfe_state.potential_et_m_per_s * cfe_state.time_step_size
        
    # ____________________________________________________________________________________
    def calculate_evaporation_from_rainfall(self, cfe_state):
        """
        Calculate evaporation from rainfall. If it is raining, take PET from rainfall
        """
        cfe_state.actual_et_from_rain_m_per_timestep = 0
        if(cfe_state.timestep_rainfall_input_m > 0):
            self.et_from_rainfall(cfe_state)
        
        cfe_state.vol_et_from_rain += cfe_state.actual_et_from_rain_m_per_timestep
        cfe_state.vol_et_to_atm += cfe_state.actual_et_from_rain_m_per_timestep
        cfe_state.volout += cfe_state.actual_et_from_rain_m_per_timestep
        
        cfe_state.actual_et_m_per_timestep += cfe_state.actual_et_from_rain_m_per_timestep
        
    # ____________________________________________________________________________________
    def calculate_evaporation_from_soil(self, cfe_state):
        """
        Calculate the evaporation from the soil.
        Logic updated to strictly follow C-code behavior (Classic/Euler).
        ODE options removed to ensure parity.
        """
        cfe_state.actual_et_from_soil_m_per_timestep = 0
        
        # If the soil moisture storage is more than wilting point, calculate ET from soil
        # Note: Root Zone logic is now handled inside et_from_soil
        if(cfe_state.soil_reservoir['storage_m'] > cfe_state.soil_reservoir['wilting_point_m']): 
            self.et_from_soil(cfe_state)

        cfe_state.vol_et_from_soil += cfe_state.actual_et_from_soil_m_per_timestep
        cfe_state.vol_et_to_atm += cfe_state.actual_et_from_soil_m_per_timestep
        cfe_state.volout += cfe_state.actual_et_from_soil_m_per_timestep 

        cfe_state.actual_et_m_per_timestep += cfe_state.actual_et_from_soil_m_per_timestep
            
        
    # ____________________________________________________________________________________
    def calculate_the_soil_moisture_deficit(self, cfe_state):
        """ Calculate the soil moisture deficit
        """
        cfe_state.soil_reservoir_storage_deficit_m = (cfe_state.soil_params['smcmax'] * cfe_state.soil_params['D'] - \
                                                        cfe_state.soil_reservoir['storage_m'])
        
    # ____________________________________________________________________________________
    def calculate_infiltration_excess_overland_flow(self, cfe_state):
        """ Calculates infiltration excess overland flow 
        by running the partitioning scheme based on the choice set in the Configuration file
        """
        if (cfe_state.timestep_rainfall_input_m > 0.0): 
            if cfe_state.surface_partitioning_scheme == "Schaake": 
                self.Schaake_partitioning_scheme(cfe_state)
            elif cfe_state.surface_partitioning_scheme == "Xinanjiang": 
                self.Xinanjiang_partitioning_scheme(cfe_state)
            else: 
                print("Problem: must specify one of Schaake of Xinanjiang partitioning scheme.\n")
                print("Program terminating.:( \n");
                sys.exit(1)
        else: 
            cfe_state.surface_runoff_depth_m = 0.0
            cfe_state.infiltration_depth_m = 0.0

    # __________________________________________________________________________________________________________
    def calculate_saturation_excess_overland_flow_from_soil(self, cfe_state):
        """ Calculates saturation excess overland flow (SOF)
        This should be run after calculate_infiltration_excess_overland_flow, then, 
        infiltration_depth_m and surface_runoff_depth_m get finalized 
        """
        # If the infiltration is more than the soil moisture deficit, 
        # additional runoff (SOF) occurs and soil get saturated
        if cfe_state.soil_reservoir_storage_deficit_m < cfe_state.infiltration_depth_m:
            diff = cfe_state.infiltration_depth_m - cfe_state.soil_reservoir_storage_deficit_m
            cfe_state.surface_runoff_depth_m += diff
            cfe_state.infiltration_depth_m -= diff
            cfe_state.soil_reservoir_storage_deficit_m = 0
        else:
            # If the infiltration is less than the soil moisture deficit,
            # Infiltration & runoff flux is as calculated in calculate_infiltration_excess_overland_flow()
            None

    # __________________________________________________________________________________________________________
    def track_infiltration_and_runoff(self, cfe_state):
        """ Tracking runoff & infiltraiton volume with final infiltration & runoff values
        """
        cfe_state.vol_partition_runoff += cfe_state.surface_runoff_depth_m
        cfe_state.vol_partition_infilt += cfe_state.infiltration_depth_m
        cfe_state.vol_to_soil += cfe_state.infiltration_depth_m
        
    # __________________________________________________________________________________________________________        
    def run_soil_moisture_scheme(self, cfe_state):
        """ Run the soil moisture scheme based on the choice set in the Configuration file
        FIX: Removed ODE option to force Classic/Euler scheme matching C source of truth.
        """
        # Add infiltration flux 
        cfe_state.soil_reservoir['storage_m'] += cfe_state.infiltration_depth_m
        
        # Calculate fluxes based on updated storage (Explicit Euler)
        self.conceptual_reservoir_flux_calc(cfe_state, cfe_state.soil_reservoir)


    # ________________________________________________________________________________________________________
    def update_outflux_from_soil(self, cfe_state):
        cfe_state.flux_perc_m = cfe_state.primary_flux_m  #percolation_flux
        cfe_state.flux_lat_m = cfe_state.secondary_flux_m # lateral_flux
        
        # Take out the outflux from soil moisture storage (Classic/Euler)
        cfe_state.soil_reservoir['storage_m'] -= cfe_state.flux_perc_m
        cfe_state.soil_reservoir['storage_m'] -= cfe_state.flux_lat_m
        

    # ________________________________________________________________________________________________________
    def calculate_groundwater_storage_deficit(self, cfe_state):
        cfe_state.gw_reservoir_storage_deficit_m = cfe_state.gw_reservoir['storage_max_m'] - cfe_state.gw_reservoir['storage_m']
        
    # __________________________________________________________________________________________________________
    def calculate_saturation_excess_overland_flow_from_gw(self, cfe_state):
        # When the groundwater storage is full, the overflowing amount goes to direct runoff
        if cfe_state.flux_perc_m > cfe_state.gw_reservoir_storage_deficit_m:
            diff = cfe_state.flux_perc_m - cfe_state.gw_reservoir_storage_deficit_m
            cfe_state.surface_runoff_depth_m += diff
            cfe_state.flux_perc_m = cfe_state.gw_reservoir_storage_deficit_m
            cfe_state.gw_reservoir['storage_m'] = cfe_state.gw_reservoir['storage_max_m']
            cfe_state.gw_reservoir_storage_deficit_m = 0
            cfe_state.vol_partition_runoff += diff 
            cfe_state.vol_partition_infilt -= diff
            
        cfe_state.gw_reservoir['storage_m']   += cfe_state.flux_perc_m
            
    # __________________________________________________________________________________________________________
    def track_volume_from_percolation_and_lateral_flow(self, cfe_state):
        # Finalize the percolation and lateral flow 
        cfe_state.vol_to_gw                += cfe_state.flux_perc_m
        cfe_state.vol_soil_to_gw           += cfe_state.flux_perc_m
        cfe_state.vol_soil_to_lat_flow     += cfe_state.flux_lat_m  #TODO add this to nash cascade as input
        cfe_state.volout                   += cfe_state.flux_lat_m
    # __________________________________________________________________________________________________________
    
    def set_flux_from_deep_gw_to_chan_m(self, cfe_state):
        cfe_state.flux_from_deep_gw_to_chan_m = cfe_state.primary_flux_m
        if (cfe_state.flux_from_deep_gw_to_chan_m > cfe_state.gw_reservoir['storage_m']): 
            cfe_state.flux_from_deep_gw_to_chan_m = cfe_state.gw_reservoir['storage_m']
            if cfe_state.verbose:
                print("WARNING: Groundwater flux larger than storage. \n")

        cfe_state.vol_from_gw += cfe_state.flux_from_deep_gw_to_chan_m
        
    # __________________________________________________________________________________________________________
    def remove_flux_from_deep_gw_to_chan_m(self, cfe_state):
        """ Just an accounting operation
        """
        cfe_state.gw_reservoir['storage_m'] -= cfe_state.flux_from_deep_gw_to_chan_m
    # __________________________________________________________________________________________________________
    def track_volume_from_giuh(self, cfe_state):
        cfe_state.vol_out_giuh += cfe_state.flux_giuh_runoff_m
        cfe_state.volout += cfe_state.flux_giuh_runoff_m
    # __________________________________________________________________________________________________________
    def track_volume_from_deep_gw_to_chan(self, cfe_state):
        cfe_state.volout += cfe_state.flux_from_deep_gw_to_chan_m 
    # __________________________________________________________________________________________________________
    def track_volume_from_nash_cascade(self, cfe_state):
        cfe_state.vol_in_nash += cfe_state.flux_lat_m
        cfe_state.vol_out_nash += cfe_state.flux_nash_lateral_runoff_m
    # __________________________________________________________________________________________________________
    def add_up_total_flux_discharge(self, cfe_state):
        cfe_state.flux_Qout_m = cfe_state.flux_giuh_runoff_m + cfe_state.flux_nash_lateral_runoff_m + cfe_state.flux_from_deep_gw_to_chan_m
        cfe_state.total_discharge = cfe_state.flux_Qout_m * cfe_state.catchment_area_km2 * 1000000.0 / cfe_state.time_step_size
    # __________________________________________________________________________________________________________
    def update_current_time(self, cfe_state):
        cfe_state.current_time_step += 1
        cfe_state.current_time      += pd.Timedelta(value=cfe_state.time_step_size, unit='s')


    # __________________________________________________________________________________________________________
    # __________________________________________________________________________________________________________
    # MAIN MODEL FUNCTION
    def run_cfe(self, cfe_state):
        
        # Rainfall and ET 
        self.calculate_input_rainfall_and_PET(cfe_state)
        self.calculate_evaporation_from_rainfall(cfe_state)
        self.calculate_evaporation_from_soil(cfe_state)
        
        # Infiltration partitioning
        self.calculate_the_soil_moisture_deficit(cfe_state)
        self.calculate_infiltration_excess_overland_flow(cfe_state)
        self.calculate_saturation_excess_overland_flow_from_soil(cfe_state)
        self.track_infiltration_and_runoff(cfe_state)

        # Soil moisture reservoir
        self.run_soil_moisture_scheme(cfe_state)
        self.update_outflux_from_soil(cfe_state)
        
        # Groundwater reservoir
        self.calculate_groundwater_storage_deficit(cfe_state)
        self.calculate_saturation_excess_overland_flow_from_gw(cfe_state)
        self.track_volume_from_percolation_and_lateral_flow(cfe_state)
        self.conceptual_reservoir_flux_calc(cfe_state, cfe_state.gw_reservoir)  
        self.set_flux_from_deep_gw_to_chan_m(cfe_state)
        self.check_is_fabs_less_than_epsilon(cfe_state) 
        self.remove_flux_from_deep_gw_to_chan_m(cfe_state)
        
        # Surface runoff rounting
        self.convolution_integral(cfe_state)
        self.track_volume_from_giuh(cfe_state)
        self.track_volume_from_deep_gw_to_chan(cfe_state)
        
        # Lateral flow rounting
        self.nash_cascade(cfe_state)
        self.track_volume_from_nash_cascade(cfe_state)
        self.add_up_total_flux_discharge(cfe_state)
        
        # Time
        self.update_current_time(cfe_state)
        
        return
    
    # __________________________________________________________________________________________________________
    # __________________________________________________________________________________________________________
    # __________________________________________________________________________________________________________
    
    # __________________________________________________________________________________________________________
    def nash_cascade(self,cfe_state):
        """
            Solve for the flow through the Nash cascade to delay the 
            arrival of the lateral flow into the channel
        """
        Q = np.zeros(cfe_state.num_lateral_flow_nash_reservoirs)
        
        for i in range(cfe_state.num_lateral_flow_nash_reservoirs):
            
            Q[i] = cfe_state.K_nash * cfe_state.nash_storage[i]
            
            cfe_state.nash_storage[i] -= Q[i]
            
            if i == 0:
                
                cfe_state.nash_storage[i] += cfe_state.flux_lat_m
                
            else:
                
                cfe_state.nash_storage[i] += Q[i-1]
        
        cfe_state.flux_nash_lateral_runoff_m = Q[cfe_state.num_lateral_flow_nash_reservoirs - 1]
        
        return
    

    # __________________________________________________________________________________________________________
    def convolution_integral(self,cfe_state):
        """
            This function solves the convolution integral involving N GIUH ordinates.
            
            Inputs:
                Schaake_output_runoff_m
                num_giuh_ordinates
                giuh_ordinates
            Outputs:
                runoff_queue_m_per_timestep
        """

        N=cfe_state.num_giuh_ordinates

        cfe_state.runoff_queue_m_per_timestep[N] = 0.0
        
        
        for i in range(cfe_state.num_giuh_ordinates): 

            cfe_state.runoff_queue_m_per_timestep[i] += cfe_state.giuh_ordinates[i] * cfe_state.surface_runoff_depth_m
            
        cfe_state.flux_giuh_runoff_m = cfe_state.runoff_queue_m_per_timestep[0]
        
        # __________________________________________________________________
        # shift all the entries in preperation for the next timestep

        for i in range(cfe_state.num_giuh_ordinates): 
            cfe_state.runoff_queue_m_per_timestep[i] = cfe_state.runoff_queue_m_per_timestep[i+1]

        return
    

    # __________________________________________________________________________________________________________
    def et_from_rainfall(self,cfe_state):
        
        """
            iff it is raining, take PET from rainfall first.  Wet veg. is efficient evaporator.
        """

        # If rainfall exceeds PET, actual AET from rainfall is equal to the PET
        if cfe_state.timestep_rainfall_input_m > cfe_state.potential_et_m_per_timestep:
            cfe_state.actual_et_from_rain_m_per_timestep = cfe_state.potential_et_m_per_timestep
            cfe_state.timestep_rainfall_input_m -= cfe_state.actual_et_from_rain_m_per_timestep

        # If rainfall is less than PET, all rainfall gets consumed as AET
        else: 
            cfe_state.actual_et_from_rain_m_per_timestep = cfe_state.timestep_rainfall_input_m
            cfe_state.timestep_rainfall_input_m = 0.0
    
        cfe_state.reduced_potential_et_m_per_timestep = cfe_state.potential_et_m_per_timestep - cfe_state.actual_et_from_rain_m_per_timestep
        
        return
                
                
    # __________________________________________________________________________________________________________
    ########## SINGLE OUTLET EXPONENTIAL RESERVOIR ###############
    ##########                -or-                 ###############
    ##########    TWO OUTLET NONLINEAR RESERVOIR   ###############                        
    def conceptual_reservoir_flux_calc(self,cfe_state,reservoir):
        """
            This function calculates the flux from a linear, or nonlinear 
            conceptual reservoir with one or two outlets, or from an
            exponential nonlinear conceptual reservoir with only one outlet.
            In the non-exponential instance, each outlet can have its own
            activation storage threshold.  Flow from the second outlet is 
            turned off by setting the discharge coeff. to 0.0.
        """

        if reservoir['is_exponential'] == True: 
            flux_exponential = np.exp(reservoir['exponent_primary'] * \
                                      reservoir['storage_m'] / \
                                      reservoir['storage_max_m']) - 1.0
            cfe_state.primary_flux_m = reservoir['coeff_primary'] * flux_exponential
            cfe_state.secondary_flux_m=0.0
            return
    
        cfe_state.primary_flux_m=0.0
        
        storage_above_threshold_m = reservoir['storage_m'] - reservoir['storage_threshold_primary_m']
        
        if storage_above_threshold_m > 0.0:
                               
            storage_diff = reservoir['storage_max_m'] - reservoir['storage_threshold_primary_m']
            storage_ratio = storage_above_threshold_m / storage_diff
            storage_power = np.power(storage_ratio, reservoir['exponent_primary'])
            
            cfe_state.primary_flux_m = reservoir['coeff_primary'] * storage_power

            if cfe_state.primary_flux_m > storage_above_threshold_m:
                cfe_state.primary_flux_m = storage_above_threshold_m
                
        cfe_state.secondary_flux_m = 0.0
            
        storage_above_threshold_m = reservoir['storage_m'] - reservoir['storage_threshold_secondary_m']
        
        if storage_above_threshold_m > 0.0:
            
            storage_diff = reservoir['storage_max_m'] - reservoir['storage_threshold_secondary_m']
            storage_ratio = storage_above_threshold_m / storage_diff
            storage_power = np.power(storage_ratio, reservoir['exponent_secondary'])
            
            cfe_state.secondary_flux_m = reservoir['coeff_secondary'] * storage_power
            
            if cfe_state.secondary_flux_m > (storage_above_threshold_m - cfe_state.primary_flux_m):
                cfe_state.secondary_flux_m = storage_above_threshold_m - cfe_state.primary_flux_m
                
        return
    
    
    # __________________________________________________________________________________________________________
    #  SCHAAKE RUNOFF PARTITIONING SCHEME
    def Schaake_partitioning_scheme(self, cfe_state):
        """
        Matches C implementation including frozen soil logic.
        """
        if 0 < cfe_state.timestep_rainfall_input_m:
            
            if 0 > cfe_state.soil_reservoir_storage_deficit_m:
                cfe_state.surface_runoff_depth_m = cfe_state.timestep_rainfall_input_m
                cfe_state.infiltration_depth_m = 0.0
            else:
                schaake_exp_term = np.exp( - cfe_state.Schaake_adjusted_magic_constant_by_soil_type * cfe_state.timestep_d)
                Schaake_parenthetical_term = (1.0 - schaake_exp_term)
                Ic = cfe_state.soil_reservoir_storage_deficit_m * Schaake_parenthetical_term
                Px = cfe_state.timestep_rainfall_input_m
                
                cfe_state.infiltration_depth_m = (Px * (Ic / (Px + Ic)))
                
                if 0.0 < (cfe_state.timestep_rainfall_input_m - cfe_state.infiltration_depth_m):
                    cfe_state.surface_runoff_depth_m = cfe_state.timestep_rainfall_input_m - cfe_state.infiltration_depth_m
                else:
                    cfe_state.surface_runoff_depth_m = 0.0
                    
                cfe_state.infiltration_depth_m = cfe_state.timestep_rainfall_input_m - cfe_state.surface_runoff_depth_m
                    
        else:
            cfe_state.surface_runoff_depth_m = 0.0
            cfe_state.infiltration_depth_m = 0.0
            
        # FIX: Impermeable fraction due to frozen soil (Matches C lines 235-257)
        # Note: Ensure 'ice_fraction_schaake' is in cfe_state
        ice_fraction = cfe_state.soil_reservoir.get('ice_fraction_schaake', 0.0)
        
        if ice_fraction > 1.0E-2:
            factor = 1.0
            cv_frz = 3 
            # Using 'D' for soil_depth as per C struct usage
            field_capacity_m = cfe_state.soil_reservoir['soil_water_content_field_capacity'] * cfe_state.soil_params['D'] 
            field_capacity = field_capacity_m / cfe_state.soil_params['D']
            
            frz_fact = cfe_state.soil_params['smcmax'] / field_capacity * (0.412 / 0.468)
            # Assuming ice_content_threshold is available in parameters
            ice_threshold = cfe_state.soil_params.get('ice_content_threshold', 0.0) 
            frzx = ice_threshold * frz_fact
            
            acrt = cv_frz * frzx / ice_fraction
            sum1 = 1.0
            
            for i1 in range(1, cv_frz):
                k = 1
                for i2 in range(i1 + 1, cv_frz):
                    k *= i2
                sum1 += np.power(acrt, (cv_frz - i1)) / float(k)
                
            factor = 1.0 - np.exp(-acrt) * sum1
            
            # Apply factor to infiltration
            cfe_state.infiltration_depth_m = factor * cfe_state.infiltration_depth_m
            cfe_state.surface_runoff_depth_m = cfe_state.timestep_rainfall_input_m - cfe_state.infiltration_depth_m

        return
    
    # __________________________________________________________________________________________________________
    def Xinanjiang_partitioning_scheme(self, cfe_state): 
        """
        Adapted to match CFE C implementation (Source of Truth).
        """
        # partition the total soil water in the column between free water and tension water
        free_water_m = cfe_state.soil_reservoir['storage_m'] - cfe_state.soil_reservoir['storage_threshold_primary_m']

        if (0.0 < free_water_m):
            tension_water_m = cfe_state.soil_reservoir['storage_threshold_primary_m']
        else: 
            free_water_m = 0.0
            tension_water_m = cfe_state.soil_reservoir['storage_m']
        
        # estimate the maximum free water and tension water available in the soil column
        max_free_water_m = cfe_state.soil_reservoir['storage_max_m'] - cfe_state.soil_reservoir['storage_threshold_primary_m']
        max_tension_water_m = cfe_state.soil_reservoir['storage_threshold_primary_m']

        # check that the free_water_m and tension_water_m do not exceed the maximum
        if(max_free_water_m < free_water_m): 
            free_water_m = max_free_water_m

        if(max_tension_water_m < tension_water_m): 
            tension_water_m = max_tension_water_m

        # FIX: Read parameters from state instead of hardcoding to 1 (Matches C implementation)
        a_Xinanjiang_inflection_point_parameter = cfe_state.soil_params['a_inflection_point_parameter']
        b_Xinanjiang_shape_parameter = cfe_state.soil_params['b_shape_parameter']
        x_Xinanjiang_shape_parameter = cfe_state.soil_params['x_shape_parameter']

        # Logic matches C lines 473-485
        if ((tension_water_m/max_tension_water_m) <= (0.5 - a_Xinanjiang_inflection_point_parameter)): 
            pervious_runoff_m = cfe_state.timestep_rainfall_input_m * \
                (np.power((0.5 - a_Xinanjiang_inflection_point_parameter),\
                    (1.0 - b_Xinanjiang_shape_parameter)) * \
                        np.power((1.0 - (tension_water_m/max_tension_water_m)),\
                            b_Xinanjiang_shape_parameter))
        else: 
            pervious_runoff_m = cfe_state.timestep_rainfall_input_m * \
                (1.0 - np.power((0.5 + a_Xinanjiang_inflection_point_parameter), \
                    (1.0 - b_Xinanjiang_shape_parameter)) * \
                        np.power((1.0 - (tension_water_m/max_tension_water_m)),\
                            (b_Xinanjiang_shape_parameter)))
    
        cfe_state.surface_runoff_depth_m = pervious_runoff_m * \
             (1.0 - np.power((1.0 - (free_water_m/max_free_water_m)), x_Xinanjiang_shape_parameter))

        if(cfe_state.surface_runoff_depth_m < 0.0): 
            cfe_state.surface_runoff_depth_m = 0.0
    
        if(cfe_state.surface_runoff_depth_m > cfe_state.timestep_rainfall_input_m): 
             cfe_state.surface_runoff_depth_m = cfe_state.timestep_rainfall_input_m
        
        cfe_state.infiltration_depth_m = cfe_state.timestep_rainfall_input_m - cfe_state.surface_runoff_depth_m

        return
                            
    # __________________________________________________________________________________________________________
    def et_from_soil(self, cfe_state):
        """
        Take AET from soil moisture storage.
        Matches C implementation including Root Zone AET logic.
        """
        cfe_state.actual_et_from_soil_m_per_timestep = 0
        
        # FIX: Check for Root Zone AET (Matches C line 388)
        # Note: Ensure 'is_aet_rootzone' and related profile variables are initialized in your cfe_state object
        if cfe_state.soil_reservoir.get('is_aet_rootzone', False):
            
            # Helper vars for readability
            max_rootzone_layer = cfe_state.soil_reservoir['max_rootzone_layer']
            layer_storage_m = cfe_state.soil_reservoir['smc_profile'][max_rootzone_layer] * \
                              cfe_state.soil_reservoir['delta_soil_layer_depth_m'][max_rootzone_layer]
            wltsmc = cfe_state.soil_params['wltsmc']
            field_capacity = cfe_state.soil_reservoir['soil_water_content_field_capacity']

            if cfe_state.soil_reservoir['smc_profile'][max_rootzone_layer] <= wltsmc:
                cfe_state.actual_et_from_soil_m_per_timestep = 0
            
            elif cfe_state.soil_reservoir['smc_profile'][max_rootzone_layer] >= field_capacity:
                cfe_state.actual_et_from_soil_m_per_timestep = np.minimum(cfe_state.reduced_potential_et_m_per_timestep, layer_storage_m)
            
            else:
                Budyko_numerator = cfe_state.soil_reservoir['smc_profile'][max_rootzone_layer] - wltsmc
                Budyko_denominator = field_capacity - wltsmc
                Budyko = Budyko_numerator / Budyko_denominator
                
                cfe_state.actual_et_from_soil_m_per_timestep = np.minimum(Budyko * cfe_state.reduced_potential_et_m_per_timestep, layer_storage_m)

            # Update state
            cfe_state.reduced_potential_et_m_per_timestep -= cfe_state.actual_et_from_soil_m_per_timestep
            
            # Remove moisture from specific layer
            layer_depth = cfe_state.soil_reservoir['delta_soil_layer_depth_m'][max_rootzone_layer]
            cfe_state.soil_reservoir['smc_profile'][max_rootzone_layer] -= (cfe_state.actual_et_from_soil_m_per_timestep / layer_depth)
            
            # Update total storage
            cfe_state.soil_reservoir['storage_m'] -= cfe_state.actual_et_from_soil_m_per_timestep

        # Standard "Bucket" Scheme (Matches C line 413)
        elif cfe_state.reduced_potential_et_m_per_timestep > 0:
            
            if cfe_state.soil_reservoir['storage_m'] >= cfe_state.soil_reservoir['storage_threshold_primary_m']:
                cfe_state.actual_et_from_soil_m_per_timestep = np.minimum(cfe_state.reduced_potential_et_m_per_timestep, 
                                                       cfe_state.soil_reservoir['storage_m'])
                               
            elif ((cfe_state.soil_reservoir['storage_m'] > cfe_state.soil_reservoir['wilting_point_m']) and 
                  (cfe_state.soil_reservoir['storage_m'] < cfe_state.soil_reservoir['storage_threshold_primary_m'])):
            
                Budyko_numerator = cfe_state.soil_reservoir['storage_m'] - cfe_state.soil_reservoir['wilting_point_m']
                Budyko_denominator = cfe_state.soil_reservoir['storage_threshold_primary_m'] - \
                                     cfe_state.soil_reservoir['wilting_point_m']
                Budyko = Budyko_numerator / Budyko_denominator

                cfe_state.actual_et_from_soil_m_per_timestep = np.minimum(Budyko * cfe_state.reduced_potential_et_m_per_timestep, cfe_state.soil_reservoir['storage_m'])
                               
            cfe_state.soil_reservoir['storage_m'] -= cfe_state.actual_et_from_soil_m_per_timestep
            cfe_state.reduced_potential_et_m_per_timestep -= cfe_state.actual_et_from_soil_m_per_timestep
        
        return
            
            
    # __________________________________________________________________________________________________________
    def check_is_fabs_less_than_epsilon(self,cfe_state,epsilon=1.0e-9):
        """ in the instance of calling the gw reservoir the secondary flux should be zero- verify
            From Line 157 of https://github.com/NOAA-OWP/cfe/blob/master/original_author_code/cfe.c
        """
        a = cfe_state.secondary_flux_m
        if np.abs(a) < epsilon:
            cfe_state.is_fabs_less_than_epsilon = True
        else:
            print("problem with nonzero flux point 1\n")
            cfe_state.is_fabs_less_than_epsilon = False 
    
    # __________________________________________________________________________________________________________
    # __________________________________________________________________________________________________________
    # ODE METHODS BELOW ARE PRESERVED BUT UNUSED TO MAINTAIN C-CODE PARITY
    # __________________________________________________________________________________________________________

    def soil_moisture_flux_ode(self, t, S, cfe_state, reservoir):
        """
        Soil reservoir module that solves ODE
        """
        storage_above_threshold_m = S - reservoir['storage_threshold_primary_m']
        storage_diff = reservoir['storage_max_m'] - reservoir['storage_threshold_primary_m']
        storage_ratio = np.minimum(storage_above_threshold_m / storage_diff, 1)

        perc_lat_switch = np.multiply(S - reservoir['storage_threshold_primary_m'] > 0, 1)
        ET_switch = np.multiply(S - reservoir['wilting_point_m'] > 0, 1)

        storage_above_threshold_m_paw = S - reservoir['wilting_point_m']
        storage_diff_paw = reservoir['storage_threshold_primary_m'] - reservoir['wilting_point_m']
        storage_ratio_paw = np.minimum(storage_above_threshold_m_paw/storage_diff_paw, 1) # Equation 11 (Ogden's document)
        dS = cfe_state.infiltration_depth_m -1 * perc_lat_switch * (reservoir['coeff_primary'] + reservoir['coeff_secondary']) * storage_ratio - ET_switch * cfe_state.reduced_potential_et_m_per_timestep * storage_ratio_paw
        return dS

    # __________________________________________________________________________________________________________
    # __________________________________________________________________________________________________________
    def jac(self, t, S, cfe_state, reservoir):
        # The Jacobian matrix of the equation conceptual_reservoir_flux_calc. Calculated as df/dS = (dS/dt)/dS.
        storage_diff = reservoir['storage_max_m'] - reservoir['storage_threshold_primary_m']
    
        perc_lat_switch = np.multiply(S - reservoir['storage_threshold_primary_m'] > 0, 1)
        ET_switch = np.multiply((S - reservoir['wilting_point_m'] > 0) and (S - reservoir['storage_threshold_primary_m'] < 0), 1)
    
        storage_diff_paw = reservoir['storage_threshold_primary_m'] - reservoir['wilting_point_m']
    
        dfdS = -1 * perc_lat_switch * (reservoir['coeff_primary'] + reservoir['coeff_secondary']) * 1/storage_diff - ET_switch * cfe_state.reduced_potential_et_m_per_timestep * 1/storage_diff_paw
        return [dfdS]
    
    # __________________________________________________________________________________________________________
    # __________________________________________________________________________________________________________
    def soil_moisture_flux_calc_with_ode(self, cfe_state, reservoir):
        """
            This function solves the soil moisture mass balance.
        """

        # Initialization
        y0 = [reservoir['storage_m']]
        t = np.array([0, 0.05, 0.15, 0.3, 0.6, 1.0]) # ODE time descritization of one time step

        # Solve and ODE
        sol = odeint(
            self.soil_moisture_flux_ode,
            y0,
            t,
            args=(cfe_state, reservoir),
            tfirst=True,
            Dfun=self.jac
        )

        # Finalize results
        ts_concat = t
        ys_concat = np.concatenate(sol, axis=0)

        # Estimate fluxes at each ODE time descritization
        t_proportion = np.diff(ts_concat)
        ys_avg = np.convolve(ys_concat, np.ones(2), 'valid') / 2

        lateral_flux = np.zeros(ys_avg.shape)
        perc_lat_switch = ys_avg - reservoir['storage_threshold_primary_m'] > 0
        lateral_flux[perc_lat_switch] = reservoir['coeff_secondary'] * np.minimum(
            (ys_avg[perc_lat_switch] - reservoir['storage_threshold_primary_m']) / (
                        reservoir['storage_max_m'] - reservoir['storage_threshold_primary_m']), 1)
        lateral_flux_frac = lateral_flux * t_proportion

        perc_flux = np.zeros(ys_avg.shape)
        perc_flux[perc_lat_switch] = reservoir['coeff_primary'] * np.minimum(
            (ys_avg[perc_lat_switch] - reservoir['storage_threshold_primary_m']) / (
                        reservoir['storage_max_m'] - reservoir['storage_threshold_primary_m']), 1)
        perc_flux_frac = perc_flux * t_proportion

        et_from_soil = np.zeros(ys_avg.shape)
        ET_switch = ys_avg - cfe_state.soil_params['wltsmc']* cfe_state.soil_params['D'] > 0
        et_from_soil[ET_switch] = cfe_state.reduced_potential_et_m_per_timestep * np.minimum(
            (ys_avg[ET_switch] - cfe_state.soil_params['wltsmc']* cfe_state.soil_params['D']) / (reservoir['storage_threshold_primary_m'] - cfe_state.soil_params['wltsmc']* cfe_state.soil_params['D']), 1)
        et_from_soil_frac = et_from_soil * t_proportion

        infilt_to_soil = np.repeat(cfe_state.infiltration_depth_m, ys_avg.shape)
        infilt_to_soil_frac = infilt_to_soil * t_proportion
        
        # Get the scale factor
        sum_outflux = lateral_flux_frac + perc_flux_frac + et_from_soil_frac
        if sum_outflux.any() == 0:
            flux_scale = 0
            if cfe_state.infiltration_depth_m > 0:
                # To account for mass balance error by ODE
                final_storage_m = y0[0] + cfe_state.infiltration_depth_m
            else:
                final_storage_m = y0[0]
        else:
            flux_scale = (
                (ys_concat[0] - ys_concat[-1]) + np.sum(infilt_to_soil_frac)
            ) / np.sum(sum_outflux)
            final_storage_m = ys_concat[-1]

        # Scale the fluxes
        scaled_lateral_flux = lateral_flux_frac * flux_scale
        scaled_perc_flux = perc_flux_frac * flux_scale
        scaled_et_flux = et_from_soil_frac * flux_scale

        # Pass the results
        cfe_state.primary_flux_m = math.fsum(scaled_perc_flux)
        cfe_state.secondary_flux_m = math.fsum(scaled_lateral_flux)
        cfe_state.actual_et_from_soil_m_per_timestep = math.fsum(scaled_et_flux)
        # reservoir['storage_m'] = ys_concat[-1]
        cfe_state.soil_reservoir["storage_m"] = final_storage_m
        
        return
