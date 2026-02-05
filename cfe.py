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
        If the soil moisture calculation scheme is 'classic', calculate the evaporation from the soil
        Elseif the soil moisture calculation scheme is 'ode', do nothing, because evaporation from the soil will be calculated within run_soil_moisture_scheme
        """
        if cfe_state.soil_params['scheme'].lower() == 'classic':
            
            cfe_state.actual_et_from_soil_m_per_timestep = 0
            # If the soil moisture storage is more than wilting point, calculate ET from soil
            if(cfe_state.soil_reservoir['storage_m'] > cfe_state.soil_reservoir['wilting_point_m']): 
                self.et_from_soil(cfe_state)

            cfe_state.vol_et_from_soil += cfe_state.actual_et_from_soil_m_per_timestep
            cfe_state.vol_et_to_atm += cfe_state.actual_et_from_soil_m_per_timestep
            cfe_state.volout += cfe_state.actual_et_from_soil_m_per_timestep 

            cfe_state.actual_et_m_per_timestep += cfe_state.actual_et_from_soil_m_per_timestep
            
        elif cfe_state.soil_params['scheme'].lower() == 'ode':
            None
        
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
        """
        if cfe_state.soil_params['scheme'].lower() == 'classic':
            # Add infiltration flux and calculate the reservoir flux 
            cfe_state.soil_reservoir['storage_m'] += cfe_state.infiltration_depth_m
            self.conceptual_reservoir_flux_calc(cfe_state, cfe_state.soil_reservoir)
        elif cfe_state.soil_params['scheme'].lower() == 'ode':
            # Infiltration flux is added witin the ODE scheme
            self.soil_moisture_flux_calc_with_ode(cfe_state, cfe_state.soil_reservoir)

    # ________________________________________________________________________________________________________
    def update_outflux_from_soil(self, cfe_state):
        cfe_state.flux_perc_m = cfe_state.primary_flux_m  #percolation_flux
        cfe_state.flux_lat_m = cfe_state.secondary_flux_m # lateral_flux
        
        # If the soil moisture scheme is classic, take out the outflux from soil moisture storage
        # If ODE, outfluxes are already subtracted from the soil moisture storage
        if cfe_state.soil_params['scheme'].lower() == 'classic':
            cfe_state.soil_reservoir['storage_m'] -= cfe_state.flux_perc_m
            cfe_state.soil_reservoir['storage_m'] -= cfe_state.flux_lat_m
        
        # If ODE, track actual ET from soil
        if cfe_state.soil_params['scheme'].lower() == 'ode':
            cfe_state.vol_et_from_soil += cfe_state.actual_et_from_soil_m_per_timestep
            cfe_state.vol_et_to_atm += cfe_state.actual_et_from_soil_m_per_timestep
            cfe_state.volout += cfe_state.actual_et_from_soil_m_per_timestep
            cfe_state.actual_et_m_per_timestep += cfe_state.actual_et_from_soil_m_per_timestep
            
        elif cfe_state.soil_params['scheme'].lower() == 'classic':
            None

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
    def Schaake_partitioning_scheme(self,cfe_state):
        """
            This subtroutine takes water_input_depth_m and partitions it into surface_runoff_depth_m and
            infiltration_depth_m using the scheme from Schaake et al. 1996. 
            !--------------------------------------------------------------------------------
            modified by FLO April 2020 to eliminate reference to ice processes, 
            and to de-obfuscate and use descriptive and dimensionally consistent variable names.
            
            inputs:
              timestep_d
              Schaake_adjusted_magic_constant_by_soil_type = C*Ks(soiltype)/Ks_ref, where C=3, and Ks_ref=2.0E-06 m/s
              column_total_soil_moisture_deficit_m (soil_reservoir_storage_deficit_m)
              water_input_depth_m (timestep_rainfall_input_m) amount of water input to soil surface this time step [m]
            outputs:
              surface_runoff_depth_m      amount of water partitioned to surface water this time step [m]
              infiltration_depth_m
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
                    
                cfe_state.infiltration_depth_m =  cfe_state.timestep_rainfall_input_m - cfe_state.surface_runoff_depth_m
                    
        else:
            
            cfe_state.surface_runoff_depth_m = 0.0
            cfe_state.infiltration_depth_m = 0.0

        # --- MODIFICATION START: Added Frozen Soil Logic to match C code ---
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
        # --- MODIFICATION END ---
            
        return
